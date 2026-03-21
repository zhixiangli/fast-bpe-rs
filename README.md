# fast-bpe-rs

A high-performance [Byte Pair Encoding (BPE)](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer written in **Rust**, with Python bindings.

## Why this exists

BPE is at the heart of every major LLM today — GPT, LLaMA, Mistral, and friends all use it to convert raw text into the token sequences the model actually sees. **Getting tokenizer training right, and fast, matters.**

The standard Python BPE implementations are correct but slow — training on large corpora becomes a real bottleneck. Existing Rust ports are faster by virtue of the language, but most carry over the same naïve O(n·V) algorithm. This project starts from Rust and rethinks the algorithm itself, using a **doubly-linked list** to represent token chains and a **frequency-indexed BTreeMap** to find the next best merge in O(log V) instead of a full scan.

## Algorithm improvements

| Phase | Naïve BPE | fast-bpe-rs |
|---|---|---|
| Per-merge rescan | O(n) | O(kᵢ) — only occurrences of merged pair |
| Max-pair lookup | O(V) | O(log V) — BTreeMap min |
| Merge application | O(n) | O(kᵢ) — in-place linked-list edits |
| **Total training** | **O(n · V)** | **O(Σ kᵢ · log V) ≈ O(n log V)** |

Where **n** is corpus size, **V** is vocabulary size, and **kᵢ** is the number of occurrences of the pair merged at step i.

The key insight: after each merge, only the immediate neighbours of every affected position change. Instead of rescanning the whole corpus, the linked-list structure lets us jump directly to those positions and update counts locally. The BTreeMap keeps pairs ordered by frequency so the next best merge is always at the front.

## Results

Benchmarks below use a 5 MB corpus and compare `fast-bpe-rs` against `minbpe` and `rustbpe`. They are intended to show relative behavior rather than serve as a hardware-independent standard.

### Training (vocab size = 4,096)

| System | Time (s) | Throughput (MB/s) | Peak RAM (MB) | Speedup vs. `minbpe` BasicTokenizer |
|---|---:|---:|---:|---:|
| `minbpe` BasicTokenizer | 447.3 | 0.011 | 418 | 1.0× |
| `minbpe` RegexTokenizer | 583.1 | 0.009 | 521 | 0.77× |
| `rustbpe` | 25.4 | 0.197 | 63 | 17.6× |
| **`fast-bpe-rs`** | **6.0** | **0.83** | **48** | **74.5×** |

Even on a single thread, `fast-bpe-rs` is about **4.2× faster than `rustbpe`** in this setup, largely because incremental updates avoid repeating most of the pair-counting work after each merge.

### Encoding

Encoding applies the learned merge rules to unseen text from the same 5 MB corpus.

| System | Time (s) | Throughput (MB/s) | Peak RAM (MB) |
|---|---:|---:|---:|
| `minbpe` BasicTokenizer | 1.47 | 3.40 | 52 |
| `minbpe` RegexTokenizer | 1.82 | 2.75 | 67 |
| `rustbpe` | 0.178 | 28.1 | 24 |
| **`fast-bpe-rs`** | **0.120** | **41.7** | **19** |

### Decoding

Decoding is mostly dominated by token-to-bytes lookup, so the gap is smaller but still measurable.

| System | Time (s) | Throughput (MB/s) | Peak RAM (MB) |
|---|---:|---:|---:|
| `minbpe` BasicTokenizer | 0.391 | 12.8 | 38 |
| `minbpe` RegexTokenizer | 0.387 | 12.9 | 41 |
| `rustbpe` | 0.057 | 87.3 | 16 |
| **`fast-bpe-rs`** | **0.053** | **94.2** | **14** |

### Training throughput vs. vocabulary size

As vocabulary size grows, the benefit of incremental updates becomes more pronounced: the naïve training cost grows roughly linearly with the number of merges, while `fast-bpe-rs` only updates the neighborhoods touched by each merge.

| Vocab size | `minbpe` Regex (MB/s) | `rustbpe` (MB/s) | `fast-bpe-rs` (MB/s) | `fast-bpe-rs` speedup vs. `minbpe` Regex |
|---|---:|---:|---:|---:|
| 1,024 | 0.038 | 0.47 | 1.62 | **43×** |
| 2,048 | 0.018 | 0.28 | 1.12 | **62×** |
| 4,096 | 0.009 | 0.197 | 0.83 | **92×** |
| 8,192 | 0.004 | 0.11 | 0.61 | **153×** |

In other words, the advantage widens as the merge schedule gets longer, which matches the asymptotic behavior described above.

## Quick start

### Installation

```bash
pip install fast-bpe-rs
```

If no prebuilt wheel exists for your platform, pip will compile from source — you'll need a recent [Rust toolchain](https://rustup.rs) installed.

### Train

```python
from fast_bpe_rs import BPE

# The argument is a regex pattern used to pre-split text into chunks.
# r"(?s).+" treats the whole input as one chunk (simplest case).
bpe = BPE(r"(?s).+")

# Learn 258 merges on the given corpus
bpe.train(258, ["low low low low", "lower lower", "newest newest newest"])
```

A GPT-style split pattern for real corpora:

```python
bpe = BPE(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
bpe.train(50_000, corpus_lines)
```

### Encode

```python
ids = bpe.encode("low lower newest")
print(ids)  # e.g. [260, 262, 259, 261, ...]
```

### Decode

```python
text = bpe.decode_to_string(ids)
print(text)  # "low lower newest"
```


## License

[Apache 2.0](LICENSE)
