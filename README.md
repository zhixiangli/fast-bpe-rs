# fast-bpe-rs

A blazing-fast [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer — written in Rust, usable from Python.

---

## What makes it fast?

Most BPE implementations rescan the entire corpus after every merge. `fast-bpe-rs` doesn't.

It uses a **doubly-linked list** to represent token chains and a **frequency-indexed BTreeMap** to find the next best merge. After each merge, only the immediate neighbours of affected positions are updated — skipping the vast majority of work.

| Phase | Naïve BPE | fast-bpe-rs |
|---|---|---|
| Per-merge rescan | O(n) | O(kᵢ) — only affected positions |
| Max-pair lookup | O(V) | O(log V) — BTreeMap |
| **Total training** | **O(n · V)** | **O(n log V)** |

---

## Benchmarks

Tested on a 5 MB corpus. Numbers show relative behavior, not absolute hardware performance.

### Training (vocab size = 4,096)

| | Time (s) | Throughput (MB/s) | Peak RAM (MB) | vs. minbpe |
|---|---:|---:|---:|---:|
| minbpe BasicTokenizer | 447.3 | 0.011 | 418 | 1× |
| minbpe RegexTokenizer | 583.1 | 0.009 | 521 | 0.77× |
| rustbpe | 25.4 | 0.197 | 63 | 17.6× |
| **fast-bpe-rs** | **6.0** | **0.83** | **48** | **74.5×** |

### Encoding & Decoding

| | Encode (MB/s) | Decode (MB/s) |
|---|---:|---:|
| minbpe BasicTokenizer | 3.40 | 12.8 |
| rustbpe | 28.1 | 87.3 |
| **fast-bpe-rs** | **41.7** | **94.2** |

### The advantage grows with vocabulary size

| Vocab size | fast-bpe-rs speedup vs. minbpe Regex |
|---|---:|
| 1,024 | 43× |
| 2,048 | 62× |
| 4,096 | 92× |
| 8,192 | 153× |

The longer the merge schedule, the more work is skipped — which matches the algorithm's design.

---

## Quick start

### Install

```bash
pip install fast-bpe-rs
```

> No prebuilt wheel for your platform? `pip` will compile from source. You'll need a [Rust toolchain](https://rustup.rs) installed first.

### Train

```python
from fast_bpe_rs import BPE

bpe = BPE(r"(?s).+")  # regex pattern for pre-splitting text
bpe.train(258, ["low low low low", "lower lower", "newest newest newest"])
```

For real corpora, use a GPT-style split pattern:

```python
bpe = BPE(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
bpe.train(50_000, corpus_lines)
```

### Encode & Decode

```python
ids = bpe.encode("low lower newest")
text = bpe.decode_to_string(ids)
```

---

## How should I write my commits?

This project uses [Release Please](https://github.com/googleapis/release-please) to prepare releases, so commit messages should follow the [Conventional Commits](https://www.conventionalcommits.org/) format. That helps the release automation determine version bumps and generate changelog entries correctly.

Examples:

- `feat: add support for custom regex patterns`
- `fix: handle invalid split regex in python bindings`
- `docs: clarify installation requirements`

---

## License

[Apache 2.0](LICENSE)
