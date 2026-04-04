# fast-bpe-rs: A Fast Rust BPE Library

A blazing-fast Rust Byte Pair Encoding (BPE) tokenizer with Python bindings for training, encoding, and decoding BPE models.

## Naive Vs `fast-bpe-rs`

Let `N` be the total number of token positions after splitting, `M` the number of merges to learn, and `k` the number of occurrences touched by the current merge.

| Aspect | Naive BPE trainer | `fast-bpe-rs` |
| --- | --- | --- |
| Corpus representation | Plain token lists such as `Vec<Vec<u32>>` | Deduplicated weighted training chains |
| Per-merge work | Recount pairs across the full corpus | Update only neighborhoods touched by the merge |
| Sequence updates | Rebuild or rewrite token lists repeatedly | In-place splicing in a sparse linked structure backed by `Vec<Option<Node>>` |
| Pair statistics | Recomputed from scratch each round | Maintained incrementally as `pair -> {count, locations}` and `count -> set of pairs` |
| Best-pair lookup | Usually depends on the latest full recount | Pulled from the highest non-empty count bucket |
| Repeated chunks | Counted again and again | Stored once with a frequency weight |
| Parallelism | Often minimal in simple implementations | Parallel chunk counting and initial pair aggregation with `rayon` |
| Training time complexity | Typically `O(MN)` because each merge triggers another global count pass | `O(N)` setup, then per merge roughly `O(k)` local updates instead of `O(N)` rescans |
| Space complexity | Usually `O(N)` plus temporary pair counts | Higher than naive: `O(N)` corpus state plus pair-location indexes and count buckets |

## Setup

### Install from PyPI

```bash
pip install fast-bpe-rs
```

## Use

### Small example

```python
from fast_bpe_rs import BPE

bpe = BPE(r"(?s).+")
bpe.train(258, ["low low low low", "lower lower", "newest newest newest"])

ids = bpe.encode("low lower newest")
text = bpe.decode_to_string(ids)
```

### GPT-style split pattern

```python
from fast_bpe_rs import BPE

bpe = BPE(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

bpe.train(32768, corpus_lines)
```

### Special tokens

```python
from fast_bpe_rs import BPE

bpe = BPE(
    r"(?s).+",
    {
        "<pad>": 600,
        "<eos>": 601,
    },
)

bpe.train(605, ["a<pad>a"])
ids = bpe.encode("a<pad><eos>a")
```

## API

- `BPE(split_pattern, special_tokens=None)`
- `train(vocab_size, docs)`
- `encode(text) -> list[int]`
- `decode(token_ids) -> bytes`
- `decode_to_string(token_ids) -> str`

## License

[Apache 2.0](LICENSE)
