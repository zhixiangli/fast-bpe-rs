# fast-bpe-rs

Fast Byte Pair Encoding (BPE) training and tokenization in Rust, with Python bindings built with PyO3.

The repository currently exposes:

- a Rust `BPE` type via `fast_bpe_rs::BPE`
- a Python `BPE` class via `from fast_bpe_rs import BPE`
- training from Python strings with `train(...)`
- training from PyArrow string arrays with `train_arrow(...)`
- `encode(...)`, `decode(...)`, and `decode_to_string(...)`
- caller-defined special tokens with fixed token IDs

It does not currently expose a save/load or merge-export API.

## Requirements

- Python `>= 3.10`
- Rust `>= 1.85`

## Install

Install the Python package:

```bash
pip install fast-bpe-rs
```

If you want to use `train_arrow(...)`, install the Arrow extra so `pyarrow` is available:

```bash
pip install "fast-bpe-rs[arrow]"
```

If you want to use the Rust crate directly:

```bash
cargo add fast-bpe-rs
```

## Python Quick Start

`fast-bpe-rs` is published on PyPI as `fast-bpe-rs`, but the Python import name is `fast_bpe_rs`.

```python
from fast_bpe_rs import BPE

bpe = BPE(r"(?s).+")
bpe.train(258, ["banana banana"])

ids = bpe.encode("banana banana")
raw = bpe.decode(ids)
text = bpe.decode_to_string(ids)

print(ids)
print(raw)
print(text)
```

## Split Patterns

In Python, `split_pattern` is required. Each regex match becomes its own chunk, and merges never cross chunk boundaries.

Common choices:

- `r"(?s).+"`: treat each input string as one mergeable chunk
- `r"\S+"`: keep merges inside non-whitespace spans
- a GPT-style pattern: closer to common tokenizer behavior for natural language

Example GPT-style pattern:

```python
GPT_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)

bpe = BPE(GPT_SPLIT_PATTERN)
```

The Rust API also exposes this default pattern through `BPE::default()` and `BPE::new(None, ...)`.

## Special Tokens

Special tokens are matched literally and keep the exact IDs you assign them.

```python
from fast_bpe_rs import BPE

bpe = BPE(
    r"(?s).+",
    {
        "<pad>": 900,
        "<eos>": 901,
    },
)

bpe.train(905, ["a<pad>a", "<pad><eos><pad>"])
ids = bpe.encode("a<pad><eos>a")

print(ids)  # [97, 900, 901, 97]
print(bpe.decode([900, 901]))  # b"<pad><eos>"
```

Rules enforced by the code:

- special token IDs must be unique
- special token IDs must be `>= 256`
- special tokens do not merge with surrounding text
- special tokens do not merge with each other

## Training from PyArrow

The Python bindings can train directly from a PyArrow string `Array` or `ChunkedArray`.

```python
import pyarrow as pa
from fast_bpe_rs import BPE

bpe = BPE(r"(?s).+")
docs = pa.array(["hello world", "hello there", "hello again"])

bpe.train_arrow(257, docs)
print(bpe.encode("hello world"))
```

Behavior to know:

- `train_arrow(...)` expects string Arrow data
- null entries are skipped
- whitespace-only strings are skipped
- Arrow buffers are read without copying the text into Python-owned `str` objects first

## Rust Quick Start

```rust
use fast_bpe_rs::BPE;

let mut bpe = BPE::default();
bpe.train(257, ["banana banana"]);

let ids = bpe.encode("banana banana");
let bytes = bpe.decode(ids);

assert_eq!(bytes, b"banana banana");
```

If you want a custom split pattern or special tokens in Rust:

```rust
use fast_bpe_rs::BPE;

let mut bpe = BPE::new(
    Some(r"\S+"),
    Some([("<pad>", 900_u32), ("<eos>", 901_u32)]),
)
.expect("valid config");
```

## Public API

Python:

- `BPE(split_pattern, special_tokens=None)`
- `train(vocab_size, docs)`
- `train_arrow(vocab_size, arrow_array)`
- `encode(text) -> list[int]`
- `decode(token_ids) -> bytes`
- `decode_to_string(token_ids) -> str`

Rust:

- `BPE::default()`
- `BPE::new(split_pattern, special_tokens) -> Result<BPE, BPEError>`
- `BPE::train(vocab_size, docs)`
- `BPE::encode(text) -> Vec<u32>`
- `BPE::decode(token_ids) -> Vec<u8>`

## Behavior Notes

- The base vocabulary is the raw byte range `0..=255`.
- `decode(...)` silently skips unknown token IDs.
- `decode_to_string(...)` raises `UnicodeDecodeError` if the decoded bytes are not valid UTF-8.
- Calling `train(...)` again replaces previously learned merges while keeping the byte vocabulary and configured special tokens.

## Why It Is Fast

Compared with a textbook BPE trainer, `fast-bpe-rs` avoids repeated full-corpus rescans during training.

| Bottleneck | Naive approach | `fast-bpe-rs` |
| --- | --- | --- |
| Best-pair lookup | Recount or rescan every round | Frequency buckets for the current best pair |
| Applying a merge | Rewrite token arrays | In-place splice in a sparse linked structure |
| Pair-count updates | Rebuild counts from scratch | Update only neighboring pairs touched by the merge |
| Repeated chunks | Process each copy independently | Deduplicate chunks and weight them by frequency |
| Initial counting | Often sequential | Parallelized with Rayon |

At a high level, training is `O(N)` setup plus local `O(k)` updates per learned merge, rather than repeated `O(N)` full rescans.

## Repository Notes

This repository also includes:

- `python/tests/` for Python binding and parity tests
- `python/train_hf_dataset.py` for timing training on a few supported Hugging Face datasets

## License

[Apache 2.0](LICENSE)
