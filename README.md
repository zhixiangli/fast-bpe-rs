# fast-bpe-rs

A blazing-fast [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer — written in Rust, usable from Python.

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

## License

[Apache 2.0](LICENSE)
