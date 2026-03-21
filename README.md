# fast-bpe-rs

A small Rust implementation of **Byte Pair Encoding (BPE)** focused on being **simple, correct, and fast**.

## What is BPE?

BPE is a way to turn text into reusable pieces called **tokens**.

It starts at the byte level, so every character is first treated as raw bytes. Then it repeatedly:

1. finds the pair of neighboring tokens that appears most often,
2. merges that pair into one new token,
3. repeats until the vocabulary reaches the target size.

Example:

- Text: `banana banana`
- Frequent pairs like `a` + `n` or `an` + `a` appear many times.
- BPE learns merges such as `an`, then maybe `ana`, then larger chunks if they keep repeating.

The result is a tokenizer that keeps common patterns as single tokens while rare text can still fall back to bytes.

## How this implementation works

At a high level, this crate:

- splits input text into chunks using a regex,
- stores each chunk as a linked chain of token nodes,
- counts every adjacent token pair,
- always merges the most frequent pair during training,
- remembers the learned merges so encoding can replay them later,
- decodes by expanding token IDs back to their original bytes.

This keeps the code close to the core BPE idea: **count pairs, merge the best one, update the neighbors, repeat**.

## Why this version is much faster

The main speedup comes from **updating only what changed**.

A slower BPE implementation often rescans large parts of the text after every merge. That wastes work because most pairs did not change.

This version is faster because it:

- tracks pair counts incrementally instead of recounting from scratch,
- stores where each pair appears, so merges can jump directly to affected locations,
- updates only the local neighbors around each merge,
- uses linked-node chains so a merge is just a small pointer update, not a full string rebuild,
- keeps pair frequencies grouped so the next best merge is quick to find.

In simple terms: **instead of rebuilding the world after each merge, it fixes only the tiny part that moved**.

## Who this is for

This project is a good fit if you want:

- a readable BPE implementation in Rust,
- a compact reference for learning how BPE works,
- a faster training approach than naive full rescans.

It is especially useful for people who want to understand the idea without reading a huge tokenizer codebase.


## Python usage with uv

This repository now includes Python bindings powered by `pyo3` and `maturin`, so you can call the Rust tokenizer directly from Python.

```bash
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
uv run pytest
```

Example:

```python
from fast_bpe_rs import BPE

bpe = BPE(r"(?s).+")
bpe.train(258, ["banana banana"])
encoded = bpe.encode("banana banana")
assert bpe.decode_to_string(encoded) == "banana banana"
```
