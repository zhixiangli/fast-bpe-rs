from contextlib import redirect_stdout
from io import StringIO

import pytest
import tiktoken
from fast_bpe_rs import BPE
from tiktoken._educational import SimpleBytePairEncoding


def _train_tiktoken_reference(
    training_doc: str,
    split_pattern: str,
    vocab_size: int,
) -> SimpleBytePairEncoding:
    with redirect_stdout(StringIO()):
        return SimpleBytePairEncoding.train(
            training_doc,
            vocab_size=vocab_size,
            pat_str=split_pattern,
        )


def _train_tiktoken_reference_with_special_tokens(
    training_doc: str,
    split_pattern: str,
    vocab_size: int,
    special_tokens: dict[str, int],
) -> tiktoken.Encoding:
    educational_bpe = _train_tiktoken_reference(training_doc, split_pattern, vocab_size)
    return tiktoken.Encoding(
        "fast-bpe-rs-test",
        pat_str=split_pattern,
        mergeable_ranks=educational_bpe.mergeable_ranks,
        special_tokens=special_tokens,
    )


@pytest.mark.parametrize(
    ("training_doc", "sample"),
    [
        ("abababa", "abababa"),
        ("abababa", "ababa"),
        ("banana banana", "banana banana"),
        ("banana banana", "banana"),
        ("go go stop", "go go stop"),
        ("go go stop", "go stop"),
        ("mississippi river", "mississippi river"),
        ("mississippi river", "river"),
    ],
)
def test_fast_bpe_rs_matches_tiktoken_for_encoding_and_decoding(
    training_doc: str,
    sample: str,
) -> None:
    split_pattern = r"(?s).+"
    vocab_size = 257

    fast_bpe = BPE(split_pattern)
    fast_bpe.train(vocab_size, [training_doc])

    tiktoken_bpe = _train_tiktoken_reference(training_doc, split_pattern, vocab_size)

    fast_encoded = fast_bpe.encode(sample)
    tiktoken_encoded = tiktoken_bpe.encode(sample, visualise=None)

    assert fast_encoded == tiktoken_encoded
    assert fast_bpe.decode(fast_encoded) == tiktoken_bpe.decode_bytes(tiktoken_encoded)
    assert fast_bpe.decode_to_string(fast_encoded) == tiktoken_bpe.decode(
        tiktoken_encoded
    )


@pytest.mark.parametrize(
    ("training_doc", "sample", "special_tokens"),
    [
        (
            "banana banana",
            "banana<pad><eos> banana",
            {"<pad>": 900, "<eos>": 901},
        ),
        (
            "go go stop",
            "<pad>go<end?>stop<pad>",
            {"<pad>": 800, "<end?>": 801},
        ),
    ],
)
def test_fast_bpe_rs_matches_tiktoken_with_special_tokens(
    training_doc: str,
    sample: str,
    special_tokens: dict[str, int],
) -> None:
    split_pattern = r"(?s).+"
    vocab_size = 257

    fast_bpe = BPE(split_pattern, special_tokens)
    fast_bpe.train(vocab_size, [training_doc])

    tiktoken_bpe = _train_tiktoken_reference_with_special_tokens(
        training_doc,
        split_pattern,
        vocab_size,
        special_tokens,
    )

    fast_encoded = fast_bpe.encode(sample)
    tiktoken_encoded = tiktoken_bpe.encode(sample, allowed_special="all")

    assert fast_encoded == tiktoken_encoded
    assert fast_bpe.decode(fast_encoded) == tiktoken_bpe.decode_bytes(tiktoken_encoded)
    assert fast_bpe.decode_to_string(fast_encoded) == tiktoken_bpe.decode(
        tiktoken_encoded
    )
