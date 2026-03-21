from contextlib import redirect_stdout
from io import StringIO

import pytest
import tiktoken
from fast_bpe_rs import BPE
from tiktoken._educational import SimpleBytePairEncoding


def _train_tiktoken_encoding(
    training_doc: str,
    split_pattern: str,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None,
) -> tiktoken.Encoding:
    with redirect_stdout(StringIO()):
        educational_bpe = SimpleBytePairEncoding.train(
            training_doc,
            vocab_size=vocab_size,
            pat_str=split_pattern,
        )

    return tiktoken.Encoding(
        name="fast-bpe-rs-test",
        pat_str=split_pattern,
        mergeable_ranks=educational_bpe.mergeable_ranks,
        special_tokens=special_tokens or {},
    )


def _assert_fast_bpe_matches_tiktoken(
    training_doc: str,
    sample: str,
    split_pattern: str,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None,
) -> None:
    fast_bpe = BPE(split_pattern, special_tokens)
    fast_bpe.train(vocab_size, [training_doc])

    tiktoken_bpe = _train_tiktoken_encoding(
        training_doc,
        split_pattern,
        vocab_size,
        special_tokens,
    )

    encode_kwargs = {"allowed_special": "all"} if special_tokens else {}
    fast_encoded = fast_bpe.encode(sample)
    tiktoken_encoded = tiktoken_bpe.encode(sample, **encode_kwargs)

    assert fast_encoded == tiktoken_encoded
    assert fast_bpe.decode(fast_encoded) == tiktoken_bpe.decode_bytes(tiktoken_encoded)
    assert fast_bpe.decode_to_string(fast_encoded) == tiktoken_bpe.decode(
        tiktoken_encoded
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
    _assert_fast_bpe_matches_tiktoken(
        training_doc=training_doc,
        sample=sample,
        split_pattern=r"(?s).+",
        vocab_size=257,
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
    _assert_fast_bpe_matches_tiktoken(
        training_doc=training_doc,
        sample=sample,
        split_pattern=r"(?s).+",
        vocab_size=257,
        special_tokens=special_tokens,
    )
