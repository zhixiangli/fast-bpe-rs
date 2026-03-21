from contextlib import redirect_stdout
from io import StringIO

import pytest
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
    assert (
        fast_bpe.decode_to_string(fast_encoded)
        == tiktoken_bpe.decode(tiktoken_encoded)
    )
