import pytest
from fast_bpe_rs import BPE


def test_python_can_train_encode_and_decode_roundtrip() -> None:
    bpe = BPE(r"(?s).+")
    bpe.train(258, ["banana banana"])

    encoded = bpe.encode("banana banana")

    assert encoded
    assert any(token > 255 for token in encoded)
    assert bpe.decode(encoded) == b"banana banana"
    assert bpe.decode_to_string(encoded) == "banana banana"


def test_python_split_pattern_scopes_merges() -> None:
    bpe = BPE(r"\S+")
    bpe.train(257, ["go go", "go stop"])

    assert bpe.encode("go stop go") == [
        256,
        ord("s"),
        ord("t"),
        ord("o"),
        ord("p"),
        256,
    ]


def test_python_special_tokens_keep_custom_ids_without_merging() -> None:
    bpe = BPE(r"(?s).+", {"<pad>": 900, "<eos>": 901})
    bpe.train(905, ["a<pad>a", "<pad><eos><pad>"])

    assert bpe.encode("a<pad><eos>a") == [ord("a"), 900, 901, ord("a")]
    assert bpe.decode([900, 901]) == b"<pad><eos>"


def test_python_invalid_regex_raises_value_error() -> None:
    with pytest.raises(ValueError, match="invalid split regex"):
        BPE(r"(")


def test_python_decode_to_string_invalid_utf8_raises_unicode_decode_error() -> None:
    bpe = BPE(r"(?s).")

    with pytest.raises(UnicodeDecodeError) as exc_info:
        bpe.decode_to_string([0xFF])

    err = exc_info.value
    assert err.encoding == "utf-8"
    assert err.object == b"\xff"
    assert err.start == 0
    assert err.end == 1
    assert err.reason == "invalid utf-8"


def test_python_can_train_from_pyarrow_string_array() -> None:
    pyarrow = pytest.importorskip("pyarrow")

    docs = pyarrow.array(["banana banana", None, "banana"], type=pyarrow.string())
    bpe = BPE(r"(?s).+")
    bpe.train_arrow(258, docs)

    encoded = bpe.encode("banana banana")
    assert encoded
    assert any(token > 255 for token in encoded)
