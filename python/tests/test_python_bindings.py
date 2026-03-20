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

    assert bpe.encode("go stop go") == [256, ord("s"), ord("t"), ord("o"), ord("p"), 256]
