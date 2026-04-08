"""Train fast_bpe_rs on supported Hugging Face datasets."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass

from datasets import load_dataset
from fast_bpe_rs import BPE


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    config: str | None = None
    split: str = "train"


DATASET_SPECS = {
    "wikitext-103-raw-v1": DatasetSpec(
        repo="Salesforce/wikitext",
        config="wikitext-103-raw-v1",
    ),
    "wikitext-2-raw-v1": DatasetSpec(
        repo="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
    ),
    "tinystories": DatasetSpec(repo="roneneldan/TinyStories"),
}
DEFAULT_DATASET = "wikitext-103-raw-v1"
DEFAULT_TARGET_VOCAB_SIZE = 1 << 15
DEFAULT_RUNS = 3

REGEX = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train fast_bpe_rs on a supported Hugging Face dataset train split "
            "and print timing."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_SPECS),
        default=DEFAULT_DATASET,
        help=f"Dataset key (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=positive_int,
        default=DEFAULT_TARGET_VOCAB_SIZE,
        help=f"Target vocabulary size (default: {DEFAULT_TARGET_VOCAB_SIZE}).",
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=DEFAULT_RUNS,
        help=f"Number of training runs (default: {DEFAULT_RUNS}).",
    )
    return parser.parse_args()


def load_text_column(dataset: str):
    """Load text column as a PyArrow ChunkedArray from dataset train split."""
    spec = DATASET_SPECS[dataset]
    dataset_split = load_dataset(spec.repo, spec.config, split=spec.split)
    return dataset_split.data.column("text")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger("python.train_hf_dataset")
    args = parse_args()
    docs = load_text_column(args.dataset)

    for run in range(1, args.runs + 1):
        bpe = BPE(REGEX)
        start_ns = time.perf_counter_ns()
        bpe.train_arrow(args.target_vocab_size, docs)
        elapsed_ns = time.perf_counter_ns() - start_ns
        logger.info(
            "python.train_hf_dataset dataset=%s duration_s=%.6f run=%s",
            args.dataset,
            elapsed_ns / 1_000_000_000,
            run,
        )


if __name__ == "__main__":
    main()
