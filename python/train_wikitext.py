"""Train fast_bpe_rs on WikiText with Hugging Face datasets."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from datasets import load_dataset
from fast_bpe_rs import BPE

DATASET_REPO = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-103-raw-v1"
SUPPORTED_DATASET_CONFIGS = ("wikitext-103-raw-v1", "wikitext-2-raw-v1")
DEFAULT_TARGET_VOCAB_SIZE = 1 << 15
DEFAULT_RUNS = 3

REGEX = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fast_bpe_rs on the WikiText train split and print timing."
    )
    parser.add_argument(
        "--dataset-config",
        choices=SUPPORTED_DATASET_CONFIGS,
        default=DEFAULT_DATASET_CONFIG,
        help=f"WikiText dataset config (default: {DEFAULT_DATASET_CONFIG}).",
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


def load_wikitext_arrow(dataset_config: str):
    """Load WikiText and return a PyArrow ChunkedArray of non-empty stripped texts."""
    dataset = load_dataset(DATASET_REPO, dataset_config, split="train")
    col = dataset.data.column("text")
    return col


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger("python.train_wikitext")
    args = parse_args()
    docs = load_wikitext_arrow(args.dataset_config)

    for run in range(1, args.runs + 1):
        bpe = BPE(REGEX)
        start_ns = time.perf_counter_ns()
        bpe.train_arrow(args.target_vocab_size, docs)
        elapsed_ns = time.perf_counter_ns() - start_ns
        logger.info(
            "python.train_wikitext duration_ns=%s duration_ms=%s "
            "duration_s=%.6f run=%s",
            elapsed_ns,
            elapsed_ns // 1_000_000,
            elapsed_ns / 1_000_000_000,
            run,
        )


if __name__ == "__main__":
    main()
