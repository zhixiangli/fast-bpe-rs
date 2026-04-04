"""Train fast_bpe_rs on WikiText with Hugging Face datasets."""

from __future__ import annotations

import argparse
import time

from datasets import load_dataset
from fast_bpe_rs import BPE

DATASET_REPO = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-103-raw-v1"
SUPPORTED_DATASET_CONFIGS = ("wikitext-103-raw-v1", "wikitext-2-raw-v1")
DATASET_SPLIT = "train"
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
        help=(
            "WikiText dataset config to train on "
            f"(default: {DEFAULT_DATASET_CONFIG})."
        ),
    )
    parser.add_argument(
        "--target-vocab-size",
        type=positive_int,
        default=DEFAULT_TARGET_VOCAB_SIZE,
        help=(
            "Target vocabulary size for BPE training "
            f"(default: {DEFAULT_TARGET_VOCAB_SIZE})."
        ),
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=DEFAULT_RUNS,
        help=f"Number of training runs to execute (default: {DEFAULT_RUNS}).",
    )
    return parser.parse_args()


def load_wikitext_train_docs(dataset_config: str) -> tuple[list[str], int]:
    dataset = load_dataset(DATASET_REPO, dataset_config, split=DATASET_SPLIT)
    total_rows = len(dataset)
    docs = [text.strip() for text in dataset["text"] if text and text.strip()]
    return docs, total_rows


def main() -> None:
    args = parse_args()
    docs, total_rows = load_wikitext_train_docs(args.dataset_config)
    print(
        "RUN_CONTEXT "
        f"dataset_repo={DATASET_REPO} "
        f"dataset_config={args.dataset_config} "
        f"dataset_split={DATASET_SPLIT} "
        f"split_rows={total_rows} "
        f"docs_loaded={len(docs)} "
        f"rows_filtered_out={total_rows - len(docs)} "
        f"target_vocab_size={args.target_vocab_size} "
        f"runs={args.runs}"
    )

    for run in range(1, args.runs + 1):
        bpe = BPE(REGEX)
        start_ns = time.perf_counter_ns()
        bpe.train(args.target_vocab_size, docs)
        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
        print(f"TRAIN_RUN run={run} elapsed_ms={elapsed_ms}")

    print("TRAINING_COMPLETE finished=true")


if __name__ == "__main__":
    main()
