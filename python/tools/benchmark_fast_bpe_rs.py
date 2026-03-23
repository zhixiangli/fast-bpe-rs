from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any

import cpuinfo
import datasets
import distro
import psutil
from datasets import load_dataset
from fast_bpe_rs import BPE
from memory_profiler import memory_usage

GPT4_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
VOCAB_SIZE = 4096
RUNS = 5
BYTES_PER_MB = 1024 * 1024
NANOSECONDS_PER_SECOND = 1000 * 1000 * 1000
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MEMORY_PROFILER_INTERVAL_SECONDS = 0.005


datasets.disable_progress_bar()
datasets.utils.logging.set_verbosity_error()


def _command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _hardware_info() -> dict[str, object]:
    info = cpuinfo.get_cpu_info()
    return {
        "cpu": info.get("brand_raw", "unavailable"),
        "ram_mb": round(psutil.virtual_memory().total / BYTES_PER_MB, 6),
        "os": distro.name(pretty=True) or platform.platform(),
        "python_version": platform.python_version(),
        "rust_version": _command_output(["rustc", "--version"]),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS") or "default",
    }


def _load_wikitext(dataset_name: str, dataset_config: str) -> list[str]:
    dataset = load_dataset(dataset_name, dataset_config)
    return [
        text
        for split in dataset.values()
        for text in split["text"]
        if text and not text.isspace()
    ]


def _measure_phase(
    name: str,
    input_bytes: int,
    runs: int,
    action: Callable[[], object],
) -> dict[str, object]:
    # Warmup: one untimed run to populate caches and trigger lazy initialization.
    action()

    # --- Speed pass (isolated from memory-profiling overhead) ---
    wall_times: list[float] = []
    for _ in range(runs):
        gc.collect()
        gc.disable()
        try:
            start_ns = time.perf_counter_ns()
            action()
            elapsed_ns = time.perf_counter_ns() - start_ns
        finally:
            gc.enable()
        wall_times.append(elapsed_ns / NANOSECONDS_PER_SECOND)

    # --- Memory pass (separate runs so profiling overhead cannot skew timings) ---
    process = psutil.Process()
    peak_mbs: list[float] = []
    for _ in range(runs):
        gc.collect()
        baseline_mb = process.memory_info().rss / BYTES_PER_MB
        peak_mb = memory_usage(
            (action, (), {}),
            interval=MEMORY_PROFILER_INTERVAL_SECONDS,
            max_usage=True,
            multiprocess=False,
        )
        peak_mbs.append(max(0.0, float(peak_mb) - baseline_mb))

    median_wall = statistics.median(wall_times)
    input_mb = input_bytes / BYTES_PER_MB
    return {
        "name": name,
        "input_bytes": input_bytes,
        "runs": runs,
        "wall_time_seconds_median": round(median_wall, 6),
        "wall_time_seconds_mean": round(statistics.mean(wall_times), 6),
        "wall_time_seconds_stdev": (
            round(statistics.stdev(wall_times), 6) if runs > 1 else 0.0
        ),
        "wall_time_seconds_min": round(min(wall_times), 6),
        "wall_time_seconds_max": round(max(wall_times), 6),
        "throughput_mb_s": round(input_mb / median_wall, 6),
        "peak_ram_mb_median": round(statistics.median(peak_mbs), 6),
        "peak_ram_mb_max": round(max(peak_mbs), 6),
        "wall_time_seconds_runs": [round(v, 6) for v in wall_times],
        "peak_ram_mb_runs": [round(v, 6) for v in peak_mbs],
    }


def _benchmark_library(
    *,
    library: str,
    split_pattern: str,
    vocab_size: int,
    runs: int,
    input_text: str,
    train_input: str,
    build_trained: Callable[[], Any],
) -> dict[str, object]:
    input_bytes = len(input_text.encode("utf-8"))
    train_result = _measure_phase("train", input_bytes, runs, build_trained)
    trained = build_trained()

    def encode() -> object:
        return trained.encode(input_text)

    def decode() -> object:
        return trained.decode(trained.encode(input_text))

    return {
        "library": library,
        "vocab_size": vocab_size,
        "split_pattern": split_pattern,
        "train_input": train_input,
        "phases": {
            "train": train_result,
            "encode": _measure_phase("encode", input_bytes, runs, encode),
            "decode": _measure_phase("decode", input_bytes, runs, decode),
        },
    }


def _benchmark_fast_bpe_rs(
    docs: list[str],
    corpus_text: str,
    *,
    runs: int,
    vocab_size: int,
    split_pattern: str,
    train_mode: str,
) -> dict[str, object]:
    train_docs = docs if train_mode == "docs" else [corpus_text]
    train_input = "wikitext documents" if train_mode == "docs" else "joined corpus text"
    return _benchmark_library(
        library="fast-bpe-rs",
        split_pattern=split_pattern,
        vocab_size=vocab_size,
        runs=runs,
        input_text=corpus_text,
        train_input=train_input,
        build_trained=lambda: _train_fast_bpe_rs(split_pattern, vocab_size, train_docs),
    )


def _train_fast_bpe_rs(split_pattern: str, vocab_size: int, docs: list[str]) -> BPE:
    trained = BPE(split_pattern)
    trained.train(vocab_size, docs)
    return trained


def _benchmark_rustbpe(
    corpus_text: str,
    *,
    runs: int,
    vocab_size: int,
    split_pattern: str,
) -> dict[str, object]:
    rustbpe = importlib.import_module("rustbpe")
    return _benchmark_library(
        library="rustbpe Tokenizer",
        split_pattern=split_pattern,
        vocab_size=vocab_size,
        runs=runs,
        input_text=corpus_text,
        train_input="joined corpus text",
        build_trained=lambda: _train_rustbpe(
            rustbpe,
            corpus_text,
            vocab_size,
            split_pattern,
        ),
    )


def _train_rustbpe(
    rustbpe: Any,
    corpus_text: str,
    vocab_size: int,
    split_pattern: str,
) -> Any:
    trained = rustbpe.Tokenizer()
    trained.train_from_iterator(
        [corpus_text],
        vocab_size=vocab_size,
        pattern=split_pattern,
    )
    return trained


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    docs = _load_wikitext(args.dataset_name, args.dataset_config)
    corpus_text = "\n".join(docs)
    result: dict[str, object] = {
        "hardware": _hardware_info(),
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "documents": len(docs),
            "input_bytes": len(corpus_text.encode("utf-8")),
        },
        "benchmark": _benchmark_fast_bpe_rs(
            docs,
            corpus_text,
            runs=args.runs,
            vocab_size=args.vocab_size,
            split_pattern=args.split_pattern,
            train_mode="docs",
        ),
    }

    if args.compare_rustbpe:
        result["comparison"] = {
            "input_mode": "joined corpus text",
            "split_pattern": args.split_pattern,
            "benchmarks": [
                _benchmark_fast_bpe_rs(
                    docs,
                    corpus_text,
                    runs=args.runs,
                    vocab_size=args.vocab_size,
                    split_pattern=args.split_pattern,
                    train_mode="corpus",
                ),
                _benchmark_rustbpe(
                    corpus_text,
                    runs=args.runs,
                    vocab_size=args.vocab_size,
                    split_pattern=args.split_pattern,
                ),
            ],
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fast-bpe-rs on WikiText and optionally compare it against "
            "rustbpe using the exact same joined corpus input."
        )
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--split-pattern", default=GPT4_SPLIT_PATTERN)
    parser.add_argument(
        "--compare-rustbpe",
        action="store_true",
        help=(
            "Benchmark fast-bpe-rs and rustbpe on the same joined corpus text while "
            "preserving the default fast-bpe-rs benchmark output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(benchmark(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
