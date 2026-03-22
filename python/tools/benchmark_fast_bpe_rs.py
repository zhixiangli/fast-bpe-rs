from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections.abc import Callable, Sequence

import cpuinfo
import datasets
import distro
import psutil
from datasets import load_dataset
from fast_bpe_rs import BPE

GPT4_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
VOCAB_SIZE = 4096
RUNS = 5
BYTES_PER_MB = 1024 * 1024
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"


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


def _load_wikitext() -> list[str]:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    return [
        text
        for split in dataset.values()
        for text in split["text"]
        if text and not text.isspace()
    ]


def _measure_phase(
    name: str,
    input_bytes: int,
    action: Callable[[], object],
) -> dict[str, object]:
    wall_times: list[float] = []
    peak_mbs: list[float] = []

    for _ in range(RUNS):
        tracemalloc.start()
        started = time.perf_counter()
        action()
        wall_times.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mbs.append(peak / BYTES_PER_MB)

    median_wall_time = statistics.median(wall_times)
    return {
        "name": name,
        "input_bytes": input_bytes,
        "runs": RUNS,
        "wall_time_seconds_median": round(median_wall_time, 6),
        "throughput_mb_s": round((input_bytes / BYTES_PER_MB) / median_wall_time, 6),
        "peak_ram_mb": round(max(peak_mbs), 6),
        "wall_time_seconds_runs": [round(value, 6) for value in wall_times],
        "peak_ram_mb_runs": [round(value, 6) for value in peak_mbs],
    }


def benchmark() -> dict[str, object]:
    docs = _load_wikitext()
    corpus_text = "\n".join(docs)
    corpus_bytes = corpus_text.encode("utf-8")

    train_result = _measure_phase(
        "train",
        len(corpus_bytes),
        lambda: BPE(GPT4_SPLIT_PATTERN).train(VOCAB_SIZE, docs),
    )

    trained = BPE(GPT4_SPLIT_PATTERN)
    trained.train(VOCAB_SIZE, docs)
    encoded = trained.encode(corpus_text)

    return {
        "hardware": _hardware_info(),
        "dataset": {
            "name": DATASET_NAME,
            "config": DATASET_CONFIG,
            "documents": len(docs),
            "input_bytes": len(corpus_bytes),
        },
        "benchmark": {
            "library": "fast-bpe-rs",
            "vocab_size": VOCAB_SIZE,
            "split_pattern": GPT4_SPLIT_PATTERN,
            "phases": {
                "train": train_result,
                "encode": _measure_phase(
                    "encode",
                    len(corpus_bytes),
                    lambda: trained.encode(corpus_text),
                ),
                "decode": _measure_phase(
                    "decode",
                    len(corpus_bytes),
                    lambda: trained.decode(encoded),
                ),
            },
        },
    }


def main() -> None:
    print(json.dumps(benchmark(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
