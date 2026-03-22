from __future__ import annotations

import os
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cpuinfo
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


@dataclass(frozen=True)
class PhaseResult:
    name: str
    wall_time_seconds: float
    throughput_mb_s: float
    peak_ram_mb: float


def _command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _print_hardware_info() -> None:
    info = cpuinfo.get_cpu_info()
    total_ram_mb = psutil.virtual_memory().total / BYTES_PER_MB
    rayon_threads = os.environ.get("RAYON_NUM_THREADS") or "default"

    print("Hardware info")
    print(f"  CPU: {info.get('brand_raw', 'unavailable')}")
    print(f"  RAM: {total_ram_mb:.2f} MB")
    print(f"  OS: {distro.name(pretty=True) or platform.platform()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Rust: {_command_output(['rustc', '--version'])}")
    print(f"  Rayon threads: {rayon_threads}")


def _load_wikitext_lines() -> list[str]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
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
) -> PhaseResult:
    wall_times: list[float] = []
    peak_mbs: list[float] = []

    for _ in range(RUNS):
        tracemalloc.start()
        started = time.perf_counter()
        action()
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        wall_times.append(elapsed)
        peak_mbs.append(peak / BYTES_PER_MB)

    median_wall_time = statistics.median(wall_times)
    return PhaseResult(
        name=name,
        wall_time_seconds=median_wall_time,
        throughput_mb_s=(input_bytes / BYTES_PER_MB) / median_wall_time,
        peak_ram_mb=max(peak_mbs),
    )


def _print_phase_result(result: PhaseResult) -> None:
    print(result.name)
    print(f"  wall_time_seconds_median: {result.wall_time_seconds:.6f}")
    print(f"  throughput_mb_s: {result.throughput_mb_s:.6f}")
    print(f"  peak_ram_mb: {result.peak_ram_mb:.6f}")


def run_benchmark() -> None:
    _print_hardware_info()

    docs = _load_wikitext_lines()
    corpus_text = "\n".join(docs)
    corpus_bytes = corpus_text.encode("utf-8")

    print("Dataset")
    print("  name: Salesforce/wikitext")
    print("  config: wikitext-2-raw-v1")
    print(f"  documents: {len(docs)}")
    print(f"  input_bytes: {len(corpus_bytes)}")
    print(f"  vocab_size: {VOCAB_SIZE}")
    print(f"  split_pattern: {GPT4_SPLIT_PATTERN}")

    train_result = _measure_phase(
        name="train",
        input_bytes=len(corpus_bytes),
        action=lambda: BPE(GPT4_SPLIT_PATTERN).train(VOCAB_SIZE, docs),
    )

    trained = BPE(GPT4_SPLIT_PATTERN)
    trained.train(VOCAB_SIZE, docs)
    encoded = trained.encode(corpus_text)
    decoded = trained.decode(encoded)
    assert decoded == corpus_bytes

    encode_result = _measure_phase(
        name="encode",
        input_bytes=len(corpus_bytes),
        action=lambda: trained.encode(corpus_text),
    )
    decode_result = _measure_phase(
        name="decode",
        input_bytes=len(decoded),
        action=lambda: trained.decode(encoded),
    )

    print("Results")
    for result in (train_result, encode_result, decode_result):
        _print_phase_result(result)


def test_fast_bpe_rs_benchmark() -> None:
    run_benchmark()


if __name__ == "__main__":
    run_benchmark()
