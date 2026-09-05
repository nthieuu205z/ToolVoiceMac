#!/usr/bin/env python
"""Measure translation concurrency with a synthetic backend or real Gemini backend.

The default mode is dry-run and uses a deterministic latency backend. Pass
``--real`` only when an API measurement has been explicitly approved.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.performance_harness import Measurement, measure, write_json_report
from pipeline.models import Segment
from pipeline.translate import translate_segments


class SyntheticTranslationBackend:
    """Latency-only backend; it never stores or returns line content."""

    def __init__(self, latency_seconds: float, jitter_seconds: float, seed: int):
        self.latency_seconds = max(0.0, latency_seconds)
        self.jitter_seconds = max(0.0, jitter_seconds)
        self.random = random.Random(seed)

    def translate(self, texts, durations, context=""):
        delay = self.latency_seconds + self.random.uniform(0.0, self.jitter_seconds)
        time.sleep(delay)
        return ["ok"] * len(texts)


def _segments(count: int) -> list[Segment]:
    # Text is deliberately not written to reports or stdout.
    return [Segment(float(index), float(index + 1), "x") for index in range(count)]


def _parse_workers(value: str) -> list[int]:
    workers = []
    for item in value.split(","):
        parsed = int(item.strip())
        if parsed < 1:
            raise ValueError("workers must be positive")
        workers.append(parsed)
    return workers


def _run_case(batch_count: int, workers: int, latency: float, jitter: float) -> Measurement:
    # BATCH_SIZE is 48, so this creates multiple independent translation requests.
    from pipeline.translate import BATCH_SIZE

    segments = _segments(max(1, batch_count) * BATCH_SIZE)

    def operation():
        backend = SyntheticTranslationBackend(latency, jitter, seed=workers + batch_count)
        translate_segments(
            backend,  # type: ignore[arg-type]
            segments,
            workers=workers,
        )
        return len(segments)

    measurement, _ = measure(
        f"translation-synthetic-workers-{workers}",
        operation,
        item_count=len(segments),
        warm=True,
        metadata={
            "stage": "translate",
            "provider": "synthetic-latency",
            "device": "cloud",
            "workers": workers,
            "batch_count": batch_count,
            "latency_seconds": latency,
            "jitter_seconds": jitter,
        },
    )
    return measurement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--workers", default="1,2,4,6")
    parser.add_argument("--latency", type=float, default=0.08)
    parser.add_argument("--jitter", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path(".hermes/perf/translation-baseline.json"))
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--real", action="store_true", help="reserved for an explicitly approved API run")
    args = parser.parse_args()

    if args.real:
        parser.error("real API mode is intentionally disabled in this harness; use an approved provider-specific runner")
    if args.batches < 1:
        parser.error("--batches must be positive")
    if args.latency < 0 or args.jitter < 0:
        parser.error("latency and jitter must be non-negative")
    try:
        workers = _parse_workers(args.workers)
    except ValueError as exc:
        parser.error(str(exc))

    measurements = [
        _run_case(args.batches, worker, args.latency, args.jitter)
        for worker in workers
    ]
    write_json_report(
        args.output,
        measurements,
        run_metadata={
            "kind": "translation-baseline",
            "mode": "synthetic-dry-run",
            "batch_count": args.batches,
        },
    )
    for item in measurements:
        print(
            f"{item.label}: {item.elapsed_seconds:.3f}s, "
            f"{item.items_per_second:.3f} lines/s, error={item.error_type or 'none'}"
        )
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
