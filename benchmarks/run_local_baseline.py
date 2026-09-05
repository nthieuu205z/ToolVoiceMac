#!/usr/bin/env python
"""Measure current local STT and OmniVoice MPS throughput without writing media output."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from benchmarks.performance_harness import Measurement, measure, write_json_report
from pipeline import custom_voices
from pipeline.omnivoice_speech import OmniVoiceSynthesizer
from pipeline.segmentation import plan_utterances
from pipeline.stt import transcribe_regions
from pipeline.whisper_stt import WhisperTranscriber
from pipeline.audio import read_wav


def _sync_mps() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def _sample_regions(regions, sample_count: int):
    if len(regions) <= sample_count:
        return regions
    step = (len(regions) - 1) / max(1, sample_count - 1)
    indices = sorted({round(index * step) for index in range(sample_count)})
    return [regions[index] for index in indices]


def _measure_whisper(source_wav: Path, sample_count: int) -> list[Measurement]:
    samples, rate = read_wav(source_wav)
    total_duration = len(samples) / rate
    regions = plan_utterances(
        source_wav,
        samples,
        rate,
        total_duration,
        max_gap=settings.max_utterance_gap,
        max_duration=settings.max_utterance_seconds,
    )
    sampled = _sample_regions(regions, sample_count)
    transcriber = WhisperTranscriber(
        settings.whisper_model,
        settings.whisper_compute_type,
        cpu_batch_size=settings.whisper_cpu_batch_size,
    )

    def clean_case_dir(name: str) -> Path:
        path = Path(".hermes/perf") / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Warm the model once outside the measured cases so worker comparisons are not
    # dominated by model loading. The measured calls still include real inference.
    with tempfile.TemporaryDirectory(prefix="toolvietsub-whisper-warmup-") as tmp:
        transcribe_regions(
            transcriber,
            samples,
            rate,
            sampled[:1],
            Path(tmp),
            workers=1,
        )

    measurements: list[Measurement] = []
    for workers in (1, 2, 4):
        def operation(workers=workers):
            case_dir = clean_case_dir(f".whisper-case-workers-{workers}")
            result = transcribe_regions(
                transcriber,  # type: ignore[arg-type]
                samples,
                rate,
                sampled,
                case_dir,
                workers=workers,
            )
            return len(result[1])

        measurement, _ = measure(
            f"whisper-cpu-workers-{workers}",
            operation,
            item_count=len(sampled),
            warm=True,
            metadata={
                "stage": "transcribe",
                "provider": "whisper",
                "device": "cpu",
                "workers": workers,
                "model": settings.whisper_model,
                "sample_count": len(sampled),
                "region_count": len(regions),
            },
        )
        measurements.append(measurement)
    return measurements


def _measure_omnivoice(batch_sizes: tuple[int, ...]) -> list[Measurement]:
    custom_voices.configure(settings.custom_voices_dir)
    voice_id = "clone-dunglai"
    if not custom_voices.is_custom(voice_id):
        raise RuntimeError("Required local benchmark voice is not available")

    text = "Xin chào, đây là phép đo hiệu năng."
    synthesizer = OmniVoiceSynthesizer(
        num_step=settings.omnivoice_num_step,
        batch_size=max(batch_sizes),
        whisper_model=settings.whisper_model,
        whisper_compute_type=settings.whisper_compute_type,
    )

    # Load the singleton and reference text before measuring batch throughput.
    _sync_mps()
    synthesizer.synthesize_batch([text], voice_id)
    _sync_mps()

    measurements: list[Measurement] = []
    for batch_size in batch_sizes:
        texts = [text] * batch_size

        def operation(texts=texts, batch_size=batch_size):
            # A fresh adapter changes only the requested batch override; the model
            # singleton remains shared, matching production behavior.
            adapter = OmniVoiceSynthesizer(
                num_step=settings.omnivoice_num_step,
                batch_size=batch_size,
                whisper_model=settings.whisper_model,
                whisper_compute_type=settings.whisper_compute_type,
            )
            _sync_mps()
            result = adapter.synthesize_batch(texts, voice_id)
            _sync_mps()
            return len(result)

        measurement, _ = measure(
            f"omnivoice-mps-batch-{batch_size}",
            operation,
            item_count=batch_size,
            warm=True,
            metadata={
                "stage": "synthesize",
                "provider": "omnivoice",
                "device": synthesizer.device() or "unknown",
                "batch_size": batch_size,
                "num_step": settings.omnivoice_num_step,
                "voice": "custom-local",
            },
        )
        measurements.append(measurement)
    return measurements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wav", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path(".hermes/perf/local-baseline.json"))
    args = parser.parse_args()

    if not args.source_wav.is_file():
        parser.error("--source-wav must point to an existing WAV file")
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")

    started = time.perf_counter()
    measurements = _measure_whisper(args.source_wav, args.sample_count)
    measurements.extend(_measure_omnivoice((1, 2, 4, 8)))
    write_json_report(
        args.output,
        measurements,
        run_metadata={
            "kind": "local-baseline",
            "sample_count": args.sample_count,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stt_provider": "whisper",
            "tts_provider": "omnivoice",
        },
    )

    for item in measurements:
        print(
            f"{item.label}: {item.elapsed_seconds:.3f}s, "
            f"{item.items_per_second:.3f} items/s, "
            f"peak_rss={item.peak_rss_mb:.1f}MB, error={item.error_type or 'none'}"
        )
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
