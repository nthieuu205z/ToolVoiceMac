"""Batch TTS contract: one provider call per group, ordered output, safe fallback."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import JobCancelledError
from pipeline.models import TTS_SAMPLE_RATE, Segment
from pipeline.tts import _chia_lo, synthesize_segments


def segments(*specs: tuple[float, float, str]) -> list[Segment]:
    return [Segment(start=s, end=e, text="x", text_vi=vi) for s, e, vi in specs]


def pcm(seconds: float = 1.0) -> bytes:
    return np.zeros(int(TTS_SAMPLE_RATE * seconds), dtype="<i2").tobytes()


class LoBackend:
    batch_size = 32

    def __init__(self, *, batch_error: bool = False, bad_text: str = ""):
        self.batch_error = batch_error
        self.bad_text = bad_text
        self.batch_calls: list[list[str]] = []
        self.single_calls: list[str] = []

    def synthesize_batch(self, texts, voice_id):
        self.batch_calls.append(list(texts))
        if self.batch_error:
            raise RuntimeError("batch failed")
        return [pcm() for _ in texts]

    def synthesize(self, text, voice_id):
        self.single_calls.append(text)
        if text == self.bad_text:
            raise RuntimeError("single failed")
        return pcm()

    def transcribe_clip(self, wav_path):
        return "en", ""

    def translate(self, texts, durations, context=""):
        return list(texts)


class ProgressBackend(LoBackend):
    batch_size = 2


def test_batch_progress_reports_each_completed_batch():
    backend = ProgressBackend()
    segs = segments(
        (0.0, 2.0, "a"),
        (3.0, 5.0, "b"),
        (6.0, 8.0, "c"),
    )
    progress = []

    synthesize_segments(
        backend,
        segs,
        "voice",
        total_duration=20.0,
        progress=lambda stage, fraction, message: progress.append((fraction, message)),
    )

    assert [round(fraction, 3) for fraction, _ in progress] == [0.0, 0.667, 1.0, 1.0]
    assert "lô 1/2" in progress[1][1]
    assert "lô 2/2" in progress[2][1]


def test_batch_provider_is_called_once_and_keeps_original_order():
    backend = LoBackend()
    segs = segments((0.0, 2.0, "a"), (3.0, 5.0, "b"), (6.0, 8.0, "c"))

    fitted, warnings = synthesize_segments(backend, segs, "voice", total_duration=20.0)

    assert [segment.text_vi for segment, _ in fitted] == ["a", "b", "c"]
    assert not warnings
    assert backend.batch_calls == [["a", "b", "c"]]
    assert backend.single_calls == []


def test_unsafe_mps_backend_uses_single_item_batches():
    class UnsafeBackend(LoBackend):
        mps_batch_safe = False

    backend = UnsafeBackend()
    segs = segments((0.0, 2.0, "a"), (3.0, 5.0, "b"), (6.0, 8.0, "c"))

    synthesize_segments(backend, segs, "voice", total_duration=20.0)

    assert backend.batch_calls == [["a"], ["b"], ["c"]]


def test_failed_unsafe_mps_batch_does_not_retry_as_a_multi_item_batch():
    class UnsafeBackend(LoBackend):
        mps_batch_safe = False
        batch_error = True

    backend = UnsafeBackend()
    segs = segments((0.0, 2.0, "a"), (3.0, 5.0, "b"))

    synthesize_segments(backend, segs, "voice", total_duration=20.0)

    assert backend.batch_calls == [["a"], ["b"]]


def test_mismatched_batch_output_falls_back_to_every_item():
    class ShortBackend(LoBackend):
        def synthesize_batch(self, texts, voice_id):
            self.batch_calls.append(list(texts))
            return [pcm()]

    backend = ShortBackend()
    segs = segments((0.0, 2.0, "a"), (3.0, 5.0, "b"), (6.0, 8.0, "c"))

    fitted, warnings = synthesize_segments(backend, segs, "voice", total_duration=20.0)

    assert backend.single_calls == ["a", "b", "c"]
    assert len(fitted) == 3
    assert warnings == []


def test_failed_batch_falls_back_to_single_items():
    backend = LoBackend(batch_error=True, bad_text="b")
    segs = segments((0.0, 2.0, "a"), (3.0, 5.0, "b"), (6.0, 8.0, "c"))

    fitted, warnings = synthesize_segments(backend, segs, "voice", total_duration=20.0)

    assert backend.single_calls == ["a", "b", "c"]
    assert [segment.text_vi for segment, _ in fitted] == ["a", "c"]
    assert any("1/3" in warning for warning in warnings)


def test_cancellation_happens_before_next_batch():
    backend = LoBackend()
    segs = segments((0.0, 2.0, "a"), (3.0, 5.0, "b"))

    with pytest.raises(JobCancelledError):
        synthesize_segments(
            backend,
            segs,
            "voice",
            should_cancel=lambda: True,
            total_duration=20.0,
        )

    assert backend.batch_calls == []


def test_batch_sizes_are_balanced():
    assert _chia_lo(53, 32) == [27, 26]
    assert _chia_lo(33, 32) == [17, 16]
    assert _chia_lo(0, 32) == []
    for count in (1, 7, 40, 100, 129):
        sizes = _chia_lo(count, 32)
        assert sum(sizes) == count
        assert max(sizes) <= 32
        assert max(sizes) - min(sizes) <= 1


def test_spoken_duration_is_saved_after_batch_fit():
    backend = LoBackend()
    segs = segments((0.0, 5.0, "a"))

    synthesize_segments(backend, segs, "voice", total_duration=20.0)

    assert segs[0].spoken_duration > 0
