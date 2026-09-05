"""Raw text synthesis keeps provider concerns separate from video timing."""

from __future__ import annotations

import pytest

from pipeline.errors import JobCancelledError
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.speech_synthesis import _chia_lo, synthesize_texts


def pcm(marker: int) -> bytes:
    return int(marker).to_bytes(2, "little", signed=True) * TTS_SAMPLE_RATE


class BatchFailsThenSingles:
    batch_size = 16

    def __init__(self, bad_text: str):
        self.bad_text = bad_text
        self.batch_calls: list[tuple[list[str], str, str]] = []
        self.single_calls: list[tuple[str, str, str]] = []

    def synthesize_batch(self, texts, voice_id, *, language="vi-VN"):
        self.batch_calls.append((list(texts), voice_id, language))
        raise RuntimeError("batch unavailable")

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        self.single_calls.append((text, voice_id, language))
        if text == self.bad_text:
            raise RuntimeError("single item failed: secret submitted text")
        return pcm({"one": 1, "three": 3}[text])


class RecordingBackend:
    batch_size = 0

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        self.calls.append((text, voice_id, language))
        return pcm(1)


def test_batch_fallback_preserves_indexes_and_marks_failed_items():
    backend = BatchFailsThenSingles(bad_text="two")

    audio, warnings = synthesize_texts(
        backend,
        ["one", "two", "three"],
        "voice",
        language="en-US",
    )

    assert audio == [pcm(1), None, pcm(3)]
    assert backend.batch_calls == [(["one", "two", "three"], "voice", "en-US")]
    assert backend.single_calls == [
        ("one", "voice", "en-US"),
        ("two", "voice", "en-US"),
        ("three", "voice", "en-US"),
    ]
    assert any("1/3" in warning for warning in warnings)
    assert all("secret submitted text" not in warning for warning in warnings)


def test_cancellation_runs_before_first_provider_call():
    backend = RecordingBackend()

    with pytest.raises(JobCancelledError):
        synthesize_texts(
            backend,
            ["one"],
            "voice",
            language="en-US",
            should_cancel=lambda: True,
        )

    assert backend.calls == []


def test_single_synthesis_preserves_order_language_and_progress():
    backend = RecordingBackend()
    progress: list[tuple[str, float, str]] = []

    audio, warnings = synthesize_texts(
        backend,
        ["one", "two"],
        "voice",
        language="en-US",
        progress=lambda stage, fraction, message: progress.append(
            (stage, fraction, message)
        ),
    )

    assert audio == [pcm(1), pcm(1)]
    assert warnings == []
    assert backend.calls == [
        ("one", "voice", "en-US"),
        ("two", "voice", "en-US"),
    ]
    assert [(stage, fraction) for stage, fraction, _ in progress] == [
        ("synthesize", 0.5),
        ("synthesize", 1.0),
    ]


def test_batch_sizes_are_balanced():
    assert _chia_lo(53, 32) == [27, 26]
    assert _chia_lo(33, 32) == [17, 16]
    assert _chia_lo(0, 32) == []
    for count in (1, 7, 40, 100, 129):
        sizes = _chia_lo(count, 32)
        assert sum(sizes) == count
        assert max(sizes) <= 32
        assert max(sizes) - min(sizes) <= 1
