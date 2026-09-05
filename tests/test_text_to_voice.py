"""Dedicated Text -> Voice synthesis and audio export behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline import text_to_voice
from pipeline.audio import read_wav, write_wav
from pipeline.errors import (
    FFmpegError,
    FFmpegNotFoundError,
    InvalidTextError,
    JobCancelledError,
    SpeechServiceError,
    TextTooLongError,
)
from pipeline.ffmpeg_utils import resolve, run
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.text_to_voice import (
    TextToVoiceOptions,
    build_mp3_cmd,
    chunk_text,
    encode_mp3,
    run_text_to_voice,
)


def pcm(seconds: float, marker: int = 1) -> bytes:
    sample_count = round(TTS_SAMPLE_RATE * seconds)
    return np.full(sample_count, marker, dtype="<i2").tobytes()


class FakeSynth:
    batch_size = 0

    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.calls = []

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        self.calls.append((text, voice_id, language))
        return self.outputs.pop(0) if self.outputs else pcm(0.1)


@pytest.mark.parametrize(
    ("text", "language", "max_chars", "expected"),
    [
        (
            "Hello world.\nHow are you? Fine!",
            "en-US",
            18,
            ["Hello world.", "How are you?", "Fine!"],
        ),
        (
            "Xin chào! Bạn khỏe không? Tốt.",
            "vi-VN",
            20,
            ["Xin chào!", "Bạn khỏe không?", "Tốt."],
        ),
    ],
)
def test_chunk_text_preserves_order_and_sentence_content(
    text, language, max_chars, expected
):
    chunks = chunk_text(text, language, max_chars=max_chars)

    assert chunks == expected
    assert all(len(chunk) <= max_chars for chunk in chunks)


def test_chunk_text_normalizes_line_endings_and_outer_whitespace():
    assert chunk_text("  Hello.\r\nWorld.  ", "en", max_chars=20) == [
        "Hello.",
        "World.",
    ]


def test_oversized_sentence_prefers_whitespace_then_uses_hard_limit():
    assert chunk_text("alpha beta gamma", "en-US", max_chars=10) == [
        "alpha beta",
        "gamma",
    ]
    assert chunk_text("abcdefghijkl", "en-US", max_chars=5) == [
        "abcde",
        "fghij",
        "kl",
    ]


def test_empty_and_oversized_text_are_rejected(tmp_path):
    with pytest.raises(InvalidTextError):
        run_text_to_voice(
            FakeSynth(), "  ", tmp_path, TextToVoiceOptions("v", "vi-VN")
        )

    with pytest.raises(TextTooLongError):
        run_text_to_voice(
            FakeSynth(),
            "x" * 50_001,
            tmp_path,
            TextToVoiceOptions("v", "vi-VN"),
        )


def test_runner_joins_only_successful_chunks_with_200ms_silence(
    tmp_path, monkeypatch
):
    def fake_encode_mp3(wav_path, mp3_path):
        Path(mp3_path).write_bytes(b"fake-mp3")

    monkeypatch.setattr(text_to_voice, "encode_mp3", fake_encode_mp3)
    synth = FakeSynth(outputs=[pcm(0.1, 1), None, pcm(0.1, 3)])

    result = run_text_to_voice(
        synth,
        "One. Two. Three.",
        tmp_path,
        TextToVoiceOptions("voice", "en"),
    )

    samples, rate = read_wav(Path(result.wav_path))
    assert rate == TTS_SAMPLE_RATE
    assert len(samples) == round(rate * 0.4)
    assert np.all(samples[: round(rate * 0.1)] == 1)
    assert np.all(samples[round(rate * 0.1) : round(rate * 0.3)] == 0)
    assert np.all(samples[round(rate * 0.3) :] == 3)
    assert result.attempted_count == 3
    assert result.spoken_count == 2
    assert any("1/3" in warning for warning in result.warnings)
    assert Path(result.mp3_path).read_bytes() == b"fake-mp3"
    assert synth.calls == [
        ("One.", "voice", "en-US"),
        ("Two.", "voice", "en-US"),
        ("Three.", "voice", "en-US"),
    ]


def test_runner_reports_each_text_stage_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(
        text_to_voice,
        "encode_mp3",
        lambda _wav_path, mp3_path: Path(mp3_path).write_bytes(b"mp3"),
    )
    updates = []

    run_text_to_voice(
        FakeSynth(outputs=[pcm(0.1)]),
        "Hello.",
        tmp_path,
        TextToVoiceOptions("voice", "en-US"),
        progress=lambda stage, fraction, message: updates.append(
            (stage, fraction, message)
        ),
    )

    first_seen = list(dict.fromkeys(stage for stage, _, _ in updates))
    assert first_seen == ["prepare", "synthesize", "assemble", "export"]
    assert updates[-1][:2] == ("export", 1.0)


def test_all_failed_chunks_raise_without_publishing_outputs(tmp_path):
    with pytest.raises(SpeechServiceError):
        run_text_to_voice(
            FakeSynth(outputs=[None, None]),
            "One. Two.",
            tmp_path,
            TextToVoiceOptions("voice", "en-US"),
        )

    assert not (tmp_path / "output.wav").exists()
    assert not (tmp_path / "output.mp3").exists()


def test_cancellation_before_synthesis_makes_no_provider_call_or_output(tmp_path):
    synth = FakeSynth(outputs=[pcm(0.1)])

    with pytest.raises(JobCancelledError):
        run_text_to_voice(
            synth,
            "Hello.",
            tmp_path,
            TextToVoiceOptions("voice", "en-US"),
            should_cancel=lambda: True,
        )

    assert synth.calls == []
    assert not (tmp_path / "output.wav").exists()
    assert not (tmp_path / "output.mp3").exists()


def test_cancellation_after_synthesis_happens_before_wav_write(tmp_path):
    synthesized = False

    class CancelAfterSynthesis:
        batch_size = 0

        def synthesize(self, text, voice_id, *, language="vi-VN"):
            nonlocal synthesized
            synthesized = True
            return pcm(0.1)

    with pytest.raises(JobCancelledError):
        run_text_to_voice(
            CancelAfterSynthesis(),
            "Hello.",
            tmp_path,
            TextToVoiceOptions("voice", "en-US"),
            should_cancel=lambda: synthesized,
        )

    assert not (tmp_path / "output.wav").exists()
    assert not (tmp_path / "output.mp3").exists()


def test_cancellation_before_mp3_encode_removes_unpublished_wav(
    tmp_path, monkeypatch
):
    cancel_now = False
    encode_calls = []

    def observe_progress(stage, fraction, _message):
        nonlocal cancel_now
        if stage == "assemble" and fraction == 1.0:
            cancel_now = True

    monkeypatch.setattr(
        text_to_voice,
        "encode_mp3",
        lambda *args: encode_calls.append(args),
    )

    with pytest.raises(JobCancelledError):
        run_text_to_voice(
            FakeSynth(outputs=[pcm(0.1)]),
            "Hello.",
            tmp_path,
            TextToVoiceOptions("voice", "en-US"),
            progress=observe_progress,
            should_cancel=lambda: cancel_now,
        )

    assert encode_calls == []
    assert not (tmp_path / "output.wav").exists()
    assert not (tmp_path / "output.mp3").exists()


def test_mp3_failure_removes_both_unpublished_outputs(tmp_path, monkeypatch):
    def fail_after_partial_write(_wav_path, mp3_path):
        Path(mp3_path).write_bytes(b"partial")
        raise FFmpegError("encode failed")

    monkeypatch.setattr(text_to_voice, "encode_mp3", fail_after_partial_write)

    with pytest.raises(FFmpegError):
        run_text_to_voice(
            FakeSynth(outputs=[pcm(0.1)]),
            "Hello.",
            tmp_path,
            TextToVoiceOptions("voice", "en-US"),
        )

    assert not (tmp_path / "output.wav").exists()
    assert not (tmp_path / "output.mp3").exists()


def test_mp3_command_uses_configured_ffmpeg_and_audio_only(tmp_path):
    command = build_mp3_cmd(
        tmp_path / "output.wav",
        tmp_path / "output.mp3",
        ffmpeg="/bin/ffmpeg",
    )

    assert command[:4] == ["/bin/ffmpeg", "-y", "-loglevel", "error"]
    assert command[-1].endswith("output.mp3")
    codec = command.index("-codec:a")
    assert command[codec : codec + 2] == ["-codec:a", "libmp3lame"]
    assert command[command.index("-map") : command.index("-map") + 2] == [
        "-map",
        "0:a:0",
    ]


def test_ffmpeg_exports_valid_mono_24khz_mp3(tmp_path):
    try:
        ffmpeg = resolve("ffmpeg")
        ffprobe = resolve("ffprobe")
    except FFmpegNotFoundError as exc:
        pytest.skip(str(exc))

    wav_path = write_wav(
        tmp_path / "source.wav",
        np.ones(round(TTS_SAMPLE_RATE * 0.1), dtype="<i2"),
    )
    mp3_path = tmp_path / "output.mp3"

    encode_mp3(wav_path, mp3_path)
    probe = run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(mp3_path),
        ],
        timeout=120,
    )
    streams = json.loads(probe.stdout)["streams"]
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")

    assert audio["codec_name"] == "mp3"
    assert audio["sample_rate"] == "24000"
    assert audio["channels"] == 1
    assert build_mp3_cmd(wav_path, mp3_path, ffmpeg=ffmpeg)[0] == ffmpeg
