from __future__ import annotations

import io
import wave

import pytest

from pipeline.audio import pcm_to_wav_bytes
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.voice_previews import VoicePreviewStore


def test_preview_paths_are_separate_per_language(tmp_path):
    store = VoicePreviewStore(tmp_path)

    assert store.path("clone-safe", "vi-VN") == tmp_path / "clone-safe" / "vi-VN.wav"
    assert store.path("clone-safe", "en-US") == tmp_path / "clone-safe" / "en-US.wav"


def test_unknown_voice_or_language_cannot_escape_preview_root(tmp_path):
    store = VoicePreviewStore(tmp_path)

    with pytest.raises(ValueError):
        store.path("../../outside", "vi-VN")
    with pytest.raises(ValueError):
        store.path("clone-safe", "../../outside")


def test_existing_preview_is_ready_and_missing_preview_uses_recorded_status(tmp_path):
    store = VoicePreviewStore(tmp_path)
    path = store.path("clone-safe", "vi-VN")

    assert store.status("clone-safe", "vi-VN") == "error"
    store.set_pending("clone-safe", "vi-VN")
    assert store.status("clone-safe", "vi-VN") == "pending"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"RIFF")
    assert store.status("clone-safe", "vi-VN") == "ready"


def test_pcm_to_wav_bytes_returns_mono_16bit_project_rate():
    pcm = b"\x00\x00" * (TTS_SAMPLE_RATE // 10)

    value = pcm_to_wav_bytes(pcm)

    assert value.startswith(b"RIFF")
    with wave.open(io.BytesIO(value), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == TTS_SAMPLE_RATE
        assert wav.readframes(wav.getnframes()) == pcm
