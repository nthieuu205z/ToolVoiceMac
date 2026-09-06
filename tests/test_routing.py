"""Định tuyến giọng theo loại: giọng nhân bản -> engine clone; giọng dựng sẵn -> tts_provider.

Mỗi job dùng một giọng, nên chọn engine một lần theo voice_id.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import custom_voices
from pipeline.audio import write_wav
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.voices import available_voices, default_voice, is_available, route_provider


def _make_clone(voice_id: str = "clone-x", display: str = "X") -> str:
    write_wav(custom_voices.sample_path(voice_id), np.zeros(2400, dtype="<i2"), TTS_SAMPLE_RATE)
    custom_voices.register(voice_id, display)
    return voice_id


# ── route_provider ───────────────────────────────────────────────────────────
def test_clone_voice_routes_to_clone_engine():
    assert route_provider("clone-abc", tts_provider="edge", clone_provider="omnivoice") == "omnivoice"


def test_preset_voice_routes_to_tts_provider():
    assert route_provider("vi-VN-HoaiMyNeural", tts_provider="edge", clone_provider="omnivoice") == "edge"
    assert route_provider("Charon", tts_provider="gemini", clone_provider="omnivoice") == "gemini"


def test_clone_voice_always_routes_to_omnivoice():
    assert route_provider("clone-abc", tts_provider="edge", clone_provider="omnivoice") == "omnivoice"


def test_clone_voice_without_clone_engine_falls_back_to_tts_provider():
    # Khi clone tắt, caller sẽ từ chối custom voice; hàm vẫn giữ provider preset cho tương thích.
    assert route_provider("clone-abc", tts_provider="edge", clone_provider=None) == "edge"


def test_unknown_provider_is_rejected_instead_of_misrouting():
    with pytest.raises(ValueError):
        route_provider("Charon", tts_provider="legacy", clone_provider="omnivoice")


# ── available_voices / is_available ──────────────────────────────────────────
def test_available_voices_adds_clones_even_with_non_clone_preset_engine():
    _make_clone("clone-x", "X")
    ids = [v.id for v in available_voices(tts_provider="edge", clone_provider="omnivoice")]
    assert "clone-x" in ids                       # clone hiện ra dù preset engine là edge
    assert "vi-VN-HoaiMyNeural" in ids            # vẫn có preset edge
    assert "truc-ly" not in ids                   # không rò preset của engine khác


def test_available_voices_no_clone_engine_hides_clones():
    _make_clone("clone-x", "X")
    ids = [v.id for v in available_voices(tts_provider="edge", clone_provider=None)]
    assert "clone-x" not in ids
    assert "vi-VN-HoaiMyNeural" in ids


def test_is_available_accepts_clone_when_engine_present():
    _make_clone("clone-x", "X")
    assert is_available("clone-x", tts_provider="edge", clone_provider="omnivoice")
    assert not is_available("clone-x", tts_provider="edge", clone_provider=None)
    assert is_available("vi-VN-HoaiMyNeural", tts_provider="edge", clone_provider=None)
    assert not is_available("khong-ton-tai", tts_provider="edge", clone_provider="omnivoice")


def test_available_voices_filters_by_canonical_language():
    vietnamese = available_voices("edge", "omnivoice", language="vi-VN")
    english = available_voices("edge", "omnivoice", language="en-US")

    assert all(voice.supports("vi-VN") for voice in vietnamese)
    assert all(voice.supports("en-US") for voice in english)
    assert "vi-VN-HoaiMyNeural" not in {voice.id for voice in english}
    assert "en-US-AvaMultilingualNeural" in {voice.id for voice in english}


def test_custom_voice_supports_vietnamese_and_english():
    voice_id = _make_clone("clone-multi", "Multi")
    voice = next(v for v in available_voices("edge", "omnivoice") if v.id == voice_id)
    assert voice.supported_languages == ("vi-VN", "en-US")


def test_is_available_rejects_a_voice_that_cannot_speak_the_language():
    assert not is_available(
        "vi-VN-HoaiMyNeural", "edge", None, language="en-US",
    )
    assert is_available(
        "en-US-AvaMultilingualNeural", "edge", None, language="en-US",
    )


# ── default_voice ────────────────────────────────────────────────────────────
def test_default_voice_returns_a_preset_for_normal_providers():
    assert default_voice("edge").startswith("vi-VN")   # có giọng dựng sẵn


def test_default_voice_returns_the_first_voice_for_the_requested_language():
    assert default_voice("edge", language="en-US") == "en-US-AvaMultilingualNeural"


def test_default_voice_on_omnivoice_without_clones_raises_clear_error():
    # OmniVoice không có giọng dựng sẵn; chưa có giọng nhân bản → không có mặc định.
    # Trước đây nổ IndexError khó hiểu; nay phải là lỗi rõ ràng, hành động được.
    with pytest.raises(ValueError):
        default_voice("omnivoice")


def test_default_voice_on_omnivoice_returns_a_clone_when_present():
    _make_clone("clone-d", "D")
    assert default_voice("omnivoice") == "clone-d"
