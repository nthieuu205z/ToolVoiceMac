"""Opt-in calls to real speech and translation providers."""

from __future__ import annotations

import os

import pytest

from backend.config import settings
from pipeline.backends import build_backend
from pipeline.edge_speech import EdgeSynthesizer

pytestmark = pytest.mark.provider_smoke


def _require_smoke_enabled() -> None:
    if os.getenv("RUN_PROVIDER_SMOKE") != "1":
        pytest.skip("set RUN_PROVIDER_SMOKE=1 to call real providers")


def _assert_pcm(pcm: bytes) -> None:
    assert len(pcm) >= 2
    assert len(pcm) % 2 == 0


def test_real_edge_english_returns_pcm():
    _require_smoke_enabled()
    pcm = EdgeSynthesizer().synthesize(
        "This is an English smoke test.",
        "en-US-AvaMultilingualNeural",
        language="en-US",
    )
    _assert_pcm(pcm)


@pytest.mark.parametrize(
    ("language", "text"),
    [("vi-VN", "Xin chào."), ("en-US", "Hello.")],
)
def test_real_omnivoice_clone_returns_pcm(language, text):
    _require_smoke_enabled()
    voice_id = os.getenv("PROVIDER_SMOKE_CLONE_VOICE_ID")
    if not voice_id:
        pytest.skip("set PROVIDER_SMOKE_CLONE_VOICE_ID to an authorized local clone")
    backend = build_backend(settings.provider_config_for("omnivoice"))
    _assert_pcm(backend.synthesize(text, voice_id, language=language))


@pytest.mark.parametrize(
    ("target", "source", "expected_token"),
    [("vi-VN", "Hello", "xin chào"), ("en-US", "Xin chào", "hello")],
)
def test_real_gemini_translation_is_aligned(target, source, expected_token):
    _require_smoke_enabled()
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY is not configured")
    backend = build_backend(settings.provider_config_for("edge"))
    translated = backend.translate([source], [1.0], target_language=target)
    assert len(translated) == 1
    assert expected_token in translated[0].strip().casefold()
