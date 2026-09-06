from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.config import settings
from backend.routes import voices
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.speech_runtime import PreviewBusyError
from pipeline.voice_previews import VoicePreviewStore


def _pcm() -> bytes:
    return b"\x00\x00" * (TTS_SAMPLE_RATE // 10)


def _configure_store(tmp_path, monkeypatch) -> VoicePreviewStore:
    store = VoicePreviewStore(tmp_path)
    monkeypatch.setattr(voices, "preview_store", store)
    return store


def test_voice_list_exposes_language_preview_maps_and_compatibility_url(
    client, tmp_path, monkeypatch
):
    store = _configure_store(tmp_path, monkeypatch)
    path = store.path("en-US-AvaMultilingualNeural", "en-US")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"RIFFenglish")

    voice = next(
        item
        for item in client.get("/api/voices", params={"language": "en-US"}).json()
        if item["id"] == "en-US-AvaMultilingualNeural"
    )

    assert voice["preview_url"].endswith(
        "/api/voices/en-US-AvaMultilingualNeural/preview?language=en-US"
    )
    assert voice["preview_urls"]["en-US"] == voice["preview_url"]
    assert voice["preview_status"]["en-US"] == "ready"
    assert set(voice["preview_urls"]) == {"vi-VN", "en-US"}


def test_preview_endpoint_defaults_to_vietnamese_and_accepts_explicit_language(
    client, tmp_path, monkeypatch
):
    store = _configure_store(tmp_path, monkeypatch)
    vi = store.path("en-US-AvaMultilingualNeural", "vi-VN")
    en = store.path("en-US-AvaMultilingualNeural", "en-US")
    vi.parent.mkdir(parents=True)
    vi.write_bytes(b"RIFFvietnamese")
    en.write_bytes(b"RIFFenglish")

    default = client.get("/api/voices/en-US-AvaMultilingualNeural/preview")
    english = client.get(
        "/api/voices/en-US-AvaMultilingualNeural/preview",
        params={"language": "en-US"},
    )

    assert default.content == b"RIFFvietnamese"
    assert english.content == b"RIFFenglish"
    assert 'filename="en-US-AvaMultilingualNeural-en-US.wav"' in english.headers[
        "content-disposition"
    ]


def test_text_preview_truncates_to_first_sentence_and_returns_wav(
    client, monkeypatch
):
    seen = []
    monkeypatch.setattr(
        voices,
        "_synthesize_preview_text",
        lambda text, voice, language: seen.append((text, voice, language)) or _pcm(),
    )

    response = client.post(
        "/api/voices/en-US-AvaMultilingualNeural/preview-text",
        json={"text": "First sentence. Second sentence.", "language": "en-US"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    assert seen == [
        ("First sentence.", "en-US-AvaMultilingualNeural", "en-US")
    ]


def test_text_preview_uses_exactly_300_characters_without_an_early_terminator(
    client, monkeypatch
):
    seen = []
    monkeypatch.setattr(
        voices,
        "_synthesize_preview_text",
        lambda text, voice, language: seen.append(text) or _pcm(),
    )

    response = client.post(
        "/api/voices/en-US-AvaMultilingualNeural/preview-text",
        json={"text": "x" * 400, "language": "en-US"},
    )

    assert response.status_code == 200
    assert seen == ["x" * 300]


def test_text_preview_rejects_empty_and_unsupported_voice_language_pair(client):
    empty = client.post(
        "/api/voices/en-US-AvaMultilingualNeural/preview-text",
        json={"text": "  ", "language": "en-US"},
    )
    unsupported = client.post(
        "/api/voices/vi-VN-HoaiMyNeural/preview-text",
        json={"text": "Hello", "language": "en-US"},
    )

    assert empty.status_code == 400
    assert unsupported.status_code == 400


def test_busy_preview_returns_retryable_409(client, monkeypatch):
    @contextmanager
    def busy_preview_context(provider):
        raise PreviewBusyError(provider)
        yield

    monkeypatch.setattr(voices.speech_activity, "preview", busy_preview_context)
    response = client.post(
        "/api/voices/en-US-AvaMultilingualNeural/preview-text",
        json={"text": "Hello", "language": "en-US"},
    )

    assert response.status_code == 409
    assert "thử lại" in response.json()["detail"].lower()


def test_text_preview_timeout_returns_504(client, monkeypatch):
    class TimedOutFuture:
        def result(self, timeout):
            raise TimeoutError

    monkeypatch.setattr(
        voices._preview_executor,
        "submit",
        lambda *args, **kwargs: TimedOutFuture(),
    )

    response = client.post(
        "/api/voices/en-US-AvaMultilingualNeural/preview-text",
        json={"text": "Hello", "language": "en-US"},
    )

    assert response.status_code == 504


def test_preview_regeneration_transitions_to_pending_and_rejects_duplicate(
    client, tmp_path, monkeypatch
):
    store = _configure_store(tmp_path, monkeypatch)
    dispatched = []
    monkeypatch.setattr(
        voices._preview_executor,
        "submit",
        lambda fn, *args: dispatched.append((fn, args)) or SimpleNamespace(),
    )

    first = client.post(
        "/api/voices/en-US-AvaMultilingualNeural/preview/regenerate",
        params={"language": "en-US"},
    )
    second = client.post(
        "/api/voices/en-US-AvaMultilingualNeural/preview/regenerate",
        params={"language": "en-US"},
    )

    assert first.status_code == 202
    assert first.json() == {"status": "pending"}
    assert second.status_code == 409
    assert store.status("en-US-AvaMultilingualNeural", "en-US") == "pending"
    assert len(dispatched) == 1


def test_custom_voice_creation_dispatches_both_language_previews(
    client, tmp_path, monkeypatch
):
    store = _configure_store(tmp_path / "previews", monkeypatch)
    monkeypatch.setattr(settings, "clone_tts_provider", "omnivoice")
    monkeypatch.setattr(
        type(settings), "resolved_clone_provider", property(lambda self: "omnivoice")
    )
    monkeypatch.setattr(
        voices,
        "decode_to_pcm",
        lambda raw, rate: b"\x00\x00" * (rate * 5),
    )
    calls = []
    monkeypatch.setattr(
        voices._preview_executor,
        "submit",
        lambda fn, *args: calls.append((fn, args)) or SimpleNamespace(),
    )

    response = client.post(
        "/api/voices/custom",
        data={"name": "Bilingual Test"},
        files={"audio": ("sample.wav", b"sample", "audio/wav")},
    )

    assert response.status_code == 200
    voice_id = response.json()["id"]
    assert store.status(voice_id, "vi-VN") == "pending"
    assert store.status(voice_id, "en-US") == "pending"
    assert len(calls) == 1
