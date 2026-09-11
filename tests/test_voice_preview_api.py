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

    assert voice["preview_url"].startswith(
        "/api/voices/en-US-AvaMultilingualNeural/preview?language=en-US&v="
    )
    assert voice["preview_urls"]["en-US"] == voice["preview_url"]
    assert voice["preview_status"]["en-US"] == "ready"
    assert set(voice["preview_urls"]) == {"vi-VN", "en-US"}


def test_voice_routes_reject_unknown_language_as_bad_request(client):
    listed = client.get("/api/voices", params={"language": "xx-YY"})
    preview = client.get(
        "/api/voices/en-US-AvaMultilingualNeural/preview",
        params={"language": "xx-YY"},
    )

    assert listed.status_code == 400
    assert preview.status_code == 400


def test_replacing_preview_changes_catalog_url_and_serves_new_audio(client, tmp_path, monkeypatch):
    store = _configure_store(tmp_path, monkeypatch)
    voice_id = "en-US-AvaMultilingualNeural"
    path = store.path(voice_id, "en-US")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"RIFFold")

    def catalog_url():
        return next(v for v in client.get("/api/voices?language=en-US").json()
                    if v["id"] == voice_id)["preview_url"]

    old_url = catalog_url()
    assert catalog_url() == old_url
    assert client.get(old_url).content == b"RIFFold"
    path.write_bytes(b"RIFFreplacement")
    new_url = catalog_url()
    assert new_url != old_url
    assert client.get(new_url).content == b"RIFFreplacement"


def test_preview_requires_cache_revalidation(client, tmp_path, monkeypatch):
    store = _configure_store(tmp_path, monkeypatch)
    path = store.path("en-US-AvaMultilingualNeural", "en-US")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"RIFFaudio")
    response = client.get("/api/voices/en-US-AvaMultilingualNeural/preview?language=en-US")
    assert response.headers.get("cache-control") == "no-cache"


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


def test_timed_out_preview_keeps_provider_busy_until_synthesis_finishes(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from pipeline.speech_runtime import SpeechActivity

    entered, release = Event(), Event()
    activity = SpeechActivity()
    monkeypatch.setattr(voices, "speech_activity", activity)
    monkeypatch.setattr(voices, "_PREVIEW_TIMEOUT_SECONDS", 0.05)

    def slow_synthesis(*args):
        entered.set()
        assert release.wait(3)
        return _pcm()

    monkeypatch.setattr(voices, "_synthesize_preview_text", slow_synthesis)
    with ThreadPoolExecutor(max_workers=2) as executor:
        monkeypatch.setattr(voices, "_preview_executor", executor)
        try:
            response = client.post(
                "/api/voices/en-US-AvaMultilingualNeural/preview-text",
                json={"text": "Hello", "language": "en-US"},
            )
            assert response.status_code == 504
            assert entered.is_set()
            second = client.post(
                "/api/voices/en-US-AvaMultilingualNeural/preview-text",
                json={"text": "Hello", "language": "en-US"},
            )
            assert second.status_code == 409
            with pytest.raises(PreviewBusyError):
                with activity.preview("edge"):
                    pass
        finally:
            release.set()
    with activity.preview("edge"):
        pass


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


def test_custom_voice_creation_dispatches_default_vietnamese_preview(
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
    assert store.status(voice_id, "en-US") == "error"
    assert len(calls) == 1


def test_failed_text_preview_releases_provider_for_retry(client, monkeypatch):
    from pipeline.speech_runtime import SpeechActivity
    activity = SpeechActivity()
    monkeypatch.setattr(voices, "speech_activity", activity)

    def fail(*args):
        raise RuntimeError("synthesis failed")

    monkeypatch.setattr(voices, "_synthesize_preview_text", fail)
    with pytest.raises(RuntimeError, match="synthesis failed"):
        client.post(
            "/api/voices/en-US-AvaMultilingualNeural/preview-text",
            json={"text": "Hello", "language": "en-US"},
        )
    with activity.preview("edge"):
        pass
