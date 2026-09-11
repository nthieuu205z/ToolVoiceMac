"""Language/tag persistence and API side effects, with only synthesis replaced."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend.config import settings
from backend.routes import voices
from pipeline import custom_voices
from pipeline.audio import write_wav
from pipeline.voice_previews import VoicePreviewStore
from pipeline.voices import available_voices


@pytest.fixture
def voice_api(client, monkeypatch, tmp_path):
    monkeypatch.setattr(type(settings), "resolved_clone_provider", property(lambda self: "omnivoice"))
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(voices, "decode_to_pcm", lambda raw, rate: b"\0\0" * rate * 3)
    monkeypatch.setattr(voices, "preview_store", VoicePreviewStore(tmp_path / "previews"))
    submitted = []
    monkeypatch.setattr(voices._preview_executor, "submit", lambda fn, *args: submitted.append((fn, args)) or SimpleNamespace())
    return client, submitted


def upload(client, **data):
    return client.post("/api/voices/custom", data={"name": "Narrator", **data}, files={"audio": ("sample.wav", b"sample", "audio/wav")})


def test_new_voice_persists_selected_language_tags_and_generates_only_that_preview(voice_api, monkeypatch):
    client, submitted = voice_api
    response = upload(client, language="en-US", tags=json.dumps([" Warm ", "Narration", "Warm"]))
    assert response.status_code == 200
    body = response.json()
    assert body["supported_languages"] == ["en-US"]
    assert body["tags"] == ["Warm", "Narration"]
    voice_id = body["id"]
    disk = json.loads(custom_voices.safe_metadata_path(voice_id).read_text())
    assert disk["supported_languages"] == ["en-US"]
    assert disk["tags"] == ["Warm", "Narration"]
    assert not any(v.id == voice_id for v in available_voices("edge", "omnivoice", "vi-VN"))
    listed = next(v for v in client.get("/api/voices?language=en-US").json() if v["id"] == voice_id)
    assert listed["tags"] == ["Warm", "Narration"]
    generated = []
    monkeypatch.setattr(voices, "_preview_backend", lambda vid: ("omnivoice", SimpleNamespace(synthesize=lambda text, voice, language: generated.append(language) or b"\0\0" * 100)))
    for fn, args in submitted:
        fn(*args)
    assert generated == ["en-US"]
    assert voices.preview_store.status(voice_id, "en-US") == "ready"
    assert not voices.preview_store.path(voice_id, "vi-VN").exists()


def test_new_voice_defaults_to_vietnamese_only(voice_api):
    client, _ = voice_api
    body = upload(client).json()
    assert body["supported_languages"] == ["vi-VN"]
    assert body["tags"] == []
    assert set(body["preview_status"]) == {"vi-VN"}


@pytest.mark.parametrize("data", [{"language": "fr-FR"}, {"tags": "not-json"}, {"tags": '"Warm"'}, {"tags": '[2]'}, {"tags": json.dumps(["x" * 25])}, {"tags": json.dumps([str(i) for i in range(9)])}, {"tags": '[""]'}])
def test_invalid_metadata_is_rejected_before_sample_is_saved(voice_api, data):
    client, submitted = voice_api
    assert upload(client, **data).status_code == 400
    assert custom_voices.list_custom() == []
    assert not list(custom_voices._dir.glob("*.wav"))
    assert submitted == []


def test_legacy_voice_stays_bilingual_until_explicit_language_edit(voice_api):
    client, submitted = voice_api
    voice_id = "clone-legacy"
    write_wav(custom_voices.sample_path(voice_id), np.zeros(10, dtype="<i2"))
    custom_voices.safe_metadata_path(voice_id).write_text(json.dumps({"id": voice_id, "display_name": "Old", "created_at": 123}))
    assert custom_voices.get(voice_id).supported_languages == ("vi-VN", "en-US")
    sample = custom_voices.sample_path(voice_id).read_bytes()
    response = client.patch(f"/api/voices/custom/{voice_id}", json={"name": " Renamed ", "tags": ["Deep"]})
    assert response.status_code == 200
    assert response.json()["supported_languages"] == ["vi-VN", "en-US"]
    assert custom_voices.get(voice_id).display_name == "Renamed"
    response = client.patch(f"/api/voices/custom/{voice_id}", json={"language": "en-US"})
    assert response.status_code == 200
    assert custom_voices.get(voice_id).supported_languages == ("en-US",)
    assert custom_voices.get(voice_id).created_at == 123
    assert custom_voices.get(voice_id).tags == ("Deep",)
    assert custom_voices.sample_path(voice_id).read_bytes() == sample
    assert submitted == []
    assert not any(v["id"] == voice_id for v in client.get("/api/voices?language=vi-VN").json())


@pytest.mark.parametrize("data", [{"name": " "}, {"name": "x" * 41}, {"language": "xx"}, {"tags": ["x" * 25]}, {"tags": None}, {"language": None}, {"name": None}, {"created_at": 0}])
def test_invalid_patch_does_not_change_metadata(voice_api, data):
    client, _ = voice_api
    voice_id = upload(client).json()["id"]
    original = custom_voices.safe_metadata_path(voice_id).read_bytes()
    assert client.patch(f"/api/voices/custom/{voice_id}", json=data).status_code in {400, 422}
    assert custom_voices.safe_metadata_path(voice_id).read_bytes() == original


def test_patch_unknown_voice_returns_404(voice_api):
    client, _ = voice_api
    assert client.patch("/api/voices/custom/clone-missing", json={"name": "New"}).status_code == 404


def test_queued_creation_preview_keeps_selected_language_after_metadata_edit(voice_api, monkeypatch):
    client, submitted = voice_api
    voice_id = upload(client, language="vi-VN").json()["id"]
    response = client.patch(f"/api/voices/custom/{voice_id}", json={"language": "en-US"})
    assert response.status_code == 200
    assert len(submitted) == 1
    generated = []
    monkeypatch.setattr(
        voices, "_preview_backend",
        lambda vid: ("omnivoice", SimpleNamespace(
            synthesize=lambda text, voice, language: generated.append(language) or b"\0\0" * 100
        )),
    )
    for fn, args in submitted:
        fn(*args)
    assert generated == ["vi-VN"]
    assert voices.preview_store.status(voice_id, "vi-VN") == "ready"
    assert not voices.preview_store.path(voice_id, "en-US").exists()
    assert custom_voices.get(voice_id).supported_languages == ("en-US",)
