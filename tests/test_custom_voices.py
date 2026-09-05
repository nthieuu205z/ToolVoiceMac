"""Nhân bản giọng từ audio mẫu: kho lưu, danh sách giọng, engine, và API.

Kho giọng được fixture autouse trong conftest cách ly sẵn vào thư mục tạm.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app
from pipeline import custom_voices
from pipeline.audio import write_wav
from pipeline.omnivoice_speech import OmniVoiceSynthesizer
from pipeline.voices import is_valid, native_id, voices_for


def _seed(name: str = "Giọng Của Tôi") -> custom_voices.CustomVoice:
    voice_id = custom_voices.unique_id(name)
    write_wav(custom_voices.sample_path(voice_id), np.zeros(24_000, dtype="<i2"), 24_000)
    return custom_voices.register(voice_id, name)


# ─── kho lưu ───

def test_vietnamese_names_become_url_safe_slugs():
    assert custom_voices.slugify("Giọng Của Tôi") == "giong-cua-toi"
    assert custom_voices.slugify("Đông Đô 2026!") == "dong-do-2026"
    assert custom_voices.slugify("😀😀") == "giong"  # không còn gì thì vẫn phải có slug


def test_a_registered_voice_appears_in_the_store():
    voice = _seed()
    assert custom_voices.get(voice.id).display_name == "Giọng Của Tôi"
    assert custom_voices.is_custom(voice.id)


def test_register_requires_a_safe_id_and_existing_sample():
    with pytest.raises(ValueError):
        custom_voices.register("clone-../escape", "Bad")
    with pytest.raises(FileNotFoundError):
        custom_voices.register("clone-missing", "Missing")


def test_a_voice_without_its_sample_file_does_not_exist():
    """Mất file mẫu thì engine không học được gì — giọng phải biến khỏi danh sách."""
    voice = _seed()
    custom_voices.sample_path(voice.id).unlink()
    assert custom_voices.get(voice.id) is None


def test_removing_a_voice_never_recycles_its_id():
    """Engine giữ giọng đã đăng ký trong RAM theo id — id cũ dùng lại sẽ đọc bằng giọng CŨ."""
    first = _seed("Test")
    custom_voices.remove(first.id)
    second = _seed("Test")
    assert second.id != first.id


def test_duplicate_names_get_distinct_ids():
    a, b = _seed("Trùng Tên"), _seed("Trùng Tên")
    assert a.id != b.id


# ─── hòa vào danh sách giọng ───

def test_clones_show_up_only_on_the_omnivoice_provider():
    voice = _seed()
    assert any(v.id == voice.id for v in voices_for("omnivoice"))
    assert not any(v.id == voice.id for v in voices_for("edge"))


def test_a_clone_id_passes_job_validation():
    """create_job chặn giọng lạ bằng is_valid — giọng nhân bản phải qua được cửa này."""
    voice = _seed()
    assert is_valid(voice.id, "omnivoice") is True
    assert native_id(voice.id, "omnivoice") == voice.id


# ─── engine ───

def test_omnivoice_is_the_only_clone_implementation():
    assert OmniVoiceSynthesizer.__module__ == "pipeline.omnivoice_speech"


# ─── API ───

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "edge")
    # Không nạp engine thật trong test: chặn luồng tạo nghe thử và bước decode ffmpeg.
    monkeypatch.setattr("backend.routes.voices._generate_preview", lambda vid: None)
    monkeypatch.setattr("backend.routes.voices.decode_to_pcm",
                        lambda raw, rate: b"\x00\x00" * (rate * 5))  # 5 giây im lặng
    yield TestClient(app)


def _upload(client, name="Giọng Test"):
    return client.post("/api/voices/custom",
                       data={"name": name},
                       files={"audio": ("mau.mp3", b"fake-mp3-bytes", "audio/mpeg")})


def test_cloning_enabled_whenever_a_clone_engine_exists(client, monkeypatch):
    # Có OmniVoice → bật, kể cả khi preset là edge.
    assert client.get("/api/voices/cloning").json() == {"enabled": True}
    monkeypatch.setattr(settings, "tts_provider", "edge")
    assert client.get("/api/voices/cloning").json() == {"enabled": True}
    # Tắt hẳn nhân bản.
    monkeypatch.setattr(settings, "clone_tts_provider", "none")
    assert client.get("/api/voices/cloning").json() == {"enabled": False}


def test_clone_upload_does_not_depend_on_preset_provider(client, monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "gemini")

    response = _upload(client)

    assert response.status_code == 200
    assert response.json()["custom"] is True


def test_uploading_a_sample_creates_a_selectable_voice(client):
    body = _upload(client).json()
    assert body["custom"] is True

    voices = client.get("/api/voices").json()
    mine = next(v for v in voices if v["id"] == body["id"])
    assert "Giọng Test" in mine["display_name"]
    assert mine["custom"] is True


def test_cloning_works_on_any_preset_provider(client, monkeypatch):
    # Định tuyến: preset=edge nhưng giọng nhân bản vẫn tạo được (sẽ đọc bằng engine clone).
    monkeypatch.setattr(settings, "tts_provider", "edge")
    assert _upload(client).json()["custom"] is True


def test_cloning_is_refused_only_when_disabled(client, monkeypatch):
    monkeypatch.setattr(settings, "clone_tts_provider", "none")
    assert _upload(client).status_code == 400


def test_a_sample_shorter_than_two_seconds_is_refused(client, monkeypatch):
    monkeypatch.setattr("backend.routes.voices.decode_to_pcm",
                        lambda raw, rate: b"\x00\x00" * rate)  # 1 giây
    response = _upload(client)
    assert response.status_code == 400
    assert "giây" in response.json()["detail"]


def test_garbage_that_ffmpeg_cannot_decode_is_refused(client, monkeypatch):
    from pipeline.errors import FFmpegError

    def boom(raw, rate):
        raise FFmpegError("not audio")

    monkeypatch.setattr("backend.routes.voices.decode_to_pcm", boom)
    assert _upload(client).status_code == 400


def test_a_long_sample_is_trimmed_to_twelve_seconds(client, monkeypatch):
    monkeypatch.setattr("backend.routes.voices.decode_to_pcm",
                        lambda raw, rate: b"\x00\x00" * (rate * 60))  # 1 phút
    body = _upload(client).json()

    from pipeline.audio import read_wav

    samples, rate = read_wav(custom_voices.sample_path(body["id"]))
    assert len(samples) / rate == pytest.approx(12.0)


def test_deleting_a_voice_removes_it_everywhere(client):
    voice_id = _upload(client).json()["id"]
    assert client.delete(f"/api/voices/custom/{voice_id}").status_code == 200

    assert custom_voices.get(voice_id) is None
    assert not any(v["id"] == voice_id for v in client.get("/api/voices").json())


def test_deleting_an_unknown_voice_is_a_404(client):
    assert client.delete("/api/voices/custom/clone-khong-co").status_code == 404


def test_a_preset_voice_cannot_be_deleted_through_this_route(client):
    assert client.delete("/api/voices/custom/truc-ly").status_code == 404