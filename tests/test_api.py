"""Kiểm tra các route mà giao diện gọi tới, không đụng ffmpeg lẫn Gemini."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.job_contracts import JobArtifact
from backend.job_manager import Job, manager
from backend.main import app
from pipeline.voices import voices_for

VOICES = voices_for(settings.tts_provider)


def _put(job: Job) -> Job:
    manager._jobs[job.id] = job
    return job


@pytest.fixture
def client():
    yield TestClient(app)
    manager._jobs.clear()  # dọn sổ đăng ký giữa các test
    manager._futures.clear()


def test_voices_endpoint_shape_matches_frontend_expectations(client):
    body = client.get("/api/voices").json()
    # Tính trong thân test: danh sách giọng phụ thuộc kho giọng nhân bản mà fixture
    # autouse đã cách ly — VOICES ở cấp module được tính TRƯỚC khi fixture chạy.
    assert len(body) == len(voices_for(settings.tts_provider))
    assert {"id", "display_name", "preview_url", "custom"} <= set(body[0])
    assert body[0]["provider"] == settings.tts_provider
    assert body[0]["supported_languages"] == ["vi-VN"]


def test_languages_endpoint_lists_supported_workflows(client):
    assert client.get("/api/languages").json() == [
        {
            "code": "vi-VN",
            "display_name": "Tiếng Việt",
            "english_name": "Vietnamese",
            "video_dubbing": True,
            "text_to_voice": True,
        },
        {
            "code": "en-US",
            "display_name": "English (US)",
            "english_name": "English",
            "video_dubbing": True,
            "text_to_voice": True,
        },
    ]


def test_voices_endpoint_filters_by_language(client, monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "edge")

    body = client.get("/api/voices", params={"language": "en"}).json()

    assert "vi-VN-HoaiMyNeural" not in {voice["id"] for voice in body}
    assert "en-US-AvaMultilingualNeural" in {voice["id"] for voice in body}
    assert all("en-US" in voice["supported_languages"] for voice in body)


def test_voice_preview_endpoint_serves_a_known_preview(client, tmp_path, monkeypatch):
    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: tmp_path))
    preview = tmp_path / "vi-VN-HoaiMyNeural.wav"
    preview.write_bytes(b"RIFFfake")

    response = client.get("/api/voices/vi-VN-HoaiMyNeural/preview")

    assert response.status_code == 200
    assert response.content == b"RIFFfake"


def test_voice_preview_endpoint_rejects_unknown_voice(client, tmp_path, monkeypatch):
    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: tmp_path))

    response = client.get("/api/voices/not-a-real-voice/preview")

    assert response.status_code == 404


def test_voice_list_follows_the_configured_provider(client, monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "gemini")
    assert len(client.get("/api/voices").json()) == 10

    monkeypatch.setattr(settings, "tts_provider", "edge")
    ids = [v["id"] for v in client.get("/api/voices").json()]
    assert ids[:2] == ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]
    assert len(ids) == 14


def test_edge_offers_native_and_multilingual_voices(client, monkeypatch):
    """12 giọng "Multilingual" đọc được tiếng Việt — đã kiểm chứng bằng Whisper nghe lại."""
    monkeypatch.setattr(settings, "tts_provider", "edge")
    ids = [v["id"] for v in client.get("/api/voices").json()]

    assert sum(1 for i in ids if i.startswith("vi-VN-")) == 2
    assert sum(1 for i in ids if "Multilingual" in i) == 12
    assert len(set(ids)) == len(ids)  # không trùng lặp


def test_a_voice_from_the_other_provider_is_rejected(client, monkeypatch):
    """Chọn giọng Gemini trong khi đang chạy edge-tts thì phải chặn, không để hỏng giữa chừng."""
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "tts_provider", "edge")

    response = client.post(
        "/api/jobs",
        files={"video": ("a.mp4", b"data", "video/mp4")},
        data={"voice_id": "Charon"},
    )
    assert response.status_code == 400


def test_index_page_is_served_at_root(client):
    # Không bám vào tên thương hiệu (thiết kế đổi được) — bám vào chức năng của trang.
    page = client.get("/").text
    assert 'src="app.js?v=' in page


def test_current_job_is_empty_when_idle(client):
    assert client.get("/api/jobs/current").json() == {}


def test_unknown_job_returns_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_job_telemetry_endpoint_returns_monitor_payload(client, tmp_path):
    job = Job(
        id="telemetry",
        filename="clip.mp4",
        workdir=tmp_path,
        voice_id="Kore",
        status="running",
        started_at=100.0,
        stage_started_at=110.0,
        stage_fraction=0.5,
        percent=50.0,
        engine="edge",
        device="network",
    )
    manager._jobs[job.id] = job

    response = client.get("/api/jobs/telemetry/telemetry")

    assert response.status_code == 200
    assert response.json()["telemetry_only"] is True
    assert response.json()["engine"] == "edge"


def test_rejects_unknown_voice(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    response = client.post(
        "/api/jobs",
        files={"video": ("a.mp4", b"data", "video/mp4")},
        data={"voice_id": "KhongCoThat"},
    )
    assert response.status_code == 400


def test_rejects_upload_when_api_key_missing(client, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    response = client.post(
        "/api/jobs",
        files={"video": ("a.mp4", b"data", "video/mp4")},
        data={"voice_id": VOICES[0].id},
    )
    assert response.status_code == 500
    assert "dịch vụ dịch thuật" in response.json()["detail"]


def test_a_second_upload_is_accepted_while_a_job_runs(client, monkeypatch, tmp_path):
    """Nhiều video cùng lúc: video thứ hai vào danh sách chứ không bị 409 như trước."""
    from pipeline.models import MediaInfo, PipelineResult

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    # KHÔNG để test ghi vào thư mục jobs/ thật — job ma sẽ được restore vào phiên chạy thật.
    monkeypatch.setattr(type(settings), "jobs_dir", property(lambda self: tmp_path))
    monkeypatch.setattr("backend.routes.jobs.probe_video",
                        lambda p: MediaInfo(duration=1.0, video_codec="h264", has_audio=True))
    monkeypatch.setattr("backend.routes.jobs.run_pipeline",
                        lambda *a, **k: PipelineResult(video_path="v.mp4", srt_path="v.srt"))
    _put(Job(id="busy", filename="a.mp4", workdir=tmp_path, voice_id="Kore", status="running"))

    response = client.post(
        "/api/jobs",
        files={"video": ("b.mp4", b"data", "video/mp4")},
        data={"voice_id": VOICES[0].id},
    )
    assert response.status_code == 200
    assert response.json()["job_id"]

    ids = [j["job_id"] for j in client.get("/api/jobs").json()["jobs"]]
    assert "busy" in ids and len(ids) == 2


def test_create_job_normalizes_and_forwards_target_language(client, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from pipeline.models import MediaInfo, PipelineResult

    captured = {}
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(type(settings), "jobs_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(
        "backend.routes.jobs.probe_video",
        lambda path: MediaInfo(duration=1.0, video_codec="h264", has_audio=True),
    )
    monkeypatch.setattr(
        manager,
        "start",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(id="english-job"),
    )
    pipeline_call = {}
    monkeypatch.setattr(
        "backend.routes.jobs.run_pipeline",
        lambda backend, video, workdir, options, progress, media, should_cancel: (
            pipeline_call.update(options=options, media=media)
            or PipelineResult(
                video_path=str(workdir / "output_en.mp4"),
                srt_path=str(workdir / "output_en.srt"),
            )
        ),
    )

    response = client.post(
        "/api/jobs",
        files={"video": ("a.mp4", b"data", "video/mp4")},
        data={"voice_id": "en-US-AvaMultilingualNeural", "target_language": "en"},
    )

    assert response.status_code == 200
    assert captured["job_type"] == "video_dubbing"
    assert captured["target_language"] == "en-US"
    assert captured["input_label"] == "a.mp4"

    result = captured["runner"](None, lambda *args: None, lambda: False)
    assert pipeline_call["options"].target_language == "en-US"
    assert [artifact.filename for artifact in result.artifacts] == ["a_en.mp4", "a_en.srt"]


def test_create_job_rejects_a_voice_that_cannot_speak_the_target_language(
    client, monkeypatch, tmp_path
):
    from pipeline.models import MediaInfo

    starts = []
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(type(settings), "jobs_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(
        "backend.routes.jobs.probe_video",
        lambda path: MediaInfo(duration=1.0, video_codec="h264", has_audio=True),
    )
    monkeypatch.setattr(manager, "start", lambda **kwargs: starts.append(kwargs))

    response = client.post(
        "/api/jobs",
        files={"video": ("a.mp4", b"data", "video/mp4")},
        data={"voice_id": "vi-VN-HoaiMyNeural", "target_language": "en-US"},
    )

    assert response.status_code == 400
    assert starts == []


def test_the_job_list_is_newest_first(client, tmp_path):
    _put(Job(id="old", filename="a.mp4", workdir=tmp_path, voice_id="v",
             status="done", created_at=100.0))
    _put(Job(id="new", filename="b.mp4", workdir=tmp_path, voice_id="v",
             status="running", created_at=200.0))

    ids = [j["job_id"] for j in client.get("/api/jobs").json()["jobs"]]
    assert ids == ["new", "old"]


def test_download_is_refused_until_the_job_finishes(client, tmp_path):
    _put(Job(id="abc", filename="a.mp4", workdir=tmp_path, voice_id="Kore", status="running"))
    assert client.get("/api/jobs/abc/download/video").status_code == 409


def test_generic_artifact_download_serves_declared_audio(client, tmp_path):
    audio = tmp_path / "output.wav"
    audio.write_bytes(b"RIFFfake")
    _put(Job(
        id="speech",
        filename="Hello",
        input_label="Hello",
        workdir=tmp_path,
        voice_id="Ava",
        job_type="text_to_voice",
        target_language="en-US",
        status="done",
        artifacts=[
            JobArtifact("wav", "wav", "hello.wav", "audio/wav", str(audio))
        ],
    ))

    response = client.get("/api/jobs/speech/artifacts/wav")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert "hello.wav" in response.headers["content-disposition"]
    assert client.get("/api/jobs/speech/artifacts/mp3").status_code == 404


def test_download_serves_the_result_with_a_vietnamese_suffix(client, tmp_path):
    video = tmp_path / "output.mp4"
    video.write_bytes(b"fake")
    _put(Job(
        id="abc", filename="phim.mkv", workdir=tmp_path, voice_id="Kore",
        status="done",
        artifacts=[
            JobArtifact("video", "video", "phim_vi.mp4", "video/mp4", str(video)),
            JobArtifact(
                "subtitle", "subtitle", "phim_vi.srt", "application/x-subrip", str(video)
            ),
        ],
    ))
    response = client.get("/api/jobs/abc/download/video")
    assert response.status_code == 200
    assert "phim_vi.mp4" in response.headers["content-disposition"]


def test_download_serves_english_results_with_an_english_suffix(client, tmp_path):
    video = tmp_path / "output_en.mp4"
    subtitles = tmp_path / "output_en.srt"
    video.write_bytes(b"fake")
    subtitles.write_text("", encoding="utf-8")
    _put(Job(
        id="english", filename="movie.mkv", workdir=tmp_path, voice_id="Ava",
        target_language="en-US", status="done",
        artifacts=[
            JobArtifact("video", "video", "movie_en.mp4", "video/mp4", str(video)),
            JobArtifact(
                "subtitle", "subtitle", "movie_en.srt", "application/x-subrip",
                str(subtitles),
            ),
        ],
    ))

    video_response = client.get("/api/jobs/english/download/video")
    srt_response = client.get("/api/jobs/english/download/srt")

    assert "movie_en.mp4" in video_response.headers["content-disposition"]
    assert "movie_en.srt" in srt_response.headers["content-disposition"]


# ─── số liệu chất lượng mà giao diện dựa vào để cảnh báo ───

def _done_job(tmp_path, *, attempted, spoken, warnings=()):
    job = Job(id="q", filename="a.mp4", workdir=tmp_path, voice_id="Kore", status="done",
              attempted_count=attempted, spoken_count=spoken, warnings=list(warnings),
              degraded=attempted > 0 and spoken < attempted)
    manager._jobs[job.id] = job
    return job


def test_snapshot_reports_how_many_lines_were_actually_spoken(client, tmp_path):
    _done_job(tmp_path, attempted=100, spoken=23)
    body = client.get("/api/jobs/q").json()
    assert body["spoken_count"] == 23
    assert body["attempted_count"] == 100
    assert body["silent_ratio"] == 0.77


def test_mostly_silent_result_is_flagged_degraded(client, tmp_path):
    """77% câm là đúng thứ đã xảy ra thật — giao diện phải báo đỏ chứ không ăn mừng."""
    _done_job(tmp_path, attempted=252, spoken=58)
    assert client.get("/api/jobs/q").json()["degraded"] is True


def test_any_partial_result_is_flagged_degraded(client, tmp_path):
    _done_job(tmp_path, attempted=100, spoken=95)
    assert client.get("/api/jobs/q").json()["degraded"] is True


def test_a_clean_run_is_never_degraded(client, tmp_path):
    _done_job(tmp_path, attempted=40, spoken=40)
    body = client.get("/api/jobs/q").json()
    assert body["degraded"] is False
    assert body["silent_ratio"] == 0.0


def test_silent_ratio_is_zero_when_nothing_was_attempted(client, tmp_path):
    _done_job(tmp_path, attempted=0, spoken=0)
    assert client.get("/api/jobs/q").json()["silent_ratio"] == 0.0


def test_a_running_job_is_never_degraded(client, tmp_path):
    job = Job(
        id="q", filename="a.mp4", workdir=tmp_path, voice_id="Kore",
        status="running", attempted_count=100, spoken_count=0,
    )
    manager._jobs[job.id] = job
    assert client.get("/api/jobs/q").json()["degraded"] is False


def test_warnings_reach_the_frontend(client, tmp_path):
    _done_job(tmp_path, attempted=10, spoken=10, warnings=["hết hạn mức"])
    assert client.get("/api/jobs/q").json()["warnings"] == ["hết hạn mức"]
