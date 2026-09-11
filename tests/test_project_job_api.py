"""The public project API, retained names, and protected diagnostic files."""

import json
import os
import threading
from types import SimpleNamespace

import pytest

from backend.job_manager import Job, JobManager
from backend.routes import text_jobs, jobs
from tests.test_text_job_api import fake_successful_text_runner, wait_until_terminal


BASE = {"voice_id": "en-US-AvaMultilingualNeural", "language": "en-US"}


@pytest.fixture(autouse=True)
def offline_backend(monkeypatch):
    from tests.test_speech_project import SequenceSpeech
    monkeypatch.setattr(text_jobs, "_make_backend", lambda *a, **kw: SequenceSpeech())
    monkeypatch.setattr("pipeline.text_to_voice.encode_mp3", lambda wav, mp3: mp3.write_bytes(wav.read_bytes()))


@pytest.mark.parametrize("length, explicit, want_project", [(50_000, False, False), (50_001, False, True), (200_000, False, True), (12, True, True)])
def test_project_mode_boundaries_and_queued_manifest(client, monkeypatch, length, explicit, want_project):
    captured = {}
    def hold_job(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=kwargs["job_id"])
    monkeypatch.setattr(text_jobs.manager, "start", hold_job)
    response = client.post("/api/jobs/text", json={**BASE, "text": "ạ" * length, "project": explicit, "name": "  My book  "})
    assert response.status_code == 200
    assert captured["name"] == "My book"
    assert captured["is_project"] is want_project
    if want_project:
        manifest = json.loads((captured["workdir"] / "project.json").read_text())
        assert manifest["status"] == "queued"
        assert manifest["total_characters"] == length
        assert sum(part["characters"] for part in manifest["parts"]) == length


@pytest.mark.parametrize("name", ["", "   ", "a" * 81])
def test_invalid_names_rejected_before_disk_writes(client, jobs_dir, name):
    response = client.post("/api/jobs/text", json={**BASE, "text": "hello", "name": name})
    assert response.status_code == 422
    assert list(jobs_dir.iterdir()) == []


def test_named_short_job_survives_restart_and_names_downloads(client, monkeypatch, jobs_dir):
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)
    response = client.post("/api/jobs/text", json={**BASE, "text": "Hello.", "name": "  My / book  "})
    job_id = response.json()["job_id"]
    job = wait_until_terminal(client, job_id)
    assert job["name"] == "My / book"
    assert job["is_project"] is False
    assert {a["filename"] for a in job["artifacts"]} == {"My book.wav", "My book.mp3"}
    text_jobs.manager._futures[job_id].result(timeout=2)
    reborn = JobManager()
    reborn.restore(jobs_dir)
    assert reborn.get(job_id).snapshot()["name"] == "My / book"
    assert reborn.get(job_id).is_project is False


def test_legacy_job_has_empty_name_and_no_project(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "job.json").write_text(json.dumps({"job_id": "legacy", "filename": "a.mp4", "status": "done"}))
    reborn = JobManager()
    assert reborn.restore(tmp_path) == 1
    assert reborn.get("legacy").snapshot()["name"] == ""
    assert reborn.get("legacy").snapshot()["is_project"] is False


def test_project_endpoint_and_zip_download(client, monkeypatch):
    from tests.test_speech_project import SequenceSpeech
    monkeypatch.setattr(text_jobs, "_make_backend", lambda *a, **kw: SequenceSpeech())
    monkeypatch.setattr("pipeline.text_to_voice.encode_mp3", lambda wav, mp3: mp3.write_bytes(wav.read_bytes()))
    response = client.post("/api/jobs/text", json={**BASE, "text": "A sentence. Another.", "project": True, "name": "Book"})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = wait_until_terminal(client, job_id)
    assert job["status"] == "done"
    assert job["is_project"] is True
    assert [a["id"] for a in job["artifacts"]] == ["project_zip"]
    manifest = client.get(f"/api/jobs/{job_id}/project").json()
    assert manifest["status"] == "done"
    assert manifest["parts"][0]["spoken_count"] == 2
    download = client.get(f"/api/jobs/{job_id}/artifacts/project_zip")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"


def test_project_read_rejects_other_job_types_and_symlink(client, jobs_dir):
    root = jobs_dir / "job"
    root.mkdir()
    job = Job("job", "input", root, "voice")
    jobs.manager._jobs[job.id] = job
    assert client.get("/api/jobs/job/project").status_code == 404
    job.is_project = True
    assert client.get("/api/jobs/job/project").status_code == 404
    job.job_type = "text_to_voice"
    outside = jobs_dir / "outside.json"
    outside.write_text('{"private":true}')
    (root / "project.json").symlink_to(outside)
    assert client.get("/api/jobs/job/project").status_code == 404


def test_prune_never_deletes_completed_projects(tmp_path):
    manager = JobManager()
    for index in range(3):
        root = tmp_path / str(index)
        root.mkdir()
        os.utime(root, (index + 1, index + 1))
        manager._jobs[str(index)] = Job(str(index), "input", root, "voice", status="done", is_project=index == 0)
    manager.prune(tmp_path, keep=1)
    assert (tmp_path / "0").exists()
    assert manager.get("0") is not None
    assert not (tmp_path / "1").exists()


def test_video_name_validation_precedes_upload(client, jobs_dir):
    response = client.post("/api/jobs", data={"voice_id": "vi-VN-HoaiMyNeural", "name": "x" * 81}, files={"video": ("a.mp4", b"video", "video/mp4")})
    assert response.status_code == 422
    assert list(jobs_dir.iterdir()) == []


def test_backend_initialization_failure_marks_project_manifest_error(client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("backend unavailable")
    monkeypatch.setattr(text_jobs, "_make_backend", unavailable)
    response = client.post("/api/jobs/text", json={**BASE, "text": "Hello.", "project": True})
    job_id = response.json()["job_id"]
    assert wait_until_terminal(client, job_id)["status"] == "error"
    text_jobs.manager._futures[job_id].result(timeout=2)
    assert client.get(f"/api/jobs/{job_id}/project").json()["status"] == "error"
    assert client.get(f"/api/jobs/{job_id}/artifacts/project_zip").status_code == 409


def test_interrupted_project_manifest_is_terminal_after_restore(tmp_path):
    from pipeline.speech_project import initialize_project
    from pipeline.text_to_voice import TextToVoiceOptions
    root = tmp_path / "interrupted"
    initialize_project("Hello.", root, TextToVoiceOptions("voice", "en-US"), name="Book")
    (root / "project.zip").write_bytes(b"unpublished stale archive")
    (root / "job.json").write_text(json.dumps({"job_id": "interrupted", "filename": "Hello", "job_type": "text_to_voice", "is_project": True, "status": "running"}))
    reborn = JobManager()
    reborn.restore(tmp_path)
    assert reborn.get("interrupted").status == "error"
    assert json.loads((root / "project.json").read_text())["status"] == "error"
    assert not (root / "project.zip").exists()


def test_queued_project_cancel_updates_manifest_without_starting_speech(tmp_path):
    from pipeline.speech_project import initialize_project
    from pipeline.text_to_voice import TextToVoiceOptions
    from backend.job_contracts import JobRunResult
    manager = JobManager(max_workers=1)
    release = threading.Event()
    blocker_dir = tmp_path / "blocker"
    blocker_dir.mkdir()
    blocker = manager.start(filename="blocker", workdir=blocker_dir, voice_id="voice", backend_factory=lambda: None,
                            runner=lambda *args: (release.wait(2), JobRunResult([]))[1])
    root = tmp_path / "queued"
    initialize_project("Hello.", root, TextToVoiceOptions("voice", "en-US"), name="Book")
    try:
        job = manager.start(filename="project", workdir=root, voice_id="voice", is_project=True,
                            job_type="text_to_voice", backend_factory=lambda: None,
                            runner=lambda *args: pytest.fail("Cancelled queued project ran"))
        assert manager.cancel(job.id).status == "cancelled"
        assert json.loads((root / "project.json").read_text())["status"] == "cancelled"
    finally:
        release.set()
        manager._futures[blocker.id].result(timeout=2)
        manager._executor.shutdown(wait=True)


def test_project_duration_override_rejects_multiple_parts_before_saving(client, monkeypatch, jobs_dir):
    from pipeline.omnivoice_settings import OmniVoiceSettings
    monkeypatch.setattr(text_jobs, "is_available", lambda *args, **kwargs: True)
    monkeypatch.setattr(text_jobs, "route_provider", lambda *args: "omnivoice")
    monkeypatch.setattr(text_jobs, "omnivoice_settings", lambda: {"supported": True, "defaults": OmniVoiceSettings().model_dump()})
    response = client.post("/api/jobs/text", json={**BASE, "text": "x" * 5001, "project": True, "omnivoice": {"duration": 8}})
    assert response.status_code == 400
    assert list(jobs_dir.iterdir()) == []


def test_named_video_uses_name_for_snapshot_and_downloads(client, monkeypatch):
    from backend.config import settings
    from pipeline.models import MediaInfo, PipelineResult
    monkeypatch.setattr(settings, "gemini_api_key", "test-only")
    monkeypatch.setattr(jobs, "is_available", lambda *args, **kwargs: True)
    monkeypatch.setattr(jobs, "_make_backend", lambda *args: None)
    monkeypatch.setattr(jobs, "probe_video", lambda path: MediaInfo(duration=1.0, width=640, height=480, video_codec="h264", has_audio=True))
    def fake_video(backend, source, root, *args):
        video = root / "output.mp4"
        subtitle = root / "output.srt"
        video.write_bytes(b"video")
        subtitle.write_text("subtitle")
        return PipelineResult(str(video), str(subtitle), attempted_count=1, spoken_count=1)
    monkeypatch.setattr(jobs, "run_pipeline", fake_video)
    response = client.post("/api/jobs", data={"voice_id": "vi-VN-HoaiMyNeural", "name": "  Named / film  "}, files={"video": ("original.mp4", b"video", "video/mp4")})
    assert response.status_code == 200
    job = wait_until_terminal(client, response.json()["job_id"])
    assert job["status"] == "done"
    assert job["name"] == "Named / film"
    assert job["filename"] == "original.mp4"
    assert {a["filename"] for a in job["artifacts"]} == {"Named film_vi.mp4", "Named film_vi.srt"}


def test_malformed_project_manifest_returns_404(client, jobs_dir):
    root = jobs_dir / "malformed"
    root.mkdir()
    (root / "project.json").write_text(json.dumps({"version": 1, "parts": [{"files": {}}]}))
    jobs.manager._jobs["malformed"] = Job("malformed", "input", root, "voice", job_type="text_to_voice", is_project=True)
    assert client.get("/api/jobs/malformed/project").status_code == 404


def test_corrupt_interrupted_project_does_not_block_restoring_other_jobs(tmp_path):
    for job_id, extra in [("malformed", {"job_type": "text_to_voice", "is_project": True, "status": "running"}), ("valid", {"status": "done"})]:
        root = tmp_path / job_id
        root.mkdir()
        (root / "job.json").write_text(json.dumps({"job_id": job_id, "filename": "input", **extra}))
    (tmp_path / "malformed" / "project.json").write_text(json.dumps({"version": 1, "parts": [{"files": {}}]}))
    (tmp_path / "malformed" / "project.zip").write_bytes(b"stale archive")
    reborn = JobManager()
    assert reborn.restore(tmp_path) == 2
    assert reborn.get("malformed").status == "error"
    assert reborn.get("valid").status == "done"
    assert json.loads((tmp_path / "malformed" / "job.json").read_text())["status"] == "error"
    assert not (tmp_path / "malformed" / "project.zip").exists()
