"""Text-to-voice jobs use the shared persistent job lifecycle."""

from __future__ import annotations

import json
import os
import stat
import time
from types import SimpleNamespace

import numpy as np

from backend.config import settings
from backend.routes import text_jobs
from pipeline.audio import write_wav
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.text_to_voice import SpeechResult


def wait_until_terminal(client, job_id: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] in {"done", "error", "cancelled"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def fake_successful_text_runner(
    synthesizer,
    text,
    workdir,
    options,
    *,
    progress,
    should_cancel,
):
    wav_path = workdir / "output.wav"
    mp3_path = workdir / "output.mp3"
    write_wav(
        wav_path,
        np.zeros(TTS_SAMPLE_RATE // 10, dtype="<i2"),
        TTS_SAMPLE_RATE,
    )
    mp3_path.write_bytes(b"ID3-fake-test-audio")
    progress("export", 1.0, "Đã xuất WAV và MP3")
    return SpeechResult(
        wav_path=str(wav_path),
        mp3_path=str(mp3_path),
        attempted_count=1,
        spoken_count=1,
    )


def test_text_job_rejects_empty_text_before_creating_workdir(client, jobs_dir):
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "  ",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 400
    assert list(jobs_dir.iterdir()) == []


def test_text_job_rejects_unsupported_voice_language_pair(client, jobs_dir):
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Hello",
            "voice_id": "vi-VN-HoaiMyNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 400
    assert "không hỗ trợ" in response.json()["detail"]
    assert list(jobs_dir.iterdir()) == []


def test_text_job_rejects_more_than_200000_characters(client, jobs_dir):
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "x" * 200_001,
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 413
    assert list(jobs_dir.iterdir()) == []


def test_text_job_accepts_exactly_50000_unicode_characters(
    client, jobs_dir, monkeypatch
):
    captured = {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=kwargs["job_id"])

    monkeypatch.setattr(text_jobs.manager, "start", fake_start)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "ạ" * 50_000,
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 200
    assert captured["job_type"] == "text_to_voice"
    assert len((captured["workdir"] / "input.txt").read_text(encoding="utf-8")) == 50_000

    runner = captured["runner"]
    closure_values = [cell.cell_contents for cell in (runner.__closure__ or ())]
    assert "ạ" * 50_000 not in closure_values


def test_text_job_rejects_unknown_language_before_creating_workdir(client, jobs_dir):
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Bonjour",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "fr",
        },
    )

    assert response.status_code == 400
    assert list(jobs_dir.iterdir()) == []


def test_text_job_routes_provider_before_creating_workdir(client, jobs_dir, monkeypatch):
    def fail_route(*args):
        raise ValueError("bad route")

    monkeypatch.setattr(text_jobs, "route_provider", fail_route)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Valid text.",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 500
    assert list(jobs_dir.iterdir()) == []


def test_text_job_reserves_a_new_workdir_without_deleting_a_collision(
    client, jobs_dir, monkeypatch
):
    existing = jobs_dir / "collision123"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("existing job", encoding="utf-8")
    ids = iter(("collision123", "reserved456"))
    captured = {}

    monkeypatch.setattr(
        text_jobs.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=next(ids)),
    )

    def fake_start(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=kwargs["job_id"])

    monkeypatch.setattr(text_jobs.manager, "start", fake_start)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Valid text.",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 200
    assert marker.read_text(encoding="utf-8") == "existing job"
    assert captured["job_id"] == "reserved456"
    assert captured["workdir"] == jobs_dir / "reserved456"


def test_text_job_returns_server_error_when_private_input_cannot_be_persisted(
    client, jobs_dir, monkeypatch
):
    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(text_jobs, "_write_private_text", fail_write)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Valid text.",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 500
    assert list(jobs_dir.iterdir()) == []


def test_private_input_is_created_exclusively_with_mode_0600(tmp_path, monkeypatch):
    input_path = tmp_path / "input.txt"
    real_open = os.open
    observed = {}

    def capture_open(path, flags, mode):
        observed.update(flags=flags, mode=mode)
        return real_open(path, flags, mode)

    monkeypatch.setattr(text_jobs.os, "open", capture_open)
    text_jobs._write_private_text(input_path, "private narration")

    assert observed["flags"] & os.O_EXCL
    assert observed["mode"] == 0o600
    assert stat.S_IMODE(input_path.stat().st_mode) == 0o600
    assert input_path.read_text(encoding="utf-8") == "private narration"


def test_cleanup_failure_is_logged(tmp_path, monkeypatch, caplog):
    workdir = tmp_path / "job"
    workdir.mkdir()

    def fail_remove(path):
        raise OSError("permission denied")

    monkeypatch.setattr(text_jobs.shutil, "rmtree", fail_remove)
    text_jobs._remove_workdir(workdir)

    assert "Không thể dọn thư mục công việc" in caplog.text
    assert str(workdir) in caplog.text


def test_text_job_cleans_workdir_when_manager_start_fails(client, jobs_dir, monkeypatch):
    def fail_start(**kwargs):
        raise RuntimeError("executor unavailable")

    monkeypatch.setattr(text_jobs.manager, "start", fail_start)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Valid text.",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 500
    assert list(jobs_dir.iterdir()) == []


def test_text_job_uses_reserved_job_id_for_workdir_and_manager(
    client, jobs_dir, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        text_jobs.uuid, "uuid4", lambda: SimpleNamespace(hex="reserved1234567")
    )

    def fake_start(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="reserved1234")

    monkeypatch.setattr(text_jobs.manager, "start", fake_start)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Valid text.",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"job_id": "reserved1234"}
    assert captured["job_id"] == "reserved1234"
    assert captured["workdir"] == jobs_dir / "reserved1234"


def test_edge_text_job_does_not_require_gemini_key(client, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)

    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Local text job.",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 200
    job = wait_until_terminal(client, response.json()["job_id"])
    assert job["status"] == "done"


def test_text_job_backend_uses_only_the_routed_speech_provider(client, monkeypatch):
    captured = {}
    sentinel = object()
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "tts_provider", "edge")

    class FakeEdgeSynthesizer:
        def __new__(cls, attempts):
            captured["attempts"] = attempts
            return sentinel

    def fake_runner(synthesizer, *args, **kwargs):
        captured["synthesizer"] = synthesizer
        return fake_successful_text_runner(synthesizer, *args, **kwargs)

    monkeypatch.setattr(text_jobs, "EdgeSynthesizer", FakeEdgeSynthesizer)
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_runner)

    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Speech only.",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )
    wait_until_terminal(client, response.json()["job_id"])

    assert captured == {
        "attempts": settings.edge_tts_attempts,
        "synthesizer": sentinel,
    }


def test_text_job_runs_in_shared_queue_and_publishes_wav_and_mp3(
    client, monkeypatch, jobs_dir
):
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Welcome to the show.",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = wait_until_terminal(client, job_id)
    listed = client.get("/api/jobs").json()["jobs"]
    assert any(item["job_id"] == job_id for item in listed)
    assert job["job_type"] == "text_to_voice"
    assert job["target_language"] == "en-US"
    assert {artifact["kind"] for artifact in job["artifacts"]} == {"wav", "mp3"}
    wav = client.get(f"/api/jobs/{job_id}/artifacts/wav")
    mp3 = client.get(f"/api/jobs/{job_id}/artifacts/mp3")
    assert wav.status_code == 200
    assert wav.headers["content-type"] == "audio/wav"
    assert mp3.status_code == 200
    assert mp3.headers["content-type"] == "audio/mpeg"

    registered = text_jobs.manager.get(job_id)
    assert registered is not None
    workdir = registered.workdir
    assert workdir.parent == jobs_dir
    assert (workdir / "input.txt").read_text(encoding="utf-8") == "Welcome to the show."
    assert stat.S_IMODE((workdir / "input.txt").stat().st_mode) == 0o600
    persisted = json.loads((workdir / "job.json").read_text(encoding="utf-8"))
    assert persisted["job_type"] == "text_to_voice"
    assert {item["path"] for item in persisted["artifacts"]} == {
        "output.wav",
        "output.mp3",
    }


def test_snapshot_and_sse_never_expose_full_input(client, monkeypatch):
    secret_text = "PRIVATE-NARRATION-" * 20
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": secret_text,
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert secret_text not in client.get(f"/api/jobs/{job_id}").text
    assert secret_text not in client.get(f"/api/jobs/{job_id}/telemetry").text
    assert secret_text not in client.get(f"/api/jobs/{job_id}/events").text


def test_text_job_normalizes_input_and_builds_an_80_character_excerpt(
    client, monkeypatch, jobs_dir
):
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)
    raw_text = "  First\r\n\tsecond   " + ("word " * 30) + "  "

    response = client.post(
        "/api/jobs/text",
        json={
            "text": raw_text,
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en",
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = wait_until_terminal(client, job_id)
    expected_text = raw_text.strip().replace("\r\n", "\n").replace("\r", "\n")
    registered = text_jobs.manager.get(job_id)
    assert registered is not None
    assert registered.workdir.parent == jobs_dir
    assert (registered.workdir / "input.txt").read_text(encoding="utf-8") == expected_text
    assert job["input_label"] == " ".join(expected_text.split())[:80]
    assert job["filename"] == job["input_label"]
    assert job["target_language"] == "en-US"


def test_text_job_download_names_are_sanitized_separately_from_display_excerpt(
    client, monkeypatch
):
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": 'Private / draft: "hello"? * final.',
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )
    job = wait_until_terminal(client, response.json()["job_id"])

    assert job["input_label"] == 'Private / draft: "hello"? * final.'
    filenames = {artifact["filename"] for artifact in job["artifacts"]}
    assert filenames == {"Private draft hello final.wav", "Private draft hello final.mp3"}
    assert all("/" not in name and '"' not in name and "*" not in name for name in filenames)


def test_completed_text_job_can_be_restored_and_deleted(client, monkeypatch, jobs_dir):
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)
    response = client.post(
        "/api/jobs/text",
        json={
            "text": "Delete all private text artifacts.",
            "voice_id": "en-US-AvaMultilingualNeural",
            "language": "en-US",
        },
    )
    job_id = response.json()["job_id"]
    wait_until_terminal(client, job_id)
    registered = text_jobs.manager.get(job_id)
    assert registered is not None
    workdir = registered.workdir
    assert workdir.parent == jobs_dir

    text_jobs.manager._jobs.clear()
    assert text_jobs.manager.restore(jobs_dir) >= 1
    assert text_jobs.manager.get(job_id).job_type == "text_to_voice"
    assert client.delete(f"/api/jobs/{job_id}").json() == {"deleted": job_id}
    assert not workdir.exists()


def test_interrupted_text_job_restore_uses_a_generic_message(jobs_dir):
    workdir = jobs_dir / "interrupted-text"
    workdir.mkdir()
    (workdir / "input.txt").write_text("private", encoding="utf-8")
    (workdir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "interrupted-text",
                "filename": "Private excerpt",
                "input_label": "Private excerpt",
                "job_type": "text_to_voice",
                "target_language": "en-US",
                "voice_id": "en-US-AvaMultilingualNeural",
                "status": "running",
                "stage": "synthesize",
            }
        ),
        encoding="utf-8",
    )

    text_jobs.manager._jobs.clear()
    assert text_jobs.manager.restore(jobs_dir) == 1
    restored = text_jobs.manager.get("interrupted-text")
    assert restored is not None
    assert restored.status == "error"
    assert "công việc" in restored.message.lower()
    assert "video" not in restored.message.lower()
