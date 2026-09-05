from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from backend.job_contracts import JobArtifact
from backend.job_manager import Job
from backend.routes import jobs as job_routes
from pipeline import custom_voices
from pipeline.audio import write_wav


def _voice(voice_id: str = "clone-safe"):
    write_wav(custom_voices.sample_path(voice_id), np.zeros(2400, dtype="<i2"), 24000)
    return custom_voices.register(voice_id, "Safe")


def test_custom_voice_ids_are_restricted_to_clone_slugs():
    assert custom_voices.is_custom("clone-safe") is True
    assert custom_voices.is_custom("clone-../outside") is False
    assert custom_voices.is_custom("clone-a/b") is False
    assert custom_voices.is_custom("clone-a\\b") is False


def test_sample_path_rejects_a_non_clone_id():
    with pytest.raises(ValueError):
        custom_voices.sample_path("clone-../outside")


def test_register_rejects_invalid_id_and_missing_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_voices, "_dir", tmp_path)
    with pytest.raises(ValueError):
        custom_voices.register("clone-../outside", "Bad")
    with pytest.raises(FileNotFoundError):
        custom_voices.register("clone-valid", "Missing")


def test_malformed_voice_metadata_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_voices, "_dir", tmp_path)
    (tmp_path / "clone-safe.json").write_text(
        json.dumps({"id": "clone-../outside", "display_name": "bad"}),
        encoding="utf-8",
    )
    (tmp_path / "outside.wav").write_bytes(b"not a voice")

    assert custom_voices.list_custom() == []


def test_registered_voice_id_must_match_metadata_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_voices, "_dir", tmp_path)
    (tmp_path / "clone-safe.wav").write_bytes(b"sample")
    (tmp_path / "clone-safe.json").write_text(
        json.dumps({"id": "clone-other", "display_name": "bad"}),
        encoding="utf-8",
    )

    assert custom_voices.list_custom() == []


def _done_job(job_dir: Path, video_path: Path) -> Job:
    return Job(
        id="job",
        filename="video.mp4",
        input_label="video.mp4",
        workdir=job_dir,
        voice_id="Kore",
        status="done",
        artifacts=[
            JobArtifact(
                "video", "video", "video_vi.mp4", "video/mp4", str(video_path)
            )
        ],
    )


def test_download_rejects_a_path_outside_the_job_directory(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _done_job(job_dir, outside))

    with pytest.raises(HTTPException) as excinfo:
        job_routes.download_video("job")

    assert excinfo.value.status_code == 404


def test_download_rejects_a_symlink_inside_job_directory(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    link = job_dir / "output.mp4"
    link.symlink_to(outside)
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _done_job(job_dir, link))

    with pytest.raises(HTTPException) as excinfo:
        job_routes.download_video("job")

    assert excinfo.value.status_code == 404


def test_download_accepts_a_regular_result_file(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"safe")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _done_job(job_dir, video))

    response = job_routes.download_video("job")

    assert response.path == video.resolve()
    assert response.filename == "video_vi.mp4"
    assert response.media_type == "video/mp4"


def test_download_rejects_a_nested_result_file(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    nested = job_dir / "nested"
    nested.mkdir(parents=True)
    video = nested / "output.mp4"
    video.write_bytes(b"safe")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _done_job(job_dir, video))

    with pytest.raises(HTTPException) as excinfo:
        job_routes.download_video("job")

    assert excinfo.value.status_code == 404


def test_generic_artifact_route_uses_exact_id_and_trusted_metadata(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    audio = job_dir / "output.wav"
    audio.write_bytes(b"RIFFsafe")
    job = Job(
        id="job",
        filename="private input",
        input_label="private input",
        workdir=job_dir,
        voice_id="Ava",
        job_type="text_to_voice",
        target_language="en-US",
        status="done",
        artifacts=[
            JobArtifact("wav", "wav", "speech.wav", "audio/wav", str(audio))
        ],
    )
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: job)

    response = job_routes.download_artifact("job", "wav")

    assert response.path == audio.resolve()
    assert response.filename == "speech.wav"
    assert response.media_type == "audio/wav"

    with pytest.raises(HTTPException) as excinfo:
        job_routes.download_artifact("job", "WAV")
    assert excinfo.value.status_code == 404
