"""Public and persisted contracts for generic job artifacts."""

from __future__ import annotations

import json

import pytest

from backend.job_contracts import JobArtifact, JobRunResult
from backend.job_manager import Job, JobManager


def _write_meta(jobs_dir, job_id: str, status: str, **extra):
    workdir = jobs_dir / job_id
    workdir.mkdir(parents=True)
    payload = {
        "job_id": job_id,
        "filename": "a.mp4",
        "status": status,
        "created_at": extra.pop("created_at", 100.0),
        **extra,
    }
    (workdir / "job.json").write_text(json.dumps(payload), encoding="utf-8")
    return workdir


def test_snapshot_exposes_typed_artifacts_without_paths(tmp_path):
    artifact = JobArtifact(
        "wav", "wav", "speech.wav", "audio/wav", str(tmp_path / "output.wav")
    )
    job = Job(
        id="text1",
        filename="Welcome…",
        input_label="Welcome…",
        workdir=tmp_path,
        voice_id="voice",
        job_type="text_to_voice",
        target_language="en-US",
        status="done",
        artifacts=[artifact],
    )

    snapshot = job.snapshot()

    assert snapshot["job_type"] == "text_to_voice"
    assert snapshot["target_language"] == "en-US"
    assert snapshot["input_label"] == "Welcome…"
    assert snapshot["artifacts"] == [
        {
            "id": "wav",
            "kind": "wav",
            "filename": "speech.wav",
            "media_type": "audio/wav",
        }
    ]
    assert "path" not in snapshot["artifacts"][0]


def test_partial_success_is_marked_degraded_only_after_publication(tmp_path):
    job = Job(
        id="text2",
        filename="Short text",
        input_label="Short text",
        workdir=tmp_path,
        voice_id="voice",
        job_type="text_to_voice",
        target_language="en-US",
        status="running",
    )

    assert job.snapshot()["degraded"] is False

    job.publish_result(JobRunResult(artifacts=[], attempted_count=3, spoken_count=2))

    assert job.snapshot()["degraded"] is True


def test_old_persisted_job_defaults_to_vietnamese_video(tmp_path):
    _write_meta(
        tmp_path,
        "old",
        "done",
        video_path="output.mp4",
        srt_path="output.srt",
    )
    manager = JobManager()

    manager.restore(tmp_path)

    restored = manager.get("old")
    assert restored is not None
    assert restored.job_type == "video_dubbing"
    assert restored.target_language == "vi-VN"


def test_generic_artifacts_and_degraded_state_round_trip(tmp_path):
    workdir = tmp_path / "text"
    workdir.mkdir()
    job = Job(
        id="text",
        filename="Hello",
        input_label="Hello",
        workdir=workdir,
        voice_id="Ava",
        job_type="text_to_voice",
        target_language="en-US",
        status="done",
        artifacts=[
            JobArtifact("wav", "wav", "hello.wav", "audio/wav", str(workdir / "out.wav"))
        ],
        degraded=True,
    )
    manager = JobManager()
    manager._persist(job)

    restored_manager = JobManager()
    assert restored_manager.restore(tmp_path) == 1
    restored = restored_manager.get("text")

    assert restored is not None
    assert restored.input_label == "Hello"
    assert restored.job_type == "text_to_voice"
    assert restored.target_language == "en-US"
    assert restored.degraded is True
    assert restored.artifacts == [
        JobArtifact("wav", "wav", "hello.wav", "audio/wav", str(workdir / "out.wav"))
    ]


@pytest.mark.parametrize(
    "unsafe_path",
    ["/tmp/private.wav", "../outside.wav", "nested/output.wav"],
)
def test_restore_rejects_unsafe_persisted_artifact_paths(tmp_path, unsafe_path):
    _write_meta(
        tmp_path,
        "unsafe",
        "done",
        artifacts=[
            {
                "id": "wav",
                "kind": "wav",
                "filename": "speech.wav",
                "media_type": "audio/wav",
                "path": unsafe_path,
            }
        ],
    )

    manager = JobManager()
    assert manager.restore(tmp_path) == 1

    assert manager.get("unsafe").artifacts == []


def test_restore_rejects_a_symlink_artifact(tmp_path):
    workdir = _write_meta(
        tmp_path,
        "unsafe",
        "done",
        artifacts=[
            {
                "id": "wav",
                "kind": "wav",
                "filename": "speech.wav",
                "media_type": "audio/wav",
                "path": "output.wav",
            }
        ],
    )
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"private")
    (workdir / "output.wav").symlink_to(outside)

    manager = JobManager()
    manager.restore(tmp_path)

    assert manager.get("unsafe").artifacts == []
