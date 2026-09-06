"""Trạng thái job phải nhất quán tại mọi thời điểm giao diện có thể đọc được nó."""

from __future__ import annotations

import json
import threading
from typing import cast

import pytest

from backend import job_manager as jm
from backend.job_contracts import JobArtifact, JobRunResult
from backend.job_manager import Job, JobManager
from pipeline.errors import NoSpeechDetectedError


def _result(workdir, *, attempted=0, spoken=0, warnings=()) -> JobRunResult:
    return JobRunResult(
        artifacts=[
            JobArtifact(
                "video", "video", "a_vi.mp4", "video/mp4", str(workdir / "output.mp4")
            ),
            JobArtifact(
                "subtitle",
                "subtitle",
                "a_vi.srt",
                "application/x-subrip",
                str(workdir / "output.srt"),
            ),
        ],
        attempted_count=attempted,
        spoken_count=spoken,
        warnings=list(warnings),
    )


def _start(manager: JobManager, tmp_path, runner=None, backend_factory=lambda: None) -> Job:
    runner = runner or (lambda backend, progress, should_cancel: _result(tmp_path))
    return manager.start(
        filename="a.mp4",
        input_label="a.mp4",
        job_type="video_dubbing",
        target_language="vi-VN",
        workdir=tmp_path,
        voice_id="Kore",
        backend_factory=backend_factory,
        runner=runner,
    )


def _wait(manager: JobManager, job: Job) -> Job:
    manager._futures[job.id].result(timeout=10)
    return job


class _SpyJob(Job):
    """Chụp lại trạng thái tại đúng khoảnh khắc `status` được gán."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "status" and value in ("done", "error"):
            object.__setattr__(self, "captured", self.snapshot())


class _BrokenFinalizationJob(Job):
    def mark_finished(self, status, message, *, now=None):
        raise NoSpeechDetectedError()


def test_result_fields_are_visible_the_moment_status_turns_done(tmp_path, monkeypatch):
    """SSE dừng khi thấy done, nên artifact và cảnh báo phải xuất hiện cùng lúc."""
    runner = lambda backend, progress, should_cancel: _result(
        tmp_path, attempted=10, spoken=3, warnings=["thiếu giọng đọc"]
    )
    monkeypatch.setattr(jm, "Job", _SpyJob)

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path, runner))

    assert job.captured["status"] == "done"
    assert job.captured["warnings"] == ["thiếu giọng đọc"]
    assert job.captured["spoken_count"] == 3
    assert job.captured["degraded"] is True
    assert [item["id"] for item in job.captured["artifacts"]] == ["video", "subtitle"]
    assert job.video_path == str(tmp_path / "output.mp4")


def test_generic_runner_receives_backend_progress_and_cancel_callback(tmp_path):
    backend = object()
    observed = {}

    def runner(actual_backend, progress, should_cancel):
        observed["backend"] = actual_backend
        observed["cancelled"] = should_cancel()
        progress("synthesize", 0.5, "Đang tạo giọng")
        return JobRunResult([])

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path, runner, lambda: backend))

    assert observed == {"backend": backend, "cancelled": False}
    assert job.status == "done"


def test_text_job_uses_text_stages_and_finishes_at_export(tmp_path):
    manager = JobManager(max_workers=1)
    job = manager.start(
        filename="Welcome back",
        input_label="Welcome back",
        job_type="text_to_voice",
        target_language="en-US",
        workdir=tmp_path,
        voice_id="Ava",
        backend_factory=lambda: None,
        runner=lambda backend, progress, should_cancel: JobRunResult([]),
    )

    _wait(manager, job)

    assert job.stage == "export"
    assert job.percent == 100.0
    assert job.snapshot()["job_type"] == "text_to_voice"


def test_pipeline_error_becomes_a_vietnamese_message(tmp_path):
    def boom(backend, progress, should_cancel):
        raise NoSpeechDetectedError()

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path, boom))

    assert job.status == "error"
    assert job.message == NoSpeechDetectedError.user_message


def test_unexpected_crash_still_surfaces_to_the_ui(tmp_path):
    manager = JobManager(max_workers=1)
    job = _wait(
        manager,
        _start(manager, tmp_path, lambda backend, progress, should_cancel: 1 / 0),
    )

    assert job.status == "error"
    assert "Lỗi không lường trước" in job.message


def test_malformed_result_finishes_as_error_without_partial_publication(tmp_path):
    artifact = JobArtifact(
        "video", "video", "a_vi.mp4", "video/mp4", str(tmp_path / "output.mp4")
    )

    def malformed_runner(backend, progress, should_cancel):
        return JobRunResult(
            artifacts=[artifact],
            warnings=["must not publish"],
            attempted_count=None,
            spoken_count=1,
        )

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path, malformed_runner))

    assert job.status == "error"
    assert job.finished_at is not None
    assert job.artifacts == []
    assert job.warnings == []
    assert job.attempted_count == 0
    assert job.spoken_count == 0
    assert job.degraded is False

    persisted = json.loads((tmp_path / "job.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "error"
    assert persisted["finished_at"] is not None
    assert persisted["artifacts"] == []


def _assert_result_rejected(job: Job, workdir) -> None:
    assert job.status == "error"
    assert job.finished_at is not None
    assert job.artifacts == []
    assert job.warnings == []
    assert job.attempted_count == 0
    assert job.spoken_count == 0
    assert job.degraded is False

    persisted = json.loads((workdir / "job.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "error"
    assert persisted["finished_at"] is not None
    assert persisted["artifacts"] == []
    assert persisted["warnings"] == []
    assert persisted["attempted_count"] == 0
    assert persisted["spoken_count"] == 0
    assert persisted["degraded"] is False


@pytest.mark.parametrize("field", ["id", "kind", "filename", "media_type", "path"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_malformed_artifact_field_finishes_as_error_without_partial_publication(
    tmp_path, field, value
):
    artifact_fields = {
        "id": "video",
        "kind": "video",
        "filename": "a_vi.mp4",
        "media_type": "video/mp4",
        "path": str(tmp_path / "output.mp4"),
    }
    artifact_fields[field] = value

    def malformed_runner(backend, progress, should_cancel):
        return JobRunResult(
            artifacts=[JobArtifact(**artifact_fields)],
            warnings=["must not publish"],
            attempted_count=4,
            spoken_count=1,
        )

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path, malformed_runner))

    _assert_result_rejected(job, tmp_path)


@pytest.mark.parametrize("unsafe_path", ["../outside.mp4", "nested/output.mp4"])
def test_unsafe_artifact_path_finishes_as_error_without_partial_publication(
    tmp_path, unsafe_path
):
    def malformed_runner(backend, progress, should_cancel):
        return JobRunResult(
            artifacts=[
                JobArtifact("video", "video", "a_vi.mp4", "video/mp4", unsafe_path)
            ],
            warnings=["must not publish"],
            attempted_count=4,
            spoken_count=1,
        )

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path, malformed_runner))

    _assert_result_rejected(job, tmp_path)


def test_non_artifact_result_item_finishes_as_error_without_partial_publication(tmp_path):
    def malformed_runner(backend, progress, should_cancel):
        return JobRunResult(
            artifacts=cast(list[JobArtifact], [object()]),
            warnings=["must not publish"],
            attempted_count=4,
            spoken_count=1,
        )

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path, malformed_runner))

    _assert_result_rejected(job, tmp_path)


def test_symlink_artifact_path_finishes_as_error_without_partial_publication(tmp_path):
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"video")
    symlink = tmp_path / "output.mp4"
    symlink.symlink_to(outside)

    def malformed_runner(backend, progress, should_cancel):
        return JobRunResult(
            artifacts=[
                JobArtifact("video", "video", "a_vi.mp4", "video/mp4", str(symlink))
            ],
            warnings=["must not publish"],
            attempted_count=4,
            spoken_count=1,
        )

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path, malformed_runner))

    _assert_result_rejected(job, tmp_path)


def test_finalization_exception_cannot_escape_the_manager_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(jm, "Job", _BrokenFinalizationJob)
    manager = JobManager(max_workers=1)

    job = _wait(manager, _start(manager, tmp_path))

    assert job.status == "error"
    assert job.finished_at is not None
    assert job.artifacts == []
    assert job.message == NoSpeechDetectedError.user_message


def test_message_is_set_before_status_flips_to_error(tmp_path, monkeypatch):
    """Cùng lý do với done: UI đọc message ngay khi thấy status error."""
    monkeypatch.setattr(jm, "Job", _SpyJob)

    def boom(backend, progress, should_cancel):
        raise NoSpeechDetectedError()

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path, boom))

    assert job.captured["status"] == "error"
    assert job.captured["message"] == NoSpeechDetectedError.user_message


def test_two_jobs_run_at_the_same_time(tmp_path):
    """Video thứ hai không phải chờ video thứ nhất."""
    both_running = threading.Barrier(3, timeout=10)

    def runner(backend, progress, should_cancel):
        both_running.wait()
        return _result(tmp_path)

    manager = JobManager(max_workers=2)
    first = _start(manager, tmp_path, runner)
    second = _start(manager, tmp_path, runner)
    both_running.wait()

    _wait(manager, first)
    _wait(manager, second)
    assert first.status == "done" and second.status == "done"


def test_a_third_job_queues_behind_the_concurrency_cap(tmp_path):
    release = threading.Event()

    def runner(backend, progress, should_cancel):
        release.wait(10)
        return _result(tmp_path)

    manager = JobManager(max_workers=2)
    jobs = [_start(manager, tmp_path, runner) for _ in range(3)]
    for _ in range(500):
        if sum(1 for job in jobs if job.status == "running") == 2:
            break
        threading.Event().wait(0.01)

    assert sum(1 for job in jobs if job.status == "running") == 2
    assert sum(1 for job in jobs if job.status == "queued") == 1

    release.set()
    for job in jobs:
        _wait(manager, job)
    assert all(job.status == "done" for job in jobs)
