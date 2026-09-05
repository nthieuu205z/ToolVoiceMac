"""Đóng trình duyệt hay khởi động lại server không được làm mất danh sách video.

Mỗi job ghi `job.json` vào thư mục của nó tại các mốc chuyển trạng thái; lúc server
khởi động, `restore()` nạp lại toàn bộ. Job đang chạy dở lúc server chết không thể
tiếp tục — phải bị đánh dấu lỗi, không được hiện "running" với một tiến trình ma.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.job_contracts import JobArtifact, JobRunResult
from backend.job_manager import Job, JobManager
from pipeline.errors import JobCancelledError


def _run_one(manager: JobManager, workdir: Path) -> Job:
    def runner(backend, progress, should_cancel):
        return JobRunResult(
            artifacts=[
                JobArtifact(
                    "video", "video", "a_vi.mp4", "video/mp4",
                    str(workdir / "output.mp4"),
                ),
                JobArtifact(
                    "subtitle", "subtitle", "a_vi.srt", "application/x-subrip",
                    str(workdir / "output.srt"),
                ),
            ],
            attempted_count=3,
            spoken_count=3,
        )

    job = manager.start(
        filename="a.mp4",
        input_label="a.mp4",
        job_type="video_dubbing",
        target_language="vi-VN",
        workdir=workdir,
        voice_id="Kore",
        backend_factory=lambda: None,
        runner=runner,
    )
    manager._futures[job.id].result(timeout=10)
    return job


def _write_meta(jobs_dir: Path, job_id: str, status: str, **extra) -> Path:
    workdir = jobs_dir / job_id
    workdir.mkdir(parents=True)
    data = {"job_id": job_id, "filename": "a.mp4", "status": status,
            "created_at": extra.pop("created_at", 100.0), **extra}
    (workdir / "job.json").write_text(json.dumps(data), encoding="utf-8")
    return workdir


def test_a_finished_job_survives_a_server_restart(tmp_path):
    workdir = tmp_path / "job1"
    workdir.mkdir()
    done = _run_one(JobManager(max_workers=1), workdir)

    reborn = JobManager()          # "server mới"
    assert reborn.restore(tmp_path) == 1

    restored = reborn.get(done.id)
    assert restored is not None
    assert restored.status == "done"
    assert restored.video_path == str(workdir / "output.mp4")
    assert restored.spoken_count == 3

    saved = json.loads((workdir / "job.json").read_text(encoding="utf-8"))
    assert saved["job_type"] == "video_dubbing"
    assert saved["target_language"] == "vi-VN"
    assert saved["input_label"] == "a.mp4"
    assert saved["degraded"] is False
    assert [item["path"] for item in saved["artifacts"]] == [
        "output.mp4", "output.srt"
    ]


def test_a_job_that_died_mid_run_is_marked_as_error(tmp_path):
    _write_meta(tmp_path, "deadjob", "running", stage="synthesize", percent=60.0)

    manager = JobManager()
    manager.restore(tmp_path)

    job = manager.get("deadjob")
    assert job.status == "error"
    assert "Máy chủ đã dừng" in job.message
    # và trạng thái sửa lại phải được ghi xuống đĩa, để lần khởi động sau khỏi sửa lại nữa
    saved = json.loads((tmp_path / "deadjob" / "job.json").read_text(encoding="utf-8"))
    assert saved["status"] == "error"


def test_a_job_stuck_cancelling_is_restored_as_cancelled(tmp_path):
    _write_meta(tmp_path, "cancelme", "cancelling")

    manager = JobManager()
    manager.restore(tmp_path)
    assert manager.get("cancelme").status == "cancelled"
    assert manager.get("cancelme").message == JobCancelledError.user_message


def test_a_corrupt_meta_file_does_not_break_the_others(tmp_path):
    _write_meta(tmp_path, "goodjob", "done")
    bad = tmp_path / "badjob"
    bad.mkdir()
    (bad / "job.json").write_text("{hỏng", encoding="utf-8")

    manager = JobManager()
    assert manager.restore(tmp_path) == 1
    assert manager.get("goodjob") is not None


def test_restored_jobs_keep_their_creation_order(tmp_path):
    _write_meta(tmp_path, "older", "done", created_at=100.0)
    _write_meta(tmp_path, "newer", "done", created_at=200.0)

    manager = JobManager()
    manager.restore(tmp_path)
    assert [j["job_id"] for j in manager.jobs()] == ["newer", "older"]


# ─── dọn thư mục ───

def test_prune_never_touches_an_active_job(tmp_path):
    """Job ĐANG CHẠY là thư mục CŨ NHẤT — nằm ngay giữa vùng bị dọn nếu thiếu chốt chặn."""
    import os

    manager = JobManager()
    # job0 đang chạy và cũ nhất; job1, job2 đã xong và mới hơn.
    for i, status in enumerate(["running", "done", "done"]):
        workdir = _write_meta(tmp_path, f"job{i}", status, created_at=float(i))
        os.utime(workdir, (i + 1, i + 1))
        manager._jobs[f"job{i}"] = Job(id=f"job{i}", filename="a", workdir=workdir,
                                       voice_id="v", status=status, created_at=float(i))

    manager.prune(tmp_path, keep=1)

    assert (tmp_path / "job0").is_dir()          # đang chạy — bất khả xâm phạm dù cũ nhất
    assert not (tmp_path / "job1").is_dir()      # đã xong, ngoài cửa sổ giữ — dọn
    assert manager.get("job1") is None           # mất file thì job cũng rời danh sách
    assert manager.get("job0") is not None


def test_prune_keeps_the_newest_finished_jobs(tmp_path):
    manager = JobManager()
    for i in range(4):
        workdir = _write_meta(tmp_path, f"job{i}", "done", created_at=float(i))
        import os
        os.utime(workdir, (i + 1, i + 1))
        manager._jobs[f"job{i}"] = Job(id=f"job{i}", filename="a", workdir=workdir,
                                       voice_id="v", status="done", created_at=float(i))

    manager.prune(tmp_path, keep=2)

    kept = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
    assert kept == ["job2", "job3"]
