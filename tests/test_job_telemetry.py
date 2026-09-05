"""Telemetry contract for the operations monitor."""

from __future__ import annotations

import json
import time
from pathlib import Path

from backend.job_manager import Job, JobManager
from pipeline.models import MediaInfo, PipelineResult
from pipeline.runner import PipelineOptions


MEDIA = MediaInfo(duration=10.0, video_codec="h264", has_audio=True)
OPTIONS = PipelineOptions(voice_id="clone-demo")


def test_job_snapshot_exposes_elapsed_eta_stage_history_and_engine():
    now = time.time()
    job = Job(
        id="abc",
        filename="clip.mp4",
        workdir=Path("jobs/abc"),
        voice_id="clone-demo",
        status="running",
        stage="synthesize",
        percent=62.5,
        created_at=now - 40,
        started_at=now - 30,
        stage_started_at=now - 8,
        stage_history=[
            {"stage": "extract", "started_at": now - 30, "ended_at": now - 25},
            {"stage": "transcribe", "started_at": now - 25, "ended_at": now - 16},
            {"stage": "translate", "started_at": now - 16, "ended_at": now - 8},
            {"stage": "synthesize", "started_at": now - 8, "ended_at": None},
        ],
        engine="omnivoice",
        device="mps:0",
        batch_size=8,
    )

    snapshot = job.snapshot(now=now)

    assert snapshot["elapsed_seconds"] == 30.0
    assert snapshot["stage_elapsed_seconds"] == 8.0
    assert snapshot["eta_seconds"] > 0
    assert snapshot["engine"] == "omnivoice"
    assert snapshot["device"] == "mps:0"
    assert snapshot["batch_size"] == 8
    assert [event["stage"] for event in snapshot["stage_history"]] == [
        "extract", "transcribe", "translate", "synthesize"
    ]


def test_job_manager_records_stage_transitions_and_completion_telemetry(tmp_path, monkeypatch):
    def pipeline(backend, video, workdir, options, progress, media, should_cancel):
        progress("extract", 1.0, "Đã tách âm thanh")
        progress("synthesize", 0.25, "Đang đọc lô 1/4")
        return PipelineResult(
            video_path=str(workdir / "output.mp4"),
            srt_path=str(workdir / "output.srt"),
            attempted_count=4,
            spoken_count=4,
        )

    monkeypatch.setattr("backend.job_manager.run_pipeline", pipeline)
    manager = JobManager(max_workers=1)
    job = manager.start(
        filename="clip.mp4",
        workdir=tmp_path,
        voice_id="clone-demo",
        video_path=tmp_path / "input.mp4",
        media=MEDIA,
        backend_factory=lambda: type(
            "Backend", (), {"engine": "omnivoice", "device": "mps:0", "batch_size": 8}
        )(),
        options=OPTIONS,
    )

    manager._futures[job.id].result(timeout=10)
    snapshot = job.snapshot()

    assert snapshot["status"] == "done"
    assert snapshot["started_at"] is not None
    assert snapshot["finished_at"] is not None
    assert snapshot["elapsed_seconds"] >= 0
    assert snapshot["engine"] == "omnivoice"
    assert snapshot["device"] == "mps:0"
    assert snapshot["batch_size"] == 8
    assert [event["stage"] for event in snapshot["stage_history"]] == [
        "extract", "synthesize", "mux"
    ]


def test_job_timer_starts_before_backend_initialization(tmp_path, monkeypatch):
    observed = {}

    class Backend:
        engine = "omnivoice"
        device = "mps:0"
        batch_size = 8

    def factory():
        observed["started_at_during_factory"] = job.started_at
        return Backend()

    monkeypatch.setattr("backend.job_manager.run_pipeline", lambda *args: PipelineResult(
        video_path="v.mp4", srt_path="v.srt"
    ))
    manager = JobManager(max_workers=1)
    job = manager.start(
        filename="clip.mp4",
        workdir=tmp_path,
        voice_id="clone-demo",
        video_path=tmp_path / "input.mp4",
        media=MEDIA,
        backend_factory=factory,
        options=OPTIONS,
    )

    manager._futures[job.id].result(timeout=10)

    assert observed["started_at_during_factory"] is not None


def test_job_manager_captures_runtime_info_from_composite_backend(tmp_path, monkeypatch):
    class Backend:
        def runtime_info(self):
            return "omnivoice", "mps:0", 8

    monkeypatch.setattr("backend.job_manager.run_pipeline", lambda *args: PipelineResult(
        video_path="v.mp4", srt_path="v.srt"
    ))
    manager = JobManager(max_workers=1)
    job = manager.start(
        filename="clip.mp4",
        workdir=tmp_path,
        voice_id="clone-demo",
        video_path=tmp_path / "input.mp4",
        media=MEDIA,
        backend_factory=Backend,
        options=OPTIONS,
    )

    manager._futures[job.id].result(timeout=10)

    assert job.snapshot()["engine"] == "omnivoice"
    assert job.snapshot()["device"] == "mps:0"
    assert job.snapshot()["batch_size"] == 8


def test_progress_updates_do_not_move_the_overall_percent_backwards():
    job = Job(
        id="abc",
        filename="clip.mp4",
        workdir=Path("jobs/abc"),
        voice_id="clone-demo",
    )

    job.update_progress("transcribe", 1.0, "Xong nhận diện", now=10.0)
    job.update_progress("extract", 0.0, "Tín hiệu trễ", now=11.0)

    assert job.snapshot()["percent"] == 30.0


def test_persisted_telemetry_round_trips(tmp_path):
    job_dir = tmp_path / "abc"
    job_dir.mkdir()
    now = time.time()
    payload = {
        "job_id": "abc",
        "filename": "clip.mp4",
        "status": "done",
        "stage": "mux",
        "created_at": now - 20,
        "started_at": now - 18,
        "finished_at": now - 2,
        "stage_started_at": now - 3,
        "stage_history": [
            {"stage": "extract", "started_at": now - 18, "ended_at": now - 15}
        ],
        "engine": "omnivoice",
        "device": "mps:0",
        "batch_size": 8,
    }
    (job_dir / "job.json").write_text(json.dumps(payload), encoding="utf-8")

    manager = JobManager()
    assert manager.restore(tmp_path) == 1
    restored = manager.get("abc")

    assert restored is not None
    assert restored.engine == "omnivoice"
    assert restored.device == "mps:0"
    assert restored.batch_size == 8
    assert restored.stage_history[0]["stage"] == "extract"
