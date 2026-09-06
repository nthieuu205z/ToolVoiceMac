"""Quản lý công việc: nhiều video cùng lúc, chạy trên một pool luồng có trần.

Pipeline toàn là lời gọi chặn (subprocess + HTTP) nên dùng thread thay vì asyncio.
Trần số job chạy đồng thời (MAX_CONCURRENT_JOBS) thấp có chủ ý: Whisper và OmniVoice
đều bị khóa suy luận toàn cục, edge-tts bị trần 2 request đồng thời — job thứ ba
chủ yếu chỉ chen hàng chứ không nhanh thêm. Vượt trần thì xếp hàng ("queued").

Mỗi job ghi trạng thái xuống `workdir/job.json` tại các mốc chuyển trạng thái, nên
đóng trình duyệt hay khởi động lại server vẫn thấy lại danh sách; job đang chạy dở
lúc server chết được đánh dấu lỗi khi khôi phục (thread của nó không sống lại được).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from backend.job_contracts import JobArtifact, JobRunResult, JobRunner
from pipeline.errors import JobCancelledError, PipelineError
from pipeline.models import STAGES_BY_JOB_TYPE, MediaInfo, overall_percent
from pipeline.runner import PipelineOptions, run_pipeline

log = logging.getLogger(__name__)

# Trên ngưỡng này thì video coi như hỏng nặng: vẫn tải về được nhưng phải cảnh báo đỏ.
SILENT_RATIO_ALERT = 0.2

# Trạng thái còn "sống": chiếm slot chạy và không được dọn thư mục.
ACTIVE_STATUSES = ("queued", "running", "cancelling")

_INTERRUPTED_MESSAGE = "Máy chủ đã dừng khi công việc này đang xử lý. Hãy chạy lại công việc."


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _safe_int(value) -> int:
    try:
        return max(0, int(str(value or 0)))
    except (TypeError, ValueError):
        return 0


def _direct_child_path(
    workdir: Path, value: object, *, require_relative: bool
) -> Path | None:
    """Resolve one artifact path without allowing traversal, nesting, or symlinks."""
    if not isinstance(value, str) or not value:
        return None
    try:
        raw = Path(value)
        if not value or ".." in raw.parts or (require_relative and raw.is_absolute()):
            return None
        if require_relative and len(raw.parts) != 1:
            return None
        root = workdir.resolve(strict=True)
        candidate = raw if raw.is_absolute() else workdir / raw
        if candidate.is_symlink():
            return None
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.parent == root else None


def _validated_artifacts(workdir: Path, artifacts: list[JobArtifact]) -> list[JobArtifact]:
    """Materialize a result's artifacts only after every persisted field is safe."""
    validated: list[JobArtifact] = []
    for artifact in artifacts:
        if not isinstance(artifact, JobArtifact):
            raise TypeError("Job result artifacts must be JobArtifact values")
        values = {
            name: getattr(artifact, name)
            for name in ("id", "kind", "filename", "media_type", "path")
        }
        if not all(isinstance(value, str) and value.strip() for value in values.values()):
            raise TypeError("Job artifact fields must be non-empty strings")
        path = _direct_child_path(workdir, values["path"], require_relative=False)
        if path is None:
            raise ValueError("Job artifact path must be a safe direct child")
        validated.append(JobArtifact(**{**values, "path": str(path)}))
    return validated


def _restore_artifacts(data: dict, workdir: Path) -> list[JobArtifact]:
    """Restore only new-format relative direct-child artifact records."""
    restored: list[JobArtifact] = []
    raw_artifacts = data.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        return restored
    for item in raw_artifacts:
        if not isinstance(item, dict):
            continue
        values = [item.get(key) for key in ("id", "kind", "filename", "media_type", "path")]
        if not all(isinstance(value, str) and value for value in values):
            continue
        path = _direct_child_path(workdir, values[4], require_relative=True)
        if path is None:
            continue
        restored.append(JobArtifact(values[0], values[1], values[2], values[3], str(path)))
    return restored


def _legacy_target_language(data: dict) -> str:
    explicit = data.get("target_language")
    if isinstance(explicit, str) and explicit:
        return explicit
    for key in ("video_path", "srt_path"):
        value = data.get(key)
        if isinstance(value, str) and Path(value).stem.endswith("_en"):
            return "en-US"
    return "vi-VN"


def _legacy_artifacts(
    data: dict, workdir: Path, target_language: str
) -> list[JobArtifact]:
    artifacts: list[JobArtifact] = []
    original_name = str(data.get("filename") or data.get("input_label") or "video")
    stem = Path(original_name).stem or "video"
    suffix = "en" if target_language == "en-US" else "vi"
    for artifact_id, kind, media_type, key in (
        ("video", "video", "video/mp4", "video_path"),
        ("subtitle", "subtitle", "application/x-subrip", "srt_path"),
    ):
        value = str(data.get(key, "") or "")
        if not value:
            continue
        path = _direct_child_path(workdir, value, require_relative=False)
        if path is None:
            continue
        filename = (
            f"{stem}_{suffix}{path.suffix or '.mp4'}"
            if artifact_id == "video"
            else f"{stem}_{suffix}.srt"
        )
        artifacts.append(
            JobArtifact(artifact_id, kind, filename, media_type, str(path))
        )
    return artifacts


@dataclass
class Job:
    id: str
    filename: str
    workdir: Path
    voice_id: str
    job_type: str = "video_dubbing"
    target_language: str = "vi-VN"
    input_label: str = ""
    status: str = "queued"  # queued | running | cancelling | cancelled | done | error
    stage: str = "extract"
    percent: float = 0.0
    message: str = ""
    artifacts: list[JobArtifact] = field(default_factory=list)
    degraded: bool = False
    attempted_count: int = 0
    spoken_count: int = 0
    warnings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    stage_started_at: float | None = None
    stage_fraction: float = 0.0
    stage_history: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    engine: str = ""
    device: str = ""
    batch_size: int = 0
    # Không giết thread giữa chừng (ffmpeg đang ghi file); pipeline tự đọc cờ này.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    telemetry_lock: threading.RLock = field(default_factory=threading.RLock,
                                             repr=False, compare=False)

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def silent_ratio(self) -> float:
        if self.attempted_count <= 0:
            return 0.0
        return (self.attempted_count - self.spoken_count) / self.attempted_count

    def _artifact_path(self, artifact_id: str) -> str:
        artifact = next((item for item in self.artifacts if item.id == artifact_id), None)
        return artifact.path if artifact is not None else ""

    @property
    def video_path(self) -> str:
        return self._artifact_path("video")

    @property
    def srt_path(self) -> str:
        return self._artifact_path("subtitle")

    def _append_event_locked(self, event: dict) -> None:
        self.events.append(event)
        del self.events[:-160]

    def update_runtime_from_backend(self, backend) -> None:
        """Capture non-sensitive backend facts for the monitor."""
        runtime_info = getattr(backend, "runtime_info", None)
        if callable(runtime_info):
            try:
                info = runtime_info()
                if isinstance(info, tuple) and len(info) == 3:
                    self.update_runtime(engine=str(info[0]), device=str(info[1]),
                                        batch_size=_safe_int(info[2]))
                    return
            except Exception:
                pass
        engine = getattr(backend, "engine", "")
        device = getattr(backend, "device", "")
        batch = getattr(backend, "batch_size", 0)
        self.update_runtime(
            engine=str(engine() if callable(engine) else engine),
            device=str(device() if callable(device) else device),
            batch_size=_safe_int(batch() if callable(batch) else batch),
        )

    def _set_status_locked(self, status: str, message: str = "") -> None:
        self.status = status
        if message:
            self.message = message
        self._append_event_locked({
            "at": time.time(),
            "kind": "status",
            "stage": self.stage,
            "status": status,
            "message": self.message,
        })

    def mark_started(self, *, now: float | None = None) -> None:
        """Bắt đầu đồng hồ và mở stage đầu tiên."""
        with self.telemetry_lock:
            if self.started_at is not None:
                return
            at = time.time() if now is None else now
            self.started_at = at
            self.stage_started_at = at
            self.stage_history.append({
                "stage": self.stage,
                "started_at": at,
                "ended_at": None,
                "duration_seconds": None,
            })
            self._append_event_locked({
                "at": at,
                "kind": "started",
                "stage": self.stage,
                "message": self.message or "Bắt đầu xử lý",
            })

    def update_runtime(self, *, engine: str | None = None,
                       device: str | None = None,
                       batch_size: int | None = None) -> None:
        """Ghi thông tin engine mà không làm lộ cấu hình nhạy cảm."""
        with self.telemetry_lock:
            if engine:
                self.engine = engine
            if device:
                self.device = device
            if batch_size is not None:
                self.batch_size = _safe_int(batch_size)

    def update_progress(self, stage: str, fraction: float, message: str,
                        *, now: float | None = None, engine: str | None = None,
                        device: str | None = None,
                        batch_size: int | None = None) -> None:
        """Cập nhật stage + telemetry nguyên tử để SSE không thấy trạng thái nửa chừng."""
        at = time.time() if now is None else now
        bounded = min(1.0, max(0.0, fraction))
        with self.telemetry_lock:
            if self.started_at is None:
                self.started_at = at
                self.stage_started_at = at
                self.stage_history.append({
                    "stage": stage,
                    "started_at": at,
                    "ended_at": None,
                    "duration_seconds": None,
                })
                self._append_event_locked({
                    "at": at,
                    "kind": "started",
                    "stage": stage,
                    "message": message or "Bắt đầu xử lý",
                })
            self.update_runtime(engine=engine, device=device, batch_size=batch_size)

            previous = self.stage_history[-1] if self.stage_history else None
            if previous is None:
                self.stage_started_at = at
                self.stage_history.append({
                    "stage": stage,
                    "started_at": at,
                    "ended_at": None,
                    "duration_seconds": None,
                })
            elif previous["stage"] != stage:
                previous["ended_at"] = at
                previous["duration_seconds"] = round(
                    max(0.0, at - float(previous["started_at"])), 3
                )
                self.stage_started_at = at
                self.stage_history.append({
                    "stage": stage,
                    "started_at": at,
                    "ended_at": None,
                    "duration_seconds": None,
                })

            previous_percent = self.percent
            self.stage = stage
            self.stage_fraction = bounded
            self.percent = max(
                previous_percent,
                overall_percent(stage, bounded, self.job_type, fallback=self.percent),
            )
            self.message = message
            self._append_event_locked({
                "at": at,
                "kind": "progress",
                "stage": stage,
                "fraction": round(bounded, 4),
                "percent": self.percent,
                "message": message,
            })

    def mark_finished(self, status: str, message: str, *, now: float | None = None) -> None:
        """Đóng stage/đồng hồ trước khi caller đổi status thành terminal."""
        at = time.time() if now is None else now
        with self.telemetry_lock:
            if self.started_at is not None and self.stage_history:
                current = self.stage_history[-1]
                if current.get("ended_at") is None:
                    current["ended_at"] = at
                    current["duration_seconds"] = round(
                        max(0.0, at - float(current["started_at"])), 3
                    )
            self.finished_at = at
            self._append_event_locked({
                "at": at,
                "kind": "finished",
                "stage": self.stage,
                "status": status,
                "message": message,
            })

    def snapshot(self, *, now: float | None = None) -> dict:
        at = time.time() if now is None else now
        with self.telemetry_lock:
            elapsed = (
                max(0.0, (self.finished_at or at) - self.started_at)
                if self.started_at is not None else 0.0
            )
            stage_elapsed = (
                max(0.0, (self.finished_at or at) - self.stage_started_at)
                if self.stage_started_at is not None else 0.0
            )
            eta = None
            if self.status == "running" and self.percent > 0:
                eta = round(elapsed * (100.0 - self.percent) / self.percent, 1)
            elif self.status == "done":
                eta = 0.0
            current_engine = self.engine
            current_device = self.device
            current_batch_size = self.batch_size
            return {
                "job_id": self.id,
                "status": self.status,
                "stage": self.stage,
                "percent": self.percent,
                "message": self.message,
                "filename": self.filename,
                "input_label": self.input_label,
                "job_type": self.job_type,
                "target_language": self.target_language,
                "voice_id": self.voice_id,
                "created_at": round(self.created_at, 3),
                "artifacts": [artifact.public() for artifact in self.artifacts],
                "warnings": self.warnings,
                "attempted_count": self.attempted_count,
                "spoken_count": self.spoken_count,
                "silent_ratio": round(self.silent_ratio, 3),
                "degraded": self.degraded,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_seconds": round(elapsed, 1),
                "stage_elapsed_seconds": round(stage_elapsed, 1),
                "stage_fraction": round(self.stage_fraction, 4),
                "eta_seconds": eta,
                "engine": current_engine,
                "device": current_device,
                "batch_size": current_batch_size,
                "stage_history": [dict(item) for item in self.stage_history],
                "events": [dict(item) for item in self.events],
            }

    def publish_result(self, result: JobRunResult) -> None:
        """Set terminal result fields before the terminal status is exposed."""
        artifacts = _validated_artifacts(self.workdir, list(result.artifacts))
        warnings = list(result.warnings)
        if not all(isinstance(warning, str) for warning in warnings):
            raise TypeError("Job result warnings must be strings")
        attempted_count = max(0, int(result.attempted_count))
        spoken_count = max(0, int(result.spoken_count))
        degraded = attempted_count > 0 and spoken_count < attempted_count
        with self.telemetry_lock:
            self.artifacts = artifacts
            self.warnings = warnings
            self.attempted_count = attempted_count
            self.spoken_count = spoken_count
            self.degraded = degraded

    def clear_result(self) -> None:
        """Discard unpublished/partial terminal data after finalization fails."""
        with self.telemetry_lock:
            self.artifacts = []
            self.warnings = []
            self.attempted_count = 0
            self.spoken_count = 0
            self.degraded = False


class JobManager:
    """Sổ đăng ký job + pool luồng. Route chỉ đọc snapshot, không đụng thread."""

    def __init__(self, max_workers: int | None = None):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers = max_workers

    # ─── tra cứu ────────────────────────────────────────────────────

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[dict]:
        """Snapshot mọi job, mới nhất trước — giao diện vẽ thẳng danh sách này."""
        ordered = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.snapshot() for job in ordered]

    @property
    def current(self) -> Job | None:
        """Job đang sống mới nhất — giữ cho /api/jobs/current cũ tiếp tục chạy."""
        active = [j for j in self._jobs.values() if j.status in ACTIVE_STATUSES]
        return max(active, key=lambda j: j.created_at) if active else None

    def is_busy(self) -> bool:
        with self._lock:
            return any(j.status in ACTIVE_STATUSES for j in self._jobs.values())

    # ─── vòng đời ───────────────────────────────────────────────────

    def start(
        self,
        *,
        job_id: str | None = None,
        filename: str,
        workdir: Path,
        voice_id: str,
        backend_factory,
        input_label: str = "",
        job_type: str = "video_dubbing",
        target_language: str = "vi-VN",
        runner: JobRunner | None = None,
        video_path: Path | None = None,
        media: MediaInfo | None = None,
        options: PipelineOptions | None = None,
    ) -> Job:
        """Nhận job mới. Quá trần chạy đồng thời thì job nằm hàng đợi, không từ chối."""
        if job_type not in STAGES_BY_JOB_TYPE:
            raise ValueError(f"Loại công việc không được hỗ trợ: {job_type}")
        if runner is None:
            if video_path is None or media is None or options is None:
                raise TypeError("runner is required")
            target_language = options.target_language

            def legacy_video_runner(backend, progress, should_cancel) -> JobRunResult:
                result = run_pipeline(
                    backend, video_path, workdir, options, progress, media, should_cancel
                )
                return JobRunResult(
                    artifacts=[
                        JobArtifact(
                            "video", "video", Path(result.video_path).name,
                            "video/mp4", result.video_path,
                        ),
                        JobArtifact(
                            "subtitle", "subtitle", Path(result.srt_path).name,
                            "application/x-subrip", result.srt_path,
                        ),
                    ],
                    warnings=result.warnings,
                    attempted_count=result.attempted_count,
                    spoken_count=result.spoken_count,
                )

            runner = legacy_video_runner

        job = Job(
            id=job_id or uuid.uuid4().hex[:12],
            filename=filename,
            input_label=input_label or filename,
            job_type=job_type,
            target_language=target_language,
            workdir=workdir,
            voice_id=voice_id,
            stage=STAGES_BY_JOB_TYPE[job_type][0],
        )
        with self._lock:
            self._jobs[job.id] = job
        try:
            self._persist(job)
            future = self._ensure_executor().submit(self._run, job, backend_factory, runner)
        except Exception:
            with self._lock:
                self._jobs.pop(job.id, None)
                self._futures.pop(job.id, None)
            raise
        self._futures[job.id] = future
        return job

    def cancel(self, job_id: str) -> Job:
        """Yêu cầu dừng. Ném RuntimeError nếu job đã kết thúc — route đổi thành HTTP 409."""
        job = self.get(job_id)
        if job is None:
            raise LookupError(job_id)
        with self._lock:
            if job.status not in ACTIVE_STATUSES:
                raise RuntimeError(f"Công việc đã kết thúc ({job.status}), không hủy được.")
            job.cancel_event.set()

            future = self._futures.get(job_id)
            if job.status == "queued" and future is not None and future.cancel():
                # Còn nằm trong hàng đợi, chưa chiếm luồng nào — hủy được ngay lập tức.
                job.message = JobCancelledError.user_message
                job._set_status_locked("cancelled", JobCancelledError.user_message)
            else:
                job._set_status_locked("cancelling", "Đang dừng…")
        self._persist(job)
        log.info("Job %s: người dùng yêu cầu hủy ở bước %s (→ %s)", job.id, job.stage, job.status)
        return job

    def delete(self, job_id: str) -> str:
        """Xóa một job terminal cùng toàn bộ thư mục output của nó."""
        job = self.get(job_id)
        if job is None:
            raise LookupError(job_id)
        with self._lock:
            if job.status in ACTIVE_STATUSES:
                raise RuntimeError("Công việc đang chạy, không thể xóa.")
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)
        shutil.rmtree(job.workdir, ignore_errors=True)
        return job_id

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            workers = self._max_workers
            if workers is None:
                from backend.config import settings

                workers = settings.max_concurrent_jobs
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, workers), thread_name_prefix="pipeline"
            )
        return self._executor

    @staticmethod
    def _finish_failed_job(
        job: Job, status: str, message: str, *, clear_result: bool
    ) -> None:
        """Always publish a terminal failure, even if telemetry finalization breaks."""
        with job.telemetry_lock:
            if clear_result:
                job.clear_result()
            job.message = message
            try:
                job.mark_finished(status, message)
            except Exception:
                job.finished_at = time.time()
            job.status = status

    def _run(self, job: Job, backend_factory, runner, *legacy_args) -> None:
        # Internal compatibility for old direct callers; production dispatch is typed.
        if legacy_args:
            video_path = backend_factory
            media = runner
            backend_factory, options = legacy_args

            def legacy_runner(backend, progress, should_cancel) -> JobRunResult:
                result = run_pipeline(
                    backend, video_path, job.workdir, options, progress, media, should_cancel
                )
                return JobRunResult(
                    artifacts=[
                        JobArtifact("video", "video", Path(result.video_path).name,
                                    "video/mp4", result.video_path),
                        JobArtifact("subtitle", "subtitle", Path(result.srt_path).name,
                                    "application/x-subrip", result.srt_path),
                    ],
                    warnings=result.warnings,
                    attempted_count=result.attempted_count,
                    spoken_count=result.spoken_count,
                )

            runner = legacy_runner

        if job.cancel_requested:   # bấm hủy khi còn trong hàng đợi
            with job.telemetry_lock:
                job.message = JobCancelledError.user_message
                job.mark_finished("cancelled", job.message)
                job.status = "cancelled"
            self._persist(job)
            return

        with job.telemetry_lock:
            job.status = "running"
            job.message = "Đang khởi tạo pipeline"
            job.mark_started()
            job._append_event_locked({
                "at": time.time(),
                "kind": "status",
                "stage": job.stage,
                "status": "running",
                "message": job.message,
            })
        self._persist(job)

        def progress(stage: str, fraction: float, message: str) -> None:
            # Trong lúc đang dừng thì đừng ghi đè thông điệp "Đang dừng…".
            if job.cancel_requested:
                return
            job.update_progress(stage, fraction, message)

        publishing = False
        try:
            log.info("Job %s: bắt đầu pipeline, voice=%s", job.id, job.voice_id)
            backend = backend_factory()
            job.update_runtime_from_backend(backend)
            self._persist(job)
            result = runner(backend, progress, job.cancel_event.is_set)
            if not isinstance(result, JobRunResult):
                raise TypeError("Job runner must return JobRunResult")
            publishing = True
            with job.telemetry_lock:
                job.publish_result(result)
                final_stage = STAGES_BY_JOB_TYPE[job.job_type][-1]
                job.update_progress(final_stage, 1.0, "Hoàn tất")
                job.mark_finished("done", job.message)
                job.status = "done"
        except JobCancelledError as exc:
            log.info("Job %s đã dừng theo yêu cầu ở bước %s", job.id, job.stage)
            self._finish_failed_job(
                job, "cancelled", exc.user_message, clear_result=publishing
            )
        except PipelineError as exc:
            log.warning("Job %s lỗi ở bước %s: %s", job.id, job.stage, exc)
            self._finish_failed_job(
                job, "error", exc.user_message, clear_result=publishing
            )
        except Exception as exc:  # lỗi ngoài dự kiến — vẫn phải hiện được lên UI
            log.exception("Job %s hỏng bất ngờ", job.id)
            self._finish_failed_job(
                job,
                "error",
                f"Lỗi không lường trước: {exc}",
                clear_result=publishing,
            )
        self._persist(job)

    # ─── lưu và khôi phục ───────────────────────────────────────────

    def _persist(self, job: Job) -> None:
        """Ghi trạng thái xuống thư mục của job. Đĩa hỏng không được phép giết pipeline."""
        with job.telemetry_lock:
            data = job.snapshot()
            persisted_artifacts = []
            for artifact in job.artifacts:
                path = _direct_child_path(job.workdir, artifact.path, require_relative=False)
                if path is None:
                    log.warning("Bỏ qua artifact không an toàn %s của job %s", artifact.id, job.id)
                    continue
                persisted_artifacts.append({**artifact.public(), "path": path.name})
            data["artifacts"] = persisted_artifacts
            by_id = {item["id"]: item["path"] for item in persisted_artifacts}
            data["video_path"] = by_id.get("video", "")
            data["srt_path"] = by_id.get("subtitle", "")
        try:
            (job.workdir / "job.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("Không ghi được job.json cho %s: %s", job.id, exc)

    def restore(self, jobs_dir: Path) -> int:
        """Nạp lại các job từ đĩa lúc khởi động server.

        Job đang chạy dở lúc server chết không thể tiếp tục (thread đã mất) — đánh dấu
        lỗi và nói thẳng, thay vì hiện "running" mãi mãi với một tiến trình ma.
        """
        if not jobs_dir.is_dir():
            return 0

        count = 0
        for meta in jobs_dir.glob("*/job.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                job_type = str(data.get("job_type", "video_dubbing"))
                target_language = _legacy_target_language(data)
                artifacts = _restore_artifacts(data, meta.parent)
                if not artifacts and "artifacts" not in data:
                    artifacts = _legacy_artifacts(data, meta.parent, target_language)
                job = Job(
                    id=data["job_id"],
                    filename=data.get("filename", ""),
                    workdir=meta.parent,
                    voice_id=data.get("voice_id", ""),
                    job_type=job_type,
                    target_language=target_language,
                    input_label=str(data.get("input_label", data.get("filename", ""))),
                    status=data.get("status", "error"),
                    stage=data.get("stage", STAGES_BY_JOB_TYPE.get(job_type, ("extract",))[0]),
                    percent=float(data.get("percent", 0.0)),
                    message=data.get("message", ""),
                    artifacts=artifacts,
                    degraded=bool(data.get("degraded", False)),
                    attempted_count=int(data.get("attempted_count", 0)),
                    spoken_count=int(data.get("spoken_count", 0)),
                    warnings=list(data.get("warnings", [])),
                    created_at=float(data.get("created_at", meta.stat().st_mtime)),
                    started_at=_optional_float(data.get("started_at")),
                    finished_at=_optional_float(data.get("finished_at")),
                    stage_started_at=_optional_float(data.get("stage_started_at")),
                    stage_fraction=float(data.get("stage_fraction", 0.0)),
                    stage_history=list(data.get("stage_history", [])),
                    events=list(data.get("events", [])),
                    engine=str(data.get("engine", "")),
                    device=str(data.get("device", "")),
                    batch_size=_safe_int(data.get("batch_size", 0)),
                )
            except (OSError, ValueError, KeyError) as exc:
                log.warning("Bỏ qua job.json hỏng tại %s: %s", meta, exc)
                continue

            if job.status == "cancelling":
                job.status = "cancelled"
                job.message = JobCancelledError.user_message
                self._persist(job)
            elif job.status in ("queued", "running"):
                job.status = "error"
                job.message = _INTERRUPTED_MESSAGE
                self._persist(job)

            self._jobs[job.id] = job
            count += 1

        if count:
            log.info("Khôi phục %d job từ %s", count, jobs_dir)
        return count

    def prune(self, jobs_dir: Path, keep: int = 10) -> None:
        """Dọn các thư mục job cũ nhất, không bao giờ đụng vào job đang sống."""
        if not jobs_dir.is_dir():
            return
        active_dirs = {j.workdir.resolve() for j in self._jobs.values()
                       if j.status in ACTIVE_STATUSES}
        by_dir = {j.workdir.resolve(): j.id for j in self._jobs.values()}

        dirs = sorted(
            (d for d in jobs_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for stale in dirs[keep:]:
            resolved = stale.resolve()
            if resolved in active_dirs:
                continue
            shutil.rmtree(stale, ignore_errors=True)
            job_id = by_dir.get(resolved)
            if job_id:  # file đã mất thì đừng để job ma trong danh sách
                self._jobs.pop(job_id, None)
                self._futures.pop(job_id, None)


manager = JobManager()
