"""Vòng đời một công việc: nhận video → chạy pipeline → trả file kết quả."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import settings
from backend.job_contracts import JobArtifact, JobRunResult, normalize_job_name
from backend.job_manager import Job, manager
from pipeline.backends import CompositeBackend, build_backend
from pipeline.errors import UnsupportedMediaError
from pipeline.languages import normalize_language_code
from pipeline.models import PipelineResult
from pipeline.probe import probe_video
from pipeline.runner import PipelineOptions, run_pipeline
from pipeline.voices import is_available, route_provider
from pipeline.speech_project import read_project_manifest, safe_download_stem

router = APIRouter()

_UPLOAD_CHUNK = 1024 * 1024
_SSE_POLL_SECONDS = 0.4
_SSE_HEARTBEAT_SECONDS = 15.0
# Giữ lại các job gần nhất để còn tải kết quả về; cũ hơn thì dọn cho nhẹ đĩa.
_KEEP_JOB_DIRS = 10


def _make_backend(tts_provider: str) -> CompositeBackend:
    """Engine giọng đọc CHỈ ĐỊNH cho job — provider đã được định tuyến theo giọng đã chọn."""
    return build_backend(settings.provider_config_for(tts_provider))


def _video_job_result(
    result: PipelineResult, filename: str, language: str, name: str = ""
) -> JobRunResult:
    suffix = "vi" if language == "vi-VN" else "en"
    stem = safe_download_stem(name or Path(filename).stem)
    return JobRunResult(
        artifacts=[
            JobArtifact(
                "video",
                "video",
                f"{stem}_{suffix}{Path(result.video_path).suffix}",
                "video/mp4",
                result.video_path,
            ),
            JobArtifact(
                "subtitle",
                "subtitle",
                f"{stem}_{suffix}.srt",
                "application/x-subrip",
                result.srt_path,
            ),
        ],
        warnings=result.warnings,
        attempted_count=result.attempted_count,
        spoken_count=result.spoken_count,
    )


@router.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    voice_id: str = Form(...),
    target_language: str = Form("vi-VN"),
    name: str | None = Form(None),
) -> dict:
    try:
        job_name = normalize_job_name(name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    # Bước dịch luôn cần Gemini, kể cả khi nhận diện và giọng đọc đã chạy miễn phí.
    if not settings.gemini_api_key:
        raise HTTPException(500, "Chưa cấu hình dịch vụ dịch thuật. Hãy điền khóa trong file .env.")
    try:
        language = normalize_language_code(target_language)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    clone_provider = settings.resolved_clone_provider
    if not is_available(
        voice_id,
        settings.tts_provider,
        clone_provider,
        language=language,
    ):
        raise HTTPException(400, "Giọng đọc không hỗ trợ ngôn ngữ đã chọn.")

    # Định tuyến một lần theo giọng: giọng nhân bản → OmniVoice; còn lại → tts_provider.
    try:
        effective_tts = route_provider(voice_id, settings.tts_provider, clone_provider)
    except ValueError as exc:
        raise HTTPException(500, f"Cấu hình nhà cung cấp không hợp lệ: {exc}") from exc

    manager.prune(settings.jobs_dir, keep=_KEEP_JOB_DIRS)
    workdir = settings.jobs_dir / uuid.uuid4().hex[:12]
    workdir.mkdir(parents=True)

    video_path = workdir / f"input{Path(video.filename or 'video.mp4').suffix or '.mp4'}"
    await _save_upload(video, video_path)

    # ffprobe trước khi tiêu tốn bất kỳ token Gemini nào.
    try:
        media = probe_video(video_path)
    except UnsupportedMediaError as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(400, exc.user_message) from exc

    options = PipelineOptions(
        voice_id=voice_id,
        target_language=language,
        max_utterance_seconds=settings.max_utterance_seconds,
        max_utterance_gap=settings.max_utterance_gap,
        sentence_level_timing=settings.sentence_level_timing,
        stt_workers=settings.stt_workers,
        tts_workers=settings.tts_workers,
        translate_workers=settings.translate_workers,
        tts_max_speedup=settings.tts_max_speedup,
        tts_fill_slowdown=settings.tts_fill_slowdown,
        tts_daily_budget=settings.tts_daily_budget,
        tts_is_metered=effective_tts == "gemini",
        # OmniVoice tự vá lỗ hổng im lặng (postprocess) nên KHÔNG chạy bước đọc-lại tốn kém
        # (mỗi lần là một single-synth, không gộp lô).
        resynthesize_holes=effective_tts != "omnivoice",
    )
    filename = video.filename or video_path.name

    def runner(backend, progress, should_cancel) -> JobRunResult:
        result = run_pipeline(
            backend,
            video_path,
            workdir,
            options,
            progress,
            media,
            should_cancel,
        )
        return _video_job_result(result, filename, language, job_name)

    job = manager.start(
        filename=filename,
        input_label=filename,
        name=job_name,
        job_type="video_dubbing",
        target_language=language,
        workdir=workdir,
        voice_id=voice_id,
        backend_factory=lambda: _make_backend(effective_tts),
        runner=runner,
    )
    return {"job_id": job.id}


@router.get("/api/jobs")
def list_jobs() -> dict:
    """Danh sách mọi job, mới nhất trước — giao diện vẽ thẳng từ đây."""
    return {"jobs": manager.jobs()}


async def _save_upload(video: UploadFile, dest: Path) -> None:
    """Ghi từng khối để video lớn không phải nằm hết trong RAM, và chặn file quá cỡ."""
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while chunk := await video.read(_UPLOAD_CHUNK):
            written += len(chunk)
            if written > limit:
                out.close()
                shutil.rmtree(dest.parent, ignore_errors=True)
                raise HTTPException(413, f"Video vượt quá giới hạn {settings.max_upload_mb} MB.")
            out.write(chunk)
    if written == 0:
        raise HTTPException(400, "File tải lên rỗng.")


@router.get("/api/jobs/current")
def current_job() -> dict:
    job = manager.current
    return job.snapshot() if job else {}


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _require(job_id).snapshot()


@router.get("/api/jobs/{job_id}/telemetry")
def job_telemetry(job_id: str) -> dict:
    """Detailed monitor payload kept separate from the compact job list."""
    job = _require(job_id)
    payload = job.snapshot()
    payload["telemetry_only"] = True
    return payload


@router.get("/api/jobs/{job_id}/project")
def job_project(job_id: str) -> dict:
    job = _require(job_id)
    if job.job_type != "text_to_voice" or not job.is_project:
        raise HTTPException(404, "Công việc này không phải dự án giọng đọc.")
    try:
        manifest = read_project_manifest(job.workdir)
        if job.status in ("error", "cancelled"):
            manifest.update(status=job.status, error=job.message)
            for part in manifest["parts"]:
                if part.get("status") == "running":
                    part.update(status=job.status, error=job.message)
        return manifest
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(404, "Không tìm thấy thông tin dự án.") from None


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Yêu cầu dừng. Pipeline dừng ở mốc an toàn gần nhất, không giết thread giữa chừng."""
    try:
        return manager.cancel(job_id).snapshot()
    except LookupError:
        raise HTTPException(404, "Không tìm thấy công việc này.") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Xóa job đã kết thúc và dọn toàn bộ output của job."""
    try:
        return {"deleted": manager.delete(job_id)}
    except LookupError:
        raise HTTPException(404, "Không tìm thấy công việc này.") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """SSE bằng cách theo dõi trạng thái job — không cần hàng đợi giữa thread và event loop."""
    _require(job_id)

    async def stream():
        last: dict | None = None
        idle = 0.0
        while True:
            job = manager.get(job_id)
            if job is None:
                break

            snapshot = job.snapshot()
            if snapshot != last:
                yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                last = snapshot
                idle = 0.0
            elif idle >= _SSE_HEARTBEAT_SECONDS:
                yield ": keepalive\n\n"  # giữ kết nối qua proxy/trình duyệt
                idle = 0.0

            if job.status in ("done", "error", "cancelled"):
                break

            await asyncio.sleep(_SSE_POLL_SECONDS)
            idle += _SSE_POLL_SECONDS

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/jobs/{job_id}/download/video")
def download_video(job_id: str) -> FileResponse:
    return _download_artifact(job_id, "video")


@router.get("/api/jobs/{job_id}/download/srt")
def download_srt(job_id: str) -> FileResponse:
    return _download_artifact(job_id, "subtitle")


@router.get("/api/jobs/{job_id}/artifacts/{artifact_id}")
def download_artifact(job_id: str, artifact_id: str) -> FileResponse:
    return _download_artifact(job_id, artifact_id)


def _download_artifact(job_id: str, artifact_id: str) -> FileResponse:
    job = _require_done(job_id)
    artifact = next((item for item in job.artifacts if item.id == artifact_id), None)
    if artifact is None:
        raise HTTPException(404, "Không tìm thấy file kết quả.")
    path = _job_file(job, artifact.path)
    return _serve(path, artifact.filename, artifact.media_type)


def _job_file(job: Job, value: str) -> Path:
    """Chỉ phục vụ file nằm trong thư mục của chính job."""
    try:
        raw = Path(value)
        root = job.workdir.resolve(strict=True)
        path = raw.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(404, "Không tìm thấy file kết quả.") from None
    if ".." in raw.parts or raw.is_symlink() or path != root and root not in path.parents:
        raise HTTPException(404, "Không tìm thấy file kết quả.")
    if path.parent != root:
        raise HTTPException(404, "Không tìm thấy file kết quả.")
    return path


def _serve(path: Path, download_name: str, media_type: str | None = None) -> FileResponse:
    if not path.is_file():
        raise HTTPException(404, "Không tìm thấy file kết quả.")
    return FileResponse(path, filename=download_name, media_type=media_type)


def _require(job_id: str) -> Job:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy công việc này.")
    return job


def _require_done(job_id: str) -> Job:
    job = _require(job_id)
    if job.status != "done":
        raise HTTPException(409, "Công việc chưa hoàn tất.")
    return job
