"""Create persistent Text -> Voice jobs in the shared job queue."""

from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import settings
from backend.job_contracts import JobArtifact, JobRunResult
from backend.job_manager import manager
from pipeline.backends import LazyGemini
from pipeline.edge_speech import EdgeSynthesizer
from pipeline.languages import normalize_language_code
from pipeline.text_to_voice import SpeechResult, TextToVoiceOptions, run_text_to_voice
from pipeline.voices import is_available, route_provider

router = APIRouter()
log = logging.getLogger(__name__)

_KEEP_JOB_DIRS = 10
_MAX_TEXT_CHARACTERS = 50_000
_MAX_INPUT_LABEL_CHARACTERS = 80
_MAX_DOWNLOAD_STEM_CHARACTERS = 40
_UNSAFE_FILENAME = re.compile(r"[^\w .-]+", re.UNICODE)


class TextJobRequest(BaseModel):
    text: str
    voice_id: str
    language: str


def _make_backend(tts_provider: str):
    """Build only the speech synthesizer selected for this text job."""
    config = settings.provider_config_for(tts_provider)
    if tts_provider == "edge":
        return EdgeSynthesizer(config.edge_tts_attempts)
    if tts_provider == "gemini":
        return LazyGemini(config)
    if tts_provider == "omnivoice":
        from pipeline.omnivoice_speech import OmniVoiceSynthesizer

        return OmniVoiceSynthesizer(
            whisper_model=config.whisper_model,
            whisper_compute_type=config.whisper_compute_type,
            num_step=config.omnivoice_num_step,
            batch_size=config.omnivoice_batch_size,
        )
    raise ValueError(f"Nhà cung cấp TTS không hợp lệ: {tts_provider}")


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _input_label(text: str) -> str:
    return " ".join(text.split())[:_MAX_INPUT_LABEL_CHARACTERS]


def _download_stem(input_label: str) -> str:
    cleaned = _UNSAFE_FILENAME.sub("", input_label[:_MAX_DOWNLOAD_STEM_CHARACTERS])
    cleaned = " ".join(cleaned.split()).strip(" .-")
    return cleaned or "speech"


def _text_job_result(result: SpeechResult, input_label: str) -> JobRunResult:
    stem = _download_stem(input_label)
    return JobRunResult(
        artifacts=[
            JobArtifact("wav", "wav", f"{stem}.wav", "audio/wav", result.wav_path),
            JobArtifact("mp3", "mp3", f"{stem}.mp3", "audio/mpeg", result.mp3_path),
        ],
        warnings=result.warnings,
        attempted_count=result.attempted_count,
        spoken_count=result.spoken_count,
    )


def _reserve_workdir(jobs_dir: Path) -> tuple[str, Path]:
    while True:
        job_id = uuid.uuid4().hex[:12]
        workdir = jobs_dir / job_id
        try:
            workdir.mkdir(parents=True, mode=0o700)
        except FileExistsError:
            continue
        return job_id, workdir


def _write_private_text(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


def _remove_workdir(workdir: Path | None) -> None:
    if workdir is None:
        return
    try:
        shutil.rmtree(workdir)
    except FileNotFoundError:
        return
    except OSError:
        log.exception("Không thể dọn thư mục công việc %s", workdir)


@router.post("/api/jobs/text")
def create_text_job(request: TextJobRequest) -> dict:
    text = _normalize_text(request.text)
    if not text:
        raise HTTPException(400, "Hãy nhập nội dung cần đọc.")
    if len(text) > _MAX_TEXT_CHARACTERS:
        raise HTTPException(413, "Nội dung vượt quá giới hạn 50.000 ký tự.")
    try:
        language = normalize_language_code(request.language)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    clone_provider = settings.resolved_clone_provider
    if not is_available(
        request.voice_id,
        settings.tts_provider,
        clone_provider,
        language=language,
    ):
        raise HTTPException(400, "Giọng đọc không hỗ trợ ngôn ngữ đã chọn.")
    try:
        effective_tts = route_provider(
            request.voice_id, settings.tts_provider, clone_provider
        )
    except ValueError as exc:
        raise HTTPException(
            500, f"Cấu hình nhà cung cấp không hợp lệ: {exc}"
        ) from exc

    input_label = _input_label(text)
    options = TextToVoiceOptions(
        voice_id=request.voice_id,
        language=language,
        max_characters=_MAX_TEXT_CHARACTERS,
    )

    manager.prune(settings.jobs_dir, keep=_KEEP_JOB_DIRS)
    workdir: Path | None = None
    try:
        job_id, workdir = _reserve_workdir(settings.jobs_dir)
        input_path = workdir / "input.txt"
        _write_private_text(input_path, text)
    except OSError as exc:
        _remove_workdir(workdir)
        raise HTTPException(500, "Không thể lưu nội dung công việc.") from exc

    def runner(backend, progress, should_cancel) -> JobRunResult:
        result = run_text_to_voice(
            backend,
            text,
            workdir,
            options,
            progress=progress,
            should_cancel=should_cancel,
        )
        return _text_job_result(result, input_label)

    try:
        job = manager.start(
            job_id=job_id,
            filename=input_label,
            input_label=input_label,
            job_type="text_to_voice",
            target_language=language,
            workdir=workdir,
            voice_id=request.voice_id,
            backend_factory=lambda: _make_backend(effective_tts),
            runner=runner,
        )
    except Exception as exc:
        _remove_workdir(workdir)
        raise HTTPException(500, "Không thể thêm công việc vào hàng đợi.") from exc
    return {"job_id": job.id}
