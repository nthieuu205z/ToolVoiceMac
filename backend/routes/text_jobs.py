"""Create persistent Text -> Voice jobs in the shared job queue."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from backend.config import settings
from backend.job_contracts import JobArtifact, JobRunResult, normalize_job_name
from backend.job_manager import manager
from pipeline.backends import LazyGemini
from pipeline.edge_speech import EdgeSynthesizer
from pipeline.languages import normalize_language_code
from pipeline import omnivoice_settings as omni_options
from pipeline.omnivoice_settings import OmniVoiceSettings
from pipeline.text_to_voice import chunk_text
from pipeline.text_to_voice import SpeechResult, TextToVoiceOptions, run_text_to_voice
from pipeline.voices import is_available, route_provider
from pipeline.speech_project import initialize_project, run_speech_project, safe_download_stem

router = APIRouter()
log = logging.getLogger(__name__)

_KEEP_JOB_DIRS = 10
_MAX_TEXT_CHARACTERS = 200_000
_SHORT_TEXT_CHARACTERS = 50_000
_MAX_INPUT_LABEL_CHARACTERS = 80


class TextJobRequest(BaseModel):
    text: str
    voice_id: str
    language: str
    omnivoice: OmniVoiceSettings | None = None
    name: str | None = None
    project: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        return normalize_job_name(value) if value is not None else None


def _make_backend(tts_provider: str, *, config=None, generation_settings=None):
    """Build only the speech synthesizer selected for this text job."""
    config = config or settings.provider_config_for(tts_provider)
    if tts_provider == "edge":
        return EdgeSynthesizer(config.edge_tts_attempts)
    if tts_provider == "gemini":
        return LazyGemini(config)
    if tts_provider == "omnivoice":
        from pipeline.omnivoice_speech import OmniVoiceSynthesizer

        return OmniVoiceSynthesizer(
            generation_settings=generation_settings,
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
    return safe_download_stem(input_label)


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


@router.get("/api/omnivoice/settings")
def omnivoice_settings() -> dict:
    capabilities = omni_options.installed_capabilities()
    if capabilities["supported"]:
        capabilities["defaults"]["num_step"] = settings.omnivoice_num_step
    return capabilities


@router.get("/api/omnivoice/runtime")
def omnivoice_runtime() -> dict:
    from pipeline.omnivoice_speech import runtime_status

    return runtime_status()


@router.post("/api/jobs/text")
def create_text_job(request: TextJobRequest) -> dict:
    text = _normalize_text(request.text)
    if not text:
        raise HTTPException(400, "Hãy nhập nội dung cần đọc.")
    if len(text) > _MAX_TEXT_CHARACTERS:
        raise HTTPException(413, "Nội dung vượt quá giới hạn 200.000 ký tự.")
    is_project = request.project or len(text) > _SHORT_TEXT_CHARACTERS
    name = request.name or ""
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

    resolved = None
    provider_config = None
    if request.omnivoice is not None and effective_tts != "omnivoice":
        raise HTTPException(400, "Cấu hình nâng cao chỉ áp dụng cho OmniVoice.")
    if effective_tts == "omnivoice":
        provider_config = settings.provider_config_for(effective_tts)
        capabilities = omnivoice_settings()
        if request.omnivoice is not None and not capabilities["supported"]:
            raise HTTPException(400, "OmniVoice đã cài không hỗ trợ cấu hình nâng cao.")
        if capabilities["supported"]:
            overrides = request.omnivoice.model_dump(exclude_unset=True) if request.omnivoice else {}
            resolved = OmniVoiceSettings(**(capabilities["defaults"] | overrides))
            if resolved.duration is not None and len(chunk_text(text, language)) != 1:
                raise HTTPException(400, "Duration chỉ hỗ trợ một câu tối đa 1.000 ký tự; để trống khi có nhiều đoạn.")

    input_label = _input_label(text)
    options = TextToVoiceOptions(
        voice_id=request.voice_id,
        language=language,
        max_characters=_SHORT_TEXT_CHARACTERS,
    )

    manager.prune(settings.jobs_dir, keep=_KEEP_JOB_DIRS)
    workdir: Path | None = None
    try:
        job_id, workdir = _reserve_workdir(settings.jobs_dir)
        input_path = workdir / "input.txt"
        _write_private_text(input_path, text)
        if is_project:
            initialize_project(text, workdir, options, name=name or input_label)
    except OSError as exc:
        _remove_workdir(workdir)
        raise HTTPException(500, "Không thể lưu nội dung công việc.") from exc

    def runner(backend, progress, should_cancel) -> JobRunResult:
        private_text = input_path.read_text(encoding="utf-8")
        if is_project:
            result = run_speech_project(
                backend, private_text, workdir, options, name=name or input_label,
                progress=progress, should_cancel=should_cancel,
            )
            return JobRunResult(
                artifacts=[JobArtifact("project_zip", "zip", result.filename, "application/zip", result.zip_path)],
                warnings=result.warnings, attempted_count=result.attempted_count, spoken_count=result.spoken_count,
            )
        result = run_text_to_voice(
            backend,
            private_text,
            workdir,
            options,
            progress=progress,
            should_cancel=should_cancel,
        )
        return _text_job_result(result, name or input_label)

    try:
        job = manager.start(
            job_id=job_id,
            filename=input_label,
            input_label=input_label,
            name=name,
            is_project=is_project,
            job_type="text_to_voice",
            target_language=language,
            workdir=workdir,
            voice_id=request.voice_id,
            omnivoice_settings=resolved,
            backend_factory=(
                (lambda: _make_backend(effective_tts, config=provider_config, generation_settings=resolved))
                if effective_tts == "omnivoice" else (lambda: _make_backend(effective_tts))
            ),
            runner=runner,
        )
    except Exception as exc:
        _remove_workdir(workdir)
        raise HTTPException(500, "Không thể thêm công việc vào hàng đợi.") from exc
    return {"job_id": job.id}
