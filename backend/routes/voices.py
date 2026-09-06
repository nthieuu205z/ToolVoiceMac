"""Danh sách giọng đọc + tạo/xóa giọng nhân bản từ audio mẫu (OmniVoice)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import logging

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from backend.config import settings
from pipeline import custom_voices
from pipeline.audio import decode_to_pcm, pcm_to_wav_bytes, write_wav
from pipeline.backends import build_backend
from pipeline.errors import FFmpegError
from pipeline.languages import normalize_language_code, require_language
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.speech_runtime import PreviewBusyError, speech_activity
from pipeline.voice_previews import VoicePreviewStore
from pipeline.voices import available_voices, is_available, route_provider

log = logging.getLogger(__name__)
router = APIRouter()
preview_store = VoicePreviewStore(settings.previews_dir)
_preview_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="voice-preview")
_PREVIEW_LANGUAGES = ("vi-VN", "en-US")
_PREVIEW_TIMEOUT_SECONDS = 30


def _require_voice(voice_id: str, language: str):
    try:
        canonical = normalize_language_code(language)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    voices = available_voices(
        settings.tts_provider,
        settings.resolved_clone_provider,
        language=canonical,
    )
    voice = next((item for item in voices if item.id == voice_id), None)
    if voice is None:
        raise HTTPException(400, "Giọng đọc không hỗ trợ ngôn ngữ đã chọn.")
    return voice, canonical


def _preview_path(voice_id: str, language: str = "vi-VN"):
    """Return the exact language-specific path for a known voice."""
    _voice, canonical = _require_voice(voice_id, language)
    return preview_store.path(voice_id, canonical)


def _preview_url(voice_id: str, language: str) -> str:
    return (
        f"/api/voices/{voice_id}/preview?language={language}"
    )


def _preview_backend(voice_id: str):
    provider = route_provider(
        voice_id, settings.tts_provider, settings.resolved_clone_provider
    )
    return provider, build_backend(settings.provider_config_for(provider))


def _synthesize_preview_text(text: str, voice_id: str, language: str) -> bytes:
    _provider, backend = _preview_backend(voice_id)
    return backend.synthesize(text, voice_id, language=language)


def _generate_fixed_previews(voice_id: str, languages) -> None:
    _provider, backend = _preview_backend(voice_id)
    preview_store.generate_languages(voice_id, languages, backend)


def _generate_preview(voice_id: str) -> None:
    """Compatibility hook for background custom-voice preview generation."""
    _generate_fixed_previews(voice_id, _PREVIEW_LANGUAGES)


class TextPreviewRequest(BaseModel):
    text: str
    language: str


def _clone_synthesizer():
    """Engine OmniVoice đọc giọng nhân bản."""
    if settings.resolved_clone_provider == "omnivoice":
        from pipeline.omnivoice_speech import OmniVoiceSynthesizer

        return OmniVoiceSynthesizer(
            whisper_model=settings.whisper_model,
            whisper_compute_type=settings.whisper_compute_type,
            num_step=settings.omnivoice_num_step,
            batch_size=settings.omnivoice_batch_size,
        )
    raise RuntimeError("OmniVoice chưa được cài để xử lý giọng nhân bản")

# Audio mẫu: giữ tối đa 12 giây cho OmniVoice.
_SAMPLE_MAX_SECONDS = 12.0
_SAMPLE_MIN_SECONDS = 2.0
_SAMPLE_MAX_UPLOAD = 30 * 1024 * 1024

# Cùng câu với scripts/generate_voice_previews.py — file nghe thử phải nghe giống nhau.
_PREVIEW_TEXT = "Xin chào, đây là giọng đọc tiếng Việt dùng để lồng tiếng cho video của bạn."


@router.get("/api/voices")
def list_voices(language: str | None = None) -> list[dict]:
    """Kèm `preview_url` khi đã có file nghe thử; chưa có thì để rỗng, giao diện tự ẩn nút."""
    selected_language = normalize_language_code(language or "vi-VN")
    result = []
    for voice in available_voices(
        settings.tts_provider, settings.resolved_clone_provider, language=language,
    ):
        preview_urls = {
            code: _preview_url(voice.id, code)
            for code in voice.supported_languages
        }
        preview_status = {
            code: preview_store.status(voice.id, code)
            for code in voice.supported_languages
        }
        selected_url = preview_urls.get(selected_language, "")
        result.append({
            "id": voice.id,
            "display_name": voice.display_name,
            "provider": voice.provider,
            "supported_languages": voice.supported_languages,
            "preview_url": (
                selected_url
                if preview_status.get(selected_language) == "ready"
                else ""
            ),
            "preview_urls": preview_urls,
            "preview_status": preview_status,
            "custom": custom_voices.is_custom(voice.id),
        })
    return result


@router.get("/api/voices/{voice_id}/preview")
def preview_voice(voice_id: str, language: str = "vi-VN") -> FileResponse:
    """Serve an explicitly known voice preview for the Voice Lab player."""
    try:
        _voice, canonical = _require_voice(voice_id, language)
    except HTTPException as exc:
        if exc.status_code == 400:
            raise HTTPException(404, "Không tìm thấy giọng đọc này.") from exc
        raise
    path = preview_store.path(voice_id, canonical)
    if not path.is_file() and canonical == "vi-VN":
        legacy = settings.previews_dir / f"{voice_id}.wav"
        if legacy.is_file():
            path = legacy
    if not path.is_file():
        raise HTTPException(404, "Giọng này chưa có file demo.")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=f"{voice_id}-{canonical}.wav",
    )


@router.post("/api/voices/{voice_id}/preview-text")
def preview_text(voice_id: str, request: TextPreviewRequest) -> Response:
    text = request.text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise HTTPException(400, "Hãy nhập nội dung cần nghe thử.")
    _voice, canonical = _require_voice(voice_id, request.language)
    terminators = set(require_language(canonical).sentence_terminators)
    preview = text[:300]
    for index, character in enumerate(preview):
        if character in terminators:
            preview = preview[: index + 1]
            break
    provider = route_provider(
        voice_id, settings.tts_provider, settings.resolved_clone_provider
    )
    try:
        with speech_activity.preview(provider):
            future = _preview_executor.submit(
                _synthesize_preview_text, preview, voice_id, canonical
            )
            try:
                pcm = future.result(timeout=_PREVIEW_TIMEOUT_SECONDS)
            except FutureTimeoutError as exc:
                cancel = getattr(future, "cancel", None)
                if callable(cancel):
                    cancel()
                raise HTTPException(
                    504, "Tạo bản nghe thử quá thời gian. Hãy thử lại."
                ) from exc
    except PreviewBusyError as exc:
        raise HTTPException(409, exc.user_message) from exc
    return Response(content=pcm_to_wav_bytes(pcm), media_type="audio/wav")


@router.post("/api/voices/{voice_id}/preview/regenerate", status_code=202)
def regenerate_preview(voice_id: str, language: str = "vi-VN") -> dict:
    _voice, canonical = _require_voice(voice_id, language)
    if not preview_store.begin_generation(voice_id, canonical):
        raise HTTPException(409, "Bản nghe thử này đang được tạo. Hãy thử lại sau.")
    _preview_executor.submit(_generate_fixed_previews, voice_id, (canonical,))
    return {"status": "pending"}


@router.get("/api/voices/cloning")
def cloning_status() -> dict:
    """Nhân bản bật khi OmniVoice được cài — không phụ thuộc giọng dựng sẵn."""
    return {"enabled": settings.resolved_clone_provider is not None}


@router.post("/api/voices/custom")
async def create_custom_voice(name: str = Form(...), audio: UploadFile = File(...)) -> dict:
    if settings.resolved_clone_provider is None:
        raise HTTPException(400, "Nhân bản giọng đang tắt (CLONE_TTS_PROVIDER=none trong .env).")

    name = name.strip()
    if not 1 <= len(name) <= 40:
        raise HTTPException(400, "Tên giọng phải từ 1 đến 40 ký tự.")

    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "File audio rỗng.")
    if len(raw) > _SAMPLE_MAX_UPLOAD:
        raise HTTPException(413, "File mẫu quá lớn — chỉ cần một đoạn 3–8 giây.")

    voice_id = custom_voices.unique_id(name)
    _normalize_sample(raw, voice_id)
    voice = custom_voices.register(voice_id, name)

    from pipeline.omnivoice_speech import forget_clone

    forget_clone(voice_id)
    preview_store.clear(voice_id)
    for language in _PREVIEW_LANGUAGES:
        preview_store.set_pending(voice_id, language)
    _preview_executor.submit(_generate_preview, voice_id)

    return {
        "id": voice.id,
        "display_name": f"{voice.display_name} — giọng nhân bản",
        "preview_url": "",
        "preview_urls": {
            language: _preview_url(voice.id, language)
            for language in _PREVIEW_LANGUAGES
        },
        "preview_status": {language: "pending" for language in _PREVIEW_LANGUAGES},
        "custom": True,
    }


def _normalize_sample(raw: bytes, voice_id: str) -> None:
    """Mọi định dạng audio → WAV mono 24 kHz tối đa 12 giây, đúng thứ engine cần."""
    try:
        pcm = decode_to_pcm(raw, TTS_SAMPLE_RATE)
    except FFmpegError as exc:
        raise HTTPException(400, "Không đọc được file — cần một file audio (wav, mp3, m4a…).") from exc

    duration = len(pcm) / 2 / TTS_SAMPLE_RATE
    if duration < _SAMPLE_MIN_SECONDS:
        raise HTTPException(400, f"Mẫu chỉ dài {duration:.1f} giây — cần ít nhất 3 giây lời nói.")

    keep = int(_SAMPLE_MAX_SECONDS * TTS_SAMPLE_RATE) * 2
    samples = np.frombuffer(pcm[:keep], dtype="<i2")
    write_wav(custom_voices.sample_path(voice_id), samples, TTS_SAMPLE_RATE)


@router.delete("/api/voices/custom/{voice_id}")
def delete_custom_voice(voice_id: str) -> dict:
    if not custom_voices.is_custom(voice_id) or custom_voices.get(voice_id) is None:
        raise HTTPException(404, "Không tìm thấy giọng nhân bản này.")

    custom_voices.remove(voice_id)
    preview_store.clear(voice_id)

    # Bảo OmniVoice quên ref_text đã cache và sidecar của giọng.
    from pipeline.omnivoice_speech import forget_clone as omnivoice_forget

    omnivoice_forget(voice_id)
    return {"deleted": voice_id}
