"""Danh sách giọng đọc + tạo/xóa giọng nhân bản từ audio mẫu (OmniVoice)."""

from __future__ import annotations

import logging
import threading

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import settings
from pipeline import custom_voices
from pipeline.audio import decode_to_pcm, write_wav
from pipeline.errors import FFmpegError
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.voices import available_voices

log = logging.getLogger(__name__)
router = APIRouter()


def _preview_path(voice_id: str):
    """Return only a safe preview filename for a known voice id."""
    if not custom_voices.is_custom(voice_id) and not any(
        voice.id == voice_id
        for voice in available_voices(settings.tts_provider, settings.resolved_clone_provider)
    ):
        raise HTTPException(404, "Không tìm thấy giọng đọc này.")
    return settings.previews_dir / f"{voice_id}.wav"


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
    result = []
    for voice in available_voices(
        settings.tts_provider, settings.resolved_clone_provider, language=language,
    ):
        preview = _preview_path(voice.id)
        result.append({
            "id": voice.id,
            "display_name": voice.display_name,
            "provider": voice.provider,
            "supported_languages": voice.supported_languages,
            "preview_url": f"/previews/{voice.id}.wav" if preview.is_file() else "",
            "custom": custom_voices.is_custom(voice.id),
        })
    return result


@router.get("/api/voices/{voice_id}/preview")
def preview_voice(voice_id: str) -> FileResponse:
    """Serve an explicitly known voice preview for the Voice Lab player."""
    path = _preview_path(voice_id)
    if not path.is_file():
        raise HTTPException(404, "Giọng này chưa có file demo.")
    return FileResponse(path, media_type="audio/wav", filename=f"{voice_id}.wav")


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

    # Nghe thử tạo ở nền: lần đầu phải nạp engine (~90 giây), không bắt request đợi.
    threading.Thread(target=_generate_preview, args=(voice_id,), daemon=True,
                     name=f"preview-{voice_id}").start()

    preview = _preview_path(voice.id)
    return {"id": voice.id, "display_name": f"{voice.display_name} — giọng nhân bản",
            "preview_url": f"/previews/{voice.id}.wav" if preview.is_file() else "",
            "custom": True}


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


def _generate_preview(voice_id: str) -> None:
    from pipeline.audio import pcm_to_array

    try:
        pcm = _clone_synthesizer().synthesize(_PREVIEW_TEXT, voice_id)
        preview = _preview_path(voice_id)
        write_wav(preview, pcm_to_array(pcm))
        log.info("Đã tạo file nghe thử cho giọng nhân bản %s", voice_id)
    except Exception as exc:  # thiếu nghe thử không phải lỗi chết người
        log.warning("Không tạo được nghe thử cho %s: %s", voice_id, exc)


@router.delete("/api/voices/custom/{voice_id}")
def delete_custom_voice(voice_id: str) -> dict:
    if not custom_voices.is_custom(voice_id) or custom_voices.get(voice_id) is None:
        raise HTTPException(404, "Không tìm thấy giọng nhân bản này.")

    custom_voices.remove(voice_id)
    _preview_path(voice_id).unlink(missing_ok=True)

    # Bảo OmniVoice quên ref_text đã cache và sidecar của giọng.
    from pipeline.omnivoice_speech import forget_clone as omnivoice_forget

    omnivoice_forget(voice_id)
    return {"deleted": voice_id}
