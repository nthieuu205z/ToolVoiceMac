"""Ứng dụng FastAPI: phục vụ giao diện tĩnh và các API của pipeline."""

from __future__ import annotations

import logging
import os
from toolvoice import __version__
from toolvoice.paths import profile_id
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.job_manager import manager
from backend.routes import jobs, languages, model, settings as settings_routes, text_jobs, voices
from backend.process_control import schedule_shutdown
from pipeline import custom_voices
from pipeline.ffmpeg_utils import set_binaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Nằm trong lifespan chứ KHÔNG ở cấp import: test import app hàng chục lần,
    # không được phép mỗi lần lại nạp thật model nặng vào RAM.
    settings.validate_providers()
    clone = settings.resolved_clone_provider

    if settings.stt_provider == "whisper":
        from pipeline import whisper_stt

        # Whisper nạp mất ~11s lần lạnh — nạp nền song song để bước "quét" job đầu khỏi chờ.
        whisper_stt.prewarm(settings.whisper_model, settings.whisper_compute_type)
    if clone == "omnivoice":
        from pipeline import omnivoice_speech

        omnivoice_speech.configure_optimization(
            settings.omnivoice_optimization, codec_device=settings.omnivoice_codec_device
        )
        # OmniVoice nạp mất ~30s lần lạnh — nạp nền để job giọng nhân bản đầu khỏi chờ.
        omnivoice_speech.prewarm()
    yield


app = FastAPI(title="ToolVoiceMac", docs_url=None, redoc_url=None, lifespan=_lifespan)

set_binaries(settings.ffmpeg_bin, settings.ffprobe_bin)
# Giọng nhân bản lưu trong thư mục riêng, sống qua khởi động lại server.
custom_voices.configure(settings.custom_voices_dir)
# Nạp lại danh sách job từ đĩa — đóng trình duyệt hay khởi động lại server không làm mất lịch sử.
manager.restore(settings.jobs_dir)

app.include_router(voices.router)
app.include_router(languages.router)
app.include_router(model.router)
app.include_router(jobs.router)
app.include_router(text_jobs.router)
app.include_router(settings_routes.router)


@app.get("/api/toolvoice")
def tool_identity() -> dict:
    return {"app": "toolvoice", "version": __version__, "profile_id": profile_id(), "session_id": os.environ.get("TOOLVOICE_SESSION_ID", "")}


@app.post("/api/shutdown")
def shutdown_tool() -> dict:
    """Schedule a scoped process-tree shutdown after returning the acknowledgement."""
    schedule_shutdown()
    return {"shutting_down": True}

# Mount sau cùng để các route /api/* được khớp trước.
app.mount("/previews", StaticFiles(directory=settings.previews_dir, check_dir=False), name="previews")
app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")
