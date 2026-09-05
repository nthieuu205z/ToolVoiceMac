#!/usr/bin/env python
"""Chạy pipeline trực tiếp, không qua web — dùng để kiểm thử nhanh trên một clip ngắn.

    ./.venv/bin/python scripts/run_pipeline_cli.py video.mp4 --voice Charon
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402
from pipeline import custom_voices  # noqa: E402
from pipeline.backends import build_backend  # noqa: E402
from pipeline.errors import PipelineError  # noqa: E402
from pipeline.ffmpeg_utils import set_binaries  # noqa: E402
from pipeline.languages import available_languages  # noqa: E402
from pipeline.models import overall_percent  # noqa: E402
from pipeline.runner import PipelineOptions, run_pipeline  # noqa: E402
from pipeline.voices import available_voices, default_voice  # noqa: E402



def main() -> int:
    custom_voices.configure(settings.custom_voices_dir)
    try:
        settings.validate_providers()
    except ValueError as exc:
        print(f"Cấu hình không hợp lệ: {exc}", file=sys.stderr)
        return 1
    voices = available_voices(settings.tts_provider, settings.resolved_clone_provider)
    parser = argparse.ArgumentParser(description="Lồng tiếng cho một video")
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--target-language",
        choices=[language.code for language in available_languages()],
        default="vi-VN",
    )
    parser.add_argument("--voice", default=default_voice(settings.tts_provider),
                        choices=[v.id for v in voices])
    parser.add_argument("--outdir", type=Path, default=Path("jobs/cli"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    set_binaries(settings.ffmpeg_bin, settings.ffprobe_bin)


    if not settings.gemini_api_key:
        print("Chưa có GEMINI_API_KEY — tạo .env từ .env.example rồi điền khóa.", file=sys.stderr)
        return 1
    if not args.video.is_file():
        print(f"Không tìm thấy file: {args.video}", file=sys.stderr)
        return 1

    def progress(stage: str, fraction: float, message: str) -> None:
        print(f"[{overall_percent(stage, fraction):5.1f}%] {stage:<11} {message}")

    effective_tts = settings.tts_provider
    if args.voice.startswith("clone-"):
        effective_tts = settings.resolved_clone_provider or settings.tts_provider
    print(f"nhận diện: {settings.stt_provider}   dịch: gemini   giọng đọc: {effective_tts}\n")
    backend = build_backend(settings.provider_config_for(effective_tts))
    options = PipelineOptions(
        voice_id=args.voice,
        target_language=args.target_language,
        max_utterance_seconds=settings.max_utterance_seconds,
        max_utterance_gap=settings.max_utterance_gap,
        stt_workers=settings.stt_workers,
        tts_workers=settings.tts_workers,
        tts_max_speedup=settings.tts_max_speedup,
        tts_daily_budget=settings.tts_daily_budget,
        tts_is_metered=effective_tts == "gemini",
        # OmniVoice tự vá lỗ hổng (postprocess) → bỏ bước đọc-lại tốn kém.
        resynthesize_holes=effective_tts != "omnivoice",
    )

    try:
        result = run_pipeline(backend, args.video, args.outdir, options, progress)
    except PipelineError as exc:
        print(f"\nLỗi: {exc.user_message}", file=sys.stderr)
        return 1

    print(f"\nVideo:  {result.video_path}")
    print(f"Phụ đề: {result.srt_path}")
    print(f"Ngôn ngữ gốc: {result.language or 'không rõ'} · {result.segment_count} lượt thoại")
    print(f"Đã đọc: {result.spoken_count}/{result.attempted_count} lượt")
    for warning in result.warnings:
        print(f"Lưu ý: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
