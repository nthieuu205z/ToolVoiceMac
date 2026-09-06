#!/usr/bin/env python
"""Tạo các file nghe thử đa ngôn ngữ cho giao diện.

    ./.venv/bin/python scripts/generate_voice_previews.py

Ghi ra web/static/previews/<VoiceId>/<language>.wav.
Dùng --force để tạo lại các file đã có.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402
from pipeline import custom_voices  # noqa: E402
from pipeline.backends import build_backend  # noqa: E402
from pipeline.ffmpeg_utils import set_binaries  # noqa: E402
from pipeline.voice_previews import VoicePreviewStore  # noqa: E402
from pipeline.voices import available_voices  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="tạo lại cả file đã có")
    args = parser.parse_args()

    set_binaries(settings.ffmpeg_bin, settings.ffprobe_bin)
    custom_voices.configure(settings.custom_voices_dir)
    try:
        settings.validate_providers()
    except ValueError as exc:
        print(f"Cấu hình không hợp lệ: {exc}", file=sys.stderr)
        return 1
    if settings.tts_provider == "gemini" and not settings.gemini_api_key:
        print("Chưa có GEMINI_API_KEY — tạo .env từ .env.example rồi điền khóa.", file=sys.stderr)
        return 1

    settings.previews_dir.mkdir(parents=True, exist_ok=True)
    store = VoicePreviewStore(settings.previews_dir)
    voices = available_voices(settings.tts_provider, settings.resolved_clone_provider)
    backend_cache = {}
    print(f"Giọng preset: {settings.tts_provider}; tổng cộng {len(voices)} giọng\n")

    failed = 0
    for voice in voices:
        provider = "omnivoice" if voice.custom else settings.tts_provider
        if provider not in backend_cache:
            backend_cache[provider] = build_backend(settings.provider_config_for(provider))
        backend = backend_cache[provider]
        languages = tuple(
            language
            for language in voice.supported_languages
            if args.force or not store.path(voice.id, language).is_file()
        )
        if not languages:
            print(f"  bỏ qua {voice.id} (đã có đủ ngôn ngữ)")
            continue
        try:
            if args.force:
                store.clear(voice.id)
                languages = tuple(voice.supported_languages)
            store.generate_languages(voice.id, languages, backend)
            failed_languages = [
                language
                for language in languages
                if store.status(voice.id, language) != "ready"
            ]
            if failed_languages:
                failed += len(failed_languages)
                print(
                    f"  lỗi với giọng {voice.id}: {', '.join(failed_languages)}",
                    file=sys.stderr,
                )
            else:
                print(f"  đã tạo {voice.id}: {', '.join(languages)}")
        except Exception as exc:
            failed += 1
            print(f"  lỗi với giọng {voice.id}: {exc}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
