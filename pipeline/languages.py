from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    display_name: str
    english_name: str
    gemini_tts_code: str
    omnivoice_name: str
    preview_text: str
    sentence_terminators: tuple[str, ...]


_LANGUAGES = (
    LanguageSpec(
        code="vi-VN",
        display_name="Tiếng Việt",
        english_name="Vietnamese",
        gemini_tts_code="vi-VN",
        omnivoice_name="Vietnamese",
        preview_text="Xin chào, đây là giọng đọc tiếng Việt dùng để lồng tiếng cho video của bạn.",
        sentence_terminators=(".", "?", "!", "…"),
    ),
    LanguageSpec(
        code="en-US",
        display_name="English (US)",
        english_name="English",
        gemini_tts_code="en-US",
        omnivoice_name="English",
        preview_text="Hello, this is an English voice preview for your audio and video projects.",
        sentence_terminators=(".", "?", "!", "…"),
    ),
)
_BY_CODE = {item.code.casefold(): item for item in _LANGUAGES}
_ALIASES = {"vi": "vi-VN", "vi-vn": "vi-VN", "en": "en-US", "en-us": "en-US"}


def available_languages() -> tuple[LanguageSpec, ...]:
    return _LANGUAGES


def normalize_language_code(code: str) -> str:
    key = str(code or "").strip().casefold()
    canonical = _ALIASES.get(key)
    if canonical is None and key in _BY_CODE:
        canonical = _BY_CODE[key].code
    if canonical is None:
        raise ValueError(f"Ngôn ngữ không được hỗ trợ: {code}")
    return canonical


def require_language(code: str) -> LanguageSpec:
    return _BY_CODE[normalize_language_code(code).casefold()]
