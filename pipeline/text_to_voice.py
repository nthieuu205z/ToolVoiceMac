"""Dedicated unconstrained Text -> Voice synthesis and audio export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import pcm_to_array, write_wav
from .errors import (
    InvalidTextError,
    JobCancelledError,
    SpeechServiceError,
    TextTooLongError,
)
from .ffmpeg_utils import resolve, run
from .languages import require_language
from .models import (
    TTS_SAMPLE_RATE,
    CancelFn,
    ProgressFn,
    SpeechSynthesizer,
    never_cancel,
    noop_progress,
)
from .speech_synthesis import synthesize_texts
from .speech_runtime import speech_activity


@dataclass(frozen=True)
class TextToVoiceOptions:
    voice_id: str
    language: str
    max_characters: int = 50_000
    inter_chunk_silence_ms: int = 200
    max_chunk_characters: int = 1_000


@dataclass
class SpeechResult:
    wav_path: str
    mp3_path: str
    attempted_count: int
    spoken_count: int
    warnings: list[str] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_oversized_sentence(sentence: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = sentence
    while len(remaining) > max_chars:
        split_at = max(
            (
                index
                for index, character in enumerate(remaining[: max_chars + 1])
                if character.isspace()
            ),
            default=-1,
        )
        if split_at <= 0:
            split_at = max_chars
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def chunk_text(text: str, language: str, *, max_chars: int = 1_000) -> list[str]:
    """Split normalized text after language punctuation, then at safe request sizes."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    normalized = _normalize_text(text)
    if not normalized:
        return []

    terminators = set(require_language(language).sentence_terminators)
    sentences: list[str] = []
    begin = 0
    for index, character in enumerate(normalized):
        if character in terminators:
            sentence = normalized[begin : index + 1].strip()
            if sentence:
                sentences.append(sentence)
            begin = index + 1
    tail = normalized[begin:].strip()
    if tail:
        sentences.append(tail)

    chunks: list[str] = []
    for sentence in sentences:
        chunks.extend(_split_oversized_sentence(sentence, max_chars))
    return chunks


def build_mp3_cmd(wav_path: Path, mp3_path: Path, *, ffmpeg: str) -> list[str]:
    """Build the deterministic audio-only MP3 export command."""
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-map",
        "0:a:0",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(mp3_path),
    ]


def encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    """Encode one project-rate mono WAV with the configured FFmpeg binary."""
    run(build_mp3_cmd(wav_path, mp3_path, ffmpeg=resolve("ffmpeg")), timeout=300)


def _raise_if_cancelled(should_cancel: CancelFn) -> None:
    if should_cancel():
        raise JobCancelledError()


def _remove_unpublished_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def run_text_to_voice(
    synthesizer: SpeechSynthesizer,
    text: str,
    workdir: Path,
    options: TextToVoiceOptions,
    *,
    progress: ProgressFn = noop_progress,
    should_cancel: CancelFn = never_cancel,
) -> SpeechResult:
    """Synthesize unconstrained text and atomically publish WAV plus MP3 paths."""
    normalized = _normalize_text(text)
    if not normalized:
        raise InvalidTextError()
    if len(normalized) > options.max_characters:
        raise TextTooLongError()

    language = require_language(options.language).code
    chunks = chunk_text(
        normalized,
        language,
        max_chars=options.max_chunk_characters,
    )
    progress("prepare", 1.0, f"Đã chuẩn bị {len(chunks)} đoạn văn bản")

    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    wav_path = root / "output.wav"
    mp3_path = root / "output.mp3"
    _remove_unpublished_outputs(wav_path, mp3_path)

    _raise_if_cancelled(should_cancel)
    provider = str(getattr(synthesizer, "engine", "") or "unknown")
    with speech_activity.production(provider):
        outputs, warnings = synthesize_texts(
            synthesizer,
            chunks,
            options.voice_id,
            language=language,
            progress=progress,
            should_cancel=should_cancel,
        )
    _raise_if_cancelled(should_cancel)

    successful = [pcm_to_array(value) for value in outputs if value is not None]
    if not successful:
        _remove_unpublished_outputs(wav_path, mp3_path)
        raise SpeechServiceError()

    silence_samples = round(
        TTS_SAMPLE_RATE * max(0, options.inter_chunk_silence_ms) / 1_000
    )
    silence = np.zeros(silence_samples, dtype="<i2")
    parts: list[np.ndarray] = []
    for index, samples in enumerate(successful):
        if index:
            parts.append(silence)
        parts.append(samples)
    assembled = np.concatenate(parts).astype("<i2", copy=False)

    progress("assemble", 0.0, "Đang ghép các đoạn giọng đọc")
    try:
        _raise_if_cancelled(should_cancel)
        write_wav(wav_path, assembled, TTS_SAMPLE_RATE)
        progress("assemble", 1.0, "Đã ghép xong giọng đọc")

        progress("export", 0.0, "Đang xuất WAV và MP3")
        _raise_if_cancelled(should_cancel)
        encode_mp3(wav_path, mp3_path)
        progress("export", 1.0, "Đã xuất WAV và MP3")
    except Exception:
        _remove_unpublished_outputs(wav_path, mp3_path)
        raise

    return SpeechResult(
        wav_path=str(wav_path),
        mp3_path=str(mp3_path),
        attempted_count=len(chunks),
        spoken_count=len(successful),
        warnings=warnings,
    )
