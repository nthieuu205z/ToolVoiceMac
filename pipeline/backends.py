"""Ghép ba nhà cung cấp rời thành một backend mà pipeline mong đợi.

Mỗi bước đổi độc lập qua .env:

    STT_PROVIDER=whisper|gemini    nhận diện giọng nói
    TTS_PROVIDER=edge|gemini       tạo giọng đọc preset
    CLONE_TTS_PROVIDER=omnivoice   tạo giọng đọc clone
    (dịch luôn dùng Gemini — chỉ tốn 1–4 lượt gọi cho cả video)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import SpeechRecognizer, SpeechSynthesizer, Translator


@dataclass(frozen=True)
class ProviderConfig:
    stt_provider: str
    tts_provider: str
    gemini_api_key: str
    gemini_stt_model: str
    gemini_translate_model: str
    gemini_tts_model: str
    gemini_backend: str = "developer"   # developer (AI Studio) | vertex (Google Cloud)
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_cpu_batch_size: int = 8
    edge_tts_attempts: int = 5
    omnivoice_num_step: int = 32
    omnivoice_batch_size: int = 0


class LazyGemini:
    """Chỉ dựng GeminiRunner khi thật sự có lời gọi tới Gemini.

    Nếu dựng sớm thì chạy `generate_voice_previews.py` với edge-tts cũng đòi khóa API.
    """

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._runner = None

    @property
    def engine(self) -> str:
        return "gemini"

    @property
    def device(self) -> str:
        return "cloud"

    def _get(self):
        if self._runner is None:
            from .gemini import GeminiRunner

            self._runner = GeminiRunner(
                api_key=self._config.gemini_api_key,
                stt_model=self._config.gemini_stt_model,
                translate_model=self._config.gemini_translate_model,
                tts_model=self._config.gemini_tts_model,
                use_vertex=self._config.gemini_backend == "vertex",
            )
        return self._runner

    def transcribe_clip(self, wav_path: Path) -> tuple[str, str]:
        return self._get().transcribe_clip(wav_path)

    def translate(
        self,
        texts: list[str],
        durations: list[float],
        context: str = "",
        *,
        target_language: str = "vi-VN",
    ) -> list[str]:
        return self._get().translate(
            texts, durations, context, target_language=target_language
        )

    def synthesize(
        self, text: str, voice_id: str, *, language: str = "vi-VN"
    ) -> bytes:
        return self._get().synthesize(text, voice_id, language=language)


class CompositeBackend:
    """Bám giao thức GeminiBackend, nhưng mỗi phương thức đi tới một nhà cung cấp khác nhau."""

    def __init__(
        self,
        recognizer: SpeechRecognizer,
        translator: Translator,
        synthesizer: SpeechSynthesizer,
    ):
        self._recognizer = recognizer
        self._translator = translator
        self._synthesizer = synthesizer

    def transcribe_clip(self, wav_path: Path) -> tuple[str, str]:
        return self._recognizer.transcribe_clip(wav_path)

    def translate(
        self,
        texts: list[str],
        durations: list[float],
        context: str = "",
        *,
        target_language: str = "vi-VN",
    ) -> list[str]:
        return self._translator.translate(
            texts, durations, context, target_language=target_language
        )

    def synthesize(
        self, text: str, voice_id: str, *, language: str = "vi-VN"
    ) -> bytes:
        return self._synthesizer.synthesize(text, voice_id, language=language)

    @property
    def engine(self) -> str:
        return str(getattr(self._synthesizer, "engine", ""))

    @property
    def device(self) -> str:
        value = getattr(self._synthesizer, "device", "")
        return str(value() if callable(value) else value)


    # ── gộp lô ──────────────────────────────────────────────────────────
    # Vỏ bọc phải chuyển tiếp khả năng gộp lô của provider local; nếu không pipeline
    # sẽ lặng lẽ rơi về xử lý từng câu.

    @property
    def batch_size(self) -> int:
        """0 = nhà cung cấp giọng đọc này không gộp lô được."""
        value = getattr(self._synthesizer, "batch_size", 0)
        try:
            return max(0, int(str(value() if callable(value) else value or 0)))
        except (TypeError, ValueError):
            return 0

    @property
    def mps_batch_safe(self) -> bool:
        return getattr(self._synthesizer, "mps_batch_safe", True)

    def synthesize_batch(
        self, texts: list[str], voice_id: str, *, language: str = "vi-VN"
    ) -> list[bytes]:
        return self._synthesizer.synthesize_batch(
            texts, voice_id, language=language
        )

    def runtime_info(self) -> tuple[str, str, int]:
        """Expose non-sensitive TTS runtime facts for the monitor."""
        engine = str(getattr(self._synthesizer, "engine", ""))
        device = getattr(self._synthesizer, "device", "")
        device = str(device() if callable(device) else device)
        batch = getattr(self._synthesizer, "batch_size", 0)
        try:
            batch = max(0, int(str(batch() if callable(batch) else batch or 0)))
        except (TypeError, ValueError):
            batch = 0
        return engine, device, batch

    @property
    def stt_batch_size(self) -> int:
        """0 = nhà cung cấp nhận diện này không gộp lô được (Gemini, Whisper/CPU)."""
        return getattr(self._recognizer, "stt_batch_size", 0)

    def transcribe_batch(self, samples, rate: int, regions: list) -> list[tuple[str, str]]:
        return self._recognizer.transcribe_batch(samples, rate, regions)

    @property
    def stt_timed(self) -> bool:
        """True = nhận diện này trả về mốc thời gian cấp CÂU."""
        return getattr(self._recognizer, "stt_timed", False)

    def transcribe_batch_timed(self, samples, rate: int, regions: list):
        return self._recognizer.transcribe_batch_timed(samples, rate, regions)


def build_backend(config: ProviderConfig) -> CompositeBackend:
    """Import muộn từng nhà cung cấp để không kéo phụ thuộc nặng khi không dùng tới."""
    gemini = LazyGemini(config)

    if config.stt_provider not in {"whisper", "gemini"}:
        raise ValueError(f"Nhà cung cấp STT không hợp lệ: {config.stt_provider}")
    if config.tts_provider not in {"edge", "gemini", "omnivoice"}:
        raise ValueError(f"Nhà cung cấp TTS không hợp lệ: {config.tts_provider}")

    if config.stt_provider == "whisper":
        from .whisper_stt import WhisperTranscriber

        recognizer = WhisperTranscriber(
            config.whisper_model,
            config.whisper_compute_type,
            cpu_batch_size=config.whisper_cpu_batch_size,
        )
    else:
        recognizer = gemini

    if config.tts_provider == "edge":
        from .edge_speech import EdgeSynthesizer

        synthesizer = EdgeSynthesizer(config.edge_tts_attempts)
    elif config.tts_provider == "omnivoice":
        from .omnivoice_speech import OmniVoiceSynthesizer

        # Cần model Whisper để chép ref_text của clip mẫu (một lần mỗi giọng).
        synthesizer = OmniVoiceSynthesizer(
            whisper_model=config.whisper_model,
            whisper_compute_type=config.whisper_compute_type,
            num_step=config.omnivoice_num_step,
            batch_size=config.omnivoice_batch_size,
        )
    else:
        synthesizer = gemini

    return CompositeBackend(recognizer, gemini, synthesizer)
