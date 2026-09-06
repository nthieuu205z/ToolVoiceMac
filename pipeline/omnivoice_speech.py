"""OmniVoice zero-shot TTS cho các giọng nhân bản của người dùng."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import custom_voices
from .errors import SpeechServiceError
from .languages import require_language
from .models import TTS_SAMPLE_RATE

log = logging.getLogger(__name__)

if TTS_SAMPLE_RATE != 24_000:
    raise RuntimeError("OmniVoice cần TTS_SAMPLE_RATE=24000")

_MODEL_ID = "k2-fsa/OmniVoice"
_MODEL = None
_MODEL_DEVICE: str | None = None
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_MPS_BATCH_SAFE = True
_MPS_SINGLE_SAFE = True
_MPS_BATCH_LIMIT: int | None = None
_REF_TEXT_LOCKS: dict[str, threading.Lock] = {}
_REF_TEXT_LOCKS_GUARD = threading.Lock()
_REF_TEXT_CACHE: dict[tuple[str, str, int], str] = {}
_REF_TEXT_CACHE_GUARD = threading.Lock()
_CLONE_PROMPT_CACHE: dict[tuple[str, str, int], object] = {}
_CLONE_PROMPT_CACHE_GUARD = threading.Lock()

# OmniVoice upstream nhận list[str]. Cỡ lô này là giới hạn pipeline; trên MPS,
# model được nạp trên CPU rồi chuyển sang MPS để tránh dispatch trực tiếp.
_BATCH_KNEE = 8
_VRAM_PER_ITEM_GB = 0.18


def _accel_device() -> str | None:
    from .model_store import accel_device

    return accel_device()


def _get_model():
    """Nạp một model singleton trên accelerator đã được probe."""
    global _MODEL, _MODEL_DEVICE
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        try:
            from omnivoice import OmniVoice
            import torch
        except Exception as exc:
            raise SpeechServiceError(
                "Chưa cài đủ OmniVoice và PyTorch",
                user_message=(
                    "Chưa cài đủ OmniVoice và PyTorch. Hãy chạy "
                    "`uv pip install -e '.[omnivoice]'` theo README."
                ),
            ) from exc

        accelerator = _accel_device()
        # MPS is supported by this model only through the staged loader below;
        # direct accelerate dispatch has crashed natively in observed runs.
        device = "cpu"
        if accelerator == "cuda":
            device = "cuda:0"
        elif accelerator == "mps":
            device = "mps"
        dtype = torch.float16 if accelerator in {"cuda", "mps"} else torch.float32
        load_device = "cpu" if accelerator == "mps" else device
        log.info(
            "Đang nạp OmniVoice trên %s (load=%s; lần đầu sẽ tải model ~3,3 GB)",
            device,
            "cpu-staged" if accelerator == "mps" else "direct",
        )
        try:
            _MODEL = OmniVoice.from_pretrained(
                _MODEL_ID,
                device_map=load_device,
                dtype=dtype,
            )
            if accelerator == "mps":
                tokenizer = getattr(_MODEL, "audio_tokenizer", None)
                _MODEL = _MODEL.to(device)  # type: ignore[union-attr]
                # Upstream deliberately keeps this codec on CPU for MPS.
                _keep_mps_tokenizer_on_cpu(tokenizer)
            log.info(
                "OmniVoice ready trên %s (load=%s, batch=%d, tokenizer=%s)",
                device,
                "cpu-staged" if accelerator == "mps" else "direct",
                _auto_batch_size(0),
                _tokenizer_device(_MODEL),
            )
        except Exception:
            _MODEL = None
            _MODEL_DEVICE = None
            raise
        _MODEL_DEVICE = device
        return _MODEL


def _tokenizer_device(model) -> str | None:
    """Return the codec device for diagnostics without assuming it exists."""
    tokenizer = getattr(model, "audio_tokenizer", None)
    return None if tokenizer is None else str(getattr(tokenizer, "device", "unknown"))


def _keep_mps_tokenizer_on_cpu(tokenizer) -> None:
    """Restore the upstream MPS placement after a recursive model move."""
    if tokenizer is None:
        return
    move_tokenizer = getattr(tokenizer, "to", None)
    if move_tokenizer is not None:
        move_tokenizer("cpu")


def _reset_model_for_tests() -> None:
    """Reset the singleton seam for isolated tests."""
    global _MODEL, _MODEL_DEVICE
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_DEVICE = None


def _reset_prompt_cache_for_tests() -> None:
    with _CLONE_PROMPT_CACHE_GUARD:
        _CLONE_PROMPT_CACHE.clear()


def _drop_clone_prompt_cache(voice_id: str) -> None:
    with _CLONE_PROMPT_CACHE_GUARD:
        for key in [key for key in _CLONE_PROMPT_CACHE if key[0] == voice_id]:
            _CLONE_PROMPT_CACHE.pop(key, None)


def model_device() -> str | None:
    """Thiết bị thực tế của singleton sau khi nạp; hữu ích cho log/diagnostics."""
    if _MODEL is not None:
        return str(getattr(_MODEL, "device", _MODEL_DEVICE))
    return _MODEL_DEVICE


def mps_batch_safe() -> bool:
    """Current MPS batch gate; false means caller must use single-item inference."""
    return _MPS_BATCH_SAFE


def mark_mps_batch_unsafe() -> None:
    """Disable future multi-item MPS calls after a recoverable model error."""
    global _MPS_BATCH_SAFE
    _MPS_BATCH_SAFE = False


def prewarm() -> None:
    """Nạp OmniVoice nền nếu model đã có sẵn trong cache; không tự tải model."""
    from .model_store import is_ready, omnivoice_spec

    try:
        if not is_ready(omnivoice_spec()):
            log.info("OmniVoice chưa tải về — bỏ qua nạp sẵn")
            return
    except Exception as exc:
        log.warning("Không kiểm tra được model OmniVoice: %s", exc)
        return

    def load() -> None:
        try:
            _get_model()
            log.info("OmniVoice sẵn sàng (nạp sẵn lúc boot)")
        except Exception as exc:
            log.warning("Không nạp sẵn được OmniVoice (job đầu sẽ tự nạp): %s", exc)

    threading.Thread(target=load, name="omnivoice-prewarm", daemon=True).start()


def forget_clone(voice_id: str) -> None:
    """Xóa ref_text đã cache trên đĩa và trong RAM của một giọng."""
    try:
        sidecar = custom_voices.safe_sidecar_path(voice_id)
        sidecar.unlink(missing_ok=True)
    except OSError:
        pass
    with _REF_TEXT_CACHE_GUARD:
        for key in [key for key in _REF_TEXT_CACHE if key[0] == voice_id]:
            _REF_TEXT_CACHE.pop(key, None)
    _drop_clone_prompt_cache(voice_id)


def _ref_text_lock(sample: Path) -> threading.Lock:
    key = str(sample)
    with _REF_TEXT_LOCKS_GUARD:
        return _REF_TEXT_LOCKS.setdefault(key, threading.Lock())


def _write_ref_text_atomic(sidecar: Path, text: str) -> None:
    temporary = sidecar.with_name(f".{sidecar.name}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)


def _to_pcm16(audio) -> bytes:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return b""
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _auto_batch_size(override: int) -> int:
    accelerator = _accel_device()
    if accelerator is None:
        return 0
    if accelerator == "mps" and not _MPS_SINGLE_SAFE:
        return 0
    if accelerator == "mps" and not _MPS_BATCH_SAFE:
        return 1
    if override > 0:
        if accelerator == "mps":
            return min(override, _MPS_BATCH_LIMIT or _BATCH_KNEE)
        return override
    if accelerator == "mps":
        return _MPS_BATCH_LIMIT or _BATCH_KNEE
    if accelerator != "cuda":
        return 0
    try:
        import torch

        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    except Exception:
        return _BATCH_KNEE
    return max(1, min(_BATCH_KNEE, int(free_gb * 0.75 / _VRAM_PER_ITEM_GB)))


class OmniVoiceSynthesizer:
    """Adapter trả PCM 16-bit mono và gom nhiều câu vào một lần generate."""

    engine = "omnivoice"

    def __init__(
        self,
        *,
        model=None,
        transcriber: Callable[[Path], str] | None = None,
        whisper_model: str = "small",
        whisper_compute_type: str = "int8",
        num_step: int = 32,
        batch_size: int = 0,
    ):
        self._model = model
        self._transcriber = transcriber
        self._whisper_model = whisper_model
        self._whisper_compute_type = whisper_compute_type
        self._num_step = num_step
        self._batch_override = max(0, batch_size)
        self._ref_cache: dict[str, str] = {}

    @property
    def batch_size(self) -> int:
        return _auto_batch_size(self._batch_override)

    @property
    def mps_batch_safe(self) -> bool:
        return mps_batch_safe()

    def synthesize(
        self, text: str, voice_id: str, *, language: str = "vi-VN"
    ) -> bytes:
        return self._generate([text], voice_id, language=language)[0]

    def synthesize_batch(
        self, texts: list[str], voice_id: str, *, language: str = "vi-VN"
    ) -> list[bytes]:
        if not texts:
            return []
        limit = self.batch_size
        if limit > 0 and len(texts) > limit:
            raise SpeechServiceError(
                f"OmniVoice nhận {len(texts)} câu, vượt cỡ lô tối đa {limit}",
                user_message="Lô giọng đọc quá lớn; pipeline cần chia nhỏ lô.",
            )
        return self._generate(list(texts), voice_id, language=language)

    def _generate(
        self, texts: list[str], voice_id: str, *, language: str
    ) -> list[bytes]:
        upstream_language = require_language(language).omnivoice_name
        if not custom_voices.is_custom(voice_id):
            raise SpeechServiceError(
                f"OmniVoice chỉ đọc bằng giọng nhân bản: {voice_id}",
                user_message="OmniVoice chỉ dùng được giọng nhân bản.",
            )
        sample = custom_voices.sample_path(voice_id)
        if not sample.is_file():
            raise SpeechServiceError(
                f"OmniVoice thiếu clip mẫu cho giọng {voice_id}: {sample}",
                user_message="Giọng nhân bản này thiếu file mẫu. Tạo lại giọng rồi thử lại.",
            )

        ref_text = self._ref_text(voice_id, sample)
        if not ref_text:
            raise SpeechServiceError(
                f"Không lấy được lời của clip mẫu cho giọng {voice_id}",
                user_message="Không nhận diện được lời trong clip mẫu. Hãy tạo lại giọng bằng audio rõ tiếng.",
            )

        model = self._model or _get_model()
        clone_prompt = self._clone_prompt(voice_id, sample, ref_text, model)
        count = len(texts)
        try:
            # Upstream generate có state/cache nội bộ; batch là một lệnh GPU duy nhất.
            with _INFER_LOCK:
                generation_config = self._generation_config()
                kwargs: dict[str, Any] = {
                    "language": [upstream_language] * count,
                }
                if generation_config is None:
                    kwargs.update(
                        num_step=self._num_step,
                        postprocess_output=True,
                    )
                else:
                    kwargs["generation_config"] = generation_config
                if clone_prompt is None:
                    kwargs.update(
                        ref_audio=[str(sample)] * count,
                        ref_text=[ref_text] * count,
                    )
                else:
                    kwargs["voice_clone_prompt"] = [clone_prompt] * count
                outputs = model.generate(text=list(texts), **kwargs)
        except SpeechServiceError:
            raise
        except Exception as exc:
            if _accel_device() == "mps" and count > 1:
                mark_mps_batch_unsafe()
            raise SpeechServiceError(
                f"OmniVoice không đọc được lượt thoại (giọng {voice_id}): {exc}",
                user_message="OmniVoice gặp lỗi khi tạo giọng đọc. Kiểm tra model và clip mẫu.",
            ) from exc

        arrays = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        pcms = [_to_pcm16(audio) for audio in arrays]
        if len(pcms) != count or any(not pcm for pcm in pcms):
            raise SpeechServiceError(
                f"OmniVoice trả về audio rỗng/thiếu ({len(pcms)}/{count}) cho giọng {voice_id}"
            )
        return pcms

    def _generation_config(self):
        """Build the upstream config once per call without legacy kwargs."""
        try:
            from omnivoice.models.omnivoice import OmniVoiceGenerationConfig
        except (ImportError, AttributeError):
            # Test doubles and older package builds may only accept legacy kwargs.
            return None
        return OmniVoiceGenerationConfig(
            num_step=self._num_step,
            postprocess_output=True,
        )

    def _clone_prompt(self, voice_id: str, sample: Path, ref_text: str, model):
        """Build and cache upstream VoiceClonePrompt for repeated batch calls."""
        try:
            version = sample.stat().st_mtime_ns
        except OSError:
            version = 0
        key = (voice_id, str(sample), version)
        with _CLONE_PROMPT_CACHE_GUARD:
            cached = _CLONE_PROMPT_CACHE.get(key)
        if cached is not None:
            return cached
        creator = getattr(model, "create_voice_clone_prompt", None)
        if not callable(creator):
            return None
        prompt = creator(str(sample), ref_text=ref_text, preprocess_prompt=True)
        with _CLONE_PROMPT_CACHE_GUARD:
            _CLONE_PROMPT_CACHE[key] = prompt
        return prompt

    def _ref_text(self, voice_id: str, sample: Path) -> str:
        cached = self._ref_cache.get(voice_id)
        if cached is not None:
            return cached

        with _ref_text_lock(sample):
            cached = self._ref_cache.get(voice_id)
            if cached is not None:
                return cached
            try:
                version = sample.stat().st_mtime_ns
            except OSError:
                version = 0
            cache_key = (voice_id, str(sample), version)
            with _REF_TEXT_CACHE_GUARD:
                text = _REF_TEXT_CACHE.get(cache_key)
            if text is not None:
                self._ref_cache[voice_id] = text
                return text

            sidecar = custom_voices.safe_sidecar_path(voice_id)
            text = ""
            if sidecar.is_file():
                try:
                    text = sidecar.read_text(encoding="utf-8").strip()
                except OSError as exc:
                    log.warning("Không đọc được ref_text cạnh %s: %s", sample, exc)
            if not text:
                text = self._transcribe(sample).strip()
                if text:
                    try:
                        _write_ref_text_atomic(sidecar, text)
                    except OSError as exc:
                        log.warning("Không ghi được ref_text cạnh %s: %s", sample, exc)

            self._ref_cache[voice_id] = text
            if text:
                with _REF_TEXT_CACHE_GUARD:
                    _REF_TEXT_CACHE[cache_key] = text
            return text

    def _transcribe(self, sample: Path) -> str:
        if self._transcriber is not None:
            return self._transcriber(sample)
        if not hasattr(self, "_default_tr") or self._default_tr is None:
            from .whisper_stt import WhisperTranscriber

            self._default_tr = WhisperTranscriber(self._whisper_model, self._whisper_compute_type)
        _lang, text = self._default_tr.transcribe_clip(sample)
        return text

    def device(self) -> str | None:
        """Trả accelerator được chọn cho instance hoặc model singleton."""
        if self._model is not None:
            return str(getattr(self._model, "device", "unknown"))
        return model_device() or _accel_device()
