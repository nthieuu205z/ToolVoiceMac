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
from .model_store import OMNIVOICE_MODEL_ID, OMNIVOICE_MODEL_REVISION
from .omnivoice_settings import OmniVoiceSettings

log = logging.getLogger(__name__)

if TTS_SAMPLE_RATE != 24_000:
    raise RuntimeError("OmniVoice cần TTS_SAMPLE_RATE=24000")

_MODEL = None
_MODEL_DEVICE: str | None = None
_OPTIMIZATION = "none"
_CODEC_DEVICE = "cpu"
_MODEL_OPTIMIZATION: str | None = None
_MODEL_LOADING = False
_MODEL_ERROR: str | None = None
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_MPS_BATCH_SAFE = True
_MPS_SINGLE_SAFE = True
_MPS_BATCH_LIMIT: int | None = None
_REF_TEXT_LOCKS: dict[str, threading.Lock] = {}
_REF_TEXT_LOCKS_GUARD = threading.Lock()
_REF_TEXT_CACHE: dict[tuple[str, str, int], str] = {}
_REF_TEXT_CACHE_GUARD = threading.Lock()
_CLONE_PROMPT_CACHE: dict[tuple[str, str, int, bool], object] = {}
_CLONE_PROMPT_CACHE_GUARD = threading.Lock()

# OmniVoice upstream nhận list[str]. Cỡ lô này là giới hạn pipeline; trên MPS,
# model được nạp trên CPU rồi chuyển sang MPS để tránh dispatch trực tiếp.
_BATCH_KNEE = 8
_SPLIT_MODES = ("split-cfg", "split-cfg-rms", "split-cfg-rms-gqa", "split-cfg-rms-gqa-rope")
_RMS_MODES = _SPLIT_MODES[1:]
_GQA_MODES = _SPLIT_MODES[2:]


def _accel_device() -> str | None:
    from .model_store import accel_device

    return accel_device()


def configure_optimization(mode: str, *, codec_device: str | None = None) -> None:
    """Select once at application startup; never swap a live singleton."""
    global _OPTIMIZATION, _CODEC_DEVICE
    if not isinstance(mode, str) or mode not in ("none", *_SPLIT_MODES):
        raise ValueError("Unsupported OmniVoice optimization")
    if codec_device is not None and codec_device not in ("cpu", "mps"):
        raise ValueError("Unsupported OmniVoice codec device")
    if not _MODEL_LOCK.acquire(blocking=False):
        raise RuntimeError("Restart required to change OmniVoice optimization during loading")
    try:
        requested_codec = _CODEC_DEVICE if codec_device is None else codec_device
        if _MODEL is not None and (mode != _OPTIMIZATION or requested_codec != _CODEC_DEVICE):
            raise RuntimeError("Restart required to change OmniVoice runtime configuration")
        _OPTIMIZATION = mode
        _CODEC_DEVICE = requested_codec
    finally:
        _MODEL_LOCK.release()


def _configured_batch_ceiling() -> int:
    ceiling = _MPS_BATCH_LIMIT or _BATCH_KNEE
    return min(ceiling, 2) if _OPTIMIZATION in _SPLIT_MODES else ceiling


def runtime_status() -> dict:
    """Content-free snapshot; never loads a model or waits for model loading."""
    loaded = _MODEL is not None
    state = "ready" if loaded else "loading" if _MODEL_LOADING else "error" if _MODEL_ERROR else "not_loaded"
    batch = _configured_batch_ceiling()
    if not _MPS_SINGLE_SAFE or (_MODEL_DEVICE is not None and _MODEL_DEVICE == "cpu"):
        batch = 0
    elif not _MPS_BATCH_SAFE:
        batch = 1
    return {
        "requested_optimization": _OPTIMIZATION,
        "loaded_optimization": _MODEL_OPTIMIZATION if loaded else None,
        "state": state,
        "device": _MODEL_DEVICE if loaded else None,
        "requested_codec_device": _CODEC_DEVICE,
        "codec_device": _tokenizer_device(_MODEL) if loaded else None,
        "batch_ceiling": batch,
        "experimental": _OPTIMIZATION in _SPLIT_MODES,
        "error": _MODEL_ERROR,
    }


def _get_model():
    """Nạp một model singleton trên accelerator đã được probe."""
    global _MODEL, _MODEL_DEVICE, _MODEL_OPTIMIZATION, _MODEL_LOADING, _MODEL_ERROR
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        _MODEL_LOADING = True
        _MODEL_ERROR = None
        try:
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
            if _CODEC_DEVICE == "mps" and accelerator != "mps":
                raise SpeechServiceError("Codec MPS yêu cầu GPU Apple MPS khả dụng.")
            model_class = OmniVoice
            if _OPTIMIZATION in _SPLIT_MODES:
                if accelerator != "mps":
                    message = "Split-CFG thử nghiệm yêu cầu MPS; khởi động lại với chế độ none."
                    raise SpeechServiceError(message, user_message=message)
                from .omnivoice_split_cfg import get_split_cfg_class

                try:
                    model_class = get_split_cfg_class()
                except RuntimeError:
                    message = "Source OmniVoice không khớp bản split-CFG đã kiểm chứng; khởi động lại với chế độ none."
                    raise SpeechServiceError(message, user_message=message) from None
            # Keep CPU-staged loading; codec placement is independently selectable.
            device = "mps" if accelerator == "mps" else "cpu"
            dtype = torch.float16 if accelerator == "mps" else torch.float32
            load_device = "cpu" if accelerator == "mps" else device
            log.info("Đang nạp OmniVoice trên %s (load=%s, optimization=%s)",
                     device, load_device, _OPTIMIZATION)
            # OmniVoice resolves repo IDs to main before forwarding revision kwargs.
            # Resolve the frozen snapshot ourselves so its model and codec stay paired.
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                snapshot = snapshot_download(
                    OMNIVOICE_MODEL_ID, revision=OMNIVOICE_MODEL_REVISION, local_files_only=True,
                )
            except LocalEntryNotFoundError:
                snapshot = snapshot_download(OMNIVOICE_MODEL_ID, revision=OMNIVOICE_MODEL_REVISION)
            model = model_class.from_pretrained(
                snapshot,
                device_map=load_device,
                dtype=dtype,
            )
            if accelerator == "mps":
                tokenizer = getattr(model, "audio_tokenizer", None)
                model = model.to(device)
                if _CODEC_DEVICE == "cpu":
                    _keep_mps_tokenizer_on_cpu(tokenizer)
                elif tokenizer is not None:
                    tokenizer.to("mps")
            if _OPTIMIZATION in _RMS_MODES:
                from .omnivoice_mps_norm import enable_native_rms_norm

                count = enable_native_rms_norm(model)
                log.info("Native MPS RMSNorm enabled on %d modules", count)
            if _OPTIMIZATION in _GQA_MODES:
                from .omnivoice_mps_attention import enable_native_mps_gqa

                count = enable_native_mps_gqa(model)
                log.info("Native MPS grouped attention enabled on %d layers", count)
            if _OPTIMIZATION == "split-cfg-rms-gqa-rope":
                from .omnivoice_mps_rope import enable_native_mps_rope

                count = enable_native_mps_rope(model)
                log.info("Native MPS rotary enabled on %d layers", count)
            _MODEL_DEVICE = device
            _MODEL_OPTIMIZATION = _OPTIMIZATION
            _MODEL = model
            log.info(
                "OmniVoice ready trên %s (load=%s, batch=%d, tokenizer=%s, optimization=%s)",
                device,
                "cpu-staged" if accelerator == "mps" else "direct",
                _auto_batch_size(0),
                _tokenizer_device(_MODEL),
                _MODEL_OPTIMIZATION,
            )
        except Exception:
            _MODEL = None
            _MODEL_DEVICE = None
            _MODEL_OPTIMIZATION = None
            _MODEL_ERROR = "load_failed"
            raise
        finally:
            _MODEL_LOADING = False
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
    global _MODEL, _MODEL_DEVICE, _MODEL_OPTIMIZATION, _MODEL_LOADING, _MODEL_ERROR, _OPTIMIZATION, _CODEC_DEVICE
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_DEVICE = None
        _MODEL_OPTIMIZATION = None
        _MODEL_LOADING = False
        _MODEL_ERROR = None
        _OPTIMIZATION = "none"
        _CODEC_DEVICE = "cpu"


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
    if _accel_device() != "mps" or not _MPS_SINGLE_SAFE:
        return 0
    if not _MPS_BATCH_SAFE:
        return 1
    if override > 0:
        return min(override, _configured_batch_ceiling())
    return _configured_batch_ceiling()


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
        generation_settings: OmniVoiceSettings | None = None,
    ):
        self._model = model
        self._transcriber = transcriber
        self._whisper_model = whisper_model
        self._whisper_compute_type = whisper_compute_type
        self._generation_settings = generation_settings
        self._num_step = generation_settings.num_step if generation_settings else num_step
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
                if self._generation_settings is not None:
                    timing = self._generation_settings
                    kwargs["speed"] = 1.0 if timing.duration is not None else timing.speed
                    if timing.duration is not None:
                        if count != 1:
                            raise SpeechServiceError("Duration chỉ hỗ trợ một đoạn văn bản.")
                        kwargs["duration"] = timing.duration
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
            if self._generation_settings is not None:
                raise SpeechServiceError("OmniVoice không hỗ trợ cấu hình nâng cao đã chọn.")
            # Test doubles and older package builds may only accept legacy kwargs.
            return None
        if self._generation_settings is not None:
            return OmniVoiceGenerationConfig(**self._generation_settings.generation_kwargs())
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
        preprocess = self._generation_settings.preprocess_prompt if self._generation_settings else True
        key = (voice_id, str(sample), version, preprocess)
        with _CLONE_PROMPT_CACHE_GUARD:
            cached = _CLONE_PROMPT_CACHE.get(key)
        if cached is not None:
            return cached
        creator = getattr(model, "create_voice_clone_prompt", None)
        if not callable(creator):
            return None
        prompt = creator(str(sample), ref_text=ref_text, preprocess_prompt=preprocess)
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
