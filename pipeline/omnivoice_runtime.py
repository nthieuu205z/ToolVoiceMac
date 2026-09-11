"""Session-scoped, standalone OmniVoice inference; heavy imports are lazy."""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from importlib import metadata
import math
from pathlib import Path
import platform
import random
import re
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from pipeline.languages import LanguageSpec, require_language
from pipeline.omnivoice_settings import OmniVoiceSettings

DEFAULT_CHECKPOINT = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
Scalar = str | int | float | bool | None


def validate_optimization(optimization: str) -> None:
    if not isinstance(optimization, str) or optimization not in ("none", "split-cfg"):
        raise ValueError("Unsupported optimization mode")


def validate_reference(reference_audio: Path, reference_text: str) -> None:
    if not reference_audio.is_file():
        raise FileNotFoundError("Reference audio must be a local file")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise ValueError("Explicit reference transcript required")


def validate_request(texts: list[str], language: str, seed: int,
                     settings: OmniVoiceSettings) -> LanguageSpec:
    """Shared pre-load bounds for the runtime and CLI; never rewrite text."""
    if not isinstance(settings, OmniVoiceSettings):
        raise ValueError("Invalid settings")
    if not isinstance(texts, list) or not 1 <= len(texts) <= 64:
        raise ValueError("Expected 1 to 64 text items")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Expected nonempty strings")
    if sum(map(len, texts)) > 50000:
        raise ValueError("Request exceeds character limit")
    if settings.duration is not None and (len(texts) != 1 or len(texts[0]) > 1000):
        raise ValueError("Manual duration requires one short text")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    try:
        return require_language(language)
    except ValueError:
        raise ValueError("Unsupported language") from None


def sanitize_provenance(values: Mapping[str, Any]) -> dict[str, Scalar]:
    """Allow only known facts and numeric generation scalars, never free text."""
    enums = {"runtime": {"omnivoice"}, "device": {"mps", "mps:0"}, "codec_device": {"cpu"},
             "load_device": {"cpu"}, "dtype": {"float16"},
             "language": {"vi-VN", "en-US"}, "checkpoint_source": {"local", "hub-cache"},
             "optimization": {"none", "split-cfg"}}
    versions = {f"{name}_version" for name in (
        "python", "omnivoice", "torch", "torchaudio", "transformers", "numpy", "huggingface_hub"
    )}
    result = {}
    for key, value in values.items():
        if key in enums and isinstance(value, str) and value in enums[key]:
            result[key] = value
        elif key in versions and (value is None or isinstance(value, str) and
                                  re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,79}", value)):
            result[key] = value
        elif key in {"source_sha256", "runtime_source_sha256", "optimization_source_sha256", "checkpoint_revision"}:
            if value is None or isinstance(value, str) and re.fullmatch(
                    r"[0-9a-f]{40,64}" if key == "checkpoint_revision" else r"[0-9a-f]{64}", value):
                result[key] = value
        elif (key in {"seed", "speed", "duration", "normalize_text", "sample_rate", "offline"}
              or re.fullmatch(r"generation_[a-z][a-z_0-9]*", key)):
            if value is None or type(value) in (bool, int) or type(value) is float and math.isfinite(value):
                result[key] = value
    return result


@dataclass(frozen=True)
class UpstreamSession:
    model: Any
    config_factory: Callable[..., Any]
    synchronize: Callable[[], None]
    seed: Callable[[int], None]
    provenance: Mapping[str, Scalar]


@dataclass(frozen=True)
class RuntimeResult:
    """Ordered mono signed little-endian 16-bit PCM, not WAV containers."""

    pcm: tuple[bytes, ...]
    sample_rate: int
    timings: Mapping[str, float]
    provenance: Mapping[str, Scalar]


def load_upstream(checkpoint: str, *, optimization: str = "none") -> UpstreamSession:
    """Load only local/cached weights; require a bundled local audio_tokenizer.

    Upstream's resolver ignores offline kwargs. Resolve the repo ID ourselves,
    then pass an actual directory; reject a missing codec rather than letting
    upstream fall back to downloading its separate codec repository. No global
    environment, model cache, upstream method, or installed file is modified.
    split-cfg is experimental and restricted to the supported upstream source.
    """
    validate_optimization(optimization)
    path = Path(checkpoint)
    from_cache = not path.is_dir()
    if from_cache:
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", checkpoint):
            raise FileNotFoundError("Local checkpoint directory required")
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(checkpoint, local_files_only=True, token=False))
    path = path.resolve()
    if not path.is_dir() or not (path / "audio_tokenizer").is_dir():
        raise FileNotFoundError("Checkpoint requires a local audio_tokenizer directory")

    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this runtime")
    from omnivoice.models import omnivoice as upstream

    model_class = upstream.OmniVoice
    optimization_source_sha256 = None
    if optimization == "split-cfg":
        from pipeline import omnivoice_split_cfg

        model_class = omnivoice_split_cfg.get_split_cfg_class(upstream)
        optimization_source_sha256 = hashlib.sha256(Path(omnivoice_split_cfg.__file__).read_bytes()).hexdigest()
    model = model_class.from_pretrained(
        str(path), device_map="cpu", dtype=torch.float16,
        load_asr=False, local_files_only=True,
    )
    model.to("mps")
    model.audio_tokenizer.to("cpu")
    model.eval()

    def seed(value: int) -> None:
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    revision = path.name if path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40,64}", path.name) else None
    provenance = {
        "runtime": "omnivoice", "device": str(model.device),
        "dtype": str(model.dtype).removeprefix("torch."),
        "load_device": "cpu", "codec_device": "cpu", "offline": True,
        "checkpoint_source": "hub-cache" if from_cache else "local",
        "checkpoint_revision": revision, "python_version": platform.python_version(),
        "source_sha256": hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest(),
        "runtime_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "optimization": optimization, "optimization_source_sha256": optimization_source_sha256,
    }
    for name in ("omnivoice", "torch", "torchaudio", "transformers", "numpy", "huggingface_hub"):
        try:
            provenance[f"{name}_version"] = metadata.version(name)
        except metadata.PackageNotFoundError:
            provenance[f"{name}_version"] = None
    defaults = upstream.OmniVoiceGenerationConfig()
    provenance.update({f"generation_{field.name}": getattr(defaults, field.name) for field in fields(defaults)})
    return UpstreamSession(model, upstream.OmniVoiceGenerationConfig, torch.mps.synchronize,
                           seed, sanitize_provenance(provenance))


class OmniVoiceRuntime:
    def __init__(
        self, reference_audio: Path, reference_text: str, *,
        settings: OmniVoiceSettings | None = None,
        checkpoint: str = DEFAULT_CHECKPOINT,
        optimization: str = "none",
        loader: Callable[..., UpstreamSession] | None = None,
    ):
        validate_optimization(optimization)
        self.optimization = optimization
        self.reference_audio = Path(reference_audio)
        self.reference_text = reference_text
        self.settings = settings if settings is not None else OmniVoiceSettings()
        validate_reference(self.reference_audio, reference_text)
        if not isinstance(self.settings, OmniVoiceSettings):
            raise ValueError("Invalid settings")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("Explicit checkpoint required")
        self.checkpoint = checkpoint
        self._loader = loader if loader is not None else load_upstream
        self._session = None
        self._prompt = None
        self._lock = Lock()

    @property
    def session(self) -> UpstreamSession | None:
        """Read-only diagnostic seam; callers must not run the model concurrently."""
        return self._session

    @property
    def prompt(self) -> Any:
        return self._prompt

    def generate(self, texts: list[str], *, language: str, seed: int) -> RuntimeResult:
        language_spec = validate_request(texts, language, seed, self.settings)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Runtime is already generating")
        try:
            validate_reference(self.reference_audio, self.reference_text)
            return self._generate(list(texts), language_spec, seed)
        finally:
            self._lock.release()

    def _generate(self, texts: list[str], language_spec: LanguageSpec, seed: int) -> RuntimeResult:
        total_start = perf_counter()
        timings = {"load_seconds": 0.0, "prompt_seconds": 0.0}
        if self._session is None:
            start = perf_counter()
            # Preserve the existing one-argument fake-loader seam in default mode.
            self._session = (self._loader(self.checkpoint) if self.optimization == "none"
                             else self._loader(self.checkpoint, optimization=self.optimization))
            self._session.synchronize()
            timings["load_seconds"] = perf_counter() - start
        session = self._session
        if session.model.sampling_rate != SAMPLE_RATE:
            raise ValueError("Unexpected upstream sample rate")
        session.seed(seed)
        if self._prompt is None:
            session.synchronize()
            start = perf_counter()
            self._prompt = session.model.create_voice_clone_prompt(
                ref_audio=str(self.reference_audio), ref_text=self.reference_text,
                preprocess_prompt=self.settings.preprocess_prompt,
            )
            session.synchronize()
            timings["prompt_seconds"] = perf_counter() - start
        config = session.config_factory(**self.settings.generation_kwargs())
        session.synchronize()
        start = perf_counter()
        audio = session.model.generate(
            text=list(texts), language=language_spec.omnivoice_name,
            voice_clone_prompt=self._prompt, generation_config=config,
            speed=self.settings.speed, duration=self.settings.duration,
            normalize_text=False,
        )
        session.synchronize()
        timings["generate_seconds"] = perf_counter() - start
        start = perf_counter()
        if not isinstance(audio, (list, tuple)) or len(audio) != len(texts):
            raise ValueError("Unexpected audio cardinality")
        if session.model.sampling_rate != SAMPLE_RATE:
            raise ValueError("Unexpected upstream sample rate")
        for item in audio:
            if (not isinstance(item, np.ndarray) or item.ndim != 1 or item.size == 0
                    or item.dtype.kind not in "fiu" or not np.isfinite(item).all()):
                raise ValueError("Invalid upstream audio")
        pcm = tuple((np.clip(np.asarray(item, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes() for item in audio)
        timings["pcm_seconds"] = perf_counter() - start
        timings["total_seconds"] = perf_counter() - total_start
        provenance = dict(session.provenance)
        provenance.update(optimization=self.optimization,
                          optimization_source_sha256=(session.provenance.get("optimization_source_sha256")
                                                      if self.optimization == "split-cfg" else None))
        provenance.update({f"generation_{field.name}": getattr(config, field.name) for field in fields(config)})
        provenance.update(seed=seed, language=language_spec.code, speed=self.settings.speed, duration=self.settings.duration, normalize_text=False)
        return RuntimeResult(pcm, SAMPLE_RATE, timings, sanitize_provenance(provenance))
