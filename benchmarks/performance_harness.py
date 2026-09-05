from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypeVar

try:
    import psutil
except ImportError:  # pragma: no cover - optional outside the project venv
    psutil = None


T = TypeVar("T")

# Never persist values that could contain credentials or user/content data.
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "token",
    "credential",
    "connection_string",
    "prompt",
    "transcript",
    "text",
    "audio",
    "video",
    "path",
)


@dataclass(frozen=True)
class ResourceSnapshot:
    rss_mb: float = 0.0
    cpu_percent: float | None = None
    system_memory_percent: float | None = None
    mps_allocated_mb: float | None = None
    mps_driver_mb: float | None = None


@dataclass(frozen=True)
class Measurement:
    label: str
    elapsed_seconds: float
    item_count: int
    items_per_second: float
    warm: bool
    resources_before: ResourceSnapshot
    resources_after: ResourceSnapshot
    peak_rss_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _safe_value(value: Any) -> Any:
    """Keep JSON-friendly scalar metadata and recursively remove unsafe fields."""
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if not _is_sensitive_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(type(value).__name__)


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return benchmark metadata with secret/content-bearing fields removed."""
    safe = _safe_value(metadata or {})
    return safe if isinstance(safe, dict) else {}


@lru_cache(maxsize=1)
def _torch_module():
    try:
        import torch
    except Exception:  # pragma: no cover - depends on optional accelerator stack
        return None
    return torch


def _mps_memory() -> tuple[float | None, float | None]:
    torch = _torch_module()
    if torch is None:
        return None, None
    try:
        mps = torch.mps
        if not mps.is_available():
            return None, None
        allocated = getattr(mps, "current_allocated_memory", None)
        driver = getattr(mps, "driver_allocated_memory", None)
        allocated_mb = float(allocated()) / 1024**2 if callable(allocated) else None  # type: ignore[call-overload]
        driver_mb = float(driver()) / 1024**2 if callable(driver) else None  # type: ignore[call-overload]
        return allocated_mb, driver_mb
    except Exception:  # pragma: no cover - API differs by torch version
        return None, None


def _resource_snapshot() -> ResourceSnapshot:
    rss_mb = 0.0
    cpu_percent: float | None = None
    system_memory_percent: float | None = None
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            rss_mb = process.memory_info().rss / 1024**2
            cpu_percent = float(process.cpu_percent(interval=None))
            system_memory_percent = float(psutil.virtual_memory().percent)
        except (OSError, psutil.Error):
            pass
    allocated_mb, driver_mb = _mps_memory()
    return ResourceSnapshot(
        rss_mb=round(max(0.0, rss_mb), 3),
        cpu_percent=None if cpu_percent is None else round(max(0.0, cpu_percent), 3),
        system_memory_percent=(
            None if system_memory_percent is None else round(system_memory_percent, 3)
        ),
        mps_allocated_mb=None if allocated_mb is None else round(max(0.0, allocated_mb), 3),
        mps_driver_mb=None if driver_mb is None else round(max(0.0, driver_mb), 3),
    )


def measure(
    label: str,
    operation: Callable[[], T],
    *,
    item_count: int = 1,
    warm: bool = False,
    metadata: dict[str, Any] | None = None,
) -> tuple[Measurement, T | None]:
    """Measure one operation and return a non-secret result record plus its result.

    Exceptions are converted to ``error_type`` only; exception messages are deliberately
    discarded because provider errors can contain request data or credentials.
    """
    before = _resource_snapshot()
    peak_rss_mb = before.rss_mb
    stop_sampling = threading.Event()

    def sample_peak() -> None:
        nonlocal peak_rss_mb
        while not stop_sampling.wait(0.01):
            peak_rss_mb = max(peak_rss_mb, _resource_snapshot().rss_mb)

    sampler = threading.Thread(target=sample_peak, name="benchmark-resource-sampler", daemon=True)
    sampler.start()
    started = time.perf_counter()
    result: T | None = None
    error_type: str | None = None
    try:
        result = operation()
    except Exception as exc:  # measurement must survive provider/model failures
        error_type = type(exc).__name__
    finally:
        elapsed = max(0.0, time.perf_counter() - started)
        stop_sampling.set()
        sampler.join(timeout=0.25)

    after = _resource_snapshot()
    peak_rss_mb = max(peak_rss_mb, after.rss_mb)
    count = max(0, int(item_count))
    throughput = count / elapsed if count and elapsed > 0 else 0.0
    measurement = Measurement(
        label=str(label),
        elapsed_seconds=round(elapsed, 6),
        item_count=count,
        items_per_second=round(throughput, 6),
        warm=bool(warm),
        resources_before=before,
        resources_after=after,
        peak_rss_mb=round(peak_rss_mb, 3),
        metadata=sanitize_metadata(metadata),
        error_type=error_type,
        error_message=None,
    )
    return measurement, result


def write_json_report(
    path: str | Path,
    measurements: list[Measurement],
    *,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    """Write a versioned, secret-safe benchmark report atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_metadata": sanitize_metadata(run_metadata),
        "measurements": [
            {
                **asdict(measurement),
                "metadata": sanitize_metadata(measurement.metadata),
            }
            for measurement in measurements
        ],
    }
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
