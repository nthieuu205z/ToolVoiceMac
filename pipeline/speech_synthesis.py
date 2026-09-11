"""Provider-facing raw text synthesis with indexed partial results."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .errors import JobCancelledError, QuotaExhaustedError
from .models import CancelFn, ProgressFn, SpeechSynthesizer, never_cancel, noop_progress

log = logging.getLogger(__name__)


class _BatchFallbackRequired(Exception):
    """The provider batch result cannot be used and must be retried item by item."""


def _chia_lo(so_luong: int, tran: int) -> list[int]:
    """Split work into balanced batches, none larger than ``tran``."""
    if so_luong <= 0:
        return []
    so_lo = (so_luong + tran - 1) // tran
    deu, du = divmod(so_luong, so_lo)
    return [deu + (1 if index < du else 0) for index in range(so_lo)]


def _batch_groups(texts: list[str], limit: int) -> list[list[int]]:
    """Build balanced batches, stably grouping similar-length provider inputs."""
    order = list(range(len(texts)))
    if limit > 1:
        order.sort(key=lambda index: max(1, len(texts[index])), reverse=True)
    groups: list[list[int]] = []
    offset = 0
    for size in _chia_lo(len(order), max(1, limit)):
        groups.append(order[offset : offset + size])
        offset += size
    return groups


def _provider_batching(synthesizer: SpeechSynthesizer) -> tuple[int, bool]:
    """Return the safe batch limit and whether size-one calls must use the batch API."""
    try:
        batch_size = int(getattr(synthesizer, "batch_size", 0) or 0)
    except (TypeError, ValueError):
        batch_size = 0
    if batch_size <= 0 or not hasattr(synthesizer, "synthesize_batch"):
        return 1, False
    if getattr(synthesizer, "mps_batch_safe", True) is False:
        return 1, True
    return batch_size, False


def _synthesize_one(
    synthesizer: SpeechSynthesizer,
    text: str,
    index: int,
    total: int,
    voice_id: str,
    language: str,
    should_cancel: CancelFn,
) -> tuple[bytes | None, bool]:
    """Run one provider call, returning its PCM and whether quota was exhausted."""
    if should_cancel():
        raise JobCancelledError()
    try:
        return synthesizer.synthesize(text, voice_id, language=language), False
    except JobCancelledError:
        raise
    except QuotaExhaustedError:
        return None, True
    except Exception as exc:
        log.warning(
            "Không tạo được giọng cho mục %d/%d (%s)",
            index + 1,
            total,
            type(exc).__name__,
        )
        return None, False


def _failure_warnings(failed: int, total: int, quota_hit: bool) -> list[str]:
    warnings: list[str] = []
    if quota_hit:
        warnings.append(QuotaExhaustedError.user_message)
    if failed:
        warnings.append(
            f"{failed}/{total} lượt thoại không tạo được giọng đọc và đã bị bỏ trống."
        )
    return warnings


def synthesize_texts(
    synthesizer: SpeechSynthesizer,
    texts: list[str],
    voice_id: str,
    *,
    language: str,
    progress: ProgressFn = noop_progress,
    should_cancel: CancelFn = never_cancel,
    workers: int = 1,
) -> tuple[list[bytes | None], list[str]]:
    """Synthesize raw PCM while preserving input indexes and partial success.

    Provider batch failures fall back to single-item calls. A failure remains local
    to its input index, and warning text never contains submitted content.
    """
    outputs: list[bytes | None] = [None] * len(texts)
    failed = 0
    quota_hit = False
    completed = 0
    batch_size, force_single_batch = _provider_batching(synthesizer)
    groups = _batch_groups(texts, batch_size)

    if batch_size == 1 and not force_single_batch and workers > 1:
        if should_cancel():
            raise JobCancelledError()
        if texts:
            progress("synthesize", 0.0, f"Đang tạo giọng 0/{len(texts)}")
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {
                pool.submit(
                    _synthesize_one,
                    synthesizer,
                    text,
                    index,
                    len(texts),
                    voice_id,
                    language,
                    should_cancel,
                ): index
                for index, text in enumerate(texts)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                value, hit_quota = future.result()
                outputs[index] = value
                failed += value is None
                quota_hit = quota_hit or hit_quota
                progress(
                    "synthesize",
                    completed / max(1, len(texts)),
                    f"Đang tạo giọng {completed}/{len(texts)}",
                )
        return outputs, _failure_warnings(failed, len(texts), quota_hit)

    for batch_number, indexes in enumerate(groups, start=1):
        if should_cancel():
            raise JobCancelledError()

        batch = [texts[index] for index in indexes]
        progress(
            "synthesize",
            completed / max(1, len(texts)),
            f"Đang tạo lô {batch_number}/{len(groups)} · {len(indexes)} đoạn"
            if batch_size > 1 or force_single_batch
            else f"Đang tạo giọng {completed + 1}/{len(texts)}",
        )
        values: list[bytes | None] | None = None
        use_batch = len(indexes) > 1 or force_single_batch
        if use_batch:
            try:
                batch_values = synthesizer.synthesize_batch(
                    batch, voice_id, language=language
                )
                if len(batch_values) != len(batch):
                    raise _BatchFallbackRequired()
                values = list(batch_values)
            except JobCancelledError:
                raise
            except QuotaExhaustedError:
                quota_hit = True
                failed += len(indexes)
                values = [None] * len(indexes)
            except Exception as exc:
                log.warning(
                    "Lô giọng đọc %d/%d không dùng được (%s)",
                    batch_number,
                    len(groups),
                    type(exc).__name__,
                )

        if values is None:
            values = []
            for index in indexes:
                value, hit_quota = _synthesize_one(
                    synthesizer,
                    texts[index],
                    index,
                    len(texts),
                    voice_id,
                    language,
                    should_cancel,
                )
                values.append(value)
                failed += value is None
                quota_hit = quota_hit or hit_quota

        for index, value in zip(indexes, values):
            outputs[index] = value

        completed += len(indexes)
        if batch_size > 1 or force_single_batch:
            message = (
                f"Đã xử lý lô {batch_number}/{len(groups)} · "
                f"lượt thoại {completed}/{len(texts)}"
            )
        else:
            message = f"Đang tạo giọng {completed}/{len(texts)}"
        progress(
            "synthesize",
            completed / max(1, len(texts)),
            message,
        )

    return outputs, _failure_warnings(failed, len(texts), quota_hit)
