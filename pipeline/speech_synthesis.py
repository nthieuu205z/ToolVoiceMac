"""Provider-facing raw text synthesis with indexed partial results."""

from __future__ import annotations

import logging

from .errors import JobCancelledError, QuotaExhaustedError, SpeechServiceError
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
    """Partition consecutive input indexes into balanced provider batches."""
    order = list(range(len(texts)))
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


def synthesize_texts(
    synthesizer: SpeechSynthesizer,
    texts: list[str],
    voice_id: str,
    *,
    language: str,
    progress: ProgressFn = noop_progress,
    should_cancel: CancelFn = never_cancel,
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

    for batch_number, indexes in enumerate(groups, start=1):
        if should_cancel():
            raise JobCancelledError()

        batch = [texts[index] for index in indexes]
        values: list[bytes | None] | None = None
        use_batch = len(indexes) > 1 or force_single_batch
        if batch_size > 1 or force_single_batch:
            try:
                batch_values = synthesizer.synthesize_batch(
                    batch, voice_id, language=language
                )
                if len(batch_values) != len(batch):
                    raise _BatchFallbackRequired()
                values = list(batch_values)
            except QuotaExhaustedError:
                quota_hit = True
                failed += len(indexes)
                values = [None] * len(indexes)
            except (_BatchFallbackRequired, SpeechServiceError, RuntimeError) as exc:
                log.warning(
                    "Lô giọng đọc %d/%d không dùng được (%s)",
                    batch_number,
                    len(groups),
                    type(exc).__name__,
                )
                if force_single_batch:
                    failed += len(indexes)
                    values = [None] * len(indexes)

        if values is None:
            values = []
            for index in indexes:
                if should_cancel():
                    raise JobCancelledError()
                try:
                    values.append(
                        synthesizer.synthesize(
                            texts[index], voice_id, language=language
                        )
                    )
                except QuotaExhaustedError:
                    quota_hit = True
                    failed += 1
                    values.append(None)
                except (SpeechServiceError, RuntimeError) as exc:
                    log.warning(
                        "Không tạo được giọng cho mục %d/%d (%s)",
                        index + 1,
                        len(texts),
                        type(exc).__name__,
                    )
                    failed += 1
                    values.append(None)

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

    warnings: list[str] = []
    if quota_hit:
        warnings.append(QuotaExhaustedError.user_message)
    if failed:
        warnings.append(
            f"{failed}/{len(texts)} lượt thoại không tạo được giọng đọc và đã bị bỏ trống."
        )
    return outputs, warnings
