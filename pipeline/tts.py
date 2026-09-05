"""Bước tạo giọng đọc: mỗi lượt phát ngôn đích → một đoạn PCM vừa khung thời gian gốc."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .audio import (
    duration_of,
    fit_to_window,
    longest_internal_silence,
    pcm_to_array,
    truncate_with_fade,
)
from .errors import JobCancelledError, QuotaExhaustedError
from .models import CancelFn, ProgressFn, Segment, SpeechSynthesizer, never_cancel, noop_progress
from .speech_runtime import speech_activity
from .speech_synthesis import synthesize_texts

log = logging.getLogger(__name__)

# Chỉ cảnh báo khi lời thoại tràn khung đủ nhiều để tai nghe ra.
_OVERFLOW_WARN_SECONDS = 0.3
# Phần khoảng lặng KHÔNG cho mượn. Đây là chỗ hồi của cả chuỗi: khi một câu tràn cứng
# đẩy các câu sau lùi lại, mỗi khoảng lặng chỉ hấp thụ được phần không bị mượn mất.
# Đo thật: cho mượn cạn kiệt → độ lệch tan 0,03s/bước, 5 câu tràn kéo 133 câu lùi theo.
_BORROW_RESERVE = 0.4

# Chống TTS "chạy hoang": model tự hồi quy thỉnh thoảng bịa thêm lời khi đầu vào quá
# ngắn — đo thật: chữ "Và" (2 ký tự) sinh ra 7,1 giây giọng nói, kéo 15 câu sau lệch
# 3–6 giây. Đọc chậm nhất còn hợp lý là ~8 ký tự/giây,
# vượt trần đó là audio bịa — cắt bỏ, từ thật luôn nằm ở phần đầu.
_RUNAWAY_CHARS_PER_SECOND = 8.0
_RUNAWAY_HEADROOM_SECONDS = 1.0
# Sàn cố định: model tốn một khoảng thời gian nền cho MỌI lượt đọc bất kể dài ngắn (ngữ điệu, hơi thở,
# đuôi câu), gần như không theo số ký tự. Ngưỡng tuyến tính len/8+1 quá chặt với câu
# ngắn — "Thứ hai," (8 ký tự) chỉ được 2,0s trong khi giọng thật ~3s → cắt cụt tiếng thật
# (nghe "ngắt đột ngột"). Đo raw: đọc ngắn bình thường ≤3,5s, chạy hoang ≥6s. Sàn 4s tách
# sạch hai loại — câu ngắn thật sống, còn "Và"→8s vẫn bị cắt.
_RUNAWAY_MIN_SECONDS = 4.0

# Lỗ hổng im lặng dài hơn mức này ở GIỮA một lượt đọc là bất thường, không phải nghỉ lấy hơi
# giữa câu. Đọc lại tối đa _RESYNTH_ATTEMPTS lần, giữ bản có lỗ hổng nhỏ nhất. Chỉ bật cho
# backend local miễn phí;
# Gemini TTS tính tiền theo lượt nên để tắt.
_MAX_INTERNAL_SILENCE = 1.3
_RESYNTH_ATTEMPTS = 2


def _draw_without_hole(synth, text: str, *, initial: bytes | None = None) -> bytes:
    """Đọc `text`; nếu audio có lỗ hổng im lặng bất thường thì đọc lại, giữ bản sạch nhất.

    `synth` là hàm text→pcm đã chốt sẵn voice_id. Các model sinh ngẫu nhiên có thể tạo
    khoảng lặng bất thường; giữ bản có khoảng lặng giữa câu nhỏ nhất.
    """
    best = initial if initial is not None else synth(text)
    best_gap = longest_internal_silence(pcm_to_array(best)) if best else 0.0
    for _ in range(_RESYNTH_ATTEMPTS):
        if best_gap <= _MAX_INTERNAL_SILENCE:
            break
        cand = synth(text)
        gap = longest_internal_silence(pcm_to_array(cand)) if cand else float("inf")
        if gap < best_gap:
            best, best_gap = cand, gap
    return best


def synthesize_segments(
    backend: SpeechSynthesizer,
    segments: list[Segment],
    voice_id: str,
    *,
    language: str = "vi-VN",
    workers: int = 4,
    max_speedup: float = 1.5,
    total_duration: float | None = None,
    next_utterance_start: float | None = None,
    resynthesize_bad: bool = False,
    fill_slowdown: float = 1.0,
    progress: ProgressFn = noop_progress,
    should_cancel: CancelFn = never_cancel,
    progress_base: int = 0,
    progress_total: int | None = None,
) -> tuple[list[tuple[Segment, np.ndarray]], list[str]]:
    """Tạo giọng đọc thô rồi ép riêng từng kết quả vào khung thời gian video.

    Trả về ([(lượt phát ngôn, mẫu âm thanh)], cảnh báo).

    Lời đích dài hơn khung gốc thì trước hết cho MƯỢN khoảng lặng phía sau (tới tận
    mốc câu kế tiếp) — đọc tốc độ tự nhiên tràn vào chỗ im lặng nghe thật hơn hẳn
    giọng bị tăng tốc. Chỉ khi hết cả chỗ mượn mới tăng tốc, tối đa `max_speedup`.

    Một lượt hỏng không giết cả video — nó chỉ để lại khoảng lặng và một cảnh báo.
    Nhưng hết hạn mức theo ngày thì mọi lượt sau đó cũng hỏng, nên được báo riêng.

    `next_utterance_start`: mốc lượt phát ngôn ĐẦU TIÊN của cụm kế tiếp, khi video
    được xử lý theo cụm gối đầu — để lượt cuối cụm không mượn lố sang cụm sau.
    `progress_base`/`progress_total`: quy số đếm về toàn video thay vì cụm hiện tại.
    """
    todo = [(index, segment) for index, segment in enumerate(segments) if segment.target_text.strip()]
    if not todo:
        return [], ["Không có lời thoại nào để đọc."]

    grand_total = progress_total if progress_total is not None else len(todo)
    progress(
        "synthesize",
        progress_base / max(1, grand_total),
        f"Đang tạo giọng đọc ({progress_base}/{grand_total})",
    )

    segments_to_read = [segment for _, segment in todo]
    provider = str(getattr(backend, "engine", "") or "unknown")
    with speech_activity.production(provider):
        raw, warnings = synthesize_texts(
            backend,
            [segment.target_text for segment in segments_to_read],
            voice_id,
            language=language,
            progress=lambda stage, fraction, message: progress(
                stage,
                (progress_base + fraction * len(todo)) / max(1, grand_total),
                message,
            ),
            should_cancel=should_cancel,
        )

        # Provider local miễn phí được phép đọc lại đúng những mẫu có lỗ hổng xấu.
        if resynthesize_bad:
            for compact_index, pcm in enumerate(raw):
                if pcm is None:
                    continue
                if longest_internal_silence(pcm_to_array(pcm)) <= _MAX_INTERNAL_SILENCE:
                    continue
                segment = segments_to_read[compact_index]
                try:
                    raw[compact_index] = _draw_without_hole(
                        lambda text: backend.synthesize(
                            text, voice_id, language=language
                        ),
                        segment.target_text,
                        initial=pcm,
                    )
                except QuotaExhaustedError:
                    raw[compact_index] = None
                    if QuotaExhaustedError.user_message not in warnings:
                        warnings.insert(0, QuotaExhaustedError.user_message)
                except Exception as exc:
                    log.warning(
                        "Không đọc lại được lượt thoại %d (%s)",
                        todo[compact_index][0],
                        type(exc).__name__,
                    )
                    raw[compact_index] = None

    if should_cancel():
        raise JobCancelledError()

    results: dict[int, np.ndarray] = {}
    overflowed = 0
    fit_failed = 0

    def _do_fit(item: tuple[int, bytes | None]):
        compact_index, pcm = item
        index, segment = todo[compact_index]
        if pcm is None:
            return index, None, 0.0
        try:
            samples, overflow = _fit_one(
                segment,
                pcm,
                max_speedup,
                _allowed_window(segments, index, total_duration, next_utterance_start),
                fill_slowdown,
            )
            return index, samples, overflow
        except Exception as exc:
            log.warning("Không dựng được lượt thoại %d: %s", index, type(exc).__name__)
            return index, None, 0.0

    if raw:
        # Mỗi lần ép khung gọi ffmpeg atempo (thả GIL), nên đa luồng rút ngắn đáng kể.
        fit_workers = min(8, max(1, workers), (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=fit_workers) as pool:
            fitted_raw = pool.map(_do_fit, list(enumerate(raw)))
            for compact_index, (index, samples, overflow) in enumerate(fitted_raw):
                if samples is None:
                    if raw[compact_index] is not None:
                        fit_failed += 1
                    continue
                results[index] = samples
                # Phụ đề bám theo thời lượng đọc thật, không theo khung thời gian gốc.
                segments[index].spoken_duration = duration_of(samples)
                if overflow > _OVERFLOW_WARN_SECONDS:
                    overflowed += 1

    synthesis_failed = sum(pcm is None for pcm in raw)
    if fit_failed:
        prior_warning = (
            f"{synthesis_failed}/{len(todo)} lượt thoại không tạo được giọng đọc "
            "và đã bị bỏ trống."
        )
        if prior_warning in warnings:
            warnings.remove(prior_warning)
        failed = synthesis_failed + fit_failed
        warnings.append(
            f"{failed}/{len(todo)} lượt thoại không tạo được giọng đọc và đã bị bỏ trống."
        )
    elif synthesis_failed:
        expected_warning = (
            f"{synthesis_failed}/{len(todo)} lượt thoại không tạo được giọng đọc "
            "và đã bị bỏ trống."
        )
        if expected_warning not in warnings:
            warnings.append(expected_warning)

    if overflowed:
        warnings.append(
            f"{overflowed} lượt thoại dài hơn cả khung gốc lẫn khoảng lặng kế tiếp dù đã "
            "tăng tốc — các câu ngay sau bị lùi lại một chút, không mất chữ nào."
        )

    fitted = [(segments[index], results[index]) for index in sorted(results)]
    progress(
        "synthesize",
        min(1.0, (progress_base + len(todo)) / max(1, grand_total)),
        f"Đã tạo {progress_base + len(fitted)}/{grand_total} lượt thoại",
    )
    return fitted, warnings


def _allowed_window(
    segments: list[Segment],
    index: int,
    total_duration: float | None,
    next_utterance_start: float | None = None,
) -> float:
    """Lời thoại được phép kéo dài tới đâu: khung gốc + PHẦN CHO MƯỢN của khoảng lặng.

    Không cho mượn cả khoảng lặng: phần chừa lại (_BORROW_RESERVE) là chỗ để độ lệch
    dồn toa tự tan. Khoảng lặng ngắn hơn mức chừa thì không mượn được gì — câu phải
    vừa khung gốc (tăng tốc) hoặc chịu lùi. Sau lượt cuối, ranh giới lần lượt là:
    mốc cụm kế tiếp (xử lý theo cụm) → hết video → đành dùng khung gốc.
    """
    segment = segments[index]
    if index + 1 < len(segments):
        next_start = segments[index + 1].start
    elif next_utterance_start is not None:
        next_start = next_utterance_start
    else:
        next_start = total_duration
    if next_start is None:
        return segment.duration
    borrowable = max(0.0, (next_start - segment.end) - _BORROW_RESERVE)
    return segment.duration + borrowable


def _fit_one(
    segment: Segment,
    pcm: bytes,
    max_speedup: float,
    allowed_duration: float,
    fill_slowdown: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Chống chạy hoang rồi nhét lời thoại vừa khung thời gian video."""
    samples = pcm_to_array(pcm)
    sane = max(
        len(segment.target_text) / _RUNAWAY_CHARS_PER_SECOND
        + _RUNAWAY_HEADROOM_SECONDS,
        _RUNAWAY_MIN_SECONDS,
    )
    if duration_of(samples) > sane:
        log.warning(
            "TTS chạy hoang: %.1fs audio cho %d ký tự (%.40r) — cắt về %.1fs",
            duration_of(samples),
            len(segment.target_text),
            segment.target_text,
            sane,
        )
        samples = truncate_with_fade(samples, sane)

    return fit_to_window(
        samples,
        segment.duration,
        allowed_duration,
        max_speedup,
        fill_slowdown=fill_slowdown,
    )
