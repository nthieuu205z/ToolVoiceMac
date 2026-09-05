"""Bước tạo giọng đọc: mỗi lượt phát ngôn đích → một đoạn PCM vừa khung thời gian gốc."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _draw_without_hole(synth, text: str) -> bytes:
    """Đọc `text`; nếu audio có lỗ hổng im lặng bất thường thì đọc lại, giữ bản sạch nhất.

    `synth` là hàm text→pcm đã chốt sẵn voice_id. Các model sinh ngẫu nhiên có thể tạo
    khoảng lặng bất thường; giữ bản có khoảng lặng giữa câu nhỏ nhất.
    """
    best = synth(text)
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
    """Tạo giọng đọc theo lô GPU hoặc song song cho provider không hỗ trợ batch.

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
    todo = [(i, seg) for i, seg in enumerate(segments) if seg.target_text.strip()]
    if not todo:
        return [], ["Không có lời thoại nào để đọc."]

    grand_total = progress_total if progress_total is not None else len(todo)
    results: dict[int, np.ndarray] = {}
    warnings: list[str] = []
    overflowed = 0
    failed = 0
    quota_hit = False

    progress("synthesize", progress_base / max(1, grand_total),
             f"Đang tạo giọng đọc ({progress_base}/{grand_total})")

    # Engine GPU đọc cả một lô trong một lượt gọi. Backend nào không gộp được thì trả 0.
    batch_size = int(getattr(backend, "batch_size", 0))
    if batch_size > 1 and getattr(backend, "mps_batch_safe", True) is False:
        batch_size = 1
    if batch_size > 0 and hasattr(backend, "synthesize_batch"):
        return _synthesize_theo_lo(
            backend, segments, todo, voice_id, language=language, max_speedup=max_speedup,
            total_duration=total_duration, next_utterance_start=next_utterance_start,
            resynthesize_bad=resynthesize_bad, fill_slowdown=fill_slowdown,
            progress=progress, should_cancel=should_cancel,
            progress_base=progress_base, grand_total=grand_total,
        )

    progress("synthesize", progress_base / max(1, grand_total),
             f"Đang đọc từng câu ({progress_base}/{grand_total})")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_render_one, backend, seg, voice_id, max_speedup,
                        _allowed_window(segments, i, total_duration,
                                        next_utterance_start), resynthesize_bad,
                        fill_slowdown, language): (i, seg)
            for i, seg in todo
        }
        for done, future in enumerate(as_completed(futures), start=1):
            if should_cancel():
                # Bỏ những lượt còn xếp hàng; lượt đang chạy tự kết thúc trong vài giây.
                pool.shutdown(wait=False, cancel_futures=True)
                raise JobCancelledError()

            index, _ = futures[future]
            try:
                samples, overflow = future.result()
            except QuotaExhaustedError:
                quota_hit = True
                failed += 1
                continue
            except Exception as exc:
                log.warning("Không đọc được lượt thoại %d: %s", index, exc)
                failed += 1
                continue

            progress("synthesize", (progress_base + done) / max(1, grand_total),
                     f"Đang đọc lượt thoại {progress_base + done}/{grand_total}")
            results[index] = samples
            # Phụ đề bám theo thời lượng đọc thật, không theo khung thời gian gốc.
            segments[index].spoken_duration = duration_of(samples)
            if overflow > _OVERFLOW_WARN_SECONDS:
                overflowed += 1

    if quota_hit:
        warnings.append(QuotaExhaustedError.user_message)
    if failed:
        warnings.append(f"{failed}/{len(todo)} lượt thoại không tạo được giọng đọc và đã bị bỏ trống.")
    if overflowed:
        warnings.append(
            f"{overflowed} lượt thoại dài hơn cả khung gốc lẫn khoảng lặng kế tiếp dù đã "
            "tăng tốc — các câu ngay sau bị lùi lại một chút, không mất chữ nào."
        )

    fitted = [(segments[i], results[i]) for i in sorted(results)]
    progress("synthesize", min(1.0, (progress_base + len(fitted)) / max(1, grand_total)),
             f"Đã tạo {progress_base + len(fitted)}/{grand_total} lượt thoại")
    return fitted, warnings


def _allowed_window(segments: list[Segment], index: int, total_duration: float | None,
                    next_utterance_start: float | None = None) -> float:
    """Lời thoại được phép kéo dài tới đâu: khung gốc + PHẦN CHO MƯỢN của khoảng lặng.

    Không cho mượn cả khoảng lặng: phần chừa lại (_BORROW_RESERVE) là chỗ để độ lệch
    dồn toa tự tan. Khoảng lặng ngắn hơn mức chừa thì không mượn được gì — câu phải
    vừa khung gốc (tăng tốc) hoặc chịu lùi. Sau lượt cuối, ranh giới lần lượt là:
    mốc cụm kế tiếp (xử lý theo cụm) → hết video → đành dùng khung gốc.
    """
    seg = segments[index]
    if index + 1 < len(segments):
        next_start = segments[index + 1].start
    elif next_utterance_start is not None:
        next_start = next_utterance_start
    else:
        next_start = total_duration
    if next_start is None:
        return seg.duration
    borrowable = max(0.0, (next_start - seg.end) - _BORROW_RESERVE)
    return seg.duration + borrowable


def _fit_one(seg: Segment, pcm: bytes, max_speedup: float,
             allowed_duration: float, fill_slowdown: float = 1.0) -> tuple[np.ndarray, float]:
    """Chống chạy hoang rồi nhét lời thoại vừa khung thời gian. Dùng chung cho cả hai đường."""
    samples = pcm_to_array(pcm)

    sane = max(
        len(seg.target_text) / _RUNAWAY_CHARS_PER_SECOND + _RUNAWAY_HEADROOM_SECONDS,
        _RUNAWAY_MIN_SECONDS,
    )
    if duration_of(samples) > sane:
        log.warning("TTS chạy hoang: %.1fs audio cho %d ký tự (%.40r) — cắt về %.1fs",
                    duration_of(samples), len(seg.target_text), seg.target_text, sane)
        samples = truncate_with_fade(samples, sane)

    return fit_to_window(samples, seg.duration, allowed_duration, max_speedup,
                         fill_slowdown=fill_slowdown)


def _render_one(
    backend: SpeechSynthesizer, seg: Segment, voice_id: str, max_speedup: float,
    allowed_duration: float, resynthesize_bad: bool = False, fill_slowdown: float = 1.0,
    language: str = "vi-VN",
) -> tuple[np.ndarray, float]:
    """Gọi TTS một lần cho câu bình thường — tenacity bên trong GeminiRunner lo việc thử
    lại khi LỖI. Chỉ khi bật `resynthesize_bad` (backend local, miễn phí) và phát hiện
    lỗ hổng im lặng bất thường mới đọc lại vì CHẤT LƯỢNG — không phải gọi lại mù.
    """
    if resynthesize_bad:
        pcm = _draw_without_hole(
            lambda text: backend.synthesize(text, voice_id, language=language),
            seg.target_text,
        )
    else:
        pcm = backend.synthesize(seg.target_text, voice_id, language=language)
    return _fit_one(seg, pcm, max_speedup, allowed_duration, fill_slowdown)


def _synthesize_single_fallback(
    backend: SpeechSynthesizer,
    seg: Segment,
    voice_id: str,
    *,
    language: str = "vi-VN",
    resynthesize_bad: bool = False,
    prefer_batch: bool = False,
) -> list[bytes]:
    """Synthesize one segment and keep failures local to that segment."""
    try:
        if prefer_batch and hasattr(backend, "synthesize_batch"):
            pcms = backend.synthesize_batch(
                [seg.target_text], voice_id, language=language
            )
            if len(pcms) != 1:
                raise RuntimeError("provider không trả đúng một audio cho lô đơn")
            return pcms
        if resynthesize_bad:
            return [_draw_without_hole(
                lambda text: backend.synthesize(
                    text, voice_id, language=language
                ),
                seg.target_text,
            )]
        return [backend.synthesize(seg.target_text, voice_id, language=language)]
    except Exception as exc:
        log.warning("Không đọc được lượt thoại: %s", exc)
        return [b""]


def _chia_lo(so_luong: int, tran: int) -> list[int]:
    """Chia `so_luong` việc thành các lô ĐỀU NHAU, không lô nào vượt `tran`.

    Chia đều thay vì cắt thẳng theo trần vì mọi hàng trong lô phải chạy tới khi hàng dài
    nhất đọc xong — một lô cuối lèo tèo 5 câu vẫn tốn gần đủ thời gian của một lô đầy.
    53 câu, trần 32: hai lô 27+26 chứ không phải 32+21.
    """
    if so_luong <= 0:
        return []
    so_lo = (so_luong + tran - 1) // tran
    deu, du = divmod(so_luong, so_lo)
    return [deu + (1 if i < du else 0) for i in range(so_lo)]


def _lo_theo_do_dai(todo: list[tuple[int, Segment]], tran: int) -> list[list[int]]:
    """Xếp câu ĐỘ DÀI GẦN NHAU vào cùng lô — GIỮ NGUYÊN số lô như chia đều.

    Vì sao: vòng sinh token tự hồi quy chạy tới khi câu DÀI NHẤT trong lô đọc xong, và một
    bước tốn gần như nhau ở batch 1 hay batch 64 (nghẽn ở phóng kernel, không phải sức tính).
    Nên chi phí một lô ≈ SỐ FRAME của câu dài nhất, gần như KHÔNG phụ thuộc số câu. Chia đều
    theo thứ tự gốc → lô nào cũng dính một câu dài → mọi lô chạy tới đỉnh. Sắp giảm dần rồi
    vẫn chia thành ĐÚNG bấy nhiêu lô: câu dài dồn vào ít lô đầu, các lô sau chỉ chạy đúng độ
    dài ngắn của chúng → tổng frame giảm hẳn (đo thật: generate 21s → ~8s).

    VRAM không tăng: lô dài nhất vẫn `tran` câu như một lô chia-đều bất kỳ (vốn cũng dính câu
    dài nhất do trộn ngẫu nhiên), lại còn ít đệm hơn vì các câu trong lô dài xấp xỉ nhau.
    Không đổi số lô nên không phát sinh lô thừa. Cùng công thức tự ép mọi loại GPU.
    """
    if not todo:
        return []
    lens = [max(1, len(seg.target_text)) for _, seg in todo]
    order = sorted(range(len(todo)), key=lambda k: lens[k], reverse=True)
    batches: list[list[int]] = []
    i = 0
    for sz in _chia_lo(len(order), max(1, tran)):
        batches.append(order[i : i + sz])
        i += sz
    return batches


def _synthesize_theo_lo(
    backend: SpeechSynthesizer, segments: list[Segment], todo: list[tuple[int, Segment]],
    voice_id: str, *, language: str, max_speedup: float, total_duration: float | None,
    next_utterance_start: float | None, resynthesize_bad: bool, fill_slowdown: float,
    progress: ProgressFn, should_cancel: CancelFn,
    progress_base: int, grand_total: int,
) -> tuple[list[tuple[Segment, np.ndarray]], list[str]]:
    """Đọc cả lô một lượt trên GPU. Giữ nguyên hai lời hứa của đường cũ:

    - Hủy giữa chừng vẫn ăn: kiểm tra trước mỗi lô (lô chạy vài giây, không phải cả video).
    - Một lượt hỏng không giết cả video: lô nào ném lỗi thì lùi về đọc từng câu trong lô đó,
      để chỉ đúng câu hỏng bị bỏ trống chứ không mất cả 32 câu cùng nó.
    """
    warnings: list[str] = []
    failed = 0
    quota_hit = False
    xong = 0

    # ── ĐỌC theo lô trên GPU (câu độ dài gần nhau → không phí frame chờ câu dài nhất) ──
    raw: dict[int, bytes] = {}
    batch_limit = max(1, int(getattr(backend, "batch_size", 1)))
    if getattr(backend, "mps_batch_safe", True) is False:
        batch_limit = 1
    batches = _lo_theo_do_dai(todo, batch_limit)
    total_batches = len(batches)
    for batch_number, grp in enumerate(batches, start=1):
        if should_cancel():
            raise JobCancelledError()

        lo = [todo[k] for k in grp]
        if len(lo) == 1:
            pcms = _synthesize_single_fallback(
                backend,
                lo[0][1],
                voice_id,
                language=language,
                resynthesize_bad=resynthesize_bad,
                prefer_batch=getattr(backend, "mps_batch_safe", True) is False,
            )
        else:
            try:
                pcms = backend.synthesize_batch(
                    [seg.target_text for _, seg in lo],
                    voice_id,
                    language=language,
                )
                if len(pcms) != len(lo):
                    raise RuntimeError(
                        f"provider trả {len(pcms)} audio cho {len(lo)} câu trong lô"
                    )
            except QuotaExhaustedError:
                quota_hit = True
                failed += len(lo)
                xong += len(lo)
                continue
            except Exception as exc:
                log.warning("Lô %d lượt thoại hỏng (%s) — đọc lại từng câu một", len(lo), exc)
                pcms = []
                for _, seg in lo:
                    pcms.extend(_synthesize_single_fallback(
                        backend,
                        seg,
                        voice_id,
                        language=language,
                        resynthesize_bad=resynthesize_bad,
                    ))

        for (index, seg), pcm in zip(lo, pcms):
            xong += 1
            if not pcm:
                failed += 1
                continue
            # Lượt xấu (lỗ hổng im lặng bất thường) hiếm — đọc lại RIÊNG câu đó, không đọc lại cả lô.
            if resynthesize_bad and longest_internal_silence(pcm_to_array(pcm)) > _MAX_INTERNAL_SILENCE:
                pcm = _draw_without_hole(
                    lambda text: backend.synthesize(
                        text, voice_id, language=language
                    ),
                    seg.target_text,
                )
            raw[index] = pcm

        progress(
            "synthesize",
            (progress_base + xong) / max(1, grand_total),
            f"Đã xử lý lô {batch_number}/{total_batches} · "
            f"lượt thoại {progress_base + xong}/{grand_total}",
        )

    if should_cancel():
        raise JobCancelledError()

    # ── ÉP KHUNG song song: mỗi _fit_one gọi ffmpeg atempo (thả GIL) — đa luồng ăn thật.
    # Đo: 267 lượt ép tuần tự 12,5s → song song còn ~2s. Kết quả gom lại đúng thứ tự index.
    results: dict[int, np.ndarray] = {}
    overflowed = 0

    def _do_fit(item: tuple[int, bytes]):
        index, pcm = item
        try:
            samples, overflow = _fit_one(
                segments[index], pcm, max_speedup,
                _allowed_window(segments, index, total_duration, next_utterance_start),
                fill_slowdown,
            )
            return index, samples, overflow
        except Exception as exc:
            log.warning("Không dựng được lượt thoại %d: %s", index, exc)
            return index, None, 0.0

    if raw:
        with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as pool:
            for index, samples, overflow in pool.map(_do_fit, list(raw.items())):
                if samples is None:
                    failed += 1
                    continue
                results[index] = samples
                # Phụ đề bám theo thời lượng đọc thật, không theo khung thời gian gốc.
                segments[index].spoken_duration = duration_of(samples)
                if overflow > _OVERFLOW_WARN_SECONDS:
                    overflowed += 1

    if quota_hit:
        warnings.append(QuotaExhaustedError.user_message)
    if failed:
        warnings.append(f"{failed}/{len(todo)} lượt thoại không tạo được giọng đọc và đã bị bỏ trống.")
    if overflowed:
        warnings.append(
            f"{overflowed} lượt thoại dài hơn cả khung gốc lẫn khoảng lặng kế tiếp dù đã "
            "tăng tốc — các câu ngay sau bị lùi lại một chút, không mất chữ nào."
        )

    fitted = [(segments[i], results[i]) for i in sorted(results)]
    progress("synthesize", min(1.0, (progress_base + len(todo)) / max(1, grand_total)),
             f"Đã tạo {progress_base + len(fitted)}/{grand_total} lượt thoại")
    return fitted, warnings
