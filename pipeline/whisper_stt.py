"""Nhận diện giọng nói bằng Whisper chạy ngay trên máy — miễn phí, không hạn mức.

Vì sao không dùng mốc thời gian của Whisper: ffmpeg đã cho ranh giới chính xác tuyệt đối
(xem pipeline/segmentation.py). Whisper chỉ được giao đúng một việc là chép chữ cho từng
vùng đã cắt sẵn, nên không cần tin vào mốc thời gian của nó.

Whisper dùng CPU/int8 trên Mac vì faster-whisper không hỗ trợ MPS.
Gộp lô CPU mặc định 8 vùng để giữ tốc độ và mốc câu.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

# Nạp model tốn vài giây và ngốn RAM — dùng chung cho mọi job trong tiến trình.
_models: dict[tuple[str, str], object] = {}
_load_lock = threading.Lock()
# Nhiều job chạy song song vẫn trỏ vào CÙNG một model object; faster-whisper không
# hứa an toàn khi gọi đồng thời, nên khóa suy luận phải là toàn cục chứ không phải
# per-instance (mỗi job tạo một WhisperTranscriber riêng, khóa riêng thì vô nghĩa).
_infer_lock = threading.Lock()


# CPU batched inference is materially faster on Apple Silicon even though
# faster-whisper itself cannot place the model on MPS. Keep a conservative
# default/ceiling; the batch is measured in feature chunks, not model copies.
DEFAULT_CPU_BATCH_SIZE = 8
MAX_CPU_BATCH_SIZE = 8


def _clamp_cpu_batch_size(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_CPU_BATCH_SIZE
    return max(0, min(parsed, MAX_CPU_BATCH_SIZE))


def _thiet_bi() -> tuple[str, str]:
    """faster-whisper dùng CPU trên Mac; MPS dành cho OmniVoice."""
    return "cpu", "int8"


def _get_model(name: str, compute_type: str):
    device, _ = _thiet_bi()
    key = (name, compute_type)
    with _load_lock:
        if key not in _models:
            from faster_whisper import WhisperModel  # nạp muộn: import nặng

            log.info("Đang nạp Whisper '%s' trên %s (lần đầu sẽ tải model về)", name, device.upper())
            _models[key] = WhisperModel(name, device=device, compute_type=compute_type)
        return _models[key]


@lru_cache(maxsize=1)
def _get_pipeline(model):
    from faster_whisper import BatchedInferencePipeline

    return BatchedInferencePipeline(model=model)


def prewarm(model: str = "small", compute_type: str = "int8") -> None:
    """Nạp model Whisper trong luồng nền lúc server khởi động — job đầu khỏi chờ tải model.

    Chỉ nạp khi model đã nằm trên đĩa — không tự ý tải 484 MB lúc boot.
    """
    from .model_store import is_ready, whisper_spec

    try:
        if not is_ready(whisper_spec(model)):
            log.info("Whisper chưa tải về — bỏ qua nạp sẵn, chờ người dùng bấm tải")
            return
    except Exception as exc:
        log.warning("Không kiểm tra được model Whisper: %s", exc)
        return

    def load() -> None:
        try:
            _get_pipeline(_get_model(model, compute_type))   # nạp cả BatchedInferencePipeline
            log.info("Whisper sẵn sàng (%s)", _thiet_bi()[0].upper())
        except Exception as exc:
            log.warning("Không nạp sẵn được Whisper (job đầu sẽ tự nạp): %s", exc)

    threading.Thread(target=load, name="whisper-prewarm", daemon=True).start()


_SENTENCE_END = (".", "!", "?", "…")


def sentences_from_words(
    words: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Gom các TỪ (start, end, text) của Whisper thành CÂU, cắt sau từ kết bằng .!?…

    Trả về [(start_câu, end_câu, văn_bản)]. Mốc câu = mốc từ đầu → mốc từ cuối của câu, là
    mốc căn chỉnh thật của Whisper. Câu cuối chưa có dấu kết vẫn được trả (mảnh dang dở) —
    merge_sentence_fragments sẽ ghép nó với mảnh mở của vùng kế. Nhờ tách theo câu, mỗi câu
    thành một lượt đọc riêng, đặt đúng thời điểm câu tiếng Anh.
    """
    sentences: list[tuple[float, float, str]] = []
    parts: list[str] = []
    start: float | None = None
    end = 0.0
    for w_start, w_end, w_text in words:
        token = w_text.strip()
        if not token:
            continue
        if start is None:
            start = w_start
        parts.append(token)
        end = w_end
        if token.endswith(_SENTENCE_END):
            sentences.append((start, end, " ".join(parts)))
            parts, start = [], None
    if parts and start is not None:
        sentences.append((start, end, " ".join(parts)))
    return sentences


def _vung_chua(regions: list, boundaries: list[tuple[float, float]], start: float,
               end: float) -> int | None:
    """Vùng nào chồng lấn nhiều nhất với [start, end]. None nếu không chồng vào đâu.

    Dùng chồng lấn chứ không dùng điểm giữa: một mảnh trả về có thể lệch nhẹ khỏi mốc ta
    đưa vào, và gán nhầm vùng nghĩa là lời thoại nhảy sang khung thời gian khác — hỏng
    đồng bộ hình–tiếng theo cách rất khó thấy.
    """
    tot_nhat, tot_diem = None, 0.0
    for i, (s, e) in enumerate(boundaries):
        chong = min(end, e) - max(start, s)
        if chong > tot_diem:
            tot_nhat, tot_diem = i, chong
    return tot_nhat


class WhisperTranscriber:
    """Chép lời cho từng đoạn audio ngắn. Bám giao thức `transcribe_clip` của pipeline.

    Gộp lô CPU qua `transcribe_batch` — pipeline/stt.py sẽ ưu tiên dùng nó.
    """

    def __init__(self, model: str = "small", compute_type: str = "int8",
                 *, cpu_batch_size: int = DEFAULT_CPU_BATCH_SIZE):
        self._model_name = model
        self._compute_type = compute_type
        self._cpu_batch_size = _clamp_cpu_batch_size(cpu_batch_size)

    @property
    def stt_batch_size(self) -> int:
        return self._cpu_batch_size

    @property
    def stt_timed(self) -> bool:
        """BatchedInferencePipeline giữ được mốc câu trên CPU."""
        return self.stt_batch_size > 0

    def _chep(self, wav_path: Path) -> tuple[str, str]:
        model = _get_model(self._model_name, self._compute_type)
        with _infer_lock:
            # Giữ VAD dù ffmpeg đã cắt đúng vùng có tiếng: đo thật trên 50 clip, tắt VAD chỉ
            # nhanh hơn 2/55 giây và chép ra CHÍNH XÁC cùng số chữ — không đáng để bỏ đi lớp
            # chắn cuối cùng chống Whisper bịa lời trên đoạn im lặng.
            segments, info = model.transcribe(str(wav_path), vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments)
        return info.language or "", text.strip()

    def transcribe_clip(self, wav_path: Path) -> tuple[str, str]:
        return self._chep(wav_path)

    def transcribe_batch(self, samples, rate: int, regions: list) -> list[tuple[str, str]]:
        """Chép lời cho NHIỀU vùng trong một lượt gộp lô CPU. Trả về [(ngôn ngữ, lời)] đúng thứ tự.

        Mốc thời gian vẫn do ffmpeg quyết định — ta truyền thẳng chúng vào `clip_timestamps`
        nên Whisper không được phép tự cắt lại. Nguyên tắc của dự án giữ nguyên: ffmpeg lo
        THỜI GIAN, Whisper lo NỘI DUNG (xem pipeline/segmentation.py).
        """
        import numpy as np

        model = _get_model(self._model_name, self._compute_type)
        pipe = _get_pipeline(model)

        # faster-whisper nhận float32 mono 16 kHz; source.wav đã đúng thế (STT_SAMPLE_RATE).
        audio = np.asarray(samples, dtype="float32") / 32768.0

        moc = [{"start": r.start, "end": r.end} for r in regions]
        bien = [(r.start, r.end) for r in regions]

        with _infer_lock:
            segments, info = pipe.transcribe(audio, clip_timestamps=moc,
                                             batch_size=self.stt_batch_size)
            chu: list[list[str]] = [[] for _ in regions]
            for seg in segments:
                i = _vung_chua(regions, bien, seg.start, seg.end)
                if i is not None:
                    chu[i].append(seg.text.strip())

        lang = info.language or ""
        return [(lang, " ".join(c).strip()) for c in chu]

    def transcribe_batch_timed(
        self, samples, rate: int, regions: list,
    ) -> tuple[str, list[tuple[float, float, str]]]:
        """Như transcribe_batch nhưng GIỮ mốc thời gian cấp CÂU của Whisper.

        Với clip_timestamps, Whisper trả về MỘT segment cho mỗi vùng (không tự tách câu).
        Nên ta bật `word_timestamps=True` lấy mốc từng TỪ rồi tự tách câu theo dấu chấm câu
        (xem sentences_from_words) — cho mốc câu THẬT bên trong vùng. Mỗi câu → một Segment
        đặt đúng thời điểm, để TTS đọc từng câu và bám hình. Vùng ffmpeg
        vẫn lo THỜI GIAN THÔ (clip_timestamps ràng buộc Whisper chỉ chép chỗ có tiếng).
        """
        import numpy as np

        model = _get_model(self._model_name, self._compute_type)
        pipe = _get_pipeline(model)
        audio = np.asarray(samples, dtype="float32") / 32768.0

        moc = [{"start": r.start, "end": r.end} for r in regions]
        bien = [(r.start, r.end) for r in regions]

        with _infer_lock:
            segments, info = pipe.transcribe(audio, clip_timestamps=moc,
                                             batch_size=self.stt_batch_size,
                                             word_timestamps=True)
            pieces: list[tuple[float, float, str]] = []
            for seg in segments:
                words = [(float(w.start), float(w.end), w.word)
                         for w in (getattr(seg, "words", None) or [])]
                # Không có mốc từ (hiếm) → giữ cả segment làm một câu, đừng bỏ lời thoại.
                cau = sentences_from_words(words) if words else (
                    [(float(seg.start), float(seg.end), seg.text.strip())]
                    if seg.text.strip() else [])
                for s, e, t in cau:
                    if t.strip() and _vung_chua(regions, bien, s, e) is not None:
                        pieces.append((s, e, t.strip()))

        return info.language or "", pieces
