"""CompositeBackend phải chuyển tiếp khả năng gộp lô của nhà cung cấp giọng đọc.

Bài học đắt: gộp lô của provider local đã chạy đúng, test đơn vị xanh, benchmark nhanh gấp nhiều lần
— nhưng trong app THẬT nó không hề chạy. Vì pipeline nói chuyện với CompositeBackend, mà
vỏ bọc đó chỉ phơi ra synthesize(). tts.py hỏi `backend.batch_size`, không thấy, và lặng lẽ
lùi về đọc từng câu một. Không lỗi, không cảnh báo, chỉ là chậm y như cũ.

Benchmark phải đi qua đúng CompositeBackend như ứng dụng thật.
"""

from __future__ import annotations

import numpy as np

from pipeline.backends import CompositeBackend
from pipeline.models import TTS_SAMPLE_RATE, Segment
from pipeline.tts import synthesize_segments


class SynthCoLo:
    """Nhà cung cấp giọng đọc có gộp lô trên GPU."""

    batch_size = 32

    def __init__(self):
        self.lo_da_goi: list[list[str]] = []
        self.le_da_goi: list[str] = []

    def synthesize_batch(self, texts, voice_id, *, language="vi-VN"):
        self.lo_da_goi.append(list(texts))
        return [np.zeros(TTS_SAMPLE_RATE, dtype="<i2").tobytes() for _ in texts]

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        self.le_da_goi.append(text)
        return np.zeros(TTS_SAMPLE_RATE, dtype="<i2").tobytes()


class SynthKhongLo:
    """Provider không gộp lô được (edge-tts hoặc Gemini)."""

    def __init__(self):
        self.le_da_goi: list[str] = []

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        self.le_da_goi.append(text)
        return np.zeros(TTS_SAMPLE_RATE, dtype="<i2").tobytes()


class RecordingSynthesizer:
    batch_size = 8

    def __init__(self):
        self.single_calls = []
        self.batch_calls = []

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        self.single_calls.append((text, voice_id, language))
        return b"\x00\x00"

    def synthesize_batch(self, texts, voice_id, *, language="vi-VN"):
        self.batch_calls.append((list(texts), voice_id, language))
        return [b"\x00\x00" for _ in texts]


class RecordingTranslator:
    def __init__(self):
        self.calls = []

    def translate(self, texts, durations, context="", *, target_language="vi-VN"):
        self.calls.append((list(texts), list(durations), context, target_language))
        return list(texts)


def _backend(synth):
    return CompositeBackend(recognizer=None, translator=None, synthesizer=synth)


def _segs():
    return [Segment(start=i * 3.0, end=i * 3.0 + 2.0, text="x", text_vi=c)
            for i, c in enumerate(["a", "b", "c"])]


def test_vo_boc_phoi_ra_batch_size_cua_nha_cung_cap():
    assert _backend(SynthCoLo()).batch_size == 32


def test_vo_boc_phoi_ra_engine_va_device_cua_nha_cung_cap():
    class RuntimeSynth(SynthCoLo):
        engine = "omnivoice"
        device = "mps:0"

    backend = _backend(RuntimeSynth())

    assert backend.engine == "omnivoice"
    assert backend.device == "mps:0"


def test_vo_boc_phoi_ra_mps_batch_safety():
    class UnsafeSynth(SynthCoLo):
        mps_batch_safe = False

    assert _backend(UnsafeSynth()).mps_batch_safe is False


def test_composite_forwards_language_to_single_and_batch_synthesis():
    synth = RecordingSynthesizer()
    backend = CompositeBackend(recognizer=None, translator=None, synthesizer=synth)

    backend.synthesize("hello", "voice", language="en-US")
    backend.synthesize_batch(["one", "two"], "voice", language="en-US")

    assert synth.single_calls == [("hello", "voice", "en-US")]
    assert synth.batch_calls == [(["one", "two"], "voice", "en-US")]


def test_composite_forwards_target_language_to_translation():
    translator = RecordingTranslator()
    backend = CompositeBackend(recognizer=None, translator=translator, synthesizer=None)

    result = backend.translate(["hello"], [1.0], "earlier", target_language="en-US")

    assert result == ["hello"]
    assert translator.calls == [(["hello"], [1.0], "earlier", "en-US")]


def test_vo_boc_bao_0_khi_nha_cung_cap_khong_gop_lo_duoc():
    """edge-tts/Gemini không có batch_size — vỏ bọc phải trả 0, không được nổ AttributeError."""
    assert _backend(SynthKhongLo()).batch_size == 0


def test_pipeline_that_su_di_duong_gop_lo_qua_vo_boc():
    """Đây là bài test đáng lẽ phải có từ đầu: đi qua ĐÚNG đối tượng mà app truyền vào."""
    synth = SynthCoLo()
    segs = _segs()

    synthesize_segments(_backend(synth), segs, "v", total_duration=30.0)

    assert synth.lo_da_goi == [["a", "b", "c"]]   # một lượt gọi cho cả ba
    assert synth.le_da_goi == []                  # KHÔNG được rơi về đọc từng câu


def test_nha_cung_cap_khong_gop_lo_van_chay_duong_cu():
    synth = SynthKhongLo()
    segs = _segs()

    fitted, _ = synthesize_segments(_backend(synth), segs, "v", workers=1, total_duration=30.0)

    assert sorted(synth.le_da_goi) == ["a", "b", "c"]
    assert len(fitted) == 3


# ─── nhận diện lời cũng gộp lô, và cũng phải được chuyển tiếp ────────────

class NhanDienCoLo:
    stt_batch_size = 32

    def __init__(self):
        self.lo_da_goi: list[int] = []

    def transcribe_batch(self, samples, rate, regions):
        self.lo_da_goi.append(len(regions))
        return [("en", f"loi thoai {i}") for i in range(len(regions))]

    def transcribe_clip(self, path):
        raise AssertionError("khong duoc goi khi da co gop lo")


class NhanDienKhongLo:
    def transcribe_clip(self, path):
        return "en", "loi thoai"


def test_vo_boc_phoi_ra_stt_batch_size():
    b = CompositeBackend(recognizer=NhanDienCoLo(), translator=None, synthesizer=None)
    assert b.stt_batch_size == 32


def test_vo_boc_bao_0_khi_nhan_dien_khong_gop_lo_duoc():
    """Provider không có stt_batch_size — phải trả 0, không được nổ AttributeError."""
    b = CompositeBackend(recognizer=NhanDienKhongLo(), translator=None, synthesizer=None)
    assert b.stt_batch_size == 0


def test_pipeline_that_su_di_duong_gop_lo_khi_nhan_dien():
    import numpy as np

    from pipeline.segmentation import Region
    from pipeline.stt import transcribe_regions

    nd = NhanDienCoLo()
    b = CompositeBackend(recognizer=nd, translator=None, synthesizer=None)
    regions = [Region(i * 5.0, i * 5.0 + 4.0) for i in range(3)]
    samples = np.zeros(16000 * 20, dtype="<i2")

    lang, segs, _ = transcribe_regions(b, samples, 16000, regions, workdir=None)

    assert nd.lo_da_goi == [3]          # một lượt gọi cho cả ba vùng
    assert lang == "en"
    assert len(segs) == 3
