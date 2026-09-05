"""Kiểu dữ liệu dùng chung giữa các bước của pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

# Thứ tự các bước video khớp với `data-stage` trong web/static/index.html.
# Phụ đề dựng SAU giọng đọc: mốc thời gian của cue bám theo thời lượng đọc thật,
# nếu dựng trước thì phụ đề trải đều trên khung gốc và trôi khỏi tiếng nói.
VIDEO_STAGES = (
    "extract",
    "transcribe",
    "translate",
    "synthesize",
    "subtitle",
    "assemble",
    "mux",
)
TEXT_STAGES = ("prepare", "synthesize", "assemble", "export")

STAGES_BY_JOB_TYPE = {
    "video_dubbing": VIDEO_STAGES,
    "text_to_voice": TEXT_STAGES,
}
STAGE_WEIGHTS_BY_JOB_TYPE = {
    "video_dubbing": {
        "extract": 5,
        "transcribe": 25,
        "translate": 15,
        "synthesize": 40,
        "subtitle": 2,
        "assemble": 8,
        "mux": 5,
    },
    "text_to_voice": {
        "prepare": 5,
        "synthesize": 75,
        "assemble": 10,
        "export": 10,
    },
}

# Tên tương thích cho mọi caller video cũ và các kiểm thử frontend hiện tại.
STAGES = list(VIDEO_STAGES)
STAGE_WEIGHTS = STAGE_WEIGHTS_BY_JOB_TYPE["video_dubbing"]

# Định dạng âm thanh Gemini TTS trả về, cũng là định dạng track lồng tiếng.
TTS_SAMPLE_RATE = 24_000
# Định dạng âm thanh đưa vào nhận diện giọng nói.
STT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class MediaInfo:
    """Kết quả ffprobe của video đầu vào."""

    duration: float
    video_codec: str
    has_audio: bool
    width: int = 0
    height: int = 0


@dataclass
class Segment:
    """Một lượt phát ngôn. Mốc thời gian do ffmpeg xác định, nội dung do Gemini chép."""

    start: float
    end: float
    text: str
    target_text: str = ""
    # Thời lượng giọng đọc thật sau khi tạo, thường ngắn hơn khung gốc.
    # Phụ đề bám theo con số này để không trôi ra khỏi tiếng nói.
    spoken_duration: float = 0.0
    # Mốc phát THẬT sau bước xếp chỗ (plan_placement): thường bằng `start`, nhưng bị
    # đẩy lùi khi lượt trước tràn khung — phụ đề phải bám theo đây, không theo `start`.
    placed_start: float | None = None

    @property
    def text_vi(self) -> str:
        """Tên tương thích trong thời gian chuyển sang nội dung đích trung lập."""
        return self.target_text

    @text_vi.setter
    def text_vi(self, value: str) -> None:
        self.target_text = value

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def cue_start(self) -> float:
        """Mốc bắt đầu cho phụ đề: mốc phát thật nếu đã xếp chỗ, không thì mốc gốc."""
        return self.placed_start if self.placed_start is not None else self.start

    @property
    def cue_span(self) -> float:
        """Khoảng thời gian phụ đề được trải ra: theo giọng đọc nếu đã có, không thì theo khung."""
        return self.spoken_duration if self.spoken_duration > 0 else self.duration

    def shifted(self, offset: float) -> Segment:
        return replace(self, start=self.start + offset, end=self.end + offset)


@dataclass
class PipelineResult:
    video_path: str
    srt_path: str
    language: str = ""
    target_language: str = "vi-VN"
    segment_count: int = 0
    attempted_count: int = 0
    spoken_count: int = 0
    warnings: list[str] = field(default_factory=list)


# progress(stage, fraction_within_stage 0..1, message)
ProgressFn = Callable[[str, float, str], None]

# Trả True nghĩa là người dùng đã bấm hủy; pipeline dừng ở mốc an toàn gần nhất.
CancelFn = Callable[[], bool]


def noop_progress(stage: str, fraction: float, message: str) -> None:
    """Mặc định: không báo tiến trình (dùng cho test và CLI im lặng)."""


def never_cancel() -> bool:
    return False


def overall_percent(
    stage: str,
    fraction: float,
    job_type: str = "video_dubbing",
    fallback: float | None = None,
) -> float:
    """Quy đổi tiến độ theo loại job; stage cũ không làm hỏng dữ liệu khôi phục."""
    stages = STAGES_BY_JOB_TYPE.get(job_type)
    weights = STAGE_WEIGHTS_BY_JOB_TYPE.get(job_type)
    if stages is None or weights is None or stage not in stages:
        return fallback if fallback is not None else 0.0

    total = sum(weights.values())
    done = sum(weights[item] for item in stages[: stages.index(stage)])
    fraction = min(1.0, max(0.0, fraction))
    return round((done + weights[stage] * fraction) / total * 100, 1)


class SpeechRecognizer(Protocol):
    def transcribe_clip(self, wav_path) -> tuple[str, str]:
        """Trả về (mã ngôn ngữ nguồn, nguyên văn lời thoại) của một đoạn audio ngắn."""
        ...


class Translator(Protocol):
    def translate(
        self,
        texts: list[str],
        durations: list[float],
        context: str = "",
        *,
        target_language: str = "vi-VN",
    ) -> list[str]:
        """Dịch sang ngôn ngữ đích, trả về đúng số phần tử và đúng thứ tự như đầu vào."""
        ...


class SpeechSynthesizer(Protocol):
    def synthesize(
        self, text: str, voice_id: str, *, language: str = "vi-VN"
    ) -> bytes:
        """Trả về PCM 16-bit little-endian, mono, TTS_SAMPLE_RATE Hz."""
        ...


class GeminiBackend(SpeechRecognizer, Translator, SpeechSynthesizer, Protocol):
    """Tên tương thích tạm thời cho backend có đủ ba khả năng của pipeline."""
