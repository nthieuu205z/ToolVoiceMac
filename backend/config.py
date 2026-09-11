"""Cấu hình đọc từ .env. Khóa API không bao giờ nằm trong mã nguồn."""

from __future__ import annotations

from pathlib import Path
from importlib.resources import files
from toolvoice.paths import data_dir
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = data_dir()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""

    # Khóa API đi tới endpoint nào:
    #   developer — generativelanguage.googleapis.com (khóa từ aistudio.google.com)
    #   vertex    — aiplatform.googleapis.com, Vertex AI Express Mode (khóa từ Google Cloud)
    # Cùng một khóa nhưng hai endpoint; project nào bật API nào thì dùng cái đó.
    gemini_backend: str = "developer"

    # Nhà cung cấp cho từng bước.
    stt_provider: str = "whisper"   # whisper (miễn phí, chạy trên máy) | gemini
    tts_provider: str = "edge"      # edge (cần mạng) | gemini
    # Bước dịch luôn dùng Gemini — chỉ tốn 1–4 lượt gọi cho cả video.

    # Engine đọc giọng NHÂN BẢN, định tuyến riêng khỏi giọng dựng sẵn.
    #   omnivoice  nhân bản zero-shot chất lượng cao (CC-BY-NC, cần cài + tải model ~3,3 GB).
    #   none       tắt tính năng nhân bản.
    clone_tts_provider: str = "omnivoice"

    # Model Gemini — đổi được khi Google cập nhật, không cần sửa code.
    gemini_stt_model: str = "gemini-3.5-flash"
    gemini_translate_model: str = "gemini-3.5-flash"
    gemini_tts_model: str = "gemini-2.5-flash-preview-tts"

    # Whisper chạy trên máy: tiny | base | small | medium | large-v3 (càng lớn càng chậm, càng chuẩn)
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    # faster-whisper không có backend MPS; trên Apple Silicon vẫn gộp feature chunks trên CPU.
    # 0 = đường từng vùng, 8 = mức đã đo nhanh hơn rõ rệt trên M1 Max 64 GB.
    whisper_cpu_batch_size: int = 8

    # edge-tts bóp tần suất — thử lại là bắt buộc.
    edge_tts_attempts: int = 5


    # OmniVoice (engine giọng nhân bản). num_step = số bước sinh: 32 mặc định (chất lượng),
    # hạ 16–24 để nhanh hơn. batch_size 0 = tự suy từ VRAM, cạp ở knee khoảng 8.
    omnivoice_num_step: int = 32
    omnivoice_batch_size: int = 0
    # Opt-in lúc khởi động; singleton không đổi mode khi đã nạp.
    omnivoice_optimization: Literal["none", "split-cfg", "split-cfg-rms", "split-cfg-rms-gqa", "split-cfg-rms-gqa-rope"] = "split-cfg-rms-gqa-rope"
    # Opt-in after codec compatibility checks on the installed MPS stack.
    omnivoice_codec_device: Literal["cpu", "mps"] = "mps"


    # ffmpeg: để trống thì tìm trong PATH.
    ffmpeg_bin: str = ""
    ffprobe_bin: str = ""

    # Tinh chỉnh pipeline. Trần lượt phát ngôn là hạt đồng bộ hình–tiếng:
    # càng ngắn càng bám sát mốc câu gốc (xem pipeline/segmentation.py).
    max_utterance_seconds: float = 12.0
    max_utterance_gap: float = 0.5
    # Dùng mốc thời gian cấp CÂU của Whisper: mỗi câu là một lượt đọc đặt đúng thời điểm câu
    # tiếng Anh, thay vì nhồi cả vùng ffmpeg (2–4 câu) làm một khối. Bám hình sát hơn,
    # hết cảnh "Hôm ...(nghỉ)... nay". Đặt False để về cấp vùng.
    sentence_level_timing: bool = True
    # Gemini STT chịu được nhiều luồng song song (đặt 8 trong .env); Whisper thì
    # tự tuần tự hóa bên trong nên chạy whisper hãy hạ về 1 cho khỏi tranh CPU.
    stt_workers: int = 1
    # Đo thực nghiệm: edge-tts 6 luồng → 25/32 hỏng; 2 luồng → 10/10.
    tts_workers: int = 2
    # Số lô dịch chạy song song. Nút cổ chai là model xuất token, không phải CPU — đo thật:
    # 4 lô tuần tự 63,7s → 6 lô song song ~11s. Vertex chịu được; hạ về 2–3 nếu dùng
    # Developer API free tier (RPM thấp) để tránh 429.
    translate_workers: int = 6
    # Trần tăng tốc để ép câu tiếng Việt (dài hơn khe gốc) vừa khung. 1,5× nghe rõ là
    # "nói nhanh"; 1,3× êm hơn hẳn mà timing vẫn khá sát — phần dư tràn sang câu sau rồi
    # tự tan ở khoảng lặng kế tiếp (xem pipeline/audio.py::fit_to_window). Nới lên nếu
    # muốn dub bám hình chặt hơn, hạ xuống nếu muốn giọng êm hơn nữa.
    tts_max_speedup: float = 1.3
    # Sàn kéo-CHẬM để lấp khung khi tiếng Việt xong sớm hơn hình
    # nhanh hơn giọng Anh). 0,9× kéo dài thêm tối đa ~11% — dưới ngưỡng tai — để bám hình
    # thay vì để im lặng cụt lủn. Đặt 1,0 để tắt (giữ hành vi cũ "không bao giờ kéo chậm").
    # Chỉ áp cho câu NGẮN hơn khung; câu dài vẫn tăng tốc như thường.
    tts_fill_slowdown: float = 0.9
    tts_daily_budget: int = 90
    max_upload_mb: int = 8192

    # Số video xử lý đồng thời. Đặt thấp có chủ ý: Whisper/OmniVoice bị khóa suy luận
    # toàn cục và edge-tts bị trần 2 request đồng thời — job thứ ba chủ yếu chen hàng
    # chứ không nhanh thêm. Video vượt trần sẽ xếp hàng chờ, không bị từ chối.
    max_concurrent_jobs: int = 2

    def validate_providers(self) -> None:
        """Reject unsupported configuration before a job can be misrouted."""
        stt = (self.stt_provider or "").strip().lower()
        tts = (self.tts_provider or "").strip().lower()
        clone = (self.clone_tts_provider or "").strip().lower()
        if stt not in {"whisper", "gemini"}:
            raise ValueError(f"Nhà cung cấp STT không hợp lệ: {self.stt_provider}")
        if tts not in {"edge", "gemini"}:
            raise ValueError(f"Nhà cung cấp preset không hợp lệ: {self.tts_provider}")
        if clone not in {"", "none", "omnivoice"}:
            raise ValueError(f"Nhà cung cấp clone không hợp lệ: {self.clone_tts_provider}")


    @property
    def provider_config(self):
        return self.provider_config_for(self.tts_provider)

    def provider_config_for(self, tts_provider: str):
        """ProviderConfig với engine giọng đọc CHỈ ĐỊNH — để định tuyến theo giọng mỗi job."""
        from pipeline.backends import ProviderConfig

        return ProviderConfig(
            stt_provider=self.stt_provider,
            tts_provider=tts_provider,
            gemini_api_key=self.gemini_api_key,
            gemini_backend=self.gemini_backend,
            gemini_stt_model=self.gemini_stt_model,
            gemini_translate_model=self.gemini_translate_model,
            gemini_tts_model=self.gemini_tts_model,
            whisper_model=self.whisper_model,
            whisper_compute_type=self.whisper_compute_type,
            whisper_cpu_batch_size=self.whisper_cpu_batch_size,
            edge_tts_attempts=self.edge_tts_attempts,
            omnivoice_num_step=self.omnivoice_num_step,
            omnivoice_batch_size=self.omnivoice_batch_size,
        )

    @property
    def resolved_clone_provider(self) -> str | None:
        """Engine nhân bản thực tế: OmniVoice hoặc tắt."""
        choice = (self.clone_tts_provider or "").strip().lower()
        if choice in ("", "none"):
            return None
        if choice == "omnivoice":
            import importlib.util

            if importlib.util.find_spec("omnivoice") is None:
                return None
        return "omnivoice" if choice == "omnivoice" else None

    @property
    def model_specs(self) -> list:
        """Các model cần tải về máy, theo nhà cung cấp đang cấu hình."""
        from pipeline.model_store import omnivoice_spec, whisper_spec

        specs = []
        if self.stt_provider == "whisper":
            specs.append(whisper_spec(self.whisper_model))
        clone = self.resolved_clone_provider
        if clone == "omnivoice":
            specs.append(omnivoice_spec())
        return specs

    @property
    def jobs_dir(self) -> Path:
        path = ROOT / "jobs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def static_dir(self) -> Path:
        return Path(str(files("web").joinpath("static")))

    @property
    def previews_dir(self) -> Path:
        return ROOT / "previews"

    @property
    def custom_voices_dir(self) -> Path:
        return ROOT / "custom_voices"


settings = Settings()
