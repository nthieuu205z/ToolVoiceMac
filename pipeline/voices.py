"""Danh sách giọng đọc, theo từng nhà cung cấp.

edge-tts (miễn phí, không hạn mức) chỉ có 2 giọng tiếng Việt.
Gemini có 30 giọng đa ngôn ngữ nhưng bị trần 100 lượt gọi mỗi ngày ở bản miễn phí.

Khi nhà cung cấp đổi danh sách, chỉ cần sửa dữ liệu ở đây.
"""

from __future__ import annotations

from dataclasses import dataclass

from .languages import normalize_language_code


@dataclass(frozen=True)
class Voice:
    id: str            # định danh an toàn cho URL và form
    display_name: str
    # Tên mà nhà cung cấp thật sự nhận, khi khác `id`.
    native_id: str = ""
    provider: str = ""
    supported_languages: tuple[str, ...] = ("vi-VN",)
    tags: tuple[str, ...] = ()

    @property
    def native(self) -> str:
        return self.native_id or self.id

    def supports(self, language: str) -> bool:
        return normalize_language_code(language) in self.supported_languages

    @property
    def custom(self) -> bool:
        from . import custom_voices

        return custom_voices.is_custom(self.id)


# Giọng neural của Microsoft Edge. Không cần API key, không trần theo ngày.
#
# Hai giọng đầu là giọng tiếng Việt bản địa. Mười hai giọng còn lại là giọng
# "Multilingual" — tuy mang nhãn en-US/fr-FR/… nhưng đọc được tiếng Việt.
# Đã kiểm chứng: cho từng giọng đọc một câu tiếng Việt rồi bắt Whisper nghe lại,
# cả 12 giọng đều được nhận là tiếng Việt với độ khớp 0.94–1.00.
EDGE_VOICES: list[Voice] = [
    Voice("vi-VN-HoaiMyNeural", "Hoài My — nữ, giọng Việt bản địa", provider="edge"),
    Voice("vi-VN-NamMinhNeural", "Nam Minh — nam, giọng Việt bản địa", provider="edge"),

    Voice("en-US-AvaMultilingualNeural", "Ava — nữ, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("en-US-EmmaMultilingualNeural", "Emma — nữ, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("fr-FR-VivienneMultilingualNeural", "Vivienne — nữ, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("de-DE-SeraphinaMultilingualNeural", "Seraphina — nữ, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("pt-BR-ThalitaMultilingualNeural", "Thalita — nữ, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),

    Voice("en-US-AndrewMultilingualNeural", "Andrew — nam, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("en-US-BrianMultilingualNeural", "Brian — nam, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("en-AU-WilliamMultilingualNeural", "William — nam, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("fr-FR-RemyMultilingualNeural", "Rémy — nam, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("de-DE-FlorianMultilingualNeural", "Florian — nam, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("it-IT-GiuseppeMultilingualNeural", "Giuseppe — nam, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
    Voice("ko-KR-HyunsuMultilingualNeural", "Hyunsu — nam, đa ngôn ngữ", provider="edge",
          supported_languages=("vi-VN", "en-US")),
]

# Giọng dựng sẵn của Gemini. Google không công bố giới tính từng giọng — hãy bấm nghe thử.
GEMINI_VOICES: list[Voice] = [
    Voice("Charon", "Charon — dẫn chuyện, nhiều thông tin", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
    Voice("Kore", "Kore — chắc chắn, dứt khoát", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
    Voice("Sulafat", "Sulafat — ấm áp", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
    Voice("Puck", "Puck — tươi tắn, sôi nổi", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
    Voice("Aoede", "Aoede — nhẹ nhàng, thoáng đãng", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
    Voice("Iapetus", "Iapetus — trong trẻo, rõ chữ", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
    Voice("Achird", "Achird — thân thiện", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
    Voice("Vindemiatrix", "Vindemiatrix — dịu dàng", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
    Voice("Gacrux", "Gacrux — chững chạc, trưởng thành", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
    Voice("Rasalgethi", "Rasalgethi — thuyết minh, mạch lạc", provider="gemini",
          supported_languages=("vi-VN", "en-US")),
]

# OmniVoice không có giọng dựng sẵn — nó là engine nhân bản thuần, nên danh sách gốc rỗng;
# giọng của nó hoàn toàn là các giọng nhân bản người dùng đã lưu (ghép vào bên dưới).
_BY_PROVIDER = {"edge": EDGE_VOICES, "gemini": GEMINI_VOICES, "omnivoice": []}

# OmniVoice là engine duy nhất dùng giọng nhân bản.
_CLONE_PROVIDERS = {"omnivoice"}


def _clone_voices() -> list[Voice]:
    """Giọng nhân bản người dùng đã lưu."""
    from . import custom_voices

    return [
        Voice(c.id, f"{c.display_name} — giọng nhân bản", c.id,
              provider="omnivoice", supported_languages=c.supported_languages, tags=c.tags)
        for c in custom_voices.list_custom()
    ]


def voices_for(provider: str) -> list[Voice]:
    if provider not in _BY_PROVIDER:
        raise ValueError(f"Nhà cung cấp giọng đọc không hợp lệ: {provider}")
    base = _BY_PROVIDER[provider]
    if provider not in _CLONE_PROVIDERS:
        return base
    return base + _clone_voices()


def available_voices(
    tts_provider: str,
    clone_provider: str | None,
    language: str | None = None,
) -> list[Voice]:
    """Giọng cho giao diện/kiểm tra khi ĐỊNH TUYẾN theo loại giọng.

    = giọng dựng sẵn của `tts_provider` + giọng nhân bản (nếu có engine clone). Nhờ vậy
    dù đặt TTS_PROVIDER=edge, người dùng vẫn thấy và chọn được giọng nhân bản — chúng sẽ
    được đọc bằng engine clone (OmniVoice) qua `route_provider`.
    """
    if tts_provider not in {"edge", "gemini"}:
        raise ValueError(f"Nhà cung cấp preset không hợp lệ: {tts_provider}")
    voices = list(_BY_PROVIDER[tts_provider])
    if clone_provider in _CLONE_PROVIDERS:
        voices += _clone_voices()
    if language is None:
        return voices
    canonical = normalize_language_code(language)
    return [voice for voice in voices if voice.supports(canonical)]


def route_provider(voice_id: str, tts_provider: str, clone_provider: str | None) -> str:
    """Engine đọc cho MỘT giọng: giọng nhân bản → engine clone; còn lại → tts_provider.

    Mỗi job chỉ dùng một giọng, nên đây là một quyết định duy nhất cho cả job (không phải
    mỗi câu). Không có engine clone thì đành trả tts_provider — không còn đường nào khác.
    """
    from . import custom_voices

    if tts_provider not in {"edge", "gemini"}:
        raise ValueError(f"Nhà cung cấp preset không hợp lệ: {tts_provider}")

    if custom_voices.is_custom(voice_id):
        return "omnivoice" if clone_provider == "omnivoice" else tts_provider
    return tts_provider


def is_valid(voice_id: str, provider: str) -> bool:
    return any(v.id == voice_id for v in voices_for(provider))


def is_available(
    voice_id: str,
    tts_provider: str,
    clone_provider: str | None,
    language: str | None = None,
) -> bool:
    return any(
        v.id == voice_id
        for v in available_voices(tts_provider, clone_provider, language=language)
    )


def default_voice(provider: str, language: str | None = None) -> str:
    """Giọng mặc định của provider. OmniVoice không có giọng dựng sẵn nên khi CHƯA có giọng
    nhân bản nào, danh sách rỗng — báo lỗi rõ ràng thay vì IndexError khó hiểu."""
    voices = voices_for(provider)
    if language is not None:
        canonical = normalize_language_code(language)
        voices = [voice for voice in voices if voice.supports(canonical)]
    if not voices:
        raise ValueError(
            f"Nhà cung cấp '{provider}' chưa có giọng nào để chọn mặc định. OmniVoice chỉ đọc "
            f"bằng giọng nhân bản — hãy tạo/chỉ định một giọng nhân bản (id 'clone-…'), hoặc "
            f"đổi TTS_PROVIDER sang edge hoặc gemini."
        )
    return voices[0].id


def native_id(voice_id: str, provider: str) -> str:
    """Slug trong URL → tên mà nhà cung cấp thật sự nhận."""
    for voice in voices_for(provider):
        if voice.id == voice_id:
            return voice.native
    return voice_id
