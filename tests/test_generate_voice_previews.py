from __future__ import annotations

from types import SimpleNamespace


def test_preview_script_routes_each_voice_language_to_the_correct_provider(
    monkeypatch, tmp_path
):
    import scripts.generate_voice_previews as previews

    custom = SimpleNamespace(
        id="clone-demo", custom=True, supported_languages=("vi-VN", "en-US")
    )
    preset = SimpleNamespace(
        id="en-US-AvaMultilingualNeural",
        custom=False,
        supported_languages=("vi-VN", "en-US"),
    )
    built: list[str] = []
    generated = []

    class FakeSettings:
        ffmpeg_bin = ""
        ffprobe_bin = ""
        tts_provider = "edge"
        resolved_clone_provider = "omnivoice"
        gemini_api_key = "present"
        previews_dir = tmp_path
        custom_voices_dir = tmp_path / "voices"

        def provider_config_for(self, provider):
            return provider

        def validate_providers(self):
            return None

    class FakeParser:
        def add_argument(self, *args, **kwargs):
            return None

        def parse_args(self):
            return SimpleNamespace(force=True)

    class FakeBackend:
        def __init__(self, provider):
            self.provider = provider

    class FakeStore:
        def __init__(self, root):
            assert root == tmp_path

        def clear(self, voice_id):
            generated.append(("clear", voice_id))

        def path(self, voice_id, language):
            return tmp_path / voice_id / f"{language}.wav"

        def generate_languages(self, voice_id, languages, backend):
            generated.append((voice_id, tuple(languages), backend.provider))

        def status(self, voice_id, language):
            return "ready"

    monkeypatch.setattr(previews, "settings", FakeSettings())
    monkeypatch.setattr(previews.argparse, "ArgumentParser", lambda: FakeParser())
    monkeypatch.setattr(previews, "set_binaries", lambda *args: None)
    monkeypatch.setattr(previews.custom_voices, "configure", lambda path: None)
    monkeypatch.setattr(previews, "available_voices", lambda *args: [preset, custom])
    monkeypatch.setattr(
        previews,
        "build_backend",
        lambda config: built.append(config) or FakeBackend(config),
    )
    monkeypatch.setattr(previews, "VoicePreviewStore", FakeStore)

    assert previews.main() == 0
    assert built == ["edge", "omnivoice"]
    assert generated == [
        ("clear", preset.id),
        (preset.id, ("vi-VN", "en-US"), "edge"),
        ("clear", custom.id),
        (custom.id, ("vi-VN", "en-US"), "omnivoice"),
    ]
