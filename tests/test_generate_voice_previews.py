from __future__ import annotations

from types import SimpleNamespace


def test_preview_script_routes_custom_voice_to_omnivoice(monkeypatch, tmp_path):
    import scripts.generate_voice_previews as previews

    custom = SimpleNamespace(id="clone-demo", custom=True)
    preset = SimpleNamespace(id="Charon", custom=False)
    built: list[str] = []

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
        def synthesize(self, text, voice_id):
            return b"\x00\x00"

    monkeypatch.setattr(previews, "settings", FakeSettings())
    monkeypatch.setattr(previews.argparse, "ArgumentParser", lambda: FakeParser())
    monkeypatch.setattr(previews, "set_binaries", lambda *args: None)
    monkeypatch.setattr(previews.custom_voices, "configure", lambda path: None)
    monkeypatch.setattr(previews, "available_voices", lambda *args: [preset, custom])
    monkeypatch.setattr(previews, "build_backend", lambda config: built.append(config) or FakeBackend())
    monkeypatch.setattr(previews, "pcm_to_array", lambda pcm: pcm)
    monkeypatch.setattr(previews, "write_wav", lambda *args: None)

    assert previews.main() == 0
    assert built == ["edge", "omnivoice"]
