from __future__ import annotations

from types import SimpleNamespace


def test_cli_routes_clone_provider_without_legacy_configuration(monkeypatch, tmp_path):
    import scripts.run_pipeline_cli as cli

    class FakeSettings:
        gemini_api_key = "present"
        ffmpeg_bin = ""
        ffprobe_bin = ""
        stt_provider = "whisper"
        tts_provider = "edge"
        resolved_clone_provider = "omnivoice"

        whisper_model = "small"
        whisper_compute_type = "int8"
        max_utterance_seconds = 12.0
        max_utterance_gap = 0.5
        stt_workers = 1
        tts_workers = 2
        tts_max_speedup = 1.3
        tts_daily_budget = 90
        tts_fill_slowdown = 0.9
        custom_voices_dir = tmp_path / "voices"

        def provider_config_for(self, provider):
            return provider

        def validate_providers(self):
            return None

    class FakeArgs:
        voice = "clone-demo"
        video = tmp_path / "input.mp4"
        outdir = tmp_path / "out"
        verbose = False

    built = []
    options = []
    available = []

    monkeypatch.setattr(cli, "settings", FakeSettings())
    monkeypatch.setattr(cli.custom_voices, "configure", lambda directory: None)
    monkeypatch.setattr(
        cli,
        "available_voices",
        lambda *args: available.append(args) or [SimpleNamespace(id="clone-demo")],
        raising=False,
    )
    monkeypatch.setattr(cli, "default_voice", lambda provider: "clone-demo")

    monkeypatch.setattr(cli, "build_backend", lambda config: built.append(config) or object())
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda *args: options.append(args[3]) or SimpleNamespace(
            video_path="video.mp4", srt_path="video.srt", language="en",
            segment_count=1, spoken_count=1, attempted_count=1, warnings=[],
        ),
    )
    monkeypatch.setattr(cli, "set_binaries", lambda *args: None)
    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", lambda self: FakeArgs())

    FakeArgs.video.write_bytes(b"video")
    assert cli.main() == 0
    assert available == [("edge", "omnivoice")]
    assert built == ["omnivoice"]
    assert options[0].tts_is_metered is False
    assert options[0].resynthesize_holes is False
