from __future__ import annotations

from types import SimpleNamespace


def _configure_real_cli(monkeypatch, tmp_path):
    import scripts.run_pipeline_cli as cli

    settings = SimpleNamespace(
        gemini_api_key="present",
        ffmpeg_bin="",
        ffprobe_bin="",
        stt_provider="whisper",
        tts_provider="edge",
        resolved_clone_provider="omnivoice",
        whisper_model="small",
        whisper_compute_type="int8",
        max_utterance_seconds=12.0,
        max_utterance_gap=0.5,
        stt_workers=1,
        tts_workers=2,
        tts_max_speedup=1.3,
        tts_daily_budget=90,
        tts_fill_slowdown=0.9,
        custom_voices_dir=tmp_path / "voices",
        provider_config_for=lambda provider: provider,
        validate_providers=lambda: None,
    )
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    built = []
    options = []

    monkeypatch.setattr(cli, "settings", settings)
    monkeypatch.setattr(cli.custom_voices, "configure", lambda directory: None)
    monkeypatch.setattr(cli, "set_binaries", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "build_backend",
        lambda config: built.append(config) or object(),
    )
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda *args: options.append(args[3]) or SimpleNamespace(
            video_path="video.mp4",
            srt_path="video.srt",
            language="en",
            target_language=args[3].target_language,
            segment_count=1,
            spoken_count=1,
            attempted_count=1,
            warnings=[],
        ),
    )
    return cli, video, built, options


def test_cli_rejects_explicit_incompatible_voice_before_building_backend(
    monkeypatch, tmp_path
):
    cli, video, built, options = _configure_real_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "run_pipeline_cli.py",
            str(video),
            "--target-language",
            "en-US",
            "--voice",
            "vi-VN-HoaiMyNeural",
        ],
    )

    assert cli.main() == 1
    assert built == []
    assert options == []


def test_cli_chooses_an_english_compatible_default_voice(monkeypatch, tmp_path):
    cli, video, built, options = _configure_real_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run_pipeline_cli.py", str(video), "--target-language", "en-US"],
    )

    assert cli.main() == 0
    assert built == ["edge"]
    assert options[0].voice_id == "en-US-AvaMultilingualNeural"
    assert options[0].target_language == "en-US"


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
        target_language = "en-US"
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
    monkeypatch.setattr(
        cli,
        "default_voice",
        lambda provider, language=None: "clone-demo",
    )
    monkeypatch.setattr(cli, "is_available", lambda *args, **kwargs: True)

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
    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_known_args",
        lambda self: (SimpleNamespace(target_language="en-US"), []),
    )
    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", lambda self: FakeArgs())

    FakeArgs.video.write_bytes(b"video")
    assert cli.main() == 0
    assert available == [("edge", "omnivoice")]
    assert built == ["omnivoice"]
    assert options[0].tts_is_metered is False
    assert options[0].resynthesize_holes is False
    assert options[0].target_language == "en-US"
