from __future__ import annotations

import sys
import types
from pathlib import Path

from scripts import check_setup


def test_setup_output_names_both_supported_languages_and_audio_formats():
    output = check_setup.capabilities_summary()
    assert output.count("\n") == 3
    assert "vi-VN" in output
    assert "en-US" in output
    assert "Video Dubbing" in output
    assert "Text → Voice" in output
    assert "WAV" in output
    assert "MP3" in output


def test_user_documentation_covers_multilingual_job_workflows():
    readme = Path("README.md").read_text(encoding="utf-8")
    mac = Path("HUONG_DAN_MAC.md").read_text(encoding="utf-8")
    env = Path(".env.example").read_text(encoding="utf-8")
    for text in (readme, mac):
        assert "Video Dubbing" in text
        assert "Text → Voice" in text
        assert "vi-VN" in text and "en-US" in text
        assert "WAV" in text and "MP3" in text
        assert "50.000" in text
    assert "RUN_PROVIDER_SMOKE=1" in readme
    assert "PROVIDER_SMOKE_CLONE_VOICE_ID" in env


def test_check_tts_checks_omnivoice_even_with_gemini_presets(monkeypatch):
    import scripts.check_setup as check_setup

    monkeypatch.setattr(check_setup.settings, "tts_provider", "gemini")
    monkeypatch.setattr(check_setup.settings, "clone_tts_provider", "omnivoice")
    monkeypatch.setattr(
        type(check_setup.settings),
        "resolved_clone_provider",
        property(lambda self: "omnivoice"),
    )
    monkeypatch.setattr(check_setup, "voices_for", lambda provider: [])

    imported: list[str] = []
    real_import = __import__

    def tracked_import(name, *args, **kwargs):
        if name in {"omnivoice", "torch", "edge_tts"}:
            imported.append(name)
            return types.ModuleType(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracked_import)

    assert check_setup.check_tts() is True
    assert "omnivoice" in imported
    assert "torch" in imported
    assert "edge_tts" not in imported
