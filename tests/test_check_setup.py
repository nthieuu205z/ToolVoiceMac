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


def test_docs_disclose_network_text_submission_and_explicit_cli_language():
    readme = Path("README.md").read_text(encoding="utf-8")
    mac = Path("HUONG_DAN_MAC.md").read_text(encoding="utf-8")
    project = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "nội dung được gửi tới dịch vụ Microsoft" in readme
    assert "--target-language en-US" in readme
    assert "đa ngôn ngữ" in project
    assert "Khóa Gemini API (bắt buộc cho Video Dubbing" in mac


def test_text_to_voice_setup_can_be_ready_without_gemini(monkeypatch, capsys):
    monkeypatch.setattr(check_setup, "check_ffmpeg", lambda: True)
    monkeypatch.setattr(check_setup, "check_stt", lambda: True)
    monkeypatch.setattr(check_setup, "check_tts", lambda: True)
    monkeypatch.setattr(check_setup, "check_translate", lambda: False)

    assert check_setup.main() == 0
    output = capsys.readouterr().out
    assert "Text → Voice: sẵn sàng" in output
    assert "Video Dubbing: chưa sẵn sàng" in output


def test_gemini_tts_is_not_ready_without_a_gemini_key(monkeypatch, capsys):
    monkeypatch.setattr(check_setup.settings, "tts_provider", "gemini")
    monkeypatch.setattr(check_setup.settings, "clone_tts_provider", "none")
    monkeypatch.setattr(check_setup.settings, "gemini_api_key", "")
    monkeypatch.setattr(check_setup, "voices_for", lambda provider: [])
    monkeypatch.setattr(
        type(check_setup.settings),
        "resolved_clone_provider",
        property(lambda self: None),
    )

    assert check_setup.check_tts() is False
    assert "Gemini TTS cần GEMINI_API_KEY" in capsys.readouterr().out


def test_missing_optional_clone_does_not_disable_preset_text_to_voice(monkeypatch, capsys):
    monkeypatch.setattr(check_setup.settings, "tts_provider", "edge")
    monkeypatch.setattr(check_setup.settings, "clone_tts_provider", "omnivoice")
    monkeypatch.setattr(
        type(check_setup.settings),
        "resolved_clone_provider",
        property(lambda self: None),
    )
    monkeypatch.setattr(check_setup, "voices_for", lambda provider: [])
    monkeypatch.setitem(sys.modules, "edge_tts", types.ModuleType("edge_tts"))

    assert check_setup.check_tts() is True
    assert "Giọng nhân bản OmniVoice chưa sẵn sàng" in capsys.readouterr().out


def test_check_tts_checks_omnivoice_even_with_gemini_presets(monkeypatch):
    import scripts.check_setup as check_setup

    monkeypatch.setattr(check_setup.settings, "tts_provider", "gemini")
    monkeypatch.setattr(check_setup.settings, "gemini_api_key", "test-key")
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
