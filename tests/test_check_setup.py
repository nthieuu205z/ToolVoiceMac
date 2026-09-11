from __future__ import annotations

import re
import shlex
import subprocess
import sys
import types
from pathlib import Path

from scripts import check_setup


def test_setup_output_names_both_supported_languages_and_audio_formats():
    output = check_setup.capabilities_summary()
    assert "vi-VN" in output
    assert "en-US" in output
    assert "Video Dubbing" in output
    assert "Text → Voice" in output
    assert "WAV" in output
    assert "MP3" in output


def test_user_documentation_covers_languages_exports_and_distinct_job_limits():
    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "HUONG_DAN_MAC.md"):
        text = (root / name).read_text(encoding="utf-8")
        for term in ("Video Dubbing", "Text → Voice", "vi-VN", "en-US", "WAV", "MP3"):
            assert term in text, f"{name} must explain {term}"
        # Permit common digit grouping styles while preserving both distinct limits.
        for number in (r"50[.,_ ]?000", r"200[.,_ ]?000"):
            assert re.search(rf"(?<!\d){number}(?!\d)", text), f"{name} is missing a text limit"
        assert "project" in text.casefold()


def test_docs_disclose_which_services_receive_content():
    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "HUONG_DAN_MAC.md"):
        paragraphs = (root / name).read_text(encoding="utf-8").split("\n\n")
        for provider, service in (("Edge", "Microsoft"), ("Gemini", "Google")):
            assert any(
                provider in paragraph and service in paragraph
                and re.search(r"gửi|truyền", paragraph, re.IGNORECASE)
                and re.search(r"văn bản|nội dung", paragraph, re.IGNORECASE)
                for paragraph in paragraphs
            ), f"{name} must disclose {provider} content submission to {service}"


def test_documented_toolvoice_commands_use_the_installed_launcher_interface():
    root = Path(__file__).resolve().parents[1]
    help_result = subprocess.run(
        [sys.executable, "-m", "toolvoice", "--help"],
        cwd=root, capture_output=True, text=True, timeout=10,
    )
    assert help_result.returncode == 0, help_result.stderr
    supported_flags = set(re.findall(r"--[a-z][a-z-]*", help_result.stdout))
    assert {"--doctor", "--no-browser", "--port", "--data-dir"} <= supported_flags
    for name in ("README.md", "HUONG_DAN_MAC.md"):
        text = (root / name).read_text(encoding="utf-8")
        commands = [
            shlex.split(line, comments=True)
            for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", text, re.DOTALL)
            for line in block.splitlines()
            if line.strip().startswith("toolvoice")
        ]
        assert commands, f"{name} must show how to start toolvoice"
        for command in commands:
            assert command[0] == "toolvoice"
            documented_flags = {word.split("=", 1)[0] for word in command[1:] if word.startswith("--")}
            assert documented_flags <= supported_flags, f"Unsupported command in {name}: {command}"


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
