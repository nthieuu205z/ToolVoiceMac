from __future__ import annotations

import sys
import types


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
