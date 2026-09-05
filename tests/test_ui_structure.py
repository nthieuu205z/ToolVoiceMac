from pathlib import Path


def test_primary_upload_panel_is_before_optional_voice_lab():
    html = Path("web/static/index.html").read_text(encoding="utf-8")

    assert html.index('id="uploadPanel"') < html.index('id="voiceSettings"')


def test_voice_cards_have_explicit_preview_and_selection_controls():
    source = Path("web/static/app.js").read_text(encoding="utf-8")

    assert "voice-preview" in source
    assert "voice-select-button" in source
    assert "new Audio" in source
    assert "preview_url" in source


def test_upload_control_ids_are_wired_to_the_new_job_flow():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    source = Path("web/static/app.js").read_text(encoding="utf-8")

    for element_id in ("uploadPanel", "uploadForm", "videoInput", "voiceSelect", "startButton", "uploadProgress"):
        assert f'id="{element_id}"' in html
    assert 'xhr.open("POST", "/api/jobs")' in source
    assert 'form.append("video", state.selectedFile)' in source
    assert 'form.append("voice_id", voiceId)' in source


def test_frontend_cache_busts_static_assets():
    html = Path("web/static/index.html").read_text(encoding="utf-8")

    assert 'href="style.css?v=' in html
    assert 'src="app.js?v=' in html


def test_runtime_settings_and_shutdown_controls_are_present():
    html = Path("web/static/index.html").read_text(encoding="utf-8")

    for element_id in ("geminiSettingsForm", "geminiApiKey", "geminiBackend", "geminiSettingsHint", "shutdownButton"):
        assert f'id="{element_id}"' in html
