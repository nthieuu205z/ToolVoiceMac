from pathlib import Path


def test_upload_form_uses_the_same_submit_button_id_as_javascript():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    js = Path("web/static/app.js").read_text(encoding="utf-8")

    assert 'id="startButton"' in html
    assert '$("#startButton")' in js
    assert 'xhr.open("POST", "/api/jobs")' in js
    assert 'form.append("video", state.selectedFile)' in js
    assert 'form.append("voice_id", voiceId)' in js


def test_upload_error_surface_has_a_recoverable_message():
    js = Path("web/static/app.js").read_text(encoding="utf-8")

    assert "Không tải được video." in js
    assert "Mất kết nối khi tải video." in js
    assert "uploadBusy = false" in js


def test_voice_cards_expose_preview_control_contract():
    js = Path("web/static/app.js").read_text(encoding="utf-8")

    assert "preview_url" in js
    assert "new Audio" in js
    assert "voice-preview" in js
    assert "voice-select-button" in js


def test_voice_lab_handlers_are_declared_once():
    js = Path("web/static/app.js").read_text(encoding="utf-8")

    assert js.count("function renderVoiceLab(") == 1
    assert js.count("function deleteVoice(") == 1
    assert js.count("function setupVoiceLab(") == 1
