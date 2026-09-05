from pathlib import Path


def test_operations_console_uses_one_voice_lab_implementation():
    js = Path("web/static/app.js").read_text(encoding="utf-8")

    assert js.count("function renderVoiceLab(") == 1
    assert js.count("function deleteVoice(") == 1
    assert js.count("function setupVoiceLab(") == 1


def test_operations_console_exposes_voice_preview_and_upload_paths():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    js = Path("web/static/app.js").read_text(encoding="utf-8")

    assert 'id="voiceList"' in html
    assert "voice-preview" in js
    assert "new Audio" in js
    assert 'xhr.open("POST", "/api/jobs")' in js
    assert 'form.append("video", state.selectedFile)' in js
    assert 'form.append("voice_id", voiceId)' in js


def test_operations_console_has_recoverable_upload_states():
    js = Path("web/static/app.js").read_text(encoding="utf-8")

    assert "Không tải được video." in js
    assert "Mất kết nối khi tải video." in js
    assert "Đã hủy tải video." in js
    assert "uploadBusy = false" in js
