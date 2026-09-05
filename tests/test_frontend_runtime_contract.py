from pathlib import Path


ROOT = Path("web/static")


def test_page_does_not_depend_on_external_font_for_basic_rendering():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert '--sans: "Be Vietnam Pro", "Noto Sans", -apple-system' in css
    assert '--mono: "JetBrains Mono", ui-monospace' in css
    assert "font-synthesis: none" in css


def test_upload_form_has_a_visible_file_input_fallback():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="videoInput"' in html
    assert 'accept="video/*,.mp4,.mkv,.mov,.avi,.webm"' in html
    assert 'input.click()' in js
    assert 'input.addEventListener("change"' in js


def test_upload_state_accepts_a_video_file_even_when_browser_omits_mime():
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "file.type.startsWith(\"video/\")" in js
    assert "/\\.(mp4|mkv|mov|avi|webm)$/i.test(file.name)" in js
    assert "selectedUpload" in js
    assert "startButton" in js


def test_upload_submission_has_an_explicit_ready_path():
    js = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "state.selectedFile = isVideoFile(file) ? file : null" in js
    assert "Boolean(state.selectedFile && $(\"#voiceSelect\").value" in js
    assert "xhr.send(form)" in js
    assert "xhr.onload" in js


def test_voice_preview_has_a_real_audio_endpoint_and_failure_state():
    js = (ROOT / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert "/api/voices/${encodeURIComponent(voice.id)}/preview" in js
    assert "new Audio" in js
    assert "audio.play()" in js
    assert "Chưa có file demo cho giọng này." in js
    assert 'id="voiceList"' in html
