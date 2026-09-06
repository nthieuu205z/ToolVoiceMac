from pathlib import Path


ROOT = Path("web/static")


def test_bootstrap_does_not_pass_document_to_css_selector_helper():
    source = (ROOT / "app.js").read_text(encoding="utf-8")

    assert '$(document).addEventListener' not in source
    assert 'document.addEventListener("click"' in source


def test_upload_file_selection_updates_a_visible_ready_state():
    source = (ROOT / "app.js").read_text(encoding="utf-8")

    assert "function isVideoFile(file)" in source
    assert "state.selectedFile = isVideoFile(file) ? file : null" in source
    assert 'selected.hidden = !state.selectedFile' in source
    assert 'button.disabled = !ready' in source
    assert 'button.setAttribute("aria-disabled", String(!ready))' in source


def test_upload_form_has_an_actionable_error_summary_and_cancel_control():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    source = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="uploadError"' in html
    assert 'id="uploadErrorText"' in html
    assert 'id="cancelUploadButton"' in html
    assert 'xhr.timeout = 120000' in source
    assert 'xhr.ontimeout' in source
    assert 'uploadRequest.abort()' in source


def test_upload_sends_the_selected_file_with_its_filename():
    source = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'form.append("video", state.selectedFile)' in source
    assert 'form.append("filename", state.selectedFile.name)' in source
    assert 'form.append("voice_id", voiceId)' in source
    assert 'xhr.send(form)' in source


def test_ui_assets_are_cache_busted_after_bugfix():
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert 'style.css?v=20260906-38' in html
    assert 'app.js?v=20260906-38' in html


def test_ui_uses_a_vietnamese_safe_font_stack():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert "Be+Vietnam+Pro" in html
    assert '--sans: "Be Vietnam Pro", "Noto Sans"' in css
    assert '--mono: "JetBrains Mono"' in css
    assert "font-synthesis: none" in css


def test_voice_lab_has_bounded_card_layout_and_readable_metadata():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert ".voice-panel { margin-top: 13px; padding: 20px; overflow: hidden; }" in css
    assert ".voice-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));" in css
    assert "max-width: none;" in css
    assert ".voice-item { display: grid; grid-template-columns: 30px minmax(0, 1fr) 193px;" in css
    assert ".voice-item-main { min-width: 0; overflow: hidden; }" in css
    assert ".voice-item-main small { display: block; margin-top: 4px;" in css
    assert "white-space: nowrap;" in css


def test_voice_lab_actions_cannot_escape_the_card():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert ".voice-actions { display: flex; align-items: center; justify-content: flex-end; gap: 7px; width: 193px; min-width: 0; flex: 0 0 193px; }" in css
    assert ".voice-preview {" in css
    assert "min-width: 0;" in css
    assert ".voice-preview span:last-child { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }" in css
    assert ".voice-delete { width: 34px; height: 34px;" in css


def test_voice_lab_mobile_rules_keep_grid_inside_the_panel():
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert ".voice-list { grid-template-columns: 1fr; }" in css
    assert ".voice-panel { margin-top: 13px; padding: 20px; overflow: hidden; }" in css
    assert ".main-content { width: 100%; margin: 0; padding: 0 15px 24px; overflow-x: hidden; }" in css


def test_voice_lab_uses_tooltip_friendly_full_names_for_truncated_cards():
    source = (ROOT / "app.js").read_text(encoding="utf-8")

    assert 'name.title = voice.display_name' in source
    assert 'id.title = voice.id' in source
