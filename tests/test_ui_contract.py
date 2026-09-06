from pathlib import Path


def test_new_job_upload_contract_is_wired_end_to_end():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    js = Path("web/static/app.js").read_text(encoding="utf-8")

    for element_id in ("uploadForm", "videoInput", "videoLanguage", "videoVoiceSelect", "startButton", "uploadProgress", "uploadBar"):
        assert f'id="{element_id}"' in html
    assert '$("#uploadForm").addEventListener("submit", submitUpload)' in js
    assert 'xhr.open("POST", "/api/jobs")' in js
    assert 'form.append("video", state.selectedFile)' in js
    assert 'form.append("voice_id", voiceId)' in js


def test_page_declares_an_inline_favicon_without_an_extra_request():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    assert 'rel="icon" href="data:image/svg+xml,' in html


def test_voice_lab_has_select_preview_and_delete_actions():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    js = Path("web/static/app.js").read_text(encoding="utf-8")

    assert 'id="voiceList"' in html
    assert 'id="toggleVoiceForm"' in html
    assert "voice-select-button" in js
    assert "voice-preview" in js
    assert "voice-delete" in js
    assert "new Audio" in js


def test_pipeline_graph_declares_all_runtime_stages():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    js = Path("web/static/app.js").read_text(encoding="utf-8")
    assert 'id="pipelineGraph"' in html
    assert "function renderGraphNodes(job)" in js
    for stage in ("extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux"):
        assert f'"{stage}"' in js
