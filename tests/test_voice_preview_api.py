from pathlib import Path


def test_voice_preview_api_is_explicitly_exposed_in_the_voice_route():
    source = Path("backend/routes/voices.py").read_text(encoding="utf-8")

    assert '@router.get("/api/voices/{voice_id}/preview")' in source
    assert 'media_type="audio/wav"' in source


def test_voice_list_marks_demo_only_when_the_file_exists():
    source = Path("backend/routes/voices.py").read_text(encoding="utf-8")

    assert '"preview_url": f"/previews/{voice.id}.wav" if preview.is_file() else ""' in source


def test_new_custom_voice_returns_the_preview_url_when_already_available():
    source = Path("backend/routes/voices.py").read_text(encoding="utf-8")

    assert 'preview = _preview_path(voice.id)' in source
    assert '"preview_url": f"/previews/{voice.id}.wav" if preview.is_file() else ""' in source


def test_preview_generation_uses_the_safe_preview_path():
    source = Path("backend/routes/voices.py").read_text(encoding="utf-8")

    assert 'preview = _preview_path(voice_id)' in source
    assert 'write_wav(preview, pcm_to_array(pcm))' in source


def test_frontend_uses_api_preview_endpoint_when_available():
    source = Path("web/static/app.js").read_text(encoding="utf-8")

    assert 'function previewUrl(voice)' in source
    assert '/api/voices/${encodeURIComponent(voice.id)}/preview' in source
    assert 'audio.play()' in source
    assert 'audio.pause()' in source


def test_frontend_exposes_upload_submit_contract():
    source = Path("web/static/app.js").read_text(encoding="utf-8")

    assert 'xhr.open("POST", "/api/jobs")' in source
    assert 'form.append("video", state.selectedFile)' in source
    assert 'form.append("voice_id", voiceId)' in source
    assert 'xhr.upload.onprogress' in source
    assert 'xhr.onerror' in source
    assert 'xhr.onabort' in source


def test_frontend_upload_ids_are_consistent():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    source = Path("web/static/app.js").read_text(encoding="utf-8")

    for element_id in ("uploadForm", "videoInput", "voiceSelect", "startButton", "uploadProgress", "uploadBar", "uploadPercent"):
        assert f'id="{element_id}"' in html
    assert '$("#startButton")' in source
    assert '$("#uploadForm")' in source
    assert '$("#videoInput")' in source
    assert '$("#voiceSelect")' in source


def test_voice_lab_has_demo_status_and_actions():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    source = Path("web/static/app.js").read_text(encoding="utf-8")

    assert 'id="voiceList"' in html
    assert 'id="voiceCount"' in html
    assert "Demo sẵn sàng" in source
    assert "Chưa có demo" in source
    assert "voice-preview" in source
    assert "voice-select-button" in source
    assert "voice-delete" in source


def test_voice_lab_handlers_are_not_duplicated():
    source = Path("web/static/app.js").read_text(encoding="utf-8")

    assert source.count("function renderVoiceLab(") == 1
    assert source.count("function setupVoiceLab(") == 1
    assert source.count("async function deleteVoice(") == 1
    assert source.count("function submitUpload(") == 1
    assert source.count("function togglePreview(") == 1


def test_preview_files_are_available_for_current_custom_voices():
    preview_dir = Path("web/static/previews")
    custom_ids = {"clone-demo-hoai-my-2", "clone-dung-lai-lap-trinh", "clone-doxumxue", "clone-dunglai", "clone-duyluan", "clone-lamdinh"}

    for voice_id in custom_ids:
        assert (preview_dir / f"{voice_id}.wav").is_file()
        assert (preview_dir / f"{voice_id}.wav").stat().st_size > 44
