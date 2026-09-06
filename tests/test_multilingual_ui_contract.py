from pathlib import Path

HTML = Path("web/static/index.html").read_text(encoding="utf-8")
JS = Path("web/static/app.js").read_text(encoding="utf-8")
CSS = Path("web/static/style.css").read_text(encoding="utf-8")


def test_new_job_uses_accessible_video_and_text_tabs():
    assert 'id="jobModeTabs" role="tablist"' in HTML
    assert 'id="videoJobTab" role="tab"' in HTML
    assert 'id="textJobTab" role="tab"' in HTML
    assert 'id="videoJobPanel" role="tabpanel"' in HTML
    assert 'id="textJobPanel" role="tabpanel"' in HTML
    assert all(key in JS for key in ("ArrowLeft", "ArrowRight", "Home", "End"))


def test_video_form_has_language_and_preserves_upload_contract():
    assert 'id="videoLanguage"' in HTML
    assert 'id="videoVoiceSelect"' in HTML
    assert 'id="videoInput"' in HTML
    assert 'form.append("target_language"' in JS
    assert 'xhr.open("POST", "/api/jobs")' in JS


def test_text_form_has_language_voice_preview_and_submit_controls():
    for element_id in (
        "textInput",
        "textCharacterCount",
        "textDurationEstimate",
        "textLanguage",
        "textVoiceSelect",
        "textFixedPreviewButton",
        "textPreviewButton",
        "textStartButton",
    ):
        assert f'id="{element_id}"' in HTML
    assert 'apiJson("/api/jobs/text"' in JS
    assert "AbortController" in JS


def test_text_jobs_render_typed_audio_downloads_and_text_stages():
    assert 'job.job_type === "text_to_voice"' in JS
    assert '/download/wav' in JS
    assert '/download/mp3' in JS
    assert 'prepare: { label:' in JS
    assert 'export: { label:' in JS
    assert "const TEXT_STAGES" in JS


def test_language_catalog_failure_does_not_block_other_boot_requests():
    assert 'catch (error) { state.languages = [' in JS
    assert 'await loadLanguages(); await Promise.all' not in JS


def test_job_mode_drafts_have_separate_storage_keys():
    for key in (
        "sub.video.language",
        "sub.video.voice",
        "sub.text.language",
        "sub.text.voice",
        "sub.voiceLab.language",
    ):
        assert key in JS
    assert "state.jobDrafts" in JS
    assert "function activateJobMode" in JS
    assert 'text: sessionStorage.getItem("sub.text.draft") || ""' in JS
    assert 'sessionStorage.setItem("sub.text.draft", event.target.value)' in JS
    assert 'sessionStorage.removeItem("sub.text.draft")' in JS


def test_voice_lab_has_a_language_selector_and_preview_retry_contract():
    assert 'id="voiceLabLanguage"' in HTML
    assert "preview_status" in JS
    assert "/preview/regenerate?language=" in JS
    assert "Đang tạo bản nghe thử" in JS
    assert "Tạo lại demo" in JS
    assert "function pollVoicePreview" in JS
    assert "window.setTimeout(() => pollVoicePreview" in JS


def test_on_demand_preview_can_be_stopped_without_starting_another_request():
    assert "if (activePreview?.button === button) { abortTextPreview(); return; }" in JS


def test_multilingual_composer_has_accessible_responsive_styles():
    assert ".job-mode-tabs" in CSS
    assert ":focus-visible" in CSS
    assert "min-height: 44px" in CSS
    assert "min-height: 160px" in CSS
    assert "prefers-reduced-motion" in CSS
