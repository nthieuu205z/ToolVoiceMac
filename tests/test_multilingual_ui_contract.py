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
    assert "artifactLabel(artifact)" in JS
    assert "/artifacts/${encodeURIComponent(artifact.id)}" in JS
    assert 'prepare: { label:' in JS
    assert 'export: { label:' in JS
    assert "const TEXT_STAGES" in JS


def test_queue_copy_is_generic_and_job_cards_render_type_and_language():
    assert "JOB QUEUE" in HTML
    assert "HÀNG ĐỢI XỬ LÝ" in HTML
    assert "VIDEO QUEUE" not in HTML
    assert "Hàng đợi video" not in HTML
    assert "jobTypeLabel(job)" in JS
    assert "languageName(job.target_language)" in JS
    assert "jobTypeIcon(job)" in JS
    assert "job-type-icon" in JS


def test_job_type_icons_render_for_queue_and_selected_job():
    assert 'className = `file-icon icon ${jobTypeIcon(job)}`' in JS
    assert ".job-type-icon.icon-sound" in CSS
    assert ".file-icon.icon-sound" in CSS
    assert ".job-card-actions { display: flex; flex-wrap: wrap;" in CSS
    assert ".selected-actions { display: flex; flex-wrap: wrap;" in CSS


def test_text_job_icons_stay_inside_their_queue_and_selected_boxes():
    assert ".job-type-icon.icon-sound::before {" in CSS
    assert ".job-type-icon.icon-sound::after {" in CSS
    assert ".file-icon.icon-sound::before {" in CSS
    assert ".file-icon.icon-sound::after {" in CSS
    assert ".job-type-icon.icon-sound::before, .file-icon.icon-sound::before" not in CSS


def test_selected_job_long_title_wraps_without_overflowing_the_panel():
    assert ".workspace-grid > * { min-width: 0; }" in CSS
    assert ".selected-file > div:last-child { min-width: 0; flex: 1; }" in CSS
    assert "#selectedFilename { display: -webkit-box;" in CSS
    assert "overflow-wrap: anywhere" in CSS
    assert '$("#selectedFilename").title = jobTitle(job);' in JS


def test_graph_stage_definitions_cover_both_job_types():
    assert 'video_dubbing: ["extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux"]' in JS
    assert 'text_to_voice: ["prepare", "synthesize", "assemble", "export"]' in JS
    assert "renderGraphNodes(job)" in JS


def test_done_job_renders_public_artifacts_instead_of_hardcoded_video_links():
    assert "job.artifacts" in JS
    assert "/artifacts/${encodeURIComponent(artifact.id)}" in JS


def test_language_catalog_failure_does_not_block_other_boot_requests():
    assert 'catch (error) { state.languages = [' in JS
    assert 'await loadLanguages(); await Promise.all' not in JS


def test_initial_voice_load_waits_to_sync_until_language_catalog_settles():
    assert "async function loadVoices({ sync = true } = {})" in JS
    assert "if (sync) syncComposerVoices();" in JS
    assert "loadVoices({ sync: false })" in JS
    init = JS.split("(async function init()", 1)[1]
    assert init.index("loadVoices({ sync: false })") < init.index("syncComposerVoices()")


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


def test_voice_options_advertise_their_supported_languages():
    assert "option.dataset.languages = voice.supported_languages.join" in JS


def test_voice_lab_has_a_language_selector_and_preview_retry_contract():
    assert 'id="voiceLabLanguage"' in HTML
    assert "preview_status" in JS
    assert "/preview/regenerate?language=" in JS
    assert "Đang tạo bản nghe thử" in JS
    assert "Tạo lại demo" in JS
    assert "function pollVoicePreview" in JS
    assert "window.setTimeout(() => pollVoicePreview" in JS


def test_voice_lab_is_split_into_two_native_collapsible_panels():
    assert '<details class="voice-disclosure" id="voiceListDisclosure" open>' in HTML
    assert '<summary class="voice-disclosure-summary" id="voiceListToggle">' in HTML
    assert '<details class="voice-disclosure" id="voiceCreateDisclosure">' in HTML
    assert '<summary class="voice-disclosure-summary" id="toggleVoiceForm">' in HTML
    assert 'id="voiceForm" hidden' not in HTML
    assert "voiceCreateDisclosure.open = false" in JS


def test_voice_cards_do_not_reserve_a_fixed_action_column():
    assert ".voice-list { display: grid; grid-template-columns: 1fr;" in CSS
    assert ".voice-item { display: grid; grid-template-columns: 36px minmax(0, 1fr) auto;" in CSS
    assert ".voice-actions { display: flex; flex-wrap: wrap;" in CSS
    assert "width: 193px" not in CSS


def test_on_demand_preview_can_be_stopped_without_starting_another_request():
    assert "if (activePreview?.button === button) { abortTextPreview(); return; }" in JS


def test_on_demand_preview_releases_audio_state_and_object_url_on_every_terminal_path():
    assert "function releaseTextPreview" in JS
    assert "URL.revokeObjectURL(state.textPreviewUrl)" in JS
    assert "let token = null;" in JS
    assert "const finish = () => releaseTextPreview(token);" in JS
    assert "releaseTextPreview(token);" in JS.split("catch (error)", 1)[1]


def test_multilingual_composer_has_accessible_responsive_styles():
    assert ".job-mode-tabs" in CSS
    assert ":focus-visible" in CSS
    assert "min-height: 44px" in CSS
    assert "min-height: 160px" in CSS
    assert "prefers-reduced-motion" in CSS
    assert ".preview-actions .button.small { min-height: 44px; }" in CSS
    assert ".voice-preview, .voice-select-button, .voice-delete, .upload-cancel { min-height: 44px; }" in CSS
    assert ".voice-actions .voice-delete { flex: 0 0 44px; }" in CSS
    assert ".upload-cancel { display: inline-flex; min-height: 44px;" in CSS
