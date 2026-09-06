/* Sub. Ops Console — realtime monitor client. */
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const STAGES_BY_JOB_TYPE = {
  video_dubbing: ["extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux"],
  text_to_voice: ["prepare", "synthesize", "assemble", "export"],
};
const STAGES = STAGES_BY_JOB_TYPE.video_dubbing;
const TEXT_STAGES = STAGES_BY_JOB_TYPE.text_to_voice;
const STAGE_META = {
  prepare: { label: "Chuẩn bị nội dung", detail: "Chia câu và chuẩn hóa văn bản", icon: "icon-language", short: "Văn bản" },
  extract: { label: "Tách âm thanh", detail: "Chuẩn bị audio từ video", icon: "icon-upload", short: "Input video" },
  transcribe: { label: "Nhận diện giọng nói", detail: "Whisper tạo transcript", icon: "icon-wave", short: "Whisper STT" },
  translate: { label: "Dịch thuật", detail: "Gemini chuyển ngữ", icon: "icon-language", short: "Gemini" },
  synthesize: { label: "Tạo giọng đọc", detail: "Dựng batch audio", icon: "icon-sound", short: "Speech" },
  subtitle: { label: "Tạo phụ đề", detail: "Căn cue theo giọng đọc", icon: "icon-caption", short: "Subtitle" },
  assemble: { label: "Ghép âm thanh", detail: "Đặt các segment vào timeline", icon: "icon-layers", short: "Assemble" },
  mux: { label: "Xuất video", detail: "Đóng gói video hoàn tất", icon: "icon-export", short: "Video" },
  export: { label: "Xuất âm thanh", detail: "Đóng gói WAV và MP3", icon: "icon-export", short: "WAV / MP3" },
};
const STAGE_LABELS = Object.fromEntries(Object.entries(STAGE_META).map(([stage, meta]) => [stage, `Đang ${meta.label.toLowerCase()}`]));
const ACTIVE_STATUSES = new Set(["queued", "running", "cancelling"]);
const STATUS_META = {
  queued: { label: "Đang chờ", tone: "queued" }, running: { label: "Đang chạy", tone: "running" },
  cancelling: { label: "Đang dừng", tone: "running" }, cancelled: { label: "Đã hủy", tone: "cancelled" },
  done: { label: "Hoàn tất", tone: "done" }, error: { label: "Lỗi", tone: "error" },
};
const MAX_BYTES = 8 * 1024 ** 3;
const JOB_MODES = ["video", "text"];
const LANGUAGE_RATES = { "vi-VN": 16.6, "en-US": 14.0 };
const state = { voices: [], languages: [], jobs: [], selectedJobId: null, eventSources: new Map(), lastEvents: [], connected: false, refreshing: false, uploadBusy: false, textBusy: false, cloneBusy: false, voiceSource: "", previewToken: 0, graphJobType: "", jobMode: "video", selectedFile: null, textPreviewController: null, textPreviewUrl: "", jobDrafts: { video: { language: localStorage.getItem("sub.video.language") || "vi-VN", voiceId: localStorage.getItem("sub.video.voice") || "" }, text: { language: localStorage.getItem("sub.text.language") || "vi-VN", voiceId: localStorage.getItem("sub.text.voice") || "", text: sessionStorage.getItem("sub.text.draft") || "" } } };

function fmtBytes(bytes) { if (!Number.isFinite(Number(bytes))) return "—"; const value = Number(bytes); if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`; if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`; if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`; return `${value} B`; }
function fmtDuration(seconds) { if (!Number.isFinite(Number(seconds))) return "—"; const total = Math.max(0, Math.round(Number(seconds))); const hours = Math.floor(total / 3600); const minutes = Math.floor((total % 3600) / 60); const secs = total % 60; if (hours) return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`; return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`; }
function fmtEta(seconds) { if (seconds == null || !Number.isFinite(Number(seconds))) return "ETA —"; return Number(seconds) <= 0 ? "Xong" : `ETA ${fmtDuration(seconds)}`; }
function fmtClock(timestamp) { if (!timestamp) return "—"; return new Date(Number(timestamp) * 1000).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
function relativeTime(timestamp) { if (!timestamp) return "—"; const delta = Math.max(0, Date.now() / 1000 - Number(timestamp)); if (delta < 60) return `${Math.round(delta)}s trước`; if (delta < 3600) return `${Math.round(delta / 60)}m trước`; return `${Math.round(delta / 3600)}h trước`; }
function voiceName(id) { return state.voices.find(voice => voice.id === id)?.display_name || id || "Chưa chọn giọng"; }
function voiceInitials(name) { return String(name || "?").trim().split(/\s+/).map(part => part[0]).join("").slice(0, 2).toUpperCase(); }
function stageLabel(stage) { return STAGE_META[stage]?.label || stage || "Đang chờ"; }
function stagesFor(job) { return STAGES_BY_JOB_TYPE[job?.job_type] || STAGES_BY_JOB_TYPE.video_dubbing; }
function jobTypeLabel(job) { return job?.job_type === "text_to_voice" ? "Text → Voice" : "Video Dubbing"; }
function languageName(code) { return state.languages.find(language => language.code === code)?.display_name || code || "—"; }
function jobTitle(job) { return job?.input_label || job?.filename || job?.job_id || "Không tên"; }
function artifactLabel(artifact) { return ({ video: "Video", subtitle: "SRT", wav: "WAV", mp3: "MP3" })[artifact?.kind] || artifact?.kind?.toUpperCase() || "File"; }
function artifactLinks(job, className = "mini-button") { return (job.artifacts || []).map((artifact, index) => `<a class="${className}${index === 0 && className.includes("button") ? " primary" : ""}" href="/api/jobs/${encodeURIComponent(job.job_id)}/artifacts/${encodeURIComponent(artifact.id)}">${escapeHtml(artifactLabel(artifact))}</a>`).join(""); }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char])); }
function escapeAttr(value) { return escapeHtml(value); }
function announce(message) { const status = $("#srStatus"); if (status) status.textContent = message; }
function toast(message, tone = "") { const item = document.createElement("div"); item.className = `toast ${tone}`; item.textContent = message; $("#toastStack").append(item); window.setTimeout(() => item.remove(), 4200); }
function setConnection(connected, message = connected ? "Đã kết nối" : "Mất kết nối") { state.connected = connected; const pill = $("#connectionPill"); pill.classList.toggle("connected", connected); $("#connectionPill .status-dot").className = `status-dot ${connected ? "live" : "error"}`; $("#connectionText").textContent = message; }
async function apiJson(url, options = {}) { const response = await fetch(url, options); const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(body.detail || `Request thất bại (${response.status})`); return body; }

function voicesForLanguage(code) { return state.voices.filter(voice => (voice.supported_languages || []).includes(code)); }
function fillLanguages(select, selected) { select.textContent = ""; state.languages.forEach(language => { const option = document.createElement("option"); option.value = language.code; option.textContent = language.display_name; select.append(option); }); select.value = state.languages.some(language => language.code === selected) ? selected : state.languages[0]?.code || "vi-VN"; }
function fillVoices(select, language, selected) { const voices = voicesForLanguage(language); select.textContent = ""; voices.forEach(voice => { const option = document.createElement("option"); option.value = voice.id; option.textContent = voice.custom ? `${voice.display_name} · Nhân bản` : voice.display_name; select.append(option); }); select.value = voices.some(voice => voice.id === selected) ? selected : voices[0]?.id || ""; return select.value; }
function activateJobMode(mode, { focus = false } = {}) { state.jobMode = JOB_MODES.includes(mode) ? mode : "video"; $$("[role=tab]", $("#jobModeTabs")).forEach(tab => { const active = tab.dataset.mode === state.jobMode; tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1; if (focus && active) tab.focus(); }); $("#videoJobPanel").hidden = state.jobMode !== "video"; $("#textJobPanel").hidden = state.jobMode !== "text"; }
function setupJobTabs() { const tabs = $$("[role=tab]", $("#jobModeTabs")); tabs.forEach((tab, index) => { tab.addEventListener("click", () => activateJobMode(tab.dataset.mode)); tab.addEventListener("keydown", event => { let next = index; if (event.key === "ArrowRight") next = (index + 1) % tabs.length; else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length; else if (event.key === "Home") next = 0; else if (event.key === "End") next = tabs.length - 1; else return; event.preventDefault(); activateJobMode(tabs[next].dataset.mode, { focus: true }); }); }); activateJobMode(state.jobMode); }
async function loadLanguages() { try { state.languages = await apiJson("/api/languages"); } catch (error) { state.languages = [{ code: "vi-VN", display_name: "Tiếng Việt" }, { code: "en-US", display_name: "English (US)" }]; toast("Không tải được catalog ngôn ngữ; đang dùng danh sách mặc định.", "error"); } fillLanguages($("#videoLanguage"), state.jobDrafts.video.language); fillLanguages($("#textLanguage"), state.jobDrafts.text.language); fillLanguages($("#voiceLabLanguage"), localStorage.getItem("sub.voiceLab.language") || "vi-VN"); }
function syncComposerVoices({ announceReset = false } = {}) { const videoLanguage = $("#videoLanguage").value; const textLanguage = $("#textLanguage").value; const oldVideo = state.jobDrafts.video.voiceId; const oldText = state.jobDrafts.text.voiceId; state.jobDrafts.video.language = videoLanguage; state.jobDrafts.text.language = textLanguage; state.jobDrafts.video.voiceId = fillVoices($("#videoVoiceSelect"), videoLanguage, oldVideo); state.jobDrafts.text.voiceId = fillVoices($("#textVoiceSelect"), textLanguage, oldText); localStorage.setItem("sub.video.voice", state.jobDrafts.video.voiceId); localStorage.setItem("sub.text.voice", state.jobDrafts.text.voiceId); if (announceReset && ((oldVideo && oldVideo !== state.jobDrafts.video.voiceId) || (oldText && oldText !== state.jobDrafts.text.voiceId))) announce("Giọng đã được đổi vì không hỗ trợ ngôn ngữ vừa chọn."); updateUploadState(); updateTextState(); }

async function loadVoices() {
  try {
    state.voices = await apiJson("/api/voices");
    state.voiceSource = "api";
    syncComposerVoices();
    renderVoiceLab();
  } catch (error) {
    state.voiceSource = "error";
    $("#videoVoiceSelect").innerHTML = '<option value="">Không tải được danh sách giọng</option>';
    $("#textVoiceSelect").innerHTML = '<option value="">Không tải được danh sách giọng</option>';
    $("#voiceList").innerHTML = '<div class="event-empty">Không tải được danh sách giọng.</div>';
    toast(error.message, "error");
  }
}

function previewUrl(voice, language = "vi-VN") { return voice.preview_urls?.[language] || voice.preview_url || `/api/voices/${encodeURIComponent(voice.id)}/preview?language=${encodeURIComponent(language)}`; }
function renderVoiceLab() {
  const list = $("#voiceList");
  const language = $("#voiceLabLanguage")?.value || "vi-VN";
  list.textContent = "";
  const custom = state.voices.filter(voice => voice.custom && (voice.supported_languages || []).includes(language));
  $("#voiceCount").textContent = `${custom.length} giọng`;
  if (!custom.length) { list.innerHTML = '<div class="event-empty">Chưa có giọng nhân bản hỗ trợ ngôn ngữ này.</div>'; return; }
  custom.forEach(voice => {
    const selected = $("#videoVoiceSelect").value === voice.id || $("#textVoiceSelect").value === voice.id;
    const previewStatus = voice.preview_status?.[language] || "error";
    const item = document.createElement("div"); item.className = `voice-item${selected ? " selected" : ""}`;
    const avatar = document.createElement("span"); avatar.className = "voice-avatar"; avatar.textContent = voiceInitials(voice.display_name);
    const body = document.createElement("div"); body.className = "voice-item-main";
    const name = document.createElement("strong"); name.textContent = voice.display_name; name.title = voice.display_name;
    const id = document.createElement("span"); id.textContent = voice.id; id.title = voice.id;
    const status = document.createElement("small"); status.className = previewStatus === "ready" ? "voice-ready" : "voice-missing"; status.textContent = previewStatus === "pending" ? "Đang tạo bản nghe thử" : previewStatus === "ready" ? "Demo sẵn sàng" : "Chưa có demo";
    body.append(name, id, status); item.append(avatar, body);
    const actions = document.createElement("div"); actions.className = "voice-actions";
    const select = document.createElement("button"); select.type = "button"; select.className = "voice-select-button"; select.textContent = selected ? "Đang chọn" : "Chọn"; select.setAttribute("aria-label", `Chọn giọng ${voice.display_name}`); select.addEventListener("click", () => { const target = state.jobMode === "text" ? $("#textVoiceSelect") : $("#videoVoiceSelect"); target.value = voice.id; target.dispatchEvent(new Event("change")); }); actions.append(select);
    const preview = document.createElement("button"); preview.type = "button"; preview.className = "voice-preview"; preview.innerHTML = `<span class="icon icon-play" aria-hidden="true"></span><span>${previewStatus === "error" ? "Tạo lại demo" : "Nghe thử"}</span>`; preview.setAttribute("aria-label", `${previewStatus === "error" ? "Tạo lại demo" : "Nghe thử"} giọng ${voice.display_name}`); preview.disabled = previewStatus === "pending"; preview.addEventListener("click", () => previewStatus === "error" ? regenerateVoicePreview(voice, language) : togglePreview(voice, preview, language)); actions.append(preview);
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "voice-delete"; remove.textContent = "×"; remove.title = `Xóa ${voice.display_name}`; remove.setAttribute("aria-label", `Xóa giọng ${voice.display_name}`); remove.addEventListener("click", () => deleteVoice(voice)); actions.append(remove); item.append(actions); list.append(item);
  });
}
let activePreview = null;
function resetPreviewButton(button) { if (!button) return; button.classList.remove("playing"); button.innerHTML = '<span class="icon icon-play" aria-hidden="true"></span><span>Nghe thử</span>'; }
function stopActivePreview() { if (!activePreview) return; activePreview.audio.pause(); resetPreviewButton(activePreview.button); activePreview = null; }
function togglePreview(voice, button, language = "vi-VN") {
  if (activePreview?.button === button) { activePreview.audio.pause(); activePreview = null; resetPreviewButton(button); return; }
  if (activePreview) { activePreview.audio.pause(); resetPreviewButton(activePreview.button); }
  const token = ++state.previewToken;
  const audio = new Audio(previewUrl(voice, language)); audio.preload = "metadata";
  activePreview = { audio, button, token };
  button.classList.add("playing"); button.innerHTML = '<span class="icon icon-pause" aria-hidden="true"></span><span>Đang phát</span>';
  const fail = () => { if (activePreview?.token !== token) return; toast("Chưa có file demo cho giọng này.", "error"); activePreview = null; resetPreviewButton(button); };
  audio.addEventListener("error", fail); audio.addEventListener("ended", () => { if (activePreview?.token !== token) return; activePreview = null; resetPreviewButton(button); }); audio.play().catch(fail);
}
async function pollVoicePreview(voiceId, language) {
  try {
    state.voices = await apiJson("/api/voices");
    syncComposerVoices(); renderVoiceLab();
    const voice = state.voices.find(item => item.id === voiceId);
    if (voice?.preview_status?.[language] === "pending") window.setTimeout(() => pollVoicePreview(voiceId, language), 1800);
  } catch (_) { window.setTimeout(() => pollVoicePreview(voiceId, language), 2600); }
}
async function regenerateVoicePreview(voice, language) { voice.preview_status = { ...(voice.preview_status || {}), [language]: "pending" }; renderVoiceLab(); try { await apiJson(`/api/voices/${encodeURIComponent(voice.id)}/preview/regenerate?language=${encodeURIComponent(language)}`, { method: "POST" }); toast("Đang tạo lại bản nghe thử."); window.setTimeout(() => pollVoicePreview(voice.id, language), 1800); } catch (error) { voice.preview_status[language] = "error"; renderVoiceLab(); toast(error.message, "error"); } }
async function deleteVoice(voice) { if (!window.confirm(`Xóa giọng “${voice.display_name}”?`)) return; try { await apiJson(`/api/voices/custom/${encodeURIComponent(voice.id)}`, { method: "DELETE" }); await loadVoices(); toast("Đã xóa giọng nhân bản.", "success"); } catch (error) { toast(error.message, "error"); } }
function setupVoiceLab() {
  const toggle = $("#toggleVoiceForm"); const form = $("#voiceForm");
  $("#voiceLabLanguage").addEventListener("change", event => { localStorage.setItem("sub.voiceLab.language", event.target.value); stopActivePreview(); renderVoiceLab(); });
  toggle.addEventListener("click", () => { form.hidden = !form.hidden; toggle.setAttribute("aria-expanded", String(!form.hidden)); });
  form.addEventListener("submit", async event => { event.preventDefault(); if (state.cloneBusy) return; const name = $("#cloneName").value.trim(); const audio = $("#cloneAudio").files[0]; const hint = $("#voiceFormHint"); if (!name || !audio) { hint.textContent = "Nhập tên và chọn audio mẫu trước khi tạo."; return; } state.cloneBusy = true; $("#createVoiceButton").disabled = true; $("#createVoiceButton").textContent = "Đang tạo…"; hint.textContent = "Đang chuẩn hóa audio và đăng ký giọng local…"; try { const data = new FormData(); data.append("name", name); data.append("audio", audio); const voice = await apiJson("/api/voices/custom", { method: "POST", body: data }); await loadVoices(); const target = state.jobMode === "text" ? $("#textVoiceSelect") : $("#videoVoiceSelect"); target.value = voice.id; target.dispatchEvent(new Event("change")); form.reset(); form.hidden = true; toggle.setAttribute("aria-expanded", "false"); hint.textContent = "Nên dùng audio 3–8 giây, một người nói, ít tạp âm."; toast("Đã tạo giọng nhân bản OmniVoice.", "success"); } catch (error) { hint.textContent = error.message; toast(error.message, "error"); } finally { state.cloneBusy = false; $("#createVoiceButton").disabled = false; $("#createVoiceButton").textContent = "Tạo giọng"; } });
}

async function loadSystemHealth() { const stack = $("#healthStack"); stack.textContent = ""; try { const data = await apiJson("/api/model"); const models = Array.isArray(data.models) ? data.models : []; const rows = models.map(model => ({ label: model.key === "omnivoice" ? "OmniVoice" : model.key === "whisper" ? "Whisper STT" : model.label, status: model.ready ? "Sẵn sàng" : model.status === "downloading" ? "Đang tải" : "Chưa sẵn sàng", tone: model.ready ? "live" : model.status === "error" ? "error" : "warn" })); rows.push({ label: "API server", status: "Online", tone: "live" }); rows.forEach(row => { const item = document.createElement("div"); item.className = "health-row"; item.innerHTML = `<span class="status-dot ${row.tone}"></span><span></span><em></em>`; item.children[1].textContent = row.label; item.children[2].textContent = row.status; stack.append(item); }); } catch (_) { stack.innerHTML = '<div class="health-row"><span class="status-dot error"></span><span>API server</span><em>Offline</em></div>'; } }
function updateMetrics() { const running = state.jobs.filter(job => job.status === "running" || job.status === "cancelling"); const queued = state.jobs.filter(job => job.status === "queued"); const done = state.jobs.filter(job => job.status === "done"); const focus = running[0] || state.jobs.find(job => job.job_id === state.selectedJobId) || null; $("#metricRunning").textContent = running.length; $("#metricQueued").textContent = queued.length; $("#metricDone").textContent = done.length; $("#navActiveCount").textContent = running.length; $("#navJobCount").textContent = state.jobs.length; $("#metricRunningNote").textContent = running.length ? `${stageLabel(focus.stage)} · ${Math.round(focus.percent || 0)}%` : "Không có job hoạt động"; $("#metricDevice").textContent = focus?.device || "—"; $("#metricEngine").textContent = focus?.engine ? `${focus.engine} · batch ${focus.batch_size || 0}` : "Chưa có engine đang chạy"; $("#lastUpdated").textContent = `Cập nhật ${new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`; }
function jobCardMarkup(job) {
  const statusMeta = STATUS_META[job.status] || STATUS_META.queued;
  const percent = Math.round(job.percent || 0);
  const action = ACTIVE_STATUSES.has(job.status) ? `<button class="mini-button danger" type="button" data-action="cancel" data-job-id="${escapeAttr(job.job_id)}">${job.status === "cancelling" ? "Đang dừng…" : "Hủy"}</button>` : "";
  const remove = !ACTIVE_STATUSES.has(job.status) ? `<button class="mini-button danger" type="button" data-action="delete-job" data-job-id="${escapeAttr(job.job_id)}">Xóa</button>` : "";
  const downloads = job.status === "done" ? `${artifactLinks(job)}${remove}` : remove;
  const title = jobTitle(job);
  return { statusMeta, percent, action, downloads, html: `<div class="job-card-top"><div class="job-card-identity"><div class="job-badge-row"><span class="job-type-badge">${escapeHtml(jobTypeLabel(job))}</span><span class="job-language">${escapeHtml(languageName(job.target_language))}</span></div><div class="job-name" title="${escapeAttr(title)}">${escapeHtml(title)}</div><div class="job-voice">${escapeHtml(voiceName(job.voice_id))}</div></div><span class="job-status ${statusMeta.tone}"><span class="status-dot ${statusMeta.tone === "running" ? "active" : statusMeta.tone === "done" ? "live" : statusMeta.tone === "error" ? "error" : "neutral"}"></span>${statusMeta.label}</span></div><div class="job-progress-row"><span>${escapeHtml(job.message || stageLabel(job.stage))}</span><strong>${percent}%</strong></div><div class="job-progress"><div class="progress-track"><span style="width:${Math.max(0, Math.min(100, Number(job.percent) || 0))}%"></span></div></div><div class="job-card-foot"><span>${escapeHtml(stageLabel(job.stage))}</span><span class="mono">${fmtDuration(job.elapsed_seconds)} · ${relativeTime(job.created_at)}</span></div>${(action || downloads) ? `<div class="job-card-actions">${action}${downloads}</div>` : ""}` };
}
function updateJobCard(card, job, {rebuild = false} = {}) {
  const { statusMeta, percent, html } = jobCardMarkup(job);
  if (rebuild || card.dataset.status !== job.status || card.dataset.stage !== job.stage || card.dataset.jobType !== job.job_type) card.innerHTML = html;
  else { const progress = card.querySelector(".progress-track > span"); card.querySelector(".job-progress-row span").textContent = job.message || stageLabel(job.stage); card.querySelector(".job-progress-row strong").textContent = `${percent}%`; progress.style.width = `${Math.max(0, Math.min(100, Number(job.percent) || 0))}%`; card.querySelector(".job-card-foot > span").textContent = stageLabel(job.stage); }
  card.dataset.status = job.status; card.dataset.stage = job.stage || ""; card.dataset.jobType = job.job_type || "video_dubbing"; card.className = `job-card ${job.job_id === state.selectedJobId ? "selected" : ""} ${ACTIVE_STATUSES.has(job.status) ? "is-active" : ""}`; card.setAttribute("aria-label", `${jobTypeLabel(job)}, ${jobTitle(job)}, ${statusMeta.label}, ${percent}%`);
}
function bindJobCard(card, job) { card.dataset.jobId = job.job_id; card.setAttribute("role", "button"); card.setAttribute("tabindex", "0"); card.addEventListener("click", event => { if (event.target.closest("button, a")) return; selectJob(job.job_id); }); card.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectJob(job.job_id); } }); }
function syncJobListViewport() {
  const list = $("#jobList");
  if (!list) return;
  const cards = $$(".job-card", list);
  if (!cards.length) { list.style.maxHeight = ""; return; }
  const visibleCards = cards.slice(0, 3);
  const gap = Number.parseFloat(getComputedStyle(list).rowGap) || 0;
  const height = visibleCards.reduce((total, card) => total + card.getBoundingClientRect().height, 0) + gap * Math.max(0, visibleCards.length - 1);
  list.style.maxHeight = `${Math.ceil(height)}px`;
}
function renderJobs() {
  const list = $("#jobList"); const existing = new Map($$(".job-card", list).map(card => [card.dataset.jobId, card])); const seen = new Set(); $("#jobsEmpty").hidden = state.jobs.length > 0;
  state.jobs.forEach(job => { let card = existing.get(job.job_id); if (!card) { card = document.createElement("article"); bindJobCard(card, job); list.append(card); } updateJobCard(card, job, {rebuild: !existing.has(job.job_id)}); seen.add(job.job_id); });
  existing.forEach((card, jobId) => { if (!seen.has(jobId)) card.remove(); }); syncJobListViewport(); updateMetrics();
}
async function loadGeminiSettings() {
  try {
    const data = await apiJson("/api/settings/gemini");
    $("#geminiBackend").value = data.backend || "developer";
    $("#geminiSettingsStatus").classList.toggle("configured", Boolean(data.configured));
    $("#geminiSettingsStatus").textContent = data.configured ? "Đã cấu hình" : "Chưa cấu hình";
  } catch (_) {}
}
function setupGeminiSettings() {
  const form = $("#geminiSettingsForm");
  if (!form) return;
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const apiKey = $("#geminiApiKey").value.trim();
    const hint = $("#geminiSettingsHint");
    if (!apiKey) { hint.textContent = "Hãy điền Gemini API key trước khi lưu."; return; }
    const button = form.querySelector("button[type=\"submit\"]");
    button.disabled = true;
    hint.textContent = "Đang lưu cấu hình local…";
    try {
      const data = await apiJson("/api/settings/gemini", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: apiKey, backend: $("#geminiBackend").value }) });
      form.reset();
      $("#geminiBackend").value = data.backend || "developer";
      $("#geminiSettingsStatus").classList.toggle("configured", Boolean(data.configured));
      $("#geminiSettingsStatus").textContent = data.configured ? "Đã cấu hình" : "Chưa cấu hình";
      hint.textContent = "Đã lưu vào .env local. Key không được hiển thị lại.";
      toast("Đã lưu cấu hình Gemini.", "success");
    } catch (error) { hint.textContent = error.message; toast(error.message, "error"); }
    finally { button.disabled = false; }
  });
  loadGeminiSettings();
}
async function deleteJob(job) {
  if (ACTIVE_STATUSES.has(job.status)) { toast("Không thể xóa job đang chạy.", "error"); return; }
  if (!window.confirm(`Xóa job “${jobTitle(job)}” và toàn bộ file kết quả?`)) return;
  try {
    await apiJson(`/api/jobs/${encodeURIComponent(job.job_id)}`, { method: "DELETE" });
    if (state.selectedJobId === job.job_id) {
      state.selectedJobId = null;
      localStorage.removeItem("sub.job");
    }
    await refreshJobs();
    renderSelectedJob();
    renderEvents(state.jobs.find(item => item.job_id === state.selectedJobId) || null);
    toast("Đã xóa job và file kết quả.", "success");
  } catch (error) { toast(error.message, "error"); }
}
function setupShutdown() {
  const button = $("#shutdownButton");
  if (!button) return;
  button.addEventListener("click", async () => {
    if (!window.confirm("Tắt ToolVietSub và dừng các tiến trình do tool khởi chạy?")) return;
    button.disabled = true;
    try { await apiJson("/api/shutdown", { method: "POST" }); button.textContent = "Đang tắt…"; }
    catch (error) { button.disabled = false; toast(error.message, "error"); }
  });
}
function renderSelectedJob() {
  const job = state.jobs.find(item => item.job_id === state.selectedJobId) || null;
  const empty = $("#selectedEmpty");
  const content = $("#selectedContent");
  if (!job) {
    empty.hidden = false;
    content.hidden = true;
    $("#selectedLiveLabel").innerHTML = '<span class="status-dot neutral"></span> IDLE';
    renderGraph(null);
    return;
  }
  empty.hidden = true;
  content.hidden = false;
  const meta = STATUS_META[job.status] || STATUS_META.queued;
  $("#selectedLiveLabel").innerHTML = `<span class="status-dot ${ACTIVE_STATUSES.has(job.status) ? "active" : meta.tone === "done" ? "live" : meta.tone === "error" ? "error" : "neutral"}"></span> ${meta.label.toUpperCase()}`;
  $("#selectedFilename").textContent = jobTitle(job);
  $("#selectedJobType").textContent = jobTypeLabel(job);
  $("#selectedLanguage").textContent = languageName(job.target_language);
  $("#selectedFileIcon").className = `file-icon ${job.job_type === "text_to_voice" ? "icon-sound" : "icon-film"}`;
  $("#selectedVoice").textContent = `Giọng: ${voiceName(job.voice_id)}`;
  $("#selectedStage").textContent = stageLabel(job.stage);
  $("#selectedPercent").textContent = `${Math.round(job.percent || 0)}%`;
  $("#selectedMessage").textContent = job.message || STAGE_META[job.stage]?.detail || "—";
  $("#selectedEta").textContent = fmtEta(job.eta_seconds);
  $("#selectedElapsed").textContent = fmtDuration(job.elapsed_seconds);
  $("#selectedStageElapsed").textContent = fmtDuration(job.stage_elapsed_seconds);
  $("#selectedSegments").textContent = job.attempted_count ? `${job.spoken_count || 0}/${job.attempted_count}` : "—";
  $("#selectedSegments").previousElementSibling.textContent = job.job_type === "text_to_voice" ? "CHUNKS" : "SEGMENTS";
  $("#selectedEngine").textContent = job.engine || "—";
  $("#selectedDevice").textContent = job.device || "—";
  $("#selectedBatch").textContent = job.batch_size || "—";
  const actions = $("#selectedActions");
  actions.textContent = "";
  if (ACTIVE_STATUSES.has(job.status)) {
    const cancel = document.createElement("button");
    cancel.className = "button danger small";
    cancel.type = "button";
    cancel.textContent = job.status === "cancelling" ? "Đang dừng…" : "Hủy job";
    cancel.disabled = job.status === "cancelling";
    cancel.addEventListener("click", () => cancelJob(job.job_id));
    actions.append(cancel);
  }
  if (job.status === "done") {
    (job.artifacts || []).forEach((artifact, index) => { const link = document.createElement("a"); link.className = `button${index === 0 ? " primary" : ""} small`; link.href = `/api/jobs/${encodeURIComponent(job.job_id)}/artifacts/${encodeURIComponent(artifact.id)}`; link.textContent = `Tải ${artifactLabel(artifact)}`; actions.append(link); });
  }
  renderGraph(job);
}
function renderGraphNodes(job) {
  const type = job?.job_type || "video_dubbing";
  if (state.graphJobType === type && $("#pipelineGraph").children.length) return;
  const track = $("#pipelineGraph"); const stages = stagesFor(job);
  track.textContent = ""; track.style.setProperty("--graph-columns", stages.length); track.setAttribute("aria-label", `Các bước ${jobTypeLabel(job)}`);
  stages.forEach((stage, index) => {
    const meta = STAGE_META[stage]; const step = document.createElement("div"); step.className = "graph-step"; step.setAttribute("role", "listitem");
    const node = document.createElement("div"); node.className = "graph-node"; node.dataset.stage = stage;
    const number = document.createElement("span"); number.className = "node-index"; number.textContent = String(index + 1).padStart(2, "0");
    const icon = document.createElement("span"); icon.className = `node-icon ${meta.icon}`; icon.setAttribute("aria-hidden", "true");
    const heading = document.createElement("strong"); heading.textContent = meta.label;
    const detail = document.createElement("small"); detail.textContent = meta.short;
    node.append(number, icon, heading, detail); step.append(node);
    if (index < stages.length - 1) { const rail = document.createElement("div"); rail.className = "graph-rail"; rail.setAttribute("aria-hidden", "true"); const link = document.createElement("span"); link.className = "graph-link"; link.dataset.flowIndex = index; const flow = document.createElement("span"); flow.className = "graph-flow"; link.append(flow); rail.append(link); step.append(rail); }
    track.append(step);
  });
  state.graphJobType = type;
}
function renderGraph(job) {
  renderGraphNodes(job);
  const stages = stagesFor(job); const currentIndex = job ? stages.indexOf(job.stage) : -1; const knownStage = currentIndex >= 0;
  $$(".graph-node").forEach((node, index) => {
    const completed = Boolean(job) && (job.status === "done" || knownStage && index < currentIndex);
    node.classList.toggle("completed", completed);
    node.classList.toggle("active", Boolean(job) && knownStage && index === currentIndex && ACTIVE_STATUSES.has(job.status));
    node.classList.toggle("failed", Boolean(job) && knownStage && index === currentIndex && job.status === "error");
    node.setAttribute("aria-current", Boolean(job) && knownStage && index === currentIndex ? "step" : "false");
  });
  const title = $("#graphDetailTitle"); const text = $("#graphDetailText"); const timer = $("#graphDetailTime");
  if (!job) { title.textContent = "Chưa có job đang chạy"; text.textContent = "Sơ đồ sẽ phát sáng theo từng bước khi pipeline hoạt động."; timer.textContent = "—"; updateGraphFlow(null); return; }
  title.textContent = knownStage ? `${stageLabel(job.stage)} · ${Math.round(job.percent || 0)}%` : `Trạng thái đã lưu · ${Math.round(job.percent || 0)}%`;
  text.textContent = knownStage ? job.message || STAGE_META[job.stage]?.detail || "Đang xử lý" : job.message || "Stage này không còn trong phiên bản hiện tại.";
  timer.textContent = fmtDuration(job.stage_elapsed_seconds);
  const ttsNode = $(".graph-node[data-stage=\"synthesize\"] small"); if (ttsNode) ttsNode.textContent = job.engine ? `${job.engine} · ${job.device || "local"}` : STAGE_META.synthesize.short;
  updateGraphFlow(knownStage ? job : null);
}
function syncGraphScrollAffordance() {
  const shell = $(".graph-scroll-shell");
  const track = $("#pipelineGraph");
  if (!shell || !track) return;
  const scrollable = track.scrollWidth > shell.clientWidth + 1;
  shell.classList.toggle("is-scrollable", scrollable);
  shell.classList.toggle("is-at-end", scrollable && track.scrollLeft + track.clientWidth >= track.scrollWidth - 1);
}
function updateGraphFlow(job) {
  const stages = stagesFor(job); const currentIndex = job ? stages.indexOf(job.stage) : -1;
  const flowIndex = currentIndex - 1;
  $$(".graph-link").forEach((link, index) => {
    const complete = Boolean(job) && (job.status === "done" || index < flowIndex);
    const active = Boolean(job) && index === flowIndex && ACTIVE_STATUSES.has(job.status);
    link.classList.toggle("flow-complete", complete);
    link.classList.toggle("flow-active", active);
    link.setAttribute("aria-label", active ? `Đang truyền từ ${stageLabel(stages[index])} sang ${stageLabel(stages[index + 1])}` : complete ? "Đã truyền xong" : "Đang chờ");
  });
  syncGraphScrollAffordance();
}
function renderEvents(job) { const list = $("#eventList"); list.textContent = ""; const events = (job?.events || state.lastEvents || []).slice(-36).reverse(); if (!events.length) { list.innerHTML = '<div class="event-empty">Chưa có sự kiện mới.</div>'; return; } events.forEach(event => { const row = document.createElement("div"); const tone = event.kind === "finished" && event.status === "error" ? "error" : event.kind === "finished" || event.kind === "status" && event.status === "done" ? "complete" : ""; row.className = `event-row ${tone}`; const heading = event.kind === "started" ? "Bắt đầu job" : event.kind === "finished" ? `Job ${STATUS_META[event.status]?.label?.toLowerCase() || event.status}` : event.kind === "status" ? (event.message || event.status) : stageLabel(event.stage); row.innerHTML = `<time class="event-time"></time><span class="event-marker"></span><div class="event-copy"><strong></strong><span></span></div>`; row.querySelector(".event-time").textContent = fmtClock(event.at); row.querySelector("strong").textContent = heading; row.querySelector(".event-copy span").textContent = event.message || (event.percent != null ? `${event.percent}%` : ""); list.append(row); }); }
function selectJob(jobId) { state.selectedJobId = jobId; localStorage.setItem("sub.job", jobId); renderJobs(); renderSelectedJob(); const job = state.jobs.find(item => item.job_id === jobId); renderEvents(job); ensureJobStream(jobId); }
async function refreshJobs() { if (state.refreshing) return; state.refreshing = true; try { const data = await apiJson("/api/jobs"); state.jobs = Array.isArray(data.jobs) ? data.jobs : []; setConnection(true); const saved = localStorage.getItem("sub.job"); const selectedExists = state.jobs.some(job => job.job_id === state.selectedJobId); if (!selectedExists) state.selectedJobId = state.jobs.some(job => job.job_id === saved) ? saved : state.jobs[0]?.job_id || null; renderJobs(); const selected = state.jobs.find(job => job.job_id === state.selectedJobId); renderSelectedJob(); renderEvents(selected); state.jobs.filter(job => ACTIVE_STATUSES.has(job.status)).forEach(job => ensureJobStream(job.job_id)); } catch (_) { setConnection(false, "API offline"); } finally { state.refreshing = false; } }
function ensureJobStream(jobId) { if (state.eventSources.has(jobId)) return; const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`); state.eventSources.set(jobId, source); source.onmessage = event => { try { const snapshot = JSON.parse(event.data); const index = state.jobs.findIndex(job => job.job_id === snapshot.job_id); if (index >= 0) state.jobs[index] = { ...state.jobs[index], ...snapshot }; else state.jobs.unshift(snapshot); if (snapshot.job_id === state.selectedJobId) { renderSelectedJob(); renderEvents(snapshot); } renderJobs(); if (!ACTIVE_STATUSES.has(snapshot.status)) { source.close(); state.eventSources.delete(jobId); if (snapshot.status === "done") toast(`${jobTypeLabel(snapshot)} đã xử lý xong.`, "success"); if (snapshot.status === "error") toast(snapshot.message || "Job gặp lỗi.", "error"); } } catch (_) {} }; source.onerror = () => { source.close(); state.eventSources.delete(jobId); window.setTimeout(() => { const current = state.jobs.find(job => job.job_id === jobId); if (current && ACTIVE_STATUSES.has(current.status)) ensureJobStream(jobId); }, 1800); }; }
async function cancelJob(jobId) { try { await apiJson(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" }); toast("Đã gửi yêu cầu dừng job."); } catch (error) { toast(error.message, "error"); } }
let uploadRequest = null;
function setupUpload() {
  const input = $("#videoInput"); const dropzone = $("#dropzone");
  input.addEventListener("change", () => setUploadFile(input.files[0] || null));
  ["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", event => setUploadFile([...event.dataTransfer.files].find(file => isVideoFile(file)) || null));
  dropzone.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } });
  $("#videoLanguage").addEventListener("change", event => { state.jobDrafts.video.language = event.target.value; localStorage.setItem("sub.video.language", event.target.value); stopActivePreview(); syncComposerVoices({ announceReset: true }); });
  $("#videoVoiceSelect").addEventListener("change", event => { state.jobDrafts.video.voiceId = event.target.value; localStorage.setItem("sub.video.voice", event.target.value); stopActivePreview(); updateUploadState(); renderVoiceLab(); });
  $("#uploadForm").addEventListener("submit", submitUpload);
  $("#cancelUploadButton").addEventListener("click", () => { if (uploadRequest) uploadRequest.abort(); });
  $("[data-action=\"open-upload\"]").addEventListener("click", () => { $("#uploadPanel").scrollIntoView({ behavior: "smooth", block: "center" }); });
  $("#newJobButton").addEventListener("click", () => { $("#uploadPanel").scrollIntoView({ behavior: "smooth", block: "center" }); });
}
function isVideoFile(file) { return Boolean(file && (file.type.startsWith("video/") || /\.(mp4|mkv|mov|avi|webm)$/i.test(file.name))); }
function showUploadError(message, focus = false) { const box = $("#uploadError"); $("#uploadErrorText").textContent = message; box.hidden = false; if (focus) box.focus(); }
function clearUploadError() { $("#uploadError").hidden = true; $("#uploadErrorText").textContent = ""; }
function setUploadFile(file) {
  clearUploadError();
  state.selectedFile = isVideoFile(file) ? file : null;
  const selected = $("#selectedUpload");
  if (file && !state.selectedFile) showUploadError("File này không phải định dạng video được hỗ trợ.", true);
  if (file && file.size > MAX_BYTES) { state.selectedFile = null; showUploadError("Video vượt quá giới hạn 8 GB.", true); }
  $("#dropTitle").textContent = state.selectedFile ? "Video đã sẵn sàng" : "Kéo thả video vào đây";
  $("#dropHint").textContent = state.selectedFile ? `${fmtBytes(state.selectedFile.size)} · sẵn sàng tải lên` : "hoặc bấm để chọn · MP4, MKV, MOV, AVI, WebM · tối đa 8 GB";
  selected.hidden = !state.selectedFile; selected.textContent = state.selectedFile?.name || ""; updateUploadState();
}
function updateUploadState() {
  const ready = Boolean(state.selectedFile && $("#videoVoiceSelect").value && !state.uploadBusy);
  const button = $("#startButton");
  button.disabled = !ready;
  button.setAttribute("aria-disabled", String(!ready));
  const hint = state.uploadBusy ? "Đang tải video lên server local…" : ready ? "Sẵn sàng — pipeline sẽ xuất hiện ở hàng đợi." : "Chọn video và giọng đọc để bắt đầu.";
  $("#uploadHint").textContent = hint;
  announce(hint);
}
function submitUpload(event) {
  event.preventDefault();
  clearUploadError();
  const voiceId = $("#videoVoiceSelect").value;
  if (!state.selectedFile || !voiceId || state.uploadBusy) {
    showUploadError(!state.selectedFile ? "Hãy chọn một video trước." : "Hãy chọn giọng đọc trước.", true);
    return;
  }
  state.uploadBusy = true;
  updateUploadState();
  $("#uploadProgress").hidden = false;
  $("#uploadBar").style.width = "0%";
  $("#uploadPercent").textContent = "0%";
  $("#uploadProgressLabel").textContent = "Đang tải video lên";

  const xhr = new XMLHttpRequest();
  uploadRequest = xhr;
  xhr.open("POST", "/api/jobs");
  xhr.timeout = 120000;
  const form = new FormData();
  form.append("video", state.selectedFile);
  form.append("filename", state.selectedFile.name);
  form.append("voice_id", voiceId);
  form.append("target_language", $("#videoLanguage").value);
  xhr.upload.onprogress = event => {
    if (!event.lengthComputable) return;
    const pct = event.loaded / event.total * 100;
    $("#uploadBar").style.width = `${pct}%`;
    $("#uploadPercent").textContent = `${Math.round(pct)}%`;
  };
  const reset = () => {
    if (uploadRequest === xhr) uploadRequest = null;
    state.uploadBusy = false;
    $("#uploadProgress").hidden = true;
    updateUploadState();
  };
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) {}
      reset();
      state.selectedFile = null;
      $("#videoInput").value = "";
      setUploadFile(null);
      refreshJobs();
      if (body.job_id) selectJob(body.job_id);
      toast("Đã thêm video vào hàng đợi.", "success");
      return;
    }
    let message = "Không tải được video.";
    try { message = JSON.parse(xhr.responseText).detail || message; } catch (_) {}
    reset();
    showUploadError(message, true);
    toast(message, "error");
  };
  xhr.onerror = () => { reset(); showUploadError("Mất kết nối khi tải video. Kiểm tra server rồi thử lại.", true); toast("Mất kết nối khi tải video.", "error"); };
  xhr.ontimeout = () => { reset(); showUploadError("Tải video quá thời gian chờ. Hãy kiểm tra server và thử lại.", true); toast("Tải video quá thời gian chờ.", "error"); };
  xhr.onabort = () => { reset(); showUploadError("Đã hủy tải video. Bạn có thể chọn lại file để thử lại.", true); toast("Đã hủy tải video.", "error"); };
  xhr.send(form);
}

function showTextError(message, focus = false) { const box = $("#textJobError"); $("#textJobErrorText").textContent = message; box.hidden = false; if (focus) box.focus(); }
function clearTextError() { $("#textJobError").hidden = true; $("#textJobErrorText").textContent = ""; }
function abortTextPreview() { if (state.textPreviewController) state.textPreviewController.abort(); state.textPreviewController = null; stopActivePreview(); if (state.textPreviewUrl) URL.revokeObjectURL(state.textPreviewUrl); state.textPreviewUrl = ""; if ($("#textPreviewButton")) { $("#textPreviewButton").disabled = false; $("#textPreviewButton").textContent = "Nghe thử nội dung"; } }
function updateTextState() {
  const text = $("#textInput").value;
  const trimmed = text.trim();
  const language = $("#textLanguage").value || "vi-VN";
  const seconds = trimmed ? Math.ceil(trimmed.length / (LANGUAGE_RATES[language] || LANGUAGE_RATES["vi-VN"])) : 0;
  $("#textCharacterCount").textContent = `${text.length.toLocaleString("vi-VN")} / 50.000 ký tự`;
  $("#textDurationEstimate").textContent = `Ước tính ${fmtDuration(seconds)}`;
  const ready = Boolean(trimmed && $("#textVoiceSelect").value);
  $("#textStartButton").disabled = !ready || state.textBusy;
  $("#textFixedPreviewButton").disabled = !$("#textVoiceSelect").value;
  $("#textPreviewButton").disabled = !ready;
}
function selectedTextVoice() { return state.voices.find(voice => voice.id === $("#textVoiceSelect").value) || null; }
function playTextFixedPreview() { const voice = selectedTextVoice(); if (!voice) return; togglePreview(voice, $("#textFixedPreviewButton"), $("#textLanguage").value); }
async function previewText() {
  const text = $("#textInput").value.trim(); const voiceId = $("#textVoiceSelect").value; const language = $("#textLanguage").value;
  if (!text || !voiceId) return;
  const button = $("#textPreviewButton");
  if (activePreview?.button === button) { abortTextPreview(); return; }
  abortTextPreview(); clearTextError();
  const controller = new AbortController(); state.textPreviewController = controller; button.disabled = true; button.textContent = "Đang tạo bản nghe thử…"; announce("Đang tạo bản nghe thử nội dung.");
  try {
    const response = await fetch(`/api/voices/${encodeURIComponent(voiceId)}/preview-text`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, language }), signal: controller.signal });
    if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Request thất bại (${response.status})`); }
    const blob = await response.blob(); if (controller.signal.aborted) return;
    state.textPreviewUrl = URL.createObjectURL(blob); const audio = new Audio(state.textPreviewUrl); const token = ++state.previewToken; activePreview = { audio, button, token }; button.textContent = "Dừng nghe thử"; announce("Đang phát bản nghe thử nội dung.");
    const finish = () => { if (activePreview?.token !== token) return; activePreview = null; button.textContent = "Nghe thử nội dung"; updateTextState(); announce("Đã dừng bản nghe thử nội dung."); };
    audio.addEventListener("ended", finish); audio.addEventListener("error", finish); await audio.play();
  } catch (error) { if (error.name !== "AbortError") { showTextError(error.message, true); button.textContent = "Thử lại nghe thử"; toast(error.message, "error"); } }
  finally { if (state.textPreviewController === controller) state.textPreviewController = null; updateTextState(); }
}
async function submitTextJob(event) {
  event.preventDefault(); clearTextError();
  const text = $("#textInput").value.trim(); const voiceId = $("#textVoiceSelect").value; const language = $("#textLanguage").value;
  if (!text || !voiceId || state.textBusy) { showTextError(!text ? "Hãy nhập nội dung cần đọc." : "Hãy chọn giọng đọc trước.", true); return; }
  abortTextPreview(); state.textBusy = true; updateTextState(); $("#textStartButton").textContent = "Đang thêm…";
  try {
    const body = await apiJson("/api/jobs/text", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, voice_id: voiceId, language }) });
    $("#textInput").value = ""; state.jobDrafts.text.text = ""; sessionStorage.removeItem("sub.text.draft"); await refreshJobs(); if (body.job_id) selectJob(body.job_id); announce("Đã thêm job giọng đọc vào hàng đợi."); toast("Đã thêm job giọng đọc vào hàng đợi.", "success");
  } catch (error) { showTextError(error.message, true); toast(error.message, "error"); }
  finally { state.textBusy = false; $("#textStartButton").textContent = "Tạo giọng đọc"; updateTextState(); }
}
function setupTextComposer() {
  $("#textInput").value = state.jobDrafts.text.text;
  $("#textInput").addEventListener("input", event => { state.jobDrafts.text.text = event.target.value; sessionStorage.setItem("sub.text.draft", event.target.value); abortTextPreview(); updateTextState(); });
  $("#textLanguage").addEventListener("change", event => { state.jobDrafts.text.language = event.target.value; localStorage.setItem("sub.text.language", event.target.value); abortTextPreview(); syncComposerVoices({ announceReset: true }); });
  $("#textVoiceSelect").addEventListener("change", event => { state.jobDrafts.text.voiceId = event.target.value; localStorage.setItem("sub.text.voice", event.target.value); abortTextPreview(); updateTextState(); renderVoiceLab(); });
  $("#textFixedPreviewButton").addEventListener("click", playTextFixedPreview);
  $("#textPreviewButton").addEventListener("click", previewText);
  $("#textJobForm").addEventListener("submit", submitTextJob);
  updateTextState();
}

$("#refreshButton").addEventListener("click", () => { loadSystemHealth(); loadVoices(); refreshJobs(); loadGeminiSettings(); });
$("#clearEvents").addEventListener("click", () => { state.lastEvents = []; renderEvents(null); });
document.addEventListener("click", event => { const button = event.target.closest("[data-action=\"cancel\"]"); if (button) { event.stopPropagation(); cancelJob(button.dataset.jobId); return; } const remove = event.target.closest("[data-action=\"delete-job\"]"); if (remove) { event.stopPropagation(); const job = state.jobs.find(item => item.job_id === remove.dataset.jobId); if (job) deleteJob(job); } });
(async function init() { setupJobTabs(); setupUpload(); setupTextComposer(); setupVoiceLab(); setupGeminiSettings(); setupShutdown(); const graphTrack = $("#pipelineGraph"); window.addEventListener("resize", syncGraphScrollAffordance); window.addEventListener("resize", syncJobListViewport); graphTrack?.addEventListener("scroll", syncGraphScrollAffordance, { passive: true }); await Promise.all([loadLanguages(), loadVoices(), loadSystemHealth(), refreshJobs()]); syncComposerVoices(); syncGraphScrollAffordance(); window.setInterval(() => { const job = state.jobs.find(item => item.job_id === state.selectedJobId); if (job && ACTIVE_STATUSES.has(job.status)) { renderSelectedJob(); updateMetrics(); } }, 1000); })();
