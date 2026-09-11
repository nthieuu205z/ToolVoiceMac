/* Workspace navigation, voice metadata and bounded project progress views. */
"use strict";
window.createWorkspaceFeatures = function ({ apiJson, loadVoices, toast, escapeHtml, languageName, syncJobListViewport = () => {} }) {
  const find = selector => document.querySelector(selector);
  let editingVoice = null;
  let projectId = null;
  let projectRequest = null;
  let projectRefreshAt = 0;
  let projectStatus = null;
  let projectSelection = 0;
  let latestProjectJob = null;

  function parseTags(value) {
    const tags = [...new Set(value.split(",").map(tag => tag.trim()).filter(Boolean))];
    if (tags.length > 8 || tags.some(tag => [...tag].length > 24)) throw new Error("Tối đa 8 nhãn, mỗi nhãn tối đa 24 ký tự.");
    return tags;
  }

  function fitVoiceList() {
    const list = find("#voiceList");
    const cards = [...list.children].filter(item => item.classList.contains("voice-item"));
    if (list.getBoundingClientRect().width === 0) return;
    const style = getComputedStyle(list);
    const gap = parseFloat(style.rowGap) || 0;
    const padding = (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0);
    const height = cards.slice(0, 6).reduce((sum, card) => sum + card.getBoundingClientRect().height, 0) + gap * Math.max(0, Math.min(6, cards.length) - 1) + padding;
    list.style.maxHeight = cards.length > 6 ? `${Math.ceil(height)}px` : "none";
  }

  function navigate() {
    const settings = location.hash === "#settings";
    find("#dashboardView").hidden = settings;
    find("#settings").hidden = !settings;
    find("#workspaceTitle").textContent = settings ? "Cấu hình workspace" : "Phòng điều khiển";
    find("#workspaceCrumb").textContent = settings ? "RUNTIME SETTINGS" : "TỔNG QUAN";
    document.querySelectorAll(".nav-item").forEach(link => {
      const active = link.getAttribute("href") === (location.hash || "#overview");
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
    });
    if (!settings) requestAnimationFrame(() => { fitVoiceList(); syncJobListViewport(); });
  }

  function editVoice(voice) {
    editingVoice = voice;
    find("#editVoiceName").value = voice.name || voice.display_name.replace(/ — giọng nhân bản$/, "");
    const select = find("#editVoiceLanguage");
    const legacy = voice.supported_languages.length > 1;
    select.querySelector('[value=""]').hidden = !legacy;
    select.value = legacy ? "" : voice.supported_languages[0];
    find("#editVoiceTags").value = (voice.tags || []).join(", ");
    find("#voiceEditError").textContent = "";
    find("#voiceEditDialog").showModal();
    find("#editVoiceName").focus();
  }

  function renderProject(job) {
    const panel = find("#speechProject");
    panel.hidden = !job?.is_project;
    if (!job?.is_project) { projectId = null; latestProjectJob = null; projectSelection++; return; }
    latestProjectJob = { job_id: job.job_id, is_project: true, status: job.status };
    const changed = projectId !== job.job_id;
    if (changed) {
      projectId = job.job_id;
      projectSelection++;
      projectRefreshAt = 0;
      projectStatus = null;
      find("#projectParts").textContent = "Đang tải các phần…";
      find("#projectSummary").textContent = "";
    }
    if ((projectRequest?.id === job.job_id && projectRequest.selection === projectSelection) || (!changed && projectStatus === job.status && Date.now() < projectRefreshAt)) return;
    const requestedId = job.job_id;
    const request = { id: requestedId, selection: projectSelection, status: job.status };
    projectRequest = request;
    projectRefreshAt = Date.now() + 2500;
    projectStatus = job.status;
    apiJson(`/api/jobs/${encodeURIComponent(requestedId)}/project`).then(project => {
      if (projectId !== requestedId || projectSelection !== request.selection) return;
      find("#projectSummary").textContent = `${project.completed_parts}/${project.total_parts} phần hoàn tất · ${Number(project.total_characters).toLocaleString("vi-VN")} ký tự`;
      const labels = { pending: "Chờ", queued: "Chờ", running: "Đang tạo", done: "Hoàn tất", error: "Lỗi", cancelled: "Đã hủy" };
      find("#projectParts").innerHTML = project.parts.map(part => `<li class="project-part"><span class="part-index">${String(part.index).padStart(3, "0")}</span><span class="part-copy"><strong>${escapeHtml(part.title)}</strong><small>${Number(part.characters).toLocaleString("vi-VN")} ký tự${part.error ? ` · ${escapeHtml(part.error)}` : ""}</small></span><span class="part-status ${escapeHtml(part.status)}">${escapeHtml(labels[part.status] || part.status)}</span></li>`).join("");
    }).catch(error => { if (projectId === requestedId && projectSelection === request.selection) find("#projectSummary").textContent = error.message; })
      .finally(() => {
        if (projectRequest !== request) return;
        projectRequest = null;
        // Coalesce a status change received during the request into one refresh.
        // Terminal jobs stop periodic rendering, so this handoff must fetch it.
        if (projectSelection === request.selection && latestProjectJob?.status !== request.status && latestProjectJob) renderProject(latestProjectJob);
      });
  }

  function setup() {
    window.addEventListener("hashchange", navigate);
    window.addEventListener("resize", fitVoiceList);
    find("#voiceListDisclosure").addEventListener("toggle", fitVoiceList);
    find("#voiceEditCancel").addEventListener("click", () => find("#voiceEditDialog").close());
    find("#voiceEditForm").addEventListener("submit", async event => {
      event.preventDefault();
      const button = find("#voiceEditSave");
      if (button.disabled || !editingVoice) return;
      try {
        const name = find("#editVoiceName").value.trim();
        if (!name) throw new Error("Hãy nhập tên giọng.");
        const language = find("#editVoiceLanguage").value;
        const tags = parseTags(find("#editVoiceTags").value);
        button.disabled = true;
        await apiJson(`/api/voices/custom/${encodeURIComponent(editingVoice.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, tags, ...(language ? { language } : {}) }) });
        if (language) find("#voiceLabLanguage").value = language;
        await loadVoices();
        find("#voiceEditDialog").close();
        toast("Đã lưu tên, ngôn ngữ và nhãn giọng.", "success");
      } catch (error) { find("#voiceEditError").textContent = error.message; }
      finally { button.disabled = false; }
    });
    navigate();
  }
  return { setup, parseTags, fitVoiceList, editVoice, renderProject };
};
