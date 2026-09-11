/* Per-job OmniVoice controls. Never changes global settings or preview options. */
"use strict";
window.omniSettings = (() => {
  const fields = [
    { key: "speed", label: "Speed · Tốc độ đọc", min: 0.5, max: 1.5, step: 0.05, help: "Tốc độ lời nói, không phải tốc độ xử lý model." },
    { key: "duration", label: "Duration · Thời lượng mục tiêu (giây)", min: 0, max: 30, step: "any", help: "Để trống để tự ước tính. Mục tiêu thủ công tối đa 30 giây để giới hạn bộ nhớ và thay thế Speed. Chỉ hỗ trợ một câu tối đa 1.000 ký tự; nhiều đoạn phải để trống. Hậu xử lý có thể thay đổi độ dài file." },
    { key: "num_step", label: "Inference Steps · Số bước", min: 4, max: 64, step: 1, help: "Ít bước có thể giảm chất lượng. 24 bước là lựa chọn thử nghiệm, không phải mặc định." },
    { key: "guidance_scale", label: "CFG · Độ bám điều kiện", min: 0, max: 4, step: 0.1, help: "Điều chỉnh classifier-free guidance; giữ mặc định nếu chưa chắc." },
    { key: "denoise", label: "Denoise", help: "Dùng điều kiện khử nhiễu khi sinh giọng.", toggle: true },
    { key: "preprocess_prompt", label: "Preprocess Prompt", help: "Tiền xử lý audio tham chiếu; thay đổi này tạo prompt cache riêng.", toggle: true },
    { key: "postprocess_output", label: "Postprocess Output", help: "Hậu xử lý audio đầu ra theo OmniVoice.", toggle: true },
  ];
  let defaults = null;
  let voice = null;
  const panel = () => document.querySelector("#omnivoiceSettings");
  const input = key => document.getElementById(`omni-${key}`);
  const range = key => document.getElementById(`omni-${key}-range`);
  function paintRange(field) {
    const slider = range(field.key);
    if (!slider) return;
    const percent = 100 * (Number(slider.value) - field.min) / (field.max - field.min);
    slider.style.setProperty("--range-fill", `${Math.max(0, Math.min(100, percent))}%`);
  }
  function active() { return Boolean(defaults && voice?.custom && voice?.provider === "omnivoice"); }
  function sync(selected = voice) {
    voice = selected;
    panel().hidden = !active();
    panel().querySelectorAll("input, button").forEach(element => { element.disabled = !active(); });
    if (defaults && active()) {
      const overridden = input("duration").value !== "";
      input("speed").disabled = overridden;
      range("speed").disabled = overridden;
    }
  }
  function validateField(field) {
    const element = input(field.key);
    let error = "";
    if (!field.toggle && !element.disabled) {
      const value = element.valueAsNumber;
      const optional = field.key === "duration" && element.value === "" && !element.validity.badInput;
      if (!optional && (!Number.isFinite(value) || !element.validity.valid || (field.key === "duration" && value <= 0))) {
        error = field.key === "duration" ? "Nhập số giây lớn hơn 0, tối đa 30 hoặc để trống." : `Nhập ${field.min}–${field.max}${field.key === "num_step" ? " (số nguyên)" : ""}.`;
      }
    }
    element.setAttribute("aria-invalid", String(Boolean(error)));
    document.getElementById(`omni-${field.key}-error`).textContent = error;
    return !error;
  }
  function reset() {
    fields.forEach(field => {
      const element = input(field.key);
      if (field.toggle) element.checked = defaults[field.key];
      else { element.value = defaults[field.key] ?? ""; if (range(field.key)) { range(field.key).value = element.value; paintRange(field); } }
    });
    sync();
    fields.forEach(validateField);
  }
  async function setup() {
    try {
      const response = await fetch("/api/omnivoice/settings");
      if (!response.ok) throw new Error("capabilities unavailable");
      const data = await response.json();
      if (!data.supported || !data.defaults || fields.some(field => !(field.key in data.defaults))) {
        throw new Error("capabilities unavailable");
      }
      defaults = Object.freeze({ ...data.defaults });
      const container = panel().querySelector(".omni-fields");
      container.innerHTML = fields.map(field => {
        const id = `omni-${field.key}`;
        const described = `${id}-help ${id}-error`;
        const bounds = `min="${field.min}" ${field.max === undefined ? "" : `max="${field.max}"`} step="${field.step}"`;
        const controls = field.toggle
          ? `<label class="omni-toggle" for="${id}"><input id="${id}" type="checkbox" aria-describedby="${described}"><span>${field.label}</span></label>`
          : `<label class="field-label" for="${id}">${field.label}</label><div class="omni-number-row">${field.key === "duration" ? "" : `<input id="${id}-range" type="range" ${bounds} aria-label="${field.label} · thanh trượt" aria-describedby="${described}">`}<input id="${id}" type="number" ${bounds} aria-describedby="${described}" ${field.key === "duration" ? 'placeholder="Tự động"' : 'required'}></div>`;
        return `<div class="omni-field">${controls}<p class="form-hint" id="${id}-help">${field.help}</p><p class="omni-error" id="${id}-error" aria-live="polite"></p></div>`;
      }).join("");
      fields.forEach(field => {
        input(field.key).addEventListener("blur", () => validateField(field));
        input(field.key).addEventListener("input", () => {
          if (range(field.key) && input(field.key).validity.valid) { range(field.key).value = input(field.key).value; paintRange(field); }
          if (field.key === "duration") { sync(); validateField(fields[0]); }
        });
        range(field.key)?.addEventListener("input", event => {
          input(field.key).value = event.target.value;
          paintRange(field);
          validateField(field);
        });
      });
      panel().querySelector("button").addEventListener("click", reset);
      reset();
    } catch (_error) {
      defaults = null;
      sync();
      panel().open = false;
      document.querySelector("#omniAvailability").textContent = "Không tải được cấu hình OmniVoice nâng cao; job dùng mặc định máy chủ.";
    }
  }
  function payload() {
    if (!active()) return null;
    const invalid = fields.filter(field => !validateField(field));
    if (invalid.length) {
      panel().open = true;
      input(invalid[0].key).focus();
      throw new Error("Kiểm tra các trường OmniVoice được đánh dấu.");
    }
    return Object.fromEntries(fields.map(field => [field.key, field.toggle ? input(field.key).checked : field.key === "duration" && input(field.key).value === "" ? null : field.key === "speed" && input(field.key).disabled ? 1 : input(field.key).valueAsNumber]));
  }
  return { setup, sync, payload };
})();
