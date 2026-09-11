/* Deterministic controller/payload exercise. No browser or provider calls. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const scriptPath = 'web/static/omnivoice-settings.js';
assert.ok(fs.existsSync(scriptPath), 'Advanced settings controller missing');
const source = fs.readFileSync(scriptPath, 'utf8');
const app = fs.readFileSync('web/static/app.js', 'utf8');
const defaults = { speed: 1, duration: null, num_step: 32, guidance_scale: 2, denoise: true, preprocess_prompt: true, postprocess_output: true };
class Element {
  constructor(attrs = {}) { this.attrs = attrs; this.value = ''; this.checked = false; this.disabled = false; this.hidden = false; this.textContent = ''; this.listeners = {}; this.style = {setProperty(key, value) {this[key] = value;}}; }
  get valueAsNumber() { return this.value === '' ? NaN : Number(this.value); }
  get validity() {
    const n = this.valueAsNumber;
    const {min, max, step} = this.attrs;
    return { badInput: false, valid: this.value === '' ? !('required' in this.attrs) : Number.isFinite(n) && (min === undefined || n >= +min) && (max === undefined || n <= +max) && (step === undefined || step === 'any' || Math.abs((n - +(min || 0)) / +step - Math.round((n - +(min || 0)) / +step)) < 1e-7) };
  }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  dispatch(type) { for (const handler of this.listeners[type] || []) handler({target: this}); }
  focus() { this.focused = true; }
}
function fixture(reply = {supported: true, defaults}, ok = true) {
  const elements = {};
  for (const id of ['omnivoiceSettings', 'omniAvailability', 'textInput', 'textVoiceSelect', 'textLanguage', 'textStartButton', 'textCharacterCount', 'textDurationEstimate', 'textFixedPreviewButton', 'textPreviewButton', 'textJobForm', 'textJobName', 'textProjectMode', 'textProjectHint', 'textJobHint']) elements[id] = new Element();
  const panel = elements.omnivoiceSettings;
  panel.hidden = true; panel.open = false;
  const reset = new Element();
  const fields = {};
  Object.defineProperty(fields, 'innerHTML', {set(markup) {
    for (const match of markup.matchAll(/<(input|p)\b([^>]*)>/g)) {
      const attrs = {};
      for (const m of match[2].matchAll(/([\w-]+)="([^"]*)"/g)) attrs[m[1]] = m[2];
      if (/\brequired\b/.test(match[2])) attrs.required = '';
      if (attrs.id) elements[attrs.id] = new Element(attrs);
    }
  }});
  panel.querySelector = selector => selector === '.omni-fields' ? fields : reset;
  panel.querySelectorAll = () => [...Object.values(elements).filter(e => e.attrs.type), reset];
  const requests = [];
  const context = vm.createContext({window: {}, document: {querySelector: s => elements[s.slice(1)], getElementById: id => elements[id]}, fetch: async url => {requests.push(url); return {ok, json: async () => reply};}, console});
  vm.runInContext(source, context);
  return {elements, panel, reset, context, requests, controller: context.window.omniSettings};
}
const clone = {id: 'clone-test', custom: true, provider: 'omnivoice'};
function json(value) { return JSON.parse(JSON.stringify(value)); }
function fn(name, next) { return app.slice(app.indexOf(`function ${name}(`), app.indexOf(`function ${next}(`)).replace(/async\s*$/, ''); }
(async () => {
  const f = fixture();
  f.controller.sync(clone);
  await f.controller.setup();
  assert.equal(f.panel.hidden, false);
  assert.equal(f.panel.open, false);
  assert.deepEqual(json(f.controller.payload()), defaults);
  const {elements: e} = f;
  e['omni-num_step-range'].value = '24'; e['omni-num_step-range'].dispatch('input');
  assert.equal(e['omni-num_step'].value, '24');
  e['omni-speed'].value = '1.25'; e['omni-speed'].dispatch('input');
  assert.equal(e['omni-speed-range'].value, '1.25');
  assert.equal(e['omni-speed-range'].style['--range-fill'], '75%');
  e['omni-duration'].value = '5'; e['omni-duration'].dispatch('input');
  assert.equal(e['omni-speed'].disabled, true);
  assert.equal(f.controller.payload().speed, 1);
  assert.equal(f.controller.payload().duration, 5);
  e['omni-duration'].value = '31';
  assert.throws(() => f.controller.payload(), 'Unbounded duration must be rejected');
  assert.match(e['omni-duration-error'].textContent, /30/);
  e['omni-duration'].value = '5';
  e['omni-num_step'].value = '3';
  assert.throws(() => f.controller.payload());
  assert.equal(f.panel.open, true); assert.equal(e['omni-num_step'].focused, true);
  assert.equal(e['omni-num_step'].attrs['aria-invalid'], 'true');
  assert.ok(e['omni-num_step-error'].textContent);
  assert.equal(e['omni-num_step'].attrs['aria-describedby'], 'omni-num_step-help omni-num_step-error');
  f.reset.dispatch('click');
  assert.deepEqual(json(f.controller.payload()), defaults);
  e['omni-num_step'].value = '24';
  for (const provider of ['edge', 'gemini']) {
    f.controller.sync({custom: false, provider});
    assert.equal(f.panel.hidden, true); assert.equal(f.controller.payload(), null);
    assert.ok(f.panel.querySelectorAll().every(el => el.disabled));
  }
  f.controller.sync(clone);
  assert.equal(f.controller.payload().num_step, 24);
  // Exercise actual submission function, capturing only its network boundary.
  const calls = [];
  Object.assign(f.context, {
    $: s => e[s.slice(1)], state: {textBusy: false, jobDrafts: {text: {}}},
    clearTextError() {}, showTextError(message) {calls.push({error: message});},
    abortTextPreview() {}, updateTextState() {}, refreshJobs: async () => {}, selectJob() {}, announce() {}, toast() {},
    sessionStorage: {removeItem() {}}, apiJson: async (url, args) => {calls.push({url, body: JSON.parse(args.body)}); return {};},
  });
  vm.runInContext('async ' + fn('submitTextJob', 'setupTextComposer'), f.context);
  e.textJobName.value = 'My speech'; e.textProjectMode.checked = true;
  e.textInput.value = 'Synthetic fixture'; e.textVoiceSelect.value = clone.id; e.textLanguage.value = 'en-US';
  await f.context.submitTextJob({preventDefault() {}});
  assert.equal(calls[0].url, '/api/jobs/text');
  assert.equal(calls[0].body.omnivoice.num_step, 24);
  assert.equal(calls[0].body.name, 'My speech');
  assert.equal(calls[0].body.project, true);
  e['omni-speed'].value = '2'; e.textInput.value = 'Synthetic fixture';
  await f.context.submitTextJob({preventDefault() {}});
  assert.ok(calls[1].error, 'Invalid advanced settings must block submission');
  f.controller.sync({custom: false, provider: 'edge'}); e.textInput.value = 'Synthetic fixture';
  await f.context.submitTextJob({preventDefault() {}});
  assert.equal('omnivoice' in calls[2].body, false);
  // An optional asset failure must not prevent unrelated app boot or legacy jobs.
  delete f.context.window.omniSettings;
  Object.assign(f.context, {
    playTextFixedPreview() {}, previewText() {}, selectedTextVoice: () => clone,
    LANGUAGE_RATES: {'en-US': 10}, fmtDuration: () => '0s',
  });
  vm.runInContext(fn('updateTextState', 'selectedTextVoice'), f.context);
  const setupStart = app.indexOf('function setupTextComposer(');
  vm.runInContext(app.slice(setupStart, app.indexOf('\n$("#refreshButton")', setupStart)), f.context);
  assert.doesNotThrow(() => f.context.setupTextComposer());
  e.textInput.value = 'a'.repeat(50001); e.textProjectMode.checked = false; f.context.updateTextState();
  assert.equal(e.textProjectMode.checked, true); assert.equal(e.textProjectMode.disabled, true);
  e.textInput.value = 'a'.repeat(200001); f.context.updateTextState(); assert.equal(e.textStartButton.disabled, true);
  assert.ok(e.omniAvailability.textContent, 'Missing controller must show default-mode warning');
  e.textInput.value = 'Synthetic asset failure.';
  await f.context.submitTextJob({preventDefault() {}});
  assert.equal(calls[3].url, '/api/jobs/text');
  assert.equal('omnivoice' in calls[3].body, false);
  // A stale server must never expose controls or silently send overrides.
  for (const [reply, ok] of [[{}, false], [{supported: false, defaults: {}}, true], [{supported: true, defaults: {}}, true]]) {
    const unavailable = fixture(reply, ok);
    unavailable.controller.sync(clone);
    await unavailable.controller.setup();
    assert.equal(unavailable.panel.hidden, true);
    assert.equal(unavailable.controller.payload(), null);
    assert.ok(unavailable.elements.omniAvailability.textContent, 'Unavailable server needs visible warning');
  }
  console.log('PASS controller: paired controls, duration, validation/focus, reset, provider isolation, real text payload, stale-server fail-closed');
})().catch(error => {console.error(error); process.exitCode = 1;});
