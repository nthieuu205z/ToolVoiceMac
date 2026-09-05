# Multilingual Text-to-Voice and Video Dubbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class Vietnamese/English Text → Voice and Vietnamese/English video dubbing while keeping one persistent live job queue and preserving the existing Vietnamese workflow.

**Architecture:** Introduce canonical language and voice-capability registries, make provider and video contracts target-language-aware, then extract raw text synthesis from video timing. A dedicated Text → Voice runner and both API creation paths feed one generalized JobManager that publishes typed artifacts; the plain-JavaScript frontend renders one two-tab composer and one adaptive queue/graph.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, NumPy, FFmpeg/FFprobe, edge-tts, Google GenAI, optional OmniVoice/PyTorch, pytest, plain HTML/CSS/JavaScript, SSE, MCP Playwright.

**Spec:** `docs/superpowers/specs/2026-09-05-multilingual-text-to-voice-design.md`

## Global Constraints

- Canonical language codes are exactly `vi-VN` and `en-US` in the first release; API boundaries may normalize `vi` and `en`.
- Existing `POST /api/jobs` callers that omit `target_language` continue to create Vietnamese video jobs.
- Existing `/download/video`, `/download/srt`, and preview requests without a language remain compatibility aliases.
- Text → Voice reads submitted text as-is, accepts at most 50,000 Unicode characters, inserts 200 ms between synthesized chunks, and publishes both WAV and MP3 or neither format as a successful result.
- On-demand text preview reads the first complete sentence or 300 characters, creates no job, and never persists the submitted preview text.
- Keep FastAPI and plain HTML/CSS/JavaScript; introduce no frontend framework or build step.
- Use MCP Playwright for browser verification at 1440×1000 and 360×800.
- Preserve all user-owned uncommitted changes. Before Task 1, invoke `superpowers:using-git-worktrees`; the isolated workspace must include the approved current working-tree baseline. If that state cannot be reproduced safely from a commit, stop and request a baseline commit instead of silently starting from `273d332` alone.
- Implement every behavior test-first and commit only task-scoped files after its tests pass.
- Do not merge or push until Task 11 and the final independent review pass.

---

### Task 1: Canonical Language Registry and Voice Capabilities

**Files:**
- Create: `pipeline/languages.py`
- Create: `backend/routes/languages.py`
- Create: `tests/test_languages.py`
- Modify: `pipeline/voices.py:14-155`
- Modify: `pipeline/custom_voices.py:1-180`
- Modify: `backend/routes/voices.py:55-67`
- Modify: `backend/main.py:14-43`
- Modify: `tests/test_routing.py:14-85`
- Modify: `tests/test_api.py:28-73`

**Interfaces:**
- Produces: `LanguageSpec`, `available_languages()`, `require_language(code)`, and `normalize_language_code(code)`.
- Produces: `Voice.provider: str`, `Voice.supported_languages: tuple[str, ...]`, `Voice.supports(language) -> bool`.
- Produces: `available_voices(tts_provider, clone_provider, language=None)` and language-aware `is_available()`/`default_voice()`.
- Produces: `GET /api/languages` and additive language/provider fields in `GET /api/voices`.
- Consumes: existing custom-voice IDs and provider-routing rules without changing their storage format.

- [ ] **Step 1: Add failing registry and normalization tests**

```python
# tests/test_languages.py
import pytest

from pipeline.languages import available_languages, normalize_language_code, require_language


def test_initial_registry_uses_canonical_bcp47_codes():
    assert [item.code for item in available_languages()] == ["vi-VN", "en-US"]
    assert require_language("vi-VN").omnivoice_name == "Vietnamese"
    assert require_language("en-US").gemini_tts_code == "en-US"


@pytest.mark.parametrize(("raw", "expected"), [
    ("vi", "vi-VN"), ("vi-vn", "vi-VN"),
    ("en", "en-US"), ("EN-us", "en-US"),
])
def test_language_aliases_normalize_at_boundaries(raw, expected):
    assert normalize_language_code(raw) == expected


def test_unknown_language_is_rejected():
    with pytest.raises(ValueError, match="Ngôn ngữ không được hỗ trợ"):
        normalize_language_code("fr")
```

- [ ] **Step 2: Run the new tests and verify they fail because the registry does not exist**

Run: `pytest tests/test_languages.py -q`
Expected: FAIL during collection with `ModuleNotFoundError: pipeline.languages`.

- [ ] **Step 3: Implement the immutable language registry**

```python
# pipeline/languages.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    display_name: str
    english_name: str
    gemini_tts_code: str
    omnivoice_name: str
    preview_text: str
    sentence_terminators: tuple[str, ...]


_LANGUAGES = (
    LanguageSpec(
        code="vi-VN",
        display_name="Tiếng Việt",
        english_name="Vietnamese",
        gemini_tts_code="vi-VN",
        omnivoice_name="Vietnamese",
        preview_text="Xin chào, đây là giọng đọc tiếng Việt dùng để lồng tiếng cho video của bạn.",
        sentence_terminators=(".", "?", "!", "…"),
    ),
    LanguageSpec(
        code="en-US",
        display_name="English (US)",
        english_name="English",
        gemini_tts_code="en-US",
        omnivoice_name="English",
        preview_text="Hello, this is an English voice preview for your audio and video projects.",
        sentence_terminators=(".", "?", "!", "…"),
    ),
)
_BY_CODE = {item.code.casefold(): item for item in _LANGUAGES}
_ALIASES = {"vi": "vi-VN", "vi-vn": "vi-VN", "en": "en-US", "en-us": "en-US"}


def available_languages() -> tuple[LanguageSpec, ...]:
    return _LANGUAGES


def normalize_language_code(code: str) -> str:
    key = str(code or "").strip().casefold()
    canonical = _ALIASES.get(key)
    if canonical is None and key in _BY_CODE:
        canonical = _BY_CODE[key].code
    if canonical is None:
        raise ValueError(f"Ngôn ngữ không được hỗ trợ: {code}")
    return canonical


def require_language(code: str) -> LanguageSpec:
    return _BY_CODE[normalize_language_code(code).casefold()]
```

- [ ] **Step 4: Add failing voice-capability tests**

```python
# append to tests/test_routing.py
def test_available_voices_filters_by_canonical_language():
    vietnamese = available_voices("edge", "omnivoice", language="vi-VN")
    english = available_voices("edge", "omnivoice", language="en-US")

    assert all(voice.supports("vi-VN") for voice in vietnamese)
    assert all(voice.supports("en-US") for voice in english)
    assert "vi-VN-HoaiMyNeural" not in {voice.id for voice in english}
    assert "en-US-AvaMultilingualNeural" in {voice.id for voice in english}


def test_custom_voice_supports_vietnamese_and_english():
    voice_id = _make_clone("clone-multi", "Multi")
    voice = next(v for v in available_voices("edge", "omnivoice") if v.id == voice_id)
    assert voice.supported_languages == ("vi-VN", "en-US")
```

- [ ] **Step 5: Extend `Voice` and language-aware catalog operations without breaking no-language callers**

```python
# pipeline/voices.py — final public shape
@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str
    native_id: str = ""
    provider: str = ""
    supported_languages: tuple[str, ...] = ("vi-VN",)

    @property
    def native(self) -> str:
        return self.native_id or self.id

    def supports(self, language: str) -> bool:
        return normalize_language_code(language) in self.supported_languages


def available_voices(
    tts_provider: str,
    clone_provider: str | None,
    language: str | None = None,
) -> list[Voice]:
    voices = list(_BY_PROVIDER[tts_provider])
    if clone_provider in _CLONE_PROVIDERS:
        voices += _clone_voices()
    if language is None:
        return voices
    canonical = normalize_language_code(language)
    return [voice for voice in voices if voice.supports(canonical)]
```

Annotate native Vietnamese Edge voices with `("vi-VN",)`, verified multilingual Edge voices and Gemini voices with `("vi-VN", "en-US")`, and custom OmniVoice voices with `("vi-VN", "en-US")`. Pass `provider=` by keyword so the existing third positional `native_id` argument cannot shift.

- [ ] **Step 6: Add the language endpoint and additive voice response fields**

```python
# backend/routes/languages.py
from fastapi import APIRouter

from pipeline.languages import available_languages

router = APIRouter()


@router.get("/api/languages")
def list_languages() -> list[dict]:
    return [
        {
            "code": item.code,
            "display_name": item.display_name,
            "english_name": item.english_name,
            "video_dubbing": True,
            "text_to_voice": True,
        }
        for item in available_languages()
    ]
```

Update `backend/routes/voices.py::list_voices(language: str | None = None)` to pass the filter and return `provider` plus `supported_languages`, while retaining `preview_url` and `custom`. Register `languages.router` before the static mount in `backend/main.py`.

- [ ] **Step 7: Run focused and compatibility tests**

Run: `pytest tests/test_languages.py tests/test_routing.py tests/test_api.py -q`
Expected: PASS; existing `/api/voices` tests are updated to use subset assertions because fields are additive.

- [ ] **Step 8: Commit the registry slice**

```bash
git add pipeline/languages.py pipeline/voices.py pipeline/custom_voices.py \
  backend/routes/languages.py backend/routes/voices.py backend/main.py \
  tests/test_languages.py tests/test_routing.py tests/test_api.py
git commit -m "feat: add multilingual language and voice registry"
```

---

### Task 2: Language-Aware Provider and Backend Contracts

**Files:**
- Modify: `pipeline/models.py:108-125`
- Modify: `pipeline/backends.py:48-188`
- Modify: `pipeline/edge_speech.py:34-75`
- Modify: `pipeline/gemini.py:111-371`
- Modify: `pipeline/omnivoice_speech.py:243-335`
- Modify: `tests/test_backend_batch_forwarding.py:21-127`
- Modify: `tests/test_edge_speech.py:1-180`
- Modify: `tests/test_gemini_backend.py:1-180`
- Modify: `tests/test_gemini_fallback.py:167-372`
- Modify: `tests/test_omnivoice_speech.py:61-205`

**Interfaces:**
- Consumes: `require_language(code)` from Task 1.
- Produces: focused `SpeechRecognizer`, `Translator`, and `SpeechSynthesizer` protocols.
- Produces: `synthesize(text, voice_id, *, language="vi-VN")` and `synthesize_batch(texts, voice_id, *, language="vi-VN")` across adapters and `CompositeBackend`.
- Produces: `translate(texts, durations, context="", *, target_language="vi-VN")` forwarding; Task 3 supplies target-neutral Gemini prompting.

- [ ] **Step 1: Write forwarding and provider-language tests that fail on the old signatures**

```python
# tests/test_backend_batch_forwarding.py
class RecordingSynthesizer:
    batch_size = 8

    def __init__(self):
        self.single_calls = []
        self.batch_calls = []

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        self.single_calls.append((text, voice_id, language))
        return b"\x00\x00"

    def synthesize_batch(self, texts, voice_id, *, language="vi-VN"):
        self.batch_calls.append((list(texts), voice_id, language))
        return [b"\x00\x00" for _ in texts]


def test_composite_forwards_language_to_single_and_batch_synthesis():
    synth = RecordingSynthesizer()
    backend = CompositeBackend(recognizer=None, translator=None, synthesizer=synth)

    backend.synthesize("hello", "voice", language="en-US")
    backend.synthesize_batch(["one", "two"], "voice", language="en-US")

    assert synth.single_calls == [("hello", "voice", "en-US")]
    assert synth.batch_calls == [(["one", "two"], "voice", "en-US")]
```

```python
# tests/test_omnivoice_speech.py
def test_english_synthesis_passes_omnivoice_language_name():
    voice_id = _make_clone("clone-english")
    model = FakeOmni()

    OmniVoiceSynthesizer(model=model, transcriber=FakeTranscriber()).synthesize(
        "Welcome", voice_id, language="en-US"
    )

    assert model.calls[0]["kwargs"]["language"] == ["English"]
```

- [ ] **Step 2: Run the focused tests and verify signature failures**

Run: `pytest tests/test_backend_batch_forwarding.py tests/test_omnivoice_speech.py -q`
Expected: FAIL with unexpected keyword argument `language`.

- [ ] **Step 3: Split capability protocols and add defaulted language keyword parameters**

```python
# pipeline/models.py
class SpeechRecognizer(Protocol):
    def transcribe_clip(self, wav_path) -> tuple[str, str]: ...


class Translator(Protocol):
    def translate(
        self,
        texts: list[str],
        durations: list[float],
        context: str = "",
        *,
        target_language: str = "vi-VN",
    ) -> list[str]: ...


class SpeechSynthesizer(Protocol):
    def synthesize(
        self, text: str, voice_id: str, *, language: str = "vi-VN"
    ) -> bytes: ...
```

Replace type annotations that only need one capability. Keep a temporary `GeminiBackend` protocol inheriting all three protocols so unchanged test doubles continue to type-check during Tasks 2–3; remove that compatibility name only after all internal imports are migrated.

- [ ] **Step 4: Thread `language` through `CompositeBackend`, `LazyGemini`, and Edge**

```python
# pipeline/backends.py — forwarding contract
def synthesize(self, text: str, voice_id: str, *, language: str = "vi-VN") -> bytes:
    return self._synthesizer.synthesize(text, voice_id, language=language)


def synthesize_batch(
    self, texts: list[str], voice_id: str, *, language: str = "vi-VN"
) -> list[bytes]:
    return self._synthesizer.synthesize_batch(texts, voice_id, language=language)
```

`EdgeSynthesizer` validates the selected catalog voice supports the canonical language before starting the async request, then continues passing the voice's native ID to `edge_tts.Communicate`. Its default remains `vi-VN`.

- [ ] **Step 5: Resolve language per Gemini speech request instead of fixing it in the constructor**

```python
# pipeline/gemini.py — essential flow
def synthesize(self, text: str, voice_id: str, *, language: str = "vi-VN") -> bytes:
    language_code = require_language(language).gemini_tts_code
    response = self._generate_speech(text, voice_id, language_code)
    return self._audio_bytes_or_raise(response)


def _speech_call(self, text: str, voice_id: str, language_code: str, *, with_language: bool):
    speech_kwargs = {"voice_config": self._voice_config(voice_id)}
    if with_language and language_code:
        speech_kwargs["language_code"] = language_code
    return self._client.models.generate_content(
        model=self._tts_model,
        contents=_TTS_INSTRUCTION + text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(**speech_kwargs),
        ),
    )
```

Preserve the current retry-without-`language_code` behavior, but pass the per-call code through both the first request and retry. Keep any constructor `tts_language_code` argument only as a deprecated default for callers that omit `language`.

- [ ] **Step 6: Map canonical language to OmniVoice single and batch generation**

```python
def synthesize(self, text: str, voice_id: str, *, language: str = "vi-VN") -> bytes:
    return self._generate([text], voice_id, language=language)[0]


def synthesize_batch(
    self, texts: list[str], voice_id: str, *, language: str = "vi-VN"
) -> list[bytes]:
    return self._generate(list(texts), voice_id, language=language)


def _generate(self, texts: list[str], voice_id: str, *, language: str) -> list[bytes]:
    upstream_language = require_language(language).omnivoice_name
    kwargs = {"language": [upstream_language] * len(texts)}
```

- [ ] **Step 7: Update test doubles and verify both default Vietnamese and explicit English**

Run: `pytest tests/test_backend_batch_forwarding.py tests/test_edge_speech.py tests/test_gemini_backend.py tests/test_gemini_fallback.py tests/test_omnivoice_speech.py -q`
Expected: PASS, including old calls that omit `language` and therefore remain Vietnamese.

- [ ] **Step 8: Commit provider contracts**

```bash
git add pipeline/models.py pipeline/backends.py pipeline/edge_speech.py pipeline/gemini.py \
  pipeline/omnivoice_speech.py tests/test_backend_batch_forwarding.py \
  tests/test_edge_speech.py tests/test_gemini_backend.py tests/test_gemini_fallback.py \
  tests/test_omnivoice_speech.py
git commit -m "refactor: make speech providers language aware"
```

---

### Task 3: Target-Neutral Video Translation and Output

**Files:**
- Modify: `pipeline/models.py:41-82`
- Modify: `pipeline/gemini.py:111-293`
- Modify: `pipeline/translate.py:1-100`
- Modify: `pipeline/tts.py:69-330`
- Modify: `pipeline/subtitles.py:117-160`
- Modify: `pipeline/runner.py:35-180`
- Modify: `backend/routes/jobs.py:37-100,188-203`
- Modify: `scripts/run_pipeline_cli.py:1-120`
- Modify: `tests/test_translate_alignment.py:1-140`
- Modify: `tests/test_runner.py:16-116`
- Modify: `tests/test_tts.py:1-280`
- Modify: `tests/test_tts_batch.py:13-172`
- Modify: `tests/test_subtitles.py:1-180`
- Modify: `tests/test_api.py:125-198`

**Interfaces:**
- Consumes: target-aware translator/TTS methods from Task 2.
- Produces: `Segment.target_text`, compatibility `Segment.text_vi` property, and `PipelineOptions.target_language` defaulting to `vi-VN`.
- Produces: target-aware video/SRT filenames while preserving Vietnamese defaults.

- [ ] **Step 1: Add failing target-text and English-runner tests**

```python
# tests/test_translate_alignment.py
class RecordingBackend:
    def __init__(self, result):
        self.result = result
        self.target_languages = []

    def translate(self, texts, durations, context="", *, target_language="vi-VN"):
        self.target_languages.append(target_language)
        return list(self.result)


def test_translation_writes_target_text_and_passes_target_language():
    backend = RecordingBackend(result=["Hello"])
    segment = Segment(0.0, 1.0, "Xin chào")

    translate_segments(backend, [segment], target_language="en-US", workers=1)

    assert segment.target_text == "Hello"
    assert backend.target_languages == ["en-US"]


def test_text_vi_alias_tracks_target_text_during_migration():
    segment = Segment(0.0, 1.0, "source", target_text="first")
    segment.text_vi = "second"
    assert segment.target_text == "second"
```

```python
# tests/test_runner.py
def test_runner_skips_translation_when_source_matches_target(stub_stages, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "translate_segments", lambda *a, **k: calls.append(k))

    run_pipeline(
        None,
        stub_stages / "in.mp4",
        stub_stages,
        PipelineOptions(voice_id="v", target_language="en-US"),
        media=MEDIA,
    )

    assert calls == []
```

- [ ] **Step 2: Run the focused tests and verify missing target-neutral fields**

Run: `pytest tests/test_translate_alignment.py tests/test_runner.py -q`
Expected: FAIL for missing `target_text` and `target_language`.

- [ ] **Step 3: Migrate `Segment` while keeping the property alias**

```python
@dataclass
class Segment:
    start: float
    end: float
    text: str
    target_text: str = ""
    spoken_duration: float = 0.0
    placed_start: float | None = None

    @property
    def text_vi(self) -> str:
        return self.target_text

    @text_vi.setter
    def text_vi(self, value: str) -> None:
        self.target_text = value
```

Update all internal construction and tests that use `text_vi=` to use `target_text=`. Keep read/write alias coverage until a later removal release.

- [ ] **Step 4: Make Gemini translation schema and prompt target-neutral**

Rename `_TranslatedLine.text_vi` to `text`, format the prompt with `LanguageSpec.display_name` and `english_name`, and compute `max_chars` from a per-language speaking-rate map held next to the prompt (`vi-VN: 16.6`, `en-US: 14.0`). The public method becomes:

```python
def translate(
    self,
    texts: list[str],
    durations: list[float],
    context: str = "",
    *,
    target_language: str = "vi-VN",
) -> list[str]:
    language = require_language(target_language)
    prompt = _TRANSLATE_PROMPT.format(
        target_name=language.english_name,
        count=len(texts),
        context=_format_context(context),
        payload=_translation_payload(texts, durations, language.code),
    )
    parsed = self._parse(self._generate_translation(prompt), _Translation)
    by_index = {line.index: line.text.strip() for line in parsed.lines}
    return [by_index.get(index, "") for index in range(len(texts))]
```

- [ ] **Step 5: Thread target language through translation, TTS, subtitles, and runner**

Add `target_language: str = "vi-VN"` to `PipelineOptions`. In `run_pipeline`, normalize detected source language, copy `text` into `target_text` when it equals the target, otherwise call `translate_segments(..., target_language=options.target_language)`. Pass `language=options.target_language` to `synthesize_segments`. Replace all internal `segment.text_vi` reads with `segment.target_text`.

- [ ] **Step 6: Add the optional video form field and target-aware compatibility filenames**

```python
@router.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    voice_id: str = Form(...),
    target_language: str = Form("vi-VN"),
) -> dict:
    language = normalize_language_code(target_language)
    if not is_available(voice_id, settings.tts_provider, clone_provider, language=language):
        raise HTTPException(400, "Giọng đọc không hỗ trợ ngôn ngữ đã chọn.")
```

Use `_vi` for `vi-VN` and `_en` for `en-US` in compatibility download filenames. Add `--target-language` with choices from the registry to the CLI and default it to `vi-VN`.

- [ ] **Step 7: Run the video regression slice**

Run: `pytest tests/test_translate_alignment.py tests/test_runner.py tests/test_tts.py tests/test_tts_batch.py tests/test_subtitles.py tests/test_api.py tests/test_run_pipeline_cli.py -q`
Expected: PASS; Vietnamese assertions remain unchanged unless they name `target_text` internally.

- [ ] **Step 8: Commit the target-neutral video pipeline**

```bash
git add pipeline/models.py pipeline/gemini.py pipeline/translate.py pipeline/tts.py \
  pipeline/subtitles.py pipeline/runner.py backend/routes/jobs.py scripts/run_pipeline_cli.py \
  tests/test_translate_alignment.py tests/test_runner.py tests/test_tts.py \
  tests/test_tts_batch.py tests/test_subtitles.py tests/test_api.py tests/test_run_pipeline_cli.py
git commit -m "feat: support target languages in video dubbing"
```

---

### Task 4: Reusable Raw Text Synthesis and Production-Priority Coordination

**Files:**
- Create: `pipeline/speech_synthesis.py`
- Create: `pipeline/speech_runtime.py`
- Create: `tests/test_speech_synthesis.py`
- Create: `tests/test_speech_runtime.py`
- Modify: `pipeline/tts.py:69-330`
- Modify: `tests/test_tts_batch.py:21-172`
- Modify: `tests/test_synthesis_quality.py:1-100`

**Interfaces:**
- Consumes: language-aware `SpeechSynthesizer` from Task 2.
- Produces: `synthesize_texts(synthesizer, texts, voice_id, *, language, progress, should_cancel) -> tuple[list[bytes | None], list[str]]`.
- Produces: process-wide `speech_activity.production(provider)` and `speech_activity.preview(provider)` coordination.
- Preserves: video-specific fitting inside `synthesize_segments()`.

- [ ] **Step 1: Write failing raw-synthesis order, fallback, cancellation, and partial-failure tests**

```python
# tests/test_speech_synthesis.py
from pipeline.models import TTS_SAMPLE_RATE


def pcm(marker: int) -> bytes:
    return (int(marker).to_bytes(2, "little", signed=True) * TTS_SAMPLE_RATE)


class BatchFailsThenSingles:
    batch_size = 16

    def __init__(self, bad_text):
        self.bad_text = bad_text
        self.single_calls = []

    def synthesize_batch(self, texts, voice_id, *, language="vi-VN"):
        raise RuntimeError("batch unavailable")

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        self.single_calls.append((text, voice_id, language))
        if text == self.bad_text:
            raise RuntimeError("single item failed")
        return pcm({"one": 1, "three": 3}[text])


class RecordingBackend:
    batch_size = 0

    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        self.calls.append((text, voice_id, language))
        return pcm(1)


def test_batch_fallback_preserves_indexes_and_marks_failed_items():
    backend = BatchFailsThenSingles(bad_text="two")

    audio, warnings = synthesize_texts(
        backend,
        ["one", "two", "three"],
        "voice",
        language="en-US",
    )

    assert audio == [pcm(1), None, pcm(3)]
    assert backend.single_calls == [
        ("one", "voice", "en-US"),
        ("two", "voice", "en-US"),
        ("three", "voice", "en-US"),
    ]
    assert any("1/3" in warning for warning in warnings)


def test_cancellation_runs_before_first_provider_call():
    backend = RecordingBackend()
    with pytest.raises(JobCancelledError):
        synthesize_texts(
            backend, ["one"], "voice", language="en-US", should_cancel=lambda: True
        )
    assert backend.calls == []
```

- [ ] **Step 2: Run the raw-synthesis tests and verify the module is absent**

Run: `pytest tests/test_speech_synthesis.py -q`
Expected: FAIL during collection with `ModuleNotFoundError`.

- [ ] **Step 3: Extract batch partitioning and indexed raw synthesis from `tts.py`**

```python
def synthesize_texts(
    synthesizer: SpeechSynthesizer,
    texts: list[str],
    voice_id: str,
    *,
    language: str,
    progress: ProgressFn = noop_progress,
    should_cancel: CancelFn = never_cancel,
) -> tuple[list[bytes | None], list[str]]:
    outputs: list[bytes | None] = [None] * len(texts)
    warnings: list[str] = []
    batch_size = max(1, int(getattr(synthesizer, "batch_size", 0) or 1))
    for offset in range(0, len(texts), batch_size):
        if should_cancel():
            raise JobCancelledError()
        batch = texts[offset:offset + batch_size]
        try:
            if batch_size == 1 or not hasattr(synthesizer, "synthesize_batch"):
                raise BatchFallbackRequired()
            values = synthesizer.synthesize_batch(batch, voice_id, language=language)
            if len(values) != len(batch):
                raise BatchFallbackRequired()
            outputs[offset:offset + len(batch)] = values
        except (BatchFallbackRequired, SpeechServiceError, RuntimeError):
            for relative_index, text in enumerate(batch):
                if should_cancel():
                    raise JobCancelledError()
                try:
                    outputs[offset + relative_index] = synthesizer.synthesize(
                        text, voice_id, language=language
                    )
                except (SpeechServiceError, RuntimeError) as exc:
                    warnings.append(
                        f"Không thể tạo giọng cho đoạn {offset + relative_index + 1}/{len(texts)}: {type(exc).__name__}"
                    )
        progress(
            "synthesize",
            (offset + len(batch)) / max(1, len(texts)),
            f"Đang tạo giọng {offset + len(batch)}/{len(texts)}",
        )
    return outputs, warnings
```

Define `BatchFallbackRequired` as a private exception in this module. Move `_chia_lo` and provider batch detection into this module, preserving the current provider-specific exception mapping and warning copy; never include submitted text in warning strings.

- [ ] **Step 4: Add failing activity-coordinator priority tests**

```python
# tests/test_speech_runtime.py
def test_preview_is_rejected_while_production_is_active():
    activity = SpeechActivity()
    with activity.production("omnivoice"):
        with pytest.raises(PreviewBusyError):
            with activity.preview("omnivoice"):
                raise AssertionError("preview body must not run")


def test_parallel_production_is_allowed_but_preview_is_single_slot():
    activity = SpeechActivity()
    with activity.production("edge"), activity.production("edge"):
        assert activity.active_production("edge") == 2
```

- [ ] **Step 5: Implement a per-provider coordinator that never blocks production behind queued preview work**

Implement `SpeechActivity` with one `threading.Condition`, per-provider active-production counts, and per-provider preview ownership. `production(provider)` waits only for an already-running preview, increments the production count, and notifies on exit. `preview(provider)` fails immediately with `PreviewBusyError` if production or another preview is active. Export one singleton `speech_activity`.

- [ ] **Step 6: Refactor video synthesis to consume raw indexed output then fit only successful PCM**

```python
raw, warnings = synthesize_texts(
    backend,
    [segment.target_text for segment in segments_to_read],
    voice_id,
    language=language,
    progress=progress,
    should_cancel=should_cancel,
)
for segment, pcm in zip(segments_to_read, raw):
    if pcm is None:
        continue
    fitted.append((segment, _fit_one(pcm, segment, max_speedup, fill_slowdown)))
```

Wrap production provider calls in `speech_activity.production(backend.engine or "unknown")`. Keep runaway-audio checks, bad-hole resynthesis, timeline fitting, and placement behavior in `pipeline/tts.py`.

- [ ] **Step 7: Run extraction and video-quality regressions**

Run: `pytest tests/test_speech_synthesis.py tests/test_speech_runtime.py tests/test_tts.py tests/test_tts_batch.py tests/test_synthesis_quality.py -q`
Expected: PASS with the same fitted segment ordering and quality warnings as before.

- [ ] **Step 8: Commit the synthesis seam**

```bash
git add pipeline/speech_synthesis.py pipeline/speech_runtime.py pipeline/tts.py \
  tests/test_speech_synthesis.py tests/test_speech_runtime.py tests/test_tts.py \
  tests/test_tts_batch.py tests/test_synthesis_quality.py
git commit -m "refactor: separate raw speech synthesis from video timing"
```

---

### Task 5: Dedicated Text-to-Voice Runner and WAV/MP3 Export

**Files:**
- Create: `pipeline/text_to_voice.py`
- Create: `tests/test_text_to_voice.py`
- Modify: `pipeline/errors.py:1-70`
- Modify: `pipeline/models.py:8-28,74-105`

**Interfaces:**
- Consumes: `LanguageSpec`, `synthesize_texts()`, `write_wav()`, `ffmpeg_utils.resolve()`, and `ffmpeg_utils.run()`.
- Produces: `TextToVoiceOptions`, `SpeechResult`, `chunk_text()`, `build_mp3_cmd()`, and `run_text_to_voice(synthesizer, text, workdir, options, *, progress=noop_progress, should_cancel=never_cancel) -> SpeechResult`.
- Produces: text stages `prepare`, `synthesize`, `assemble`, `export` and per-job stage weights.

- [ ] **Step 1: Add failing text normalization and multilingual chunk tests**

```python
# tests/test_text_to_voice.py
from pathlib import Path

import numpy as np

from pipeline.models import TTS_SAMPLE_RATE


def pcm(seconds: float) -> bytes:
    sample_count = round(TTS_SAMPLE_RATE * seconds)
    return np.ones(sample_count, dtype="<i2").tobytes()


class FakeSynth:
    batch_size = 0

    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])

    def synthesize(self, text, voice_id, *, language="vi-VN"):
        return self.outputs.pop(0) if self.outputs else pcm(0.1)


def test_chunk_text_preserves_order_and_sentence_content():
    chunks = chunk_text("Hello world.\nHow are you? Fine!", "en-US", max_chars=18)
    assert "".join(chunks).replace(" ", "") == "Helloworld.Howareyou?Fine!"
    assert all(len(chunk) <= 18 for chunk in chunks)


def test_empty_and_oversized_text_are_rejected(tmp_path):
    with pytest.raises(InvalidTextError):
        run_text_to_voice(FakeSynth(), "  ", tmp_path, TextToVoiceOptions("v", "vi-VN"))
    with pytest.raises(TextTooLongError):
        run_text_to_voice(
            FakeSynth(), "x" * 50_001, tmp_path, TextToVoiceOptions("v", "vi-VN")
        )
```

- [ ] **Step 2: Run and verify failure for the missing runner**

Run: `pytest tests/test_text_to_voice.py -q`
Expected: FAIL during collection with `ModuleNotFoundError: pipeline.text_to_voice`.

- [ ] **Step 3: Implement options, results, validation, and language-specific chunking**

```python
@dataclass(frozen=True)
class TextToVoiceOptions:
    voice_id: str
    language: str
    max_characters: int = 50_000
    inter_chunk_silence_ms: int = 200
    max_chunk_characters: int = 1_000


@dataclass
class SpeechResult:
    wav_path: str
    mp3_path: str
    attempted_count: int
    spoken_count: int
    warnings: list[str] = field(default_factory=list)
```

`chunk_text()` normalizes CRLF to LF, trims outer whitespace, splits after the registry's terminators, then splits an oversized sentence at the last whitespace before `max_chunk_characters`; when no whitespace exists it splits at the hard limit. It never translates or changes character order.

- [ ] **Step 4: Add failing assembly, silence, partial-failure, cancellation, and MP3 command tests**

```python
def test_runner_joins_successful_chunks_with_200ms_silence(tmp_path, monkeypatch):
    def fake_encode_mp3(wav_path, mp3_path):
        Path(mp3_path).write_bytes(b"fake-mp3")

    monkeypatch.setattr(text_to_voice, "encode_mp3", fake_encode_mp3)
    synth = FakeSynth(outputs=[pcm(0.1), None, pcm(0.1)])

    result = run_text_to_voice(
        synth,
        "One. Two. Three.",
        tmp_path,
        TextToVoiceOptions("voice", "en-US"),
    )

    samples, rate = read_wav(Path(result.wav_path))
    assert len(samples) == int(rate * 0.4)  # 0.1 + 0.2 silence + 0.1
    assert result.attempted_count == 3
    assert result.spoken_count == 2
    assert Path(result.mp3_path).read_bytes() == b"fake-mp3"


def test_mp3_command_uses_configured_ffmpeg_and_audio_only(tmp_path):
    command = build_mp3_cmd(tmp_path / "output.wav", tmp_path / "output.mp3", ffmpeg="/bin/ffmpeg")
    assert command[:4] == ["/bin/ffmpeg", "-y", "-loglevel", "error"]
    assert command[-1].endswith("output.mp3")
    assert ["-codec:a", "libmp3lame"] == command[command.index("-codec:a"):command.index("-codec:a") + 2]
```

- [ ] **Step 5: Implement deterministic WAV assembly and MP3 export**

Call `synthesize_texts()` with the canonical language, discard `None` chunks, and fail with `SpeechServiceError` when every chunk is `None`. Concatenate successful int16 arrays with exactly `rate * 0.2` zero samples between successful chunks. Write `output.wav`, then run:

```python
def build_mp3_cmd(wav_path: Path, mp3_path: Path, *, ffmpeg: str) -> list[str]:
    return [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path),
        "-map", "0:a:0", "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path),
    ]
```

Check cancellation before synthesis, between chunks through `synthesize_texts`, before WAV write, and before MP3 encode. If MP3 encoding fails, delete both unpublished outputs before re-raising.

Add an FFmpeg-backed integration test (skip only when `resolve("ffmpeg")` or `resolve("ffprobe")` reports the binary is unavailable) that encodes a 0.1-second WAV, runs FFprobe JSON output, and asserts `codec_name == "mp3"`, `sample_rate == "24000"`, and `channels == 1`. The unit test above remains responsible for exact command construction.

- [ ] **Step 6: Add job-type-specific stages without breaking legacy percentage calls**

```python
VIDEO_STAGES = ("extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux")
TEXT_STAGES = ("prepare", "synthesize", "assemble", "export")
STAGES_BY_JOB_TYPE = {
    "video_dubbing": VIDEO_STAGES,
    "text_to_voice": TEXT_STAGES,
}
STAGE_WEIGHTS_BY_JOB_TYPE = {
    "video_dubbing": {"extract": 5, "transcribe": 25, "translate": 15, "synthesize": 40, "subtitle": 2, "assemble": 8, "mux": 5},
    "text_to_voice": {"prepare": 5, "synthesize": 75, "assemble": 10, "export": 10},
}
```

Make `overall_percent(stage, fraction, job_type="video_dubbing", fallback=None)` return the persisted fallback for an unknown restored stage instead of raising `ValueError`.

- [ ] **Step 7: Run the runner tests**

Run: `pytest tests/test_text_to_voice.py tests/test_progress.py -q`
Expected: PASS, including valid WAV metadata and deterministic MP3 command construction.

- [ ] **Step 8: Commit the Text → Voice pipeline**

```bash
git add pipeline/text_to_voice.py pipeline/errors.py pipeline/models.py \
  tests/test_text_to_voice.py tests/test_progress.py
git commit -m "feat: add text to voice audio runner"
```

---

### Task 6: Generic Job Results, Artifacts, and Runner Dispatch

**Files:**
- Create: `backend/job_contracts.py`
- Create: `tests/test_job_artifacts.py`
- Modify: `backend/job_manager.py:53-557`
- Modify: `backend/routes/jobs.py:20-237`
- Modify: `tests/test_job_manager.py:8-132`
- Modify: `tests/test_job_persistence.py:13-135`
- Modify: `tests/test_job_telemetry.py:1-220`
- Modify: `tests/test_security_paths.py:1-240`
- Modify: `tests/test_api.py:102-248`

**Interfaces:**
- Consumes: `PipelineResult`, target-language video runner, and job-type stage weights.
- Produces: `JobArtifact`, `JobRunResult`, and `JobRunner`.
- Produces: generalized `JobManager.start(..., job_type, target_language, input_label, backend_factory, runner)`.
- Produces: `GET /api/jobs/{job_id}/artifacts/{artifact_id}` and compatibility download aliases.

- [ ] **Step 1: Add failing artifact snapshot and old-job restore tests**

```python
# tests/test_job_artifacts.py
def test_snapshot_exposes_typed_artifacts_without_paths(tmp_path):
    artifact = JobArtifact("wav", "wav", "speech.wav", "audio/wav", str(tmp_path / "output.wav"))
    job = Job(
        id="text1", filename="Welcome…", input_label="Welcome…", workdir=tmp_path,
        voice_id="voice", job_type="text_to_voice", target_language="en-US",
        status="done", artifacts=[artifact],
    )
    snapshot = job.snapshot()
    assert snapshot["job_type"] == "text_to_voice"
    assert snapshot["target_language"] == "en-US"
    assert snapshot["artifacts"] == [{
        "id": "wav", "kind": "wav", "filename": "speech.wav", "media_type": "audio/wav"
    }]
    assert "path" not in snapshot["artifacts"][0]


def test_partial_success_is_marked_degraded_only_after_publication(tmp_path):
    job = Job(
        id="text2",
        filename="Short text",
        input_label="Short text",
        workdir=tmp_path,
        voice_id="voice",
        job_type="text_to_voice",
        target_language="en-US",
        status="running",
    )
    assert job.snapshot()["degraded"] is False
    job.publish_result(JobRunResult(artifacts=[], attempted_count=3, spoken_count=2))
    assert job.snapshot()["degraded"] is True


def test_old_persisted_job_defaults_to_vietnamese_video(tmp_path):
    _write_meta(tmp_path, "old", "done", video_path="output.mp4", srt_path="output.srt")
    manager = JobManager()
    manager.restore(tmp_path)
    restored = manager.get("old")
    assert restored.job_type == "video_dubbing"
    assert restored.target_language == "vi-VN"
```

- [ ] **Step 2: Run and verify missing artifact contracts**

Run: `pytest tests/test_job_artifacts.py tests/test_job_persistence.py -q`
Expected: FAIL because `JobArtifact` and new job fields are absent.

- [ ] **Step 3: Implement artifact and generic run-result contracts**

```python
# backend/job_contracts.py
from dataclasses import dataclass, field
from typing import Callable

from pipeline.models import CancelFn, ProgressFn


@dataclass(frozen=True)
class JobArtifact:
    id: str
    kind: str
    filename: str
    media_type: str
    path: str

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "filename": self.filename, "media_type": self.media_type}


@dataclass
class JobRunResult:
    artifacts: list[JobArtifact]
    warnings: list[str] = field(default_factory=list)
    attempted_count: int = 0
    spoken_count: int = 0

    @property
    def degraded(self) -> bool:
        return self.attempted_count > 0 and self.spoken_count < self.attempted_count


JobRunner = Callable[[object, ProgressFn, CancelFn], JobRunResult]
```

- [ ] **Step 4: Generalize `Job` snapshot, progress, result publication, and persistence**

Add `job_type="video_dubbing"`, `target_language="vi-VN"`, `input_label=""`, `artifacts=[]`, and `degraded=False`. `filename` remains a compatibility/display field and equals `input_label` for text jobs. Compute percent with `overall_percent(stage, fraction, self.job_type, fallback=self.percent)`. Replace `mark_persisted_result(...)` with `publish_result(JobRunResult)` and assign artifacts, counts, warnings, and `degraded=result.degraded` under `telemetry_lock` before setting terminal status. Persist artifact paths relative to `job.workdir`; on restore, reject absolute paths, `..` components, symlinks, and resolved paths outside or below a nested directory. Persist and restore `degraded`; old metadata defaults to `False`.

- [ ] **Step 5: Refactor manager execution around an injected typed runner**

```python
def start(
    self,
    *,
    filename: str,
    input_label: str,
    job_type: str,
    target_language: str,
    workdir: Path,
    voice_id: str,
    backend_factory,
    runner: JobRunner,
) -> Job:
    job = Job(
        id=uuid.uuid4().hex[:12], filename=filename, input_label=input_label,
        job_type=job_type, target_language=target_language,
        workdir=workdir, voice_id=voice_id,
        stage=STAGES_BY_JOB_TYPE[job_type][0],
    )
    future = self._ensure_executor().submit(self._run, job, backend_factory, runner)
    return job
```

`_run` builds the backend, captures runtime telemetry, calls `runner(backend, progress, job.cancel_event.is_set)`, publishes `JobRunResult`, advances the final stage to 100%, then flips status. Existing cancel/error ordering tests remain authoritative.

- [ ] **Step 6: Adapt the video route with a result adapter and preserve its API**

```python
def _video_job_result(result: PipelineResult, filename: str, language: str) -> JobRunResult:
    suffix = "vi" if language == "vi-VN" else "en"
    return JobRunResult(
        artifacts=[
            JobArtifact("video", "video", f"{Path(filename).stem}_{suffix}{Path(result.video_path).suffix}", "video/mp4", result.video_path),
            JobArtifact("subtitle", "subtitle", f"{Path(filename).stem}_{suffix}.srt", "application/x-subrip", result.srt_path),
        ],
        warnings=result.warnings,
        attempted_count=result.attempted_count,
        spoken_count=result.spoken_count,
    )
```

Pass a closure that calls `run_pipeline()` with the already-probed media and options. Compatibility `video_path`/`srt_path` properties are derived from artifacts during persistence and restore.

- [ ] **Step 7: Add a path-safe generic artifact route and make old downloads aliases**

Resolve the requested artifact from `job.artifacts` by exact ID; use the existing `_job_file()` direct-child, symlink, and resolved-root checks on its stored path. The response filename and media type come from the trusted persisted artifact, not request parameters. `/download/video` maps to artifact `video`; `/download/srt` maps to `subtitle`.

- [ ] **Step 8: Run job, API, persistence, telemetry, and security tests**

Run: `pytest tests/test_job_artifacts.py tests/test_job_manager.py tests/test_job_persistence.py tests/test_job_telemetry.py tests/test_security_paths.py tests/test_api.py -q`
Expected: PASS, including old JSON restore and terminal-result atomicity.

- [ ] **Step 9: Commit shared job infrastructure**

```bash
git add backend/job_contracts.py backend/job_manager.py backend/routes/jobs.py \
  tests/test_job_artifacts.py tests/test_job_manager.py tests/test_job_persistence.py \
  tests/test_job_telemetry.py tests/test_security_paths.py tests/test_api.py
git commit -m "refactor: generalize jobs and result artifacts"
```

---

### Task 7: Text Job API, Persistence, Cancellation, and Downloads

**Files:**
- Create: `backend/routes/text_jobs.py`
- Create: `tests/test_text_job_api.py`
- Modify: `backend/main.py:14-43`
- Modify: `backend/job_manager.py:53-557`
- Modify: `tests/conftest.py:1-180`
- Modify: `tests/test_queue_settings_shutdown.py:1-280`

**Interfaces:**
- Consumes: `run_text_to_voice()`, `TextToVoiceOptions`, `JobRunResult`, shared manager, voice capability checks.
- Produces: `POST /api/jobs/text` JSON contract.
- Produces: persisted `input.txt`, text excerpt `input_label`, WAV/MP3 artifacts, and shared SSE/cancel/delete behavior.

- [ ] **Step 1: Add failing request validation tests**

```python
# tests/test_text_job_api.py
import time

import numpy as np

from pipeline.audio import write_wav
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.text_to_voice import SpeechResult


def wait_until_terminal(client, job_id: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] in {"done", "error", "cancelled"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def fake_successful_text_runner(
    synthesizer,
    text,
    workdir,
    options,
    *,
    progress,
    should_cancel,
):
    wav_path = workdir / "output.wav"
    mp3_path = workdir / "output.mp3"
    write_wav(wav_path, np.zeros(TTS_SAMPLE_RATE // 10, dtype="<i2"), TTS_SAMPLE_RATE)
    mp3_path.write_bytes(b"ID3-fake-test-audio")
    progress("export", 1.0, "Đã xuất WAV và MP3")
    return SpeechResult(
        wav_path=str(wav_path),
        mp3_path=str(mp3_path),
        attempted_count=1,
        spoken_count=1,
    )


def test_text_job_rejects_empty_text_before_creating_workdir(client, jobs_dir):
    response = client.post("/api/jobs/text", json={
        "text": "  ", "voice_id": "en-US-AvaMultilingualNeural", "language": "en-US",
    })
    assert response.status_code == 400
    assert list(jobs_dir.iterdir()) == []


def test_text_job_rejects_unsupported_voice_language_pair(client):
    response = client.post("/api/jobs/text", json={
        "text": "Hello", "voice_id": "vi-VN-HoaiMyNeural", "language": "en-US",
    })
    assert response.status_code == 400
    assert "không hỗ trợ" in response.json()["detail"]


def test_text_job_rejects_more_than_50000_characters(client):
    response = client.post("/api/jobs/text", json={
        "text": "x" * 50_001, "voice_id": "en-US-AvaMultilingualNeural", "language": "en-US",
    })
    assert response.status_code == 413


def test_edge_text_job_does_not_require_gemini_key(client, monkeypatch):
    monkeypatch.setattr(text_jobs.settings, "gemini_api_key", "")
    monkeypatch.setattr(text_jobs.settings, "tts_provider", "edge")
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)
    response = client.post("/api/jobs/text", json={
        "text": "Local text job.",
        "voice_id": "en-US-AvaMultilingualNeural",
        "language": "en-US",
    })
    assert response.status_code == 200
```

- [ ] **Step 2: Run and verify the endpoint is missing**

Run: `pytest tests/test_text_job_api.py -q`
Expected: FAIL with HTTP 404 for `/api/jobs/text`.

- [ ] **Step 3: Implement the Pydantic request and validate before directory allocation**

```python
class TextJobRequest(BaseModel):
    text: str
    voice_id: str
    language: str


@router.post("/api/jobs/text")
def create_text_job(request: TextJobRequest) -> dict:
    text = request.text.strip()
    if not text:
        raise HTTPException(400, "Hãy nhập nội dung cần đọc.")
    if len(text) > 50_000:
        raise HTTPException(413, "Nội dung vượt quá giới hạn 50.000 ký tự.")
    language = normalize_language_code(request.language)
    clone_provider = settings.resolved_clone_provider
    if not is_available(request.voice_id, settings.tts_provider, clone_provider, language=language):
        raise HTTPException(400, "Giọng đọc không hỗ trợ ngôn ngữ đã chọn.")
```

Only after validation: prune, reserve/create a workdir, write normalized text to `input.txt`, compute an 80-character whitespace-normalized excerpt, route the provider, and construct `TextToVoiceOptions`.

- [ ] **Step 4: Add failing successful-job and privacy tests**

```python
def test_text_job_runs_in_shared_queue_and_publishes_wav_and_mp3(client, monkeypatch, jobs_dir):
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)
    response = client.post("/api/jobs/text", json={
        "text": "Welcome to the show.",
        "voice_id": "en-US-AvaMultilingualNeural",
        "language": "en-US",
    })
    job_id = response.json()["job_id"]
    job = wait_until_terminal(client, job_id)
    assert job["job_type"] == "text_to_voice"
    assert {artifact["kind"] for artifact in job["artifacts"]} == {"wav", "mp3"}


def test_snapshot_and_sse_never_expose_full_input(client, monkeypatch):
    secret_text = "PRIVATE-NARRATION-" * 20
    monkeypatch.setattr(text_jobs, "run_text_to_voice", fake_successful_text_runner)
    response = client.post("/api/jobs/text", json={
        "text": secret_text,
        "voice_id": "en-US-AvaMultilingualNeural",
        "language": "en-US",
    })
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert secret_text not in client.get(f"/api/jobs/{job_id}").text
    assert secret_text not in client.get(f"/api/jobs/{job_id}/telemetry").text
    assert secret_text not in client.get(f"/api/jobs/{job_id}/events").text
```

- [ ] **Step 5: Adapt `SpeechResult` to `JobRunResult` and register with the shared manager**

```python
def _text_job_result(result: SpeechResult, input_label: str) -> JobRunResult:
    return JobRunResult(
        artifacts=[
            JobArtifact("wav", "wav", f"{input_label[:40] or 'speech'}.wav", "audio/wav", result.wav_path),
            JobArtifact("mp3", "mp3", f"{input_label[:40] or 'speech'}.mp3", "audio/mpeg", result.mp3_path),
        ],
        warnings=result.warnings,
        attempted_count=result.attempted_count,
        spoken_count=result.spoken_count,
    )
```

Sanitize download filenames separately from display excerpts; the actual files remain fixed `output.wav` and `output.mp3` inside the workdir. Register `text_jobs.router` before the static mount.

- [ ] **Step 6: Test mixed queue concurrency, queued cancellation, restore, and deletion**

Add tests that submit one stubbed video job and one stubbed text job to the same `JobManager(max_workers=1)`, assert the second queues, cancel it before execution, and confirm deleting a completed text job removes `input.txt`, WAV, MP3, and `job.json` through the existing exclusive deletion claim.

- [ ] **Step 7: Run all text-job and shared-lifecycle tests**

Run: `pytest tests/test_text_job_api.py tests/test_job_manager.py tests/test_job_persistence.py tests/test_queue_settings_shutdown.py tests/test_security_paths.py -q`
Expected: PASS with text jobs visible through existing list/status/SSE endpoints.

- [ ] **Step 8: Commit the text-job API**

```bash
git add backend/routes/text_jobs.py backend/main.py backend/job_manager.py \
  tests/test_text_job_api.py tests/conftest.py tests/test_queue_settings_shutdown.py \
  tests/test_job_manager.py tests/test_job_persistence.py tests/test_security_paths.py
git commit -m "feat: add persistent text to voice jobs"
```

---

### Task 8: Multilingual Fixed Previews and On-Demand Text Preview

**Files:**
- Create: `pipeline/voice_previews.py`
- Create: `tests/test_voice_previews.py`
- Modify: `pipeline/audio.py:62-84`
- Modify: `backend/routes/voices.py:23-154`
- Modify: `backend/main.py:18-32`
- Modify: `scripts/generate_voice_previews.py:1-180`
- Modify: `tests/test_voice_preview_api.py:1-220`
- Modify: `tests/test_voice_upload_contract.py:1-180`
- Modify: `tests/test_generate_voice_previews.py:1-200`

**Interfaces:**
- Consumes: language registry, language-aware backends, `speech_activity.preview(provider)`, voice routing.
- Produces: `preview_path(voice_id, language)`, thread-safe preview status, background regeneration, and WAV byte encoding.
- Produces: `GET /api/voices/{voice_id}/preview?language=...` and `POST /api/voices/{voice_id}/preview-text`.
- Produces: `POST /api/voices/{voice_id}/preview/regenerate?language=...` for retrying a failed fixed preview.

- [ ] **Step 1: Add failing language-specific path and fixed-preview status tests**

```python
# tests/test_voice_previews.py
def test_preview_paths_are_separate_per_language(tmp_path):
    store = VoicePreviewStore(tmp_path)
    assert store.path("clone-safe", "vi-VN") == tmp_path / "clone-safe" / "vi-VN.wav"
    assert store.path("clone-safe", "en-US") == tmp_path / "clone-safe" / "en-US.wav"


def test_unknown_voice_or_language_cannot_escape_preview_root(tmp_path):
    store = VoicePreviewStore(tmp_path)
    with pytest.raises(ValueError):
        store.path("../../outside", "vi-VN")
    with pytest.raises(ValueError):
        store.path("clone-safe", "../../outside")
```

- [ ] **Step 2: Run and verify the preview store is absent**

Run: `pytest tests/test_voice_previews.py -q`
Expected: FAIL during collection with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the preview store and state transitions**

`VoicePreviewStore` receives the trusted preview root and accepts only catalog-style IDs matching `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`; it canonicalizes language through `require_language()` before constructing a path. API callers additionally require the voice to exist in the catalog. The store keeps `pending|ready|error` in a lock-protected map; an existing file always reports `ready`. `generate_languages(voice_id, languages, synthesizer)` creates the voice directory, sets each language pending, synthesizes `LanguageSpec.preview_text`, writes a temporary WAV in the same directory, atomically replaces `<language>.wav`, and records ready/error without failing voice creation.

- [ ] **Step 4: Add PCM-to-WAV bytes for non-persistent preview responses**

```python
def pcm_to_wav_bytes(pcm: bytes, rate: int = TTS_SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()
```

Test the returned RIFF header, mono channel count, 16-bit width, and sample rate using `wave.open(io.BytesIO(value))`.

- [ ] **Step 5: Update voice listing and fixed-preview routes with compatibility defaults**

`GET /api/voices?language=en-US` returns `preview_urls` and `preview_status` maps plus the old scalar `preview_url` for the selected/default language. `GET /api/voices/{voice_id}/preview` normalizes a defaulted query parameter `language="vi-VN"`, serves only the store's exact file, and names it `<voice_id>-<language>.wav`.

Creating or replacing a custom voice first removes all old per-language previews and cached reference text, then starts one background call for both `("vi-VN", "en-US")`; deleting it removes the whole trusted preview subdirectory and clears status/cache entries.

- [ ] **Step 6: Add failing on-demand preview validation, truncation, busy, and no-persistence tests**

```python
from contextlib import contextmanager

from pipeline.errors import PreviewBusyError
from pipeline.models import TTS_SAMPLE_RATE


def pcm() -> bytes:
    return b"\x00\x00" * (TTS_SAMPLE_RATE // 10)


@contextmanager
def busy_preview_context(provider):
    raise PreviewBusyError(provider)
    yield


def test_text_preview_truncates_to_first_sentence_and_returns_wav(client, monkeypatch):
    seen = []
    monkeypatch.setattr(voices, "_synthesize_preview_text", lambda text, voice, language: seen.append(text) or pcm())
    response = client.post(
        "/api/voices/en-US-AvaMultilingualNeural/preview-text",
        json={"text": "First sentence. Second sentence.", "language": "en-US"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert seen == ["First sentence."]


def test_busy_preview_returns_retryable_409(client, monkeypatch):
    monkeypatch.setattr(voices.speech_activity, "preview", busy_preview_context)
    response = client.post(
        "/api/voices/en-US-AvaMultilingualNeural/preview-text",
        json={"text": "Hello", "language": "en-US"},
    )
    assert response.status_code == 409
    assert "thử lại" in response.json()["detail"].lower()
```

- [ ] **Step 7: Implement bounded on-demand preview**

Add a Pydantic body with `text` and `language`. Reject empty input, validate the voice/language pair, and select `text[:sentence_end]` only when the first terminator occurs within the first 300 characters; otherwise select exactly `text[:300]`. Route/build the selected provider and enter `speech_activity.preview(backend.engine)` non-blockingly. Run synthesis through a bounded future with a 30-second timeout, return `Response(content=pcm_to_wav_bytes(pcm), media_type="audio/wav")`, map `PreviewBusyError` to 409, and map timeout to 504. Do not write the text or WAV to disk and do not log raw input.

Add `POST /api/voices/{voice_id}/preview/regenerate?language=...`: validate the same voice/language pair, atomically transition only that entry to `pending`, dispatch `VoicePreviewStore.generate_languages()` in the existing background executor, and return `202 {"status": "pending"}`. Return `409` when that same preview is already pending.

- [ ] **Step 8: Regenerate preview script behavior and test the full preview slice**

Update `scripts/generate_voice_previews.py` to iterate each voice's supported languages and use the same store/path/text rules as the API.

Run: `pytest tests/test_voice_previews.py tests/test_voice_preview_api.py tests/test_voice_upload_contract.py tests/test_generate_voice_previews.py tests/test_api.py -q`
Expected: PASS for old Vietnamese preview URLs and new Vietnamese/English maps.

- [ ] **Step 9: Commit preview support**

```bash
git add pipeline/voice_previews.py pipeline/audio.py backend/routes/voices.py backend/main.py \
  scripts/generate_voice_previews.py tests/test_voice_previews.py \
  tests/test_voice_preview_api.py tests/test_voice_upload_contract.py \
  tests/test_generate_voice_previews.py tests/test_api.py
git commit -m "feat: add multilingual voice previews"
```

---

### Task 9: Accessible Two-Tab New Job Composer and Voice Lab Language UX

**Files:**
- Modify: `web/static/index.html:65-116`
- Modify: `web/static/style.css:142-272,349-468`
- Modify: `web/static/app.js:4-80,273-371`
- Create: `tests/test_multilingual_ui_contract.py`
- Modify: `tests/test_ui_contract.py:4-31`
- Modify: `tests/test_frontend_runtime_contract.py:15-51`
- Modify: `tests/test_upload_browser_contract.py:1-180`
- Modify: `tests/test_frontend_boot.py:1-220`

**Interfaces:**
- Consumes: `GET /api/languages`, language-filtered voice payloads, video `target_language`, text-job API, and preview APIs.
- Produces: accessible `#jobModeTabs`, `#videoJobPanel`, `#textJobPanel`, `#videoLanguage`, `#textLanguage`, `#textInput`, `#textPreviewButton`, and `#textJobForm`.
- Produces: one language-aware Voice Lab player and retained per-tab draft state.

- [ ] **Step 1: Add failing structural and JavaScript contract tests**

```python
# tests/test_multilingual_ui_contract.py
from pathlib import Path

HTML = Path("web/static/index.html").read_text(encoding="utf-8")
JS = Path("web/static/app.js").read_text(encoding="utf-8")


def test_new_job_uses_accessible_video_and_text_tabs():
    assert 'id="jobModeTabs" role="tablist"' in HTML
    assert 'id="videoJobTab" role="tab"' in HTML
    assert 'id="textJobTab" role="tab"' in HTML
    assert 'id="videoJobPanel" role="tabpanel"' in HTML
    assert 'id="textJobPanel" role="tabpanel"' in HTML
    assert "ArrowLeft" in JS and "ArrowRight" in JS and "Home" in JS and "End" in JS


def test_text_form_has_language_voice_preview_and_submit_controls():
    for element_id in ("textInput", "textCharacterCount", "textLanguage", "textVoiceSelect", "textPreviewButton", "textStartButton"):
        assert f'id="{element_id}"' in HTML
    assert 'apiJson("/api/jobs/text"' in JS
    assert "AbortController" in JS
```

- [ ] **Step 2: Run and verify the new UI contract fails**

Run: `pytest tests/test_multilingual_ui_contract.py tests/test_ui_contract.py -q`
Expected: FAIL because the tabs and text controls are absent.

- [ ] **Step 3: Replace the single upload form with the approved tab composer**

Use two native `<button type="button" role="tab">` controls with `aria-selected`, `aria-controls`, and `tabindex`. Each panel owns its inputs and one primary CTA. Keep the current video dropzone and XHR upload inside `#videoJobPanel`; add the labeled textarea, character/estimated-duration status, language/voice selects, fixed preview, text preview, and submit action inside `#textJobPanel`.

- [ ] **Step 4: Implement tab state, keyboard navigation, and separate drafts**

```javascript
const JOB_MODES = ["video", "text"];
state.jobDrafts = {
  video: { language: "vi-VN", voiceId: "" },
  text: { language: "vi-VN", voiceId: "", text: "" },
};

function activateJobMode(mode, { focus = false } = {}) {
  state.jobMode = JOB_MODES.includes(mode) ? mode : "video";
  $$("[role=tab]", $("#jobModeTabs")).forEach(tab => {
    const active = tab.dataset.mode === state.jobMode;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (focus && active) tab.focus();
  });
  $("#videoJobPanel").hidden = state.jobMode !== "video";
  $("#textJobPanel").hidden = state.jobMode !== "text";
}
```

Handle Left/Right/Home/End without page scrolling. Switching tabs must not clear selected files, text, language, or voice.

- [ ] **Step 5: Load languages once and filter voice selectors without losing compatible selections**

Fetch languages and all voices during initialization. `voicesForLanguage(code)` filters `supported_languages`. When language changes, retain the selected voice only if compatible; otherwise select the first compatible voice and announce the reset. Keep `localStorage` keys separate: `sub.video.language`, `sub.video.voice`, `sub.text.language`, `sub.text.voice`, and `sub.voiceLab.language`.

Compute the visible duration estimate locally from trimmed text with registry UI rates (`vi-VN: 16.6` characters/second, `en-US: 14.0` characters/second), round up to whole seconds, and label it as an estimate rather than a promised output duration.

- [ ] **Step 6: Submit each workflow through its correct transport**

Video remains XHR and appends `target_language`. Text uses JSON `fetch` through `apiJson`; disable only the active submit button while pending. On success, clear text but keep text language/voice, refresh/select the new job, and announce “Đã thêm job giọng đọc vào hàng đợi.”

- [ ] **Step 7: Implement one-at-a-time fixed and on-demand preview lifecycle**

Keep one active audio object globally. Fixed preview chooses `voice.preview_urls[selectedLanguage]`. On-demand preview aborts the previous controller, posts the current text/language, converts the WAV blob to an object URL, revokes old URLs, and exposes loading/playing/stopped/retry states through button text plus the existing polite live region. Changing text, language, or voice aborts an in-flight preview and returns the control to idle.

- [ ] **Step 8: Add Voice Lab language selection and preview states**

Add `#voiceLabLanguage`; render only custom voices supporting it. Use `preview_status[language]` to show “Đang tạo bản nghe thử”, “Nghe thử”, or “Tạo lại demo”. The retry action posts to `/api/voices/${encodeURIComponent(voice.id)}/preview/regenerate?language=${encodeURIComponent(language)}`, changes to pending immediately, and refreshes the voice catalog until ready/error. Do not disable selecting a voice because its fixed preview is pending/error.

- [ ] **Step 9: Add responsive and accessibility styles**

Use the existing semantic tokens. Tabs are a two-column segmented control on desktop and mobile, controls are at least 44 px high, textarea is at least 160 px on mobile, labels remain visible, and errors render next to the active form. Add visible `:focus-visible` styles and `prefers-reduced-motion` coverage. Do not create new page-level horizontal scrolling.

- [ ] **Step 10: Run static UI and JavaScript checks**

Run: `pytest tests/test_multilingual_ui_contract.py tests/test_ui_contract.py tests/test_frontend_runtime_contract.py tests/test_upload_browser_contract.py tests/test_frontend_boot.py -q`
Run: `node --check web/static/app.js`
Expected: both commands PASS.

- [ ] **Step 11: Commit the composer and Voice Lab UX**

```bash
git add web/static/index.html web/static/style.css web/static/app.js \
  tests/test_multilingual_ui_contract.py tests/test_ui_contract.py \
  tests/test_frontend_runtime_contract.py tests/test_upload_browser_contract.py \
  tests/test_frontend_boot.py
git commit -m "feat: add video and text job composer"
```

---

### Task 10: Shared Queue, Typed Artifacts, and Adaptive Execution Graph

**Files:**
- Modify: `web/static/index.html:55-104`
- Modify: `web/static/style.css:142-219,287-348,394-468`
- Modify: `web/static/app.js:6-39,82-270`
- Modify: `tests/test_multilingual_ui_contract.py`
- Modify: `tests/test_frontend_requested_fixes.py:1-220`
- Modify: `tests/test_frontend_bugfix_regression.py:1-240`
- Modify: `tests/test_ui_structure.py:1-240`
- Modify: `tests/test_ui_motion_contract.py:1-180`

**Interfaces:**
- Consumes: generalized job snapshot and public artifacts from Tasks 6–7.
- Produces: `STAGES_BY_JOB_TYPE`, typed job cards/downloads, adaptive Selected Job metadata, and dynamic Execution Graph nodes.
- Preserves: per-job SSE live updates and the single queue list.

- [ ] **Step 1: Add failing queue and graph contract tests**

```python
def test_queue_copy_is_generic_and_job_cards_render_type_and_language():
    assert "JOB QUEUE" in HTML
    assert "HÀNG ĐỢI XỬ LÝ" in HTML
    assert "VIDEO QUEUE" not in HTML
    assert "jobTypeLabel(job)" in JS
    assert "languageName(job.target_language)" in JS


def test_graph_stage_definitions_cover_both_job_types():
    assert 'video_dubbing: ["extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux"]' in JS
    assert 'text_to_voice: ["prepare", "synthesize", "assemble", "export"]' in JS
    assert "renderGraphNodes(job)" in JS


def test_done_job_renders_public_artifacts_instead_of_hardcoded_video_links():
    assert "job.artifacts" in JS
    assert "/artifacts/${encodeURIComponent(artifact.id)}" in JS
```

- [ ] **Step 2: Run and verify old video-only queue assumptions fail**

Run: `pytest tests/test_multilingual_ui_contract.py tests/test_frontend_requested_fixes.py tests/test_ui_structure.py -q`
Expected: FAIL on generic copy, typed artifacts, and text stages.

- [ ] **Step 3: Introduce stage metadata maps and dynamic graph DOM**

```javascript
const STAGES_BY_JOB_TYPE = {
  video_dubbing: ["extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux"],
  text_to_voice: ["prepare", "synthesize", "assemble", "export"],
};

function stagesFor(job) {
  return STAGES_BY_JOB_TYPE[job?.job_type] || STAGES_BY_JOB_TYPE.video_dubbing;
}
```

Replace seven static graph nodes with one empty `#pipelineGraph`; `renderGraphNodes(job)` creates nodes and rails using trusted local metadata and `textContent`. Cache the last rendered job type to avoid rebuilding on every 1-second timer tick.

- [ ] **Step 4: Render queue cards with job type, language, and artifact actions**

Use `input_label || filename` as the title, a visible type badge with icon and text, canonical language display, and the existing voice/status/progress fields. Render artifact links from `job.artifacts` using known `kind` labels; construct only the route with encoded job/artifact IDs. Keep `data-job-id`, `data-status`, and add `data-job-type`.

- [ ] **Step 5: Adapt Selected Job and terminal notifications**

Text jobs show a document/audio icon, excerpt, language, voice, chunk count, WAV/MP3 downloads, and text stage names. Video jobs retain filename, segment count, video/SRT actions, and seven stages. Replace “Video đã xử lý xong” with a job-type-aware message. Delete/cancel confirmations say “job” rather than “video/project”.

- [ ] **Step 6: Make completion independent of the persisted final stage name**

For `status === "done"`, mark every node and rail for the selected job type complete. For active/error states, derive current index from `stagesFor(job)` and fall back to a neutral graph detail when a restored unknown stage is encountered. This guarantees both video `mux` and text `export` light correctly.

- [ ] **Step 7: Preserve the graph-only mobile horizontal scroll contract**

Set the dynamic graph grid column count via `style.setProperty("--graph-columns", stages.length)`. Keep overflow on `.graph-scroll-shell`/`.graph-track`; all cards, labels, and buttons wrap inside 360 px. Type/status badges include text so color is not the only signal.

- [ ] **Step 8: Run queue, graph, responsive-contract, and syntax tests**

Run: `pytest tests/test_multilingual_ui_contract.py tests/test_frontend_requested_fixes.py tests/test_frontend_bugfix_regression.py tests/test_ui_structure.py tests/test_ui_motion_contract.py -q`
Run: `node --check web/static/app.js`
Expected: PASS, including the prior Export-node and Voice Lab overflow regressions.

- [ ] **Step 9: Commit shared queue rendering**

```bash
git add web/static/index.html web/static/style.css web/static/app.js \
  tests/test_multilingual_ui_contract.py tests/test_frontend_requested_fixes.py \
  tests/test_frontend_bugfix_regression.py tests/test_ui_structure.py \
  tests/test_ui_motion_contract.py
git commit -m "feat: generalize queue and execution graph"
```

---

### Task 11: Documentation, Full Regression, Real Smoke Tests, and MCP Playwright

**Files:**
- Modify: `README.md:1-210`
- Modify: `HUONG_DAN_MAC.md:1-260`
- Modify: `.env.example:1-160`
- Modify: `pyproject.toml:1-120`
- Modify: `scripts/check_setup.py:1-260`
- Modify: `tests/test_check_setup.py:1-220`
- Create: `tests/test_provider_smoke.py`
- Modify: `tests/test_ui_contract.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Consumes: all completed backend and frontend contracts.
- Produces: `capabilities_summary() -> str`, user-facing Vietnamese setup/usage documentation, opt-in provider smoke tests, complete regression evidence, and browser QA evidence.

- [ ] **Step 1: Add failing documentation/setup contract assertions**

```python
# tests/test_check_setup.py
from scripts import check_setup


def test_setup_output_names_both_supported_languages_and_audio_formats():
    output = check_setup.capabilities_summary()
    assert "vi-VN" in output
    assert "en-US" in output
    assert "WAV" in output
    assert "MP3" in output
```

Add static documentation tests only for critical commands/API names; do not snapshot prose.

- [ ] **Step 2: Update documentation and setup diagnostics**

Implement `capabilities_summary()` as a deterministic four-line summary naming `vi-VN`, `en-US`, video dubbing, Text → Voice, WAV, and MP3; print it from the setup script before provider-specific checks. Document the two New Job tabs, video target-language selection, Text → Voice limits, both output formats, shared queue, fixed/on-demand previews, provider capability filtering, and language-extension procedure. Update examples from “always Vietnamese” to “selected target language”. Explain that video jobs require Gemini while local/Edge Text → Voice can work without a Gemini key.

- [ ] **Step 3: Run formatting and deterministic static checks**

Run: `node --check web/static/app.js`
Run: `git diff --check`
Expected: both PASS with no syntax or whitespace errors.

- [ ] **Step 4: Run focused backend integration suites**

Run: `pytest tests/test_languages.py tests/test_backends.py tests/test_backend_batch_forwarding.py tests/test_runner.py tests/test_translate_alignment.py tests/test_speech_synthesis.py tests/test_speech_runtime.py tests/test_text_to_voice.py tests/test_job_manager.py tests/test_job_persistence.py tests/test_job_artifacts.py tests/test_text_job_api.py tests/test_voice_previews.py tests/test_voice_preview_api.py tests/test_security_paths.py -q`
Expected: PASS.

- [ ] **Step 5: Run the complete deterministic pytest suite**

Run: `pytest -q`
Expected: PASS; only documented platform/model skips are allowed. Any new warning or unexpected skip is a failure to investigate.

- [ ] **Step 6: Run opt-in provider smoke tests**

Register the `provider_smoke` marker in `pyproject.toml`, then add explicit opt-in tests:

```python
# tests/test_provider_smoke.py
import os

import pytest

from backend.config import settings
from pipeline.backends import build_backend
from pipeline.edge_speech import EdgeSynthesizer

pytestmark = pytest.mark.provider_smoke


def _require_smoke_enabled():
    if os.getenv("RUN_PROVIDER_SMOKE") != "1":
        pytest.skip("set RUN_PROVIDER_SMOKE=1 to call real providers")


def test_real_edge_english_returns_pcm():
    _require_smoke_enabled()
    pcm = EdgeSynthesizer().synthesize(
        "This is an English smoke test.",
        "en-US-AvaMultilingualNeural",
        language="en-US",
    )
    assert len(pcm) >= 2 and len(pcm) % 2 == 0


@pytest.mark.parametrize(("language", "text"), [
    ("vi-VN", "Xin chào."),
    ("en-US", "Hello."),
])
def test_real_omnivoice_clone_returns_pcm(language, text):
    _require_smoke_enabled()
    voice_id = os.getenv("PROVIDER_SMOKE_CLONE_VOICE_ID")
    if not voice_id:
        pytest.skip("set PROVIDER_SMOKE_CLONE_VOICE_ID to an authorized local clone")
    backend = build_backend(settings.provider_config_for("omnivoice"))
    pcm = backend.synthesize(text, voice_id, language=language)
    assert len(pcm) >= 2 and len(pcm) % 2 == 0


@pytest.mark.parametrize(("target", "source", "expected_token"), [
    ("vi-VN", "Hello", "xin chào"),
    ("en-US", "Xin chào", "hello"),
])
def test_real_gemini_translation_is_aligned(target, source, expected_token):
    _require_smoke_enabled()
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY is not configured")
    backend = build_backend(settings.provider_config_for("edge"))
    translated = backend.translate([source], [1.0], target_language=target)
    assert len(translated) == 1
    assert expected_token in translated[0].strip().casefold()
```

With configured providers and authorized clone samples, run:

```bash
RUN_PROVIDER_SMOKE=1 pytest -q -m provider_smoke
```

Expected: English Edge produces non-empty PCM; one custom OmniVoice sample produces valid Vietnamese and English PCM; Gemini translation returns aligned Vietnamese and English target output. If a paid/network provider is intentionally unavailable, record that exact skip separately and do not represent it as a pass.

- [ ] **Step 7: Start the local QA server**

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Expected: server reaches startup without provider-validation errors and `GET /api/languages` returns both languages.

- [ ] **Step 8: Verify desktop UX with MCP Playwright at 1440×1000**

Use MCP Playwright, not ad-hoc browser automation:

1. Open the exact URL printed by Uvicorn, normally `http://127.0.0.1:8000`; if port 8000 was unavailable, restart once on an explicitly chosen free port and use that printed URL.
2. Confirm Video Dubbing and Text → Voice tabs expose correct `role`, `aria-selected`, roving `tabindex`, and keyboard Left/Right/Home/End behavior.
3. Enter text, switch tabs twice, and confirm both text and selected video draft survive.
4. Change language to English and confirm every voice option advertises `en-US`; select Vietnamese and confirm incompatible native-English-only choices disappear.
5. Play a fixed English custom-voice preview; confirm one audio plays and the button returns to idle on end.
6. Request an on-demand preview; assert one POST occurs, no job appears, loading/playing states announce, and changing language aborts the request.
7. Submit a short real English Edge Text → Voice input; observe live SSE progress without F5, then download and inspect WAV and MP3.
8. Submit the authorized short Gemini video smoke fixture targeting English; observe live progress, then download video and SRT. If the real provider precondition is unavailable, record this check as blocked rather than passed.
9. Select each job and confirm the Execution Graph renders 4 or 7 stages and all nodes/links complete at terminal success.
10. Emulate `prefers-reduced-motion: reduce` and repeat tab switching plus graph completion; verify state changes do not depend on animation events.
11. Assert no application console errors.

Correct the first URL instruction during execution based on the actual bound port; never invent a second server if 8000 is active.

- [ ] **Step 9: Verify mobile UX with MCP Playwright at 360×800**

Repeat tab, language, voice, preview, queue, Selected Job, and graph completion checks. Assert `document.documentElement.scrollWidth === document.documentElement.clientWidth`; only the graph shell may have `scrollWidth > clientWidth`. Verify all primary controls are at least 44 CSS px high and Voice Lab names/actions stay inside their cards.

- [ ] **Step 10: Run an independent two-axis review**

Dispatch one reviewer for spec compliance against the approved design and another reviewer for repository standards/regressions. Resolve every Critical or Important finding, rerun the affected focused tests, then rerun `pytest -q`, `node --check`, `git diff --check`, and both MCP Playwright viewport checks.

- [ ] **Step 11: Commit documentation and final QA contracts**

```bash
git add README.md HUONG_DAN_MAC.md .env.example pyproject.toml scripts/check_setup.py \
  tests/test_check_setup.py tests/test_provider_smoke.py tests/test_ui_contract.py \
  tests/test_api.py tests/test_runner.py
git commit -m "docs: document multilingual audio workflows"
```

- [ ] **Step 12: Record final evidence before integration**

Capture the exact pytest pass/skip count, Node syntax result, diff-check result, smoke-test availability/result, MCP Playwright desktop/mobile assertions, console-error count, and independent review verdict in the implementation task's final handoff. Do not merge or push as part of this task unless the user explicitly asks after reviewing that evidence.
