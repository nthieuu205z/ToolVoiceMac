# Multilingual Text-to-Voice and Video Dubbing Design

**Date:** 2026-09-05  
**Project:** ToolVietSubMac  
**Status:** Approved in design discussion; awaiting written-spec review

## 1. Summary

ToolVietSubMac will support two job types in one shared processing queue:

1. `video_dubbing`: upload a video, automatically detect the source language, translate to a selected target language, synthesize the selected voice, and export a dubbed video plus subtitles.
2. `text_to_voice`: submit text in a selected language, synthesize it without translation, and export both WAV and MP3 audio.

Vietnamese (`vi-VN`) and English (`en-US`) are the first supported target languages. Language handling must be registry-driven so later languages can be added without adding another set of hard-coded branches throughout the pipeline.

The implementation will preserve the current Vietnamese video workflow and existing API clients while introducing target-neutral language, voice, job, and artifact contracts.

## 2. Goals

- Add Text → Voice as a first-class asynchronous job.
- Let users dub videos to Vietnamese or English.
- Keep video and text jobs in one queue, scheduler, persistence model, SSE update system, and cancellation flow.
- Export WAV and MP3 for every successful Text → Voice job.
- Filter voices by language and provider capability before job creation.
- Support pre-rendered voice previews for every supported language.
- Support an on-demand preview of the beginning of Text → Voice input without creating a queue job.
- Remove Vietnamese-specific assumptions from shared domain and provider interfaces.
- Make adding a future language primarily a registry/catalog/configuration change plus provider verification tests.
- Preserve the existing local-first privacy and filesystem safety guarantees.

## 3. Non-goals

- Automatic language detection for Text → Voice input.
- Translating Text → Voice input before synthesis.
- SSML, per-word pronunciation controls, timeline editing, or a waveform editor.
- User-configurable chunk pauses in the first release.
- Audio formats other than WAV and MP3.
- Replacing the current queue with a general-purpose workflow/DAG engine.
- Silently changing the selected voice or provider after a synthesis failure.
- Supporting a language that has not been verified for at least one configured TTS provider.

## 4. Locked Product Decisions

- English applies to both Text → Voice and video dubbing.
- Text → Voice exports both WAV and MP3.
- Video and text jobs share one queue, renamed from **VIDEO QUEUE** to **JOB QUEUE / HÀNG ĐỢI XỬ LÝ**.
- The New Job panel uses two accessible tabs: **Video Dubbing** and **Text → Voice**.
- A job card carries a visible `VIDEO` or `TEXT` type badge and a language label.
- Video input language remains automatic; the user explicitly selects the output language.
- Text → Voice reads the submitted text as-is; the user explicitly selects its spoken language.
- Voice Lab includes a language selector. Clicking a voice preview plays the pre-rendered sample for that selected language.
- Custom voices receive pre-rendered Vietnamese and English previews in the background immediately after creation.
- Text → Voice includes an on-demand preview action that reads at most the first 300 characters of the current text. It does not create or persist a job.
- The current dark Sub. Ops Console visual language remains in place.

## 5. Current-State Constraints

The current pipeline is a seven-stage video-only runner in `pipeline/runner.py`. `PipelineOptions`, `PipelineResult`, `Job`, and the download API all assume video input and video/SRT output.

The following Vietnamese-specific assumptions must be removed carefully:

- `Segment.text_vi` represents translated/spoken target text.
- Gemini translation prompts and schemas are fixed to Vietnamese.
- Gemini TTS defaults to `vi-VN`.
- OmniVoice generation passes `Vietnamese` directly.
- `Voice` entries do not describe supported languages.
- `synthesize_segments()` combines raw synthesis with video timing and speed fitting.

The Git working tree already contains substantial unrelated/uncommitted work. Implementation must begin in an isolated `codex/` worktree created from the intended current baseline and must not stage or overwrite unrelated changes.

## 6. Architectural Approach

Use a capability-driven shared job platform with specialized runners.

```text
New Job tabs
  ├─ Video Dubbing ── POST /api/jobs ──────┐
  └─ Text → Voice ─── POST /api/jobs/text ─┤
                                            v
                                  Shared JobManager
                               queue / SSE / cancel / restore
                                      /            \
                          VideoDubbingRunner    TextToVoiceRunner
                                      \            /
                                       Job artifacts
```

Three registries/interfaces cut across both runners:

- `LanguageRegistry`: canonical language metadata and provider mappings.
- `VoiceCatalog`: provider, custom/preset status, supported languages, and preview state.
- `SpeechSynthesizer`: target-language-aware single and batch synthesis.

The Text → Voice runner must not create fake timed `Segment` objects and call the video fitting path. Video fitting may speed up, slow down, truncate, or borrow timeline gaps; those behaviors would damage unconstrained text narration.

## 7. Language Model

Create `pipeline/languages.py` with an immutable `LanguageSpec` and a registry lookup API.

```python
@dataclass(frozen=True)
class LanguageSpec:
    code: str                  # canonical BCP-47 code
    display_name: str          # Vietnamese UI label
    english_name: str
    gemini_tts_code: str
    omnivoice_name: str
    preview_text: str
    sentence_terminators: tuple[str, ...]
```

Initial entries:

| Canonical code | UI label | Gemini TTS | OmniVoice | Preview language |
|---|---|---|---|---|
| `vi-VN` | Tiếng Việt | `vi-VN` | `Vietnamese` | Vietnamese fixed sentence |
| `en-US` | English (US) | `en-US` | `English` | English fixed sentence |

Public operations:

```python
def available_languages() -> tuple[LanguageSpec, ...]: ...
def require_language(code: str) -> LanguageSpec: ...
def normalize_language_code(code: str) -> str: ...
```

`normalize_language_code()` may accept existing short aliases (`vi`, `en`) at API boundaries but all snapshots, persistence files, provider calls, and artifact metadata use canonical BCP-47 codes.

Adding a language requires:

1. A `LanguageSpec` entry.
2. At least one compatible voice/provider entry.
3. Provider mapping tests and one real smoke-test record.
4. Translation prompt coverage if video dubbing targets that language.
5. Segmentation/chunking punctuation coverage for that script.

## 8. Provider Interfaces

Replace the broad `GeminiBackend` protocol with focused capability protocols while keeping `CompositeBackend` as the composition root.

```python
class SpeechRecognizer(Protocol):
    def transcribe_clip(self, wav_path: Path) -> tuple[str, str]: ...

class Translator(Protocol):
    def translate(
        self,
        texts: list[str],
        durations: list[float],
        *,
        target_language: str,
        context: str = "",
    ) -> list[str]: ...

class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str, voice_id: str, *, language: str) -> bytes: ...
```

Batch-capable synthesizers additionally expose:

```python
def synthesize_batch(
    self,
    texts: list[str],
    voice_id: str,
    *,
    language: str,
) -> list[bytes]: ...
```

Provider adapters resolve canonical language codes through `LanguageRegistry`:

- Gemini receives `gemini_tts_code` and a prompt that names the target language.
- OmniVoice receives `omnivoice_name` for every single and batch item.
- Edge validates that the selected voice advertises the same language before synthesis.

Fallback from batch synthesis to individual synthesis is allowed. Fallback to a different voice or a different provider is not allowed.

## 9. Voice Catalog and Preview Model

Extend the public voice representation:

```json
{
  "id": "clone-my-voice",
  "display_name": "Giọng của tôi",
  "provider": "omnivoice",
  "custom": true,
  "supported_languages": ["vi-VN", "en-US"],
  "preview_urls": {
    "vi-VN": "/api/voices/clone-my-voice/preview?language=vi-VN",
    "en-US": "/api/voices/clone-my-voice/preview?language=en-US"
  },
  "preview_status": {
    "vi-VN": "ready",
    "en-US": "pending"
  }
}
```

`preview_status` values are `pending`, `ready`, or `error`. A voice is selectable when synthesis is supported even if its preview is still pending.

Preview files use a language-specific, path-safe layout:

```text
previews/<voice_id>/<language_code>.wav
```

Creating or replacing a custom voice invalidates all preview files and cached reference text for that voice, then starts background generation for `vi-VN` and `en-US`. Preview generation uses the fixed sentence from `LanguageSpec.preview_text`.

Preset voices only advertise language entries they can synthesize correctly. A provider voice that is genuinely multilingual may advertise multiple canonical codes.

## 10. Target-Neutral Video Pipeline

Evolve `Segment` from Vietnamese-specific to target-neutral:

```python
@dataclass
class Segment:
    start: float
    end: float
    text: str
    target_text: str = ""
```

Keep a temporary `text_vi` property alias that reads/writes `target_text` during migration. New code and tests must use `target_text`.

Add `target_language: str = "vi-VN"` to `PipelineOptions`; the default preserves existing callers. Translation, synthesis, subtitle construction, filename suffixes, warning text, and result metadata use the target language.

Video execution remains:

```text
extract → transcribe → translate → synthesize → subtitle → assemble → mux
```

Source language is still detected by STT. Translation is skipped only when the detected source and canonical target language are equivalent. The skip must preserve 1:1 segments by copying `text` to `target_text`.

Video artifacts:

- `video`: dubbed MP4.
- `subtitle`: SRT in the target language.

## 11. Text → Voice Runner

Create a dedicated `pipeline/text_to_voice.py` runner.

```python
@dataclass(frozen=True)
class TextToVoiceOptions:
    voice_id: str
    language: str
    max_characters: int = 50_000
    inter_chunk_silence_ms: int = 200

@dataclass
class SpeechResult:
    wav_path: str
    mp3_path: str
    attempted_count: int
    spoken_count: int
    warnings: list[str]
```

Execution stages:

```text
prepare → synthesize → assemble → export
```

Behavior:

1. Normalize line endings and surrounding whitespace without rewriting user wording.
2. Reject empty input and input over 50,000 Unicode characters.
3. Split by language-specific sentence terminators, then by safe provider request size without reordering text.
4. Use batch synthesis where supported and individual fallback for a failed batch.
5. Join successful PCM chunks in original order with 200 ms silence between chunks.
6. Write lossless WAV at the project TTS sample rate.
7. Encode MP3 with the configured FFmpeg binary.
8. Return counts and warnings through the same degraded-result rules as video jobs.

If no chunk succeeds, the job is `error` and no artifact is published. If at least one chunk succeeds, publish WAV and MP3, set `degraded=true` when `spoken_count < attempted_count`, and surface the failed chunk count without including full private text in the snapshot.

## 12. Shared Job and Artifact Contracts

Extend `Job` instead of creating a second manager:

```python
job_type: Literal["video_dubbing", "text_to_voice"]
target_language: str
input_label: str
artifacts: list[JobArtifact]
```

```python
@dataclass(frozen=True)
class JobArtifact:
    id: str
    kind: Literal["video", "subtitle", "wav", "mp3"]
    filename: str
    media_type: str
    path: str
```

The snapshot adds these fields while retaining the current `filename`, `video_path`, and `srt_path` compatibility fields for video jobs:

```json
{
  "job_id": "abc123",
  "job_type": "text_to_voice",
  "target_language": "en-US",
  "input_label": "Welcome back. Today…",
  "voice_id": "en-US-AvaMultilingualNeural",
  "status": "running",
  "stage": "synthesize",
  "percent": 63.0,
  "message": "Đang tạo giọng 4/7",
  "artifacts": [],
  "attempted_count": 7,
  "spoken_count": 3,
  "warnings": [],
  "degraded": false
}
```

The job manager dispatches a runner based on `job_type`, but queue ordering, maximum concurrency, cancellation, persistence, pruning, telemetry, and SSE snapshots remain shared.

Stage weights become job-type-specific. Unknown restored stages must not crash percentage calculation; they produce the persisted percentage and a neutral UI state.

The full Text → Voice input is stored only as `input.txt` inside that job's work directory. `input_label` is a whitespace-normalized excerpt capped at 80 characters.

## 13. API Design

### Languages

```http
GET /api/languages
```

Returns canonical language code, display names, and whether video dubbing and Text → Voice are available.

### Voices

```http
GET /api/voices?language=vi-VN
GET /api/voices/{voice_id}/preview?language=en-US
```

Omitting `language` from `GET /api/voices` preserves the current all-voices behavior. Omitting it from the preview endpoint defaults to `vi-VN` for compatibility.

### Video jobs

The existing endpoint remains valid:

```http
POST /api/jobs
Content-Type: multipart/form-data

video=<file>
voice_id=<id>
target_language=vi-VN
```

`target_language` defaults to `vi-VN` when omitted.

### Text jobs

```http
POST /api/jobs/text
Content-Type: application/json

{
  "text": "Welcome back…",
  "voice_id": "en-US-AvaMultilingualNeural",
  "language": "en-US"
}
```

Returns `{"job_id": "..."}` after validation and persistence.

### On-demand text preview

```http
POST /api/voices/{voice_id}/preview-text
Content-Type: application/json

{
  "text": "Welcome back…",
  "language": "en-US"
}
```

The server normalizes and truncates to the first complete sentence or 300 characters, whichever comes first, synthesizes one temporary WAV response, and does not create a job or write job history. The browser cancels an obsolete request with `AbortController` when the user changes voice/language or requests another preview.

Production synthesis has priority. An on-demand preview attempts to acquire the selected provider's preview slot without blocking active production work; when unavailable it returns `409` with a Vietnamese retry message. Provider model locks and rate limits remain authoritative.

### Artifacts

```http
GET /api/jobs/{job_id}/artifacts/{artifact_id}
```

The route serves only an artifact declared by that job and whose resolved path is a direct child of the job work directory. Existing `/download/video` and `/download/srt` routes remain as compatibility aliases.

## 14. Frontend UX

### New Job panel

- Use a native accessible tab pattern: `role=tablist`, two `role=tab` buttons, `aria-selected`, `aria-controls`, keyboard Left/Right/Home/End, and matching `role=tabpanel` regions.
- Keep separate in-memory drafts when switching tabs.
- Keep one primary CTA per active tab.
- Video tab retains drag/drop, upload progress, cancel-upload, and preflight model readiness.
- Text tab contains a visible text label, character count, estimated duration, language select, filtered voice select, fixed-preview button, on-demand text-preview button, and submit CTA.
- Changing language clears an incompatible selected voice and explains why.
- Submitting a text job clears the text draft but retains language and voice for repeated work.
- On-demand preview shows loading, playing, retry, and stopped states without layout shift; only one preview audio element may play at once.

### Voice Lab

- Add one language selector above the voice collection.
- Filter the visible voice list to voices supporting that language.
- A preview button uses the selected language's pre-rendered URL and state.
- Pending preview shows a disabled control with “Đang tạo bản nghe thử”.
- Error preview offers a retry-generation action without disabling use of the voice itself.
- Voice labels wrap or truncate with an accessible full-name disclosure and never overflow their button.

### Shared queue and selected job

- Rename the section to **JOB QUEUE / HÀNG ĐỢI XỬ LÝ**.
- Show a text-and-icon type badge (`VIDEO` or `TEXT`), language label, voice, current stage, percent, and result artifacts.
- Text jobs use their excerpt as the card title; full input is never injected into queue HTML.
- Selected Job adapts its metadata and actions to the job type.
- The Execution Graph renders seven video stages or four text stages from a stage definition map. Completed export always lights when the job is done.
- Only the graph track may scroll horizontally on narrow screens; the rest of the page must fit at 360 px.

### Accessibility and responsive requirements

- Visible focus rings with at least 3:1 non-text contrast.
- Normal text contrast at least 4.5:1.
- Controls at least 44 CSS px high where practical and never below WCAG 2.2's 24×24 CSS px target requirement.
- No hover-only actions; icon controls have accessible names.
- Dynamic queue/preview updates use one polite atomic live region and do not move focus.
- Respect `prefers-reduced-motion`; correctness never depends on transition completion.
- Validate desktop at 1440×1000 and mobile at 360×800, plus keyboard-only operation.

## 15. Error Handling and Recovery

Validation occurs before creating a work directory or reserving a queue slot when possible:

- Unknown language: `400`.
- Unsupported voice/language pair: `400`.
- Empty Text → Voice input: `400`.
- Text over 50,000 characters: `413`.
- Missing Gemini key blocks video-job creation because the source language is not known until after STT and translation may be required. For Text → Voice, it blocks only Gemini TTS; configured local/Edge providers remain usable.
- Busy on-demand preview provider: `409` with retry guidance.
- Preview synthesis timeout: `504` with retry guidance.

Runtime behavior:

- Batch failure retries the same chunks individually with the same provider, voice, and language.
- Partial synthesis publishes artifacts only when at least one chunk succeeds and marks the job degraded.
- Cancellation checks run between chunks and before WAV/MP3 export. A cancelled job publishes no new artifact after cancellation is observed.
- FFmpeg MP3 failure leaves the job in `error`; an internal WAV temporary file is not advertised as a successful two-format result.
- API errors state both the cause and a recovery action in Vietnamese.

## 16. Persistence, Privacy, and Filesystem Safety

- Persist new fields with schema defaults so old video jobs restore as `video_dubbing` and `vi-VN`.
- Persist artifacts by relative identity, then resolve and validate paths at download time.
- Never accept a client-supplied artifact path or filename as a filesystem destination.
- Keep input text, generated audio, previews, model caches, and job metadata local.
- Do not log submitted full text, API keys, clone reference transcripts, or absolute artifact paths in client-facing messages.
- Deleting a job removes its `input.txt`, temporary chunks, WAV, MP3, and metadata using the manager's existing exclusive deletion claim.
- Pruning treats concurrently disappearing job directories as benign.

## 17. Testing Strategy

### Characterization and unit tests

- Lock current Vietnamese video behavior before changing shared interfaces.
- Test language normalization, unknown codes, and provider mapping.
- Test voice filtering and multilingual capability declarations.
- Assert exact `vi-VN`/`en-US` and `Vietnamese`/`English` values passed to Gemini and OmniVoice.
- Test Edge voice/language validation.
- Test target-neutral translation alignment and the temporary `text_vi` alias.
- Test multilingual sentence chunking, stable ordering, batch fallback, cancellation, partial failure, and 200 ms joins.
- Inspect WAV sample rate/channels/sample width and validate MP3 with FFprobe.

### API and job tests

- Create/list/status/cancel/delete/restore both job types.
- Validate defaults for old video clients.
- Validate unsupported language/voice combinations and text size limits.
- Verify generic artifact downloads and path traversal/symlink rejection.
- Verify full input text is absent from snapshots, SSE payloads, and telemetry.
- Verify preview pending/ready/error state and background regeneration after custom voice replacement.
- Verify production work takes priority over an on-demand preview.
- Verify mixed video/text queue ordering and concurrency reservations.

### Browser tests

All browser verification uses MCP Playwright:

- Desktop 1440×1000 and mobile 360×800.
- Accessible tab semantics and keyboard navigation.
- Draft preservation when switching tabs.
- Language-dependent voice filtering and incompatible-selection reset.
- Fixed language preview and on-demand text preview lifecycle.
- Live SSE progress for a text job and a video job without F5.
- Shared queue type/language labels and correct download actions.
- Adaptive Execution Graph completion for both stage sets.
- No Voice Lab overflow, no page-level horizontal overflow, and reduced-motion behavior.
- No application console errors.

### Real-provider smoke tests

- English Edge preset voice.
- Vietnamese and English OmniVoice using one authorized custom voice when the model is installed.
- Vietnamese and English video translation using Gemini.
- Smoke tests are opt-in and excluded from the deterministic default pytest suite.

## 18. Migration and Delivery Phases

### Phase 1: Characterize and introduce language capabilities

- Add regression tests around the current Vietnamese pipeline.
- Add `LanguageRegistry` and language-aware `Voice` metadata.
- Keep default target `vi-VN`; no visible behavior changes.

### Phase 2: Generalize providers and video target text

- Split backend protocols.
- Thread canonical target language through translation and TTS providers.
- Migrate `text_vi` to `target_text` with a compatibility alias.
- Prove the existing Vietnamese suite still passes.

### Phase 3: Add Text → Voice backend

- Extract reusable raw text synthesis from video timing logic.
- Add the dedicated runner, generic artifacts, shared-job dispatch, API, persistence, cancellation, and security tests.
- Export deterministic WAV and MP3 results.

### Phase 4: Add multilingual previews

- Generate language-specific fixed previews for custom voices.
- Add preview state to the voice API.
- Add bounded on-demand text preview with production-priority coordination.

### Phase 5: Integrate the frontend

- Add the approved two-tab composer.
- Rename and generalize the queue.
- Adapt Selected Job and Execution Graph by job type.
- Preserve the existing static HTML/CSS/JavaScript stack and no-build workflow.

### Phase 6: Enable English video dubbing and complete QA

- Expose target language in video creation.
- Verify English translation, TTS, subtitles, filenames, artifacts, and UI labels.
- Run full pytest, provider smoke tests, MCP Playwright matrices, and documentation checks.

Each phase must end in working, reviewable software. A phase may not rely on an untested future migration to restore Vietnamese behavior.

## 19. Acceptance Criteria

The feature is complete when:

1. A user can submit Vietnamese or English text, watch it update live in the shared queue, and download valid WAV and MP3 files.
2. A user can upload a video, choose Vietnamese or English output, and download a dubbed MP4 and matching SRT.
3. Voice choices always match the selected language and provider capabilities.
4. Voice Lab plays a pre-rendered custom-voice preview in the selected language.
5. Text → Voice can preview the first sentence or first 300 characters without creating a job.
6. Existing clients that submit only `video` and `voice_id` still receive Vietnamese output.
7. Existing/restored video jobs remain visible and downloadable.
8. Queue progress updates without page refresh for both job types.
9. The Execution Graph reaches a completed export state for both job types.
10. The full deterministic test suite passes, and MCP Playwright passes the desktop/mobile matrix without application console errors.

## 20. Implementation Constraints

- Use TDD for every behavior change: failing test, minimal implementation, passing test, review.
- Use an isolated `codex/` worktree because the current `main` working tree is dirty.
- Preserve user-owned uncommitted files and changes.
- Keep FastAPI plus plain HTML/CSS/JavaScript; do not introduce a frontend build step or framework.
- Use MCP Playwright for browser verification.
- Use subagent-driven implementation with independent tasks and review gates.
- Do not merge or push until the complete suite and final review pass.
