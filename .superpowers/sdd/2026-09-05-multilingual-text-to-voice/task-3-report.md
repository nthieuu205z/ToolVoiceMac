# Task 3 Report: Target-Neutral Video Translation and Output

## Status

Implemented the approved Task 3 brief on top of `aa74f55`. The video pipeline now carries a canonical target language through translation and speech synthesis, skips translation when source and target match, stores translated text in `Segment.target_text`, and preserves the `text_vi` read/write alias during migration. English runs use persistent `output_en.*` artifacts so compatibility downloads can expose `_en`, while the default Vietnamese path retains `output.mp4`, `output.srt`, and `_vi` download names.

## TDD Evidence

Every pytest invocation sanitized both proxy-bypass variables by splitting their comma-separated values and removing only tokens exactly equal to `::1` or `::1/128`, then passing all other tokens unchanged through `env NO_PROXY=... no_proxy=...`.

### RED 1 — migration and runner branch

Command:

```text
pytest tests/test_translate_alignment.py tests/test_runner.py -q
```

Result: exit 1 with the intended three failures:

```text
FAILED test_translation_writes_target_text_and_passes_target_language
TypeError: translate_segments() got an unexpected keyword argument 'target_language'
FAILED test_text_vi_alias_tracks_target_text_during_migration
TypeError: Segment.__init__() got an unexpected keyword argument 'target_text'
FAILED test_runner_skips_translation_when_source_matches_target
TypeError: PipelineOptions.__init__() got an unexpected keyword argument 'target_language'
```

### GREEN 1

The same command completed at 100% with exit 0:

```text
....................                                                     [100%]
```

### RED 2 — complete Task 3 video slice

Command:

```text
pytest tests/test_translate_alignment.py tests/test_runner.py tests/test_tts.py tests/test_tts_batch.py tests/test_subtitles.py tests/test_api.py tests/test_run_pipeline_cli.py -q
```

Result: exit 1 with 17 expected failures. They covered the old Gemini `text_vi` response schema, absent runner/TTS language forwarding, old `output.*` English names, absent TTS `language` argument, pre-migration `text_vi=` fixtures, ignored API target language/voice capability, `_vi` English downloads, and absent CLI propagation.

### GREEN 2 — focused Task 3 regression slice

The same command completed at 100% with exit 0:

```text
........................................................................ [ 65%]
......................................                                   [100%]
```

Only the pre-existing Starlette `httpx` deprecation warning was emitted.

### Full suite — run once

Command:

```text
pytest -q
```

Result: exit 0, 100%, no failures:

```text
........................................................................ [ 14%]
........................................................................ [ 29%]
........................................................................ [ 44%]
........................................................................ [ 59%]
........................................................................ [ 74%]
........................................................................ [ 89%]
.....................................................                    [100%]
```

Warnings were limited to the existing Starlette `httpx` deprecation and Python 3.13 `audioop` deprecation from `pydub`.

## Changed Files

- `pipeline/models.py` — `target_text`, compatibility `text_vi` property, neutral translator contract wording.
- `pipeline/gemini.py` — neutral response schema/prompt, language names, and per-language speaking-rate payloads.
- `pipeline/translate.py` — target-language forwarding and `target_text` writes.
- `pipeline/tts.py` — `target_text` reads and explicit language forwarding across single, retry, fallback, and batch paths.
- `pipeline/subtitles.py` — subtitle text reads from `target_text` with source fallback.
- `pipeline/runner.py` — target option, normalized source comparison, translation skip/copy, explicit synthesis language, and English artifact names.
- `backend/routes/jobs.py` — optional target form field, normalization, language-aware voice validation, and `_vi`/`_en` compatibility names.
- `scripts/run_pipeline_cli.py` — registry-backed `--target-language` choices/default and pipeline propagation.
- `tests/test_translate_alignment.py`, `tests/test_runner.py`, `tests/test_tts.py`, `tests/test_tts_batch.py`, `tests/test_api.py`, `tests/test_run_pipeline_cli.py` — Task 3 behavior coverage and migration.
- `tests/conftest.py`, `tests/test_gemini_fallback.py`, `tests/test_backend_batch_forwarding.py` — shared/internal test-contract migration required by `target_text` and explicit provider language arguments.

## Compatibility Checks

- `PipelineOptions.target_language` defaults to `vi-VN`; existing API clients may omit the new form field.
- Vietnamese runs retain internal `output.mp4`/`output.srt` and download as `<stem>_vi.mp4`/`<stem>_vi.srt`.
- English runs persist as `output_en.mp4`/`output_en.srt`, allowing restored jobs to retain `_en` download naming without adding generic job metadata in Task 3.
- `Segment.text_vi` reads and writes the same `target_text` value; only its dedicated compatibility test and property remain in the Python tree.
- Unsupported source languages still translate: normalization is used for supported aliases, while unregistered detected source codes remain unequal to the canonical target.
- Runner always supplies `language`, including explicit `vi-VN`, so Gemini's omitted-language sentinel is not accidentally selected.
- Existing video route and CLI defaults remain Vietnamese; existing provider clients and the complete prior suite pass unchanged.

## Self-Review

- Checked every brief step against the diff: schema/prompt, rate map, target threading, skip branch, alias, route, filenames, CLI, focused tests, and full suite are covered.
- Searched all Python files for `text_vi`/`text_vi=`; only the compatibility property and its setter test remain.
- Checked all production TTS call sites, including single-item batch fallback and quality re-synthesis, for explicit language forwarding.
- Ran `git diff --check` and Python byte-compilation for every changed production module; both exited 0.
- Reviewed the worktree status and diff; changes are limited to Task 3 production, tests, and this report.

## Concerns

No blocking concerns. The full suite still emits two unrelated upstream deprecation warnings. Compatibility suffix detection deliberately relies on Task 3's persistent `output_en.*` naming because generic job-language persistence is outside this task's scope.
