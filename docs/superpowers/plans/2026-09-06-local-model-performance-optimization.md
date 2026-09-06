# Local Model Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve local-model latency and unified-memory behavior on the target M1 Max by measuring the real production paths, optimizing OmniVoice MPS execution, retaining Whisper's supported CPU path, and exposing safe performance profiles and adaptive fallback.

**Architecture:** Add a small immutable performance-profile module, extend provider batching with a total-character budget, and isolate MPS failure classification/adaptive state in a dedicated controller. Keep model inference serialized, coordinate prewarming through one process-wide queue, and publish only non-sensitive effective runtime facts. Benchmark and quality gates decide whether `balanced` becomes the final default; unmeasured candidate values do not silently replace the current behavior.

**Tech Stack:** Python 3.12 arm64, FastAPI, Pydantic v2, PyTorch MPS, OmniVoice 0.2.1, faster-whisper/CTranslate2 CPU inference, NumPy, pytest, project-local benchmark scripts.

**Spec:** `docs/superpowers/specs/2026-09-06-local-model-performance-optimization-design.md`

## Global Constraints

- Target validation machine: Apple M1 Max, 10 CPU cores, 32 GPU cores, 64 GB unified memory.
- Use the exact project interpreter: `/Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/python` and its pytest.
- Whisper remains on CPU `int8` on Apple Silicon; do not claim or implement faster-whisper MPS support.
- OmniVoice remains CPU-staged PyTorch MPS `float16`; keep its audio tokenizer on CPU.
- Keep `_INFER_LOCK`; this plan does not authorize concurrent `model.generate()` calls on the same OmniVoice singleton.
- Preserve output ordering, partial-result semantics, cancellation, job persistence, privacy, SSE, and typed artifact publication.
- Keep Edge and Gemini behavior unchanged and avoid importing torch/OmniVoice when clone support is disabled.
- Benchmark metadata must exclude prompts, text, transcripts, audio/video paths, credentials, and exception messages.
- Real model benchmarks are opt-in and use only an authorized local clone voice ID.
- Report blocked real-model prerequisites as blocked, never passed.
- Do not stage or delete `web/static/previews/`; it contains user-generated local preview files.
- Every production behavior is implemented test-first. Each task is committed independently after focused tests pass.
- Adoption thresholds: ≥10% warm-median gain for profile/batching changes, ≥5% for cache/prewarm changes, p95 regression ≤10%, candidate balanced MPS driver memory <8 GB, and no correctness or quality regression.

---

### Task 1: Repeated Benchmark Series and Baseline Schema

**Files:**
- Modify: `benchmarks/performance_harness.py:39-223`
- Modify: `benchmarks/run_local_baseline.py:1-206`
- Modify: `benchmarks/README.md:1-22`
- Modify: `tests/test_performance_harness.py:1-86`
- Create: `tests/test_local_baseline_cli.py`

**Interfaces:**
- Produces: `SeriesSummary` with `cold_seconds`, `warm_seconds`, `median_seconds`, `p95_seconds`, `error_count`, `items_per_second`, `generated_audio_seconds_per_wall_second`, `peak_rss_mb`, `peak_mps_allocated_mb`, and `peak_mps_driver_mb`.
- Produces: `measure_series(label, operation_factory, *, warm_runs, item_count, generated_audio_seconds, metadata) -> SeriesSummary`.
- Produces: local benchmark CLI arguments `--voice-id`, `--warm-runs`, `--whisper-batches`, `--whisper-workers`, `--omnivoice-batches`, `--omnivoice-steps`, and `--output`.
- Consumes: existing `Measurement`, `measure()`, `_sync_mps()`, `WhisperTranscriber`, and `OmniVoiceSynthesizer`.

- [ ] **Step 1: Write failing statistics and sanitization tests**

```python
# append to tests/test_performance_harness.py
from benchmarks.performance_harness import measure_series


def test_measure_series_reports_cold_median_p95_errors_and_audio_rate(monkeypatch):
    durations = iter([9.0, 5.0, 1.0, 3.0, 2.0, 4.0])
    monkeypatch.setattr("benchmarks.performance_harness._timed_call", lambda operation: (next(durations), operation()))

    summary, results = measure_series(
        "omnivoice-balanced",
        lambda run_index: (lambda: run_index),
        warm_runs=5,
        item_count=2,
        generated_audio_seconds=10.0,
        metadata={"profile": "balanced", "text": "never persist"},
    )

    assert results == [0, 1, 2, 3, 4, 5]
    assert summary.cold_seconds == 9.0
    assert summary.warm_seconds == [5.0, 1.0, 3.0, 2.0, 4.0]
    assert summary.median_seconds == 3.0
    assert summary.p95_seconds == 5.0
    assert summary.generated_audio_seconds_per_wall_second == 10.0 / 3.0
    assert summary.metadata == {"profile": "balanced"}
```

```python
def test_measure_series_counts_failures_without_persisting_exception_messages():
    def factory(index):
        def operation():
            if index == 2:
                raise RuntimeError("private input must not enter report")
            return index
        return operation

    summary, _ = measure_series(
        "failure-case",
        factory,
        warm_runs=3,
        item_count=1,
        generated_audio_seconds=0.0,
    )

    assert summary.error_count == 1
    assert "private input" not in json.dumps(summary.to_dict())
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_performance_harness.py::test_measure_series_reports_cold_median_p95_errors_and_audio_rate \
  tests/test_performance_harness.py::test_measure_series_counts_failures_without_persisting_exception_messages -q
```

Expected: FAIL because `measure_series` and `SeriesSummary` do not exist.

- [ ] **Step 3: Implement deterministic series aggregation**

Add this public shape to `benchmarks/performance_harness.py`:

```python
@dataclass(frozen=True)
class SeriesSummary:
    label: str
    cold_seconds: float | None
    warm_seconds: list[float]
    median_seconds: float | None
    p95_seconds: float | None
    error_count: int
    item_count: int
    items_per_second: float
    generated_audio_seconds_per_wall_second: float
    peak_rss_mb: float
    peak_mps_allocated_mb: float | None
    peak_mps_driver_mb: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

Implement `_timed_call(operation)` with `time.perf_counter()`. Implement nearest-rank p95 as `sorted_values[max(0, ceil(0.95 * n) - 1)]`. Run exactly one cold call followed by `warm_runs` calls. Keep successful return values for quality checks; failures contribute only their exception type internally and increment `error_count`.

Extend `write_json_report()` to accept both `Measurement` and `SeriesSummary`, write `schema_version: 2`, and continue reading/writing schema-1-compatible measurement fields.

- [ ] **Step 4: Add failing CLI matrix tests**

```python
# tests/test_local_baseline_cli.py
import subprocess
import sys


def test_local_baseline_help_exposes_repeatable_matrix_options():
    result = subprocess.run(
        [sys.executable, "benchmarks/run_local_baseline.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for option in (
        "--voice-id", "--warm-runs", "--whisper-batches",
        "--whisper-workers", "--omnivoice-batches", "--omnivoice-steps",
    ):
        assert option in result.stdout


def test_local_baseline_requires_an_explicit_authorized_voice_for_omnivoice():
    result = subprocess.run(
        [sys.executable, "benchmarks/run_local_baseline.py", "--source-wav", "missing.wav"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--voice-id" in result.stderr
```

- [ ] **Step 5: Run CLI tests and verify RED**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest tests/test_local_baseline_cli.py -q
```

Expected: FAIL because the new CLI options are absent.

- [ ] **Step 6: Extend the benchmark CLI without changing production settings**

Implement comma-separated positive integer parsing. Defaults:

```python
warm_runs = 5
whisper_batches = (4, 8, 12, 16)
whisper_workers = (1, 2, 4)
omnivoice_batches = (1, 2, 4, 8)
omnivoice_steps = (16, 24, 32)
```

The benchmark creates fresh adapter instances but shares production model singletons. It synchronizes MPS before and after every timed OmniVoice call. It records short, medium, long, and mixed-length fixture labels, never fixture content. It calculates generated audio seconds from returned PCM byte lengths using `TTS_SAMPLE_RATE` and 16-bit mono encoding.

The CLI must validate `--source-wav`, `--voice-id`, and `--warm-runs >= 5` before loading a model. Default output remains under `.hermes/perf/`.

- [ ] **Step 7: Run focused tests**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_performance_harness.py tests/test_local_baseline_cli.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/performance_harness.py benchmarks/run_local_baseline.py benchmarks/README.md \
  tests/test_performance_harness.py tests/test_local_baseline_cli.py
git commit -m "perf: add repeated local model benchmarks"
```

---

### Task 2: Immutable Local Performance Profiles

**Files:**
- Create: `pipeline/performance_profiles.py`
- Create: `tests/test_performance_profiles.py`
- Modify: `backend/config.py:12-180`
- Modify: `pipeline/backends.py:19-34,184-221`
- Modify: `backend/routes/text_jobs.py:40-55`
- Modify: `scripts/run_pipeline_cli.py:86-105`
- Modify: `.env.example:69-119`
- Modify: `README.md:214-293`
- Modify: `tests/test_backends.py:1-87`
- Modify: `tests/test_run_pipeline_cli.py:1-130`

**Interfaces:**
- Produces: immutable `LocalPerformanceProfile`.
- Produces: `resolve_local_performance_profile(name, *, num_step_override=0, batch_size_override=0, character_limit_override=0) -> LocalPerformanceProfile`.
- Produces: settings fields `local_performance_profile`, `omnivoice_batch_character_limit`, and `prewarm_local_models`.
- Adds to `ProviderConfig`: `performance_profile`, `omnivoice_batch_character_limit`, and `prewarm_local_models`.
- Consumes: existing advanced overrides `OMNIVOICE_NUM_STEP` and `OMNIVOICE_BATCH_SIZE`; use `0` as “profile value” rather than a second default.

- [ ] **Step 1: Write failing profile tests**

```python
# tests/test_performance_profiles.py
import pytest

from pipeline.performance_profiles import resolve_local_performance_profile


def test_balanced_profile_is_the_candidate_default_shape():
    profile = resolve_local_performance_profile("balanced")
    assert profile.name == "balanced"
    assert profile.omnivoice_num_step == 24
    assert profile.omnivoice_batch_size == 4
    assert profile.omnivoice_batch_character_limit == 900
    assert profile.whisper_cpu_batch_size == 8
    assert profile.whisper_workers == 1


def test_profiles_resolve_locked_quality_and_turbo_values():
    quality = resolve_local_performance_profile("quality")
    turbo = resolve_local_performance_profile("turbo")
    assert (quality.omnivoice_num_step, quality.omnivoice_batch_size, quality.omnivoice_batch_character_limit) == (32, 2, 600)
    assert (turbo.omnivoice_num_step, turbo.omnivoice_batch_size, turbo.omnivoice_batch_character_limit) == (16, 4, 1200)


def test_explicit_overrides_win_and_are_clamped():
    profile = resolve_local_performance_profile(
        "balanced",
        num_step_override=20,
        batch_size_override=3,
        character_limit_override=700,
    )
    assert (profile.omnivoice_num_step, profile.omnivoice_batch_size) == (20, 3)
    assert profile.omnivoice_batch_character_limit == 700


@pytest.mark.parametrize("name", ["", "fastest", "mps"])
def test_unknown_profile_is_rejected(name):
    with pytest.raises(ValueError, match="LOCAL_PERFORMANCE_PROFILE"):
        resolve_local_performance_profile(name)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest tests/test_performance_profiles.py -q
```

Expected: FAIL because `pipeline.performance_profiles` does not exist.

- [ ] **Step 3: Implement the profile deep module**

```python
# pipeline/performance_profiles.py
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LocalPerformanceProfile:
    name: str
    omnivoice_num_step: int
    omnivoice_batch_size: int
    omnivoice_batch_character_limit: int
    whisper_cpu_batch_size: int = 8
    whisper_workers: int = 1
    prewarm_local_models: bool = True
```

Keep `_PROFILES` private. Validate profile names by exact normalized lowercase value. Clamp advanced overrides to:

- steps: `8..64`;
- item batch: `1..8`;
- character limit: `128..4000`.

A zero override means use the profile value. Do not scatter profile-name conditionals outside this module.

- [ ] **Step 4: Add failing settings/provider forwarding tests**

```python
# append to tests/test_performance_profiles.py
from backend.config import Settings


def test_settings_resolve_profile_into_provider_config():
    settings = Settings(
        _env_file=None,
        local_performance_profile="turbo",
        omnivoice_num_step=0,
        omnivoice_batch_size=0,
        omnivoice_batch_character_limit=0,
    )
    config = settings.provider_config_for("omnivoice")
    assert config.performance_profile == "turbo"
    assert config.omnivoice_num_step == 16
    assert config.omnivoice_batch_size == 4
    assert config.omnivoice_batch_character_limit == 1200


def test_invalid_profile_fails_provider_validation():
    settings = Settings(_env_file=None, local_performance_profile="fastest")
    with pytest.raises(ValueError, match="LOCAL_PERFORMANCE_PROFILE"):
        settings.validate_providers()
```

- [ ] **Step 5: Run forwarding tests and verify RED**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_performance_profiles.py::test_settings_resolve_profile_into_provider_config \
  tests/test_performance_profiles.py::test_invalid_profile_fails_provider_validation -q
```

Expected: FAIL because settings/config do not expose profiles.

- [ ] **Step 6: Wire profiles through configuration and every OmniVoice construction path**

Add settings:

```python
local_performance_profile: str = "quality"
omnivoice_num_step: int = 0
omnivoice_batch_size: int = 0
omnivoice_batch_character_limit: int = 0
prewarm_local_models: bool = True
```

`provider_config_for()` resolves exactly once and copies effective values into `ProviderConfig`. Update `build_backend()`, text jobs, previews, generation script, and CLI paths to consume effective config fields rather than rereading raw settings.

Update `.env.example` with `LOCAL_PERFORMANCE_PROFILE=quality` during the measurement phase, advanced override semantics, and the fact that Whisper stays CPU on Apple Silicon. Update README with the three profiles and the rule that Task 8 promotes balanced only if its benchmark and quality gates pass.

- [ ] **Step 7: Run focused configuration tests**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_performance_profiles.py tests/test_backends.py tests/test_run_pipeline_cli.py \
  tests/test_text_job_api.py tests/test_generate_voice_previews.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add pipeline/performance_profiles.py backend/config.py pipeline/backends.py \
  backend/routes/text_jobs.py scripts/run_pipeline_cli.py .env.example README.md \
  tests/test_performance_profiles.py tests/test_backends.py tests/test_run_pipeline_cli.py
git commit -m "perf: add local inference profiles"
```

---

### Task 3: Length-Aware Speech Batch Planning

**Files:**
- Modify: `pipeline/speech_synthesis.py:18-204`
- Modify: `pipeline/models.py` batch-capable protocol section
- Modify: `pipeline/backends.py:131-165`
- Modify: `pipeline/omnivoice_speech.py:244-291`
- Modify: `tests/test_speech_synthesis.py`
- Modify: `tests/test_tts_batch.py:1-196`
- Modify: `tests/test_backend_batch_forwarding.py`

**Interfaces:**
- Changes: `_batch_groups(texts, item_limit, character_limit) -> list[list[int]]`.
- Adds optional provider property: `batch_character_limit: int`; `0` means unlimited.
- Keeps: output arrays indexed to original input order.
- Consumes: effective item/character limits from the resolved OmniVoice profile.

- [ ] **Step 1: Write failing grouping tests**

```python
# append to tests/test_speech_synthesis.py
from pipeline.speech_synthesis import _batch_groups


def test_batch_groups_respect_item_and_character_limits():
    texts = ["a" * 500, "b" * 450, "c" * 300, "d" * 100]
    groups = _batch_groups(texts, item_limit=3, character_limit=700)
    assert sorted(index for group in groups for index in group) == [0, 1, 2, 3]
    assert all(len(group) <= 3 for group in groups)
    assert all(sum(len(texts[index]) for index in group) <= 700 for group in groups)


def test_one_oversized_input_is_kept_whole_in_a_single_item_batch():
    texts = ["x" * 1200, "short"]
    groups = _batch_groups(texts, item_limit=4, character_limit=900)
    assert [0] in groups
    assert sorted(index for group in groups for index in group) == [0, 1]


def test_mixed_lengths_are_grouped_deterministically_and_outputs_keep_input_order():
    texts = ["a" * 20, "b" * 600, "c" * 30, "d" * 590]
    assert _batch_groups(texts, 2, 900) == _batch_groups(texts, 2, 900)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_speech_synthesis.py -q
```

Expected: FAIL because `_batch_groups` accepts only one limit.

- [ ] **Step 3: Implement two-dimensional batch planning**

Sort indexes by normalized length descending with original index as the tie breaker. Build groups greedily: add the next index only if both item count and total characters remain within limits; otherwise start a new group. An oversized item forms a one-item group. Return groups ordered by the first original index they contain only if doing so does not mix lengths; result assignment remains by original index regardless of execution order.

Update `_provider_batching()` to return `(item_limit, character_limit, force_single_batch)`. Providers without `batch_character_limit` return `0` and keep current behavior.

- [ ] **Step 4: Add failing forwarding and ordered-output tests**

```python
# append to tests/test_backend_batch_forwarding.py
def test_composite_forwards_batch_character_limit():
    class Synth:
        batch_size = 4
        batch_character_limit = 900
    assert _backend(Synth()).batch_character_limit == 900
```

```python
# append to tests/test_tts_batch.py
def test_character_limited_batches_publish_results_in_original_order():
    class CharacterLimitedBackend(LoBackend):
        batch_size = 4
        batch_character_limit = 5

    backend = CharacterLimitedBackend()
    segs = segments((0, 1, "aaaa"), (2, 3, "b"), (4, 5, "cccc"))
    fitted, warnings = synthesize_segments(backend, segs, "voice", total_duration=10)

    assert [segment.target_text for segment, _ in fitted] == ["aaaa", "b", "cccc"]
    assert warnings == []
    assert all(sum(len(text) for text in call) <= 5 for call in backend.batch_calls)
```

- [ ] **Step 5: Implement provider forwarding**

Add `CompositeBackend.batch_character_limit` with the same safe integer conversion pattern as `batch_size`. Add `OmniVoiceSynthesizer.batch_character_limit`. Do not add the property to Edge or Gemini adapters.

- [ ] **Step 6: Run focused batching tests**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_speech_synthesis.py tests/test_tts_batch.py tests/test_backend_batch_forwarding.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/speech_synthesis.py pipeline/models.py pipeline/backends.py \
  pipeline/omnivoice_speech.py tests/test_speech_synthesis.py tests/test_tts_batch.py \
  tests/test_backend_batch_forwarding.py
git commit -m "perf: bound speech batches by input length"
```

---

### Task 4: Adaptive MPS Batch Controller and Failure Classification

**Files:**
- Create: `pipeline/accelerator_policy.py`
- Create: `tests/test_accelerator_policy.py`
- Modify: `pipeline/errors.py:1-75`

**Interfaces:**
- Produces: `AcceleratorFailureKind` enum values `out_of_memory`, `runtime`, `unsupported`.
- Produces: `classify_accelerator_failure(exc, *, device) -> AcceleratorFailureKind | None`.
- Produces: thread-safe `AdaptiveBatchController`.
- Produces: internal `AcceleratorRetryRequired(retry_limit: int, fallback_reason: str)`; its message contains no submitted text.

- [ ] **Step 1: Write failing classification tests**

```python
# tests/test_accelerator_policy.py
from pipeline.accelerator_policy import AcceleratorFailureKind, classify_accelerator_failure


def test_mps_failure_classifier_accepts_only_accelerator_signatures():
    assert classify_accelerator_failure(RuntimeError("MPS backend out of memory"), device="mps") == AcceleratorFailureKind.OUT_OF_MEMORY
    assert classify_accelerator_failure(RuntimeError("Metal command buffer failed"), device="mps") == AcceleratorFailureKind.RUNTIME
    assert classify_accelerator_failure(NotImplementedError("operator not supported on MPS"), device="mps") == AcceleratorFailureKind.UNSUPPORTED
    assert classify_accelerator_failure(ValueError("bad input"), device="mps") is None
    assert classify_accelerator_failure(RuntimeError("MPS failure"), device="cpu") is None
```

- [ ] **Step 2: Write failing controller tests**

```python
from pipeline.accelerator_policy import AdaptiveBatchController


def test_controller_reduces_along_ladder_and_requires_cooldown_plus_successes():
    now = [100.0]
    controller = AdaptiveBatchController((4, 2, 1), cooldown_seconds=120, required_successes=3, clock=lambda: now[0])

    assert controller.effective_limit == 4
    controller.record_accelerator_failure("out_of_memory")
    assert controller.effective_limit == 2
    for _ in range(3):
        controller.record_success()
    assert controller.probe_limit() is None
    now[0] = 220.0
    assert controller.probe_limit() == 4
    controller.record_probe_success(4)
    assert controller.effective_limit == 4


def test_non_accelerator_failure_does_not_change_limit():
    controller = AdaptiveBatchController((4, 2, 1))
    controller.record_non_accelerator_failure()
    assert controller.effective_limit == 4
    assert controller.fallback_reason == ""
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest tests/test_accelerator_policy.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 4: Implement the isolated policy module**

Use `threading.Lock` and `time.monotonic`. Never inspect or persist full exception text outside the classifier. Classification may inspect lowercase exception text only inside the function, then return the enum. The public fallback reason is one of `mps_out_of_memory`, `mps_runtime_failure`, or `mps_unsupported_operation`.

The controller:

- normalizes a strictly descending unique ladder ending in `1`;
- reduces one step per classified failure;
- resets success count on failure;
- permits one outstanding probe only after cooldown and required successes;
- rejects stale probe completions;
- exposes an immutable snapshot dict with requested/effective/probe/fallback fields.

- [ ] **Step 5: Run tests including thread safety**

Add a test that calls `record_accelerator_failure` concurrently from eight threads and asserts the effective limit remains a valid ladder value and never drops below 1.

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest tests/test_accelerator_policy.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/accelerator_policy.py pipeline/errors.py tests/test_accelerator_policy.py
git commit -m "perf: add recoverable MPS batch policy"
```

---

### Task 5: Integrate Adaptive Retry Into OmniVoice Batch Execution

**Files:**
- Modify: `pipeline/omnivoice_speech.py:23-357`
- Modify: `pipeline/speech_synthesis.py:92-204`
- Modify: `pipeline/backends.py:131-165`
- Modify: `tests/test_omnivoice_speech.py:150-330`
- Modify: `tests/test_speech_synthesis.py`
- Modify: `tests/test_tts_batch.py:111-176`

**Interfaces:**
- `OmniVoiceSynthesizer` owns one shared process controller for production MPS calls.
- `mps_batch_safe` remains as a compatibility property and returns `effective_limit > 1`.
- Produces: `adaptive_batch_snapshot() -> dict[str, object]` for telemetry/tests.
- `AcceleratorRetryRequired` tells `synthesize_texts()` to repartition only the failed group.

- [ ] **Step 1: Replace permanent downgrade expectations with failing adaptive tests**

```python
# append to tests/test_omnivoice_speech.py
def test_mps_oom_reduces_only_one_ladder_step(monkeypatch):
    import pipeline.omnivoice_speech as ov

    controller = ov._new_batch_controller_for_tests(requested_limit=4)
    monkeypatch.setattr(ov, "_MPS_BATCH_CONTROLLER", controller)
    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")

    model = FakeOmni(error=RuntimeError("MPS backend out of memory"))
    synth = OmniVoiceSynthesizer(model=model, transcriber=FakeTranscriber(text="mẫu"), batch_size=4)

    with pytest.raises(AcceleratorRetryRequired) as raised:
        synth.synthesize_batch(["a", "b", "c", "d"], _make_clone())

    assert raised.value.retry_limit == 2
    assert synth.batch_size == 2
    assert synth.adaptive_batch_snapshot()["fallback_reason"] == "mps_out_of_memory"


def test_content_error_does_not_reduce_mps_batch(monkeypatch):
    import pipeline.omnivoice_speech as ov

    controller = ov._new_batch_controller_for_tests(requested_limit=4)
    monkeypatch.setattr(ov, "_MPS_BATCH_CONTROLLER", controller)
    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")

    voice_id = _make_clone("clone-content-error")
    synth = OmniVoiceSynthesizer(
        model=FakeOmni(error=ValueError("bad text")),
        transcriber=FakeTranscriber(text="mẫu"),
        batch_size=4,
    )

    with pytest.raises(SpeechServiceError):
        synth.synthesize_batch(["a", "b"], voice_id)

    assert synth.batch_size == 4
    assert synth.adaptive_batch_snapshot()["fallback_reason"] == ""
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_omnivoice_speech.py::test_mps_oom_reduces_only_one_ladder_step \
  tests/test_omnivoice_speech.py::test_content_error_does_not_reduce_mps_batch -q
```

Expected: FAIL because current code permanently switches every future batch to 1 after any multi-item exception.

- [ ] **Step 3: Integrate the controller at the raw exception boundary**

In `_generate()` catch raw exceptions before wrapping them. If device is MPS and classification succeeds:

1. update the controller;
2. if the failed call had more than one item, raise `AcceleratorRetryRequired` with the new limit;
3. if the failed call had one item, raise the existing safe `SpeechServiceError`.

For all non-classified failures, preserve current `SpeechServiceError` behavior and do not change controller state.

Resolve the ladder from the requested profile batch size:

```python
def batch_ladder(requested: int) -> tuple[int, ...]:
    if requested >= 4:
        return (requested, 2, 1)
    if requested == 3:
        return (3, 2, 1)
    if requested == 2:
        return (2, 1)
    return (1,)
```

- [ ] **Step 4: Write failing retry/repartition/cancellation tests**

```python
# append to tests/test_speech_synthesis.py
def test_accelerator_retry_repartitions_only_the_failed_group_and_keeps_order():
    backend = AdaptiveFakeBackend(initial_limit=4, retry_limit=2)
    outputs, warnings = synthesize_texts(
        backend,
        ["a", "bb", "ccc", "dddd"],
        "voice",
        language="en-US",
    )
    assert outputs == [b"a", b"bb", b"ccc", b"dddd"]
    assert backend.calls == [["dddd", "ccc", "bb", "a"], ["dddd", "ccc"], ["bb", "a"]]
    assert warnings == []


def test_cancellation_wins_before_an_accelerator_retry_batch():
    backend = AdaptiveFakeBackend(initial_limit=4, retry_limit=2)
    cancel = iter([False, True])
    with pytest.raises(JobCancelledError):
        synthesize_texts(
            backend,
            ["a", "b", "c", "d"],
            "voice",
            language="en-US",
            should_cancel=lambda: next(cancel, True),
        )
    assert len(backend.calls) == 1
```

Define `AdaptiveFakeBackend.synthesize_batch()` in the test file: first call raises `AcceleratorRetryRequired(2, "mps_out_of_memory")`; subsequent calls return each input encoded as bytes.

- [ ] **Step 5: Convert the static batch loop to a bounded work queue**

Use `collections.deque` of index groups. On `AcceleratorRetryRequired`, check cancellation, split that group using the new item limit and current character limit, and prepend the smaller groups in execution order. Track attempted `(indexes, retry_limit)` pairs and raise a safe `SpeechServiceError` if the same partition would repeat, preventing infinite retries.

Do not count the failed accelerator attempt as failed output. Progress advances only after a group produces terminal values.

- [ ] **Step 6: Run focused retry and legacy fallback tests**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_accelerator_policy.py tests/test_omnivoice_speech.py \
  tests/test_speech_synthesis.py tests/test_tts_batch.py tests/test_backend_batch_forwarding.py -q
```

Expected: PASS, including legacy non-accelerator batch-to-single fallback tests.

- [ ] **Step 7: Commit**

```bash
git add pipeline/omnivoice_speech.py pipeline/speech_synthesis.py pipeline/backends.py \
  tests/test_omnivoice_speech.py tests/test_speech_synthesis.py tests/test_tts_batch.py
git commit -m "perf: recover OmniVoice MPS batch failures"
```

---

### Task 6: Coordinated Local Model Prewarming

**Files:**
- Create: `pipeline/model_prewarm.py`
- Create: `tests/test_model_prewarm.py`
- Modify: `pipeline/whisper_stt.py:138-162`
- Modify: `pipeline/omnivoice_speech.py:162-181`
- Modify: `backend/main.py:21-38`
- Modify: `backend/config.py`
- Modify: `tests/test_queue_settings_shutdown.py`

**Interfaces:**
- Produces: `ModelPrewarmCoordinator.enqueue(key, loader)`, `start()`, `snapshot()`, and `wait_for_idle(timeout)`.
- Changes `whisper_stt.prewarm()` and `omnivoice_speech.prewarm()` to synchronous, idempotent load-attempt functions; thread ownership moves to the coordinator.
- Consumes: `settings.prewarm_local_models` and existing offline `is_ready()` checks.

- [ ] **Step 1: Write failing coordinator-order tests**

```python
# tests/test_model_prewarm.py
from pipeline.model_prewarm import ModelPrewarmCoordinator


def test_prewarm_runs_one_loader_at_a_time_in_queue_order():
    events = []
    coordinator = ModelPrewarmCoordinator()
    coordinator.enqueue("whisper", lambda: events.extend(["whisper-start", "whisper-end"]))
    coordinator.enqueue("omnivoice", lambda: events.extend(["omnivoice-start", "omnivoice-end"]))
    coordinator.start()
    assert coordinator.wait_for_idle(2.0)
    assert events == ["whisper-start", "whisper-end", "omnivoice-start", "omnivoice-end"]


def test_duplicate_model_key_is_loaded_once():
    calls = []
    coordinator = ModelPrewarmCoordinator()
    coordinator.enqueue("whisper", lambda: calls.append("load"))
    coordinator.enqueue("whisper", lambda: calls.append("duplicate"))
    coordinator.start()
    assert coordinator.wait_for_idle(2.0)
    assert calls == ["load"]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest tests/test_model_prewarm.py -q
```

Expected: FAIL because the coordinator does not exist.

- [ ] **Step 3: Implement one-worker coordinator**

Use a `deque`, `Condition`, one daemon thread, and sets for queued/completed keys. `snapshot()` returns only keys/status/timestamps/error type. It must not retain exception messages. A loader exception marks that key `error`, logs once, and continues to the next loader.

- [ ] **Step 4: Write failing lifespan/prewarm tests**

```python
# append to tests/test_model_prewarm.py
def test_lifespan_queues_whisper_before_omnivoice(monkeypatch):
    queued = []
    monkeypatch.setattr("backend.main.settings.prewarm_local_models", True)
    monkeypatch.setattr("backend.main.settings.stt_provider", "whisper")
    monkeypatch.setattr("backend.main.settings.resolved_clone_provider", "omnivoice")
    monkeypatch.setattr("backend.main.prewarm_coordinator.enqueue", lambda key, loader: queued.append(key))
    # Enter the FastAPI lifespan with TestClient.
    with TestClient(app):
        pass
    assert queued[:2] == ["whisper", "omnivoice"]
```

Use monkeypatch seams for readiness/loaders so the test never imports or loads a real model.

- [ ] **Step 5: Move thread ownership out of model adapters**

Make both adapter `prewarm()` functions perform their existing offline readiness check and one synchronous singleton load attempt. In `backend/main.py`, enqueue Whisper first and OmniVoice second, then call `prewarm_coordinator.start()`. When `PREWARM_LOCAL_MODELS=false`, enqueue nothing.

Do not wait for the queue before yielding from lifespan; server readiness remains independent of prewarm completion.

- [ ] **Step 6: Run focused startup tests**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_model_prewarm.py tests/test_queue_settings_shutdown.py \
  tests/test_omnivoice_speech.py tests/test_whisper_cpu_batching.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/model_prewarm.py pipeline/whisper_stt.py pipeline/omnivoice_speech.py \
  backend/main.py backend/config.py tests/test_model_prewarm.py tests/test_queue_settings_shutdown.py
git commit -m "perf: coordinate local model prewarming"
```

---

### Task 7: Rich Non-Sensitive Runtime Telemetry

**Files:**
- Create: `pipeline/runtime_info.py`
- Create: `tests/test_runtime_info.py`
- Modify: `pipeline/omnivoice_speech.py`
- Modify: `pipeline/backends.py:155-165`
- Modify: `backend/job_manager.py:153-410`
- Modify: `backend/routes/model.py:28-40`
- Modify: `tests/test_job_telemetry.py:1-213`
- Modify: `tests/test_model_routes.py`

**Interfaces:**
- Produces immutable `RuntimeInfo` with engine/device/profile/requested/effective/character-limit/fallback/memory fields.
- `CompositeBackend.runtime_info() -> RuntimeInfo`.
- `Job.update_runtime_from_backend()` accepts both legacy 3-tuples and `RuntimeInfo`.
- Adds snapshot fields: `performance_profile`, `requested_batch_size`, `batch_character_limit`, `fallback_reason`, `mps_allocated_mb`, `mps_driver_mb`.

- [ ] **Step 1: Write failing runtime-info tests**

```python
# tests/test_runtime_info.py
from pipeline.runtime_info import RuntimeInfo


def test_runtime_info_public_dict_contains_only_safe_scalar_fields():
    info = RuntimeInfo(
        engine="omnivoice",
        device="mps:0",
        performance_profile="balanced",
        requested_batch_size=4,
        effective_batch_size=2,
        batch_character_limit=900,
        fallback_reason="mps_out_of_memory",
        mps_allocated_mb=1200.5,
        mps_driver_mb=2400.0,
    )
    assert info.public()["effective_batch_size"] == 2
    serialized = json.dumps(info.public())
    assert "text" not in serialized
    assert "path" not in serialized
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest tests/test_runtime_info.py -q
```

Expected: FAIL because `RuntimeInfo` does not exist.

- [ ] **Step 3: Implement runtime-info sampling**

`RuntimeInfo.public()` returns JSON-safe numbers/strings only. Add `mps_memory_snapshot()` that uses `torch.mps.current_allocated_memory()` and `driver_allocated_memory()` when available, returning `None` otherwise. It catches API/version exceptions and never initializes a model.

`OmniVoiceSynthesizer.runtime_info()` combines profile fields, controller snapshot, model device, and current MPS memory. `CompositeBackend.runtime_info()` forwards a provider-supplied `RuntimeInfo`; otherwise it builds one from legacy engine/device/batch properties.

- [ ] **Step 4: Write failing job compatibility tests**

```python
# append to tests/test_job_telemetry.py
from pipeline.runtime_info import RuntimeInfo


def test_job_manager_captures_rich_runtime_info(tmp_path):
    class Backend:
        def runtime_info(self):
            return RuntimeInfo(
                engine="omnivoice", device="mps:0", performance_profile="balanced",
                requested_batch_size=4, effective_batch_size=2,
                batch_character_limit=900, fallback_reason="mps_out_of_memory",
                mps_allocated_mb=1100.0, mps_driver_mb=2200.0,
            )
    manager = JobManager(max_workers=1)
    job = manager.start(
        filename="clip.mp4",
        input_label="clip.mp4",
        job_type="video_dubbing",
        target_language="vi-VN",
        workdir=tmp_path,
        voice_id="clone-demo",
        backend_factory=Backend,
        runner=lambda backend, progress, should_cancel: JobRunResult([]),
    )
    manager._futures[job.id].result(timeout=10)

    snapshot = job.snapshot()
    assert snapshot["batch_size"] == 2
    assert snapshot["performance_profile"] == "balanced"
    assert snapshot["fallback_reason"] == "mps_out_of_memory"
```

Keep the existing tuple test unchanged to prove backward compatibility.

- [ ] **Step 5: Extend Job fields and additive API output**

Add default-safe fields to `Job`. Persist and restore them using the same defaulting rules as current engine/device/batch values. The model status route adds a top-level `runtime` object only when OmniVoice is loaded; absence remains valid.

Do not expose the selected voice ID, prompt cache content, reference transcript, or model/cache paths through model status.

- [ ] **Step 6: Run focused telemetry tests**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_runtime_info.py tests/test_job_telemetry.py tests/test_job_persistence.py \
  tests/test_model_routes.py tests/test_backends.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pipeline/runtime_info.py pipeline/omnivoice_speech.py pipeline/backends.py \
  backend/job_manager.py backend/routes/model.py tests/test_runtime_info.py \
  tests/test_job_telemetry.py tests/test_model_routes.py
git commit -m "perf: expose effective local model runtime"
```

---

### Task 8: Run the M1 Max Matrix and Select Production Defaults

**Files:**
- Create: `.hermes/perf/local-model-optimization-m1max.json` (local evidence; do not commit)
- Create: `.hermes/perf/audio/` A/B files (local evidence; do not commit)
- Create: `docs/performance/local-model-m1max.md`
- Modify only if gates support it: `pipeline/performance_profiles.py`
- Modify only if gates support it: `backend/config.py`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `tests/test_performance_profiles.py`

**Interfaces:**
- Consumes: Tasks 1–7 benchmark/profile/controller/telemetry interfaces.
- Produces: a committed, content-safe decision report with medians, p95, memory peaks, error counts, quality result, accepted defaults, and rejected experiments.

- [ ] **Step 1: Verify the authorized benchmark prerequisites without printing private data**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/python - <<'PY'
import os
from backend.config import settings
from pipeline import custom_voices
custom_voices.configure(settings.custom_voices_dir)
voice_id = os.getenv("PROVIDER_SMOKE_CLONE_VOICE_ID", "")
print(f"authorized_voice_configured={bool(voice_id)}")
print(f"authorized_voice_available={bool(voice_id and custom_voices.is_custom(voice_id))}")
PY
```

Do not print the voice ID. If either value is false, record OmniVoice matrix/quality as `BLOCKED` and do not claim optimized defaults.

- [ ] **Step 2: Run the repeatable baseline matrix**

With an authorized voice configured, run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/python \
  benchmarks/run_local_baseline.py \
  --source-wav .hermes/perf/fixtures/authorized-source.wav \
  --voice-id "$PROVIDER_SMOKE_CLONE_VOICE_ID" \
  --warm-runs 5 \
  --whisper-batches 4,8,12,16 \
  --whisper-workers 1,2,4 \
  --omnivoice-batches 1,2,4,8 \
  --omnivoice-steps 16,24,32 \
  --output .hermes/perf/local-model-optimization-m1max.json
```

The fixture must be a project-authorized, non-private WAV. If no such fixture exists, stop and ask for one rather than substituting user job content.

- [ ] **Step 3: Evaluate candidates programmatically**

Add an analysis subcommand to `run_local_baseline.py` or a pure helper in the benchmark module that prints a table from the JSON report without content fields. Select settings by these exact rules:

1. Reject any case with errors or missing output.
2. Reject default candidates with MPS driver memory ≥8192 MB.
3. Compare balanced against quality on mixed-length warm median and p95.
4. Require balanced median gain ≥10% and p95 regression ≤10%.
5. Compare batch 4 against batch 2; if throughput gain <5%, choose batch 2.
6. For Whisper, choose the lowest-median combination only if its gain over current batch 8/workers 1 is ≥5% and p95 does not regress >10%; otherwise retain batch 8/workers 1.

- [ ] **Step 4: Produce blind A/B quality files**

Generate quality/balanced/turbo outputs with neutral filenames:

```text
.hermes/perf/audio/sample-a.wav
.hermes/perf/audio/sample-b.wav
.hermes/perf/audio/sample-c.wav
```

Keep the profile mapping in a separate local JSON not shown during listening. Validate each file as mono 24 kHz, non-empty, without excessive internal silence or runaway duration. Record only `pass`, `fail`, or `preference` in the committed report; do not commit audio or fixture text.

- [ ] **Step 5: Apply the adoption decision test-first**

If balanced passes, update the expected default in `tests/test_performance_profiles.py` first and verify RED if the implementation default differs. Then change only the accepted values in `_PROFILES`/Settings. If balanced fails quality, keep `quality` as default and document balanced as explicit. If batch 4 misses the 5% gate, set balanced item limit to 2.

- [ ] **Step 6: Write the decision report**

`docs/performance/local-model-m1max.md` must include:

- hardware/software versions;
- baseline and candidate median/p95;
- generated-audio-seconds per wall second;
- peak RSS and MPS memory;
- quality verdict;
- selected defaults;
- rejected candidates and exact gate missed;
- blocked prerequisites;
- statement that Whisper remains CPU because its installed backend lacks MPS support.

Do not include serial number, hardware UUID, user paths, voice ID, fixture text, transcripts, or credentials.

- [ ] **Step 7: Run profile and benchmark tests**

Run:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_performance_profiles.py tests/test_performance_harness.py \
  tests/test_local_baseline_cli.py tests/test_omnivoice_generation_config.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit only code/tests/docs, not local evidence or audio**

```bash
git add pipeline/performance_profiles.py backend/config.py README.md .env.example \
  tests/test_performance_profiles.py docs/performance/local-model-m1max.md
git diff --cached --name-only
```

Verify `.hermes/perf/` and `web/static/previews/` are absent from staged output, then commit:

```bash
git commit -m "perf: select measured M1 Max defaults"
```

If prerequisites are blocked and no production defaults change, commit only the blocked evidence report/docs with:

```bash
git commit -m "docs: record local model benchmark blockers"
```

---

### Task 9: Full Regression, Real-Model Smoke, and Final Review

**Files:**
- Modify only for verified documentation corrections: `README.md`, `.env.example`, `docs/performance/local-model-m1max.md`
- No production source changes unless a failing regression receives its own RED/GREEN fix cycle.

**Interfaces:**
- Verifies every interface and acceptance gate from Tasks 1–8.

- [ ] **Step 1: Run deterministic focused suites**

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_performance_profiles.py tests/test_accelerator_policy.py \
  tests/test_performance_harness.py tests/test_local_baseline_cli.py \
  tests/test_speech_synthesis.py tests/test_tts_batch.py \
  tests/test_omnivoice_speech.py tests/test_omnivoice_generation_config.py \
  tests/test_model_prewarm.py tests/test_runtime_info.py \
  tests/test_job_telemetry.py tests/test_job_persistence.py \
  tests/test_cancel.py tests/test_text_to_voice.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full deterministic regression**

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest -o addopts='' -q
```

Expected: zero failures. Record the exact pass/skip/warning count in the performance report.

- [ ] **Step 3: Run opt-in OmniVoice provider smoke when authorized**

```bash
RUN_PROVIDER_SMOKE=1 \
NO_PROXY= no_proxy= \
/Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_provider_smoke.py -m provider_smoke -q
```

Missing `PROVIDER_SMOKE_CLONE_VOICE_ID` is `BLOCKED/SKIP`, not pass. When available, require both Vietnamese and English PCM smoke cases to pass.

- [ ] **Step 4: Inject deterministic fallback/cancellation cases**

Run the exact tests covering:

```bash
NO_PROXY= no_proxy= /Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/.venv/bin/pytest \
  tests/test_accelerator_policy.py \
  tests/test_speech_synthesis.py -k 'accelerator_retry or cancellation' \
  tests/test_cancel.py tests/test_text_to_voice.py -k 'cancel' -q
```

Expected: PASS; no retry occurs after cancellation and output order remains stable.

- [ ] **Step 5: Inspect staged scope and sensitive-data exclusions**

```bash
git diff --check
git status --short
git diff --cached --name-only
```

Verify no `.hermes/perf/`, `web/static/previews/`, custom voice files, generated WAV/MP3, `.env`, credentials, transcripts, or absolute local paths are staged.

- [ ] **Step 6: Conduct final spec and standards review**

Review the range from the pre-plan base to HEAD against:

- `docs/superpowers/specs/2026-09-06-local-model-performance-optimization-design.md`;
- adoption gates in section 10;
- privacy/cancellation/artifact compatibility;
- MPS failure classification and bounded retry;
- absence of false GPU claims for Whisper.

Any actionable finding receives a new failing test before a fix. Re-run the focused and full suites after fixes.

- [ ] **Step 7: Commit final documentation evidence**

```bash
git add README.md .env.example docs/performance/local-model-m1max.md
git commit -m "docs: record local model performance verification"
```

Skip this commit if those files did not change.

- [ ] **Step 8: Present integration options**

Do not merge automatically. Report measured gains, selected defaults, blocked real-model checks, exact test totals, and the branch commit range. Then offer:

1. Push and create a pull request.
2. Keep the branch for additional local benchmarks.
3. Merge locally only if explicitly requested after verification.
