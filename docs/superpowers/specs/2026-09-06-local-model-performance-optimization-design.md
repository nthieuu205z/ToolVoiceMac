# Local Model Performance Optimization Design

**Date:** 2026-09-06
**Project:** ToolVietSubMac
**Status:** Approved in design discussion

## 1. Summary

ToolVietSubMac will optimize local model execution for the target Apple Silicon machine without pretending that every workload belongs on the GPU.

The target machine is a MacBook Pro with Apple M1 Max, 10 CPU cores, 32 GPU cores, and 64 GB unified memory. The current environment uses Python 3.12.14 arm64, PyTorch 2.13.0, OmniVoice 0.2.1, faster-whisper 1.2.1, CTranslate2 4.8.1, Transformers 5.16.1, and Accelerate 1.14.0. PyTorch MPS is built and available.

The design keeps Whisper on its supported optimized CPU path and concentrates GPU work on OmniVoice through measured MPS policies. It adds repeatable benchmark evidence, named performance profiles, length-aware batches, recoverable accelerator fallback, coordinated model prewarming, and non-sensitive runtime telemetry.

No optimization ships solely because it uses MPS. Every production change must pass latency, memory, correctness, cancellation, and audio-quality gates on the target machine.

## 2. Baseline Evidence

The current committed baseline at `.hermes/perf/local-baseline-m1max.json` is useful directional evidence but contains one warm measurement per case, so it is not sufficient for a final adoption decision.

### Whisper `small`, CPU `int8`, 12 sampled regions

| Workers | Wall time | Throughput | Peak RSS |
|---:|---:|---:|---:|
| 1 | 8.508 s | 1.410 items/s | 2784 MB |
| 2 | 8.236 s | 1.457 items/s | 2696 MB |
| 4 | 8.404 s | 1.428 items/s | 2599 MB |

The current evidence does not justify more than two outer workers. Because `faster-whisper` uses CTranslate2 and has no MPS backend, Apple GPU acceleration is not a valid implementation target for this adapter.

### OmniVoice, MPS `float16`, 32 generation steps

| Batch | Wall time | Throughput | MPS driver memory after |
|---:|---:|---:|---:|
| 1 | 6.871 s | 0.146 items/s | 2211 MB |
| 2 | 13.167 s | 0.152 items/s | 2211 MB |
| 4 | 25.627 s | 0.156 items/s | 3235 MB |
| 8 | 51.270 s | 0.156 items/s | 4259 MB |

Batch 4 and batch 8 have effectively equal throughput in this sample, while batch 8 adds roughly 1 GB of driver memory and doubles per-batch latency. The production default must therefore not assume that the largest safe batch is fastest.

## 3. Goals

- Reduce warm OmniVoice synthesis time on the target M1 Max while controlling unified-memory growth.
- Reduce first-use latency without loading Whisper and OmniVoice concurrently in an uncontrolled way.
- Preserve output ordering, missing-item behavior, cancellation, privacy, persistence, and artifact publication.
- Replace the permanent one-error MPS downgrade with classified, bounded, recoverable fallback.
- Make the effective profile, device, batch, and fallback state observable without exposing prompts, transcripts, paths, or credentials.
- Produce repeatable cold/warm median and p95 benchmark evidence for every adopted setting.
- Preserve CPU/CUDA behavior and retain Edge/Gemini provider paths unchanged.

## 4. Non-goals

- Forcing faster-whisper onto MPS.
- Replacing Whisper with a different ASR engine in this optimization wave.
- Porting OmniVoice to MLX or CoreML without an independently verified compatible adapter.
- Disabling PyTorch allocator safety limits.
- Running multiple concurrent calls against the same OmniVoice model object.
- Automatically choosing a different voice or provider after a model failure.
- Persisting submitted text, audio, transcripts, absolute paths, or exception messages in benchmark reports.
- Claiming GPU utilization when only memory allocation is measurable.

## 5. Locked Architecture

### 5.1 Workload placement

- Whisper remains `cpu/int8` on Apple Silicon.
- OmniVoice remains `mps/float16`, loaded through CPU staging and then moved to MPS.
- The OmniVoice audio tokenizer remains on CPU because the current upstream MPS path requires it.
- CUDA behavior remains available for supported NVIDIA systems and is not derived from Apple-specific limits.

### 5.2 Named performance profiles

Create immutable profiles in `pipeline/performance_profiles.py`:

| Profile | OmniVoice steps | Initial max items | Initial max characters | Intent |
|---|---:|---:|---:|---|
| `quality` | 32 | 2 | 600 | Preserve current generation quality with bounded latency |
| `balanced` | 24 | 4 | 900 | Candidate default; reduce generation work while keeping robust speech |
| `turbo` | 16 | 4 | 1200 | Explicit speed-first mode |

`balanced` is the intended candidate default. `quality` remains the configured default while the benchmark and blind quality gates are being built. Task 8 may promote `balanced` to the default only if it passes section 10; otherwise `quality` remains the default. The implementation may not silently substitute unmeasured values.

The selected profile is configured by `LOCAL_PERFORMANCE_PROFILE=quality|balanced|turbo`. Existing explicit `OMNIVOICE_NUM_STEP` and `OMNIVOICE_BATCH_SIZE` values remain supported as advanced overrides. An explicit override is reported in telemetry and takes precedence over the profile field it replaces.

### 5.3 Length-aware batching

The batching contract gains two independent limits:

- maximum item count;
- maximum total normalized character count.

Inputs remain stably grouped by similar length to limit longest-sequence padding. Groups preserve original indexes when results are reassembled. No batch may exceed either effective limit. Single inputs longer than the character budget remain single-item batches rather than being truncated.

The provider interface exposes `batch_size` and `batch_character_limit`. Edge/Gemini keep their current behavior. OmniVoice resolves both values from the active profile and adaptive controller.

### 5.4 Adaptive MPS fallback

Replace `_MPS_BATCH_SAFE` as the production decision source with a thread-safe `AdaptiveBatchController` scoped to OmniVoice MPS.

The controller maintains:

- requested item limit;
- current effective item limit;
- consecutive accelerator failure count;
- monotonic cooldown deadline;
- last non-sensitive fallback reason;
- successful call count since fallback.

The fallback ladder is `4 → 2 → 1` for `balanced` and `turbo`, and `2 → 1` for `quality`. An explicit lower override starts at that lower value.

Only classified accelerator failures change the controller:

- MPS out-of-memory/allocation failures;
- Metal/MPS native runtime failures;
- unsupported MPS operation failures.

Malformed input, missing voice assets, empty model output, cancellation, and ordinary provider/content errors do not reduce the batch limit.

After a reduction, the controller keeps the lower limit for a 120-second cooldown and at least three successful calls. It then permits one probe at the next higher ladder step. A successful probe restores that step; a failed probe restarts cooldown. The model remains serialized behind `_INFER_LOCK`.

If single-item MPS inference fails with a classified accelerator error, the job returns the existing safe `SpeechServiceError`. This wave does not move the loaded OmniVoice model back to CPU mid-process because that migration is expensive and unverified.

### 5.5 Coordinated prewarming

Create a process-wide `ModelPrewarmCoordinator` with one daemon worker and an ordered queue.

On Apple Silicon with both local models ready:

1. prewarm Whisper first;
2. wait for its load attempt to finish;
3. prewarm OmniVoice second.

This prevents simultaneous CPU model loading, MPS context initialization, and unified-memory pressure. A real job may still call the existing lazy singleton loaders; singleton locks guarantee only one model instance.

On systems requiring only one local model, only that model is queued. Startup never downloads a missing model. Prewarm failures remain warnings and do not make the server unavailable.

### 5.6 Benchmark system

Extend `benchmarks/performance_harness.py` with synchronized repeated-series measurements:

- one cold run where feasible;
- at least five warm runs;
- median;
- p95;
- error count;
- items per second;
- generated audio seconds per wall second for TTS;
- peak RSS;
- peak/current MPS allocated and driver memory.

`benchmarks/run_local_baseline.py` accepts the benchmark voice ID explicitly and runs:

- Whisper CPU batch sizes `4, 8, 12, 16` with workers `1, 2, 4`;
- OmniVoice profiles `quality, balanced, turbo`;
- item batches `1, 2, 4, 8` where allowed;
- short, medium, long, and mixed-length fixtures;
- cold and warm series in separate processes when testing MPS environment variables.

Reports remain under `.hermes/perf/` and exclude private content and user paths.

### 5.7 Telemetry

Extend runtime facts additively with:

- `performance_profile`;
- `requested_batch_size`;
- `effective_batch_size`;
- `batch_character_limit`;
- `fallback_reason`;
- `mps_allocated_mb`;
- `mps_driver_mb`.

The job snapshot and model status API may expose these non-sensitive values. Existing `engine`, `device`, and `batch_size` fields remain compatible; `batch_size` represents the current effective item limit.

## 6. Data Flow

```text
Settings
  └─ LOCAL_PERFORMANCE_PROFILE + advanced overrides
       └─ resolve_performance_profile()
            ├─ Whisper CPU batch/worker candidates
            └─ OmniVoice steps + requested item/character limits

Text/video synthesis
  └─ synthesize_texts()
       └─ length-aware group builder
            └─ OmniVoiceSynthesizer
                 ├─ AdaptiveBatchController.effective_limit()
                 ├─ serialized model.generate() on MPS
                 ├─ classified fallback and bounded retry
                 └─ runtime_info() telemetry
```

## 7. Error Handling

- Cancellation always wins before another batch and before publishing artifacts.
- Adaptive retries preserve original indexes and retry only the affected group.
- If a reduced group still fails for a non-accelerator reason, existing per-item fallback behavior applies without changing GPU capability.
- Benchmark failures record only exception type, never exception text.
- Unsupported profile names fail configuration validation before jobs start.
- Explicit batch/step overrides are clamped to documented safe positive ranges.
- Prewarm failures are logged once and lazy load remains available.

## 8. Compatibility

- `TTS_PROVIDER=edge|gemini` is unchanged.
- `CLONE_TTS_PROVIDER=none` does not import torch or OmniVoice.
- Existing callers that omit performance profile use `quality` until the measured adoption task explicitly promotes `balanced`.
- Existing tests and fakes that expose only `batch_size` remain valid; the character limit defaults to unlimited for providers that do not advertise it.
- Existing runtime tuple implementations remain accepted while the project adapter returns richer runtime metadata.
- Existing video and Text → Voice artifact contracts remain unchanged.

## 9. Testing Strategy

### Deterministic tests

- Profile resolution, validation, override precedence, and serialization.
- Length-aware grouping respects both limits and preserves output indexes.
- Failure classifier changes capability only for accelerator failures.
- Batch ladder reduction, cooldown, successful re-probe, and thread safety.
- Cancellation during a retry does not run another batch.
- Prewarm order and no-download behavior.
- Rich runtime telemetry remains secret-safe and backward-compatible.
- Benchmark series calculates median/p95 and sanitizes reports.

### Real-model benchmarks

Real model runs are opt-in and require an authorized local clone voice ID. They run on the exact project venv and target machine. Each adopted profile requires:

- one cold run;
- at least five warm runs;
- short, medium, long, and mixed-length cases;
- both `vi-VN` and `en-US`;
- cancellation injection between batches;
- MPS memory sampling;
- output count and non-empty PCM validation.

### Audio quality

Generate neutral A/B WAV files under `.hermes/perf/audio/` for the three profiles. Use the same authorized voice and fixed non-private fixture text. Record blind preference plus automated checks for non-empty audio, sample rate, internal silence, runaway duration, and missing outputs.

## 10. Adoption Gates

A production optimization is accepted only when all applicable gates pass:

- at least 10% lower warm median end-to-end synthesis time for profile or batching changes;
- at least 5% lower warm median for low-risk cache/prewarm changes;
- p95 does not regress by more than 10%;
- no increase in missing/empty outputs;
- no cancellation, ordering, persistence, or artifact-publication regression;
- no native crash;
- MPS driver memory remains below 8 GB for the candidate balanced profile on the target M1 Max;
- balanced profile passes blind intelligibility/speaker-similarity review against quality;
- turbo is labeled speed-first and never becomes default automatically;
- full deterministic pytest passes;
- real provider/model checks that lack prerequisites are reported blocked, not passed.

## 11. Rollback Rules

- If `balanced` misses its quality gate, keep `quality` as the default and leave `balanced` explicit.
- If batch 4 improves throughput by less than 5% over batch 2 for mixed-length inputs, balanced defaults to batch 2.
- If character-budget batching does not improve median or memory by 5%, remove it rather than shipping inactive complexity.
- If re-probe causes repeated native failures, keep the controller at batch 1 for the process and report the reason; do not loop indefinitely.
- If coordinated prewarm increases time-to-ready or peak memory, retain lazy loading and disable prewarm through configuration.
- MLX/CoreML work requires a separate design and benchmark because no compatible adapter is established in this scope.
