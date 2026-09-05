from __future__ import annotations


def test_whisper_cpu_batch_size_is_configurable(monkeypatch):
    from pipeline import whisper_stt

    monkeypatch.setattr(whisper_stt, "_thiet_bi", lambda: ("cpu", "int8"))

    transcriber = whisper_stt.WhisperTranscriber(cpu_batch_size=8)

    assert transcriber.stt_batch_size == 8
    assert transcriber.stt_timed is True


def test_whisper_cpu_batch_size_zero_disables_batching(monkeypatch):
    from pipeline import whisper_stt

    monkeypatch.setattr(whisper_stt, "_thiet_bi", lambda: ("cpu", "int8"))

    transcriber = whisper_stt.WhisperTranscriber(cpu_batch_size=0)

    assert transcriber.stt_batch_size == 0
    assert transcriber.stt_timed is False


def test_whisper_cpu_batch_size_is_clamped_to_safe_limit(monkeypatch):
    from pipeline import whisper_stt

    monkeypatch.setattr(whisper_stt, "_thiet_bi", lambda: ("cpu", "int8"))

    transcriber = whisper_stt.WhisperTranscriber(cpu_batch_size=999)

    assert transcriber.stt_batch_size == whisper_stt.MAX_CPU_BATCH_SIZE


def test_backend_forwards_cpu_batch_size_to_whisper(monkeypatch):
    from pipeline import backends

    seen = {}

    class FakeWhisper:
        def __init__(self, model, compute_type, *, cpu_batch_size):
            seen.update(
                model=model,
                compute_type=compute_type,
                cpu_batch_size=cpu_batch_size,
            )

    config = backends.ProviderConfig(
        stt_provider="whisper",
        tts_provider="edge",
        gemini_api_key="",
        gemini_stt_model="m",
        gemini_translate_model="m",
        gemini_tts_model="m",
        # This field is introduced by the implementation immediately after RED.
        whisper_cpu_batch_size=6,
    )

    # Import is lazy in build_backend, so patch the actual module seam too.
    monkeypatch.setattr("pipeline.whisper_stt.WhisperTranscriber", FakeWhisper)
    backends.build_backend(config)

    assert seen == {"model": "small", "compute_type": "int8", "cpu_batch_size": 6}
