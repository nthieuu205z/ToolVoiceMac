"""OmniVoice adapter tests without downloading the model."""

from __future__ import annotations

import builtins
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pipeline import custom_voices
from pipeline.audio import pcm_to_array, write_wav
from pipeline.errors import SpeechServiceError
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.omnivoice_speech import OmniVoiceSynthesizer


class FakeOmni:
    def __init__(self, audio: np.ndarray | None = None, error: Exception | None = None):
        self.audio = audio if audio is not None else np.full(2400, 0.3, dtype=np.float32)
        self.error = error
        self.calls: list[dict] = []

    def generate(self, text, ref_audio, ref_text, **kwargs):
        self.calls.append({"text": text, "ref_audio": ref_audio, "ref_text": ref_text, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        count = len(text) if isinstance(text, (list, tuple)) else 1
        return [self.audio for _ in range(count)]


class FakeTranscriber:
    def __init__(self, text: str = "lời tham chiếu", error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> str:
        self.calls.append(Path(path))
        if self.error is not None:
            raise self.error
        return self.text


def _make_clone(voice_id: str = "clone-test", display: str = "Giọng Test") -> str:
    write_wav(
        custom_voices.sample_path(voice_id),
        np.zeros(2400, dtype="<i2"),
        TTS_SAMPLE_RATE,
    )
    custom_voices.register(voice_id, display)
    return voice_id


def _sidecar(voice_id: str) -> Path:
    return custom_voices.safe_sidecar_path(voice_id)


def test_synthesize_returns_pcm16_and_passes_vietnamese_generation_options():
    voice_id = _make_clone()
    audio = (0.5 * np.sin(
        2 * np.pi * 220 * np.arange(TTS_SAMPLE_RATE // 2) / TTS_SAMPLE_RATE
    )).astype(np.float32)
    model = FakeOmni(audio=audio)
    synth = OmniVoiceSynthesizer(
        model=model,
        transcriber=FakeTranscriber(text="câu mẫu"),
        num_step=16,
    )

    pcm = synth.synthesize("Xin chào", voice_id)

    arr = pcm_to_array(pcm)
    assert len(arr) == len(audio)
    assert arr.dtype == np.dtype("<i2")
    assert int(np.abs(arr).max()) > 1000
    assert model.calls[0]["text"] == ["Xin chào"]
    assert model.calls[0]["ref_text"] == ["câu mẫu"]
    assert model.calls[0]["kwargs"]["language"] == ["Vietnamese"]
    assert model.calls[0]["kwargs"]["generation_config"].num_step == 16
    assert model.calls[0]["kwargs"]["generation_config"].postprocess_output is True


def test_ref_text_is_computed_once_and_persisted():
    voice_id = _make_clone()
    transcriber = FakeTranscriber(text="mẫu")
    synth = OmniVoiceSynthesizer(model=FakeOmni(), transcriber=transcriber)

    synth.synthesize("a", voice_id)
    synth.synthesize("b", voice_id)

    assert len(transcriber.calls) == 1
    assert _sidecar(voice_id).read_text(encoding="utf-8").strip() == "mẫu"


def test_ref_text_is_reused_from_sidecar_without_transcribing():
    voice_id = _make_clone()
    _sidecar(voice_id).write_text("có sẵn", encoding="utf-8")
    transcriber = FakeTranscriber(error=AssertionError("không được gọi transcriber"))
    model = FakeOmni()

    OmniVoiceSynthesizer(model=model, transcriber=transcriber).synthesize("x", voice_id)

    assert transcriber.calls == []
    assert model.calls[0]["ref_text"] == ["có sẵn"]


def test_synthesize_batch_generates_all_items_in_one_call():
    voice_id = _make_clone()
    model = FakeOmni()
    synth = OmniVoiceSynthesizer(model=model, transcriber=FakeTranscriber(text="m"))
    texts = ["câu một", "câu hai", "câu ba"]

    pcms = synth.synthesize_batch(texts, voice_id)

    assert len(pcms) == 3
    assert all(pcms)
    assert len(model.calls) == 1
    assert model.calls[0]["text"] == texts
    assert model.calls[0]["ref_audio"] == [str(custom_voices.sample_path(voice_id))] * 3


def test_synthesize_batch_rejects_more_than_configured_batch_limit(monkeypatch):
    import pipeline.omnivoice_speech as ov

    voice_id = _make_clone("clone-batch-limit")
    model = FakeOmni()
    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")
    monkeypatch.setattr(ov, "_MPS_BATCH_LIMIT", 2)

    with pytest.raises(SpeechServiceError, match="cỡ lô"):
        OmniVoiceSynthesizer(model=model, batch_size=2, transcriber=FakeTranscriber()).synthesize_batch(
            ["a", "b", "c"], voice_id
        )


def test_mps_batch_failure_disables_future_multi_item_batches(monkeypatch):
    import pipeline.omnivoice_speech as ov

    class FailingModel:
        def generate(self, *, text, **kwargs):
            if len(text) > 1:
                raise RuntimeError("MPS batch failed")
            return [np.ones(2400, dtype=np.float32) for _ in text]

    voice_id = _make_clone("clone-batch-fails")
    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")
    monkeypatch.setattr(ov, "_MPS_BATCH_SAFE", True)

    with pytest.raises(SpeechServiceError):
        OmniVoiceSynthesizer(
            model=FailingModel(),
            transcriber=FakeTranscriber(text="mẫu"),
            batch_size=2,
        ).synthesize_batch(["a", "b"], voice_id)

    assert ov.mps_batch_safe() is False
    assert OmniVoiceSynthesizer(model=FakeOmni()).batch_size == 1


def test_mps_batch_failure_is_retried_as_single_items_by_pipeline(monkeypatch):
    import pipeline.omnivoice_speech as ov
    from pipeline.models import Segment
    from pipeline.tts import synthesize_segments

    class FailingModel:
        def generate(self, *, text, **kwargs):
            if len(text) > 1:
                raise RuntimeError("MPS batch failed")
            return [np.ones(2400, dtype=np.float32) for _ in text]

    voice_id = _make_clone("clone-batch-recover")
    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")
    monkeypatch.setattr(ov, "_MPS_BATCH_SAFE", True)
    synth = OmniVoiceSynthesizer(
        model=FailingModel(),
        transcriber=FakeTranscriber(text="mẫu"),
        batch_size=2,
    )
    segments = [
        Segment(0.0, 2.0, "a", "a"),
        Segment(3.0, 5.0, "b", "b"),
    ]
    backend = type(
        "Backend",
        (),
        {
            "batch_size": 2,
            "mps_batch_safe": True,
            "synthesize_batch": synth.synthesize_batch,
            "synthesize": synth.synthesize,
        },
    )()

    fitted, warnings = synthesize_segments(
        backend, segments, voice_id, total_duration=10.0
    )

    assert len(fitted) == 2
    assert warnings == []


def test_empty_batch_is_a_noop():
    voice_id = _make_clone()
    model = FakeOmni()

    assert OmniVoiceSynthesizer(model=model).synthesize_batch([], voice_id) == []
    assert model.calls == []


def test_non_custom_voice_is_rejected():
    with pytest.raises(SpeechServiceError):
        OmniVoiceSynthesizer(model=FakeOmni()).synthesize("x", "preset")


def test_missing_sample_is_rejected():
    with pytest.raises(SpeechServiceError):
        OmniVoiceSynthesizer(model=FakeOmni()).synthesize("x", "clone-missing")


def test_empty_audio_is_rejected():
    voice_id = _make_clone()
    with pytest.raises(SpeechServiceError):
        OmniVoiceSynthesizer(
            model=FakeOmni(audio=np.zeros(0, dtype=np.float32)),
            transcriber=FakeTranscriber(text="m"),
        ).synthesize("x", voice_id)


def test_model_error_is_wrapped():
    voice_id = _make_clone()
    with pytest.raises(SpeechServiceError):
        OmniVoiceSynthesizer(
            model=FakeOmni(error=RuntimeError("boom")),
            transcriber=FakeTranscriber(text="m"),
        ).synthesize("x", voice_id)


def test_batch_override_is_honored_on_cuda_and_mps(monkeypatch):
    import pipeline.omnivoice_speech as ov

    for device in ("cuda", "mps"):
        monkeypatch.setattr(ov, "_accel_device", lambda device=device: device)
        assert OmniVoiceSynthesizer(model=FakeOmni(), batch_size=5).batch_size == 5


def test_batch_override_is_clamped_to_a_positive_value():
    synth = OmniVoiceSynthesizer(model=FakeOmni(), batch_size=-4)

    assert synth._batch_override == 0


def test_default_mps_batch_is_capped_at_the_knee(monkeypatch):
    import pipeline.omnivoice_speech as ov

    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")

    assert OmniVoiceSynthesizer(model=FakeOmni()).batch_size == 8


def test_default_mps_batch_is_capped_even_with_a_large_override(monkeypatch):
    import pipeline.omnivoice_speech as ov

    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")

    assert OmniVoiceSynthesizer(model=FakeOmni(), batch_size=64).batch_size == 8


def test_mps_batch_can_be_disabled_without_falling_back_to_cpu(monkeypatch):
    import pipeline.omnivoice_speech as ov

    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")
    monkeypatch.setattr(ov, "_MPS_BATCH_SAFE", False)

    assert OmniVoiceSynthesizer(model=FakeOmni(), batch_size=8).batch_size == 1


def test_mps_batch_safety_flag_is_not_a_generic_probe(monkeypatch):
    import pipeline.omnivoice_speech as ov

    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")
    monkeypatch.setattr(ov, "_MPS_BATCH_SAFE", False)

    assert OmniVoiceSynthesizer(model=FakeOmni()).batch_size == 1


def test_mps_single_safety_flag_can_disable_accelerator(monkeypatch):
    import pipeline.omnivoice_speech as ov

    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")
    monkeypatch.setattr(ov, "_MPS_SINGLE_SAFE", False)

    assert OmniVoiceSynthesizer(model=FakeOmni()).batch_size == 0


def test_marking_mps_batch_unsafe_reduces_future_batch_size(monkeypatch):
    import pipeline.omnivoice_speech as ov

    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")
    try:
        ov.mark_mps_batch_unsafe()
        assert OmniVoiceSynthesizer(model=FakeOmni()).batch_size == 1
    finally:
        ov._MPS_BATCH_SAFE = True


def test_cpu_has_no_batch(monkeypatch):
    import pipeline.omnivoice_speech as ov

    monkeypatch.setattr(ov, "_accel_device", lambda: None)

    assert OmniVoiceSynthesizer(model=FakeOmni(), batch_size=5).batch_size == 0


def test_mps_load_is_staged_through_cpu_then_moved(monkeypatch):
    import sys
    import types
    import pipeline.omnivoice_speech as ov

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"

    class FakeModel:
        device = "cpu"

        def __init__(self):
            self.to_calls: list[str] = []
            self.audio_tokenizer = FakeTokenizer()

        def to(self, device):
            self.to_calls.append(device)
            self.audio_tokenizer.to(device)
            self.device = device
            return self

    class FakeTokenizer:
        def __init__(self):
            self.device = "cpu"
            self.to_calls: list[str] = []

        def to(self, device):
            self.to_calls.append(device)
            self.device = device
            return self

    class FakeOmniVoice:
        model = FakeModel()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.args = args
            cls.kwargs = kwargs
            return cls.model

    ov._reset_model_for_tests()
    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "omnivoice", types.SimpleNamespace(OmniVoice=FakeOmniVoice))

    ov._get_model()

    assert FakeOmniVoice.kwargs["device_map"] == "cpu"
    assert FakeOmniVoice.kwargs["dtype"] == "float16"
    assert FakeOmniVoice.model.to_calls[-1] == "mps"
    assert FakeOmniVoice.model.audio_tokenizer.device == "cpu"
    assert FakeOmniVoice.model.audio_tokenizer.to_calls[-1:] == ["cpu"]
    assert ov._MODEL_DEVICE == "mps"
    assert ov._tokenizer_device(FakeOmniVoice.model) == "cpu"


def test_mps_staged_model_reports_the_model_device(monkeypatch):
    import pipeline.omnivoice_speech as ov

    class FakeModel:
        device = "mps:0"

    ov._reset_model_for_tests()
    ov._MODEL = FakeModel()
    ov._MODEL_DEVICE = "mps"

    assert OmniVoiceSynthesizer().device() == "mps:0"


def test_model_load_is_serialized(monkeypatch):
    import sys
    import types
    import pipeline.omnivoice_speech as ov

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"

    class FakeModel:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

    class FakeOmniVoice:
        calls = 0
        model = FakeModel()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls += 1
            return cls.model

    ov._reset_model_for_tests()
    monkeypatch.setattr(ov, "_accel_device", lambda: "mps")
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "omnivoice", types.SimpleNamespace(OmniVoice=FakeOmniVoice))

    first = threading.Thread(target=ov._get_model)
    second = threading.Thread(target=ov._get_model)
    first.start()
    second.start()
    first.join()
    second.join()

    assert FakeOmniVoice.calls == 1
    assert ov._MODEL_DEVICE == "mps"


def test_missing_torch_is_reported_as_installation_error(monkeypatch):
    import pipeline.omnivoice_speech as ov

    ov._reset_model_for_tests()
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(SpeechServiceError, match="PyTorch"):
        ov._get_model()


def test_reference_text_is_transcribed_once_across_synthesizers():
    voice_id = _make_clone("clone-concurrent")
    calls = 0
    started = threading.Event()

    def transcribe(_sample):
        nonlocal calls
        calls += 1
        started.set()
        time.sleep(0.05)
        return "câu mẫu"

    synths = [
        OmniVoiceSynthesizer(model=FakeOmni(), transcriber=transcribe)
        for _ in range(2)
    ]
    results = []

    def run(synth):
        results.append(synth._ref_text(voice_id, custom_voices.sample_path(voice_id)))

    threads = [threading.Thread(target=run, args=(synth,)) for synth in synths]
    for thread in threads:
        thread.start()
    assert started.wait(2)
    for thread in threads:
        thread.join(timeout=5)

    assert results == ["câu mẫu", "câu mẫu"]
    assert calls == 1


def test_forget_clone_removes_reftext_and_shared_cache():
    import pipeline.omnivoice_speech as ov

    voice_id = _make_clone("clone-cache-clear")
    synth = OmniVoiceSynthesizer(model=FakeOmni(), transcriber=lambda _sample: "câu mẫu")
    assert synth._ref_text(voice_id, custom_voices.sample_path(voice_id)) == "câu mẫu"
    assert any(key[0] == voice_id for key in ov._REF_TEXT_CACHE)

    ov.forget_clone(voice_id)

    assert not _sidecar(voice_id).exists()
    assert not any(key[0] == voice_id for key in ov._REF_TEXT_CACHE)


def test_reference_cache_key_changes_when_sample_changes():
    voice_id = _make_clone("clone-sample-replaced")
    sample = custom_voices.sample_path(voice_id)
    sidecar = _sidecar(voice_id)
    calls = []
    synth = OmniVoiceSynthesizer(
        model=FakeOmni(),
        transcriber=lambda _sample: calls.append(1) or f"câu {len(calls)}",
    )

    assert synth._ref_text(voice_id, sample) == "câu 1"
    sidecar.unlink()
    sample.write_bytes(sample.read_bytes() + b"\x00")
    synth._ref_cache.clear()

    assert synth._ref_text(voice_id, sample) == "câu 2"
    assert len(calls) == 2


def test_empty_reference_text_is_reported_before_model_inference():
    voice_id = _make_clone("clone-empty-reference")
    synth = OmniVoiceSynthesizer(model=FakeOmni(), transcriber=lambda _sample: " ")

    with pytest.raises(SpeechServiceError, match="Không lấy được lời"):
        synth.synthesize("xin chào", voice_id)
