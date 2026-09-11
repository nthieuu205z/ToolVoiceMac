"""Synthetic contracts; only the upstream session is substituted."""
from dataclasses import asdict, dataclass
import importlib
import importlib.util

import numpy as np
import pytest

from pipeline.omnivoice_settings import OmniVoiceSettings


def runtime_module():
    assert importlib.util.find_spec("pipeline.omnivoice_runtime") is not None, "standalone runtime is missing"
    return importlib.import_module("pipeline.omnivoice_runtime")


@dataclass
class FakeConfig:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = 0.1
    fade_duration: float = 0.1


class FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.prompt_calls = []
        self.generate_calls = []
        self.prompt = object()
        self.output: object = None
        self.error = None

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(kwargs)
        return self.prompt

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.error:
            raise self.error
        if self.output is not None:
            return self.output
        return [np.array([-2.0, -0.5, 0, 0.5, 2.0], dtype=np.float32) for _ in kwargs["text"]]


def make_upstream(tmp_path):
    module = runtime_module()
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"synthetic reference; fake upstream does not decode")
    model = FakeModel()
    calls, events = [], []
    session = module.UpstreamSession(
        model=model, config_factory=FakeConfig,
        synchronize=lambda: events.append("sync"),
        seed=lambda value: events.append(("seed", value)),
        provenance={"runtime": "omnivoice", "device": "mps", "dtype": "float16"},
    )

    def loader(checkpoint):
        calls.append(checkpoint)
        events.append("load")
        return session

    return module, reference, model, loader, calls, events


def test_order_config_pcm_and_session_prompt_reuse(tmp_path):
    module, reference, model, loader, calls, events = make_upstream(tmp_path)
    runtime = module.OmniVoiceRuntime(reference, "Explicit transcript", loader=loader)
    assert calls == []
    first = runtime.generate(["Second input", "First input"], language="vi-VN", seed=1234)
    second = runtime.generate(["Another"], language="en-US", seed=5678)
    assert len(calls) == 1
    assert model.prompt_calls == [{"ref_audio": str(reference), "ref_text": "Explicit transcript", "preprocess_prompt": True}]
    assert [call["text"] for call in model.generate_calls] == [["Second input", "First input"], ["Another"]]
    assert [call["language"] for call in model.generate_calls] == ["Vietnamese", "English"]
    call = model.generate_calls[0]
    assert call["voice_clone_prompt"] is model.prompt
    assert call["speed"] == 1.0 and call["duration"] is None
    assert call["normalize_text"] is False
    assert asdict(call["generation_config"]) == asdict(FakeConfig())
    assert set(call) == {"text", "language", "voice_clone_prompt", "speed", "duration", "generation_config", "normalize_text"}
    assert isinstance(first.pcm, tuple) and len(first.pcm) == 2
    assert np.frombuffer(first.pcm[0], dtype="<i2").tolist() == [-32767, -16383, 0, 16383, 32767]
    assert second.sample_rate == 24000
    assert second.timings["load_seconds"] == second.timings["prompt_seconds"] == 0
    assert set(first.timings) == {"load_seconds", "prompt_seconds", "generate_seconds", "pcm_seconds", "total_seconds"}
    assert all(value >= 0 for value in first.timings.values())
    assert [e for e in events if isinstance(e, tuple)] == [("seed", 1234), ("seed", 5678)]
    assert events.count("sync") >= 5
    assert runtime.session.model is model and runtime.prompt is model.prompt
    assert first.provenance["generation_t_shift"] == 0.1
    assert first.provenance["generation_num_step"] == 32
    assert first.provenance["seed"] == 1234
    assert first.provenance["language"] == "vi-VN"
    assert first.provenance["optimization"] == "none"
    assert first.provenance["optimization_source_sha256"] is None


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64])
@pytest.mark.parametrize("samples,expected", [
    ([-1, 0, 1], [-32767, 0, 32767]),
    ([-2, 0, 2], [-32767, 0, 32767]),
    ([-2, 2], [-32767, 32767]),
])
def test_pcm_normalizes_numeric_dtypes_without_overflow(tmp_path, dtype, samples, expected):
    module, reference, model, loader, _, _ = make_upstream(tmp_path)
    model.output = [np.array(samples, dtype=dtype)]
    runtime = module.OmniVoiceRuntime(reference, "transcript", loader=loader)
    with np.errstate(over="raise", invalid="raise"):
        result = runtime.generate(["hello"], language="vi", seed=1)
    assert isinstance(result.pcm, tuple) and len(result.pcm) == 1
    assert isinstance(result.pcm[0], bytes)
    assert np.frombuffer(result.pcm[0], dtype="<i2").tolist() == expected
    assert result.sample_rate == 24000


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pcm_preserves_fractional_truncation(tmp_path, dtype):
    module, reference, model, loader, _, _ = make_upstream(tmp_path)
    model.output = [np.array([-2, -0.5, 0, 0.5, 2], dtype=dtype)]
    runtime = module.OmniVoiceRuntime(reference, "transcript", loader=loader)
    with np.errstate(over="raise", invalid="raise"):
        result = runtime.generate(["hello"], language="vi", seed=1)
    assert np.frombuffer(result.pcm[0], dtype="<i2").tolist() == [-32767, -16383, 0, 16383, 32767]


def test_pcm_float64_rounds_to_float32_before_scaling(tmp_path):
    module, reference, model, loader, _, _ = make_upstream(tmp_path)
    model.output = [np.array([np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0)], dtype=np.float64)]
    runtime = module.OmniVoiceRuntime(reference, "transcript", loader=loader)
    with np.errstate(over="raise", invalid="raise"):
        result = runtime.generate(["hello"], language="vi", seed=1)
    assert np.frombuffer(result.pcm[0], dtype="<i2").tolist() == [-32767, 32767]


@pytest.mark.parametrize("texts,language,seed,settings", [
    ([], "vi", 1, {}), ([" "], "vi", 1, {}), ([1], "vi", 1, {}),
    ("hello", "vi", 1, {}), (["x"] * 65, "vi", 1, {}),
    (["x" * 50001], "vi", 1, {}), (["x"], "private-language", 1, {}),
    (["x"], "vi", True, {}), (["x"], "vi", -1, {}),
    (["x"], "vi", 2**32, {}), (["x"], "vi", 1.5, {}),
    (["x", "y"], "vi", 1, {"duration": 2.0}),
    (["x" * 1001], "vi", 1, {"duration": 2.0}),
])
def test_invalid_requests_never_load(tmp_path, texts, language, seed, settings):
    module, reference, model, loader, calls, _ = make_upstream(tmp_path)
    runtime = module.OmniVoiceRuntime(reference, "transcript", settings=OmniVoiceSettings(**settings), loader=loader)
    with pytest.raises(ValueError):
        runtime.generate(texts, language=language, seed=seed)
    assert calls == [] and model.prompt_calls == []


@pytest.mark.parametrize("case", ["missing", "directory", "blank", "none", "bad-settings", "empty-checkpoint"])
def test_invalid_reference_and_options_never_load(tmp_path, case):
    module, reference, _, loader, calls, _ = make_upstream(tmp_path)
    transcript, settings, checkpoint = "transcript", None, "k2-fsa/OmniVoice"
    if case == "missing":
        reference.unlink()
    elif case == "directory":
        reference = tmp_path
    elif case == "blank":
        transcript = " \n "
    elif case == "none":
        transcript = None
    elif case == "bad-settings":
        settings = {"speed": 1.0}
    elif case == "empty-checkpoint":
        checkpoint = ""
    with pytest.raises((ValueError, FileNotFoundError)):
        runtime = module.OmniVoiceRuntime(reference, transcript, settings=settings, checkpoint=checkpoint, loader=loader)
        runtime.generate(["hello"], language="vi", seed=1)
    assert calls == []


@pytest.mark.parametrize("output", [
    [], [np.zeros(2), np.zeros(2)], [np.array([])], [np.zeros((1, 2))],
    [np.array([float("nan")])], [np.array([float("inf")])],
    [np.array([1j])], [np.array(["private"])], [[0.1, 0.2]],
    np.array([0.1]), [np.array([1], dtype=object)],
    [np.array([True])], [np.array(0.1)], [np.array(["0.1"])],
    [np.array([np.nan], dtype=np.float16)],
    [np.array([np.inf], dtype=np.float16)],
    [np.array([-np.inf], dtype=np.float16)],
])
def test_rejects_malformed_audio_before_pcm(tmp_path, output):
    module, reference, model, loader, _, _ = make_upstream(tmp_path)
    model.output = output
    runtime = module.OmniVoiceRuntime(reference, "transcript", loader=loader)
    with pytest.raises(ValueError):
        runtime.generate(["hello"], language="vi", seed=1)
    assert len(model.generate_calls) == 1


def test_wrong_sample_rate_rejected_before_generate(tmp_path):
    module, reference, model, loader, _, _ = make_upstream(tmp_path)
    model.sampling_rate = 16000
    runtime = module.OmniVoiceRuntime(reference, "transcript", loader=loader)
    with pytest.raises(ValueError):
        runtime.generate(["hello"], language="vi", seed=1)
    assert model.generate_calls == []


def test_upstream_failure_is_not_retried_and_releases_lock(tmp_path):
    module, reference, model, loader, calls, _ = make_upstream(tmp_path)
    runtime = module.OmniVoiceRuntime(reference, "transcript", loader=loader)
    model.error = RuntimeError("private failure")
    with pytest.raises(RuntimeError) as caught:
        runtime.generate(["hello"], language="vi", seed=1)
    assert caught.value is model.error and len(model.generate_calls) == 1
    model.error = None
    assert runtime.generate(["hello"], language="vi", seed=1).sample_rate == 24000
    assert len(calls) == 1 and len(model.prompt_calls) == 1


def test_concurrent_generation_rejected_without_blocking(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    module, reference, model, loader, _, _ = make_upstream(tmp_path)
    entered, release = Event(), Event()
    generate = model.generate

    def blocked(**kwargs):
        entered.set()
        assert release.wait(5)
        return generate(**kwargs)

    model.generate = blocked
    runtime = module.OmniVoiceRuntime(reference, "transcript", loader=loader)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runtime.generate, ["one"], language="vi", seed=1)
        try:
            assert entered.wait(5)
            competing = pool.submit(runtime.generate, ["two"], language="vi", seed=2)
            with pytest.raises(RuntimeError):
                competing.result(timeout=1)
        finally:
            release.set()
        assert len(first.result(timeout=5).pcm) == 1
    assert len(model.generate_calls) == 1


def test_settings_forwarded_without_mutating_or_normalizing_text(tmp_path):
    module, reference, model, loader, _, _ = make_upstream(tmp_path)
    settings = OmniVoiceSettings(speed=1.25, duration=3.0, num_step=40, guidance_scale=3.0,
                                 denoise=False, preprocess_prompt=False, postprocess_output=False)
    runtime = module.OmniVoiceRuntime(reference, "  transcript  ", settings=settings, loader=loader)
    result = runtime.generate(["  unchanged!?  "], language="en", seed=1)
    assert model.generate_calls[0]["text"] == ["  unchanged!?  "]
    assert model.generate_calls[0]["speed"] == 1.25
    assert model.generate_calls[0]["duration"] == 3.0
    assert model.generate_calls[0]["generation_config"].num_step == 40
    assert model.prompt_calls[0]["preprocess_prompt"] is False
    assert model.prompt_calls[0]["ref_text"] == "  transcript  "
    assert result.provenance["generation_postprocess_output"] is False


def test_provenance_filters_untrusted_strings_and_nested_values(tmp_path):
    module, reference, _, loader, _, _ = make_upstream(tmp_path)
    session = loader("fixture")
    session.provenance.update({"reference_text": "private", "checkpoint": str(reference),
                               "device": "/private/location", "nested": {"secret": "value"},
                               "torch_version": "2.13.0", "source_sha256": "a" * 64})
    runtime = module.OmniVoiceRuntime(reference, "private transcript", loader=lambda _: session)
    result = runtime.generate(["private input"], language="vi", seed=1)
    import json
    serialized = json.dumps(result.provenance, allow_nan=False)
    assert "private" not in serialized and "secret" not in serialized
    assert result.provenance["torch_version"] == "2.13.0"
    assert result.provenance["source_sha256"] == "a" * 64
    assert all(value is None or type(value) in (str, int, float, bool) for value in result.provenance.values())


def test_import_and_constructor_do_not_import_heavy_or_application_modules(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"synthetic")
    code = '''
import importlib.abc, sys
from pathlib import Path
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'torchaudio', 'omnivoice', 'transformers', 'backend', 'dotenv', 'huggingface_hub'} or fullname == 'pipeline.omnivoice_speech':
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
from pipeline.omnivoice_runtime import OmniVoiceRuntime
runtime = OmniVoiceRuntime(Path(sys.argv[1]), 'explicit transcript')
assert runtime.session is None and runtime.prompt is None
'''
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(reference)],
                               cwd=Path(__file__).resolve().parents[1], env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                               text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def fake_loader_packages(monkeypatch, tmp_path):
    """Replace only heavyweight/library boundaries, not the runtime loader."""
    from types import ModuleType, SimpleNamespace
    import sys

    module = runtime_module()
    events = []
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    (snapshot / "audio_tokenizer").mkdir(parents=True)
    source = tmp_path / "synthetic_upstream.py"
    source.write_text("# Synthetic source fixture only\n", encoding="utf-8")
    model = FakeModel()
    model.device = "cpu"
    model.dtype = "float16"
    model.audio_tokenizer = SimpleNamespace(to=lambda device: events.append(("codec.to", device)))

    def to(device):
        model.device = device
        events.append(("model.to", device))
        return model

    model.to = to
    model.eval = lambda: events.append("eval")

    class FakeOmniVoice:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            events.append(("from_pretrained", checkpoint, kwargs))
            return model

    torch = ModuleType("torch")
    torch.float16 = "float16"
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    torch.mps = SimpleNamespace(synchronize=lambda: events.append("sync"))
    torch.manual_seed = lambda seed: events.append(("torch.seed", seed))
    hub = ModuleType("huggingface_hub")

    def snapshot_download(repo_id, **kwargs):
        events.append(("resolve", repo_id, kwargs))
        assert kwargs == {"local_files_only": True, "token": False}
        return str(snapshot)

    hub.snapshot_download = snapshot_download
    package = ModuleType("omnivoice")
    package.__path__ = []
    models = ModuleType("omnivoice.models")
    models.__path__ = []
    upstream = ModuleType("omnivoice.models.omnivoice")
    upstream.__file__ = str(source)
    upstream.OmniVoice = FakeOmniVoice
    upstream.OmniVoiceGenerationConfig = FakeConfig
    for name, value in {"torch": torch, "huggingface_hub": hub, "omnivoice": package,
                        "omnivoice.models": models, "omnivoice.models.omnivoice": upstream}.items():
        monkeypatch.setitem(sys.modules, name, value)
    return module, snapshot, model, events, torch, source


@pytest.mark.parametrize("from_cache", [False, True])
def test_real_loader_cpu_stages_mps_keeps_codec_cpu_offline(monkeypatch, tmp_path, from_cache):
    import hashlib
    import random

    module, snapshot, model, events, _, source = fake_loader_packages(monkeypatch, tmp_path)
    checkpoint = "k2-fsa/OmniVoice" if from_cache else str(snapshot)
    assert callable(getattr(module, "load_upstream", None)), "real offline loader is missing"
    session = module.load_upstream(checkpoint)
    assert session.model is model
    assert ("from_pretrained", str(snapshot), {"device_map": "cpu", "dtype": "float16", "load_asr": False,
                                              "local_files_only": True}) in events
    assert events[-3:] == [("model.to", "mps"), ("codec.to", "cpu"), "eval"]
    resolutions = [e for e in events if isinstance(e, tuple) and e[0] == "resolve"]
    assert len(resolutions) == int(from_cache)
    session.seed(123)
    draw = random.random(), np.random.random()
    session.seed(123)
    assert draw == (random.random(), np.random.random())
    assert events[-1] == ("torch.seed", 123)
    session.synchronize()
    assert events[-1] == "sync"
    facts = session.provenance
    assert facts["checkpoint_revision"] == "a" * 40
    assert facts["checkpoint_source"] == ("hub-cache" if from_cache else "local")
    assert facts["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert facts["dtype"] == "float16" and facts["device"] == "mps"
    assert facts["load_device"] == facts["codec_device"] == "cpu"
    assert facts["offline"] is True
    assert all(f"{name}_version" in facts for name in ("python", "omnivoice", "torch", "torchaudio", "transformers", "numpy", "huggingface_hub"))
    assert len(facts["runtime_source_sha256"]) == 64
    assert "generation_audio_chunk_threshold" in facts


@pytest.mark.parametrize("case", ["missing-local", "missing-codec", "no-mps", "url"])
def test_real_loader_prerequisites_fail_without_model_or_download(monkeypatch, tmp_path, case):
    module, snapshot, _, events, torch, _ = fake_loader_packages(monkeypatch, tmp_path)
    checkpoint = str(snapshot)
    if case == "missing-local":
        checkpoint = str(tmp_path / "missing")
    elif case == "missing-codec":
        (snapshot / "audio_tokenizer").rmdir()
    elif case == "no-mps":
        torch.backends.mps.is_available = lambda: False
    elif case == "url":
        checkpoint = "https://private.example/checkpoint"
    assert callable(getattr(module, "load_upstream", None)), "real offline loader is missing"
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        module.load_upstream(checkpoint)
    assert not any(isinstance(e, tuple) and e[0] in {"from_pretrained", "resolve"} for e in events)


def test_default_runtime_uses_real_loader_and_unknown_revision_is_null(monkeypatch, tmp_path):
    module, snapshot, model, events, _, _ = fake_loader_packages(monkeypatch, tmp_path)
    local = tmp_path / "local-model"
    snapshot.rename(local)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"synthetic")
    runtime = module.OmniVoiceRuntime(reference, "explicit", checkpoint=str(local))
    assert events == []
    result = runtime.generate(["test"], language="en", seed=1)
    assert runtime.session.model is model
    assert result.provenance["checkpoint_revision"] is None
    assert result.provenance["generation_audio_chunk_threshold"] == 30.0


def test_loader_failure_releases_lock_without_hidden_retry(tmp_path):
    module, reference, _, good_loader, _, _ = make_upstream(tmp_path)
    calls = []

    def loader(checkpoint):
        calls.append(checkpoint)
        if len(calls) == 1:
            raise RuntimeError("synthetic loader failure")
        return good_loader(checkpoint)

    runtime = module.OmniVoiceRuntime(reference, "explicit", loader=loader)
    with pytest.raises(RuntimeError):
        runtime.generate(["test"], language="en", seed=1)
    assert len(calls) == 1
    assert len(runtime.generate(["test"], language="en", seed=1).pcm) == 1
    assert len(calls) == 2


def test_indexed_mps_device_provenance_is_retained(tmp_path):
    module, reference, _, loader, _, _ = make_upstream(tmp_path)
    session = loader("fixture")
    session.provenance["device"] = "mps:0"
    runtime = module.OmniVoiceRuntime(reference, "explicit", loader=lambda _: session)
    assert runtime.generate(["test"], language="en", seed=1).provenance["device"] == "mps:0"


@pytest.mark.parametrize("optimization", ["auto", "SPLIT-CFG", "", None, 1, []])
def test_invalid_optimization_rejected_before_any_loader(tmp_path, optimization):
    module, reference, _, loader, calls, _ = make_upstream(tmp_path)
    with pytest.raises(ValueError, match="optimization"):
        module.OmniVoiceRuntime(reference, "explicit", loader=loader, optimization=optimization)
    with pytest.raises(ValueError, match="optimization"):
        module.load_upstream("/synthetic/missing", optimization=optimization)
    assert calls == []


def test_split_runtime_routes_explicit_mode_once_and_retains_order_provenance(tmp_path):
    module, reference, model, original_loader, _, _ = make_upstream(tmp_path)
    calls = []

    def loader(checkpoint, *, optimization):
        calls.append((checkpoint, optimization))
        session = original_loader(checkpoint)
        session.provenance.update(optimization=optimization, optimization_source_sha256="b" * 64)
        return session

    runtime = module.OmniVoiceRuntime(reference, "explicit", loader=loader, optimization="split-cfg")
    first = runtime.generate(["Second", "First", "Third"], language="en", seed=1234)
    second = runtime.generate(["Next"], language="vi", seed=5678)
    assert calls == [(module.DEFAULT_CHECKPOINT, "split-cfg")]
    assert len(model.prompt_calls) == 1
    assert [call["text"] for call in model.generate_calls] == [["Second", "First", "Third"], ["Next"]]
    assert first.pcm == second.pcm * 3
    assert first.provenance["optimization"] == "split-cfg"
    assert first.provenance["optimization_source_sha256"] == "b" * 64
    assert first.provenance["generation_num_step"] == 32
    assert first.provenance["generation_guidance_scale"] == 2.0
    assert second.timings["load_seconds"] == second.timings["prompt_seconds"] == 0


@pytest.mark.parametrize("optimization", ["none", "split-cfg"])
def test_loader_selects_original_or_explicit_subclass_without_mutation(monkeypatch, tmp_path, optimization):
    import hashlib
    import sys
    from pathlib import Path

    module, snapshot, model, events, _, source = fake_loader_packages(monkeypatch, tmp_path)
    assert importlib.util.find_spec("pipeline.omnivoice_split_cfg") is not None, "split CFG module is missing"
    split = importlib.import_module("pipeline.omnivoice_split_cfg")
    upstream = sys.modules["omnivoice.models.omnivoice"]
    original = upstream.OmniVoice
    before = dict(vars(original))
    classes = []
    original_load = original.from_pretrained.__func__

    def load(cls, *args, **kwargs):
        classes.append(cls)
        return original_load(cls, *args, **kwargs)

    monkeypatch.setattr(original, "from_pretrained", classmethod(load))
    if optimization == "split-cfg":
        # Only the source-fingerprint boundary is substituted for this fake package.
        monkeypatch.setattr(split, "SUPPORTED_UPSTREAM_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    session = module.load_upstream(str(snapshot), optimization=optimization)
    assert session.model is model and len(classes) == 1
    assert upstream.OmniVoice is original
    assert all(vars(original)[key] is value for key, value in before.items() if key != "from_pretrained")
    if optimization == "none":
        assert classes == [original]
        assert session.provenance["optimization_source_sha256"] is None
    else:
        candidate = classes[0]
        assert candidate is not original and issubclass(candidate, original)
        assert candidate._generate_iterative is split._generate_iterative
        assert set(vars(candidate)) <= {"__module__", "__doc__", "_generate_iterative", "__firstlineno__", "__static_attributes__"}
        assert session.provenance["optimization_source_sha256"] == hashlib.sha256(Path(split.__file__).read_bytes()).hexdigest()
    assert session.provenance["optimization"] == optimization
    assert session.provenance["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert events[-3:] == [("model.to", "mps"), ("codec.to", "cpu"), "eval"]


@pytest.mark.parametrize("case", ["mismatch", "missing", "unavailable"])
def test_split_source_guard_fails_closed_before_model_load(monkeypatch, tmp_path, case):
    import sys

    module, snapshot, _, events, _, source = fake_loader_packages(monkeypatch, tmp_path)
    if case == "missing":
        source.unlink()
    elif case == "unavailable":
        sys.modules["omnivoice.models.omnivoice"].__file__ = None
    with pytest.raises(RuntimeError, match="split-cfg.*source") as caught:
        module.load_upstream(str(snapshot), optimization="split-cfg")
    assert str(tmp_path) not in str(caught.value)
    assert caught.value.__suppress_context__
    assert events == []


@pytest.mark.parametrize("value", ["private-mode", "SPLIT-CFG", None, ["split-cfg"]])
def test_optimization_provenance_drops_unknown_enum_and_malformed_hash(value):
    module = runtime_module()
    assert module.sanitize_provenance({"optimization": value, "optimization_source_sha256": "/private/hash"}) == {}


@pytest.mark.parametrize("value", ["none", "split-cfg"])
def test_optimization_provenance_retains_only_approved_enum_and_hash(value):
    module = runtime_module()
    expected = {"optimization": value, "optimization_source_sha256": "f" * 64}
    assert module.sanitize_provenance(expected) == expected
