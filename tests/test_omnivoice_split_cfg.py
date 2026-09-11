"""CPU-only synthetic differential gates; never import/load upstream models.

The optional oracle reads only allowlisted function ASTs from the fingerprinted
installed source. Small tensors prove shape/RNG contracts, not real performance.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
from importlib import metadata
import importlib.util
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

SUPPORTED_SOURCE = "631050ba94775b9c8a72ec3b2d84777c38317e2e502b1998d3128fedf88846ff"


def split_module():
    assert importlib.util.find_spec("pipeline.omnivoice_split_cfg") is not None, "split CFG module is missing"
    return importlib.import_module("pipeline.omnivoice_split_cfg")


def test_import_and_opt_in_constructor_are_lazy_without_heavy_dependencies(tmp_path):
    split_module()
    reference = tmp_path / "synthetic.wav"
    reference.write_bytes(b"not decoded")
    code = '''
import importlib.abc, sys
from pathlib import Path
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'torchaudio', 'omnivoice', 'transformers', 'backend', 'huggingface_hub'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Guard())
from pipeline import omnivoice_split_cfg
from pipeline.omnivoice_runtime import OmniVoiceRuntime
import run_omnivoice
runtime = OmniVoiceRuntime(Path(sys.argv[1]), 'explicit', optimization='split-cfg')
assert runtime.session is None and runtime.prompt is None
'''
    completed = subprocess.run([sys.executable, "-B", "-c", code, str(reference)],
                               cwd=Path(__file__).resolve().parents[1],
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                               text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def test_public_class_factory_checks_source_before_returning_subclass(tmp_path):
    split = split_module()
    source = tmp_path / "synthetic-source.py"
    source.write_text("# unsupported synthetic source\n")
    upstream = SimpleNamespace(__file__=str(source), OmniVoice=type("Original", (), {}))
    with pytest.raises(RuntimeError, match="split-cfg.*source") as caught:
        split.get_split_cfg_class(upstream)
    assert str(tmp_path) not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.fixture
def cpu_oracle(monkeypatch):
    """Only trusted function bodies are executed, not package/model initialization."""
    torch = pytest.importorskip("torch", reason="CPU tensor differential requires optional torch")
    try:
        distribution = metadata.distribution("omnivoice")
        source_file = distribution.locate_file("omnivoice/models/omnivoice.py")
        source = Path(source_file).read_bytes()
    except (metadata.PackageNotFoundError, OSError):
        pytest.skip("CPU differential requires readable installed OmniVoice source; no weights loaded")
    if hashlib.sha256(source).hexdigest() != SUPPORTED_SOURCE:
        pytest.skip("CPU differential oracle requires the supported upstream source fingerprint")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniVoice")
    functions.update({node.name: node for node in model.body if isinstance(node, ast.FunctionDef)})
    names = ("_gumbel_sample", "_get_time_steps", "_filter_top_k", "_predict_tokens_with_scoring", "_generate_iterative")
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(functions[name] for name in names)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {"torch": torch, "F": torch.nn.functional, "math": math, "logger": logging.getLogger(__name__)}
    exec(compile(module, "<fingerprinted-upstream-test-oracle>", "exec"), namespace)
    # Let the production override lazily import its real upstream helpers without
    # importing the heavyweight model package. The helpers above are unmodified.
    upstream = ModuleType("omnivoice.models.omnivoice")
    upstream._get_time_steps = namespace["_get_time_steps"]
    upstream._gumbel_sample = namespace["_gumbel_sample"]
    for name in ("omnivoice", "omnivoice.models"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "omnivoice.models.omnivoice", upstream)
    # Tiny tensor tests do not benefit from CPU thread-pool overhead.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch, namespace
    finally:
        torch.set_num_threads(threads)


def make_mock(torch, oracle, target_lens, prefixes, *, poison_padding=False):
    inputs = []
    for index, (target, prefix) in enumerate(zip(target_lens, prefixes)):
        length = prefix + target
        ids = (torch.arange(2 * length).reshape(1, 2, length) + index * 3) % 23
        ids[..., -target:] = 31
        audio = torch.ones((1, length), dtype=torch.bool)
        audio[:, :max(1, prefix // 2)] = False
        inputs.append({"input_ids": ids, "audio_mask": audio})

    class MockModel:
        device = "cpu"
        config = SimpleNamespace(num_audio_codebook=2, audio_mask_id=31)

        def __init__(self):
            self.calls, self.prepare_calls, self.scoring_calls = [], [], []

        def _prepare_inference_inputs(self, *args):
            self.prepare_calls.append(args)
            return inputs[int(args[0])]

        def __call__(self, **kwargs):
            self.calls.append({key: value.clone() for key, value in kwargs.items()})
            ids = kwargs["input_ids"].float()
            if poison_padding:
                padded = (ids == 31).all(dim=1) & ~kwargs["audio_mask"]
                ids = ids.masked_fill(padded[:, None, :], 10000)
            attention = kwargs["attention_mask"][:, 0].float()
            weights = attention / attention.sum(-1, keepdim=True).clamp_min(1)
            context = torch.einsum("bij,bcj->bci", weights, ids)
            vocab = torch.arange(32).view(1, 1, 1, 32)
            # Sequence-independent arithmetic means differences expose masks and
            # index/order mistakes, not real backend accumulation differences.
            logits = torch.sin(ids[..., None] * .13 + context[..., None] * .07 + vocab * .37)
            return SimpleNamespace(logits=logits.to(torch.float16))

        def _predict_tokens_with_scoring(self, conditional, unconditional, config):
            tokens, scores = oracle["_predict_tokens_with_scoring"](self, conditional, unconditional, config)
            self.scoring_calls.append((conditional.clone(), unconditional.clone(), tokens.clone(), scores.clone(), config))
            return tokens, scores

    size = len(target_lens)
    task = SimpleNamespace(batch_size=size, texts=list(map(str, range(size))), target_lens=list(target_lens),
                           ref_texts=[f"synthetic-{i}" for i in range(size)], ref_audio_tokens=[None] * size,
                           langs=["en"] * size, instructs=[f"instruction-{i}" for i in range(size)])
    return MockModel(), task, inputs


@pytest.mark.parametrize("seed", [1234, 5678])
@pytest.mark.parametrize("target_lens,prefixes", [
    ([3], [4]), ([3, 5], [8, 1]), ([2, 5, 3], [11, 1, 5]), ([3, 3, 3], [1, 9, 4]),
])
@pytest.mark.parametrize("options", [
    {},
    {"num_step": 3, "guidance_scale": 0.0, "position_temperature": 0.0,
     "class_temperature": 0.8, "layer_penalty_factor": 0.25, "t_shift": 0.7, "denoise": False},
    {"num_step": 4, "guidance_scale": 3.0, "class_temperature": 0.6},
])
def test_split_matches_original_tokens_rng_scoring_masks_and_shapes(cpu_oracle, seed, target_lens, prefixes, options):
    torch, oracle = cpu_oracle
    split = split_module()
    config = SimpleNamespace(num_step=32, guidance_scale=2.0, t_shift=0.1, denoise=True,
                             layer_penalty_factor=5.0, position_temperature=5.0, class_temperature=0.0)
    config.__dict__.update(options)
    config_before = vars(config).copy()
    baseline, task, inputs = make_mock(torch, oracle, target_lens, prefixes)
    candidate, _, _ = make_mock(torch, oracle, target_lens, prefixes)
    input_copies = [{key: value.clone() for key, value in item.items()} for item in inputs]
    torch.manual_seed(seed)
    expected = oracle["_generate_iterative"](baseline, task, config)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    actual = split._generate_iterative(candidate, task, config)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert len(actual) == len(expected) == len(target_lens)
    for index, (left, right) in enumerate(zip(expected, actual)):
        assert torch.equal(left, right), f"item order/tokens differ at {index}"
        assert tuple(right.shape) == (2, target_lens[index])
        assert not (right == 31).any()
    assert vars(config) == config_before
    assert candidate.prepare_calls == baseline.prepare_calls
    assert [args[-1] for args in candidate.prepare_calls] == [config.denoise] * len(target_lens)
    for item, copy in zip(inputs, input_copies):
        assert all(torch.equal(item[key], copy[key]) for key in item)
    assert len(candidate.scoring_calls) == len(baseline.scoring_calls) > 0
    for left, right in zip(baseline.scoring_calls, candidate.scoring_calls):
        assert right[-1] is left[-1] is config
        assert all(torch.equal(a, b) for a, b in zip(left[:-1], right[:-1]))
        assert right[0].dtype == right[1].dtype == right[3].dtype == torch.float32
    batch, conditional_length, target_length = len(target_lens), max(a + b for a, b in zip(target_lens, prefixes)), max(target_lens)
    assert target_length < conditional_length
    assert len(baseline.calls) == config.num_step
    assert len(candidate.calls) == config.num_step * 2
    for step, original in enumerate(baseline.calls):
        conditional, unconditional = candidate.calls[2 * step:2 * step + 2]
        assert tuple(conditional["input_ids"].shape) == (batch, 2, conditional_length)
        assert tuple(unconditional["input_ids"].shape) == (batch, 2, target_length)
        for key, value in original.items():
            expected_unconditional = value[batch:, ..., :target_length, :target_length] if key == "attention_mask" else value[batch:, ..., :target_length]
            assert torch.equal(conditional[key], value[:batch])
            assert torch.equal(unconditional[key], expected_unconditional)
        for index, length in enumerate(target_lens):
            mask = unconditional["attention_mask"][index, 0]
            assert mask[:length, :length].all()
            assert not mask[:length, length:].any() and not mask[length:, :length].any()
            assert torch.equal(mask[length:, length:], torch.eye(target_length - length, dtype=torch.bool))


@pytest.mark.parametrize("seed", [1234, 5678])
def test_split_padding_isolated_from_real_target_outputs(cpu_oracle, seed):
    torch, oracle = cpu_oracle
    split = split_module()
    config = SimpleNamespace(num_step=4, guidance_scale=2.0, t_shift=.1, denoise=True,
                             layer_penalty_factor=5.0, position_temperature=5.0, class_temperature=.8)
    regular, task, _ = make_mock(torch, oracle, [2, 5, 3], [11, 1, 5])
    poisoned, _, _ = make_mock(torch, oracle, [2, 5, 3], [11, 1, 5], poison_padding=True)
    torch.manual_seed(seed)
    expected = split._generate_iterative(regular, task, config)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    actual = split._generate_iterative(poisoned, task, config)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for left, right in zip(regular.scoring_calls, poisoned.scoring_calls):
        assert all(torch.equal(a, b) for a, b in zip(left[:-1], right[:-1]))


def test_split_only_promotes_target_logits_to_float32(cpu_oracle):
    """Prefix/padding logits must not inflate the float32 scoring workspace."""
    torch, oracle = cpu_oracle
    from torch.utils._python_dispatch import TorchDispatchMode

    promoted = []

    class TrackPromotions(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if (func == torch.ops.aten._to_copy.default
                    and isinstance(result, torch.Tensor)
                    and result.ndim == 4 and result.shape[-1] == 32
                    and result.dtype == torch.float32 and args[0].dtype == torch.float16):
                promoted.append(result.numel())
            return result

    model, task, _ = make_mock(torch, oracle, [3, 5], [18, 11])
    config = SimpleNamespace(num_step=2, guidance_scale=2.0, t_shift=.1, denoise=True,
                             layer_penalty_factor=5.0, position_temperature=5.0, class_temperature=0.0)
    with TrackPromotions():
        result = split_module()._generate_iterative(model, task, config)

    assert promoted, 'must observe actual float32 scoring conversions'
    # One item's real targets: 2 codebooks * at most 5 frames * 32 vocabulary.
    assert max(promoted) <= 320
    assert [tuple(tokens.shape) for tokens in result] == [(2, 3), (2, 5)]
