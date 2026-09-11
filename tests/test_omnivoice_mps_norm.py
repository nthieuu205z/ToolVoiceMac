"""Small real MPS norm checks; no checkpoints, voice files, or downloads."""
from types import SimpleNamespace

import pytest


def make_model():
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    if not torch.backends.mps.is_available():
        pytest.skip('requires Apple MPS for the native normalization contract')
    llm = torch.nn.ModuleList([Qwen3RMSNorm(128, eps=1e-6) for _ in range(5)])
    llm.config = SimpleNamespace(model_type='qwen3', num_hidden_layers=1)
    return SimpleNamespace(llm=llm.to(device='mps', dtype=torch.float16))


def test_native_norm_preserves_weights_and_normalization_for_strided_inputs():
    import torch
    from pipeline.omnivoice_mps_norm import enable_native_rms_norm
    model = make_model()
    weights = [m.weight for m in model.llm]
    torch.manual_seed(741)
    x = torch.randn(2, 3, 7, 128, device='mps', dtype=torch.float16).transpose(1, 2)
    expected = model.llm[0](x)
    assert enable_native_rms_norm(model) == 5
    actual = model.llm[0](x)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
    assert all(m.weight is w for m, w in zip(model.llm, weights))
    assert enable_native_rms_norm(model) == 5
    assert set(model.llm.state_dict()) == {f'{i}.weight' for i in range(5)}


def test_native_norm_validates_every_module_before_changing_any():
    from pipeline.omnivoice_mps_norm import enable_native_rms_norm
    model = make_model()
    before = [m.forward for m in model.llm]
    model.llm[-1].to('cpu')
    with pytest.raises(RuntimeError, match='MPS'):
        enable_native_rms_norm(model)
    assert [m.forward for m in model.llm] == before


def test_native_norm_requires_a_native_mps_kernel(monkeypatch):
    import torch
    from pipeline.omnivoice_mps_norm import enable_native_rms_norm
    model = make_model()
    before = [m.forward for m in model.llm]
    monkeypatch.setattr(torch._C, '_dispatch_has_kernel_for_dispatch_key', lambda *args: False)
    with pytest.raises(RuntimeError, match='kernel'):
        enable_native_rms_norm(model)
    assert [m.forward for m in model.llm] == before
