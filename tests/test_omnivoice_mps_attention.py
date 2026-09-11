"""Native MPS attention contracts using small real tensors, never checkpoints."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("length", [7, 65, 1030])
@pytest.mark.parametrize("mask_kind", ["bool", "additive", "padded_diagonal"])
def test_native_gqa_matches_repeated_kv_and_preserves_rng(length, mask_kind):
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from pipeline.omnivoice_mps_attention import mps_gqa_attention_forward

    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    torch.manual_seed(952)
    # Real Qwen3 head width and grouping, with non-contiguous input layouts.
    q = torch.randn(1, length, 4, 128, device="mps", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(1, length, 2, 128, device="mps", dtype=torch.float16).transpose(1, 2)
    v = torch.randn_like(k)
    mask = torch.ones(1, 1, length, length, device="mps", dtype=torch.bool)
    mask[..., -2:] = False
    mask[..., -2:, :] = False
    if mask_kind == "padded_diagonal":
        mask[..., -2, -2] = True
        mask[..., -1, -1] = True
    if mask_kind == "additive":
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, -float("inf"))
    module = SimpleNamespace(num_key_value_groups=2, is_causal=False, training=False)
    expected, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=128 ** -0.5)
    rng = torch.mps.get_rng_state().clone()
    actual, weights = mps_gqa_attention_forward(module, q, k, v, mask, scaling=128 ** -0.5)
    assert weights is None and actual.shape == (1, length, 4, 128)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
    assert torch.equal(torch.mps.get_rng_state(), rng)


def test_cpu_and_asymmetric_attention_keep_upstream_behavior():
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from pipeline.omnivoice_mps_attention import mps_gqa_attention_forward

    module = SimpleNamespace(num_key_value_groups=2, is_causal=True, training=False)
    q, k, v = torch.randn(1, 4, 1, 8), torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
    expected, _ = sdpa_attention_forward(module, q, k, v, None)
    actual, _ = mps_gqa_attention_forward(module, q, k, v, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_activation_rejects_wrong_backbone_without_changing_configuration():
    from pipeline.omnivoice_mps_attention import enable_native_mps_gqa

    config = SimpleNamespace(model_type="unsupported", _attn_implementation="sdpa")
    with pytest.raises(RuntimeError, match="Qwen3"):
        enable_native_mps_gqa(SimpleNamespace(llm=SimpleNamespace(config=config)))
    assert config._attn_implementation == "sdpa"


def make_small_backbone():
    import torch
    from transformers import Qwen3Config, Qwen3Model

    if not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                        vocab_size=16, attn_implementation="sdpa")
    return SimpleNamespace(llm=Qwen3Model(config).to(device="mps", dtype=torch.float16).eval())


def test_activation_preserves_full_backbone_outputs_weights_and_upstream_backend():
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from pipeline.omnivoice_mps_attention import enable_native_mps_gqa

    model = make_small_backbone()
    original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    parameters = dict(model.llm.named_parameters())
    keys = set(model.llm.state_dict())
    x = torch.randn(2, 7, 32, device="mps", dtype=torch.float16)
    mask = torch.ones(2, 1, 7, 7, device="mps", dtype=torch.bool)
    mask[1, :, :, -2:] = False
    mask[1, :, -2:, :] = False
    with torch.inference_mode():
        expected = model.llm(inputs_embeds=x, attention_mask=mask).last_hidden_state
        rng = torch.mps.get_rng_state().clone()
        assert enable_native_mps_gqa(model) == 2
        assert torch.equal(torch.mps.get_rng_state(), rng)
        actual = model.llm(inputs_embeds=x, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
    assert set(model.llm.state_dict()) == keys
    assert all(p is parameters[name] for name, p in model.llm.named_parameters())
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original_sdpa


def test_activation_checks_every_layer_before_selecting_backend():
    from pipeline.omnivoice_mps_attention import enable_native_mps_gqa

    model = make_small_backbone()
    model.llm.layers[-1].self_attn.to("cpu")
    with pytest.raises(RuntimeError, match="MPS"):
        enable_native_mps_gqa(model)
    assert model.llm.config._attn_implementation == "sdpa"
