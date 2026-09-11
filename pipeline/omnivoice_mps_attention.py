"""Opt-in native grouped attention for OmniVoice's MPS Qwen3 backbone.

Keep the upstream Qwen3 forward, weights, rotary embeddings and masks. Register
a separate attention implementation, selected only by this model at startup.
"""
from __future__ import annotations

_BACKEND = "omnivoice_mps_gqa"


def mps_gqa_attention_forward(
    module, query, key, value, attention_mask, dropout=0.0, scaling=None,
    is_causal=None, position_bias=None, **kwargs,
):
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    # Only the square, inference-time self-attention used by this TTS model.
    # Delegate decoding, training and positional-bias semantics to upstream.
    if (query.device.type != "mps" or query.dtype != torch.float16
            or query.shape[2] <= 1 or query.shape[2] != key.shape[2]
            or query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]
            or getattr(module, "training", False) or dropout != 0.0
            or position_bias is not None or kwargs.get("output_attentions", False)):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout,
            scaling=scaling, is_causal=is_causal, position_bias=position_bias, **kwargs,
        )
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    causal = query.shape[2] > 1 and attention_mask is None and causal
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0,
        scale=scaling, is_causal=causal, enable_gqa=True,
    )
    return output.transpose(1, 2).contiguous(), None


def _check_native_gqa() -> None:
    """Check this installed MPS build without advancing the generation RNG."""
    import torch
    import torch.nn.functional as functional

    q = (torch.arange(16 * 7 * 128, dtype=torch.float32).reshape(1, 16, 7, 128) / 1000).sin()
    k = q[:, ::2].contiguous().to(device="mps", dtype=torch.float16)
    q = q.to(device="mps", dtype=torch.float16)
    v = k.flip(-1)
    mask = torch.ones(1, 1, 7, 7, device="mps", dtype=torch.bool)
    mask[..., -1] = False
    expected = functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1), attn_mask=mask,
    )
    actual = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    if not torch.allclose(actual, expected, rtol=0.002, atol=0.002):
        raise RuntimeError("Native MPS grouped attention failed its numerical check")


def enable_native_mps_gqa(model) -> int:
    """Validate before selecting native GQA; never change other models' SDPA."""
    import torch
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    llm = getattr(model, "llm", None)
    config = getattr(llm, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise RuntimeError("Native MPS grouped attention requires the supported Qwen3 backbone")
    if config._attn_implementation not in ("sdpa", _BACKEND):
        raise RuntimeError("Native MPS grouped attention requires the SDPA implementation")
    layers = [m for m in llm.modules() if type(m) is Qwen3Attention]
    if len(layers) != config.num_hidden_layers or not layers:
        raise RuntimeError("Native MPS grouped attention encountered an unsupported layer layout")
    for layer in layers:
        if (layer.training or layer.head_dim != 128 or layer.num_key_value_groups != 2
                or layer.sliding_window is not None
                or any(p.device.type != "mps" or p.dtype != torch.float16 for p in layer.parameters())):
            raise RuntimeError("Native MPS grouped attention requires inference on supported MPS float16 layers")
    _check_native_gqa()
    AttentionInterface.register(_BACKEND, mps_gqa_attention_forward)
    AttentionMaskInterface.register(_BACKEND, sdpa_mask)
    llm.set_attn_implementation(_BACKEND)
    return len(layers)
