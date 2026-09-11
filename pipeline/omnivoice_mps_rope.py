"""Opt-in Metal rotary embedding for OmniVoice's supported Qwen3 backbone.

Keep FP16 multiplication boundaries before the final addition. The kernel only
accepts the dense B,T,H,D storage used by this model; other layouts and autograd
use Transformers. No torch import or shader compilation occurs at module import.
"""
from __future__ import annotations

from functools import update_wrapper
from types import FunctionType, MethodType

# Transformers 5.16.1: models/qwen3/modeling_qwen3.py::Qwen3Attention.forward.
# Updating this fingerprint requires renewed tensor/token/PCM parity checks.
_FORWARD_SHA256 = "6c8fb032ff3742438fb80dab4427c11ef04e12fdc02f08abd11ff5adaa0fd1c3"
_SHADER = None
_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
kernel void rope_pair(
    device const half* q, device const half* k,
    device const half* c, device const half* s,
    device half* oq, device half* ok,
    constant long& nq, constant long& nk,
    constant long& seq, constant long& cos_batch,
    uint tid [[thread_position_in_grid]]) {
    bool query = tid < uint(nq);
    uint i = query ? tid : tid - uint(nq);
    if (i >= (query ? uint(nq) : uint(nk))) return;
    uint dim = i & 127u;
    uint token = i >> (query ? 11u : 10u);
    // At most two items. A shared cosine table needs one subtraction,
    // not division/modulo for every scalar element.
    if (cos_batch == 1 && token >= uint(seq)) token -= uint(seq);
    uint ci = token * 128u + dim;
    uint j = dim < 64u ? i + 64u : i - 64u;
    float x = float(query ? q[i] : k[i]);
    float rotated = float(query ? q[j] : k[j]);
    if (dim < 64u) rotated = -rotated;
    // Materialize both half products: contracting or retaining FP32 products
    // would change upstream rounding and can change diffusion token choices.
    volatile half a = half(x * float(c[ci]));
    volatile half b = half(rotated * float(s[ci]));
    half result = half(float(a) + float(b));
    if (query) oq[i] = result; else ok[i] = result;
}
"""


def _shader():
    global _SHADER
    if _SHADER is None:
        import torch

        _SHADER = torch.mps.compile_shader(_SOURCE)
    return _SHADER


def _supported(q, k, cos, sin, unsqueeze_dim):
    import torch

    tensors = (q, k, cos, sin)
    if (unsqueeze_dim != 1 or q.ndim != 4 or k.ndim != 4
            or cos.ndim != 3 or sin.ndim != 3
            or q.device.type != "mps"
            or any(t.device != q.device or t.dtype != torch.float16 for t in tensors)
            or (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))):
        return False
    batch, heads, length, dim = q.shape
    return (
        batch in (1, 2) and heads == 16 and dim == 128 and length > 1
        and k.shape == (batch, 8, length, 128)
        and q.stride() == (length * 2048, 128, 2048, 1)
        and k.stride() == (length * 1024, 128, 1024, 1)
        and cos.shape == sin.shape and cos.shape[0] in (1, batch)
        and cos.shape[1:] == (length, 128)
        and cos.is_contiguous() and sin.is_contiguous()
        and q.numel() + k.numel() < 2**32
    )


def native_rotary(q, k, cos, sin, unsqueeze_dim=1):
    """Apply paired rotary embeddings, preserving upstream layout and RNG."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    if not _supported(q, k, cos, sin, unsqueeze_dim):
        return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    _shader().rope_pair(
        q, k, cos, sin, oq, ok, q.numel(), k.numel(), q.shape[2], cos.shape[0],
        threads=q.numel() + k.numel(),
    )
    return oq, ok


def _check_native_rope():
    """Compile/check this Metal build before changing any layer; consume no RNG."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    with torch.inference_mode():
        q = torch.arange(2 * 7 * 16 * 128, dtype=torch.float32).sin()
        q = q.reshape(2, 7, 16, 128).to(device="mps", dtype=torch.float16)
        k = q[:, :, :8, :].contiguous().transpose(1, 2)
        q = q.transpose(1, 2)
        phase = torch.arange(7 * 128, dtype=torch.float32).reshape(1, 7, 128)
        cos = phase.cos().to(device="mps", dtype=torch.float16)
        sin = phase.sin().to(device="mps", dtype=torch.float16)
        expected = apply_rotary_pos_emb(q, k, cos, sin)
        actual = native_rotary(q, k, cos, sin)
        if not all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                   for a, b in zip(actual, expected)):
            raise RuntimeError("Native MPS rotary failed its exact numerical check")


def enable_native_mps_rope(model):
    """Bind a private rotary function to each supported attention instance.

    Use the upstream forward bytecode with a private globals dictionary instead
    of modifying the Transformers module globally or maintaining a second copy
    of its attention logic. Source fingerprint and closure checks guard this
    narrow seam. All validation and shader preflight precede layer mutation.
    """
    import hashlib
    import inspect
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    llm = getattr(model, "llm", None)
    config = getattr(llm, "config", None)
    if (getattr(config, "model_type", None) != "qwen3"
            or getattr(config, "_attn_implementation", None) != "omnivoice_mps_gqa"
            or config.num_attention_heads != 16 or config.num_key_value_heads != 8):
        raise RuntimeError("Native MPS rotary requires the supported Qwen3 GQA backbone")
    original = Qwen3Attention.forward
    if (original.__closure__ is not None
            or "apply_rotary_pos_emb" not in original.__code__.co_names
            or hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != _FORWARD_SHA256):
        raise RuntimeError("Native MPS rotary requires the verified Qwen3 forward")
    layers = [m for m in llm.modules() if type(m) is Qwen3Attention]
    if not layers or len(layers) != config.num_hidden_layers:
        raise RuntimeError("Native MPS rotary encountered an unsupported layer layout")
    for layer in layers:
        if (layer.training or layer.head_dim != 128 or layer.num_key_value_groups != 2
                or layer.sliding_window is not None
                or getattr(layer.forward, "__func__", None) is not original
                or any(p.device.type != "mps" or p.dtype != torch.float16 for p in layer.parameters())):
            raise RuntimeError("Native MPS rotary requires unmodified MPS FP16 inference layers")
    _check_native_rope()
    namespace = dict(original.__globals__, apply_rotary_pos_emb=native_rotary)
    forward = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    forward.__kwdefaults__ = original.__kwdefaults__
    update_wrapper(forward, original)
    for layer in layers:
        layer.forward = MethodType(forward, layer)
    return len(layers)
