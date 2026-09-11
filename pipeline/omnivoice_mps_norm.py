"""Opt-in native MPS RMSNorm for the fingerprint-gated OmniVoice runtime.

Keep weights and checkpoint keys intact. Only the forward implementation changes;
this is inference-only and must be installed before publishing the model singleton.
"""
from __future__ import annotations

from types import MethodType


def _native_rms_norm_forward(self, hidden_states):
    import torch.nn.functional as functional

    return functional.rms_norm(
        hidden_states, (self.weight.numel(),), self.weight, self.variance_epsilon
    )


def enable_native_rms_norm(model) -> int:
    """Validate the entire Qwen3 norm set before installing native forwards."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    if not torch._C._dispatch_has_kernel_for_dispatch_key("aten::_fused_rms_norm", "MPS"):
        raise RuntimeError("Native MPS RMSNorm kernel is unavailable in this PyTorch build")
    llm = getattr(model, "llm", None)
    config = getattr(llm, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise RuntimeError("Native MPS RMSNorm requires the supported Qwen3 backbone")
    norms = [module for module in llm.modules() if type(module) is Qwen3RMSNorm]
    expected = 4 * config.num_hidden_layers + 1
    if len(norms) != expected:
        raise RuntimeError("Native MPS RMSNorm encountered an unsupported Qwen3 norm layout")
    for norm in norms:
        if norm.weight.device.type != "mps" or norm.weight.dtype != torch.float16:
            raise RuntimeError("Native MPS RMSNorm requires all norm weights on MPS in float16")
    for norm in norms:
        norm.forward = MethodType(_native_rms_norm_forward, norm)
    return len(norms)
