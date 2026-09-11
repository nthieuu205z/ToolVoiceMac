"""Exact rotary arithmetic and model-scoped activation; no checkpoint loads."""
from types import SimpleNamespace
import pytest


def torch_mps():
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip('MPS required')
    return torch


def operands(batch, length, cos_batch=1, offset=0):
    torch = torch_mps()
    def tensor(heads):
        raw = torch.arange(offset + batch * length * heads * 128, dtype=torch.float32).sin()
        return raw[offset:].to(device='mps', dtype=torch.float16).reshape(batch, length, heads, 128).transpose(1, 2)
    q, k = tensor(16), tensor(8)
    phase = torch.arange(cos_batch * length * 128, dtype=torch.float32).reshape(cos_batch, length, 128)
    return q, k, phase.cos().to(device='mps', dtype=torch.float16), phase.sin().to(device='mps', dtype=torch.float16)


@pytest.mark.parametrize('batch,cos_batch', [(1,1), (2,1), (2,2)])
@pytest.mark.parametrize('length', [7,65,366,757,1030])
def test_native_rope_preserves_bits_layout_inputs_and_rng(batch, cos_batch, length):
    torch = torch_mps()
    from pipeline.omnivoice_mps_rope import native_rotary, _supported
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    args = operands(batch, length, cos_batch)
    originals = [x.clone() for x in args]
    with torch.inference_mode():
        assert _supported(*args, 1)
        expected = apply_rotary_pos_emb(*args)
        before = torch.mps.get_rng_state().clone()
        actual = native_rotary(*args)
    assert torch.equal(before, torch.mps.get_rng_state())
    for got, want in zip(actual, expected):
        assert got.stride() == want.stride()
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert all(torch.equal(a, b) for a, b in zip(args, originals))


def test_all_finite_half_patterns_keep_intermediate_rounding():
    import numpy as np
    torch = torch_mps()
    from pipeline.omnivoice_mps_rope import native_rotary
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    bits = np.arange(65536, dtype=np.uint16).view(np.float16).copy()
    bits[~np.isfinite(bits)] = 0
    raw = torch.from_numpy(bits).reshape(1,32,16,128).to('mps')
    q, k = raw.transpose(1,2), raw[:,:,:8,:].contiguous().transpose(1,2)
    phase = torch.linspace(-3.14,3.14,32*128,device='mps').reshape(1,32,128)
    for cos, sin in [(phase.cos().half(),phase.sin().half()), (torch.zeros_like(phase).half(),torch.ones_like(phase).half())]:
        actual, expected = native_rotary(q,k,cos,sin), apply_rotary_pos_emb(q,k,cos,sin)
        assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(actual,expected))


def test_nonzero_storage_offsets_are_respected():
    torch = torch_mps()
    from pipeline.omnivoice_mps_rope import native_rotary
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    q,k,c,s = operands(2,7)
    def offset(t):
        # Preserve dense B,T,H,D strides while starting after sentinel values.
        dense=t.transpose(1,2).contiguous() if t.ndim==4 else t.contiguous()
        base=torch.full((dense.numel()+17,),23.0,device='mps',dtype=t.dtype)
        view=base[17:].view(dense.shape);view.copy_(dense)
        return view.transpose(1,2) if t.ndim==4 else view
    args=tuple(offset(t) for t in (q,k,c,s))
    assert all(t.storage_offset()==17 for t in args)
    expected=apply_rotary_pos_emb(*args);actual=native_rotary(*args)
    assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(actual,expected))


@pytest.mark.parametrize('kind',['cpu','float32','contiguous','batch3','unsqueeze2','grad','length1'])
def test_unsupported_paths_keep_upstream_behavior(kind):
    torch = torch_mps()
    from pipeline.omnivoice_mps_rope import native_rotary, _supported
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    args = operands(3 if kind=='batch3' else 1,1 if kind=='length1' else 7)
    axis = 1
    if kind=='cpu': args=tuple(x.cpu() for x in args)
    if kind=='float32': args=tuple(x.float() for x in args)
    if kind=='contiguous': args=tuple(x.contiguous() for x in args)
    if kind=='unsqueeze2': args=(args[0].transpose(1,2),args[1].transpose(1,2),*args[2:]);axis=2
    if kind=='grad': args=(args[0].detach().requires_grad_(),*args[1:])
    assert not _supported(*args,axis)
    actual=native_rotary(*args,unsqueeze_dim=axis)
    expected=apply_rotary_pos_emb(*args,unsqueeze_dim=axis)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert all(a.stride()==b.stride() for a,b in zip(actual,expected))
    if kind=='grad':
        got=torch.autograd.grad(actual[0].float().sum(),args[0])[0]
        want=torch.autograd.grad(expected[0].float().sum(),args[0])[0]
        assert torch.equal(got,want)


def small_model():
    torch=torch_mps()
    from transformers import Qwen3Config,Qwen3Model
    from pipeline.omnivoice_mps_attention import enable_native_mps_gqa
    from pipeline.omnivoice_mps_norm import enable_native_rms_norm
    cfg=Qwen3Config(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=16,num_key_value_heads=8,head_dim=128,vocab_size=16,attn_implementation='sdpa')
    model=SimpleNamespace(llm=Qwen3Model(cfg).to(device='mps',dtype=torch.float16).eval())
    enable_native_rms_norm(model);enable_native_mps_gqa(model)
    return model


def test_activation_is_model_scoped_and_preserves_backbone_outputs_and_weights():
    torch=torch_mps()
    from transformers.models.qwen3 import modeling_qwen3 as upstream
    from pipeline.omnivoice_mps_rope import enable_native_mps_rope
    model=small_model();untouched=small_model()
    original=upstream.apply_rotary_pos_emb;forward=upstream.Qwen3Attention.forward
    params=dict(model.llm.named_parameters());keys=set(model.llm.state_dict())
    x=torch.arange(2*7*32,dtype=torch.float32).sin().reshape(2,7,32).to(device='mps',dtype=torch.float16)
    mask=torch.ones(2,1,7,7,device='mps',dtype=torch.bool);mask[1,:,-2:,:]=False;mask[1,:,:,-2:]=False
    with torch.inference_mode():
        expected=model.llm(inputs_embeds=x,attention_mask=mask,use_cache=False).last_hidden_state
        rng=torch.mps.get_rng_state().clone()
        assert enable_native_mps_rope(model)==2
        assert torch.equal(rng,torch.mps.get_rng_state())
        actual=model.llm(inputs_embeds=x,attention_mask=mask,use_cache=False).last_hidden_state
    assert torch.equal(actual.view(torch.int16),expected.view(torch.int16))
    assert set(model.llm.state_dict())==keys
    assert all(params[k] is v for k,v in model.llm.named_parameters())
    assert upstream.apply_rotary_pos_emb is original and upstream.Qwen3Attention.forward is forward
    assert all(layer.self_attn.forward.__func__ is forward for layer in untouched.llm.layers)


@pytest.mark.parametrize('failure',['last_layer','source','preflight'])
def test_activation_failures_leave_every_forward_unchanged(monkeypatch,failure):
    from pipeline import omnivoice_mps_rope as rope
    model=small_model();before=[layer.self_attn.forward.__func__ for layer in model.llm.layers]
    if failure=='last_layer':model.llm.layers[-1].self_attn.to('cpu')
    if failure=='source':monkeypatch.setattr(rope,'_FORWARD_SHA256','invalid')
    if failure=='preflight':
        def fail():raise RuntimeError('Metal unavailable')
        monkeypatch.setattr(rope,'_check_native_rope',fail)
    with pytest.raises(RuntimeError):rope.enable_native_mps_rope(model)
    assert [layer.self_attn.forward.__func__ for layer in model.llm.layers]==before
