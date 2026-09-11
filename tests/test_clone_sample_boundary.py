import numpy as np
import pytest
from fastapi import HTTPException
from backend.routes import voices
from pipeline import custom_voices
from pipeline.audio import read_wav

RATE = 24000

def speech(seconds):
    t=np.arange(round(seconds*RATE))/RATE
    return (5000*np.sin(2*np.pi*180*t)).astype('<i2')

def test_long_reference_ends_in_pause_instead_of_cut_word(tmp_path,monkeypatch):
    monkeypatch.setattr(custom_voices, "_dir", tmp_path)
    samples=np.concatenate([speech(9.4),np.zeros(round(.4*RATE),dtype='<i2'),speech(4)])
    monkeypatch.setattr(voices,'decode_to_pcm',lambda raw,rate:samples.tobytes())
    voices._normalize_sample(b'fixture','clone-boundary')
    actual,rate=read_wav(custom_voices.sample_path('clone-boundary'))
    assert 9.4 < len(actual)/rate < 9.8
    assert np.array_equal(actual,samples[:len(actual)])
    assert np.max(np.abs(actual[-240:])) == 0

def test_continuous_long_reference_is_rejected_instead_of_cutting_mid_word(tmp_path,monkeypatch):
    monkeypatch.setattr(custom_voices, "_dir", tmp_path)
    monkeypatch.setattr(voices,'decode_to_pcm',lambda raw,rate:speech(15).tobytes())
    with pytest.raises(HTTPException) as exc:
        voices._normalize_sample(b'fixture','clone-continuous')
    assert exc.value.status_code == 400
    assert not custom_voices.sample_path('clone-continuous').exists()

def test_short_reference_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(custom_voices, "_dir", tmp_path)
    samples=speech(5)
    monkeypatch.setattr(voices,'decode_to_pcm',lambda raw,rate:samples.tobytes())
    voices._normalize_sample(b'fixture','clone-short')
    actual,_=read_wav(custom_voices.sample_path('clone-short'))
    assert np.array_equal(actual,samples)


def test_long_reference_with_quiet_cap_preserves_twelve_seconds(tmp_path,monkeypatch):
    monkeypatch.setattr(custom_voices, '_dir', tmp_path)
    samples=np.concatenate([speech(10),np.zeros(3*RATE,dtype='<i2')])
    monkeypatch.setattr(voices,'decode_to_pcm',lambda raw,rate:samples.tobytes())
    voices._normalize_sample(b'fixture','clone-tail')
    actual,_=read_wav(custom_voices.sample_path('clone-tail'))
    assert len(actual)==12*RATE


def test_brief_dip_in_voicing_is_not_treated_as_safe_pause(tmp_path,monkeypatch):
    monkeypatch.setattr(custom_voices, '_dir', tmp_path)
    samples=np.concatenate([speech(9),np.zeros(round(.08*RATE),dtype='<i2'),speech(4)])
    monkeypatch.setattr(voices,'decode_to_pcm',lambda raw,rate:samples.tobytes())
    with pytest.raises(HTTPException):
        voices._normalize_sample(b'fixture','clone-brief')
