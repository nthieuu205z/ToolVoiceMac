from __future__ import annotations

import numpy as np

from pipeline import custom_voices
from pipeline.audio import write_wav
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.omnivoice_speech import OmniVoiceSynthesizer


def _make_clone(voice_id: str = "clone-generation-config") -> str:
    write_wav(
        custom_voices.sample_path(voice_id),
        np.zeros(2400, dtype="<i2"),
        TTS_SAMPLE_RATE,
    )
    custom_voices.register(voice_id, "Generation Config")
    return voice_id


class ConfigModel:
    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append({"text": list(text), "kwargs": kwargs})
        return [np.ones(2400, dtype=np.float32) for _ in text]


def test_synthesis_passes_explicit_generation_config_to_omnivoice():
    voice_id = _make_clone()
    model = ConfigModel()
    synthesizer = OmniVoiceSynthesizer(
        model=model,
        transcriber=lambda _sample: "mẫu tham chiếu",
        num_step=12,
        batch_size=2,
    )

    synthesizer.synthesize_batch(["một", "hai"], voice_id)

    kwargs = model.calls[0]["kwargs"]
    assert kwargs["generation_config"].num_step == 12
    assert kwargs["generation_config"].postprocess_output is True
    assert "ref_audio" in kwargs
    assert "ref_text" in kwargs


def test_synthesis_does_not_pass_legacy_generation_kwargs_when_config_is_used():
    voice_id = _make_clone("clone-generation-config-no-legacy")
    model = ConfigModel()
    synthesizer = OmniVoiceSynthesizer(
        model=model,
        transcriber=lambda _sample: "mẫu tham chiếu",
        num_step=16,
        batch_size=1,
    )

    synthesizer.synthesize("một", voice_id)

    kwargs = model.calls[0]["kwargs"]
    assert "num_step" not in kwargs
    assert "postprocess_output" not in kwargs
