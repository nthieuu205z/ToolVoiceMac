from __future__ import annotations

import numpy as np

from pipeline import custom_voices
from pipeline.audio import write_wav
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.omnivoice_speech import OmniVoiceSynthesizer


def _make_clone(voice_id: str = "clone-prompt-cache") -> str:
    write_wav(
        custom_voices.sample_path(voice_id),
        np.zeros(2400, dtype="<i2"),
        TTS_SAMPLE_RATE,
    )
    custom_voices.register(voice_id, "Prompt Cache")
    return voice_id


class PromptModel:
    def __init__(self):
        self.prompt_calls = 0
        self.prompts = []
        self.generate_calls = []

    def create_voice_clone_prompt(self, ref_audio, *, ref_text, preprocess_prompt):
        self.prompt_calls += 1
        prompt = {"number": self.prompt_calls}
        self.prompts.append((ref_audio, ref_text, preprocess_prompt, prompt))
        return prompt

    def generate(self, text, **kwargs):
        self.generate_calls.append({"text": list(text), "kwargs": kwargs})
        return [np.ones(2400, dtype=np.float32) for _ in text]


def test_cached_prompt_is_reused_across_synthesizer_instances(monkeypatch):
    import pipeline.omnivoice_speech as ov

    ov._reset_prompt_cache_for_tests()
    voice_id = _make_clone("clone-prompt-cache-shared")
    model = PromptModel()
    first = OmniVoiceSynthesizer(
        model=model,
        transcriber=lambda _sample: "mẫu tham chiếu",
        batch_size=2,
    )
    second = OmniVoiceSynthesizer(
        model=model,
        transcriber=lambda _sample: "mẫu tham chiếu",
        batch_size=2,
    )

    first.synthesize_batch(["một"], voice_id)
    second.synthesize_batch(["hai"], voice_id)

    assert model.prompt_calls == 1


def test_clone_prompt_is_cached_for_repeated_batch_calls():
    import pipeline.omnivoice_speech as ov

    ov._reset_prompt_cache_for_tests()
    ov._MPS_BATCH_SAFE = True
    voice_id = _make_clone()
    model = PromptModel()
    synthesizer = OmniVoiceSynthesizer(
        model=model,
        transcriber=lambda _sample: "mẫu tham chiếu",
        batch_size=4,
    )

    synthesizer.synthesize_batch(["một", "hai"], voice_id)
    synthesizer.synthesize_batch(["ba"], voice_id)

    assert model.prompt_calls == 1
    assert len(model.generate_calls) == 2
    prompt = model.prompts[0][3]
    assert "generation_config" in model.generate_calls[0]["kwargs"]
    assert model.generate_calls[0]["kwargs"]["voice_clone_prompt"] == [prompt, prompt]
    assert model.generate_calls[1]["kwargs"]["voice_clone_prompt"] == [prompt]
    assert "ref_audio" not in model.generate_calls[0]["kwargs"]
    assert "ref_text" not in model.generate_calls[0]["kwargs"]


def test_forget_clone_invalidates_cached_prompt():
    import pipeline.omnivoice_speech as ov

    ov._reset_prompt_cache_for_tests()
    ov._MPS_BATCH_SAFE = True
    voice_id = _make_clone("clone-prompt-cache-forget")
    model = PromptModel()
    synthesizer = OmniVoiceSynthesizer(
        model=model,
        transcriber=lambda _sample: "mẫu tham chiếu",
        batch_size=2,
    )

    synthesizer.synthesize_batch(["một"], voice_id)
    ov.forget_clone(voice_id)
    synthesizer.synthesize_batch(["hai"], voice_id)

    assert model.prompt_calls == 2
