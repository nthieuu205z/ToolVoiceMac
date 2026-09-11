"""Main app opt-in contracts; no real model, jobs, voices or credentials."""
import sys
from types import SimpleNamespace

import pytest

from pipeline import omnivoice_speech as omni
from pipeline.errors import SpeechServiceError


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    import huggingface_hub
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: str(snapshot))
    for name, value in {
        '_MODEL': None, '_MODEL_DEVICE': None, '_OPTIMIZATION': 'none',
        '_MODEL_OPTIMIZATION': None, '_MODEL_LOADING': False, '_MODEL_ERROR': None,
        '_CODEC_DEVICE': 'cpu', '_MPS_BATCH_SAFE': True, '_MPS_SINGLE_SAFE': True, '_MPS_BATCH_LIMIT': None,
    }.items():
        monkeypatch.setattr(omni, name, value, raising=False)


def configure(mode):
    assert callable(getattr(omni, 'configure_optimization', None)), 'Startup mode configuration is missing'
    omni.configure_optimization(mode)


def fake_packages(monkeypatch, device='mps'):
    events = []
    codec = SimpleNamespace(device='cpu')
    codec.to = lambda value: setattr(codec, 'device', value)
    model = SimpleNamespace(device='cpu', audio_tokenizer=codec)
    def move(value):
        events.append(('move', value))
        model.device = value
        return model
    model.to = move
    class Original:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            events.append(('load', cls, kwargs))
            return model
    class Split(Original):
        pass
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float16='float16', float32='float32'))
    monkeypatch.setitem(sys.modules, 'omnivoice', SimpleNamespace(OmniVoice=Original))
    from pipeline import omnivoice_split_cfg
    def select():
        events.append(('source_guard',))
        return Split
    monkeypatch.setattr(omnivoice_split_cfg, 'get_split_cfg_class', select)
    monkeypatch.setattr(omni, '_accel_device', lambda: device)
    return Original, Split, model, events


@pytest.mark.parametrize('mode', ['none', 'split-cfg'])
def test_startup_selects_one_model_and_reports_actual_mode(monkeypatch, mode):
    original, split, model, events = fake_packages(monkeypatch)
    configure(mode)
    assert omni.runtime_status()['state'] == 'not_loaded'
    assert omni.runtime_status()['loaded_optimization'] is None
    assert omni._get_model() is model
    assert omni._get_model() is model
    loads = [event for event in events if event[0] == 'load']
    assert len(loads) == 1 and loads[0][1] is (split if mode == 'split-cfg' else original)
    assert loads[0][2]['device_map'] == 'cpu' and loads[0][2]['dtype'] == 'float16'
    if mode == 'split-cfg':
        assert events[0] == ('source_guard',)
    assert codec_device(model) == 'cpu'
    status = omni.runtime_status()
    assert status['state'] == 'ready' and status['loaded_optimization'] == mode
    assert status['requested_optimization'] == mode and status['device'] == 'mps'
    assert status['batch_ceiling'] == (2 if mode == 'split-cfg' else 8)
    configure(mode)  # Idempotent startup is harmless.
    with pytest.raises(RuntimeError):
        configure('none' if mode == 'split-cfg' else 'split-cfg')


def codec_device(model):
    return model.audio_tokenizer.device


@pytest.mark.parametrize('mode', ['', 'auto', 'SPLIT-CFG', None, 1])
def test_invalid_mode_rejected_without_changes(mode):
    with pytest.raises(ValueError):
        configure(mode)
    assert omni._OPTIMIZATION == 'none' and omni._MODEL is None


@pytest.mark.parametrize('accelerator', [None, 'cuda'])
def test_split_requires_mps_without_loading(monkeypatch, accelerator):
    _, _, _, events = fake_packages(monkeypatch, accelerator)
    configure('split-cfg')
    with pytest.raises(SpeechServiceError):
        omni._get_model()
    assert not [event for event in events if event[0] == 'load']
    assert omni._MODEL is None and omni.runtime_status()['loaded_optimization'] is None


def test_source_guard_failure_has_no_fallback_or_partial_model(monkeypatch):
    _, _, _, events = fake_packages(monkeypatch)
    from pipeline import omnivoice_split_cfg
    def fail():
        raise RuntimeError('private source path and details')
    monkeypatch.setattr(omnivoice_split_cfg, 'get_split_cfg_class', fail)
    configure('split-cfg')
    with pytest.raises(SpeechServiceError) as caught:
        omni._get_model()
    assert 'private' not in str(caught.value)
    assert events == [] and omni._MODEL is None
    assert omni.runtime_status()['state'] == 'error'
    assert omni.runtime_status()['error'] == 'load_failed'
    assert omni.runtime_status()['loaded_optimization'] is None


def test_failed_device_move_never_publishes_ready_model(monkeypatch):
    _, _, model, _ = fake_packages(monkeypatch)
    def fail(value):
        assert omni._MODEL is None
        assert omni.runtime_status()['state'] == 'loading'
        raise RuntimeError('move failed')
    model.to = fail
    configure('split-cfg')
    with pytest.raises(RuntimeError):
        omni._get_model()
    assert omni._MODEL is None and omni.runtime_status()['state'] == 'error'
    assert omni.runtime_status()['loaded_optimization'] is None


def test_split_batch_cap_preserves_mps_safety_flags(monkeypatch):
    monkeypatch.setattr(omni, '_accel_device', lambda: 'mps')
    configure('split-cfg')
    assert omni.OmniVoiceSynthesizer().batch_size == 2
    assert omni.OmniVoiceSynthesizer(batch_size=64).batch_size == 2
    assert omni.OmniVoiceSynthesizer(batch_size=1).batch_size == 1
    monkeypatch.setattr(omni, '_MPS_BATCH_SAFE', False)
    assert omni.OmniVoiceSynthesizer().batch_size == 1
    monkeypatch.setattr(omni, '_MPS_SINGLE_SAFE', False)
    assert omni.OmniVoiceSynthesizer().batch_size == 0


def test_configuration_during_model_load_rejected_without_wait(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    configure('none')
    monkeypatch.setattr(omni, '_MODEL_LOADING', True)
    omni._MODEL_LOCK.acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(configure, 'split-cfg')
        try:
            with pytest.raises(RuntimeError, match='Restart'):
                request.result(timeout=.5)
        finally:
            omni._MODEL_LOCK.release()
    assert omni._OPTIMIZATION == 'none'


def test_runtime_endpoint_never_loads_or_waits_on_model_lock(client, monkeypatch):
    configure('split-cfg')
    def forbidden(*args):
        pytest.fail('Read-only status must not load or probe model')
    monkeypatch.setattr(omni, '_get_model', forbidden)
    monkeypatch.setattr(omni, '_accel_device', forbidden)
    omni._MODEL_LOCK.acquire()
    try:
        response = client.get('/api/omnivoice/runtime')
    finally:
        omni._MODEL_LOCK.release()
    assert response.status_code == 200
    assert response.json() == {
        'requested_optimization': 'split-cfg', 'loaded_optimization': None,
        'state': 'not_loaded', 'device': None, 'batch_ceiling': 2,
        'experimental': True, 'error': None,
        'requested_codec_device': 'cpu', 'codec_device': None,
    }


def test_settings_default_and_env_opt_in(monkeypatch):
    from backend.config import Settings
    monkeypatch.delenv('OMNIVOICE_OPTIMIZATION', raising=False)
    assert Settings(_env_file=None).omnivoice_optimization == 'split-cfg-rms-gqa-rope'
    monkeypatch.setenv('OMNIVOICE_OPTIMIZATION', 'split-cfg')
    assert Settings(_env_file=None).omnivoice_optimization == 'split-cfg'
    monkeypatch.setenv('OMNIVOICE_OPTIMIZATION', 'invalid')
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_lifespan_configures_mode_before_prewarm(monkeypatch):
    import asyncio
    from backend import main
    from backend.config import Settings, settings
    from pipeline import whisper_stt
    events = []
    monkeypatch.setattr(settings, 'omnivoice_optimization', 'split-cfg', raising=False)
    monkeypatch.setattr(settings, 'omnivoice_codec_device', 'mps')
    monkeypatch.setattr(Settings, 'validate_providers', lambda self: None)
    monkeypatch.setattr(Settings, 'resolved_clone_provider', property(lambda self: 'omnivoice'))
    monkeypatch.setattr(omni, 'configure_optimization', lambda mode, *, codec_device: events.append(('configure', mode, codec_device)), raising=False)
    monkeypatch.setattr(omni, 'prewarm', lambda: events.append(('prewarm',)))
    monkeypatch.setattr(whisper_stt, 'prewarm', lambda *args: None)
    async def run():
        async with main._lifespan(main.app):
            pass
    asyncio.run(run())
    assert events == [('configure', 'split-cfg', 'mps'), ('prewarm',)]


def test_mps_codec_opt_in_places_codec_on_gpu_and_reports_actual_device(monkeypatch):
    _, _, model, _ = fake_packages(monkeypatch)
    omni.configure_optimization('none', codec_device='mps')
    assert omni.runtime_status()['codec_device'] is None
    omni._get_model()
    assert model.audio_tokenizer.device == 'mps'
    assert omni.runtime_status()['codec_device'] == 'mps'
    assert omni.runtime_status()['requested_codec_device'] == 'mps'
    with pytest.raises(RuntimeError, match='Restart'):
        omni.configure_optimization('none', codec_device='cpu')
    assert model.audio_tokenizer.device == 'mps'


def test_codec_device_rejects_unsupported_values_before_load():
    with pytest.raises(ValueError):
        omni.configure_optimization('none', codec_device='bad-device')
    assert omni._MODEL is None


def test_requested_mps_codec_requires_mps_accelerator(monkeypatch):
    fake_packages(monkeypatch, device=None)
    omni.configure_optimization('none', codec_device='mps')
    with pytest.raises(SpeechServiceError, match='MPS'):
        omni._get_model()
    assert omni.runtime_status()['codec_device'] is None


def test_codec_environment_is_validated(monkeypatch):
    from backend.config import Settings
    monkeypatch.setenv('OMNIVOICE_CODEC_DEVICE', 'mps')
    assert Settings(_env_file=None).omnivoice_codec_device == 'mps'
    monkeypatch.setenv('OMNIVOICE_CODEC_DEVICE', 'invalid')
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_native_rms_mode_activates_after_mps_move_and_reports_ready(monkeypatch):
    original, split, model, events = fake_packages(monkeypatch)
    import pipeline.omnivoice_mps_norm as norms

    def activate(loaded):
        assert loaded.device == 'mps'
        loaded.native_norms_active = True
        return 5

    monkeypatch.setattr(norms, 'enable_native_rms_norm', activate)
    configure('split-cfg-rms')
    assert omni._get_model() is model
    assert model.native_norms_active
    status = omni.runtime_status()
    assert status['loaded_optimization'] == 'split-cfg-rms'
    assert status['batch_ceiling'] == 2 and status['experimental']
    with pytest.raises(RuntimeError, match='Restart'):
        configure('split-cfg')


def test_native_rms_failure_does_not_publish_partial_model(monkeypatch):
    fake_packages(monkeypatch)
    import pipeline.omnivoice_mps_norm as norms

    def unavailable(model):
        raise RuntimeError('Native MPS RMSNorm unavailable')

    monkeypatch.setattr(norms, 'enable_native_rms_norm', unavailable)
    configure('split-cfg-rms')
    with pytest.raises(RuntimeError, match='unavailable'):
        omni._get_model()
    assert omni.runtime_status()['state'] == 'error'
    assert omni.runtime_status()['loaded_optimization'] is None


def test_native_gqa_mode_validates_environment_and_activates_before_ready(monkeypatch):
    from backend.config import Settings
    mode = 'split-cfg-rms-gqa'
    monkeypatch.setenv('OMNIVOICE_OPTIMIZATION', mode)
    assert Settings(_env_file=None).omnivoice_optimization == mode
    _, _, model, events = fake_packages(monkeypatch)
    from pipeline import omnivoice_mps_norm as norms, omnivoice_mps_attention as attention
    activated = []

    def activate(name, loaded):
        assert loaded is model and loaded.device == 'mps'
        assert omni._MODEL is None
        activated.append(name)
        return 28

    monkeypatch.setattr(norms, 'enable_native_rms_norm', lambda m: activate('norm', m))
    monkeypatch.setattr(attention, 'enable_native_mps_gqa', lambda m: activate('attention', m))
    configure(mode)
    assert omni._get_model() is model
    assert activated == ['norm', 'attention']
    assert omni.runtime_status()['loaded_optimization'] == mode
    assert omni.runtime_status()['batch_ceiling'] == 2
    with pytest.raises(RuntimeError, match='Restart'):
        configure('split-cfg-rms')


def test_native_gqa_failure_never_publishes_ready_model(monkeypatch):
    fake_packages(monkeypatch)
    from pipeline import omnivoice_mps_norm as norms, omnivoice_mps_attention as attention
    monkeypatch.setattr(norms, 'enable_native_rms_norm', lambda m: 113)

    def unavailable(model):
        raise RuntimeError('Native MPS attention unavailable')

    monkeypatch.setattr(attention, 'enable_native_mps_gqa', unavailable)
    configure('split-cfg-rms-gqa')
    with pytest.raises(RuntimeError, match='unavailable'):
        omni._get_model()
    assert omni.runtime_status()['state'] == 'error'
    assert omni.runtime_status()['loaded_optimization'] is None


def test_native_rope_mode_accepts_config_and_keeps_two_item_batch_ceiling(monkeypatch):
    from backend.config import Settings
    mode = 'split-cfg-rms-gqa-rope'
    monkeypatch.setenv('OMNIVOICE_OPTIMIZATION', mode)
    assert Settings(_env_file=None).omnivoice_optimization == mode
    omni.configure_optimization(mode, codec_device='mps')
    status = omni.runtime_status()
    assert status['requested_optimization'] == mode
    assert status['state'] == 'not_loaded'
    assert status['loaded_optimization'] is None
    assert status['batch_ceiling'] == 2
    assert status['experimental'] is True


def test_native_rope_mode_cannot_swap_a_live_model(monkeypatch):
    omni.configure_optimization('split-cfg-rms-gqa-rope', codec_device='mps')
    monkeypatch.setattr(omni, '_MODEL', object())
    with pytest.raises(RuntimeError, match='Restart'):
        omni.configure_optimization('split-cfg-rms-gqa', codec_device='mps')


@pytest.mark.parametrize('fail_preflight', [False, True])
def test_native_rope_preflight_finishes_before_model_is_published(monkeypatch, fail_preflight):
    _, _, model, _ = fake_packages(monkeypatch)
    from pipeline import omnivoice_mps_norm as norms, omnivoice_mps_attention as attention, omnivoice_mps_rope as rope
    activated = []
    def activate(name, loaded):
        assert loaded is model and model.device == 'mps'
        assert omni._MODEL is None and omni.runtime_status()['state'] == 'loading'
        activated.append(name)
        if name == 'rope' and fail_preflight:
            raise RuntimeError('Metal preflight failed')
        return 28
    monkeypatch.setattr(norms, 'enable_native_rms_norm', lambda m: activate('norm', m))
    monkeypatch.setattr(attention, 'enable_native_mps_gqa', lambda m: activate('attention', m))
    monkeypatch.setattr(rope, 'enable_native_mps_rope', lambda m: activate('rope', m))
    omni.configure_optimization('split-cfg-rms-gqa-rope', codec_device='mps')
    if fail_preflight:
        with pytest.raises(RuntimeError, match='preflight'):
            omni._get_model()
        assert omni.runtime_status()['state'] == 'error'
        assert omni.runtime_status()['loaded_optimization'] is None
    else:
        assert omni._get_model() is model
        assert omni.runtime_status()['state'] == 'ready'
        assert omni.runtime_status()['loaded_optimization'] == 'split-cfg-rms-gqa-rope'
    assert activated == ['norm','attention','rope']
