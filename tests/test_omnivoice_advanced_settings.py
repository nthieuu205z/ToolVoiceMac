"""Settings-only restoration: synthetic fixtures, no provider/model execution."""
import importlib.util
import json
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError


def options(**kwargs):
    assert importlib.util.find_spec('pipeline.omnivoice_settings'), 'Missing advanced settings validation'
    from pipeline.omnivoice_settings import OmniVoiceSettings
    return OmniVoiceSettings(**kwargs)


@pytest.mark.parametrize('field,value', [
    ('speed', .49), ('speed', 1.51), ('speed', float('nan')),
    ('num_step', 3), ('num_step', 65), ('num_step', 4.5),
    ('guidance_scale', -.1), ('guidance_scale', 4.1),
    ('duration', 0), ('duration', float('inf')), ('duration', 30.01),
    ('duration', 86400.0), ('duration', 1e100), ('denoise', 'false'),
    ('preprocess_prompt', 1), ('postprocess_output', None), ('unknown', True),
])
def test_invalid_settings(field, value):
    with pytest.raises(ValidationError):
        options(**{field: value})


def test_defaults_are_frozen_scalars():
    config = options()
    assert config.speed == 1.0 and config.num_step == 32
    assert config.guidance_scale == 2.0
    with pytest.raises(ValidationError):
        config.speed = 1.2


def test_duration_limit_accepts_thirty_seconds_or_automatic():
    assert options(duration=30.0).duration == 30.0
    assert options(duration=None).duration is None


@pytest.fixture
def fake_upstream(monkeypatch):
    @dataclass
    class GenerationConfig:
        num_step: int = 32
        guidance_scale: float = 2.0
        denoise: bool = True
        preprocess_prompt: bool = True
        postprocess_output: bool = True
    module = ModuleType('omnivoice.models.omnivoice')
    module.OmniVoiceGenerationConfig = GenerationConfig
    monkeypatch.setitem(sys.modules, 'omnivoice.models.omnivoice', module)


def test_adapter_forwards_options_and_prompt_identity(fake_upstream):
    from tests.test_omnivoice_generation_config import _make_clone, ConfigModel
    from pipeline.omnivoice_speech import OmniVoiceSynthesizer, _reset_prompt_cache_for_tests
    _reset_prompt_cache_for_tests()
    model = ConfigModel()
    prompts = []
    def create(*args, **kwargs):
        prompt = object()
        prompts.append((kwargs['preprocess_prompt'], prompt))
        return prompt
    model.create_voice_clone_prompt = create
    voice = _make_clone('clone-advanced')
    for preprocess in (True, False, True):
        synth = OmniVoiceSynthesizer(model=model, transcriber=lambda _: 'synthetic reference',
            generation_settings=options(speed=1.25, duration=5.0, num_step=24,
                guidance_scale=3.0, denoise=False, preprocess_prompt=preprocess,
                postprocess_output=False))
        synth.synthesize('Synthetic fixture', voice)
    assert [p[0] for p in prompts] == [True, False]
    kwargs = model.calls[1]['kwargs']
    assert kwargs['speed'] == 1.0 and kwargs['duration'] == 5.0
    config = kwargs['generation_config']
    assert vars(config) == dict(num_step=24, guidance_scale=3.0, denoise=False,
                               preprocess_prompt=False, postprocess_output=False)
    assert model.calls[0]['kwargs']['voice_clone_prompt'] == model.calls[2]['kwargs']['voice_clone_prompt']


def test_speed_and_settings_survive_batch_to_single_retry(fake_upstream, monkeypatch):
    from tests.test_omnivoice_generation_config import _make_clone, ConfigModel
    from pipeline import omnivoice_speech as omni
    from pipeline.errors import SpeechServiceError
    monkeypatch.setattr(omni, '_accel_device', lambda: None)
    model = ConfigModel()
    generate = model.generate
    def fail_batch(text, **kwargs):
        if len(text) > 1:
            model.calls.append({'text': text, 'kwargs': kwargs})
            raise RuntimeError('synthetic batch failure')
        return generate(text, **kwargs)
    model.generate = fail_batch
    voice = _make_clone('clone-retry-settings')
    synth = omni.OmniVoiceSynthesizer(model=model, transcriber=lambda _: 'synthetic reference',
                                     generation_settings=options(speed=1.3, num_step=24))
    with pytest.raises(SpeechServiceError):
        synth.synthesize_batch(['One', 'Two'], voice)
    for text in ['One', 'Two']:
        synth.synthesize(text, voice)
    assert len(model.calls) == 3
    assert all(c['kwargs']['speed'] == 1.3 and 'duration' not in c['kwargs']
               and c['kwargs']['generation_config'].num_step == 24 for c in model.calls)


@pytest.fixture
def captured_queue(monkeypatch):
    from backend.routes import text_jobs
    from backend.config import settings
    captured = []
    monkeypatch.setattr(settings, 'gemini_api_key', 'synthetic-test-key')
    monkeypatch.setattr(settings, 'omnivoice_num_step', 32)
    monkeypatch.setattr(settings, 'omnivoice_batch_size', 0)
    monkeypatch.setattr(text_jobs, 'is_available', lambda *a, **k: True)
    monkeypatch.setattr(text_jobs, 'route_provider', lambda *a: 'omnivoice')
    def start(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(id=kwargs.get('job_id', 'synthetic-job'))
    monkeypatch.setattr(text_jobs.manager, 'start', start)
    return captured


def submit(client, payload=None, *, text='Synthetic fixture'):
    body = dict(text=text, voice_id='clone-test', language='en-US')
    if payload is not None:
        body['omnivoice'] = payload
    return client.post('/api/jobs/text', json=body)


@pytest.mark.parametrize('payload,steps,batch', [
    (None, 32, 8), ({'speed': 1.2}, 32, 8),
    ({'num_step': 24}, 24, 8), ({'num_step': 28}, 28, 8),
])
def test_submission_snapshots_settings_without_restoring_profile_batch_limits(
    client, monkeypatch, captured_queue, payload, steps, batch,
):
    from backend.config import settings
    from pipeline import omnivoice_speech as omni
    monkeypatch.setattr(omni, '_accel_device', lambda: 'mps')
    monkeypatch.setattr(omni, '_MPS_BATCH_SAFE', True)
    monkeypatch.setattr(omni, '_MPS_SINGLE_SAFE', True)
    monkeypatch.setattr(omni, '_MPS_BATCH_LIMIT', None)
    response = submit(client, payload)
    assert response.status_code == 200, response.text
    captured = captured_queue[-1]
    assert captured.get('omnivoice_settings') is not None, 'Job settings snapshot missing'
    monkeypatch.setattr(settings, 'omnivoice_num_step', 8)
    monkeypatch.setattr(settings, 'omnivoice_batch_size', 1)
    synth = captured['backend_factory']()
    assert synth._num_step == steps and synth.batch_size == batch
    assert synth._generation_settings is captured['omnivoice_settings']
    assert synth._generation_settings.speed == (payload or {}).get('speed', 1.0)
    assert not hasattr(synth, '_performance_profile')
    assert not hasattr(synth, 'batch_character_limit')


@pytest.mark.parametrize('provider', ['edge', 'gemini'])
def test_other_providers_reject_overrides_and_keep_legacy_factory(
    client, jobs_dir, monkeypatch, captured_queue, provider,
):
    from backend.routes import text_jobs as route
    monkeypatch.setattr(route, 'route_provider', lambda *a: provider)
    response = submit(client, {'num_step': 24})
    assert response.status_code == 400
    assert not captured_queue and list(jobs_dir.iterdir()) == []
    sentinel = object()
    monkeypatch.setattr(route, '_make_backend', lambda selected: sentinel if selected == provider else None)
    assert submit(client).status_code == 200
    assert captured_queue[-1]['backend_factory']() is sentinel
    assert captured_queue[-1].get('omnivoice_settings') is None


@pytest.mark.parametrize('text', ['One. Two.', 'x' * 1001], ids=['multiple-sentences', 'over-1000-characters'])
def test_duration_scope_rejected_before_files_or_queue(client, jobs_dir, captured_queue, text):
    response = submit(client, {'duration': 2.0}, text=text)
    assert response.status_code == 400
    assert 'Duration' in response.json()['detail']
    assert not captured_queue and list(jobs_dir.iterdir()) == []


def test_invalid_input_rejected_before_files_or_queue(client, jobs_dir, captured_queue):
    assert submit(client, {'num_step': 3}).status_code == 422
    assert submit(client, {'duration': 86400.0}).status_code == 422
    assert not captured_queue and list(jobs_dir.iterdir()) == []


def test_capabilities_keep_config_default(client, monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, 'omnivoice_num_step', 36)
    response = client.get('/api/omnivoice/settings')
    assert response.status_code == 200
    body = response.json()
    assert body['supported'] is True
    assert body['defaults']['num_step'] == 36
    assert set(body['defaults']) == {'speed', 'duration', 'num_step', 'guidance_scale', 'denoise', 'preprocess_prompt', 'postprocess_output'}


def test_capability_detection_is_source_only_and_fails_closed(tmp_path, monkeypatch):
    options()
    from pipeline import omnivoice_settings as module
    package = tmp_path / 'package'
    (package / 'models').mkdir(parents=True)
    source = package / 'models' / 'omnivoice.py'
    source.write_text('class OmniVoiceGenerationConfig:\n    num_step: int = 32\n    guidance_scale: float = 2.0\n    denoise: bool = True\n    preprocess_prompt: bool = True\n    postprocess_output: bool = True\nclass OmniVoice:\n    def generate(self, text, speed=1.0, duration=None, generation_config=None): pass\n')
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda _: SimpleNamespace(origin=str(package / '__init__.py')))
    assert module.installed_capabilities()['supported'] is True
    source.write_text('class OmniVoice: pass\n')
    assert module.installed_capabilities() == {'supported': False, 'defaults': {}}


def test_unsupported_server_rejects_overrides(client, jobs_dir, captured_queue, monkeypatch):
    options()
    from pipeline import omnivoice_settings as module
    monkeypatch.setattr(module, 'installed_capabilities', lambda: {'supported': False, 'defaults': {}})
    assert submit(client, {'num_step': 24}).status_code == 400
    assert not captured_queue and list(jobs_dir.iterdir()) == []
    assert submit(client).status_code == 200


def test_manager_persists_scalar_settings_and_restore_is_independent(tmp_path, monkeypatch):
    from backend.job_manager import JobManager
    config = options(num_step=24, speed=1.2)
    mgr = JobManager()
    monkeypatch.setattr(mgr, '_ensure_executor', lambda: SimpleNamespace(submit=lambda *a: SimpleNamespace()))
    workdir = tmp_path / 'job'
    workdir.mkdir()
    job = mgr.start(filename='fixture', voice_id='clone-test', workdir=workdir,
                    job_type='text_to_voice',
                    backend_factory=lambda: None, runner=lambda *a: None, omnivoice_settings=config)
    saved = json.loads((workdir / 'job.json').read_text())['omnivoice_settings']
    assert saved == config.model_dump()
    assert all(value is None or type(value) in (str, int, float, bool) for value in saved.values())
    assert not {'text', 'ref_text', 'gemini_api_key', 'path', 'workdir'} & saved.keys()
    snapshot = job.snapshot()
    snapshot['omnivoice_settings']['num_step'] = 4
    assert job.snapshot()['omnivoice_settings']['num_step'] == 24
    restored = JobManager()
    assert restored.restore(tmp_path) == 1
    assert restored.get(job.id).snapshot()['omnivoice_settings'] == saved
