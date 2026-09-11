"""User-facing launcher contracts: isolated data and conservative server reuse."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def test_launcher_is_available_without_loading_model():
    assert importlib.util.find_spec('toolvoice') is not None, 'terminal launcher package is missing'
    result = subprocess.run([sys.executable, '-m', 'toolvoice', '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--no-browser' in result.stdout and '--port' in result.stdout


def test_config_keeps_writable_data_outside_installed_code(tmp_path):
    data = tmp_path / 'persistent data'
    data.mkdir()
    (data / '.env').write_text('WHISPER_MODEL=base\n')
    env = dict(os.environ, TOOLVOICE_DATA_DIR=str(data))
    code = '''import json
from backend.config import settings
from backend.routes.settings import ENV_PATH
print(json.dumps({'jobs':str(settings.jobs_dir),'voices':str(settings.custom_voices_dir),'previews':str(settings.previews_dir),'env':str(ENV_PATH),'model':settings.whisper_model,'has_ui':(settings.static_dir/'index.html').is_file()}))'''
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values == {'jobs':str(data/'jobs'), 'voices':str(data/'custom_voices'), 'previews':str(data/'previews'), 'env':str(data/'.env'), 'model':'base', 'has_ui':True}


def test_health_identity_belongs_to_this_data_store(client):
    response = client.get('/api/toolvoice')
    assert response.status_code == 200
    assert response.json()['app'] == 'toolvoice'
    assert len(response.json()['profile_id']) == 64


def test_refuses_unrelated_service_without_stopping_it(tmp_path):
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        sock.listen()
        port = sock.getsockname()[1]
        result = subprocess.run([sys.executable, '-m', 'toolvoice', '--no-browser', '--data-dir', str(tmp_path), '--port', str(port)], capture_output=True, text=True, timeout=8)
        assert result.returncode == 1
        assert 'Không mở được cổng' in result.stderr
        assert sock.getsockname()[1] == port


def test_second_launch_reuses_same_profile_and_keeps_user_data(tmp_path):
    import socket
    import time
    import urllib.request
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    data = tmp_path/'data'
    data.mkdir()
    (data/'.env').write_text('CLONE_TTS_PROVIDER=none\nSTT_PROVIDER=gemini\n')
    (data/'user-note.txt').write_text('keep me')
    cmd = [sys.executable, '-m', 'toolvoice', '--no-browser', '--data-dir', str(data), '--port', str(port)]
    env = dict(os.environ, TOOLVOICE_LOG_DIR=str(tmp_path/'logs'))
    with (tmp_path/'server.log').open('w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise AssertionError((tmp_path/'server.log').read_text())
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/toolvoice', timeout=.3) as response:
                        identity = json.load(response)
                    break
                except OSError:
                    time.sleep(.05)
            else:
                raise AssertionError('server did not become ready')
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=8)
            assert result.returncode == 0, result.stderr
            assert process.poll() is None
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/toolvoice') as response:
                assert json.load(response)['session_id'] == identity['session_id']
        finally:
            process.terminate()
            process.wait(timeout=10)
    assert (data/'user-note.txt').read_text() == 'keep me'
    assert (data/'.env').read_text() == 'CLONE_TTS_PROVIDER=none\nSTT_PROVIDER=gemini\n'


def test_failed_startup_returns_failure_and_releases_profile_lock(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / '.env').write_text('STT_PROVIDER=unsupported\nCLONE_TTS_PROVIDER=none\n')
    import socket
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    env = dict(os.environ, TOOLVOICE_LOG_DIR=str(tmp_path / 'logs'))
    cmd = [sys.executable, '-m', 'toolvoice', '--no-browser', '--data-dir', str(data), '--port', str(port)]
    for _ in range(2):
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=10)
        assert result.returncode != 0
        assert 'Application startup failed' in result.stderr
        assert not (data / 'server.json').exists()
