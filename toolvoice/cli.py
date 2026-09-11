"""Start the local web app, or open an already running matching instance."""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import logging
import os
import platform
import secrets
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from . import __version__
from .paths import data_dir, log_dir, profile_id


def _identity(port: int) -> dict:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/toolvoice', timeout=1) as response:
            value = json.loads(response.read(4096))
            return value if isinstance(value, dict) else {}
    except (OSError, ValueError, urllib.error.URLError):
        return {}


def _matches(identity: dict, session: str | None = None) -> bool:
    return (identity.get('app') == 'toolvoice'
            and identity.get('profile_id') == profile_id()
            and (session is None or identity.get('session_id') == session))


def _open(port: int, browser: bool) -> None:
    url = f'http://127.0.0.1:{port}/'
    print(f'ToolVoiceMac: {url}', flush=True)
    if browser:
        webbrowser.open(url)


def _open_when_ready(port: int, session: str, stop: threading.Event) -> None:
    for _ in range(120):
        if stop.is_set():
            return
        if _matches(_identity(port), session):
            webbrowser.open(f'http://127.0.0.1:{port}/')
            return
        stop.wait(0.5)


def _platform_error() -> str | None:
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        return 'ToolVoiceMac yêu cầu macOS trên Apple Silicon (M1 trở lên).'
    version = platform.mac_ver()[0]
    if version and int(version.split('.')[0]) < 14:
        return 'ToolVoiceMac yêu cầu macOS 14 trở lên.'
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='toolvoice', description='Khởi động ToolVoiceMac và mở giao diện trong trình duyệt.')
    parser.add_argument('--version', action='version', version=f'ToolVoiceMac {__version__}')
    parser.add_argument('--port', type=int, default=8000, help='Cổng local (mặc định: 8000)')
    parser.add_argument('--no-browser', action='store_true', help='Chạy server, không tự mở trình duyệt')
    parser.add_argument('--data-dir', type=Path, help='Thư mục dữ liệu riêng (mặc định: macOS Application Support)')
    parser.add_argument('--doctor', action='store_true', help='Kiểm tra môi trường, không nạp model')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port phải nằm trong khoảng 1–65535')
    if args.data_dir:
        os.environ['TOOLVOICE_DATA_DIR'] = str(args.data_dir.expanduser().resolve())
    error = _platform_error()
    if error:
        print(error, file=sys.stderr)
        return 1
    # Import config only after applying --data-dir; this does not load a model.
    from backend.config import settings
    binaries = {'ffmpeg': settings.ffmpeg_bin or 'ffmpeg', 'ffprobe': settings.ffprobe_bin or 'ffprobe'}
    missing = [name for name, command in binaries.items() if shutil.which(command) is None]
    if args.doctor:
        from importlib.metadata import version
        print(json.dumps({'version': __version__, 'python': platform.python_version(), 'data_dir': str(data_dir()),
                          'optimization': settings.omnivoice_optimization, 'codec': settings.omnivoice_codec_device,
                          'missing_binaries': missing, 'packages': {name: version(name) for name in ['torch','torchaudio','transformers','omnivoice']}}, indent=2))
        return int(bool(missing))
    if missing:
        print('Thiếu ffmpeg/ffprobe. Cài bằng: brew install ffmpeg', file=sys.stderr)
        return 1
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = root / 'server.json'
    stop = threading.Event()
    # Keep the lock inode in place: unlinking it would permit concurrent owners.
    with (root / '.server.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # The owner may still be starting. Wait for its identity, never kill it.
            for _ in range(20):
                try:
                    state = json.loads(state_path.read_text())
                    port = int(state['port'])
                    if _matches(_identity(port), state['session_id']):
                        _open(port, not args.no_browser)
                        return 0
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                time.sleep(0.25)
            print('ToolVoiceMac đang khởi động hoặc đã giữ thư mục dữ liệu này. Kiểm tra cửa sổ Terminal đang chạy.', file=sys.stderr)
            return 1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', args.port))
                sock.listen(128)
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE and _matches(_identity(args.port)):
                    _open(args.port, not args.no_browser)
                    return 0
                print(f'Không mở được cổng {args.port}; không dừng tiến trình đang dùng cổng. Thử --port khác.', file=sys.stderr)
                return 1
            session = secrets.token_hex(16)
            os.environ['TOOLVOICE_SESSION_ID'] = session
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
            state_path.write_text(json.dumps({'port': args.port, 'session_id': session, 'pid': os.getpid()}))
            state_path.chmod(0o600)
            logs = log_dir()
            logs.mkdir(parents=True, exist_ok=True, mode=0o700)
            handler = logging.FileHandler(logs / 'toolvoice.log', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s — %(message)s'))
            logging.getLogger().addHandler(handler)
            logging.getLogger().setLevel(logging.INFO)
            print(f'ToolVoiceMac {__version__} · dữ liệu: {root}\nCtrl+C để dừng. Log: {logs / "toolvoice.log"}', flush=True)
            _open(args.port, False)
            if not args.no_browser:
                threading.Thread(target=_open_when_ready, args=(args.port, session, stop), daemon=True).start()
            try:
                import uvicorn
                uvicorn.Server(uvicorn.Config('backend.main:app', host='127.0.0.1', port=args.port, log_level='info')).run(sockets=[sock])
            finally:
                stop.set()
                state_path.unlink(missing_ok=True)
                logging.getLogger().removeHandler(handler)
                handler.close()
    return 0
