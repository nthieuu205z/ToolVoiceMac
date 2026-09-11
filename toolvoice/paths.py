"""Writable user data stays outside the installed application."""
import hashlib
import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get('TOOLVOICE_DATA_DIR', Path.home() / 'Library/Application Support/ToolVoiceMac')).expanduser().resolve()


def log_dir() -> Path:
    return Path(os.environ.get('TOOLVOICE_LOG_DIR', Path.home() / 'Library/Logs/ToolVoiceMac')).expanduser().resolve()


def profile_id() -> str:
    return hashlib.sha256(str(data_dir()).encode()).hexdigest()
