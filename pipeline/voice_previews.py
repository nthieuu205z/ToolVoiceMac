"""Language-specific, path-safe voice preview storage."""

from __future__ import annotations

import os
import re
import shutil
import threading
import uuid
from pathlib import Path

from .audio import pcm_to_array, write_wav
from .languages import require_language

_VOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VALID_STATUSES = {"pending", "ready", "error"}


class VoicePreviewStore:
    """Own preview paths and lock-protected generation states."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._statuses: dict[tuple[str, str], str] = {}

    def path(self, voice_id: str, language: str) -> Path:
        if not isinstance(voice_id, str) or not _VOICE_ID_RE.fullmatch(voice_id):
            raise ValueError(f"ID giọng không hợp lệ: {voice_id}")
        canonical = require_language(language).code
        return self.root / voice_id / f"{canonical}.wav"

    def status(self, voice_id: str, language: str) -> str:
        path = self.path(voice_id, language)
        canonical = require_language(language).code
        legacy = self.root / f"{voice_id}.wav"
        if path.is_file() or canonical == "vi-VN" and legacy.is_file():
            return "ready"
        with self._lock:
            return self._statuses.get((voice_id, canonical), "error")

    def set_status(self, voice_id: str, language: str, status: str) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"Trạng thái preview không hợp lệ: {status}")
        canonical = require_language(language).code
        self.path(voice_id, canonical)
        with self._lock:
            self._statuses[(voice_id, canonical)] = status

    def set_pending(self, voice_id: str, language: str) -> None:
        self.set_status(voice_id, language, "pending")

    def begin_generation(self, voice_id: str, language: str) -> bool:
        """Atomically claim one preview unless it is already pending."""
        canonical = require_language(language).code
        self.path(voice_id, canonical)
        with self._lock:
            key = (voice_id, canonical)
            if self._statuses.get(key) == "pending":
                return False
            self._statuses[key] = "pending"
            return True

    def clear(self, voice_id: str) -> None:
        directory = self.path(voice_id, "vi-VN").parent
        shutil.rmtree(directory, ignore_errors=True)
        (self.root / f"{voice_id}.wav").unlink(missing_ok=True)
        with self._lock:
            for key in [key for key in self._statuses if key[0] == voice_id]:
                self._statuses.pop(key, None)

    def generate_languages(self, voice_id: str, languages, synthesizer) -> None:
        for language in languages:
            spec = require_language(language)
            path = self.path(voice_id, spec.code)
            self.set_pending(voice_id, spec.code)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                pcm = synthesizer.synthesize(
                    spec.preview_text, voice_id, language=spec.code
                )
                write_wav(temporary, pcm_to_array(pcm))
                os.replace(temporary, path)
            except Exception:
                temporary.unlink(missing_ok=True)
                self.set_status(voice_id, spec.code, "error")
            else:
                self.set_status(voice_id, spec.code, "ready")