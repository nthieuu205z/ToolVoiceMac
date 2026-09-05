"""Process-wide coordination for production and preview speech calls."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from threading import Condition
from typing import Iterator


class PreviewBusyError(RuntimeError):
    """A preview cannot start while its provider is already occupied."""

    user_message = "Dịch vụ giọng đọc đang bận. Vui lòng thử lại sau."

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"speech preview provider is busy: {provider}")


class SpeechActivity:
    """Coordinate speech activity independently for each provider.

    Production calls may overlap. A preview owns one provider exclusively and is
    rejected instead of queued whenever that provider has production or preview
    work. Production waits only while an already-running preview owns the provider.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._production: defaultdict[str, int] = defaultdict(int)
        self._previews: set[str] = set()

    @contextmanager
    def production(self, provider: str) -> Iterator[None]:
        with self._condition:
            self._condition.wait_for(lambda: provider not in self._previews)
            self._production[provider] += 1
        try:
            yield
        finally:
            with self._condition:
                self._production[provider] -= 1
                if self._production[provider] == 0:
                    del self._production[provider]
                self._condition.notify_all()

    @contextmanager
    def preview(self, provider: str) -> Iterator[None]:
        with self._condition:
            if self._production[provider] or provider in self._previews:
                raise PreviewBusyError(provider)
            self._previews.add(provider)
        try:
            yield
        finally:
            with self._condition:
                self._previews.remove(provider)
                self._condition.notify_all()

    def active_production(self, provider: str) -> int:
        with self._condition:
            return self._production[provider]


speech_activity = SpeechActivity()
