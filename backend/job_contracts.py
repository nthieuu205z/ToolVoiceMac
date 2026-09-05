"""Typed contracts shared by asynchronous job runners and API snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pipeline.models import CancelFn, ProgressFn


@dataclass(frozen=True)
class JobArtifact:
    id: str
    kind: str
    filename: str
    media_type: str
    path: str

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "media_type": self.media_type,
        }


@dataclass
class JobRunResult:
    artifacts: list[JobArtifact]
    warnings: list[str] = field(default_factory=list)
    attempted_count: int = 0
    spoken_count: int = 0

    @property
    def degraded(self) -> bool:
        return self.attempted_count > 0 and self.spoken_count < self.attempted_count


JobRunner = Callable[[object, ProgressFn, CancelFn], JobRunResult]
