"""
Domain models for stem separation representation in EchoInsight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StemPaths:
    """Immutable value object holding verified paths to separated stems."""
    vocals: Path
    drums: Path
    bass: Path
    other: Path

    def __post_init__(self) -> None:
        for name, path in (
            ("vocals", self.vocals),
            ("drums", self.drums),
            ("bass", self.bass),
            ("other", self.other),
        ):
            if not isinstance(path, Path):
                object.__setattr__(self, name, Path(path))


@dataclass(frozen=True)
class SeparatedTrack:
    """Represents a fully separated track ready for mixing and analysis."""
    track_id: str
    original_audio: Path
    sample_rate: int
    duration_seconds: float
    stems: StemPaths
    model_name: str = "htdemucs"