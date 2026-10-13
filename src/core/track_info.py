"""
track_info.py — Descriptive track metadata where every field records its source.

``TrackInfo`` holds the human-facing description of a track (title, artist,
album, release date).  Each field is a ``SourcedText`` that pairs the value
with where it came from, so a missing value is stored as *unknown* rather
than as a plausible-looking placeholder.

Layer rules
-----------
- ``src.core.track_info`` imports **only** Python stdlib.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

METADATA_SOURCES: Tuple[str, ...] = ("file_tags", "spotify", "user", "unknown")
"""Allowed values for ``SourcedText.source``."""


@dataclasses.dataclass(frozen=True)
class SourcedText:
    """A text value together with the source it was obtained from.

    Attributes
    ----------
    value : Optional[str]
        The value, or ``None`` when it is not known.
    source : str
        One of ``METADATA_SOURCES``.  ``"unknown"`` if and only if
        ``value`` is ``None``.
    """

    value: Optional[str] = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.source not in METADATA_SOURCES:
            raise ValueError(
                f"source must be one of {METADATA_SOURCES}, got {self.source!r}"
            )
        if self.value is None and self.source != "unknown":
            raise ValueError(
                f"a missing value must have source 'unknown', got {self.source!r}"
            )
        if self.value is not None:
            if self.source == "unknown":
                raise ValueError("a known value must name its source")
            if not self.value.strip():
                raise ValueError("value must be non-empty when provided")

    @property
    def is_known(self) -> bool:
        """Whether a value is present."""
        return self.value is not None


@dataclasses.dataclass(frozen=True)
class TrackInfo:
    """Descriptive metadata for a track, each field with its source."""

    title: SourcedText = dataclasses.field(default_factory=SourcedText)
    artist: SourcedText = dataclasses.field(default_factory=SourcedText)
    album: SourcedText = dataclasses.field(default_factory=SourcedText)
    release_date: SourcedText = dataclasses.field(default_factory=SourcedText)
