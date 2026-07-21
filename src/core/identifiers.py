"""
identifiers.py — Core domain type aliases and generic containers.

All domain types in EchoInsight reference songs and segments through
lightweight, type-safe identifiers rather than raw strings or ints.
This module defines those identifiers plus the generic ``Estimate``
container that pairs any value with a confidence score.

Layer rules
-----------
- ``src.core.identifiers`` imports **only** Python stdlib.
- It must **never** import ``src.analysis``, ``src.config``, ``src.app``,
  or any external package.

Ownership
---------
- ``SongID``   — owned by ``SongDNA``
- ``SegmentIndex`` — owned by ``SegmentDNA`` (via ``SegmentContext.index``)
- ``URI``      — owned by ``FrameReference``
- ``Estimate`` — value container, used by any type that needs confidence
"""

from __future__ import annotations

import uuid as _uuid
from typing import Generic, NewType, TypeVar

# ---------------------------------------------------------------------------
# Domain identifiers
# ---------------------------------------------------------------------------

SongID = NewType("SongID", str)
"""A UUID v4 string that uniquely identifies a song across the entire dataset.

Every SongDNA receives a SongID at extraction time.  This ID is the
foreign key used by future aggregates (StemDNA, EmbeddingDNA, etc.) to
reference the song without coupling to the SongDNA schema.

Usage
-----
.. code-block:: python

    sid = SongID(uuid.uuid4().hex)
    song_dna = SongDNA(metadata=SongMetadata(song_id=sid, ...), ...)
"""

SegmentIndex = NewType("SegmentIndex", int)
"""0-based position of a segment within ``StructureDNA.segments``.

Used by the comparison and recommendation engines to reference specific
segments without embedding the full ``SegmentDNA`` value.

Invariants
----------
- Must be >= 0
- Must be < len(StructureDNA.segments)
"""

URI = NewType("URI", str)
"""A relative URI within the EchoInsight data root.

URIs never contain absolute filesystem paths.  Absolute resolution is
handled by ``ApplicationContext.resolve_data_path()``.

Examples
--------
- ``"frames/abc123.npz"``
- ``"ssm/abc123.npz"``
"""


def generate_song_id() -> SongID:
    """Return a new, unique ``SongID``.

    Uses :func:`uuid.uuid4` under the hood.  Collision probability is
    negligible for any realistic dataset size.
    """
    return SongID(_uuid.uuid4().hex)


# ---------------------------------------------------------------------------
# Generic value containers
# ---------------------------------------------------------------------------

T = TypeVar("T")


class Estimate(Generic[T]):
    """A value paired with a normalised confidence score.

    ``Estimate`` is the standard way to represent any measurement that
    carries uncertainty.  Instead of storing a raw value (e.g. 124.0 BPM)
    and a separate confidence somewhere else, both travel together.

    Parameters
    ----------
    value : T
        The estimated value.
    confidence : float
        Confidence in the estimate in 0.0–1.0, where 1.0 means "certain."

    Raises
    ------
    ValueError
        If *confidence* is outside the [0, 1] range.

    Examples
    --------
    .. code-block:: python

        bpm = Estimate(value=124.0, confidence=0.92)
        print(bpm.value)        # 124.0
        print(bpm.confidence)   # 0.92
    """

    __slots__ = ("value", "confidence")

    def __init__(self, value: T, confidence: float) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {confidence}"
            )
        self.value = value
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"Estimate(value={self.value!r}, confidence={self.confidence})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Estimate):
            return NotImplemented
        return self.value == other.value and self.confidence == other.confidence

    def __hash__(self) -> int:
        return hash((self.value, self.confidence))