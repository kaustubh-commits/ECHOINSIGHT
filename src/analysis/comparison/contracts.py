"""
contracts.py — Comparison engine data contracts.

This module defines the **output types** produced by the comparison engine.
These types live in ``src.analysis`` (not ``src.core``) because they are
results of computation, not domain primitives.  The domain layer
(``SongDNA``, ``SegmentDNA``, etc.) has no knowledge of comparison.

The comparison engine consumes ``SongDNA`` and ``SegmentDNA`` from
``src.core`` and produces ``ComparisonResult``, ``SegmentMatch``, and
related types.

Layer rules
-----------
- ``src.analysis.comparison.contracts`` may import from ``src.core``.
- It must **never** import from ``src.app`` or ``src.config``.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

from src.core.identifiers import SongID
from src.core.structure import SegmentDNA

# ---------------------------------------------------------------------------
# Dimension-level types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ComparisonDimension:
    """Score for a single comparison dimension (e.g. tempo, timbre).

    A comparison is decomposed into multiple dimensions, each of which
    measures similarity along a specific musical axis.  The overall
    similarity is the weighted sum of dimension scores.

    Attributes
    ----------
    dimension_id : str
        Machine-readable identifier, e.g. ``"tempo"``, ``"harmonic_profile"``,
        ``"structural_flow"``.
    label : str
        Human-readable label, e.g. ``"Tempo"``, ``"Harmonic Profile"``.
    score : float
        Normalised similarity in 0.0–1.0, where 1.0 = identical.
    weight : float
        Contribution weight for aggregating to the overall score.
    contribution : float
        ``score * weight`` — the weighted contribution to the overall score.
    evidence : str
        Human-readable explanation of why this dimension scored as it did.
        This is the **explainability layer** — it should answer "why is
        this number what it is?"
    """

    dimension_id: str
    label: str
    score: float
    weight: float
    contribution: float
    evidence: str

    def __post_init__(self) -> None:
        if not self.dimension_id.strip():
            raise ValueError("dimension_id must be non-empty")
        if not self.label.strip():
            raise ValueError("label must be non-empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"score must be in [0, 1], got {self.score}"
            )
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")
        if not self.evidence.strip():
            raise ValueError("evidence must be non-empty")


@dataclasses.dataclass(frozen=True)
class ComparisonDifference:
    """A specific point of difference between two compared entities.

    Differences highlight *where* two songs or segments diverge, providing
    actionable information beyond a single similarity score.

    Attributes
    ----------
    feature : str
        The feature that differs, e.g. ``"segment_count"``, ``"average_tempo"``,
        ``"key"``.
    value_a : float | str
        The value of the feature in the first entity.
    value_b : float | str
        The value of the feature in the second entity.
    delta : float
        Normalised difference in 0.0–1.0.  0.0 = identical, 1.0 = completely
        different along this feature.
    significance : float
        How much this difference matters, in 0.0–1.0.  A large delta in a
        musically irrelevant feature has low significance.
    """

    feature: str
    value_a: float | str
    value_b: float | str
    delta: float
    significance: float

    def __post_init__(self) -> None:
        if not self.feature.strip():
            raise ValueError("feature must be non-empty")
        if not 0.0 <= self.delta <= 1.0:
            raise ValueError(
                f"delta must be in [0, 1], got {self.delta}"
            )
        if not 0.0 <= self.significance <= 1.0:
            raise ValueError(
                f"significance must be in [0, 1], got {self.significance}"
            )


# ---------------------------------------------------------------------------
# Top-level comparison results
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing two entities (songs, segments, etc.).

    ``ComparisonResult`` is the primary output of the comparison engine.
    It contains the overall similarity score, a breakdown by dimension,
    specific differences, and human-readable explanations.

    Attributes
    ----------
    entity_a_id : str
        Identifier of the first entity (``song_id`` or
        ``song_id + segment_index``).
    entity_b_id : str
        Identifier of the second entity.
    entity_a_label : str
        Human-readable name for the first entity (song name or
        ``"Song Name — Chorus"``).
    entity_b_label : str
        Human-readable name for the second entity.
    comparison_type : str
        Type of comparison, e.g. ``"song_song"``, ``"segment_segment"``,
        ``"song_library"``.
    overall_score : float
        Weighted composite similarity in 0.0–1.0.
    dimensions : Tuple[ComparisonDimension, ...]
        Per-dimension breakdown of the comparison.  At least one dimension
        is always present.
    differences : Tuple[ComparisonDifference, ...]
        Specific points of difference.  May be empty when entities are
        nearly identical.
    strongest_dimension : str
        ``dimension_id`` of the dimension that contributed most to the
        overall score.
    weakest_dimension : str
        ``dimension_id`` of the dimension that contributed least.
    summary : str
        One-line summary of the comparison result.
    explanation : str
        Full human-readable explanation of the comparison.
    """

    entity_a_id: str
    entity_b_id: str
    entity_a_label: str
    entity_b_label: str
    comparison_type: str
    overall_score: float
    dimensions: Tuple[ComparisonDimension, ...]
    differences: Tuple[ComparisonDifference, ...] = ()
    strongest_dimension: str = ""
    weakest_dimension: str = ""
    summary: str = ""
    explanation: str = ""

    def __post_init__(self) -> None:
        if not self.entity_a_id.strip():
            raise ValueError("entity_a_id must be non-empty")
        if not self.entity_b_id.strip():
            raise ValueError("entity_b_id must be non-empty")
        if not self.comparison_type.strip():
            raise ValueError("comparison_type must be non-empty")
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError(
                f"overall_score must be in [0, 1], "
                f"got {self.overall_score}"
            )
        if not self.dimensions:
            raise ValueError("dimensions must be non-empty")


# ---------------------------------------------------------------------------
# Segment match types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SegmentMatch:
    """A single segment returned as a match from a similarity search.

    ``SegmentMatch`` wraps a ``SegmentDNA`` with match metadata: the
    similarity score, which dimensions contributed to the match, and
    a human-readable evidence string.  The ``song_id`` and
    ``segment_index`` in the embedded ``SegmentDNA`` are sufficient for
    the caller to reconstruct the full context.

    Attributes
    ----------
    song_id : SongID
        Identifier of the song the segment belongs to.
    song_label : str
        Human-readable song name.
    segment : SegmentDNA
        The matched segment.  Contains all timing, DSP, and context
        information.
    similarity_score : float
        Overall similarity of this match in 0.0–1.0.
    matching_dimensions : Tuple[str, ...]
        Dimension IDs that contributed most to the match.
    evidence : str
        Human-readable explanation of why this segment was returned.
    """

    song_id: SongID
    song_label: str
    segment: SegmentDNA
    similarity_score: float
    matching_dimensions: Tuple[str, ...]
    evidence: str

    def __post_init__(self) -> None:
        if not self.song_id:
            raise ValueError("song_id must be non-empty")
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(
                f"similarity_score must be in [0, 1], "
                f"got {self.similarity_score}"
            )
        if not self.matching_dimensions:
            raise ValueError("matching_dimensions must be non-empty")
        if not self.evidence.strip():
            raise ValueError("evidence must be non-empty")


@dataclasses.dataclass(frozen=True)
class SearchResults:
    """Results of searching for similar segments.

    ``SearchResults`` wraps a list of ``SegmentMatch`` values with
    metadata about the search itself: the query segment, how many
    candidates were searched, and how long the search took.

    Attributes
    ----------
    query_song_id : SongID
        The song the query segment belongs to.
    query_segment : SegmentDNA
        The segment that was used as the query.
    results : Tuple[SegmentMatch, ...]
        Ranked matches, ordered by ``similarity_score`` descending.
    total_candidates : int
        Total number of candidates that were searched.
    search_time_ms : float
        Wall-clock time for the search in milliseconds.
    """

    query_song_id: SongID
    query_segment: SegmentDNA
    results: Tuple[SegmentMatch, ...]
    total_candidates: int
    search_time_ms: float

    def __post_init__(self) -> None:
        if not self.query_song_id:
            raise ValueError("query_song_id must be non-empty")
        if self.total_candidates < 0:
            raise ValueError(
                f"total_candidates must be >= 0, "
                f"got {self.total_candidates}"
            )
        if self.search_time_ms < 0:
            raise ValueError(
                f"search_time_ms must be >= 0, "
                f"got {self.search_time_ms}"
            )