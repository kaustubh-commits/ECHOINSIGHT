"""
engine.py — Raw metric comparison engine.

Compares two v2 ``SongDNA`` aggregates and produces **raw metrics only**.
No scoring.  No recommendations.  No interpretation.

The report exposes four deterministic differences between two songs:

- ``tempo_difference``    — absolute BPM difference
- ``key_relationship``    — Camelot wheel circular distance (0 = same key)
- ``duration_difference`` — absolute duration difference in seconds
- ``energy_difference``   — normalised RMS energy difference in [-1.0, 1.0]

Every value is traceable to DSP estimates captured in ``SongSummary`` and
``SongMetadata``.

Usage:
    from src.analysis.comparison import compare
    report = compare(song_a, song_b)
    print(report.tempo_difference)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional

from src.core.song_dna import SongDNA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camelot wheel
# ---------------------------------------------------------------------------

# Camelot wheel numbers for each pitch class, per mode.
# Majors (B): C=8, C#=3, D=10, D#=5, E=12, F=7, F#=2, G=9, G#=4, A=11, A#=6, B=1
# Minors (A): A=8, A#=3, B=10, C=5, C#=12, D=7, D#=2, E=9, F=4, F#=11, G=6, G#=1
_MAJOR_CAMELOT: Dict[str, int] = {
    "C": 8, "C#": 3, "D": 10, "D#": 5, "E": 12, "F": 7,
    "F#": 2, "G": 9, "G#": 4, "A": 11, "A#": 6, "B": 1,
}

_MINOR_CAMELOT: Dict[str, int] = {
    "A": 8, "A#": 3, "B": 10, "C": 5, "C#": 12, "D": 7,
    "D#": 2, "E": 9, "F": 4, "F#": 11, "G": 6, "G#": 1,
}

_UNKNOWN_KEY_DISTANCE: int = 12
"""Value returned for ``key_relationship`` when either key is unknown."""


def _camelot_number(key: Optional[str]) -> Optional[int]:
    """Convert a key label (e.g. ``"C major"``, ``"A minor"``) to a
    Camelot wheel number 1..12.

    Returns ``None`` when the key is unknown or unparsable.
    """
    if not key:
        return None
    parts = key.strip().split()
    if not parts:
        return None
    root = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else ""
    if mode.startswith("maj"):
        return _MAJOR_CAMELOT.get(root)
    if mode.startswith("min"):
        return _MINOR_CAMELOT.get(root)
    # No mode → default to the major wheel (relative minor shares the
    # same wheel number, so this remains harmonically correct).
    return _MAJOR_CAMELOT.get(root)


def _camelot_distance(num_a: Optional[int], num_b: Optional[int]) -> int:
    """Circular distance between two Camelot wheel numbers.

    Returns ``_UNKNOWN_KEY_DISTANCE`` when either number is missing.
    """
    if num_a is None or num_b is None:
        return _UNKNOWN_KEY_DISTANCE
    diff = abs(num_a - num_b)
    return min(diff, 12 - diff)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ComparisonReport:
    """Raw metric differences between two songs.

    All fields are **facts** — deliberately uninterpreted.  No similarity
    score, no weighting, no recommendation is computed here.
    """

    song_a_name: str
    song_b_name: str
    tempo_difference: float
    key_relationship: int
    duration_difference: float
    energy_difference: float

    def __post_init__(self) -> None:
        if self.tempo_difference < 0:
            raise ValueError(
                f"tempo_difference must be >= 0, "
                f"got {self.tempo_difference}"
            )
        if not 0 <= self.key_relationship <= 12:
            raise ValueError(
                f"key_relationship must be in [0, 12], "
                f"got {self.key_relationship}"
            )
        if self.duration_difference < 0:
            raise ValueError(
                f"duration_difference must be >= 0, "
                f"got {self.duration_difference}"
            )
        if not -1.0 <= self.energy_difference <= 1.0:
            raise ValueError(
                f"energy_difference must be in [-1, 1], "
                f"got {self.energy_difference}"
            )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _display_name(dna: SongDNA) -> str:
    """Extract a human-readable song name from a SongDNA."""
    raw = dna.metadata.filename
    if raw:
        import pathlib

        return pathlib.Path(raw).stem
    return "Unknown"


def _tempo_difference(a: SongDNA, b: SongDNA) -> float:
    """Absolute BPM difference between the two DSP tempo estimates."""
    return abs(a.summary.tempo_estimate - b.summary.tempo_estimate)


def _key_relationship(a: SongDNA, b: SongDNA) -> int:
    """Camelot wheel circular distance (0 = same key)."""
    return _camelot_distance(
        _camelot_number(a.summary.key_estimate),
        _camelot_number(b.summary.key_estimate),
    )


def _duration_difference(a: SongDNA, b: SongDNA) -> float:
    """Absolute duration difference in seconds."""
    return abs(a.metadata.duration - b.metadata.duration)


def _energy_difference(a: SongDNA, b: SongDNA) -> float:
    """Normalised RMS energy difference in [-1.0, 1.0].

    ``(rms_a - rms_b) / max(rms_a, rms_b)``.  Zero when both are zero.
    """
    rms_a = a.summary.rms_energy_mean
    rms_b = b.summary.rms_energy_mean
    denom = max(rms_a, rms_b)
    if denom <= 1e-10:
        return 0.0
    return max(-1.0, min(1.0, (rms_a - rms_b) / denom))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare(song_a: SongDNA, song_b: SongDNA) -> ComparisonReport:
    """Compute raw metric differences between two songs.

    No scoring.  No recommendations.  Just the four facts defined on
    :class:`ComparisonReport`.

    Parameters
    ----------
    song_a : SongDNA
        First v2 song aggregate.
    song_b : SongDNA
        Second v2 song aggregate.

    Returns
    -------
    ComparisonReport
        Raw metric differences between the two songs.
    """
    return ComparisonReport(
        song_a_name=_display_name(song_a),
        song_b_name=_display_name(song_b),
        tempo_difference=_tempo_difference(song_a, song_b),
        key_relationship=_key_relationship(song_a, song_b),
        duration_difference=_duration_difference(song_a, song_b),
        energy_difference=_energy_difference(song_a, song_b),
    )


def compare_from_dicts(a: dict, b: dict) -> ComparisonReport:
    """Compute a :class:`ComparisonReport` from two SongDNA dicts.

    Convenience wrapper around :func:`compare` that reconstructs v2
    ``SongDNA`` dataclasses from deserialised JSON data produced by
    ``dataclasses.asdict`` / ``SongDNA.to_json``.
    """
    import json

    return compare(
        SongDNA.from_json(json.dumps(a)),
        SongDNA.from_json(json.dumps(b)),
    )


def compare_by_files(path_a: str, path_b: str) -> ComparisonReport:
    """Load two SongDNA JSON files and compute the raw comparison report.

    Parameters
    ----------
    path_a, path_b : str
        Paths to v2 SongDNA ``.json`` files (produced by ``to_json``).

    Returns
    -------
    ComparisonReport
    """
    import json
    import pathlib

    with pathlib.Path(path_a).open("r", encoding="utf-8") as fh:
        data_a = json.load(fh)
    with pathlib.Path(path_b).open("r", encoding="utf-8") as fh:
        data_b = json.load(fh)
    return compare_from_dicts(data_a, data_b)


def format_report(report: ComparisonReport) -> str:
    """Format a :class:`ComparisonReport` as a human-readable string.

    Parameters
    ----------
    report : ComparisonReport
        The comparison report to format.

    Returns
    -------
    str
        Formatted report.
    """
    sep = "=" * 50
    lines: List[str] = [
        sep,
        "EchoInsight — Raw Metric Comparison Report",
        sep,
        f"\n{report.song_a_name}  vs  {report.song_b_name}",
        f"\nTempo difference:     {report.tempo_difference:.2f} BPM",
        f"Key relationship:     {report.key_relationship} "
        "(Camelot wheel distance; 0 = same key)",
        f"Duration difference:  {report.duration_difference:.2f} s",
        f"Energy difference:    {report.energy_difference:+.3f} "
        "[-1.0, 1.0; positive = A more energetic]",
        "",
        "Raw metrics only — no scoring, no recommendation.",
        sep,
    ]
    return "\n".join(lines)