"""
comparison.py — SongDNA comparison engine v1.

Compares two SongDNA fingerprints across four acoustic dimensions using
deterministic metrics only.  Every score is explainable and traceable to
specific musical features.

Usage:
    from src.analysis.comparison import compare_songs
    result = compare_songs(song_a, song_b)
    print(result.explanation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.core.dna_schema import (
    IntelligenceDNA,
    RhythmDNA,
    SongDNA,
    SongMetadata,
    StemDNA,
    StructureDNA,
    TimbreDNA,
    TonalDNA,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single comparison dimension.

    Attributes
    ----------
    name : str
        Dimension name (``"tempo"``, ``"timbre"``, etc.).
    score : float
        Normalised similarity in 0.0–1.0 (1.0 = identical).
    weight : float
        Contribution weight for this dimension.
    contribution : float
        ``score * weight`` — the weighted contribution to the overall score.
    explanation : str
        Human-readable explanation of this dimension's score.
    """

    name: str
    score: float
    weight: float
    contribution: float
    explanation: str


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing two SongDNA fingerprints.

    Attributes
    ----------
    song_a_name : str
        Display name for the first song.
    song_b_name : str
        Display name for the second song.
    overall_similarity : float
        Weighted composite similarity in 0.0–1.0.
    dimensions : List[DimensionScore]
        Per-dimension breakdown of the comparison.
    strongest_dimension : str
        Name of the dimension that contributed most to the overall score.
    weakest_dimension : str
        Name of the dimension that contributed least.
    explanation : str
        Full human-readable summary of the comparison.
    """

    song_a_name: str
    song_b_name: str
    overall_similarity: float
    dimensions: List[DimensionScore] = field(default_factory=list)
    strongest_dimension: str = ""
    weakest_dimension: str = ""
    explanation: str = ""


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "tempo": 0.28,
    "timbre": 0.35,
    "harmonic": 0.28,
    "energy": 0.09,
}

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _ratio_similarity(a: float, b: float) -> float:
    """Ratio-based similarity in 0.0–1.0."""
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity rescaled to 0.0–1.0."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    cos = float(np.dot(arr_a, arr_b) / (norm_a * norm_b))
    # Clamp to [-1, 1] then rescale to [0, 1]
    return max(0.0, (1.0 + max(-1.0, min(1.0, cos))) / 2.0)


def _display_name(entry: SongDNA) -> str:
    """Extract a human-readable song name from a SongDNA."""
    raw = entry.metadata.filename
    if raw:
        import pathlib

        return pathlib.Path(raw).stem
    return "Unknown"


# ---------------------------------------------------------------------------
# Explainability helpers
# ---------------------------------------------------------------------------


def _explain_tempo(tempo_a: float, tempo_b: float, score: float) -> str:
    if score >= 0.95:
        return f"Nearly identical tempo ({tempo_a:.1f} vs {tempo_b:.1f} BPM)."
    if score >= 0.80:
        return f"Both songs fall in a similar tempo range ({tempo_a:.1f} vs {tempo_b:.1f} BPM)."
    if score >= 0.50:
        return f"Tempos differ noticeably ({tempo_a:.1f} vs {tempo_b:.1f} BPM)."
    return f"Very different tempos ({tempo_a:.1f} vs {tempo_b:.1f} BPM)."


def _explain_timbre(score: float) -> str:
    if score >= 0.90:
        return "Nearly identical spectral texture — instruments and production style are very similar."
    if score >= 0.70:
        return "MFCC profiles are fairly close — similar spectral texture."
    if score >= 0.40:
        return "Noticeable timbral differences — likely different instrumentation or production styles."
    return "Very different timbral character — fundamentally different instrumentation or mix."


def _explain_harmonic(key_a: Optional[str], key_b: Optional[str], score: float) -> str:
    if score >= 0.90:
        return f"Almost identical harmonic profile ({key_a or 'unknown'} vs {key_b or 'unknown'})."
    if score >= 0.70:
        return f"Similar harmonic character — related tonal centres ({key_a or 'unknown'} vs {key_b or 'unknown'})."
    if score >= 0.40:
        return (
            f"Moderate harmonic overlap ({key_a or 'unknown'} vs {key_b or 'unknown'})."
        )
    return f"Very different harmonic structure ({key_a or 'unknown'} vs {key_b or 'unknown'})."


def _explain_energy(rms_a: float, rms_b: float, score: float) -> str:
    if score >= 0.90:
        return "Nearly identical energy levels."
    if score >= 0.70:
        return "Similar energy levels."
    if score >= 0.40:
        return "Noticeable energy difference."
    return "Very different energy levels."


def _generate_overall_explanation(r: ComparisonResult, name_a: str, name_b: str) -> str:
    parts: List[str] = []

    # Overall similarity level
    if r.overall_similarity >= 0.80:
        parts.append(
            f"{name_a} and {name_b} are very similar (score: {r.overall_similarity:.2f})."
        )
    elif r.overall_similarity >= 0.55:
        parts.append(
            f"{name_a} and {name_b} are moderately similar (score: {r.overall_similarity:.2f})."
        )
    else:
        parts.append(
            f"{name_a} and {name_b} are not very similar (score: {r.overall_similarity:.2f})."
        )

    # Driving dimensions (contribution > 0.08)
    strong = [d for d in r.dimensions if d.contribution >= 0.08]
    if strong:
        desc = "; ".join(f"{d.name} ({d.contribution:.3f})" for d in strong)
        parts.append(f"Primary drivers: {desc}.")

    # Weak dimensions (score < 0.5)
    weak = [d for d in r.dimensions if d.score < 0.5]
    if weak:
        desc = ", ".join(f"{d.name} ({d.score:.2f})" for d in weak)
        parts.append(f"Areas of difference: {desc}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Dimension computation
# ---------------------------------------------------------------------------


def _compute_tempo(a: SongDNA, b: SongDNA, weight: float) -> DimensionScore:
    t_a = a.rhythm.tempo
    t_b = b.rhythm.tempo
    score = _ratio_similarity(t_a, t_b)
    return DimensionScore(
        name="tempo",
        score=round(score, 4),
        weight=weight,
        contribution=round(score * weight, 4),
        explanation=_explain_tempo(t_a, t_b, score),
    )


def _compute_timbre(a: SongDNA, b: SongDNA, weight: float) -> DimensionScore:
    mfcc_a = a.timbre.mfcc_mean
    mfcc_b = b.timbre.mfcc_mean
    score = _cosine_similarity(mfcc_a, mfcc_b)
    return DimensionScore(
        name="timbre",
        score=round(score, 4),
        weight=weight,
        contribution=round(score * weight, 4),
        explanation=_explain_timbre(score),
    )


def _compute_harmonic(a: SongDNA, b: SongDNA, weight: float) -> DimensionScore:
    chroma_a = a.tonal.chroma_mean
    chroma_b = b.tonal.chroma_mean
    score = _cosine_similarity(chroma_a, chroma_b)
    return DimensionScore(
        name="harmonic",
        score=round(score, 4),
        weight=weight,
        contribution=round(score * weight, 4),
        explanation=_explain_harmonic(a.tonal.key, b.tonal.key, score),
    )


def _compute_energy(a: SongDNA, b: SongDNA, weight: float) -> DimensionScore:
    rms_a = a.timbre.rms_energy_mean
    rms_b = b.timbre.rms_energy_mean
    score = _ratio_similarity(rms_a, rms_b)
    return DimensionScore(
        name="energy",
        score=round(score, 4),
        weight=weight,
        contribution=round(score * weight, 4),
        explanation=_explain_energy(rms_a, rms_b, score),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_songs(
    a: SongDNA,
    b: SongDNA,
    weights: Optional[Dict[str, float]] = None,
) -> ComparisonResult:
    """Compare two SongDNA fingerprints across all active dimensions.

    Parameters
    ----------
    a : SongDNA
        First song fingerprint.
    b : SongDNA
        Second song fingerprint.
    weights : dict or None
        Per-dimension weights (defaults to :data:`DEFAULT_WEIGHTS`).

    Returns
    -------
    ComparisonResult
        Full comparison with per-dimension scores and explanation.
    """
    w = weights or DEFAULT_WEIGHTS

    name_a = _display_name(a)
    name_b = _display_name(b)

    dims: List[DimensionScore] = [
        _compute_tempo(a, b, w.get("tempo", 0.28)),
        _compute_timbre(a, b, w.get("timbre", 0.35)),
        _compute_harmonic(a, b, w.get("harmonic", 0.28)),
        _compute_energy(a, b, w.get("energy", 0.09)),
    ]

    overall = sum(d.contribution for d in dims)

    sorted_dims = sorted(dims, key=lambda d: d.contribution, reverse=True)
    strongest = sorted_dims[0].name if sorted_dims else ""
    weakest = sorted_dims[-1].name if sorted_dims else ""

    result = ComparisonResult(
        song_a_name=name_a,
        song_b_name=name_b,
        overall_similarity=round(overall, 4),
        dimensions=dims,
        strongest_dimension=strongest,
        weakest_dimension=weakest,
        explanation="",  # filled below
    )

    explanation = _generate_overall_explanation(result, name_a, name_b)

    # Swap in the full explanation
    object.__setattr__(result, "explanation", explanation)

    return result


def compare_songs_from_dicts(
    a: dict,
    b: dict,
    weights: Optional[Dict[str, float]] = None,
) -> ComparisonResult:
    """Compare two SongDNA dictionaries loaded from JSON.

    Convenience wrapper around :func:`compare_songs` that reconstructs
    SongDNA dataclasses from deserialised JSON data.

    Parameters
    ----------
    a, b : dict
        SongDNA dictionaries as produced by ``dataclasses.asdict()``.
    weights : dict or None
        Per-dimension weights.

    Returns
    -------
    ComparisonResult
    """

    def _reconstruct(cls, data: dict):
        """Reconstruct a frozen dataclass from a dict, ignoring extra keys."""
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    def _dict_to_songdna(data: dict) -> SongDNA:
        return SongDNA(
            metadata=_reconstruct(SongMetadata, data.get("metadata", {})),
            rhythm=_reconstruct(RhythmDNA, data.get("rhythm", {})),
            timbre=_reconstruct(TimbreDNA, data.get("timbre", {})),
            tonal=_reconstruct(TonalDNA, data.get("tonal", {})),
            structure=_reconstruct(StructureDNA, data.get("structure", {})),
            intelligence=_reconstruct(IntelligenceDNA, data.get("intelligence", {})),
            stem=_reconstruct(StemDNA, data.get("stem", {})),
        )

    return compare_songs(_dict_to_songdna(a), _dict_to_songdna(b), weights=weights)


def compare_by_files(
    path_a: str,
    path_b: str,
    weights: Optional[Dict[str, float]] = None,
) -> ComparisonResult:
    """Load two SongDNA JSON files and compare them.

    Parameters
    ----------
    path_a, path_b : str
        Paths to SongDNA ``.json`` files.
    weights : dict or None
        Per-dimension weights.

    Returns
    -------
    ComparisonResult
    """
    import json

    with open(path_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    return compare_songs_from_dicts(data_a, data_b, weights=weights)


def format_report(result: ComparisonResult) -> str:
    """Format a :class:`ComparisonResult` as a human-readable string.

    Parameters
    ----------
    result : ComparisonResult
        The comparison result to format.

    Returns
    -------
    str
        Formatted report.
    """
    sep = "=" * 50
    lines: List[str] = [
        sep,
        "EchoInsight Song Comparison",
        sep,
        f"\n{result.song_a_name}  vs  {result.song_b_name}",
        f"\nOverall Similarity:  {result.overall_similarity:.4f}",
        f"Strongest dimension: {result.strongest_dimension}",
        f"Weakest dimension:   {result.weakest_dimension}",
        sep + "\n",
    ]

    for dim in result.dimensions:
        lines.append(
            f"  {dim.name:<12s}  {dim.score:.4f}  "
            f"(weight: {dim.weight:.2f}, contrib: {dim.contribution:.4f})"
        )
        lines.append(f"  {'':12s}  {dim.explanation}")

    lines.append("")
    lines.append(sep)
    lines.append("\nExplanation:")
    lines.append(f"  {result.explanation}")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)
