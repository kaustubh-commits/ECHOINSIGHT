"""
validate_comparison.py — Similarity Matrix Validation for Comparison Engine v1.

Generates a full pairwise similarity matrix, nearest-neighbour analysis,
dimension statistics, and distribution analysis.  Helps diagnose whether
the comparison engine is discriminative or suffering from feature collapse.

Usage:
    from src.analysis.validate_comparison import run_validation
    run_validation("data/dna")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.analysis.comparison import (
    DEFAULT_WEIGHTS,
    compare_songs_from_dicts,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEP: str = "=" * 60
_SUB_SEP: str = "-" * 60


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------


def _load_dataset(directory: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Load all SongDNA JSON files from *directory*.

    Returns
    -------
    List[Tuple[str, Dict]]
        ``(song_name, song_dict)`` pairs, sorted by song name.
    """
    data_path = Path(directory).resolve(strict=True)
    entries: List[Tuple[str, Dict[str, Any]]] = []

    for json_path in sorted(data_path.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                entry: Dict[str, Any] = json.load(fh)
            # Extract display name from filename metadata
            raw = entry.get("metadata", {}).get("filename", "")
            name = Path(raw).stem if raw else json_path.stem
            entries.append((name, entry))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", json_path.name, exc)

    return entries


def _similarity_matrix(
    entries: List[Tuple[str, Dict[str, Any]]],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str], List[List[Dict[str, float]]]]:
    """Build the NxN pairwise similarity matrix with dimension breakdowns.

    Parameters
    ----------
    entries : list of (name, dict)
        SongDNA entries.
    weights : dict or None
        Per-dimension weights (defaults to comparison DEFAULT_WEIGHTS).

    Returns
    -------
    overall_matrix : np.ndarray
        NxN matrix of overall similarity scores.
    names : List[str]
        Song names in matrix order.
    dim_matrices : List[List[Dict[str, float]]]
        NxN matrix of per-dimension scores.
        ``dim_matrices[i][j]`` is a dict like
        ``{"tempo": 0.91, "timbre": 0.82, "harmonic": 0.88, "energy": 0.74}``.
    """
    n = len(entries)
    names = [e[0] for e in entries]
    overall_matrix = np.zeros((n, n), dtype=np.float64)
    dim_matrices: List[List[Optional[Dict[str, float]]]] = [
        [None] * n for _ in range(n)
    ]

    for i in range(n):
        overall_matrix[i, i] = 1.0
        dim_matrices[i][i] = {
            "tempo": 1.0,
            "timbre": 1.0,
            "harmonic": 1.0,
            "energy": 1.0,
        }

        for j in range(i + 1, n):
            result = compare_songs_from_dicts(
                entries[i][1], entries[j][1], weights=weights
            )
            score = result.overall_similarity
            overall_matrix[i, j] = score
            overall_matrix[j, i] = score

            dim_scores: Dict[str, float] = {}
            for d in result.dimensions:
                dim_scores[d.name] = d.score
            dim_matrices[i][j] = dim_scores
            dim_matrices[j][i] = dim_scores

    return overall_matrix, names, dim_matrices


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def _print_matrix(
    matrix: np.ndarray, names: List[str], label: str = "Overall Similarity"
) -> None:
    """Print a formatted similarity matrix."""
    n = len(names)
    # Truncate names to fit
    short_names = [n[:18] for n in names]
    col_width = max(len(s) for s in short_names) + 2

    print(f"\n{label}:")
    print(_SUB_SEP)

    # Header
    header = " " * col_width
    for s in short_names:
        header += f"{s:>{col_width}s}"
    print(header)

    # Rows
    for i in range(n):
        row = f"{short_names[i]:<{col_width}s}"
        for j in range(n):
            row += f"{matrix[i, j]:>{col_width}.4f}"
        print(row)

    print()


def _print_nearest_farthest(
    matrix: np.ndarray,
    names: List[str],
) -> None:
    """Print nearest and farthest neighbour for each song."""
    n = len(names)
    print("Nearest and Farthest Neighbours:")
    print(_SUB_SEP)

    for i in range(n):
        scores = [(j, matrix[i, j]) for j in range(n) if j != i]
        scores.sort(key=lambda x: x[1], reverse=True)
        nearest = scores[0]
        farthest = scores[-1]

        print(f"\n  {names[i]}:")
        print(f"    Nearest:  {names[nearest[0]]} ({nearest[1]:.4f})")
        print(f"    Farthest: {names[farthest[0]]} ({farthest[1]:.4f})")

    print()


def _print_dimension_statistics(
    dim_matrices: List[List[Dict[str, float]]],
    names: List[str],
) -> None:
    """Print per-dimension average, min, max across all unique pairs."""
    n = len(names)
    dim_names = ["tempo", "timbre", "harmonic", "energy"]
    stats: Dict[str, List[float]] = {d: [] for d in dim_names}

    for i in range(n):
        for j in range(i + 1, n):
            scores = dim_matrices[i][j]
            if scores is not None:
                for d in dim_names:
                    stats[d].append(scores.get(d, 0.0))

    print("Per-Dimension Statistics (across all unique pairs):")
    print(_SUB_SEP)
    header = f"{'Dimension':<15s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Spread':>8s}"
    print(header)
    print("-" * len(header))

    for d in dim_names:
        vals = stats[d]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        spread = max_v - min_v
        print(
            f"{d:<15s} {mean_v:>8.4f} {std_v:>8.4f} {min_v:>8.4f} {max_v:>8.4f} {spread:>8.4f}"
        )

    print()


def _print_distribution_analysis(
    matrix: np.ndarray,
    names: List[str],
) -> None:
    """Print similarity distribution statistics across all unique pairs."""
    n = len(names)
    pairs: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(matrix[i, j])

    arr = np.array(pairs)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    spread = max_v - min_v

    # Histogram-like binning
    bins = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    counts, _ = np.histogram(arr, bins=bins)

    print("Similarity Distribution Analysis:")
    print(_SUB_SEP)
    print(f"  Total pairs:       {len(pairs)}")
    print(f"  Mean similarity:   {mean_v:.4f}")
    print(f"  Std deviation:     {std_v:.4f}")
    print(f"  Min similarity:    {min_v:.4f}")
    print(f"  Max similarity:    {max_v:.4f}")
    print(f"  Spread (max-min):  {spread:.4f}")
    print(f"  Coefficient of variation (std/mean): {std_v / mean_v:.4f}")
    print()

    print("  Distribution histogram:")
    for i in range(len(bins) - 1):
        pct = counts[i] / len(pairs) * 100 if len(pairs) > 0 else 0
        bar = "#" * int(pct / 2)
        print(
            f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {counts[i]:3d} songs ({pct:5.1f}%) {bar}"
        )

    print()
    # Diagnostic message
    if spread < 0.15:
        print("  ⚠ DIAGNOSTIC: Narrow spread detected (spread < 0.15).")
        print("    All songs cluster in a tight similarity band.")
        print("    Possible causes:")
        print("      - Songs are genuinely very similar (same album, same genre)")
        print("      - Global MFCC averaging is not discriminative enough")
        print("      - Single-album dataset; spread should increase with diverse songs")
    elif spread < 0.30:
        print("  ℹ Moderate spread. Reasonable for a single-album dataset.")
    else:
        print("  ✅ Good spread. Metric appears discriminative.")

    print()

    if std_v < 0.05:
        print("  ⚠ DIAGNOSTIC: Very low standard deviation (< 0.05).")
        print("    All songs are scored within a narrow range.")
        print(
            "    This suggests feature collapse — the metric cannot distinguish songs."
        )
    elif std_v < 0.10:
        print("  ℹ Low standard deviation. Expected for same-album comparisons.")
    else:
        print("  ✅ Healthy standard deviation. Good discrimination.")

    print()


def _print_dimension_breakdown(
    dim_matrices: List[List[Dict[str, float]]],
    names: List[str],
) -> None:
    """Print full dimension breakdown for every pair."""
    n = len(names)
    dim_names = ["tempo", "timbre", "harmonic", "energy"]

    print("Full Pairwise Dimension Breakdown:")
    print(_SUB_SEP)

    for i in range(n):
        for j in range(i + 1, n):
            scores = dim_matrices[i][j]
            if scores is None:
                continue
            overall = np.mean(list(scores.values()))  # unweighted mean for display
            print(f"\n  {names[i]}  vs  {names[j]}  (unweighted mean: {overall:.4f})")
            for d in dim_names:
                print(f"    {d:<12s}  {scores.get(d, 0.0):.4f}")

    print()


def _print_investigation(
    dim_matrices: List[List[Dict[str, float]]], names: List[str]
) -> None:
    """Diagnose whether timbre or harmonic similarity is inflated."""
    n = len(names)
    all_timbre: List[float] = []
    all_harmonic: List[float] = []
    all_tempo: List[float] = []
    all_energy: List[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            scores = dim_matrices[i][j]
            if scores is not None:
                all_timbre.append(scores.get("timbre", 0.0))
                all_harmonic.append(scores.get("harmonic", 0.0))
                all_tempo.append(scores.get("tempo", 0.0))
                all_energy.append(scores.get("energy", 0.0))

    print("Investigation: Is Timbre or Harmonic Similarity Inflated?")
    print(_SUB_SEP)

    def _analyze_dim(name: str, vals: List[float], threshold: float = 0.95) -> None:
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        min_v = float(np.min(vals))
        n_above = sum(1 for v in vals if v >= threshold)
        pct_above = n_above / len(vals) * 100 if vals else 0
        print(f"\n  {name}:")
        print(f"    Mean:     {mean_v:.4f}")
        print(f"    Std:      {std_v:.4f}")
        print(f"    Min:      {min_v:.4f}")
        print(f"    Max:      {float(np.max(vals)):.4f}")
        print(f"    % >= {threshold:.2f}:  {pct_above:.1f}% ({n_above}/{len(vals)})")
        if pct_above > 80:
            print(
                f"    ⚠ WARNING: {pct_above:.0f}% of pairs score above {threshold:.2f}."
            )
            print("      This dimension may not be discriminative for this dataset.")
            print("      Possible mitigations:")
            print("        - Use per-beat trajectories (FrameGrid + DTW)")
            print("        - Reduce this dimension's weight")
            print("        - Add complementary features (spectral contrast, etc.)")

    _analyze_dim("Timbre (MFCC cosine)", all_timbre)
    _analyze_dim("Harmonic (chroma cosine)", all_harmonic)
    _analyze_dim("Tempo (ratio)", all_tempo, threshold=0.90)
    _analyze_dim("Energy (RMS ratio)", all_energy, threshold=0.90)

    print()
    print("Correlation between timbre and harmonic scores:")
    if all_timbre and all_harmonic:
        corr = float(np.corrcoef(all_timbre, all_harmonic)[0, 1])
        print(f"  Pearson r = {corr:.4f}")
        if corr > 0.8:
            print("  ⚠ WARNING: Timbre and harmonic scores are highly correlated.")
            print("    This suggests they are measuring a shared underlying property")
            print("    (e.g., production style from the same album) rather than")
            print("    independent musical dimensions.")
            print(
                "    Mitigation: DTW on separate feature trajectories may decorrelate them."
            )
        elif corr > 0.5:
            print("  ℹ Moderate correlation. Expected for same-album comparisons.")
        else:
            print("  ✅ Low correlation. Dimensions are reasonably independent.")
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_validation(
    directory: str = "data/dna",
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run full similarity matrix validation on a SongDNA dataset.

    Parameters
    ----------
    directory : str
        Path to SongDNA JSON files.
    weights : dict or None
        Per-dimension weights.
    verbose : bool
        If True, print all reports to stdout.

    Returns
    -------
    Dict[str, Any]
        Summary statistics:
        - n_songs: int
        - mean_similarity: float
        - std_similarity: float
        - min_similarity: float
        - max_similarity: float
        - spread: float
        - dim_means: Dict[str, float]
        - dim_stds: Dict[str, float]
    """
    entries = _load_dataset(directory)
    if len(entries) < 2:
        logger.warning("Need at least 2 songs for validation, got %d", len(entries))
        return {"n_songs": len(entries)}

    w = weights or DEFAULT_WEIGHTS
    matrix, names, dim_matrices = _similarity_matrix(entries, weights=w)

    # Collect stats
    n = len(names)
    all_pairs: List[float] = []
    dims_collected: Dict[str, List[float]] = {
        "tempo": [],
        "timbre": [],
        "harmonic": [],
        "energy": [],
    }
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append(matrix[i, j])
            if dim_matrices[i][j] is not None:
                for d in dims_collected:
                    dims_collected[d].append(dim_matrices[i][j].get(d, 0.0))

    arr = np.array(all_pairs)
    stats: Dict[str, Any] = {
        "n_songs": n,
        "mean_similarity": float(np.mean(arr)),
        "std_similarity": float(np.std(arr)),
        "min_similarity": float(np.min(arr)),
        "max_similarity": float(np.max(arr)),
        "spread": float(np.max(arr) - np.min(arr)),
        "dim_means": {d: float(np.mean(v)) for d, v in dims_collected.items()},
        "dim_stds": {d: float(np.std(v)) for d, v in dims_collected.items()},
    }

    if verbose:
        print(_SEP)
        print("EchoInsight — Similarity Matrix Validation")
        print(f"Dataset: {directory}  ({len(entries)} songs)")
        print(_SEP)

        _print_matrix(matrix, names, "Overall Similarity Matrix")
        _print_nearest_farthest(matrix, names)
        _print_dimension_statistics(dim_matrices, names)
        _print_distribution_analysis(matrix, names)
        _print_investigation(dim_matrices, names)
        _print_dimension_breakdown(dim_matrices, names)

        print(_SEP)
        print("Validation Summary")
        print(_SEP)
        print(f"  Songs:                {stats['n_songs']}")
        print(f"  Mean similarity:      {stats['mean_similarity']:.4f}")
        print(f"  Std similarity:       {stats['std_similarity']:.4f}")
        print(f"  Min similarity:       {stats['min_similarity']:.4f}")
        print(f"  Max similarity:       {stats['max_similarity']:.4f}")
        print(f"  Spread (max-min):     {stats['spread']:.4f}")
        print()
        for d in ["tempo", "timbre", "harmonic", "energy"]:
            print(
                f"  Dim mean — {d:<12s}: {stats['dim_means'][d]:.4f}  (std: {stats['dim_stds'][d]:.4f})"
            )
        print(_SEP)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_validation()
