"""
metrics.py — Evaluation metrics for structural segmentation.

Functions to compute boundary precision/recall, boundary distance, segment
length statistics, and other quantitative diagnostics for the v2
segmentation pipeline.

Layer rules
-----------
- ``src.analysis.evaluation`` may import from ``src.core``.
- It must **never** import from ``src.app`` or ``src.config``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Boundary metrics
# ---------------------------------------------------------------------------


def boundary_distance(
    detected: Sequence[float],
    reference: Sequence[float],
) -> float:
    """Mean minimum distance (seconds) from each detected boundary to the
    closest reference boundary, and vice versa.

    Parameters
    ----------
    detected : sequence of float
        Detected boundary times in seconds (excluding 0.0 and duration).
    reference : sequence of float
        Ground-truth boundary times in seconds (excluding 0.0 and duration).

    Returns
    -------
    float
        Symmetric mean boundary distance in seconds.
    """
    detected_arr = np.asarray(sorted(detected), dtype=np.float64)
    reference_arr = np.asarray(sorted(reference), dtype=np.float64)

    if detected_arr.size == 0 or reference_arr.size == 0:
        return float("inf")

    # For each detected boundary, find the distance to the nearest reference.
    # Use broadcasting for clarity on small arrays.
    d2r = np.min(
        np.abs(detected_arr[:, None] - reference_arr[None, :]), axis=1
    )
    r2d = np.min(
        np.abs(reference_arr[:, None] - detected_arr[None, :]), axis=1
    )
    return float(np.mean(np.concatenate([d2r, r2d])))


def boundary_tolerance_hits(
    detected: Sequence[float],
    reference: Sequence[float],
    tolerance: float = 3.0,
) -> Tuple[int, int]:
    """Count matched boundaries within a tolerance window.

    A detected boundary matches a reference boundary if it lies within
    ``tolerance`` seconds of it.  Each boundary can be matched at most
    once.  Greedy matching is used for determinism.

    Parameters
    ----------
    detected : sequence of float
        Detected boundary times in seconds.
    reference : sequence of float
        Reference boundary times in seconds.
    tolerance : float
        Maximum distance (seconds) for a match.

    Returns
    -------
    tuple[int, int]
        ``(true_positives, n_reference)``.
    """
    detected_sorted = sorted(detected)
    reference_sorted = sorted(reference)

    matched = [False] * len(reference_sorted)
    true_positives = 0

    for d in detected_sorted:
        best_idx = -1
        best_dist = tolerance + 1.0
        for i, r in enumerate(reference_sorted):
            if matched[i]:
                continue
            dist = abs(d - r)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            matched[best_idx] = True
            true_positives += 1

    return true_positives, len(reference_sorted)


def boundary_precision_recall(
    detected: Sequence[float],
    reference: Sequence[float],
    tolerance: float = 3.0,
) -> Tuple[float, float, float]:
    """Compute boundary precision, recall, and F1.

    Parameters
    ----------
    detected : sequence of float
        Detected boundary times in seconds.
    reference : sequence of float
        Reference boundary times in seconds.
    tolerance : float
        Maximum distance (seconds) for a match.

    Returns
    -------
    tuple[float, float, float]
        ``(precision, recall, f1)`` in 0.0–1.0.
    """
    tp, n_ref = boundary_tolerance_hits(detected, reference, tolerance)

    precision = tp / len(detected) if detected else 0.0
    recall = tp / n_ref if n_ref > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Segment-length statistics
# ---------------------------------------------------------------------------


def segment_durations(segments: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Return the durations (seconds) of each segment.

    Parameters
    ----------
    segments : sequence of (start, end)
        Contiguous ``(start_time, end_time)`` tuples covering a track.

    Returns
    -------
    np.ndarray
        ``(n_segments,)`` array of durations.
    """
    return np.asarray(
        [end - start for start, end in segments], dtype=np.float64
    )


def segment_length_stats(
    segments: Sequence[Tuple[float, float]],
) -> dict:
    """Summary statistics for segment durations.

    Parameters
    ----------
    segments : sequence of (start, end)
        Contiguous ``(start_time, end_time)`` tuples covering a track.

    Returns
    -------
    dict
        ``n_segments``, ``mean``, ``std``, ``min``, ``max``.
    """
    durations = segment_durations(segments)
    if durations.size == 0:
        return {
            "n_segments": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "n_segments": int(durations.size),
        "mean": float(np.mean(durations)),
        "std": float(np.std(durations)),
        "min": float(np.min(durations)),
        "max": float(np.max(durations)),
    }


def segment_length_distribution(
    segments: Sequence[Tuple[float, float]],
    bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of segment durations.

    Parameters
    ----------
    segments : sequence of (start, end)
        Contiguous ``(start_time, end_time)`` tuples covering a track.
    bins : int
        Number of histogram bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(counts, bin_edges)`` as returned by ``np.histogram``.
    """
    durations = segment_durations(segments)
    if durations.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    return np.histogram(durations, bins=bins)


def boundary_count(segments: Sequence[Tuple[float, float]]) -> int:
    """Number of internal boundaries (one fewer than segment count)."""
    return max(0, len(segments) - 1)


def coverage_error(
    segments: Sequence[Tuple[float, float]],
    total_duration: float,
) -> float:
    """Absolute difference between the segment coverage and the track
    duration (seconds).

    A fully contiguous segmentation covering the whole track has an error
    of ``0.0``.
    """
    covered = sum(end - start for start, end in segments)
    return abs(covered - total_duration)