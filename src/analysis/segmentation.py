"""
segmentation.py — Structural boundary detection for the v2 extraction
pipeline.

Given a loaded audio time series, this module detects the macro-structural
sections of a song (intro, verse, chorus, etc. — labelled generically for
v0.2) and returns a list of temporally contiguous ``(start_time, end_time)``
tuples in seconds that cover the entire track.

Implementation notes
--------------------
The pipeline is deliberately simple and deterministic:

1. Compute MFCCs over the full track (13 coefficients).  MFCCs capture
   timbral/textural change, which is the primary signal for structural
   boundaries.
2. Compute a recurrence matrix (self-similarity) from the MFCCs via
   ``librosa.segment.recurrence_matrix``.  Repeating sections (verse,
   chorus) produce strong off-diagonal stripes.
3. Compute the novelty curve from the recurrence matrix via
   ``librosa.segment.recurrence_to_lag`` + summing the first difference.
   (For librosa 0.11.0, ``recurrence_to_lag`` exists in
   ``librosa.segment``.)  Falls back to spectral-flux novelty if the
   lag-domain is unavailable.
4. Peak-pick the novelty curve to find candidate boundary frames.
5. Apply agglomerative structural clustering to refine boundaries:
   ``librosa.segment.agglomerative`` clusters MFCC frames into
   ``n_segments`` groups; boundaries are the frame indices where the
   cluster label changes.
6. Merge the two candidate sets (peaks + cluster transitions) and enforce
   a minimum segment duration so the result is musically plausible.

The output is guaranteed to satisfy the Extraction Contract invariants:
- ``segments[0].start == 0.0``
- ``segments[-1].end == duration``
- ``segment[i].end == segment[i + 1].start`` for all i

Layer rules
-----------
- ``src.analysis.segmentation`` may import from ``src.core``.
- It must **never** import from ``src.app`` or ``src.config``.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOP_LENGTH: int = 512
"""Hop length in samples — must match the extraction pipeline's hop."""

N_MFCC: int = 13
"""Number of MFCC coefficients used for segmentation."""

MIN_SEGMENT_DURATION: float = 4.0
"""Minimum segment duration in seconds.  Shorter segments are merged."""

N_SEGMENTS_AGGLO: int = 16
"""Target number of segments for the agglomerative clustering step.

This is deliberately a "stretch" target — the actual number of returned
segments may be lower because merge rules enforce the minimum duration.
"""

_K = N_MFCC  # number of nearest neighbours in the recurrence matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_segments(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Detect the macro-structural segments of a song.

    Parameters
    ----------
    y : np.ndarray
        Mono audio time series.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    List[Tuple[float, float]]
        ``(start_time, end_time)`` tuples in seconds, temporally
        contiguous, covering ``[0.0, duration]`` exactly.
    """
    duration = float(len(y)) / float(sr)
    if duration <= 0.0:
        raise ValueError(f"audio duration must be positive, got {duration}")

    frames = _frames_from_time(np.array([0.0, duration]), sr)

    # --- Compute MFCCs ---------------------------------------------------
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )  # shape (N_MFCC, n_frames)
    n_frames = mfcc.shape[1]
    logger.debug("MFCC shape: %s over %.2fs", mfcc.shape, duration)

    if n_frames < 4:
        # Too short to segment meaningfully — return a single segment.
        return [(0.0, duration)]

    # --- Recurrence matrix and novelty -----------------------------------
    recurrence = _recurrence_matrix(mfcc)
    novelty = _novelty_curve(recurrence)

    peak_frames = _peak_frames(novelty, duration)

    cluster_frames = _cluster_boundaries(mfcc, n_segments=N_SEGMENTS_AGGLO)

    # --- Merge candidates ------------------------------------------------
    peak_set = set(int(f) for f in peak_frames)
    cluster_set = set(cluster_frames)
    boundary_frames = sorted(peak_set | cluster_set)

    boundary_frames = _enforce_min_segment(boundary_frames, frames, duration)

    if not boundary_frames:
        return [(0.0, duration)]

    boundary_seconds = _time_from_frames(
        np.asarray(sorted(boundary_frames), dtype=np.float64), sr
    )
    boundary_times: List[float] = [0.0] + [float(t) for t in boundary_seconds]
    boundary_times.append(duration)

    segments: List[Tuple[float, float]] = []
    for i in range(len(boundary_times) - 1):
        start = float(boundary_times[i])
        end = float(boundary_times[i + 1])
        if end - start >= 1e-3:  # ignore sub-millisecond slices
            segments.append((start, end))

    if not segments:
        return [(0.0, duration)]

    # Guarantee full coverage: snap first to 0.0 and last to duration.
    if segments[0][0] != 0.0:
        segments[0] = (0.0, segments[0][1])
    if segments[-1][1] != duration:
        segments[-1] = (segments[-1][0], duration)

    logger.debug("Detected %d segments: %s", len(segments), segments)
    return segments


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _frames_from_time(times: np.ndarray, sr: int) -> np.ndarray:
    """Convert times (seconds) to frame indices (hop-aligned)."""
    return np.round(np.asarray(times) * sr / float(HOP_LENGTH)).astype(int)


def _time_from_frames(frames: np.ndarray, sr: int) -> np.ndarray:
    """Convert frame indices to times (seconds)."""
    return np.asarray(frames) * float(HOP_LENGTH) / float(sr)


def _recurrence_matrix(mfcc: np.ndarray) -> np.ndarray:
    """Compute a binary recurrence matrix from MFCC features.

    Uses a fixed neighbourhood of ``_K`` nearest neighbours with a small
    exclusion band around the diagonal (self-similarity).
    """
    R = librosa.segment.recurrence_matrix(
        mfcc,
        k=_K,
        width=3,
        metric="cosine",
        sym=True,
    )
    return R


def _novelty_curve(recurrence: np.ndarray) -> np.ndarray:
    """Compute a novelty curve from a recurrence matrix.

    For librosa 0.11.0 we use ``librosa.segment.recurrence_to_lag`` and
    take the L1 difference along the lag axis, then sum across possible
    lags.  If that API is unavailable we fall back to the spectral flux
    of the recurrence matrix columns.
    """
    try:
        lag = librosa.segment.recurrence_to_lag(recurrence, pad=False)
        # Novelty = sum of absolute differences in the lag domain
        novelty = np.sum(np.abs(np.diff(lag, axis=1)), axis=0)
    except (AttributeError, ValueError):
        # Fallback: spectral flux across columns
        novelty = np.sum(
            np.abs(np.diff(recurrence.astype(np.float64), axis=1)), axis=0
        )

    if novelty.size == 0:
        return np.zeros(recurrence.shape[1], dtype=np.float64)

    # Normalise to [0, 1]
    mx = float(np.max(novelty))
    if mx > 0.0:
        novelty = novelty / mx
    else:
        return np.zeros_like(novelty, dtype=np.float64)

    # Smooth slightly to reduce spurious peaks
    kernel = np.hanning(5)
    kernel = kernel / kernel.sum()
    novelty = np.convolve(novelty, kernel, mode="same")
    return novelty


def _peak_frames(novelty: np.ndarray, duration: float) -> List[int]:
    """Peak-pick the novelty curve for candidate boundary frames."""
    if novelty.size == 0:
        return []

    # Select peaks above a threshold using librosa's peak_pick.
    # distance is the minimum number of frames between peaks.
    distance = max(1, int(MIN_SEGMENT_DURATION * (22050 / HOP_LENGTH) / 2))
    try:
        peaks = librosa.util.peak_pick(
            novelty,
            pre_max=int(distance / 2) + 1,
            post_max=int(distance / 2) + 1,
            pre_avg=int(distance / 2) + 1,
            post_avg=int(distance / 2) + 1,
            delta=0.15,
            wait=distance,
        )
        return [int(p) for p in peaks]
    except (ValueError, IndexError):
        # peak_pick can fail on degenerate inputs — fall back to
        # simple argmax on non-overlapping windows.
        step = max(distance, 1)
        peaks: List[int] = []
        idx = int(step / 2)
        while idx < novelty.size:
            window = novelty[max(0, idx - int(step / 2)): idx + int(step / 2) + 1]
            if window.size:
                local = int(np.argmax(window)) + max(0, idx - int(step / 2))
                p = int(local)
                if p not in peaks and p > 0 and p < novelty.size - 1:
                    peaks.append(p)
            idx += step
        return peaks


def _cluster_boundaries(mfcc: np.ndarray, n_segments: int) -> List[int]:
    """Detect boundaries via agglomerative structural clustering.

    The ``librosa.segment.agglomerative`` function clusters MFCC frames
    into ``n_segments`` perceptual groups; segment boundaries lie at the
    frame indices where the cluster label changes.

    If the first/last frames belong to the same cluster, a synthetic
    boundary is not added — coverage is guaranteed by ``detect_segments``.
    """
    n_frames = mfcc.shape[1]
    if n_frames < 2 or n_segments >= n_frames:
        return []

    # librosa.segment.agglomerative expects (n_features, n_frames).
    # In librosa 0.11.0 the `clusterer` parameter defaults to
    # AgglomerativeClustering(n_clusters=k) with Ward linkage, so no
    # extra argument is needed.
    labels = librosa.segment.agglomerative(mfcc, n_segments)

    boundaries: List[int] = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.append(i)
    return boundaries


def _enforce_min_segment(
    boundary_frames: List[int],
    frames: np.ndarray,
    duration: float,
) -> List[int]:
    """Remove boundaries that would create segments shorter than the
    minimum duration.

    ``frames`` is the array of hop-aligned frame indices for the track
    boundaries (``[0, n_frames]``).
    """
    if not boundary_frames:
        return []

    min_frames = max(1, int(MIN_SEGMENT_DURATION * (22050 / HOP_LENGTH)))

    # Filter boundaries that are too close to the start or end
    filtered = [b for b in boundary_frames if min_frames < b < frames[-1] - min_frames]

    # Iteratively remove the pair of boundaries that creates the smallest
    # segment until all segments satisfy the minimum.
    while len(filtered) > 1:
        points = [0] + sorted(filtered) + [int(frames[-1])]
        min_gap = min(points[i + 1] - points[i] for i in range(len(points) - 1))
        if min_gap >= min_frames:
            break

        # Find the smallest gap and remove the boundary between them
        min_idx = 0
        best_gap = float("inf")
        for i in range(len(points) - 1):
            gap_i = points[i + 1] - points[i]
            if gap_i < best_gap:
                best_gap = float(gap_i)
                min_idx = i
        # The boundary to remove is the later point of the smallest gap
        remove = points[min_idx + 1]
        if remove not in filtered:
            # We should never hit this with the [0] and [end] guards,
            # but guard anyway.
            if min_idx + 1 < len(points) - 1:
                remove = points[min_idx + 2]
            else:
                break
        filtered = [b for b in filtered if b != remove]

    return filtered