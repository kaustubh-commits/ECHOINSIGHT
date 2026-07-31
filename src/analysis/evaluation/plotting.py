"""
plotting.py — Plot generation for segmentation evaluation.

Renders two diagnostic plots to disk (no GUI, no pop-ups):

- ``waveform_boundaries.png`` — waveform with vertical boundary lines
  and generic segment labels.
- ``novelty_curve.png`` — novelty curve with peak-picked boundary points.

Uses Matplotlib's non-interactive ``Agg`` backend so the evaluation
workflow works headless.

Layer rules
-----------
- ``src.analysis.evaluation`` may import from ``src.core`` and
  ``src.analysis.segmentation``.
- It must **never** import from ``src.app`` or ``src.config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — no GUI. Must be set before pyplot.

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.segmentation import HOP_LENGTH, _novelty_curve, _recurrence_matrix


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------


def plot_waveform_boundaries(
    y: np.ndarray,
    sr: int,
    segments: Sequence[Tuple[float, float]],
    output_path: Path,
    labels: Optional[Sequence[str]] = None,
) -> Path:
    """Render a waveform with vertical boundary lines and segment labels.

    Parameters
    ----------
    y : np.ndarray
        Mono audio time series.
    sr : int
        Sample rate in Hz.
    segments : sequence of (start, end)
        Contiguous ``(start_time, end_time)`` tuples in seconds.
    output_path : Path
        Absolute path for the output PNG.  Parent directories are
        created automatically.
    labels : optional sequence of str
        Per-segment labels.  Defaults to ``"Segment 1"``, ``"Segment 2"``…

    Returns
    -------
    Path
        The absolute path where the PNG was saved.
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time_axis = np.arange(len(y), dtype=np.float64) / float(sr)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_axis, y, color="steelblue", linewidth=0.4)
    ax.set_xlim(0.0, float(time_axis[-1]) if time_axis.size else 1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform with Detected Segment Boundaries")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Draw boundaries
    boundaries: List[float] = []
    for start_s, end_s in segments:
        if start_s > 0.0:
            boundaries.append(start_s)
        boundaries.append(end_s)
    # Drop the final duration boundary (end of track)
    if boundaries:
        boundaries = boundaries[:-1]

    for b in boundaries:
        ax.axvline(b, color="crimson", linewidth=0.8, linestyle="--", alpha=0.8)

    # Segment labels
    for i, (start_s, end_s) in enumerate(segments):
        label = labels[i] if labels and i < len(labels) else f"Segment {i + 1}"
        ax.text(
            (start_s + end_s) / 2.0,
            0.95,
            label,
            ha="center",
            va="top",
            fontsize=8,
            alpha=0.9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_novelty_curve(
    y: np.ndarray,
    sr: int,
    output_path: Path,
) -> Path:
    """Render the novelty curve with peak-picked boundary points.

    The novelty curve is computed from the MFCC recurrence matrix via
    ``segmentation._novelty_curve``.  Peak-picked boundary frames are
    overlaid as red markers.

    Parameters
    ----------
    y : np.ndarray
        Mono audio time series.
    sr : int
        Sample rate in Hz.
    output_path : Path
        Absolute path for the output PNG.

    Returns
    -------
    Path
        The absolute path where the PNG was saved.
    """
    import librosa
    import scipy.signal

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH
    )
    recurrence = _recurrence_matrix(mfcc)
    novelty = _novelty_curve(recurrence)

    time_axis = librosa.frames_to_time(
        np.arange(novelty.size), sr=sr, hop_length=HOP_LENGTH
    )

    # Peak pick using scipy's find_peaks with a sensible distance
    min_distance = max(1, int(4.0 * sr / HOP_LENGTH))
    peaks, _ = scipy.signal.find_peaks(
        novelty,
        distance=min_distance,
        prominence=0.05,
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_axis, novelty, color="darkgreen", linewidth=0.8)
    if peaks.size:
        ax.plot(
            time_axis[peaks],
            novelty[peaks],
            "ro",
            markersize=4,
            label="Peak-picked boundaries",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Novelty")
    ax.set_title("Novelty Curve with Detected Boundary Peaks")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path