"""
visualization.py — Generate exploratory visualisations from a SongDNA
dataset.  Produces PNG plots under ``data/plots/``.

All plotting is done with **matplotlib** only.  Data loading reuses
:func:`src.analysis.explorer.load_song_dna_dataset` to avoid duplicating
JSON I/O logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from src.analysis.explorer import _extract_display_name, load_song_dna_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unpack_entries(
    entries: List[Dict[str, Any]],
) -> tuple[List[str], List[float], List[float], List[float]]:
    """Extract parallel lists of names and features from SongDNA dicts.

    Returns
    -------
    tuple of (names, tempos, rms_values, centroid_values)
        Each list is in the same order as *entries*.
    """
    names: List[str] = []
    tempos: List[float] = []
    rms_values: List[float] = []
    centroid_values: List[float] = []

    for entry in entries:
        names.append(_extract_display_name(entry))
        tempos.append(float(entry.get("rhythm", {}).get("tempo", 0)))
        rms_values.append(float(entry.get("timbre", {}).get("rms_energy_mean", 0)))
        centroid_values.append(
            float(entry.get("timbre", {}).get("spectral_centroid_mean", 0))
        )

    return names, tempos, rms_values, centroid_values


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------


def plot_tempo_vs_spectral_centroid(directory: str, output_dir: str) -> Path:
    """Scatter plot: tempo (x) vs spectral centroid (y), labelled by song.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    entries = load_song_dna_dataset(directory)
    names, tempos, _, centroids = _unpack_entries(entries)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(tempos, centroids, alpha=0.7, edgecolors="k", linewidth=0.5)

    # Label each point with the song name.
    for name, x, y in zip(names, tempos, centroids):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Tempo (BPM)")
    ax.set_ylabel("Spectral Centroid Mean (Hz)")
    ax.set_title("Tempo vs Spectral Centroid")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = Path(output_dir).resolve() / "tempo_vs_spectral_centroid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_rms_energy_distribution(directory: str, output_dir: str) -> Path:
    """Histogram of mean RMS energy across the dataset.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    entries = load_song_dna_dataset(directory)
    _, _, rms_values, _ = _unpack_entries(entries)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rms_values, bins=10, edgecolor="black", alpha=0.7)
    ax.set_xlabel("RMS Energy Mean")
    ax.set_ylabel("Number of Songs")
    ax.set_title("RMS Energy Distribution")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = Path(output_dir).resolve() / "rms_energy_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_tempo_distribution(directory: str, output_dir: str) -> Path:
    """Histogram of tempo (BPM) across the dataset.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    entries = load_song_dna_dataset(directory)
    _, tempos, _, _ = _unpack_entries(entries)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(tempos, bins=10, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Tempo (BPM)")
    ax.set_ylabel("Number of Songs")
    ax.set_title("Tempo Distribution")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = Path(output_dir).resolve() / "tempo_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def plot_chromagram(directory: str, output_dir: str) -> Path:
    """Chromagram heatmap for the first song in the dataset.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    entries = load_song_dna_dataset(directory)
    if not entries:
        logger.warning("No entries found — chromagram not generated.")
        out = Path(output_dir).resolve() / "chromagram.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    entry = entries[0]
    name = _extract_display_name(entry)
    chroma = entry.get("tonal", {}).get("chroma_mean", [])
    key = entry.get("tonal", {}).get("key", "unknown")

    fig, ax = plt.subplots(figsize=(10, 4))
    if chroma and len(chroma) == 12:
        # Simple bar chart of the average chroma vector
        pitch_classes = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]
        ax.bar(pitch_classes, chroma, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_title(f"Average Chroma — {name} (Key: {key})")
        ax.set_ylabel("Energy")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    else:
        ax.text(
            0.5,
            0.5,
            "No chroma data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()

    out = Path(output_dir).resolve() / "chromagram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def generate_all_plots(
    directory: str = "data/dna",
    output_dir: str = "data/plots",
) -> List[Path]:
    """Generate all three standard visualisation plots.

    Parameters
    ----------
    directory : str
        Directory containing SongDNA ``.json`` files (default ``data/dna``).
    output_dir : str
        Directory where ``.png`` files will be saved.  Created automatically
        if it does not exist (default ``data/plots``).

    Returns
    -------
    List[Path]
        Absolute paths to each generated plot file.  Returns an empty list
        if no SongDNA files were found.
    """
    # Quick guard: if there are no JSON files, bail early.
    entries = load_song_dna_dataset(directory)
    if not entries:
        logger.warning(
            "No SongDNA entries found in %s — no plots generated.", directory
        )
        return []

    out_paths: List[Path] = []

    out_paths.append(plot_tempo_vs_spectral_centroid(directory, output_dir))
    out_paths.append(plot_rms_energy_distribution(directory, output_dir))
    out_paths.append(plot_tempo_distribution(directory, output_dir))
    out_paths.append(plot_chromagram(directory, output_dir))

    logger.info("All %d plots saved to %s", len(out_paths), Path(output_dir).resolve())
    return out_paths
