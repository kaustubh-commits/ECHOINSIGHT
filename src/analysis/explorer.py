"""
explorer.py — SongDNA Explorer that loads a dataset of SongDNA JSON
files and computes a human-readable summary of the collection.

Usage (standalone):
    from src.analysis.explorer import summarize_dataset
    summarize_dataset("data/dna")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEPARATOR: str = "=" * 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _extract_display_name(entry: Dict[str, Any]) -> str:
    """Return a human-readable song name from a SongDNA dict.

    Prefer the basename of the original audio file's filename metadata,
    falling back to a cleaned version of the JSON filename otherwise.
    """
    raw = entry.get("metadata", {}).get("filename", "")
    if raw:
        return Path(raw).stem
    return "Unknown"


def load_song_dna_dataset(directory: str) -> List[Dict[str, Any]]:
    """Load all SongDNA JSON files from *directory* into a list of dicts.

    Parameters
    ----------
    directory : str
        Path to a directory containing ``.json`` SongDNA files.

    Returns
    -------
    List[Dict[str, Any]]
        A list of deserialised SongDNA dictionaries.  Malformed or
        unreadable files are skipped with a logged warning.
    """
    data_path = Path(directory).resolve(strict=True)
    entries: List[Dict[str, Any]] = []

    for json_path in sorted(data_path.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                entry: Dict[str, Any] = json.load(fh)
            entries.append(entry)
            logger.debug("Loaded: %s", json_path.name)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", json_path.name, exc)

    return entries


# ---------------------------------------------------------------------------
# Summarisation helpers
# ---------------------------------------------------------------------------


def _find_extreme(
    entries: List[Dict[str, Any]],
    key_path: List[str],
    *,
    largest: bool,
) -> Optional[Dict[str, Any]]:
    """Find the entry with the extreme (min/max) value along *key_path*.

    Parameters
    ----------
    entries : List[Dict[str, Any]]
        List of SongDNA dictionaries.
    key_path : List[str]
        Nested key path, e.g. ``["rhythm", "tempo"]``.
    largest : bool
        If ``True`` find the maximum; otherwise find the minimum.

    Returns
    -------
    Optional[Dict[str, Any]]
        The entry with the extreme value, or ``None`` if *entries* is empty.
    """
    if not entries:
        return None

    def _value(entry: Dict[str, Any]) -> float:
        val: Any = entry
        for key in key_path:
            val = val.get(key, float("inf") if largest else float("-inf"))
            if val is None:
                return float("-inf") if largest else float("inf")
        return float(val)

    return max(entries, key=_value) if largest else min(entries, key=_value)


def _format_entry(
    entry: Optional[Dict[str, Any]],
    label: str,
    value_key_path: List[str],
    unit: str = "",
) -> str:
    """Format a single entry line for the summary report."""
    if entry is None:
        return f"\n{label}:\n  (no data)\n"

    name = _extract_display_name(entry)
    val: Any = entry
    for key in value_key_path:
        val = val.get(key, "N/A")
    return f"\n{label}:\n{name}\n{value_key_path[-1]}: {val}{unit}\n"


def summarize_dataset(directory: str) -> None:
    """Print a clean summary report of all SongDNA files in *directory*.

    The report includes:

    * Total number of songs loaded.
    * Fastest song (highest BPM).
    * Slowest song (lowest BPM).
    * Brightest song (highest spectral centroid).
    * Darkest song (lowest spectral centroid).
    * Highest energy song (highest RMS).
    * Lowest energy song (lowest RMS).

    Parameters
    ----------
    directory : str
        Path to a directory containing SongDNA ``.json`` files.

    Returns
    -------
    None
    """
    entries = load_song_dna_dataset(directory)
    count = len(entries)

    # --- Compute extremes ------------------------------------------------
    fastest = _find_extreme(entries, ["rhythm", "tempo"], largest=True)
    slowest = _find_extreme(entries, ["rhythm", "tempo"], largest=False)
    brightest = _find_extreme(
        entries, ["timbre", "spectral_centroid_mean"], largest=True
    )
    darkest = _find_extreme(
        entries, ["timbre", "spectral_centroid_mean"], largest=False
    )
    highest_energy = _find_extreme(entries, ["timbre", "rms_energy_mean"], largest=True)
    lowest_energy = _find_extreme(entries, ["timbre", "rms_energy_mean"], largest=False)

    # --- Assemble report -------------------------------------------------
    report_parts: List[str] = [
        _SEPARATOR,
        "EchoInsight Dataset Summary\n",
        f"Songs Loaded: {count}\n",
        _format_entry(fastest, "Fastest Song", ["rhythm", "tempo"], " BPM"),
        _format_entry(slowest, "Slowest Song", ["rhythm", "tempo"], " BPM"),
        _format_entry(
            brightest,
            "Brightest Song",
            ["timbre", "spectral_centroid_mean"],
            " Hz",
        ),
        _format_entry(
            darkest,
            "Darkest Song",
            ["timbre", "spectral_centroid_mean"],
            " Hz",
        ),
        _format_entry(
            highest_energy,
            "Highest Energy Song",
            ["timbre", "rms_energy_mean"],
        ),
        _format_entry(
            lowest_energy,
            "Lowest Energy Song",
            ["timbre", "rms_energy_mean"],
        ),
        _SEPARATOR,
    ]

    report = "\n".join(report_parts)
    logger.info("Dataset summary generated for %d songs.", count)
    print(report)
