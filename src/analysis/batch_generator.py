"""
batch_generator.py — Batch SongDNA generator that recursively scans an
input directory for supported audio files and serialises each extracted
SongDNA as a JSON file under a given output directory.

Frame-level data is also persisted as NPZ files alongside the JSON.

Usage (standalone):
    from src.analysis.batch_generator import generate_dna_dataset
    generate_dna_dataset("music", "data/dna")
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Set

from src.analysis.extractor import extract_and_save

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# File extensions that will be accepted for processing.
_SUPPORTED_EXTENSIONS: Set[str] = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_supported(path: Path) -> bool:
    """Return ``True`` if *path* has a supported audio extension."""
    return path.suffix.lower() in _SUPPORTED_EXTENSIONS


def _output_path(audio_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """Compute the JSON output path mirroring the relative structure of
    *audio_path* under *input_dir* into *output_dir*.

    Example
    -------
    input_dir=/music, audio_path=/music/some/folder/song.mp3
    → output_dir/some/folder/song.json
    """
    rel = audio_path.relative_to(input_dir.resolve())
    return output_dir / rel.with_suffix(".json")


def _resolve_data_root(json_output_dir: Path) -> Path:
    """Resolve the data root directory from the JSON output directory.

    In the standard layout, the data root is the parent of ``dna/``.
    If the output directory is not named ``dna``, fall back to the
    output directory's parent.
    """
    if json_output_dir.name == "dna":
        return json_output_dir.parent.resolve()
    return json_output_dir.parent.resolve()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dna_dataset(input_directory: str, output_directory: str) -> None:
    """Recursively scan *input_directory* for supported audio files,
    extract SongDNA fingerprints with frame-level data, and save each
    result as a JSON file under *output_directory*.

    Frame-level feature arrays are saved as compressed NPZ files under
    ``<data_root>/frames/``, where ``data_root`` is inferred from the
    output directory structure.

    Parameters
    ----------
    input_directory : str
        Root directory to walk recursively for audio files.
    output_directory : str
        Root directory where JSON output files will be written.  The
        directory and any required sub-directories are created automatically.

    Returns
    -------
    None
    """
    input_path = Path(input_directory).resolve(strict=True)
    output_path = Path(output_directory).resolve()
    data_root = _resolve_data_root(output_path)

    files_found: int = 0
    files_processed: int = 0
    files_failed: int = 0

    # Walk the input tree recursively.
    for audio_path in sorted(input_path.rglob("*")):
        if not audio_path.is_file() or not _is_supported(audio_path):
            continue

        files_found += 1
        out = _output_path(audio_path, input_path, output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Full extraction with NPZ persistence
            dna = extract_and_save(str(audio_path), data_root=data_root)
            out.write_text(
                json.dumps(dataclasses.asdict(dna), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            files_processed += 1
            logger.info("Processed: %s → %s (frames: %s)",
                        audio_path, out, dna.frames.uri if dna.frames else "none")

        except Exception:
            files_failed += 1
            logger.exception("Failed to process: %s", audio_path)
            continue

    # Final summary.
    summary = (
        f"\n{'=' * 50}\n"
        f"Batch SongDNA generation complete.\n"
        f"  Files found:     {files_found}\n"
        f"  Files processed: {files_processed}\n"
        f"  Files failed:    {files_failed}\n"
        f"{'=' * 50}"
    )
    logger.info(summary)
    print(summary)