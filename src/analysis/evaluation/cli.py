"""
cli.py — Command-line interface for the segmentation evaluation workflow.

Usage::

    python -m src.analysis.evaluation.cli path/to/song.mp3 [--show-segments] [--plot]

Flags
-----
--show-segments
    Print a tabular breakdown of the detected segments: Segment #,
    Start (s), End (s), Duration (s), BPM (local), Key.
--plot
    Generate two PNG files in the output directory:
    ``waveform_boundaries.png`` and ``novelty_curve.png``.
--output-dir PATH
    Output directory (default: ``analysis_output/``).

The tool is fully headless — plots are saved to disk, never displayed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import librosa

from src.analysis.evaluation.metrics import coverage_error, segment_length_stats
from src.analysis.evaluation.plotting import (
    plot_novelty_curve,
    plot_waveform_boundaries,
)
from src.analysis.segmentation import detect_segments
from src.analysis.segment_dsp import compute_segment_dsp

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR: str = "analysis_output"


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _format_tabular_segments(segments, dsp_list) -> str:
    """Build a human-readable segment table."""
    lines = [
        f"{'Segment #':<10s} {'Start (s)':>10s} {'End (s)':>10s} "
        f"{'Duration (s)':>12s} {'BPM (local)':>12s} {'Key':>10s}",
        "-" * 70,
    ]
    for i, ((start_s, end_s), dsp) in enumerate(zip(segments, dsp_list)):
        lines.append(
            f"{i + 1:<10d} {start_s:>10.2f} {end_s:>10.2f} "
            f"{end_s - start_s:>12.2f} "
            f"{dsp.rhythm.tempo:>12.2f} "
            f"{str(dsp.harmony.key):>10s}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.analysis.evaluation.cli",
        description="Evaluate structural segmentation on an audio file.",
    )
    parser.add_argument(
        "audio", help="Path to an audio file (MP3, WAV, FLAC, OGG, M4A)."
    )
    parser.add_argument(
        "--show-segments",
        action="store_true",
        help="Print the detected segments as a table.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render waveform + novelty plots to disk.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plots (default: {DEFAULT_OUTPUT_DIR}/).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 2

    # Load (locked to 22050 Hz)
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(len(y)) / float(sr)

    # Segment + DSP
    segments = detect_segments(y, sr)
    dsp_list = compute_segment_dsp(y, sr, segments)

    # Validate coverage
    coverage_err = coverage_error(segments, duration)
    stats = segment_length_stats(segments)

    print("=" * 70)
    print("EchoInsight — Segmentation Evaluation")
    print(f"File:     {audio_path.name}")
    print(f"Duration: {duration:.2f}s")
    print(f"Segments: {stats['n_segments']}")
    print(f"Seg len:  mean {stats['mean']:.2f}s  std {stats['std']:.2f}s  "
          f"min {stats['min']:.2f}s  max {stats['max']:.2f}s")
    print(f"Coverage error: {coverage_err:.4f}s")
    print("=" * 70)

    if args.show_segments:
        print()
        print(_format_tabular_segments(segments, dsp_list))
        print()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot:
        waveform_png = plot_waveform_boundaries(
            y, sr, segments, output_dir / "waveform_boundaries.png"
        )
        novelty_png = plot_novelty_curve(y, sr, output_dir / "novelty_curve.png")
        print(f"Saved plot: {waveform_png}")
        print(f"Saved plot: {novelty_png}")

    if coverage_err > 1e-2:
        print(
            f"⚠ WARNING: coverage error {coverage_err:.4f}s — segments do not "
            "fully cover the track.  This violates the Extraction Contract."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())