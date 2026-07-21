"""
EchoInsight Infrastructure Layer (analysis).

This package implements all DSP, comparison, validation, and visualisation
logic for the EchoInsight platform.  It is the "engine room" that converts
raw audio into structured SongDNA fingerprints and provides tools for
comparing, exploring, and diagnosing those fingerprints.

Layer rules
-----------
- ``src.analysis`` may import from ``src.core`` but **never** from
  ``src.app`` or ``src.config``.
- All functions receive parameters explicitly — no global state.
- No ``ApplicationContext`` is accessed directly.
"""

from .comparison import compare_by_files, compare_songs, format_report
from .extractor import extract_and_save, extract_song_dna

__all__ = [
    "compare_by_files",
    "compare_songs",
    "extract_and_save",
    "extract_song_dna",
    "format_report",
]