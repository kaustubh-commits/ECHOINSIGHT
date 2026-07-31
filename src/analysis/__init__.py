"""
EchoInsight Infrastructure Layer (analysis).

This package implements all DSP, extraction, validation, and visualisation
logic for the EchoInsight platform.  It is the "engine room" that converts
raw audio into structured SongDNA fingerprints and provides tools for
exploring and diagnosing those fingerprints.

Layer rules
-----------
- ``src.analysis`` may import from ``src.core`` but **never** from
  ``src.app`` or ``src.config``.
- All functions receive parameters explicitly — no global state.
- No ``ApplicationContext`` is accessed directly.

Migration note
--------------
The legacy v1 comparison engine (``compare_songs`` etc.) is no longer
re-exported here.  ``src/analysis/comparison.py`` (v1 song-level engine,
bound to ``src.core.dna_schema``) is shadowed by the ``comparison/``
package (v2 comparison contracts, ``src.analysis.comparison.contracts``)
and its migration is tracked in ``docs/MIGRATION.md`` under the
"Comparison consumes SongDNA v2" criterion.  Only the extraction API is
exposed at the package level.
"""

from .extractor import extract_and_save, extract_song_dna

__all__ = [
    "extract_and_save",
    "extract_song_dna",
]
