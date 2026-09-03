"""Comparison engine package.

The raw-metric comparison engine lives in ``.engine``; the v2 comparison
data contracts live in ``.contracts``.

Public API
----------
.. code-block:: python

    from src.analysis.comparison import compare, ComparisonReport

    report = compare(song_a, song_b)
    print(report.tempo_difference)

The package re-exports the engine's public API at the top level.  The
``contracts`` sub-module (``ComparisonResult``, ``SegmentMatch``,
``SearchResults``, etc.) is intentionally **not** re-exported here — it
is imported on demand via ``from src.analysis.comparison.contracts
import ...``.
"""

from .engine import (
    ComparisonReport,
    _camelot_distance,
    _camelot_number,
    compare,
    compare_by_files,
    compare_from_dicts,
    format_report,
)

__all__ = [
    "ComparisonReport",
    "compare",
    "compare_by_files",
    "compare_from_dicts",
    "format_report",
    "_camelot_distance",
    "_camelot_number",
]