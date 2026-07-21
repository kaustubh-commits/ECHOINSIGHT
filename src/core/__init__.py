"""
EchoInsight Domain Layer.

This package defines the pure data models that represent EchoInsight's
core business concepts.  The Domain Layer imports only Python stdlib
types — no I/O, no configuration, no framework imports.

Layer rules
-----------
- ``src.core`` imports **only** Python stdlib.
- ``src.core`` must **never** import from ``src.config``, ``src.analysis``,
  or ``src.app``.
- All dataclasses are frozen for immutability.

Public API
----------
The following types are exported from the ``src.core`` package and
represent the **stable** domain model that all downstream consumers
(comparison, recommendation, manipulation, visualisation) must use.

Deprecated types from ``src.core.dna_schema`` (SongMetadata, RhythmDNA,
TimbreDNA, TonalDNA, StructureDNA, Segment, IntelligenceDNA, StemDNA)
are still importable via ``src.core.dna_schema`` directly but are no
longer part of the public API and will be removed when the extraction
pipeline is migrated to the new schema.
"""

from .analysis_manifest import AnalysisManifest, compute_config_hash
from .enums import LabelType, RepetitionRole
from .frame_reference import FrameReference
from .identifiers import (
    URI,
    Estimate,
    SegmentIndex,
    SongID,
    generate_song_id,
)
from .song_dna import SongDNA, SongMetadata, SongSummary
from .structure import (
    AlternativeSegmentation,
    BoundaryConfidence,
    SSMReference,
    SegmentContext,
    SegmentDNA,
    SegmentDSP,
    SegmentHarmony,
    SegmentRhythm,
    SegmentTimbre,
    SegmentTiming,
    StructureDNA,
)

__all__ = [
    # Identifiers
    "SongID",
    "SegmentIndex",
    "URI",
    "Estimate",
    "generate_song_id",
    # Enums
    "LabelType",
    "RepetitionRole",
    # Song-level
    "SongDNA",
    "SongMetadata",
    "SongSummary",
    # Structure
    "StructureDNA",
    "SegmentDNA",
    "SegmentTiming",
    "SegmentRhythm",
    "SegmentHarmony",
    "SegmentTimbre",
    "SegmentDSP",
    "SegmentContext",
    "BoundaryConfidence",
    "AlternativeSegmentation",
    "SSMReference",
    # Infrastructure references
    "FrameReference",
    "AnalysisManifest",
    "compute_config_hash",
]