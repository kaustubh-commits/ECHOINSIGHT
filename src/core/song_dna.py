"""
song_dna.py — The SongDNA aggregate root.

``SongDNA`` is the **sole aggregate root** in EchoInsight's domain model.
Every capability — comparison, recommendation, manipulation, visualisation —
loads a ``SongDNA`` as its unit of access.  ``SongDNA`` owns ``StructureDNA``
by composition and references frame-level data via ``FrameReference``.

Ownership diagram::

    SongDNA  (aggregate root)
     ├── SongMetadata   (value object)
     ├── SongSummary    (value object — song-level aggregates for fast filtering)
     ├── StructureDNA   (entity — composition, not reference)
     ├── FrameReference (value object — NPZ URI)
     └── AnalysisManifest (value object — provenance)

What ``SongDNA`` does NOT contain:
- Comparison results (the Comparison Engine owns those)
- Recommendation results (the Recommendation Engine owns those)
- Embeddings (future ``EmbeddingDNA`` aggregate, linked by ``song_id``)
- Stem data (future ``StemDNA`` aggregate, linked by ``song_id``)
- User annotations (future ``UserData`` aggregate)
- Raw frame arrays (those live in NPZ files)

Layer rules
-----------
- ``src.core.song_dna`` imports **only** Python stdlib + ``src.core.identifiers``,
  ``src.core.enums``, and ``src.core.structure``.
- It must **never** import ``src.analysis``, ``src.config``, ``src.app``,
  or any external package.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import typing
from typing import Optional, Tuple

from src.core.analysis_manifest import AnalysisManifest
from src.core.frame_reference import FrameReference
from src.core.identifiers import SongID
from src.core.structure import StructureDNA
from src.core.timeline import Timeline

# ---------------------------------------------------------------------------
# SongMetadata
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SongMetadata:
    """Immutable metadata describing the audio source.

    ``SongMetadata`` captures everything that is known about the *source
    file* — not the musical content (that lives in ``SongSummary`` and
    ``SegmentDNA``).

    Attributes
    ----------
    song_id : SongID
        UUID v4 that uniquely identifies this song across the entire dataset.
        Generated at extraction time.  Foreign key for future aggregates
        (StemDNA, EmbeddingDNA, etc.).
    filename : str
        Original filename of the source audio file.  Not a path — just the
        filename for display purposes.
    duration : float
        Total duration of the track in seconds.  Must be positive.
    sample_rate : int
        Sample rate in Hz (e.g. 44100, 48000).  Must be positive.
    channels : int
        Number of audio channels (1 = mono, 2 = stereo).
    bit_depth : Optional[int]
        Bit depth of the original file (e.g. 16, 24, 32).  ``None`` for
        compressed formats (MP3, AAC) where bit depth is not meaningful.
    format : str
        Original file extension/format, e.g. ``"mp3"``, ``"wav"``,
        ``"flac"``, ``"m4a"``.
    """

    song_id: SongID
    filename: str
    duration: float
    sample_rate: int
    channels: int = 1
    bit_depth: Optional[int] = None
    format: str = "mp3"

    def __post_init__(self) -> None:
        if not self.song_id:
            raise ValueError("song_id must be non-empty")
        if not self.filename.strip():
            raise ValueError("filename must be non-empty")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )
        if self.channels not in (1, 2):
            raise ValueError(
                f"channels must be 1 (mono) or 2 (stereo), got {self.channels}"
            )
        if self.bit_depth is not None and self.bit_depth <= 0:
            raise ValueError(
                f"bit_depth must be positive, got {self.bit_depth}"
            )
        if not self.format.strip():
            raise ValueError("format must be non-empty")


# ---------------------------------------------------------------------------
# SongSummary
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SongSummary:
    """Song-level aggregate statistics for fast filtering and comparison.

    ``SongSummary`` contains *estimates* computed from segment-level data.
    These are not ground truth — they exist to enable quick filtering
    (e.g. "show me all songs in C major with tempo 120-130 BPM") without
    loading per-segment data.

    The ground truth for any musical feature lives in ``SegmentDNA``.
    ``SongSummary`` values are derived from segment-level features and
    should be treated as approximations.

    Attributes
    ----------
    tempo_estimate : float
        Song-level tempo in BPM.  When ``tempo_is_fallback`` is True this
        is a placeholder, not a measurement.
    tempo_confidence : Optional[float]
        Confidence of the tempo estimate in 0.0–1.0, or ``None`` when no
        confidence was computed (the current extractor computes none).
    key_estimate : Optional[str]
        The most common key across segments, e.g. ``"C major"``,
        ``"A minor"``.  ``None`` when no key could be reliably estimated.
    key_confidence : float
        Confidence of the key estimate in 0.0–1.0.
    key_alternatives : Tuple[Tuple[str, float], ...]
        Alternative key candidates as ``(key_name, confidence)`` tuples.
        Empty tuple when no alternatives exist.
    rms_energy_mean : float
        Mean RMS energy across the entire song.  Always >= 0.
    spectral_centroid_mean : float
        Mean spectral centroid in Hz — correlates with perceived brightness.
        Always >= 0.
    spectral_bandwidth_mean : float
        Mean spectral bandwidth in Hz.  Always >= 0.
    mfcc_mean : Tuple[float, ...]
        Mean MFCC coefficients (typically 13) across the entire song.
    chroma_mean : Tuple[float, ...]
        Mean chroma vector (12 bins) across the entire song.
    tuning_offset : float
        Deviation from A440 in cents.  Positive = sharp, negative = flat.
    tempo_is_fallback : bool
        True when no tempo could be detected and ``tempo_estimate`` is the
        extractor's fixed fallback value.
    """

    tempo_estimate: float = 0.0
    tempo_confidence: Optional[float] = None
    key_estimate: Optional[str] = None
    key_confidence: float = 0.0
    key_alternatives: Tuple[Tuple[str, float], ...] = ()
    rms_energy_mean: float = 0.0
    spectral_centroid_mean: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    mfcc_mean: Tuple[float, ...] = ()
    chroma_mean: Tuple[float, ...] = ()
    tuning_offset: float = 0.0
    tempo_is_fallback: bool = False

    def __post_init__(self) -> None:
        if self.tempo_estimate < 0:
            raise ValueError(
                f"tempo_estimate must be >= 0, got {self.tempo_estimate}"
            )
        if self.tempo_confidence is not None and not 0.0 <= self.tempo_confidence <= 1.0:
            raise ValueError(
                f"tempo_confidence must be in [0, 1], "
                f"got {self.tempo_confidence}"
            )
        if self.key_estimate is not None and not self.key_estimate.strip():
            raise ValueError("key_estimate must be non-empty when provided")
        if not 0.0 <= self.key_confidence <= 1.0:
            raise ValueError(
                f"key_confidence must be in [0, 1], "
                f"got {self.key_confidence}"
            )
        if self.rms_energy_mean < 0:
            raise ValueError(
                f"rms_energy_mean must be >= 0, "
                f"got {self.rms_energy_mean}"
            )
        if self.spectral_centroid_mean < 0:
            raise ValueError(
                f"spectral_centroid_mean must be >= 0, "
                f"got {self.spectral_centroid_mean}"
            )
        if self.spectral_bandwidth_mean < 0:
            raise ValueError(
                f"spectral_bandwidth_mean must be >= 0, "
                f"got {self.spectral_bandwidth_mean}"
            )
        if self.mfcc_mean and len(self.mfcc_mean) != 13:
            raise ValueError(
                f"mfcc_mean must have 13 elements, "
                f"got {len(self.mfcc_mean)}"
            )
        if self.chroma_mean and len(self.chroma_mean) != 12:
            raise ValueError(
                f"chroma_mean must have 12 elements, "
                f"got {len(self.chroma_mean)}"
            )
        # Validate key_alternatives structure
        for i, alt in enumerate(self.key_alternatives):
            if len(alt) != 2:
                raise ValueError(
                    f"key_alternatives[{i}] must be a (str, float) tuple, "
                    f"got {alt}"
                )
            key_name, conf = alt
            if not isinstance(key_name, str) or not key_name.strip():
                raise ValueError(
                    f"key_alternatives[{i}][0] must be a non-empty string, "
                    f"got {key_name!r}"
                )
            if not isinstance(conf, (int, float)):
                raise ValueError(
                    f"key_alternatives[{i}][1] must be a float, "
                    f"got {conf!r}"
                )
            if not 0.0 <= conf <= 1.0:
                raise ValueError(
                    f"key_alternatives[{i}][1] (confidence) must be in "
                    f"[0, 1], got {conf}"
                )


# ---------------------------------------------------------------------------
# JSON reconstruction helpers
# ---------------------------------------------------------------------------


def _reconstruct_value(tp: object, value: object) -> object:
    """Recursively reconstruct a value from deserialised JSON data.

    JSON round-trips convert tuples to lists and enums to their string
    values.  This function walks the type hint of each dataclass field
    and reconstructs the original Python types.
    """
    if value is None:
        return None

    origin = typing.get_origin(tp)

    if origin is typing.Union:
        non_none = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return _reconstruct_value(non_none[0], value)
        return value

    if origin in (tuple, typing.Tuple):
        args = typing.get_args(tp)
        if args and args[-1] is Ellipsis:
            elem_tp = args[0]
            return tuple(_reconstruct_value(elem_tp, v) for v in value)
        if len(args) == len(value) and args:
            return tuple(_reconstruct_value(a, v) for a, v in zip(args, value))
        return tuple(value)

    if origin in (list, typing.List):
        args = typing.get_args(tp)
        if args:
            elem_tp = args[0]
            return [_reconstruct_value(elem_tp, v) for v in value]
        return list(value)

    if origin in (dict, typing.Dict):
        args = typing.get_args(tp)
        if len(args) == 2:
            _, val_tp = args
            return {k: _reconstruct_value(val_tp, v) for k, v in value.items()}
        return dict(value)

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _reconstruct_dataclass(tp, value)

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return tp(value)

    return value


def _reconstruct_dataclass(cls: type, data: dict) -> object:
    """Reconstruct a frozen dataclass from a dict using its type hints.

    Fields with ``init=False`` (computed fields such as
    ``SegmentTiming.duration`` and ``StructureDNA.num_segments``) are
    recomputed automatically by the constructor and therefore skipped.
    """
    hints = typing.get_type_hints(cls)
    kwargs: dict = {}
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        if field.name not in data:
            continue
        kwargs[field.name] = _reconstruct_value(hints[field.name], data[field.name])
    return cls(**kwargs)


def _migrate_tempo_fields(rhythm_like: dict) -> None:
    """Migrate one pre-2.2.0 tempo record in place.

    Before ``tempo_is_fallback`` existed, the extractor encoded a fallback
    tempo as confidence 0.0 and wrote fixed placeholder confidences
    (1.0 or 0.8) for detected tempos.  None of those numbers was measured,
    so they become ``None``; confidence 0.0 becomes ``tempo_is_fallback``.
    """
    if "tempo_is_fallback" in rhythm_like:
        return
    rhythm_like["tempo_is_fallback"] = rhythm_like.get("tempo_confidence") == 0.0
    rhythm_like["tempo_confidence"] = None


def migrate_song_dna_dict(data: dict) -> dict:
    """Upgrade a serialised SongDNA dict to the current schema (in place).

    Only data written before schema 2.2.0 is changed; current data passes
    through untouched.  Returns *data* for convenience.
    """
    summary = data.get("summary")
    if isinstance(summary, dict):
        _migrate_tempo_fields(summary)

    structure = data.get("structure")
    if isinstance(structure, dict):
        segment_lists = [structure.get("segments") or []]
        for alt in structure.get("alternative_segmentations") or []:
            segment_lists.append(alt.get("segments") or [])
        for segments in segment_lists:
            for seg in segments:
                rhythm = seg.get("dsp", {}).get("rhythm")
                if isinstance(rhythm, dict):
                    _migrate_tempo_fields(rhythm)
    return data


# ---------------------------------------------------------------------------
# SongDNA (aggregate root)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SongDNA:
    """Root aggregate for the EchoInsight domain model.

    ``SongDNA`` is the **sole aggregate root** in the system.  Every
    capability — comparison, recommendation, manipulation, visualisation —
    loads a ``SongDNA`` as its unit of access.  The structure is loaded
    eagerly as part of the aggregate because structural information is
    required by every downstream consumer.

    ``SongDNA`` is a **frozen dataclass** — immutable, hashable, and
    directly serialisable via ``dataclasses.asdict()``.

    ``SongDNA`` does NOT contain:
    - Comparison results (the Comparison Engine owns those)
    - Recommendation results (the Recommendation Engine owns those)
    - Embeddings (future ``EmbeddingDNA`` aggregate)
    - Stem data (future ``StemDNA`` aggregate)
    - User annotations (future ``UserData`` aggregate)
    - Raw frame arrays (those are in NPZ files)

    Attributes
    ----------
    metadata : SongMetadata
        Static track-level information (duration, sample rate, etc.).
    summary : SongSummary
        Song-level aggregate statistics for fast filtering.
    structure : StructureDNA
        Macro-structural organisation.  Loaded eagerly because every
        downstream consumer needs structural information.  **Composition,
        not reference** — the structure lives inside the aggregate.
    timeline : Optional[Timeline]
        Beat-aligned timeline (beats, downbeats, bars) in seconds.
        ``None`` when only summary statistics are available.
    frames : Optional[FrameReference]
        Reference to binary frame-level data (NPZ).  ``None`` when only
        summary statistics are available (lightweight extraction).
    manifest : Optional[AnalysisManifest]
        Provenance record of the extraction.  ``None`` for legacy or
        in-memory-only extractions.
    schema_version : str
        Schema version for migration, e.g. ``"2.0.0"``.

    Notes on imports
    ----------------
    ``FrameReference`` and ``AnalysisManifest`` are imported at module
    level.  There is no circular dependency: ``frame_reference.py``
    imports only ``identifiers.py``, which imports only the standard
    library.
    """

    metadata: SongMetadata
    summary: SongSummary
    structure: StructureDNA
    timeline: Optional[Timeline] = None
    frames: Optional[FrameReference] = None
    manifest: Optional[AnalysisManifest] = None
    schema_version: str = "2.2.0"  # 2.2.0: tempo_is_fallback, optional tempo confidence, Timeline.downbeats_assumed

    def __post_init__(self) -> None:
        # Validate that the structure's total_duration matches the
        # metadata duration.  This is the primary cross-field invariant.
        if abs(self.structure.total_duration - self.metadata.duration) > 1e-3:
            raise ValueError(
                f"structure.total_duration ({self.structure.total_duration}) "
                f"does not match metadata.duration "
                f"({self.metadata.duration})"
            )

        # Validate song_id consistency: the metadata song_id should match
        # the song_id on every segment.
        sid = self.metadata.song_id
        for i, seg in enumerate(self.structure.segments):
            if seg.song_id != sid:
                raise ValueError(
                    f"Segment at index {i} has song_id {seg.song_id}, "
                    f"expected {sid}"
                )

        # Validate timeline coverage when a timeline is present: the last
        # bar must extend to the track duration (the tail bar is built to
        # cover the full track in Timeline.build).
        if self.timeline is not None:
            if abs(
                self.timeline.bars[-1].end_time - self.metadata.duration
            ) > 1e-3:
                raise ValueError(
                    f"timeline.bars[-1].end_time "
                    f"({self.timeline.bars[-1].end_time}) does not match "
                    f"metadata.duration ({self.metadata.duration})"
                )

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize this ``SongDNA`` to a JSON string.

        The output is lossless: :meth:`from_json` reconstructs a
        ``SongDNA`` that compares equal (``==``) to the original.
        """
        return json.dumps(
            dataclasses.asdict(self),
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SongDNA":
        """Deserialize a ``SongDNA`` from JSON produced by :meth:`to_json`.

        Parameters
        ----------
        json_str : str
            JSON string previously produced by :meth:`to_json`.

        Returns
        -------
        SongDNA
            Reconstructed aggregate root.  Comprehensively equal to the
            original (``==``), including nested structures, enums, and
            identifiers.
        """
        data = migrate_song_dna_dict(json.loads(json_str))
        return _reconstruct_dataclass(cls, data)
