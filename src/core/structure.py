"""
structure.py — Structural segmentation data types: StructureDNA, SegmentDNA,
and all supporting value objects.

This is the most important module in the EchoInsight domain layer.  Every
capability that involves comparison, recommendation, manipulation, or
visualisation depends on the types defined here.

Ownership diagram::

    SongDNA  (aggregate root)
     └── StructureDNA  (entity, part of SongDNA aggregate)
          ├── SegmentDNA[]  (value objects — owned by StructureDNA)
          │    ├── SegmentTiming
          │    ├── SegmentDSP
          │    │    ├── SegmentRhythm
          │    │    ├── SegmentHarmony
          │    │    └── SegmentTimbre
          │    └── SegmentContext
          ├── BoundaryConfidence[]
          └── AlternativeSegmentation[]

Layer rules
-----------
- ``src.core.structure`` imports **only** Python stdlib + ``src.core.identifiers``
  and ``src.core.enums``.
- It must **never** import ``src.analysis``, ``src.config``, ``src.app``,
  or any external package.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

from src.core.enums import LabelType, RepetitionRole
from src.core.identifiers import SegmentIndex, SongID, URI

# ---------------------------------------------------------------------------
# Segment-level value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SegmentTiming:
    """Temporal position and duration of a single structural segment.

    Every time value is in **seconds** measured from the start of the song.
    This is the canonical time representation — frame indices and beat
    indices are derived references.

    Attributes
    ----------
    start_time : float
        Start time in seconds (>= 0).
    end_time : float
        End time in seconds (<= song duration, > start_time).
    duration : float
        Pre-computed duration in seconds (end_time - start_time).
    start_beat : Optional[int]
        0-based beat index at the segment start, if a beat grid is available.
        ``None`` when beat tracking was not performed or is unreliable.
    end_beat : Optional[int]
        0-based beat index at the segment end.  ``None`` when beat tracking
        was not performed or is unreliable.
    bar_count : Optional[int]
        Estimated number of bars in this segment.  ``None`` when tempo
        estimation is unreliable or the segment is too short.

    Invariants
    ----------
    - start_time >= 0
    - end_time > start_time
    - duration == end_time - start_time (validated in __post_init__)
    - start_beat >= 0 when present
    - end_beat >= 0 when present
    - bar_count >= 1 when present
    """

    start_time: float
    end_time: float
    duration: float = dataclasses.field(init=False)
    start_beat: Optional[int] = None
    end_beat: Optional[int] = None
    bar_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise ValueError(
                f"start_time must be >= 0, got {self.start_time}"
            )
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be > "
                f"start_time ({self.start_time})"
            )
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "duration", self.end_time - self.start_time)
        if self.start_beat is not None and self.start_beat < 0:
            raise ValueError(
                f"start_beat must be >= 0, got {self.start_beat}"
            )
        if self.end_beat is not None and self.end_beat < 0:
            raise ValueError(
                f"end_beat must be >= 0, got {self.end_beat}"
            )
        if self.bar_count is not None and self.bar_count < 1:
            raise ValueError(
                f"bar_count must be >= 1, got {self.bar_count}"
            )
        if self.start_beat is not None and self.end_beat is not None:
            if self.end_beat <= self.start_beat:
                raise ValueError(
                    f"end_beat ({self.end_beat}) must be > "
                    f"start_beat ({self.start_beat})"
                )


@dataclasses.dataclass(frozen=True)
class SegmentRhythm:
    """Rhythmic characteristics of a single segment.

    Attributes
    ----------
    tempo : float
        Estimated tempo in BPM within this segment.  Positive value.
    tempo_confidence : float
        Confidence of the tempo estimate in 0.0–1.0.
    onset_strength_mean : float
        Mean onset strength — a measure of how percussive the segment is.
        Higher values indicate more rhythmic activity (e.g. drums).
        Always >= 0.
    """

    tempo: float
    tempo_confidence: float
    onset_strength_mean: float

    def __post_init__(self) -> None:
        if self.tempo <= 0:
            raise ValueError(f"tempo must be positive, got {self.tempo}")
        if not 0.0 <= self.tempo_confidence <= 1.0:
            raise ValueError(
                f"tempo_confidence must be in [0, 1], "
                f"got {self.tempo_confidence}"
            )
        if self.onset_strength_mean < 0:
            raise ValueError(
                f"onset_strength_mean must be >= 0, "
                f"got {self.onset_strength_mean}"
            )


@dataclasses.dataclass(frozen=True)
class SegmentHarmony:
    """Harmonic and tonal characteristics of a single segment.

    Attributes
    ----------
    key : Optional[str]
        Detected key within this segment, e.g. ``"C major"``, ``"A minor"``.
        ``None`` when key estimation is unreliable.
    key_confidence : float
        Confidence of the key estimate in 0.0–1.0.
    chroma_mean : Tuple[float, ...]
        12-element mean chroma vector — the average pitch-class energy
        across the segment.  Each element is >= 0.
    chroma_std : Tuple[float, ...]
        12-element standard deviation of the chroma vector — captures
        harmonic variability within the segment.
    tonnetz_mean : Tuple[float, ...]
        6-element mean tonnetz (tonal centroid) vector.

    Invariants
    ----------
    - chroma_mean has exactly 12 elements
    - chroma_std has exactly 12 elements
    - tonnetz_mean has exactly 6 elements
    """

    key: Optional[str] = None
    key_confidence: float = 0.0
    chroma_mean: Tuple[float, ...] = ()
    chroma_std: Tuple[float, ...] = ()
    tonnetz_mean: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.key is not None and not self.key.strip():
            raise ValueError("key must be non-empty when provided")
        if not 0.0 <= self.key_confidence <= 1.0:
            raise ValueError(
                f"key_confidence must be in [0, 1], "
                f"got {self.key_confidence}"
            )
        if self.chroma_mean and len(self.chroma_mean) != 12:
            raise ValueError(
                f"chroma_mean must have 12 elements, "
                f"got {len(self.chroma_mean)}"
            )
        if self.chroma_std and len(self.chroma_std) != 12:
            raise ValueError(
                f"chroma_std must have 12 elements, "
                f"got {len(self.chroma_std)}"
            )
        if self.tonnetz_mean and len(self.tonnetz_mean) != 6:
            raise ValueError(
                f"tonnetz_mean must have 6 elements, "
                f"got {len(self.tonnetz_mean)}"
            )


@dataclasses.dataclass(frozen=True)
class SegmentTimbre:
    """Timbral (spectral texture) characteristics of a single segment.

    Attributes
    ----------
    rms_mean : float
        Mean RMS energy within the segment — a proxy for perceived loudness.
        Always >= 0.
    rms_std : float
        Standard deviation of RMS energy — captures energy variability.
        Always >= 0.
    spectral_centroid_mean : float
        Mean spectral centroid in Hz — correlates with perceived brightness.
        Always >= 0.
    spectral_centroid_std : float
        Standard deviation of spectral centroid in Hz.
    spectral_bandwidth_mean : float
        Mean spectral bandwidth in Hz — correlates with spectral spread.
        Always >= 0.
    mfcc_mean : Tuple[float, ...]
        MFCC coefficients averaged over the segment (typically 13 coefficients).
    mfcc_std : Tuple[float, ...]
        Standard deviation of MFCC coefficients — captures timbral variation.
    zero_crossing_rate_mean : float
        Mean zero-crossing rate — correlates with noisiness/percussiveness.
        In 0.0–1.0 range.
    spectral_rolloff_mean : float
        Mean spectral rolloff frequency in Hz — the frequency below which
        a given percentage (typically 85%) of spectral energy resides.
    spectral_contrast_mean : Tuple[float, ...]
        Mean spectral contrast per octave band (typically 7 bands).

    Invariants
    ----------
    - mfcc_mean has 13 elements when present
    - mfcc_std has 13 elements when present
    - spectral_contrast_mean has 7 elements when present
    """

    rms_mean: float = 0.0
    rms_std: float = 0.0
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    mfcc_mean: Tuple[float, ...] = ()
    mfcc_std: Tuple[float, ...] = ()
    zero_crossing_rate_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_contrast_mean: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.rms_mean < 0:
            raise ValueError(
                f"rms_mean must be >= 0, got {self.rms_mean}"
            )
        if self.rms_std < 0:
            raise ValueError(
                f"rms_std must be >= 0, got {self.rms_std}"
            )
        if self.spectral_centroid_mean < 0:
            raise ValueError(
                f"spectral_centroid_mean must be >= 0, "
                f"got {self.spectral_centroid_mean}"
            )
        if self.spectral_centroid_std < 0:
            raise ValueError(
                f"spectral_centroid_std must be >= 0, "
                f"got {self.spectral_centroid_std}"
            )
        if self.spectral_bandwidth_mean < 0:
            raise ValueError(
                f"spectral_bandwidth_mean must be >= 0, "
                f"got {self.spectral_bandwidth_mean}"
            )
        if not 0.0 <= self.zero_crossing_rate_mean <= 1.0:
            raise ValueError(
                f"zero_crossing_rate_mean must be in [0, 1], "
                f"got {self.zero_crossing_rate_mean}"
            )
        if self.spectral_rolloff_mean < 0:
            raise ValueError(
                f"spectral_rolloff_mean must be >= 0, "
                f"got {self.spectral_rolloff_mean}"
            )
        if self.mfcc_mean and len(self.mfcc_mean) != 13:
            raise ValueError(
                f"mfcc_mean must have 13 elements, "
                f"got {len(self.mfcc_mean)}"
            )
        if self.mfcc_std and len(self.mfcc_std) != 13:
            raise ValueError(
                f"mfcc_std must have 13 elements, "
                f"got {len(self.mfcc_std)}"
            )
        if self.spectral_contrast_mean and len(self.spectral_contrast_mean) != 7:
            raise ValueError(
                f"spectral_contrast_mean must have 7 elements, "
                f"got {len(self.spectral_contrast_mean)}"
            )


@dataclasses.dataclass(frozen=True)
class SegmentDSP:
    """Aggregate DSP fingerprint for a single structural segment.

    ``SegmentDSP`` composes three sub-components — rhythm, harmony, and
    timbre — into a single value object.  This decomposition prevents the
    DSP fields from becoming a flat dumping ground while keeping the public
    API simple::

        segment.dsp.rhythm.tempo       # rhythmic
        segment.dsp.harmony.key        # harmonic
        segment.dsp.timbre.rms_mean    # timbral

    Attributes
    ----------
    rhythm : SegmentRhythm
        Rhythmic characteristics of the segment.
    harmony : SegmentHarmony
        Harmonic and tonal characteristics.
    timbre : SegmentTimbre
        Timbral and spectral characteristics.
    """

    rhythm: SegmentRhythm
    harmony: SegmentHarmony
    timbre: SegmentTimbre


@dataclasses.dataclass(frozen=True)
class SegmentContext:
    """Structural context of a segment within its song.

    ``SegmentContext`` captures the segment's position in the arrangement,
    its label, its confidence, and its relationships to other segments.

    Attributes
    ----------
    index : SegmentIndex
        0-based position in ``StructureDNA.segments``.
    label : LabelType
        Structural role (verse, chorus, bridge, etc.).
    label_confidence : float
        Confidence of the label assignment in 0.0–1.0.
    repetition_role : RepetitionRole
        How this segment relates to other segments in the same song.
    repetition_group : Optional[str]
        Group identifier for repeated/varied segments, e.g. ``"verse_1"``,
        ``"chorus_3"``.  ``None`` when the segment is unique.
    novelty_score : float
        How different this segment is from the preceding segment, in 0.0–1.0.
        Higher values indicate a more dramatic structural shift.
    self_similarity_score : float
        Similarity to the most similar other segment in the song, in 0.0–1.0.
        A score of 1.0 indicates an exact duplicate.
    """

    index: SegmentIndex
    label: LabelType = LabelType.UNKNOWN
    label_confidence: float = 0.0
    repetition_role: RepetitionRole = RepetitionRole.UNIQUE
    repetition_group: Optional[str] = None
    novelty_score: float = 0.0
    self_similarity_score: float = 0.0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")
        if not 0.0 <= self.label_confidence <= 1.0:
            raise ValueError(
                f"label_confidence must be in [0, 1], "
                f"got {self.label_confidence}"
            )
        if not 0.0 <= self.novelty_score <= 1.0:
            raise ValueError(
                f"novelty_score must be in [0, 1], "
                f"got {self.novelty_score}"
            )
        if not 0.0 <= self.self_similarity_score <= 1.0:
            raise ValueError(
                f"self_similarity_score must be in [0, 1], "
                f"got {self.self_similarity_score}"
            )


# ---------------------------------------------------------------------------
# SegmentDNA
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SegmentDNA:
    """A single structural segment of a song.

    ``SegmentDNA`` represents one atomic structural unit — a verse, a chorus,
    a bridge, etc.  It contains everything the comparison and recommendation
    engines need to operate on individual musical ideas.

    This is a **value object**: it is immutable, comparable by value, and can
    only exist within a ``StructureDNA``.

    ``SegmentDNA`` does NOT know:
    - Its parent ``StructureDNA`` (that's the container's responsibility)
    - Which other segments are similar (the Comparison Engine owns that)
    - Raw frame arrays (those live in NPZ files)
    - User annotations (future ``UserData`` aggregate)

    Attributes
    ----------
    song_id : SongID
        Identifier of the song this segment belongs to.  Used by the
        comparison and recommendation engines to link back to the song
        without loading the full ``SongDNA`` aggregate.
    timing : SegmentTiming
        Temporal position and duration.
    dsp : SegmentDSP
        Rhythmic, harmonic, and timbral fingerprint.
    context : SegmentContext
        Structural context, label, repetition relationships.
    schema_version : str
        Schema version for migration, e.g. ``"2.0.0"``.
    """

    song_id: SongID
    timing: SegmentTiming
    dsp: SegmentDSP
    context: SegmentContext
    schema_version: str = "2.0.0"

    def __post_init__(self) -> None:
        if not self.song_id:
            raise ValueError("song_id must be non-empty")


# ---------------------------------------------------------------------------
# BoundaryConfidence
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BoundaryConfidence:
    """Confidence for a single segment boundary.

    One ``BoundaryConfidence`` exists for each boundary *between* segments.
    The confidence reflects how certain the segmentation algorithm is that
    a structural boundary exists at this time point.

    Attributes
    ----------
    time : float
        Boundary time in seconds from song start.
    confidence : float
        Confidence in 0.0–1.0.  Low values suggest the boundary may be
        spurious or the result of noise.
    method : str
        The algorithm or feature that detected this boundary, e.g.
        ``"novelty_curve"``, ``"chroma_change"``, ``"mfcc_change"``,
        ``"onset_clustering"``.
    """

    time: float
    confidence: float
    method: str = "unknown"

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"time must be >= 0, got {self.time}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if not self.method.strip():
            raise ValueError("method must be non-empty")


# ---------------------------------------------------------------------------
# AlternativeSegmentation
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AlternativeSegmentation:
    """An alternative segmentation of the same song.

    The segmentation algorithm may produce multiple granularities or the
    user may manually adjust boundaries.  Each alternative is stored as a
    separate ``AlternativeSegmentation``, preserving the ability to switch
    between granularities without re-analysis.

    The primary segmentation lives in ``StructureDNA.segments``.
    All alternatives live in ``StructureDNA.alternative_segmentations``.

    Attributes
    ----------
    label : str
        Human-readable label, e.g. ``"default"``, ``"fine_grained"``,
        ``"coarse"``, ``"user_adjusted"``.
    segments : Tuple[SegmentDNA, ...]
        The alternative segment list.  Must be non-empty and temporally
        contiguous (segment[n].end_time == segment[n+1].start_time for
        all n).
    confidence : float
        Overall confidence of this segmentation in 0.0–1.0.
    """

    label: str
    segments: Tuple[SegmentDNA, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must be non-empty")
        if not self.segments:
            raise ValueError("segments must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        # Validate temporal contiguity
        for i in range(len(self.segments) - 1):
            current_end = self.segments[i].timing.end_time
            next_start = self.segments[i + 1].timing.start_time
            if abs(current_end - next_start) > 1e-6:
                raise ValueError(
                    f"Segments {i} and {i + 1} are not temporally contiguous: "
                    f"segment[{i}].end_time ({current_end}) != "
                    f"segment[{i + 1}].start_time ({next_start})"
                )


# ---------------------------------------------------------------------------
# SSMReference
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SSMReference:
    """Lightweight reference to a self-similarity matrix stored as NPZ.

    The self-similarity matrix (SSM) is a 2D array where element (i, j)
    is the similarity between frame i and frame j.  It is never stored in
    the domain — only referenced.

    Attributes
    ----------
    uri : URI
        Relative URI (within the data root) to the NPZ file containing
        the SSM array.
    num_frames : int
        Number of frames in the SSM (the matrix is num_frames × num_frames).
    hop_length : int
        Hop length in samples used for frame extraction.
    sample_rate : int
        Sample rate in Hz used for extraction.
    """

    uri: URI
    num_frames: int
    hop_length: int
    sample_rate: int

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("uri must be non-empty")
        if self.num_frames <= 0:
            raise ValueError(
                f"num_frames must be positive, got {self.num_frames}"
            )
        if self.hop_length <= 0:
            raise ValueError(
                f"hop_length must be positive, got {self.hop_length}"
            )
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )


# ---------------------------------------------------------------------------
# StructureDNA (owned by SongDNA)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StructureDNA:
    """Macro-structural organisation of a song.

    ``StructureDNA`` represents *how the song is organised over time*.  It
    owns a list of ``SegmentDNA`` values and the metadata about the
    segmentation itself.

    ``StructureDNA`` does NOT contain:
    - DSP data at the song level (that belongs in ``SongSummary``)
    - Frame-level feature arrays (those are in ``FrameReference``)
    - Comparison results (the Comparison Engine owns those)
    - User annotations (future ``UserData`` aggregate)

    Attributes
    ----------
    segments : Tuple[SegmentDNA, ...]
        The primary segmentation — an ordered, non-empty, temporally
        contiguous list of segments.
    total_duration : float
        Sum of segment durations.  Should match the song's total duration
        (validated in ``__post_init__``).
    boundary_confidences : Tuple[BoundaryConfidence, ...]
        Confidence values for each boundary *between* segments.  There are
        ``num_segments - 1`` boundary confidences.
    alternative_segmentations : Tuple[AlternativeSegmentation, ...]
        Zero or more alternative segmentations.
    ssm_reference : Optional[SSMReference]
        Reference to the self-similarity matrix NPZ, if computed.  ``None``
        when SSM was not computed.
    schema_version : str
        Schema version for migration, e.g. ``"2.0.0"``.
    num_segments : int
        Number of segments (convenience, derived from ``len(segments)``).
        Computed automatically — do not pass in constructor.
    """

    segments: Tuple[SegmentDNA, ...]
    total_duration: float
    boundary_confidences: Tuple[BoundaryConfidence, ...] = ()
    alternative_segmentations: Tuple[AlternativeSegmentation, ...] = ()
    ssm_reference: Optional[SSMReference] = None
    schema_version: str = "2.0.0"
    num_segments: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        # Validate segments exist
        if not self.segments:
            raise ValueError("segments must be non-empty")

        object.__setattr__(self, "num_segments", len(self.segments))

        # Validate temporal contiguity
        computed_duration = 0.0
        for i in range(len(self.segments)):
            seg = self.segments[i]
            computed_duration += seg.timing.duration

            # Validate that segment indices match their position
            if seg.context.index != i:
                raise ValueError(
                    f"Segment at position {i} has context.index "
                    f"{seg.context.index}.  Expected {i}."
                )

            # Validate contiguity between segments
            if i > 0:
                prev_end = self.segments[i - 1].timing.end_time
                curr_start = seg.timing.start_time
                if abs(prev_end - curr_start) > 1e-6:
                    raise ValueError(
                        f"Segments {i - 1} and {i} are not contiguous: "
                        f"segments[{i - 1}].end_time ({prev_end}) != "
                        f"segments[{i}].start_time ({curr_start})"
                    )

        # Validate total_duration
        if abs(computed_duration - self.total_duration) > 1e-3:
            raise ValueError(
                f"total_duration ({self.total_duration}) does not match "
                f"computed segment durations ({computed_duration})"
            )

        # Validate boundary confidences
        expected_boundaries = len(self.segments) - 1
        if len(self.boundary_confidences) != expected_boundaries:
            raise ValueError(
                f"Expected {expected_boundaries} boundary_confidences "
                f"(one per boundary between segments), "
                f"got {len(self.boundary_confidences)}"
            )