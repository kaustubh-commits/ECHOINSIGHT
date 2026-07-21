"""
dna_schema.py — Nested SongDNA dataclass definition.

Follows the EchoInsight architecture by decomposing a song's acoustic
fingerprint into six specialised sub-dataclasses:

  * SongMetadata    — static track-level info (duration, filename)
  * RhythmDNA       — rhythmic / temporal features (tempo, beats)
  * TimbreDNA       — spectral / timbral features (MFCC, centroid, …)
  * TonalDNA        — harmonic / tonal features (key, chroma, …)
  * IntelligenceDNA — learned embeddings (future)
  * StemDNA         — per-stem features, e.g. vocals, drums (future)

Each sub-dataclass is frozen for immutability and supports JSON
serialisation via ``dataclasses.asdict()``.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

from .analysis_manifest import AnalysisManifest
from .frame_reference import FrameReference


# ---------------------------------------------------------------------------
# Sub-DNA dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SongMetadata:
    """Static metadata describing the track itself.

    Attributes
    ----------
    duration : float
        Total duration of the track in seconds.
    filename : str
        Original filename or path of the source audio file.
    """

    duration: float
    filename: str = ""

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")


@dataclasses.dataclass(frozen=True)
class RhythmDNA:
    """Rhythmic and temporal fingerprint of the track.

    Attributes
    ----------
    tempo : float
        Estimated tempo in beats per minute (BPM).
    confidence : float
        Confidence of the tempo estimate in 0.0–1.0.
    beat_frames : tuple[int, ...]
        Beat positions in frame indices, forming a universal coordinate
        system with frame-level features.
        Convert to seconds via: ``time_s = frame_idx * hop_length / sample_rate``.
    beat_count : int
        Number of detected beats.
    """

    tempo: float
    confidence: float = 0.0
    beat_frames: Tuple[int, ...] = ()
    beat_count: int = 0

    def __post_init__(self) -> None:
        if self.tempo <= 0:
            raise ValueError(f"tempo must be positive, got {self.tempo}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if len(self.beat_frames) != self.beat_count:
            raise ValueError(
                f"beat_count {self.beat_count} != "
                f"len(beat_frames) {len(self.beat_frames)}"
            )


@dataclasses.dataclass(frozen=True)
class TimbreDNA:
    """Spectral / timbral fingerprint of the track.

    Attributes
    ----------
    rms_energy_mean : float
        Mean root-mean-square energy across the track — a proxy
        for perceived loudness.
    spectral_centroid_mean : float
        Mean spectral centroid in Hz — correlates with perceived
        "brightness" or timbral sharpness.
    mfcc_mean : List[float]
        Mean Mel-Frequency Cepstral Coefficients (13 coefficients)
        aggregated over time.  Captures timbral/textural information.
    spectral_bandwidth_mean : float
        Mean spectral bandwidth in Hz — correlates with spectral spread.
    """

    rms_energy_mean: float
    spectral_centroid_mean: float
    mfcc_mean: List[float]
    spectral_bandwidth_mean: float = 0.0

    def __post_init__(self) -> None:
        if not self.mfcc_mean:
            raise ValueError("mfcc_mean must be a non-empty list")


@dataclasses.dataclass(frozen=True)
class TonalDNA:
    """Harmonic / tonal fingerprint of the track.

    Attributes
    ----------
    key : Optional[str]
        Detected musical key, e.g. "C major", "A minor".  ``None`` when
        estimation is unreliable.
    key_confidence : float
        Confidence of the key estimate in 0.0–1.0 (Krumhansl-Schmuckler
        correlation).
    chroma_mean : List[float]
        12-bin mean chroma vector — the average pitch-class energy across
        the track.
    tuning_offset : float
        Deviation from A440 in cents (positive = sharp).  Essential for
        comparing songs recorded at non-standard tunings.
    """

    key: Optional[str] = None
    key_confidence: float = 0.0
    chroma_mean: List[float] = dataclasses.field(default_factory=lambda: [0.0] * 12)
    tuning_offset: float = 0.0

    def __post_init__(self) -> None:
        if len(self.chroma_mean) != 12:
            raise ValueError(
                f"chroma_mean must have 12 bins, got {len(self.chroma_mean)}"
            )


@dataclasses.dataclass(frozen=True)
class Segment:
    """A single structural section of the song.

    Attributes
    ----------
    label : str
        Section label — algorithmic ("A", "B", "C") or user-assigned.
    start_time : float
        Start time in seconds.
    end_time : float
        End time in seconds.
    """

    label: str = ""
    start_time: float = 0.0
    end_time: float = 0.0


@dataclasses.dataclass(frozen=True)
class StructureDNA:
    """Macro-structural organisation of the track.

    Attributes
    ----------
    segments : List[Segment]
        Detected or user-defined structural sections.
    """

    segments: List[Segment] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class IntelligenceDNA:
    """Learned embeddings from pre-trained audio models (future).

    Placeholder — populated during the Embeddings phase.  Not required
    for deterministic DSP or comparison engine v1/v2.
    """


@dataclasses.dataclass(frozen=True)
class StemDNA:
    """Per-stem acoustic features (future).

    Placeholder for Demucs-based source-separation results:
    vocal, drums, bass, other — each with its own sub-fingerprint.
    """


# ---------------------------------------------------------------------------
# Top-level SongDNA
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SongDNA:
    """Immutable, nested fingerprint of a song's acoustic characteristics.

    Composes six specialised sub-dataclasses that together form the
    complete EchoInsight SongDNA.

    Attributes
    ----------
    metadata : SongMetadata
        Static track-level information.
    rhythm : RhythmDNA
        Rhythmic / temporal features.
    timbre : TimbreDNA
        Spectral / timbral features.
    tonal : TonalDNA
        Harmonic / tonal features.
    intelligence : IntelligenceDNA
        Learned embeddings (future).
    stem : StemDNA
        Per-stem features (future).
    frames : Optional[FrameReference]
        Reference to binary frame-level data, if extracted.
        ``None`` when only summary statistics are available.
    manifest : Optional[AnalysisManifest]
        Provenance record of the extraction, if available.
        ``None`` for legacy or in-memory-only extractions.
    """

    metadata: SongMetadata
    rhythm: RhythmDNA
    timbre: TimbreDNA
    tonal: TonalDNA
    structure: StructureDNA
    intelligence: IntelligenceDNA
    stem: StemDNA
    frames: Optional[FrameReference] = None
    manifest: Optional[AnalysisManifest] = None