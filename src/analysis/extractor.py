"""
extractor.py — Feature extraction pipeline that converts a raw
audio file into a v2 SongDNA aggregate (SongDNA + StructureDNA +
SegmentDNA[] + Timeline).

The pipeline is divided into deterministic stages:

1.  Load Audio (locked to 22050 Hz)
2.  Extract Frame Features
3.  Detect Beats (+ build Timeline)
4.  Segment Structure (v2 branch) — or single-global-segment (v1 branch)
5.  Compute Per-Segment DSP (v2 branch)
6.  Build Manifest
7.  Assemble SongDNA v2
8.  (Optional) 4-Stem Separation + Stem Analysis — ``extract_full_pipeline`` only

Public API
----------
- ``extract_song_dna(file_path)``   — in-memory extraction (backward compatible)
- ``extract_and_save(file_path, data_root)`` — extraction + NPZ persistence
- ``extract_full_pipeline(file_path, data_root, separate_stems)`` — unified
  Pillar 1 (SongDNA + Timeline) + optional Pillar 2 (4-stem separation)

Migration
---------
A developer-only ``_BUILD_V2`` flag selects between the two branches.
Both branches return the **v2** ``SongDNA`` type:

- ``_BUILD_V2 = False`` (default): the legacy v1 global features are
  adapted into a v2 ``SongDNA`` with a **single global segment**
  covering the whole track.  This keeps every consumer on the v2 type
  while the rest of the system migrates.
- ``_BUILD_V2 = True``: full structural segmentation via
  ``segmentation.detect_segments`` and per-segment DSP via
  ``segment_dsp.compute_segment_dsp``.

See ``docs/MIGRATION.md`` for removal criteria.
"""

from __future__ import annotations

import datetime
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from src.core.analysis_manifest import AnalysisManifest, compute_config_hash
from src.core.enums import LabelType
from src.core.frame_reference import FrameReference
from src.core.identifiers import SegmentIndex, SongID, generate_song_id
from src.core.package import SongPackage
from src.core.song_dna import SongDNA, SongMetadata, SongSummary
from src.core.structure import (
    BoundaryConfidence,
    SegmentContext,
    SegmentDNA,
    SegmentDSP,
    SegmentHarmony,
    SegmentRhythm,
    SegmentTimbre,
    SegmentTiming,
    StructureDNA,
)
from src.core.timeline import Timeline
from src.analysis.frame_store import FrameStore
from src.analysis.segmentation import detect_segments
from src.analysis.segment_dsp import compute_segment_dsp
from src.analysis.separator import EchoSeparator
from src.analysis.stem_analyzer import StemAnalysisResult, StemAnalyzer
from src.core.stems import SeparatedTrack

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration flag
# ---------------------------------------------------------------------------

# MIGRATION: This flag determines which extraction pipeline to use.
# Removal criteria: delete this flag and the v1 branch when:
# - Workspace consumes SongDNA v2
# - Visualization consumes SongDNA v2
# - Comparison consumes SongDNA v2
# - All extraction tests pass using v2
# - No production code imports dna_schema.py
_BUILD_V2 = True  # ← v2 pipeline enabled by default

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 22050          # Locked. Do not parameterize.

# Krumhansl-Schmuckler key profiles (Krumhansl & Kessler, 1982)
_MAJOR_PROFILE: np.ndarray = np.array(
    [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
        2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ]
)

_MINOR_PROFILE: np.ndarray = np.array(
    [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
        2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
    ]
)

_KEY_NAMES: List[str] = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]

N_MFCC: int = 13          # Number of MFCC coefficients to keep
HOP_LENGTH: int = 512     # Hop length (samples) for frame-level analysis
N_FFT: int = 2048         # FFT window size
FEATURE_SET_VERSION: str = "v2"
EXTRACTOR_VERSION: str = "2.2.0"  # 2.2.0: key estimation fixed (keys were mirrored)
SCHEMA_VERSION: str = "1.0.0"  # AnalysisManifest schema version


# ======================================================================
# Stage 1 — Load Audio
# ======================================================================


def _load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return the time series and sample rate.

    The sample rate is **locked to 22050 Hz** per the Extraction
    Contract.  Do not parameterize this.

    Parameters
    ----------
    file_path : str
        Path to a supported audio file (MP3, WAV, FLAC, OGG, M4A, etc.).

    Returns
    -------
    tuple[np.ndarray, int]
        ``(y, sr)`` where ``y`` is the mono audio time series and
        ``sr`` is 22050.
    """
    logger.info("Loading audio: %s", file_path)
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    logger.debug("Loaded: sr=%d, samples=%d, duration=%.2fs",
                 sr, len(y), len(y) / sr)
    return y, sr


# ======================================================================
# Stage 2 — Frame Feature Extraction
# ======================================================================


def _extract_frame_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Extract all frame-level DSP features for a track.

    Each array preserves librosa's natural output shape — no flattening.

    Returns
    -------
    dict[str, np.ndarray]
        ``"rms"``        — (n_frames,)   — RMS energy per frame
        ``"centroid"``   — (n_frames,)   — spectral centroid (Hz)
        ``"bandwidth"``  — (n_frames,)   — spectral bandwidth (Hz)
        ``"rolloff"``    — (n_frames,)   — spectral rolloff (Hz)
        ``"zcr"``        — (n_frames,)   — zero-crossing rate
        ``"mfcc"``       — (13, n_frames) — MFCC coefficients
        ``"chroma"``     — (12, n_frames) — chromagram (CQT)
        ``"contrast"``   — (7, n_frames)  — spectral contrast
        ``"tonnetz"``    — (6, n_frames)  — tonal centroid
    """
    logger.debug("Extracting frame features (hop_length=%d)", HOP_LENGTH)
    features: Dict[str, np.ndarray] = {}

    # --- 1-D features ---
    features["rms"] = librosa.feature.rms(y=y, hop_length=HOP_LENGTH).flatten()
    features["centroid"] = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    ).flatten()
    features["bandwidth"] = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    ).flatten()
    features["rolloff"] = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    ).flatten()
    features["zcr"] = librosa.feature.zero_crossing_rate(
        y=y, hop_length=HOP_LENGTH
    ).flatten()

    # --- 2-D features ---
    features["mfcc"] = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )  # shape (13, n_frames)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["chroma"] = chroma  # shape (12, n_frames)

    features["contrast"] = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )  # shape (7, n_frames)

    features["tonnetz"] = librosa.feature.tonnetz(
        chroma=chroma, sr=sr
    )  # shape (6, n_frames)

    logger.debug("Extracted %d feature arrays, %d frames each",
                 len(features), features["rms"].shape[0])
    return features


# ======================================================================
# Stage 3 — Beat Detection + Timeline
# ======================================================================


FALLBACK_TEMPO: float = 120.0
"""Placeholder tempo stored when no beats are detected (flagged as fallback)."""


def _detect_beats(y: np.ndarray, sr: int) -> Tuple[float, bool, Tuple[float, ...]]:
    """Detect beat positions (in seconds) and estimate tempo.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.

    Returns
    -------
    tuple[float, bool, tuple[float, ...]]
        ``(tempo, is_fallback, beat_times)``.
        ``tempo`` is in BPM, rounded to 2 decimal places.
        ``is_fallback`` is True when no beats were found and ``tempo`` is
        ``FALLBACK_TEMPO`` rather than a measurement.
        ``beat_times`` are beat positions in **seconds** from the start
        of the track.  The beat tracker provides no confidence score, so
        none is reported.
    """
    tempo, beat_times = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=HOP_LENGTH, trim=False, units="time"
    )
    raw_tempo = float(np.round(np.asarray(tempo).flatten()[0], 2))

    if not np.asarray(beat_times).size or raw_tempo <= 0.0:
        # Static tones, pure noise, or zero-energy audio: no tempo exists.
        # Store a placeholder so downstream types stay valid, and flag it.
        final_tempo = FALLBACK_TEMPO
        is_fallback = True
        beat_times_tuple: Tuple[float, ...] = ()
    else:
        final_tempo = raw_tempo
        is_fallback = False
        beat_times_tuple = tuple(
            float(b) for b in np.asarray(beat_times).flatten()
        )

    logger.debug(
        "Tempo: %.2f BPM (fallback=%s), %d beats detected",
        final_tempo,
        is_fallback,
        len(beat_times_tuple),
    )
    return final_tempo, is_fallback, beat_times_tuple


# ======================================================================
# Stage 4 — Global Statistics (shared helpers)
# ======================================================================


def _estimate_key(chroma_mean: np.ndarray) -> Tuple[Optional[str], float]:
    """Krumhansl-Schmuckler key-finding algorithm.

    Parameters
    ----------
    chroma_mean : np.ndarray
        12-bin chroma vector averaged over the track (or segment).

    Returns
    -------
    Tuple[Optional[str], float]
        Detected key label (e.g. ``"C major"``) and correlation confidence
        (0.0–1.0).  Returns ``(None, 0.0)`` if the chroma vector is flat.
    """
    total = float(np.sum(chroma_mean))
    if total < 1e-10:
        return None, 0.0

    chroma_norm = chroma_mean / total

    best_corr = -1.0
    best_key: Optional[str] = None

    for shift in range(12):
        # Rotate so the candidate tonic (pitch class `shift`) sits at index 0,
        # where the profiles hold their tonic weight.
        rolled = np.roll(chroma_norm, -shift)

        corr_major = float(np.corrcoef(rolled, _MAJOR_PROFILE)[0, 1])
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = f"{_KEY_NAMES[shift]} major"

        corr_minor = float(np.corrcoef(rolled, _MINOR_PROFILE)[0, 1])
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f"{_KEY_NAMES[shift]} minor"

    return best_key, best_corr


def _compute_global_statistics(
    features: Dict[str, np.ndarray], y: np.ndarray, sr: int
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Reduce frame-level features to plain summary dicts.

    This helper is shared by both the v1 (single-global-segment) and
    v2 (segmented) paths.  It returns plain dicts rather than v1 domain
    types so the extractor never imports ``dna_schema.py``.

    Parameters
    ----------
    features : dict[str, np.ndarray]
        Frame-level feature dictionary from ``_extract_frame_features``.
    y : np.ndarray
        Audio time series (for tuning estimation).
    sr : int
        Sample rate.

    Returns
    -------
    tuple[dict[str, float], dict[str, object]]
        ``(timbre_dict, tonal_dict)``:
        timbre — rms_energy_mean, spectral_centroid_mean,
                 spectral_bandwidth_mean, mfcc_mean (list)
        tonal  — key, key_confidence, chroma_mean (list), tuning_offset
    """
    timbre: Dict[str, float] = {
        "rms_energy_mean": float(np.mean(features["rms"])),
        "spectral_centroid_mean": float(np.mean(features["centroid"])),
        "spectral_bandwidth_mean": float(np.mean(features["bandwidth"])),
        "mfcc_mean": [float(v) for v in np.mean(features["mfcc"], axis=1)],
    }

    chroma_mean_arr = np.mean(features["chroma"], axis=1)  # (12,)
    key, key_confidence = _estimate_key(chroma_mean_arr)
    tuning_offset = float(librosa.estimate_tuning(y=y, sr=sr))

    tonal: Dict[str, object] = {
        "key": key,
        "key_confidence": key_confidence,
        "chroma_mean": [float(v) for v in chroma_mean_arr],
        "tuning_offset": tuning_offset,
    }

    logger.debug("Key: %s (confidence: %.2f), tuning: %.1f cents",
                 key, key_confidence, tuning_offset)
    return timbre, tonal


def _onset_strength_mean(y: np.ndarray, sr: int,
                         start_s: float = 0.0,
                         end_s: Optional[float] = None) -> float:
    """Mean onset strength over a time range (defaults to the full track)."""
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr) if end_s is not None else len(y)
    slice_y = y[start_sample:end_sample]
    if slice_y.size < sr // 4:
        return 0.0
    onset_env = librosa.onset.onset_strength(
        y=slice_y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median
    )
    return float(np.mean(onset_env)) if onset_env.size else 0.0


# ======================================================================
# Stage 5 — Manifest
# ======================================================================


def _build_manifest(sample_rate: int, hop_length: int) -> AnalysisManifest:
    """Construct an ``AnalysisManifest`` for the current extraction."""
    config_hash = compute_config_hash(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=N_FFT,
        feature_set_version=FEATURE_SET_VERSION,
    )

    # Try to get git info — gracefully handle non-git environments
    git_commit = ""
    git_branch = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if result.returncode == 0:
            git_branch = result.stdout.strip()
    except Exception:
        pass  # Not a git repository or git not installed — leave as ""

    return AnalysisManifest(
        schema_version=SCHEMA_VERSION,
        extractor_version=EXTRACTOR_VERSION,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        librosa_version=librosa.__version__,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=N_FFT,
        feature_set_version=FEATURE_SET_VERSION,
        config_hash=config_hash,
        git_commit=git_commit,
        git_branch=git_branch,
    )


# ======================================================================
# Stage 6 — Assembly (shared builders)
# ======================================================================


def _build_song_summary(
    tempo: float,
    tempo_is_fallback: bool,
    timbre: Dict[str, float],
    tonal: Dict[str, object],
) -> SongSummary:
    """Build a v2 ``SongSummary`` from global statistics."""
    return SongSummary(
        tempo_estimate=tempo,
        tempo_is_fallback=tempo_is_fallback,
        key_estimate=tonal.get("key"),  # type: ignore[arg-type]
        key_confidence=float(tonal.get("key_confidence", 0.0)),
        rms_energy_mean=timbre["rms_energy_mean"],
        spectral_centroid_mean=timbre["spectral_centroid_mean"],
        spectral_bandwidth_mean=timbre["spectral_bandwidth_mean"],
        mfcc_mean=tuple(timbre["mfcc_mean"]),  # type: ignore[arg-type]
        chroma_mean=tuple(tonal.get("chroma_mean", [])),  # type: ignore[arg-type]
        tuning_offset=float(tonal.get("tuning_offset", 0.0)),
    )


def _build_single_segment(
    song_id: SongID,
    duration: float,
    tempo: float,
    tempo_is_fallback: bool,
    timbre: Dict[str, float],
    tonal: Dict[str, object],
    onset_mean: float,
) -> SegmentDNA:
    """Build a single-global-segment ``SegmentDNA`` for the v1 adapter.

    Every global feature is placed into the segment's DSP so that the
    v2 aggregate is fully populated even when structural segmentation is
    disabled.
    """
    timing = SegmentTiming(start_time=0.0, end_time=duration)

    rhythm = SegmentRhythm(
        tempo=tempo,
        tempo_confidence=None,
        onset_strength_mean=onset_mean,
        tempo_is_fallback=tempo_is_fallback,
    )

    chroma_mean = list(tonal.get("chroma_mean", []))
    harmony = SegmentHarmony(
        key=tonal.get("key"),  # type: ignore[arg-type]
        key_confidence=float(tonal.get("key_confidence", 0.0)),
        chroma_mean=tuple(chroma_mean[:12]),
        chroma_std=tuple(0.0 for _ in range(12)),
        tonnetz_mean=tuple(0.0 for _ in range(6)),
    )

    mfcc_mean = list(timbre["mfcc_mean"])
    timbre_obj = SegmentTimbre(
        rms_mean=timbre["rms_energy_mean"],
        rms_std=0.0,
        spectral_centroid_mean=timbre["spectral_centroid_mean"],
        spectral_centroid_std=0.0,
        spectral_bandwidth_mean=timbre["spectral_bandwidth_mean"],
        mfcc_mean=tuple(mfcc_mean[:13]),
        mfcc_std=tuple(0.0 for _ in range(13)),
        zero_crossing_rate_mean=0.0,
        spectral_rolloff_mean=0.0,
        spectral_contrast_mean=tuple(0.0 for _ in range(7)),
        spectral_flux_mean=0.0,
        spectral_flux_std=0.0,
        hpss_envelope_ratio=0.0,
    )

    dsp = SegmentDSP(rhythm=rhythm, harmony=harmony, timbre=timbre_obj)

    context = SegmentContext(
        index=SegmentIndex(0),
        label=LabelType.UNKNOWN,
        label_confidence=0.0,
    )

    return SegmentDNA(
        song_id=song_id,
        timing=timing,
        dsp=dsp,
        context=context,
    )


def _assemble_song_dna_v2(
    song_id: SongID,
    filename: str,
    duration: float,
    sr: int,
    segments: List[SegmentDNA],
    summary: SongSummary,
    timeline: Optional[Timeline] = None,
    frames: Optional[FrameReference] = None,
    manifest: Optional[AnalysisManifest] = None,
) -> SongDNA:
    """Assemble a complete v2 ``SongDNA`` aggregate."""
    # One boundary record per boundary between segments.  Segmentation
    # computes no confidence, so none is stored.
    boundaries: List[BoundaryConfidence] = []
    for i in range(len(segments) - 1):
        boundaries.append(
            BoundaryConfidence(
                time=segments[i].timing.end_time,
                confidence=None,
                method="agglomerative",
            )
        )

    structure = StructureDNA(
        segments=tuple(segments),
        total_duration=duration,
        boundary_confidences=tuple(boundaries),
    )

    metadata = SongMetadata(
        song_id=song_id,
        filename=filename,
        duration=duration,
        sample_rate=sr,
        format=Path(filename).suffix.lstrip(".") or "mp3",
    )

    return SongDNA(
        metadata=metadata,
        summary=summary,
        structure=structure,
        timeline=timeline,
        frames=frames,
        manifest=manifest,
    )


def _assemble_v1_adapter(
    file_path: str,
    duration: float,
    sr: int,
    y: np.ndarray,
    features: Dict[str, np.ndarray],
    tempo: float,
    tempo_is_fallback: bool,
    beat_times: Optional[Tuple[float, ...]] = None,
    frames: Optional[FrameReference] = None,
    manifest: Optional[AnalysisManifest] = None,
) -> SongDNA:
    """Assemble a v2 ``SongDNA`` from legacy v1 global features.

    This is the v1 branch of the migration.  It produces a v2 ``SongDNA``
    whose ``StructureDNA`` contains a **single global segment** covering
    the whole track.  This keeps every consumer on the v2 type while the
    rest of the system migrates (see ``docs/MIGRATION.md``).
    """
    song_id = generate_song_id()
    timbre, tonal = _compute_global_statistics(features, y, sr)
    onset_mean = _onset_strength_mean(y, sr)

    segment = _build_single_segment(
        song_id=song_id,
        duration=duration,
        tempo=tempo,
        tempo_is_fallback=tempo_is_fallback,
        timbre=timbre,
        tonal=tonal,
        onset_mean=onset_mean,
    )

    summary = _build_song_summary(tempo, tempo_is_fallback, timbre, tonal)

    timeline = None
    if beat_times:
        timeline = Timeline.build(beats=beat_times, duration=duration)

    return _assemble_song_dna_v2(
        song_id=song_id,
        filename=file_path,
        duration=duration,
        sr=sr,
        segments=[segment],
        summary=summary,
        timeline=timeline,
        frames=frames,
        manifest=manifest,
    )


def _assemble_v2_full(
    file_path: str,
    duration: float,
    sr: int,
    features: Dict[str, np.ndarray],
    y: np.ndarray,
    tempo: float,
    tempo_is_fallback: bool,
    beat_times: Optional[Tuple[float, ...]] = None,
    frames: Optional[FrameReference] = None,
    manifest: Optional[AnalysisManifest] = None,
) -> SongDNA:
    """Assemble a full v2 ``SongDNA`` with structural segmentation."""
    song_id = generate_song_id()

    # Structural segmentation + per-segment DSP
    seg_ranges = detect_segments(y, sr)
    dsp_list = compute_segment_dsp(y, sr, seg_ranges)

    # Fall back to a single global segment if nothing was detected
    if not seg_ranges:
        seg_ranges = [(0.0, duration)]

    segments: List[SegmentDNA] = []
    for i, ((start_s, end_s), dsp) in enumerate(zip(seg_ranges, dsp_list)):
        # start_beat / end_beat are left as None: they represent indices
        # into a per-segment beat grid, which is a future refinement.
        # Frame indices are NOT beat indices; setting them here would
        # violate the semantic contract of SegmentTiming.
        timing = SegmentTiming(
            start_time=start_s,
            end_time=end_s,
        )
        context = SegmentContext(
            index=SegmentIndex(i),
            label=LabelType.UNKNOWN,
            label_confidence=0.0,
        )
        segments.append(
            SegmentDNA(
                song_id=song_id,
                timing=timing,
                dsp=dsp,
                context=context,
            )
        )

    # SongSummary from global statistics
    timbre, tonal = _compute_global_statistics(features, y, sr)
    summary = _build_song_summary(tempo, tempo_is_fallback, timbre, tonal)

    timeline = None
    if beat_times:
        timeline = Timeline.build(beats=beat_times, duration=duration)

    return _assemble_song_dna_v2(
        song_id=song_id,
        filename=file_path,
        duration=duration,
        sr=sr,
        segments=segments,
        summary=summary,
        timeline=timeline,
        frames=frames,
        manifest=manifest,
    )


# ======================================================================
# Public API — extract_song_dna (backward compatible)
# ======================================================================


def extract_song_dna(file_path: str) -> SongDNA:
    """Load an audio file and return its v2 SongDNA fingerprint.

    This is the **in-memory only** extraction path.  It computes
    summary statistics and beat information but does **not** persist
    frame-level data to NPZ.  The returned ``SongDNA`` will have
    ``frames=None`` and ``manifest=None``.

    The returned type is always the **v2** ``SongDNA`` (see
    ``docs/MIGRATION.md``).  When ``_BUILD_V2`` is False the v1
    branch constructs a v2 ``SongDNA`` with a single global segment.

    For full extraction with NPZ persistence, use ``extract_and_save()``.

    Parameters
    ----------
    file_path : str
        Path to a supported audio file (MP3, WAV, FLAC, OGG, M4A).

    Returns
    -------
    SongDNA
        Immutable v2 dataclass holding the extracted features.
    """
    # Stage 1 — Load (locked to 22050 Hz)
    y, sr = _load_audio(file_path)

    # Stage 2 — Frame features
    features = _extract_frame_features(y, sr)

    # Stage 3 — Beats
    tempo, is_fallback, beat_times = _detect_beats(y, sr)

    # Stage 4 — Global statistics
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Stage 5 — Manifest (no manifest for in-memory path)
    # Stage 6 — Assemble
    beat_times_tuple: Optional[Tuple[float, ...]] = beat_times

    if _BUILD_V2:
        return _assemble_v2_full(
            file_path=file_path,
            duration=duration,
            sr=sr,
            features=features,
            y=y,
            tempo=tempo,
            tempo_is_fallback=is_fallback,
            beat_times=beat_times_tuple,
            frames=None,
            manifest=None,
        )

    return _assemble_v1_adapter(
        file_path=file_path,
        duration=duration,
        sr=sr,
        y=y,
        features=features,
        tempo=tempo,
        tempo_is_fallback=is_fallback,
        beat_times=beat_times_tuple,
        frames=None,
        manifest=None,
    )


def extract_and_save(file_path: str, data_root: Path) -> SongDNA:
    """Extract SongDNA and persist frame-level data to NPZ storage.

    This is the **full extraction path**.  It:
    1. Loads the audio file
    2. Extracts all frame-level features
    3. Saves them as a compressed NPZ file under ``data_root / frames/``
    4. Constructs a ``FrameReference`` pointing to the NPZ file
    5. Builds an ``AnalysisManifest`` recording extraction provenance
    6. Returns a v2 ``SongDNA`` with ``frames`` and ``manifest`` populated

    Parameters
    ----------
    file_path : str
        Path to a supported audio file.
    data_root : Path
        Absolute path to the EchoInsight data root directory.
        NPZ files are saved under ``data_root / frames / {stem}.npz``.

    Returns
    -------
    SongDNA
        v2 ``SongDNA`` with ``frames`` and ``manifest`` populated.
    """
    # Stage 1 — Load
    y, sr = _load_audio(file_path)

    # Stage 2 — Frame features
    features = _extract_frame_features(y, sr)

    # Stage 3 — Beats
    tempo, is_fallback, beat_times = _detect_beats(y, sr)

    # Stage 4 — Global statistics
    duration = float(librosa.get_duration(y=y, sr=sr))

    # ---- NPZ persistence ----
    stem = Path(file_path).stem
    npz_uri = f"frames/{stem}.npz"
    npz_path = (data_root / npz_uri).resolve()
    FrameStore.save(features, npz_path)

    frame_ref = FrameStore.build_frame_reference(
        uri=npz_uri,
        features=features,
        hop_length=HOP_LENGTH,
        sample_rate=sr,
        duration=duration,
    )
    logger.info("Frame data saved to %s", npz_path)

    # Stage 5 — Manifest
    manifest = _build_manifest(sample_rate=sr, hop_length=HOP_LENGTH)
    logger.debug("Manifest: extractor=%s, hash=%s",
                 manifest.extractor_version, manifest.config_hash)

    # Stage 6 — Assemble
    beat_times_tuple: Optional[Tuple[float, ...]] = beat_times

    if _BUILD_V2:
        return _assemble_v2_full(
            file_path=file_path,
            duration=duration,
            sr=sr,
            features=features,
            y=y,
            tempo=tempo,
            tempo_is_fallback=is_fallback,
            beat_times=beat_times_tuple,
            frames=frame_ref,
            manifest=manifest,
        )

    return _assemble_v1_adapter(
        file_path=file_path,
        duration=duration,
        sr=sr,
        y=y,
        features=features,
        tempo=tempo,
        tempo_is_fallback=is_fallback,
        beat_times=beat_times_tuple,
        frames=frame_ref,
        manifest=manifest,
    )


# ======================================================================
# Unified Package Pipeline — extract_full_pipeline
# ======================================================================


def extract_full_pipeline(
    file_path: str | Path,
    data_root: Optional[Path] = None,
    separate_stems: bool = False,
    custom_model_path: Optional[str | Path] = None,
) -> SongPackage:
    """Unified entry point executing Pillar 1 and optionally Pillar 2.

    Parameters
    ----------
    file_path : str | Path
        Path to the target audio file.
    data_root : Optional[Path]
        If provided, persists frame NPZ data under ``data_root / frames/``
        and stem WAVs under ``data_root / stems/``.
    separate_stems : bool
        If True, executes HTDemucs separation and stem acoustic analysis.
    custom_model_path : Optional[str | Path]
        Path to custom fine-tuned weights (e.g. ``best_htdemucs_finetuned.th``).

    Returns
    -------
    SongPackage
        The canonical package aggregate containing the SongDNA (with
        Timeline when beats were detected) plus optional ``SeparatedTrack``
        and ``StemAnalysisResult``.

    Notes
    -----
    Pillar 1 always delegates to the canonical extraction engine
    (``extract_song_dna`` / ``extract_and_save``) — this wrapper never
    constructs ``SongDNA`` on its own.
    """
    path_obj = Path(file_path)

    # 1. Pillar 1: canonical SongDNA extraction (Timeline included).
    if data_root is not None:
        song_dna = extract_and_save(str(path_obj), data_root)
    else:
        song_dna = extract_song_dna(str(path_obj))

    separated_track: Optional[SeparatedTrack] = None
    stem_analysis: Optional[StemAnalysisResult] = None

    # 2. Pillar 2 (optional): 4-stem separation + stem acoustic analysis.
    if separate_stems:
        cache_dir = (data_root / "stems") if data_root else None
        separator = EchoSeparator(
            model_path=custom_model_path,
            cache_dir=cache_dir,
        )
        separated_track = separator.separate(path_obj)

        analyzer = StemAnalyzer()
        stem_analysis = analyzer.analyze(separated_track)

    return SongPackage(
        song_dna=song_dna,
        separated_track=separated_track,
        stem_analysis=stem_analysis,
    )