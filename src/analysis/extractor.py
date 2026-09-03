"""
extractor.py — Feature extraction pipeline that converts a raw
audio file into a v2 SongDNA aggregate (SongDNA + StructureDNA +
SegmentDNA[]) and optionally separates 4-stem tracks.

The pipeline is divided into deterministic stages:

1.  Load Audio (locked to 22050 Hz)
2.  Extract Frame Features
3.  Detect Beats
4.  Segment Structure (v2 branch) — or single-global-segment (v1 branch)
5.  Compute Per-Segment DSP (v2 branch)
6.  Build Manifest
7.  Assemble SongDNA v2
8.  (Optional) 4-Stem Separation + Stem Acoustic Analysis

Public API
----------
- ``extract_song_dna(file_path)``   — in-memory extraction (backward compatible)
- ``extract_and_save(file_path, data_root)`` — extraction + NPZ persistence
- ``extract_full_pipeline(file_path, data_root, separate_stems)`` — unified Pillar 1 + Pillar 2
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import logging
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from src.analysis.frame_store import FrameStore
from src.analysis.segment_dsp import compute_segment_dsp
from src.analysis.segmentation import detect_segments
from src.analysis.separator import EchoSeparator
from src.analysis.stem_analyzer import StemAnalysisResult, StemAnalyzer
from src.core.analysis_manifest import AnalysisManifest, compute_config_hash
from src.core.enums import LabelType
from src.core.frame_reference import FrameReference
from src.core.identifiers import SegmentIndex, SongID, generate_song_id
from src.core.song_dna import SongDNA, SongMetadata, SongSummary
from src.core.stems import SeparatedTrack
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Migration flag
# ---------------------------------------------------------------------------

_BUILD_V2 = False  # Set to True when activating full structural segmentation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 22050          # Locked per Extraction Contract
N_MFCC: int = 13
HOP_LENGTH: int = 512
N_FFT: int = 2048
FEATURE_SET_VERSION: str = "v2"
EXTRACTOR_VERSION: str = "2.0.0"
SCHEMA_VERSION: str = "1.0.0"

_MAJOR_PROFILE: np.ndarray = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE: np.ndarray = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_KEY_NAMES: List[str] = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]


@dataclass(frozen=True)
class ExtractedSongPackage:
    """Unified container for all Pillar 1 and Pillar 2 extraction outputs."""
    song_dna: SongDNA
    separated_track: Optional[SeparatedTrack] = None
    stem_analysis: Optional[StemAnalysisResult] = None


# ======================================================================
# Stage 1 — Load Audio
# ======================================================================


def _load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return the time series and sample rate."""
    logger.info("Loading audio: %s", file_path)
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    logger.debug(
        "Loaded: sr=%d, samples=%d, duration=%.2fs",
        sr, len(y), len(y) / sr
    )
    return y, sr


# ======================================================================
# Stage 2 — Frame Feature Extraction
# ======================================================================


def _extract_frame_features(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Extract all frame-level DSP features for a track."""
    logger.debug("Extracting frame features (hop_length=%d)", HOP_LENGTH)
    features: Dict[str, np.ndarray] = {}

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

    features["mfcc"] = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["chroma"] = chroma

    features["contrast"] = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    features["tonnetz"] = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    return features


# ======================================================================
# Stage 3 — Beat Detection
# ======================================================================


def _detect_beats(y: np.ndarray, sr: int) -> Tuple[float, float, Tuple[int, ...]]:
    """Detect beat positions and estimate tempo with fallback for ambient/pure tones."""
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=HOP_LENGTH
        )
        raw_tempo = float(np.round(np.asarray(tempo).flatten()[0], 2))
    except Exception:
        raw_tempo = 0.0
        beat_frames = np.array([])

    if raw_tempo <= 0.0:
        # Fallback for static tones or zero-energy audio
        final_tempo = 120.0
        confidence = 0.0
    else:
        final_tempo = raw_tempo
        confidence = 1.0

    beat_frames_tuple: Tuple[int, ...] = tuple(
        int(b) for b in np.asarray(beat_frames).flatten()
    )
    return final_tempo, confidence, beat_frames_tuple


# ======================================================================
# Stage 4 — Global Statistics (shared helpers)
# ======================================================================


def _estimate_key(chroma_mean: np.ndarray) -> Tuple[Optional[str], float]:
    """Krumhansl-Schmuckler key-finding algorithm."""
    total = float(np.sum(chroma_mean))
    if total < 1e-10:
        return None, 0.0

    chroma_norm = chroma_mean / total
    best_corr = -1.0
    best_key: Optional[str] = None

    for shift in range(12):
        rolled = np.roll(chroma_norm, shift)

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
    """Reduce frame-level features to plain summary dicts."""
    timbre: Dict[str, float] = {
        "rms_energy_mean": float(np.mean(features["rms"])),
        "spectral_centroid_mean": float(np.mean(features["centroid"])),
        "spectral_bandwidth_mean": float(np.mean(features["bandwidth"])),
        "mfcc_mean": [float(v) for v in np.mean(features["mfcc"], axis=1)],
    }

    chroma_mean_arr = np.mean(features["chroma"], axis=1)
    key, key_confidence = _estimate_key(chroma_mean_arr)
    tuning_offset = float(librosa.estimate_tuning(y=y, sr=sr))

    tonal: Dict[str, object] = {
        "key": key,
        "key_confidence": key_confidence,
        "chroma_mean": [float(v) for v in chroma_mean_arr],
        "tuning_offset": tuning_offset,
    }
    return timbre, tonal


def _onset_strength_mean(
    y: np.ndarray,
    sr: int,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
) -> float:
    """Mean onset strength over a time range."""
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
    """Construct an AnalysisManifest for the current extraction."""
    config_hash = compute_config_hash(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_fft=N_FFT,
        feature_set_version=FEATURE_SET_VERSION,
    )

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
        pass

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
# Stage 6 — Assembly
# ======================================================================


def _build_song_summary(
    tempo: float,
    tempo_confidence: float,
    timbre: Dict[str, float],
    tonal: Dict[str, object],
) -> SongSummary:
    """Build a v2 SongSummary from global statistics."""
    return SongSummary(
        tempo_estimate=tempo,
        tempo_confidence=tempo_confidence,
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
    tempo_confidence: float,
    timbre: Dict[str, float],
    tonal: Dict[str, object],
    onset_mean: float,
) -> SegmentDNA:
    """Build a single-global-segment SegmentDNA for the v1 adapter."""
    timing = SegmentTiming(start_time=0.0, end_time=duration)
    rhythm = SegmentRhythm(
        tempo=tempo,
        tempo_confidence=tempo_confidence,
        onset_strength_mean=onset_mean,
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
    frames: Optional[FrameReference] = None,
    manifest: Optional[AnalysisManifest] = None,
) -> SongDNA:
    """Assemble a complete v2 SongDNA aggregate."""
    boundaries: List[BoundaryConfidence] = []
    for i in range(len(segments) - 1):
        boundaries.append(
            BoundaryConfidence(
                time=segments[i].timing.end_time,
                confidence=0.7,
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
        frames=frames,
        manifest=manifest,
        schema_version="2.0.0",
    )


def _assemble_v1_adapter(
    file_path: str,
    duration: float,
    sr: int,
    y: np.ndarray,
    features: Dict[str, np.ndarray],
    tempo: float,
    tempo_confidence: float,
    frames: Optional[FrameReference] = None,
    manifest: Optional[AnalysisManifest] = None,
) -> SongDNA:
    """Assemble a v2 SongDNA from legacy v1 global features."""
    song_id = generate_song_id()
    timbre, tonal = _compute_global_statistics(features, y, sr)
    onset_mean = _onset_strength_mean(y, sr)

    segment = _build_single_segment(
        song_id=song_id,
        duration=duration,
        tempo=tempo,
        tempo_confidence=tempo_confidence,
        timbre=timbre,
        tonal=tonal,
        onset_mean=onset_mean,
    )

    summary = _build_song_summary(tempo, tempo_confidence, timbre, tonal)
    return _assemble_song_dna_v2(
        song_id=song_id,
        filename=file_path,
        duration=duration,
        sr=sr,
        segments=[segment],
        summary=summary,
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
    tempo_confidence: float,
    frames: Optional[FrameReference] = None,
    manifest: Optional[AnalysisManifest] = None,
) -> SongDNA:
    """Assemble a full v2 SongDNA with structural segmentation."""
    song_id = generate_song_id()

    seg_ranges = detect_segments(y, sr)
    dsp_list = compute_segment_dsp(y, sr, seg_ranges)

    if not seg_ranges:
        seg_ranges = [(0.0, duration)]

    segments: List[SegmentDNA] = []
    for i, ((start_s, end_s), dsp) in enumerate(zip(seg_ranges, dsp_list)):
        timing = SegmentTiming(start_time=start_s, end_time=end_s)
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

    timbre, tonal = _compute_global_statistics(features, y, sr)
    summary = _build_song_summary(tempo, tempo_confidence, timbre, tonal)

    return _assemble_song_dna_v2(
        song_id=song_id,
        filename=file_path,
        duration=duration,
        sr=sr,
        segments=segments,
        summary=summary,
        frames=frames,
        manifest=manifest,
    )


# ======================================================================
# Public API — extract_song_dna & extract_and_save
# ======================================================================


def extract_song_dna(file_path: str) -> SongDNA:
    """Load an audio file and return its v2 SongDNA fingerprint in-memory."""
    y, sr = _load_audio(file_path)
    features = _extract_frame_features(y, sr)
    tempo, confidence, beat_frames = _detect_beats(y, sr)
    duration = float(librosa.get_duration(y=y, sr=sr))

    if _BUILD_V2:
        return _assemble_v2_full(
            file_path=file_path,
            duration=duration,
            sr=sr,
            features=features,
            y=y,
            tempo=tempo,
            tempo_confidence=confidence,
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
        tempo_confidence=confidence,
        frames=None,
        manifest=None,
    )


def extract_and_save(file_path: str, data_root: Path) -> SongDNA:
    """Extract SongDNA and persist frame-level data to NPZ storage."""
    y, sr = _load_audio(file_path)
    features = _extract_frame_features(y, sr)
    tempo, confidence, beat_frames = _detect_beats(y, sr)
    duration = float(librosa.get_duration(y=y, sr=sr))

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
    manifest = _build_manifest(sample_rate=sr, hop_length=HOP_LENGTH)

    if _BUILD_V2:
        return _assemble_v2_full(
            file_path=file_path,
            duration=duration,
            sr=sr,
            features=features,
            y=y,
            tempo=tempo,
            tempo_confidence=confidence,
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
        tempo_confidence=confidence,
        frames=frame_ref,
        manifest=manifest,
    )


def extract_full_pipeline(
    file_path: str | Path,
    data_root: Optional[Path] = None,
    separate_stems: bool = False,
    custom_model_path: Optional[str | Path] = None,
) -> ExtractedSongPackage:
    """Unified entry point executing both Pillar 1 and Pillar 2.

    Parameters
    ----------
    file_path : str | Path
        Path to the target audio file.
    data_root : Optional[Path]
        If provided, persists frame NPZ data under data_root / frames/.
    separate_stems : bool
        If True, executes HTDemucs separation and stem acoustic analysis.
    custom_model_path : Optional[str | Path]
        Path to custom fine-tuned weights (e.g. best_htdemucs_finetuned.th).

    Returns
    -------
    ExtractedSongPackage
        Contains SongDNA, optional SeparatedTrack, and optional StemAnalysisResult.
    """
    path_obj = Path(file_path)

    # 1. Pillar 1: Extract SongDNA
    if data_root is not None:
        song_dna = extract_and_save(str(path_obj), data_root)
    else:
        song_dna = extract_song_dna(str(path_obj))

    separated_track: Optional[SeparatedTrack] = None
    stem_analysis: Optional[StemAnalysisResult] = None

    # 2. Pillar 2: 4-Stem Separation & Analysis
    if separate_stems:
        cache_dir = (data_root / "stems") if data_root else None
        separator = EchoSeparator(
            model_path=custom_model_path,
            cache_dir=cache_dir,
        )
        separated_track = separator.separate(path_obj)

        analyzer = StemAnalyzer()
        stem_analysis = analyzer.analyze(separated_track)

    return ExtractedSongPackage(
        song_dna=song_dna,
        separated_track=separated_track,
        stem_analysis=stem_analysis,
    )