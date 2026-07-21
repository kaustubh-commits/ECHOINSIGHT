"""
extractor.py — Feature extraction pipeline that converts a raw
audio file into a nested SongDNA dataclass.

The pipeline is divided into deterministic stages:

1.  Load Audio
2.  Extract Frame Features
3.  Detect Beats
4.  Compute Global Statistics
5.  Build Manifest
6.  Assemble SongDNA

Public API
----------
- ``extract_song_dna(file_path)``   — in-memory extraction (backward compatible)
- ``extract_and_save(file_path, data_root)`` — extraction + NPZ persistence
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
from src.core.dna_schema import (
    IntelligenceDNA,
    RhythmDNA,
    SongDNA,
    SongMetadata,
    StemDNA,
    StructureDNA,
    TimbreDNA,
    TonalDNA,
)
from src.core.frame_reference import FrameReference
from src.analysis.frame_store import FrameStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
FEATURE_SET_VERSION: str = "v1"
EXTRACTOR_VERSION: str = "2.0.0"
SCHEMA_VERSION: str = "1.0.0"


# ======================================================================
# Stage 1 — Load Audio
# ======================================================================


def _load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return the time series and sample rate.

    Parameters
    ----------
    file_path : str
        Path to a supported audio file (MP3, WAV, FLAC, OGG, etc.).

    Returns
    -------
    tuple[np.ndarray, int]
        ``(y, sr)`` where ``y`` is the mono audio time series and
        ``sr`` is the native sample rate.
    """
    logger.info("Loading audio: %s", file_path)
    y, sr = librosa.load(file_path, sr=None, mono=True)
    logger.debug("Loaded: sr=%d, samples=%d, duration=%.2fs", sr, len(y), len(y) / sr)
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
# Stage 3 — Beat Detection
# ======================================================================


def _detect_beats(y: np.ndarray, sr: int) -> Tuple[float, float, Tuple[int, ...]]:
    """Detect beat positions and estimate tempo.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.

    Returns
    -------
    tuple[float, float, tuple[int, ...]]
        ``(tempo, confidence, beat_frames)``.
        ``tempo`` is in BPM, rounded to 2 decimal places.
        ``confidence`` is the beat tracker's confidence (0.0–1.0).
        ``beat_frames`` are frame indices into the feature arrays.
    """
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )
    tempo = float(np.round(np.asarray(tempo).flatten()[0], 2))

    # librosa does not expose a native confidence metric for beat_track.
    # We use 1.0 as a default — future versions may derive confidence
    # from onset strength autocorrelation.
    confidence: float = 1.0

    beat_frames_tuple: Tuple[int, ...] = tuple(
        int(b) for b in np.asarray(beat_frames).flatten()
    )

    logger.debug("Tempo: %.2f BPM, %d beats detected", tempo, len(beat_frames_tuple))
    return tempo, confidence, beat_frames_tuple


# ======================================================================
# Stage 4 — Global Statistics
# ======================================================================


def _estimate_key(chroma_mean: np.ndarray) -> Tuple[Optional[str], float]:
    """Krumhansl-Schmuckler key-finding algorithm.

    Parameters
    ----------
    chroma_mean : np.ndarray
        12-bin chroma vector averaged over the track.

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
) -> Tuple[TimbreDNA, TonalDNA]:
    """Reduce frame-level features to summary statistics for SongDNA.

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
    tuple[TimbreDNA, TonalDNA]
    """
    # --- Timbre ---
    rms_mean = float(np.mean(features["rms"]))
    centroid_mean = float(np.mean(features["centroid"]))
    bandwidth_mean = float(np.mean(features["bandwidth"]))
    mfcc_mean: List[float] = np.mean(features["mfcc"], axis=1).tolist()

    timbre = TimbreDNA(
        rms_energy_mean=rms_mean,
        spectral_centroid_mean=centroid_mean,
        spectral_bandwidth_mean=bandwidth_mean,
        mfcc_mean=mfcc_mean,
    )

    # --- Tonal ---
    chroma_mean_arr = np.mean(features["chroma"], axis=1)  # (12,)
    key, key_confidence = _estimate_key(chroma_mean_arr)
    tuning_offset = float(librosa.estimate_tuning(y=y, sr=sr))

    tonal = TonalDNA(
        key=key,
        key_confidence=key_confidence,
        chroma_mean=chroma_mean_arr.tolist(),
        tuning_offset=tuning_offset,
    )

    logger.debug("Key: %s (confidence: %.2f), tuning: %.1f cents",
                 key, key_confidence, tuning_offset)
    return timbre, tonal


# ======================================================================
# Stage 5 — Build Manifest
# ======================================================================


def _build_manifest(sample_rate: int, hop_length: int) -> AnalysisManifest:
    """Construct an ``AnalysisManifest`` for the current extraction.

    Parameters
    ----------
    sample_rate : int
        Sample rate used for extraction.
    hop_length : int
        Hop length used for frame extraction.

    Returns
    -------
    AnalysisManifest
    """
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
        created_at=datetime.datetime.utcnow().isoformat() + "Z",
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


def _assemble_song_dna(
    metadata: SongMetadata,
    rhythm: RhythmDNA,
    timbre: TimbreDNA,
    tonal: TonalDNA,
    structure: StructureDNA,
    intelligence: IntelligenceDNA,
    stem: StemDNA,
    frames: Optional["FrameReference"] = None,
    manifest: Optional[AnalysisManifest] = None,
) -> SongDNA:
    """Assemble a complete ``SongDNA`` from its sub-components.

    All parameters have the same meaning as ``SongDNA`` fields.
    """
    return SongDNA(
        metadata=metadata,
        rhythm=rhythm,
        timbre=timbre,
        tonal=tonal,
        structure=structure,
        intelligence=intelligence,
        stem=stem,
        frames=frames,
        manifest=manifest,
    )


# ======================================================================
# Public API — extract_song_dna (backward compatible)
# ======================================================================


def extract_song_dna(file_path: str) -> SongDNA:
    """Load an audio file and return its nested SongDNA fingerprint.

    This is the **in-memory only** extraction path.  It computes
    summary statistics and beat information but does **not** persist
    frame-level data to NPZ.  The returned ``SongDNA`` will have
    ``frames=None`` and ``manifest=None``.

    For full extraction with NPZ persistence, use ``extract_and_save()``.

    Parameters
    ----------
    file_path : str
        Path to a supported audio file (MP3, WAV, FLAC, OGG, etc.).

    Returns
    -------
    SongDNA
        Immutable dataclass holding the extracted features.
    """
    # Stage 1 — Load
    y, sr = _load_audio(file_path)

    # Stage 2 — Frame features
    features = _extract_frame_features(y, sr)

    # Stage 3 — Beats
    tempo, confidence, beat_frames = _detect_beats(y, sr)

    # Stage 4 — Global statistics
    duration = float(librosa.get_duration(y=y, sr=sr))
    timbre, tonal = _compute_global_statistics(features, y, sr)

    # Stage 5 — Manifest (no manifest for in-memory path)
    # Stage 6 — Assemble
    return _assemble_song_dna(
        metadata=SongMetadata(duration=duration, filename=file_path),
        rhythm=RhythmDNA(
            tempo=tempo,
            confidence=confidence,
            beat_frames=beat_frames,
            beat_count=len(beat_frames),
        ),
        timbre=timbre,
        tonal=tonal,
        structure=StructureDNA(),
        intelligence=IntelligenceDNA(),
        stem=StemDNA(),
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
    6. Returns a ``SongDNA`` with ``frames`` and ``manifest`` populated

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
        ``SongDNA`` with ``frames`` and ``manifest`` populated.
    """
    # Stage 1 — Load
    y, sr = _load_audio(file_path)

    # Stage 2 — Frame features
    features = _extract_frame_features(y, sr)

    # Stage 3 — Beats
    tempo, confidence, beat_frames = _detect_beats(y, sr)

    # Stage 4 — Global statistics
    duration = float(librosa.get_duration(y=y, sr=sr))
    timbre, tonal = _compute_global_statistics(features, y, sr)

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
    return _assemble_song_dna(
        metadata=SongMetadata(duration=duration, filename=file_path),
        rhythm=RhythmDNA(
            tempo=tempo,
            confidence=confidence,
            beat_frames=beat_frames,
            beat_count=len(beat_frames),
        ),
        timbre=timbre,
        tonal=tonal,
        structure=StructureDNA(),
        intelligence=IntelligenceDNA(),
        stem=StemDNA(),
        frames=frame_ref,
        manifest=manifest,
    )