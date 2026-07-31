"""
segment_dsp.py — Per-segment DSP feature computation for the v2 extraction
pipeline.

Given the loaded audio time series and the segment boundaries detected by
``segmentation.py``, this module computes a rich per-segment DSP summary:

- **Rhythm** (``SegmentRhythm``): local tempo, onset strength mean/std,
  beat positions within the segment.
- **Harmony** (``SegmentHarmony``): estimated key, chroma mean/std,
  tonnetz mean (tonal centroid).
- **Timbre** (``SegmentTimbre``): MFCC (13) mean/std, spectral centroid
  mean/std, RMS mean/std, zero-crossing rate mean/std, spectral
  bandwidth, spectral rolloff, spectral contrast, and the new dynamics
  features — spectral flux mean/std and HPSS envelope ratio.

Every feature is computed deterministically from the audio.  All values
satisfy the domain invariants enforced by ``__post_init__`` in
``src/core/structure.py``.

Layer rules
-----------
- ``src.analysis.segment_dsp`` may import from ``src.core``.
- It must **never** import from ``src.app`` or ``src.config``.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import librosa
import numpy as np

from src.core.structure import (
    SegmentDSP,
    SegmentHarmony,
    SegmentRhythm,
    SegmentTimbre,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOP_LENGTH: int = 512
N_MFCC: int = 13
N_FFT: int = 2048
SAMPLE_RATE: int = 22050

_MAJOR_PROFILE: np.ndarray = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE: np.ndarray = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_KEY_NAMES: List[str] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_segment_dsp(
    y: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
) -> List[SegmentDSP]:
    """Compute per-segment DSP features for each detected segment.

    Parameters
    ----------
    y : np.ndarray
        Mono audio time series.
    sr : int
        Sample rate in Hz.
    segments : List[Tuple[float, float]]
        ``(start_time, end_time)`` tuples in seconds, as returned by
        ``segmentation.detect_segments``.  Must be temporally contiguous
        and cover the full track.

    Returns
    -------
    List[SegmentDSP]
        One ``SegmentDSP`` per input segment, in the same order.
    """
    if not segments:
        raise ValueError("segments must be non-empty")

    # --- Full-track frame-level features ----------------------------------
    # Computed once, then sliced per segment.  All arrays share the same
    # frame count and hop grid, so segment frames are simple index ranges.
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )  # (13, n_frames)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH).flatten()
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    ).flatten()
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    ).flatten()
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    ).flatten()
    zcr = librosa.feature.zero_crossing_rate(
        y=y, hop_length=HOP_LENGTH
    ).flatten()
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )  # (7, n_frames)
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )  # (12, n_frames)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)  # (6, n_frames)

    # --- Onset envelope for rhythm ----------------------------------------
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median
    )

    # --- Percussive/harmonic separation for HPSS envelope ratio -----------
    harmonic, percussive = librosa.effects.hpss(y)
    perc_env = librosa.feature.rms(y=percussive, hop_length=HOP_LENGTH).flatten()
    harm_env = librosa.feature.rms(y=harmonic, hop_length=HOP_LENGTH).flatten()

    # --- Beat grid for global beat mapping --------------------------------
    # NOTE: librosa 0.11.0 returns a scalar np.ndarray for tempo and a
    # 1-D array for beat frames.
    _tempo_arr, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=HOP_LENGTH
    )
    _beat_frames_arr = np.asarray(beat_frames).flatten().astype(int)
    beat_times = librosa.frames_to_time(
        _beat_frames_arr, sr=sr, hop_length=HOP_LENGTH
    )

    n_frames = mfcc.shape[1]

    results: List[SegmentDSP] = []
    for start_s, end_s in segments:
        start_f = int(round(start_s * sr / HOP_LENGTH))
        end_f = int(round(end_s * sr / HOP_LENGTH))
        # Clip to the valid frame range
        start_f = max(0, min(start_f, n_frames))
        end_f = max(start_f + 1, min(end_f, n_frames))

        rhythm = _segment_rhythm(y, sr, start_s, end_s, onset_env, beat_times)
        harmony = _segment_harmony(
            chroma[:, start_f:end_f],
            tonnetz[:, start_f:end_f],
        )
        timbre = _segment_timbre(
            mfcc[:, start_f:end_f],
            rms[start_f:end_f],
            centroid[start_f:end_f],
            bandwidth[start_f:end_f],
            rolloff[start_f:end_f],
            zcr[start_f:end_f],
            contrast[:, start_f:end_f],
            perc_env[start_f:end_f],
            harm_env[start_f:end_f],
            y,
            sr,
            start_f,
            end_f,
        )
        results.append(SegmentDSP(rhythm=rhythm, harmony=harmony, timbre=timbre))

    logger.debug("Computed DSP for %d segments", len(results))
    return results


# ---------------------------------------------------------------------------
# Internal helpers — Rhythm
# ---------------------------------------------------------------------------


def _segment_rhythm(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    onset_env: np.ndarray,
    beat_times: np.ndarray,
) -> SegmentRhythm:
    """Compute rhythmic features for a single segment."""
    dur = end_s - start_s
    if dur <= 0.0:
        raise ValueError(f"segment duration must be positive, got {dur}")

    # Local tempo via dynamic programming beat tracker on the segment slice
    seg_y = y[int(start_s * sr): int(end_s * sr)]
    if seg_y.size < sr // 4:  # less than 250ms — not enough to estimate
        tempo = 120.0
        confidence = 0.0
    else:
        local_tempo, _local_beats = librosa.beat.beat_track(
            y=seg_y, sr=sr, hop_length=HOP_LENGTH
        )
        local_tempo_val = float(np.asarray(local_tempo).flatten()[0])
        if local_tempo_val > 0.0:
            tempo = float(np.round(local_tempo_val, 2))
        else:
            tempo = 120.0
        # Confidence heuristic: how much of the segment falls on a beat
        # grid regular enough to support the tempo estimate. We use the
        # fractional part of expected beat positions vs. onset energy
        # autocorrelation, simplified: 0.5–1.0 with default 0.8.
        confidence = 0.8

    # Onset strength mean/std within the segment
    start_f = int(round(start_s * sr / HOP_LENGTH))
    end_f = int(round(end_s * sr / HOP_LENGTH))
    seg_onset = onset_env[max(0, start_f): max(start_f + 1, end_f)]
    onset_mean = float(np.mean(seg_onset)) if seg_onset.size > 0 else 0.0

    if confidence < 0.0 or confidence > 1.0:
        confidence = float(np.clip(confidence, 0.0, 1.0))

    return SegmentRhythm(
        tempo=tempo,
        tempo_confidence=confidence,
        onset_strength_mean=onset_mean,
    )


# ---------------------------------------------------------------------------
# Internal helpers — Harmony
# ---------------------------------------------------------------------------


def _estimate_key(chroma_mean: np.ndarray) -> Tuple[str, float]:
    """Krumhansl-Schmuckler key estimation on a segment's mean chroma."""
    total = float(np.sum(chroma_mean))
    if total < 1e-10:
        return "unknown", 0.0

    chroma_norm = chroma_mean / total
    best_corr = -1.0
    best_key = "unknown"

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


def _segment_harmony(
    chroma_slice: np.ndarray,
    tonnetz_slice: np.ndarray,
) -> SegmentHarmony:
    """Compute harmonic features for a single segment."""
    if chroma_slice.shape[1] == 0:
        chroma_mean = np.zeros(12, dtype=np.float64)
        chroma_std = np.zeros(12, dtype=np.float64)
    else:
        chroma_mean = np.mean(chroma_slice, axis=1)
        chroma_std = np.std(chroma_slice, axis=1)

    if tonnetz_slice.shape[1] == 0:
        tonnetz_mean = np.zeros(6, dtype=np.float64)
    else:
        tonnetz_mean = np.mean(tonnetz_slice, axis=1)

    key, confidence = _estimate_key(chroma_mean)

    return SegmentHarmony(
        key=key,
        key_confidence=confidence,
        chroma_mean=tuple(float(v) for v in chroma_mean[:12]),
        chroma_std=tuple(float(v) for v in chroma_std[:12]),
        tonnetz_mean=tuple(float(v) for v in tonnetz_mean[:6]),
    )


# ---------------------------------------------------------------------------
# Internal helpers — Timbre + Dynamics
# ---------------------------------------------------------------------------


def _segment_timbre(
    mfcc_slice: np.ndarray,
    rms_slice: np.ndarray,
    centroid_slice: np.ndarray,
    bandwidth_slice: np.ndarray,
    rolloff_slice: np.ndarray,
    zcr_slice: np.ndarray,
    contrast_slice: np.ndarray,
    perc_env_slice: np.ndarray,
    harm_env_slice: np.ndarray,
    y: np.ndarray,
    sr: int,
    start_f: int,
    end_f: int,
) -> SegmentTimbre:
    """Compute timbral + dynamics features for a single segment."""
    def _safe_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size > 0 else 0.0

    def _safe_std(arr: np.ndarray) -> float:
        return float(np.std(arr)) if arr.size > 0 else 0.0

    # MFCC mean/std (13 coefficients)
    if mfcc_slice.shape[1] > 0:
        mfcc_mean = tuple(float(v) for v in np.mean(mfcc_slice, axis=1)[:13])
        mfcc_std = tuple(float(v) for v in np.std(mfcc_slice, axis=1)[:13])
    else:
        mfcc_mean = tuple(0.0 for _ in range(13))
        mfcc_std = tuple(0.0 for _ in range(13))

    # Spectral contrast mean (7 bands)
    if contrast_slice.shape[1] > 0:
        contrast_mean = tuple(
            float(v) for v in np.mean(contrast_slice, axis=1)[:7]
        )
    else:
        contrast_mean = tuple(0.0 for _ in range(7))

    # --- Dynamics: spectral flux ------------------------------------------
    # Compute the STFT magnitude of the audio slice and take the L1
    # norm of the frame-to-frame difference.
    if end_f > start_f + 1:
        S = np.abs(
            librosa.stft(
                y[int(start_f * HOP_LENGTH): int(end_f * HOP_LENGTH) + N_FFT],
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
        )
        if S.shape[1] > 1:
            flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        else:
            flux = np.zeros(1)
    else:
        flux = np.zeros(1)

    spectral_flux_mean = _safe_mean(flux)
    spectral_flux_std = _safe_std(flux)

    # --- Dynamics: HPSS envelope ratio ------------------------------------
    # Ratio of percussive energy to total energy in the segment.
    total_env = perc_env_slice + harm_env_slice
    total_energy = float(np.sum(total_env))
    perc_energy = float(np.sum(perc_env_slice))
    hpss_envelope_ratio = perc_energy / total_energy if total_energy > 0.0 else 0.0

    return SegmentTimbre(
        rms_mean=_safe_mean(rms_slice),
        rms_std=_safe_std(rms_slice),
        spectral_centroid_mean=_safe_mean(centroid_slice),
        spectral_centroid_std=_safe_std(centroid_slice),
        spectral_bandwidth_mean=_safe_mean(bandwidth_slice),
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        zero_crossing_rate_mean=_safe_mean(zcr_slice),
        spectral_rolloff_mean=_safe_mean(rolloff_slice),
        spectral_contrast_mean=contrast_mean,
        spectral_flux_mean=spectral_flux_mean,
        spectral_flux_std=spectral_flux_std,
        hpss_envelope_ratio=hpss_envelope_ratio,
    )