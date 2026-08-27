"""Stem-aware acoustic analyzer for isolated instrument tracks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import soundfile as sf
import librosa

from src.core.stems import SeparatedTrack


@dataclass(frozen=True)
class StemAnalysisResult:
    """Acoustic feature profile extracted from separated stems."""
    tempo_bpm: float
    beat_frames: List[int]
    vocal_energy_ratio: float
    bass_energy_ratio: float
    drums_energy_ratio: float
    other_energy_ratio: float
    per_stem_rms: Dict[str, float]


class StemAnalyzer:
    """Orchestrates acoustic feature extraction across individual stems."""

    def __init__(self, sample_rate: int = 44100, hop_length: int = 512) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def analyze(self, separated_track: SeparatedTrack) -> StemAnalysisResult:
        """Extracts specialized features from each separated stem."""
        stems = {
            "drums": separated_track.stems.drums,
            "bass": separated_track.stems.bass,
            "other": separated_track.stems.other,
            "vocals": separated_track.stems.vocals,
        }

        stem_audios = {}
        stem_rms = {}

        # 1. Load stems and compute energy
        for name, path in stems.items():
            audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
            # Convert to mono for DSP analysis
            mono = np.mean(audio, axis=1)
            stem_audios[name] = mono
            rms_val = float(np.sqrt(np.mean(mono**2)))
            stem_rms[name] = rms_val

        # 2. Energy ratios
        total_rms = sum(stem_rms.values()) + 1e-9
        energy_ratios = {k: v / total_rms for k, v in stem_rms.items()}

        # 3. Dedicated Beat Tracking on Drums stem (clean transients)
        drums_audio = stem_audios["drums"]
        if np.max(np.abs(drums_audio)) > 1e-4:
            tempo, beats = librosa.beat.beat_track(
                y=drums_audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
            )
            bpm = float(np.atleast_1d(tempo)[0])
            beat_list = beats.tolist()
        else:
            bpm = 0.0
            beat_list = []

        return StemAnalysisResult(
            tempo_bpm=round(bpm, 2),
            beat_frames=beat_list,
            vocal_energy_ratio=round(energy_ratios["vocals"], 4),
            bass_energy_ratio=round(energy_ratios["bass"], 4),
            drums_energy_ratio=round(energy_ratios["drums"], 4),
            other_energy_ratio=round(energy_ratios["other"], 4),
            per_stem_rms={k: round(v, 6) for k, v in stem_rms.items()},
        )