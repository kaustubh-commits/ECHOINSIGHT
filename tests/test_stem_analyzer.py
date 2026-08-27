"""Unit tests for StemAnalyzer."""

from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from src.core.stems import StemPaths, SeparatedTrack
from src.analysis.stem_analyzer import StemAnalyzer, StemAnalysisResult


@pytest.fixture
def dummy_separated_track(tmp_path) -> SeparatedTrack:
    track_dir = tmp_path / "dummy_song"
    track_dir.mkdir(parents=True)
    
    sr = 44100
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Synthetic stems
    drums_audio = np.sin(2 * np.pi * 100 * t) * (t % 0.5 < 0.05)  # 120 BPM clicks
    bass_audio = 0.5 * np.sin(2 * np.pi * 60 * t)                # 60 Hz bass sine
    vocal_audio = 0.3 * np.sin(2 * np.pi * 440 * t)               # 440 Hz A4 tone
    other_audio = 0.2 * np.sin(2 * np.pi * 880 * t)               # 880 Hz harmonic
    
    sf.write(str(track_dir / "drums.wav"), drums_audio, sr)
    sf.write(str(track_dir / "bass.wav"), bass_audio, sr)
    sf.write(str(track_dir / "vocals.wav"), vocal_audio, sr)
    sf.write(str(track_dir / "other.wav"), other_audio, sr)
    
    stems = StemPaths(
        drums=track_dir / "drums.wav",
        bass=track_dir / "bass.wav",
        vocals=track_dir / "vocals.wav",
        other=track_dir / "other.wav",
    )
    
    return SeparatedTrack(
        track_id="dummy_song",
        original_audio=track_dir / "original.wav",
        sample_rate=sr,
        duration_seconds=duration,
        stems=stems,
        model_name="htdemucs_custom",
    )


def test_stem_analyzer_execution(dummy_separated_track):
    analyzer = StemAnalyzer()
    result = analyzer.analyze(dummy_separated_track)

    assert isinstance(result, StemAnalysisResult)
    assert result.vocal_energy_ratio > 0.0
    assert result.bass_energy_ratio > 0.0
    assert result.drums_energy_ratio > 0.0
    assert result.other_energy_ratio > 0.0
    assert sum(result.per_stem_rms.values()) > 0.0
    assert isinstance(result.tempo_bpm, float)