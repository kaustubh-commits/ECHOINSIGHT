"""Integration tests for the unified ExtractionPipeline."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.analysis.extractor import extract_full_pipeline
from src.core.package import SongPackage
from src.core.song_dna import SongDNA
from src.core.stems import SeparatedTrack


@pytest.fixture
def synthetic_audio_file(tmp_path) -> Path:
    audio_path = tmp_path / "test_track.wav"
    sr = 22050
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 120 BPM clicks + harmonic tone
    clicks = (t % 0.5 < 0.02).astype(np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    audio_data = clicks + tone
    sf.write(str(audio_path), audio_data, sr)
    return audio_path


def test_extract_full_pipeline_pillar1_only(synthetic_audio_file, tmp_path):
    pkg = extract_full_pipeline(
        file_path=synthetic_audio_file,
        data_root=tmp_path,
        separate_stems=False,
    )
    assert isinstance(pkg, SongPackage)
    assert isinstance(pkg.song_dna, SongDNA)
    assert pkg.song_dna.summary.tempo_estimate > 0
    assert pkg.separated_track is None
    assert pkg.stem_analysis is None


def test_extract_full_pipeline_with_cached_stems(synthetic_audio_file, tmp_path):
    stems_dir = tmp_path / "stems" / "test_track"
    stems_dir.mkdir(parents=True)
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    test_tone = 0.2 * np.sin(2 * np.pi * 440 * t)

    for name in ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]:
        sf.write(str(stems_dir / name), test_tone, sr)

    pkg = extract_full_pipeline(
        file_path=synthetic_audio_file,
        data_root=tmp_path,
        separate_stems=True,
    )

    assert isinstance(pkg, SongPackage)
    assert isinstance(pkg.song_dna, SongDNA)
    assert isinstance(pkg.separated_track, SeparatedTrack)
    assert pkg.stem_analysis is not None
    assert pkg.stem_analysis.vocal_energy_ratio > 0.0