"""
test_milestone_m1.py — Validation of Milestone M1 (Complete Unified Pipeline).

Tests: Audio -> ExtractionPipeline -> SongPackage -> PackageStore -> Reload.
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.analysis.extractor import extract_full_pipeline
from src.analysis.package_store import PackageStore
from src.core.package import SongPackage


@pytest.fixture
def mock_tracks(tmp_path) -> tuple[Path, Path]:
    """Generates two distinct synthetic test tracks."""
    sr = 22050
    duration = 2.5

    # Track A: 120 BPM pulses + 440 Hz
    t_a = np.linspace(0, duration, int(sr * duration), endpoint=False)
    clicks_a = (t_a % 0.5 < 0.02).astype(np.float32)
    tone_a = 0.3 * np.sin(2 * np.pi * 440 * t_a)
    track_a_path = tmp_path / "beyond.wav"
    sf.write(str(track_a_path), clicks_a + tone_a, sr)

    # Track B: 140 BPM pulses + 523.25 Hz
    t_b = np.linspace(0, duration, int(sr * duration), endpoint=False)
    clicks_b = (t_b % (60.0 / 140.0) < 0.02).astype(np.float32)
    tone_b = 0.3 * np.sin(2 * np.pi * 523.25 * t_b)
    track_b_path = tmp_path / "contact.wav"
    sf.write(str(track_b_path), clicks_b + tone_b, sr)

    # Create cached mock stems for both tracks
    for name in ["beyond", "contact"]:
        stems_dir = tmp_path / "stems" / name
        stems_dir.mkdir(parents=True, exist_ok=True)
        tone = 0.2 * np.sin(2 * np.pi * 440 * t_a)
        for stem_name in ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]:
            sf.write(str(stems_dir / stem_name), tone, sr)

    return track_a_path, track_b_path


def test_milestone_m1_complete_vertical_slice(mock_tracks, tmp_path):
    track_a_path, track_b_path = mock_tracks

    # 1. Run full extraction pipeline on Track A
    pkg_a_raw = extract_full_pipeline(
        file_path=track_a_path,
        data_root=tmp_path,
        separate_stems=True,
    )
    song_pkg_a = SongPackage(
        song_dna=pkg_a_raw.song_dna,
        separated_track=pkg_a_raw.separated_track,
        stem_analysis=pkg_a_raw.stem_analysis,
    )

    # 2. Persist SongPackage A
    pkg_a_json = tmp_path / "dna" / "beyond_package.json"
    PackageStore.save(song_pkg_a, pkg_a_json)

    # 3. Reload SongPackage A
    loaded_pkg_a = PackageStore.load(pkg_a_json)

    # 4. Verify all layers exist and are intact
    assert loaded_pkg_a.has_stems
    assert loaded_pkg_a.song_dna.metadata.filename == str(track_a_path)
    assert loaded_pkg_a.song_dna.summary.tempo_estimate > 0
    assert loaded_pkg_a.separated_track is not None
    assert loaded_pkg_a.separated_track.stems.drums.exists()
    assert loaded_pkg_a.stem_analysis is not None
    assert loaded_pkg_a.stem_analysis.drums_energy_ratio > 0

    # 5. Run full extraction on Track B
    pkg_b_raw = extract_full_pipeline(
        file_path=track_b_path,
        data_root=tmp_path,
        separate_stems=True,
    )
    song_pkg_b = SongPackage(
        song_dna=pkg_b_raw.song_dna,
        separated_track=pkg_b_raw.separated_track,
        stem_analysis=pkg_b_raw.stem_analysis,
    )
    pkg_b_json = tmp_path / "dna" / "contact_package.json"
    PackageStore.save(song_pkg_b, pkg_b_json)
    loaded_pkg_b = PackageStore.load(pkg_b_json)

    assert loaded_pkg_b.has_stems
    assert loaded_pkg_b.song_dna.metadata.filename == str(track_b_path)