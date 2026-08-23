"""
Unit tests for the stem separation engine and domain models.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.core.stems import SeparatedTrack, StemPaths
from src.analysis.separator import EchoSeparator


def test_stem_paths_dataclass(tmp_path: Path) -> None:
    paths = StemPaths(
        vocals=tmp_path / "vocals.wav",
        drums=tmp_path / "drums.wav",
        bass=tmp_path / "bass.wav",
        other=tmp_path / "other.wav",
    )
    assert paths.vocals == tmp_path / "vocals.wav"
    assert paths.drums.name == "drums.wav"
    assert paths.bass.name == "bass.wav"
    assert paths.other.name == "other.wav"


def test_separated_track_dataclass(tmp_path: Path) -> None:
    stem_paths = StemPaths(
        vocals=tmp_path / "vocals.wav",
        drums=tmp_path / "drums.wav",
        bass=tmp_path / "bass.wav",
        other=tmp_path / "other.wav",
    )
    track = SeparatedTrack(
        track_id="test_001",
        original_audio=tmp_path / "mix.wav",
        sample_rate=44100,
        duration_seconds=180.0,
        stems=stem_paths,
    )
    assert track.track_id == "test_001"
    assert track.duration_seconds == 180.0
    assert track.stems.vocals == tmp_path / "vocals.wav"
    assert track.model_name == "htdemucs"


def test_separator_initialization() -> None:
    separator = EchoSeparator(shifts=1, overlap=0.25)
    assert separator.model_name == "htdemucs"
    assert separator.shifts == 1
    assert separator.overlap == 0.25
    assert separator._model is None  # Lazy loading verification