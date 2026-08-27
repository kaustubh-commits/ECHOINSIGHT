"""Unit tests for EchoSeparator and stem separation data contracts."""

from pathlib import Path
import pytest

from src.analysis.separator import EchoSeparator
from src.core.stems import SeparatedTrack, StemPaths


def test_stem_paths_immutability():
    stems = StemPaths(
        vocals="path/vocals.wav",
        drums="path/drums.wav",
        bass="path/bass.wav",
        other="path/other.wav",
    )
    assert stems.drums == Path("path/drums.wav")
    assert stems.bass == Path("path/bass.wav")
    assert stems.other == Path("path/other.wav")
    assert stems.vocals == Path("path/vocals.wav")

    with pytest.raises((AttributeError, TypeError)):
        stems.drums = Path("new/path.wav")


def test_separated_track_structure():
    stems = StemPaths(
        vocals=Path("v.wav"),
        drums=Path("d.wav"),
        bass=Path("b.wav"),
        other=Path("o.wav"),
    )
    track = SeparatedTrack(
        track_id="test_song_01",
        original_audio=Path("test_song.wav"),
        sample_rate=44100,
        duration_seconds=180.5,
        stems=stems,
        model_name="htdemucs_custom",
    )
    assert track.track_id == "test_song_01"
    assert track.original_audio == Path("test_song.wav")
    assert track.sample_rate == 44100
    assert track.duration_seconds == 180.5
    assert track.model_name == "htdemucs_custom"
    assert track.stems.vocals == Path("v.wav")


def test_echo_separator_initialization(tmp_path):
    sep = EchoSeparator(cache_dir=tmp_path)
    assert sep.model_name == "htdemucs"
    assert sep.model_path is None
    assert not sep.is_custom_checkpoint

    dummy_checkpoint = tmp_path / "model.th"
    dummy_checkpoint.touch()

    custom_sep = EchoSeparator(model_path=dummy_checkpoint, cache_dir=tmp_path)
    assert custom_sep.is_custom_checkpoint
    assert custom_sep.model_path == dummy_checkpoint


def test_echo_separator_missing_file_raises(tmp_path):
    sep = EchoSeparator(cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        sep.separate("non_existent_audio_file.wav")


def test_echo_separator_cache_hit(tmp_path):
    track_dir = tmp_path / "cached_track"
    track_dir.mkdir(parents=True)

    for name in ["drums.wav", "bass.wav", "other.wav", "vocals.wav"]:
        (track_dir / name).write_text("dummy audio data")

    dummy_input = tmp_path / "cached_track.wav"
    dummy_input.write_text("dummy content")

    sep = EchoSeparator(cache_dir=tmp_path)
    result = sep.separate(audio_path=dummy_input, output_dir=track_dir)

    assert isinstance(result, SeparatedTrack)
    assert result.track_id == "cached_track"
    assert result.original_audio == dummy_input
    assert result.stems.drums == track_dir / "drums.wav"