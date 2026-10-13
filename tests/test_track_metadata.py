"""Tests for honest track metadata: file tags, sources, and no fabrication."""

import json
from typing import Optional

import numpy as np
import pytest
import soundfile as sf

from src.analysis.package_store import PackageStore
from src.analysis.track_metadata import build_enriched_package, read_file_tags
from src.core.enums import LabelType
from src.core.identifiers import SegmentIndex, SongID
from src.core.package import EnrichedSongPackage, SongPackage
from src.core.song_dna import SongDNA, SongMetadata, SongSummary
from src.core.spotify import SpotifyMetadata
from src.core.structure import (
    SegmentContext,
    SegmentDNA,
    SegmentDSP,
    SegmentHarmony,
    SegmentRhythm,
    SegmentTimbre,
    SegmentTiming,
    StructureDNA,
)
from src.core.track_info import SourcedText, TrackInfo


@pytest.fixture
def base_song_package() -> SongPackage:
    song_id = SongID("metadata_test_01")
    metadata = SongMetadata(
        song_id=song_id,
        filename="some_track.wav",
        duration=100.0,
        sample_rate=22050,
        format="wav",
    )
    summary = SongSummary(tempo_estimate=106.0, tempo_confidence=0.95)
    timing = SegmentTiming(start_time=0.0, end_time=100.0)
    rhythm = SegmentRhythm(tempo=106.0, tempo_confidence=0.95, onset_strength_mean=0.4)
    dsp = SegmentDSP(rhythm=rhythm, harmony=SegmentHarmony(), timbre=SegmentTimbre())
    context = SegmentContext(index=SegmentIndex(0), label=LabelType.UNKNOWN)
    segment = SegmentDNA(song_id=song_id, timing=timing, dsp=dsp, context=context)
    structure = StructureDNA(segments=(segment,), total_duration=100.0)
    song_dna = SongDNA(metadata=metadata, summary=summary, structure=structure)
    return SongPackage(song_dna=song_dna)


def _write_wav(path, title: Optional[str] = None, artist: Optional[str] = None) -> None:
    with sf.SoundFile(str(path), "w", samplerate=22050, channels=1) as f:
        if title is not None:
            f.title = title
        if artist is not None:
            f.artist = artist
        f.write(np.zeros(2205, dtype="float32"))


class _FakeProvider:
    name = "fake"

    def __init__(self, result: Optional[SpotifyMetadata]) -> None:
        self.result = result

    def lookup(self, track: TrackInfo, duration: float) -> Optional[SpotifyMetadata]:
        return self.result


# --- SourcedText invariants -------------------------------------------------


def test_sourced_text_defaults_to_unknown():
    field = SourcedText()
    assert field.value is None
    assert field.source == "unknown"
    assert not field.is_known


@pytest.mark.parametrize(
    "value, source",
    [
        ("Some title", "unknown"),  # known value without a source
        (None, "file_tags"),        # source claimed for a missing value
        ("Some title", "made_up"),  # invalid source
        ("   ", "file_tags"),       # blank value
    ],
)
def test_sourced_text_rejects_inconsistent_values(value, source):
    with pytest.raises(ValueError):
        SourcedText(value=value, source=source)


# --- File tags ---------------------------------------------------------------


def test_read_file_tags_uses_embedded_tags(tmp_path):
    path = tmp_path / "tagged.wav"
    _write_wav(path, title="Night Drive", artist="Test Artist")

    info = read_file_tags(path)

    assert info.title == SourcedText("Night Drive", "file_tags")
    assert info.artist == SourcedText("Test Artist", "file_tags")
    assert info.album == SourcedText()
    assert info.release_date == SourcedText()


def test_read_file_tags_untagged_file_is_all_unknown(tmp_path):
    path = tmp_path / "untagged.wav"
    _write_wav(path)
    assert read_file_tags(path) == TrackInfo()


def test_read_file_tags_unreadable_file_is_all_unknown(tmp_path):
    assert read_file_tags(tmp_path / "missing.mp3") == TrackInfo()


# --- Enrichment without fabrication -----------------------------------------


def test_build_without_provider_never_invents_catalogue_data(base_song_package, tmp_path):
    path = tmp_path / "untagged.wav"
    _write_wav(path)

    enriched = build_enriched_package(base_song_package, path)

    assert isinstance(enriched, EnrichedSongPackage)
    assert enriched.spotify is None
    assert not enriched.is_enriched
    assert enriched.track_info == TrackInfo()


def test_provider_fills_only_missing_fields(base_song_package, tmp_path):
    path = tmp_path / "tagged.wav"
    _write_wav(path, title="Tag Title")
    match = SpotifyMetadata(
        track_id="4cOdK2wGLETKBW3PvgPWqT",
        track_name="Catalogue Title",
        artist_name="Catalogue Artist",
        album_name="Catalogue Album",
    )

    enriched = build_enriched_package(base_song_package, path, _FakeProvider(match))

    assert enriched.is_enriched
    assert enriched.track_info.title == SourcedText("Tag Title", "file_tags")
    assert enriched.track_info.artist == SourcedText("Catalogue Artist", "spotify")
    assert enriched.track_info.album == SourcedText("Catalogue Album", "spotify")
    assert enriched.track_info.release_date == SourcedText()


def test_provider_without_match_leaves_fields_unknown(base_song_package, tmp_path):
    path = tmp_path / "untagged.wav"
    _write_wav(path)

    enriched = build_enriched_package(base_song_package, path, _FakeProvider(None))

    assert enriched.spotify is None
    assert enriched.track_info == TrackInfo()


# --- Persistence -------------------------------------------------------------


def test_track_info_round_trips(base_song_package, tmp_path):
    audio = tmp_path / "tagged.wav"
    _write_wav(audio, title="Night Drive", artist="Test Artist")
    enriched = build_enriched_package(base_song_package, audio)

    json_path = tmp_path / "pkg.json"
    PackageStore.save(enriched, json_path)
    reloaded = PackageStore.load(json_path)

    assert reloaded.track_info == enriched.track_info
    assert reloaded.spotify is None


def _saved_dict(base_song_package, spotify_metadata):
    data = PackageStore.to_dict(base_song_package)
    data["schema_version"] = "2.1.0"
    del data["track_info"]
    data["spotify_metadata"] = spotify_metadata
    return data


@pytest.mark.parametrize(
    "stub_id, art_url",
    [
        ("offline_id", None),
        ("mock_beyond_id", "https://images.example.com/beyond_art.jpg"),
        # The stub echoed real-looking IDs too; its fake art URL gives it away.
        ("4cOdK2wGLETKBW3PvgPWqT", "https://images.example.com/beyond_art.jpg"),
    ],
)
def test_old_fabricated_spotify_stubs_are_dropped_on_load(
    base_song_package, tmp_path, stub_id, art_url
):
    stub = {
        "track_id": stub_id,
        "track_name": "Beyond",
        "artist_name": "Daft Punk",
        "album_name": "Random Access Memories",
        "album_art_url": art_url,
        "popularity": 72,
    }
    json_path = tmp_path / "old.json"
    json_path.write_text(json.dumps(_saved_dict(base_song_package, stub)))

    reloaded = PackageStore.load(json_path)

    assert reloaded.spotify is None
    assert not reloaded.is_enriched
    assert reloaded.track_info is None


def test_old_file_with_real_looking_spotify_record_still_loads(base_song_package, tmp_path):
    record = {
        "track_id": "4cOdK2wGLETKBW3PvgPWqT",
        "track_name": "Some Song",
        "artist_name": "Some Artist",
        "album_name": "Some Album",
    }
    json_path = tmp_path / "old.json"
    json_path.write_text(json.dumps(_saved_dict(base_song_package, record)))

    reloaded = PackageStore.load(json_path)

    assert reloaded.spotify is not None
    assert reloaded.spotify.track_id == "4cOdK2wGLETKBW3PvgPWqT"
