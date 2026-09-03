"""Unit tests for Phase 2 Spotify Enrichment and resilience invariants."""

from pathlib import Path
import pytest

from src.analysis.package_store import PackageStore
from src.analysis.spotify_enricher import SpotifyEnricher
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


@pytest.fixture
def base_song_package() -> SongPackage:
    song_id = SongID("spotify_test_01")
    metadata = SongMetadata(
        song_id=song_id,
        filename="Daft Punk - Beyond.wav",
        duration=100.0,
        sample_rate=22050,
        format="wav",
    )
    summary = SongSummary(tempo_estimate=106.0, tempo_confidence=0.95)
    timing = SegmentTiming(start_time=0.0, end_time=100.0)
    rhythm = SegmentRhythm(tempo=106.0, tempo_confidence=0.95, onset_strength_mean=0.4)
    harmony = SegmentHarmony(
        key="C major",
        key_confidence=0.8,
        chroma_mean=(0.1,) * 12,
        chroma_std=(0.0,) * 12,
        tonnetz_mean=(0.0,) * 6,
    )
    timbre = SegmentTimbre(
        rms_mean=0.3,
        rms_std=0.0,
        spectral_centroid_mean=2000.0,
        spectral_centroid_std=0.0,
        spectral_bandwidth_mean=1500.0,
        mfcc_mean=(0.0,) * 13,
        mfcc_std=(0.0,) * 13,
        zero_crossing_rate_mean=0.0,
        spectral_rolloff_mean=0.0,
        spectral_contrast_mean=(0.0,) * 7,
        spectral_flux_mean=0.0,
        spectral_flux_std=0.0,
        hpss_envelope_ratio=0.0,
    )
    dsp = SegmentDSP(rhythm=rhythm, harmony=harmony, timbre=timbre)
    context = SegmentContext(index=SegmentIndex(0), label=LabelType.UNKNOWN, label_confidence=0.0)
    segment = SegmentDNA(song_id=song_id, timing=timing, dsp=dsp, context=context)
    structure = StructureDNA(segments=(segment,), total_duration=100.0)
    song_dna = SongDNA(metadata=metadata, summary=summary, structure=structure)
    return SongPackage(song_dna=song_dna)


def test_spotify_metadata_immutability():
    meta = SpotifyMetadata(
        track_id="4cOdK2wGLETKBW3PvgPWqT",
        track_name="Beyond",
        artist_name="Daft Punk",
        album_name="Random Access Memories",
        popularity=70,
    )
    assert meta.track_name == "Beyond"
    assert meta.artist_name == "Daft Punk"


def test_spotify_enricher_graceful_fallback(base_song_package):
    """Enrichment succeeds with fallback metadata even without credentials."""
    enricher = SpotifyEnricher(client_id=None, client_secret=None)
    enriched = enricher.enrich(base_song_package)

    assert isinstance(enriched, EnrichedSongPackage)
    assert enriched.is_enriched
    assert enriched.package.song_dna.metadata.song_id == "spotify_test_01"
    assert enriched.spotify is not None
    assert enriched.spotify.track_name != ""


def test_spotify_enricher_with_id(base_song_package):
    """Enrichment via direct track ID matches expected catalog item."""
    enricher = SpotifyEnricher()
    enriched = enricher.enrich(base_song_package, spotify_id="mock_beyond_id")

    assert enriched.spotify is not None
    assert enriched.spotify.artist_name == "Daft Punk"
    assert enriched.spotify.album_name == "Random Access Memories"
    assert "electronic" in enriched.spotify.genres


def test_enriched_song_package_serialization_roundtrip(base_song_package, tmp_path):
    enricher = SpotifyEnricher()
    enriched = enricher.enrich(base_song_package, spotify_id="mock_beyond_id")

    json_path = tmp_path / "enriched_package.json"
    PackageStore.save(enriched, json_path)

    reloaded = PackageStore.load(json_path)
    assert isinstance(reloaded, EnrichedSongPackage)
    assert reloaded.is_enriched
    assert reloaded.spotify.track_name == "Beyond"
    assert reloaded.spotify.artist_name == "Daft Punk"