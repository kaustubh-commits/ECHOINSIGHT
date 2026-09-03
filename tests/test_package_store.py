"""Unit tests for SongPackage serialization and persistence."""

import dataclasses
from pathlib import Path
import pytest

from src.analysis.package_store import PackageStore
from src.analysis.stem_analyzer import StemAnalysisResult
from src.core.enums import LabelType
from src.core.identifiers import SegmentIndex, SongID
from src.core.package import SongPackage
from src.core.song_dna import SongDNA, SongMetadata, SongSummary
from src.core.stems import SeparatedTrack, StemPaths
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
from src.core.timeline import Timeline


@pytest.fixture
def dummy_song_package(tmp_path) -> SongPackage:
    song_id = SongID("test_song_01")
    metadata = SongMetadata(
        song_id=song_id,
        filename="test_song.wav",
        duration=120.0,
        sample_rate=22050,
        format="wav",
    )
    summary = SongSummary(
        tempo_estimate=120.0,
        tempo_confidence=1.0,
        key_estimate="A minor",
        key_confidence=0.85,
        rms_energy_mean=0.25,
        spectral_centroid_mean=2100.0,
        spectral_bandwidth_mean=1800.0,
        mfcc_mean=tuple(float(i) for i in range(13)),
        chroma_mean=tuple(float(i) for i in range(12)),
        tuning_offset=0.0,
    )

    timing = SegmentTiming(start_time=0.0, end_time=120.0)
    rhythm = SegmentRhythm(tempo=120.0, tempo_confidence=1.0, onset_strength_mean=0.5)
    harmony = SegmentHarmony(
        key="A minor",
        key_confidence=0.85,
        chroma_mean=tuple(float(i) for i in range(12)),
        chroma_std=(0.0,) * 12,
        tonnetz_mean=(0.0,) * 6,
    )
    timbre = SegmentTimbre(
        rms_mean=0.25,
        rms_std=0.0,
        spectral_centroid_mean=2100.0,
        spectral_centroid_std=0.0,
        spectral_bandwidth_mean=1800.0,
        mfcc_mean=tuple(float(i) for i in range(13)),
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

    structure = StructureDNA(segments=(segment,), total_duration=120.0)
    song_dna = SongDNA(
        metadata=metadata,
        summary=summary,
        structure=structure,
        schema_version="2.0.0",
    )

    stems = StemPaths(
        vocals=tmp_path / "vocals.wav",
        drums=tmp_path / "drums.wav",
        bass=tmp_path / "bass.wav",
        other=tmp_path / "other.wav",
    )
    separated_track = SeparatedTrack(
        track_id="test_song_01",
        original_audio=tmp_path / "test_song.wav",
        sample_rate=44100,
        duration_seconds=120.0,
        stems=stems,
        model_name="htdemucs_custom",
    )
    stem_analysis = StemAnalysisResult(
        tempo_bpm=120.0,
        beat_frames=[0, 43, 86],
        vocal_energy_ratio=0.35,
        bass_energy_ratio=0.25,
        drums_energy_ratio=0.30,
        other_energy_ratio=0.10,
        per_stem_rms={"vocals": 0.15, "drums": 0.20, "bass": 0.18, "other": 0.08},
    )

    return SongPackage(
        song_dna=song_dna,
        separated_track=separated_track,
        stem_analysis=stem_analysis,
    )


def test_song_package_roundtrip(dummy_song_package, tmp_path):
    json_path = tmp_path / "package.json"
    PackageStore.save(dummy_song_package, json_path)

    loaded_package = PackageStore.load(json_path)

    assert loaded_package.has_stems
    assert loaded_package.song_dna.metadata.song_id == "test_song_01"
    assert loaded_package.song_dna.summary.key_estimate == "A minor"
    assert loaded_package.separated_track.model_name == "htdemucs_custom"
    assert loaded_package.stem_analysis.vocal_energy_ratio == 0.35
    assert loaded_package.stem_analysis.beat_frames == [0, 43, 86]


def test_song_package_preserves_timeline_roundtrip(dummy_song_package, tmp_path):
    """SongDNA.with(Timeline) -> SongPackage -> PackageStore -> Timeline intact."""
    # 120 BPM grid over a 120 s track: beats every 0.5 s, bars every 4 beats.
    duration = 120.0
    beats = tuple(round(i * 0.5, 6) for i in range(240))  # 0.0 .. 119.5
    timeline = Timeline.build(beats=beats, duration=duration)

    dna_with_timeline = dataclasses.replace(dummy_song_package.song_dna, timeline=timeline)
    package_with_timeline = dataclasses.replace(
        dummy_song_package,
        song_dna=dna_with_timeline,
        separated_track=None,
        stem_analysis=None,
    )

    json_path = tmp_path / "package_with_timeline.json"
    PackageStore.save(package_with_timeline, json_path)
    loaded = PackageStore.load(json_path)

    loaded_tl = loaded.song_dna.timeline
    assert loaded_tl is not None, "Timeline must survive PackageStore round-trip"
    assert tuple(loaded_tl.beats) == tuple(timeline.beats)
    assert tuple(loaded_tl.downbeats) == tuple(timeline.downbeats)
    assert loaded_tl.num_bars == timeline.num_bars == len(timeline.bars) == 60
    assert loaded_tl.bars[0].number == 1
    assert abs(loaded_tl.bars[0].start_time - 0.0) < 1e-6
    assert abs(loaded_tl.bars[-1].end_time - duration) < 1e-6
    assert abs(loaded_tl.bars[-1].end_time - loaded.song_dna.metadata.duration) < 1e-3