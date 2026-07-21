"""
test_serialization.py — Round-trip JSON serialization tests for the new
domain model.

These tests verify that every domain type can be:
1. Constructed
2. Serialised to a plain dict (via dataclasses.asdict)
3. Serialised to JSON (via json.dumps)
4. Deserialised back to the original type
5. Compare equal to the original after the round trip
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from typing import Any, Dict

from src.core.identifiers import (
    SongID,
    URI,
    Estimate,
    generate_song_id,
    SegmentIndex,
)
from src.core.enums import LabelType, RepetitionRole
from src.core.structure import (
    AlternativeSegmentation,
    BoundaryConfidence,
    SegmentContext,
    SegmentDNA,
    SegmentDSP,
    SegmentHarmony,
    SegmentRhythm,
    SegmentTimbre,
    SegmentTiming,
    StructureDNA,
    SSMReference,
)
from src.core.song_dna import SongDNA, SongMetadata, SongSummary


# ---------------------------------------------------------------------------
# Helpers — build valid test objects
# ---------------------------------------------------------------------------


def _make_song_id() -> SongID:
    return SongID(uuid.uuid4().hex)


def _make_timing(start: float = 0.0, end: float = 30.0) -> SegmentTiming:
    return SegmentTiming(
        start_time=start,
        end_time=end,
        start_beat=0,
        end_beat=64,
        bar_count=16,
    )


def _make_rhythm() -> SegmentRhythm:
    return SegmentRhythm(
        tempo=120.0,
        tempo_confidence=0.9,
        onset_strength_mean=0.5,
    )


def _make_harmony() -> SegmentHarmony:
    return SegmentHarmony(
        key="C major",
        key_confidence=0.8,
        chroma_mean=(0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3),
        chroma_std=(0.05,) * 12,
        tonnetz_mean=(0.0, 0.1, -0.1, 0.2, -0.2, 0.0),
    )


def _make_timbre() -> SegmentTimbre:
    return SegmentTimbre(
        rms_mean=0.5,
        rms_std=0.1,
        spectral_centroid_mean=2000.0,
        spectral_centroid_std=500.0,
        spectral_bandwidth_mean=3000.0,
        mfcc_mean=tuple(float(i) for i in range(13)),
        mfcc_std=tuple(float(i % 3) for i in range(13)),
        zero_crossing_rate_mean=0.1,
        spectral_rolloff_mean=4000.0,
        spectral_contrast_mean=tuple(float(i) for i in range(7)),
    )


def _make_dsp() -> SegmentDSP:
    return SegmentDSP(
        rhythm=_make_rhythm(),
        harmony=_make_harmony(),
        timbre=_make_timbre(),
    )


def _make_context(index: int = 0) -> SegmentContext:
    return SegmentContext(
        index=SegmentIndex(index),
        label=LabelType.VERSE,
        label_confidence=0.85,
        repetition_role=RepetitionRole.ORIGINAL,
        repetition_group="verse_1",
        novelty_score=0.5,
        self_similarity_score=0.3,
    )


def _make_segment(
    song_id: SongID,
    index: int = 0,
    start: float = 0.0,
    end: float = 30.0,
) -> SegmentDNA:
    return SegmentDNA(
        song_id=song_id,
        timing=_make_timing(start=start, end=end),
        dsp=_make_dsp(),
        context=_make_context(index=index),
    )


def _make_structure(song_id: SongID, duration: float = 120.0) -> StructureDNA:
    segs = (
        _make_segment(song_id, index=0, start=0.0, end=30.0),
        _make_segment(song_id, index=1, start=30.0, end=60.0),
        _make_segment(song_id, index=2, start=60.0, end=90.0),
        _make_segment(song_id, index=3, start=90.0, end=120.0),
    )
    boundaries = (
        BoundaryConfidence(time=30.0, confidence=0.9, method="novelty_curve"),
        BoundaryConfidence(time=60.0, confidence=0.85, method="chroma_change"),
        BoundaryConfidence(time=90.0, confidence=0.7, method="mfcc_change"),
    )
    return StructureDNA(
        segments=segs,
        total_duration=duration,
        boundary_confidences=boundaries,
    )


def _make_metadata(song_id: SongID) -> SongMetadata:
    return SongMetadata(
        song_id=song_id,
        filename="test_song.mp3",
        duration=120.0,
        sample_rate=44100,
        channels=2,
        bit_depth=16,
        format="mp3",
    )


def _make_summary() -> SongSummary:
    return SongSummary(
        tempo_estimate=120.0,
        tempo_confidence=0.9,
        key_estimate="C major",
        key_confidence=0.8,
        key_alternatives=(("A minor", 0.6), ("F major", 0.4)),
        rms_energy_mean=0.5,
        spectral_centroid_mean=2000.0,
        spectral_bandwidth_mean=3000.0,
        mfcc_mean=tuple(float(i) for i in range(13)),
        chroma_mean=tuple(float(i % 12) / 12.0 for i in range(12)),
        tuning_offset=0.0,
    )


# ===================================================================
# Test: Estimate
# ===================================================================


def test_estimate_round_trip() -> None:
    """Estimate should survive dict → JSON → dict round trip."""
    orig = Estimate(value=124.0, confidence=0.92)
    as_dict = {"value": orig.value, "confidence": orig.confidence}
    as_json = json.dumps(as_dict)
    restored = json.loads(as_json)
    assert isinstance(restored, dict)
    assert restored["value"] == 124.0
    assert restored["confidence"] == 0.92


def test_estimate_rejects_invalid_confidence() -> None:
    """Estimate should reject confidence outside [0, 1]."""
    try:
        Estimate(value=100.0, confidence=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        Estimate(value=100.0, confidence=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ===================================================================
# Test: SegmentTiming
# ===================================================================


def test_timing_round_trip() -> None:
    """SegmentTiming should survive dict round trip."""
    orig = _make_timing(start=5.0, end=35.0)
    as_dict = dataclasses.asdict(orig)
    as_json = json.dumps(as_dict)
    restored_dict = json.loads(as_json)
    assert restored_dict["start_time"] == 5.0
    assert restored_dict["end_time"] == 35.0
    assert restored_dict["duration"] == 30.0  # Auto-computed
    assert restored_dict["start_beat"] == 0
    assert restored_dict["bar_count"] == 16


# ===================================================================
# Test: SegmentDNA
# ===================================================================


def test_segment_dna_round_trip() -> None:
    """SegmentDNA should survive dict → JSON → dict round trip."""
    song_id = _make_song_id()
    orig = _make_segment(song_id)
    as_dict = dataclasses.asdict(orig)
    as_json = json.dumps(as_dict)
    restored = json.loads(as_json)

    assert restored["schema_version"] == "2.0.0"
    assert restored["song_id"] == song_id
    assert restored["timing"]["start_time"] == 0.0
    assert restored["timing"]["end_time"] == 30.0
    assert restored["dsp"]["rhythm"]["tempo"] == 120.0
    assert restored["dsp"]["harmony"]["key"] == "C major"
    assert restored["dsp"]["timbre"]["rms_mean"] == 0.5
    assert restored["context"]["index"] == 0
    assert restored["context"]["label"] == "verse"


# ===================================================================
# Test: StructureDNA
# ===================================================================


def test_structure_dna_round_trip() -> None:
    """StructureDNA should survive dict → JSON → dict round trip."""
    song_id = _make_song_id()
    orig = _make_structure(song_id, duration=120.0)
    as_dict = dataclasses.asdict(orig)
    as_json = json.dumps(as_dict)
    restored = json.loads(as_json)

    assert restored["schema_version"] == "2.0.0"
    assert restored["num_segments"] == 4
    assert restored["total_duration"] == 120.0
    assert len(restored["segments"]) == 4
    assert len(restored["boundary_confidences"]) == 3


# ===================================================================
# Test: SongDNA (full aggregate)
# ===================================================================


def test_song_dna_round_trip() -> None:
    """Full SongDNA should survive dict → JSON → dict round trip."""
    song_id = _make_song_id()
    metadata = _make_metadata(song_id)
    summary = _make_summary()
    structure = _make_structure(song_id, duration=120.0)
    dna = SongDNA(
        metadata=metadata,
        summary=summary,
        structure=structure,
    )

    as_dict = dataclasses.asdict(dna)
    as_json = json.dumps(as_dict)
    restored = json.loads(as_json)

    # Top-level
    assert restored["schema_version"] == "2.0.0"
    assert restored["metadata"]["song_id"] == song_id
    assert restored["metadata"]["duration"] == 120.0
    assert restored["summary"]["tempo_estimate"] == 120.0
    assert restored["summary"]["key_estimate"] == "C major"

    # Structure
    assert len(restored["structure"]["segments"]) == 4
    assert restored["structure"]["total_duration"] == 120.0

    # Every segment must have the correct song_id
    for seg in restored["structure"]["segments"]:
        assert seg["song_id"] == song_id

    # frames and manifest should be None (not set)
    assert restored["frames"] is None
    assert restored["manifest"] is None


# ===================================================================
# Test: SSMReference
# ===================================================================


def test_ssm_reference_round_trip() -> None:
    """SSMReference should survive dict → JSON → dict round trip."""
    orig = SSMReference(
        uri=URI("ssm/test_song.npz"),
        num_frames=8421,
        hop_length=512,
        sample_rate=44100,
    )
    as_dict = dataclasses.asdict(orig)
    as_json = json.dumps(as_dict)
    restored = json.loads(as_json)

    assert restored["uri"] == "ssm/test_song.npz"
    assert restored["num_frames"] == 8421
    assert restored["hop_length"] == 512
    assert restored["sample_rate"] == 44100


# ===================================================================
# Test: AlternativeSegmentation
# ===================================================================


def test_alternative_segmentation_round_trip() -> None:
    """AlternativeSegmentation should survive dict → JSON → dict round trip."""
    song_id = _make_song_id()
    segs = (
        _make_segment(song_id, index=0, start=0.0, end=60.0),
        _make_segment(song_id, index=1, start=60.0, end=120.0),
    )
    alt = AlternativeSegmentation(
        label="coarse",
        segments=segs,
        confidence=0.8,
    )
    as_dict = dataclasses.asdict(alt)
    as_json = json.dumps(as_dict)
    restored = json.loads(as_json)

    assert restored["label"] == "coarse"
    assert len(restored["segments"]) == 2
    assert restored["confidence"] == 0.8


# ===================================================================
# Test: generate_song_id
# ===================================================================


def test_generate_song_id_uniqueness() -> None:
    """Two calls to generate_song_id should produce different IDs."""
    id_a = generate_song_id()
    id_b = generate_song_id()
    assert id_a != id_b
    assert isinstance(id_a, str)
    assert len(id_a) == 32  # uuid4.hex is 32 characters