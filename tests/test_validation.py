"""
test_validation.py — Validation and invariant enforcement tests.

These tests verify that every domain type correctly rejects invalid state
through its ``__post_init__`` validation.  They test the *failure cases*
that ``test_serialization.py`` does not cover.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.core.identifiers import (
    SongID,
    URI,
    Estimate,
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
)
from src.core.song_dna import SongDNA, SongMetadata, SongSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sid() -> SongID:
    return SongID(uuid.uuid4().hex)


# ===================================================================
# SegmentTiming validation
# ===================================================================


class TestSegmentTimingValidation:
    def test_rejects_negative_start_time(self) -> None:
        with pytest.raises(ValueError, match="start_time"):
            SegmentTiming(start_time=-1.0, end_time=30.0)

    def test_rejects_end_before_or_equal_to_start(self) -> None:
        with pytest.raises(ValueError, match="end_time"):
            SegmentTiming(start_time=30.0, end_time=30.0)
        with pytest.raises(ValueError, match="end_time"):
            SegmentTiming(start_time=30.0, end_time=29.0)

    def test_rejects_negative_beat_indices(self) -> None:
        with pytest.raises(ValueError, match="start_beat"):
            SegmentTiming(start_time=0.0, end_time=30.0, start_beat=-1)
        with pytest.raises(ValueError, match="end_beat"):
            SegmentTiming(start_time=0.0, end_time=30.0, end_beat=-1)

    def test_rejects_end_beat_leq_start_beat(self) -> None:
        with pytest.raises(ValueError, match="end_beat"):
            SegmentTiming(
                start_time=0.0, end_time=30.0,
                start_beat=5, end_beat=5,
            )
        with pytest.raises(ValueError, match="end_beat"):
            SegmentTiming(
                start_time=0.0, end_time=30.0,
                start_beat=5, end_beat=3,
            )

    def test_rejects_invalid_bar_count(self) -> None:
        with pytest.raises(ValueError, match="bar_count"):
            SegmentTiming(start_time=0.0, end_time=30.0, bar_count=0)
        with pytest.raises(ValueError, match="bar_count"):
            SegmentTiming(start_time=0.0, end_time=30.0, bar_count=-1)

    def test_computes_duration_correctly(self) -> None:
        t = SegmentTiming(start_time=10.0, end_time=35.5)
        assert t.duration == pytest.approx(25.5)


# ===================================================================
# SegmentRhythm validation
# ===================================================================


class TestSegmentRhythmValidation:
    def test_rejects_non_positive_tempo(self) -> None:
        with pytest.raises(ValueError, match="tempo"):
            SegmentRhythm(tempo=0.0, tempo_confidence=0.9, onset_strength_mean=0.5)
        with pytest.raises(ValueError, match="tempo"):
            SegmentRhythm(tempo=-120.0, tempo_confidence=0.9, onset_strength_mean=0.5)

    def test_rejects_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="tempo_confidence"):
            SegmentRhythm(tempo=120.0, tempo_confidence=1.5, onset_strength_mean=0.5)
        with pytest.raises(ValueError, match="tempo_confidence"):
            SegmentRhythm(tempo=120.0, tempo_confidence=-0.1, onset_strength_mean=0.5)

    def test_rejects_negative_onset_strength(self) -> None:
        with pytest.raises(ValueError, match="onset_strength_mean"):
            SegmentRhythm(tempo=120.0, tempo_confidence=0.9, onset_strength_mean=-0.1)


# ===================================================================
# SegmentHarmony validation
# ===================================================================


class TestSegmentHarmonyValidation:
    def test_rejects_empty_key(self) -> None:
        with pytest.raises(ValueError, match="key"):
            SegmentHarmony(key="  ", key_confidence=0.8)

    def test_rejects_invalid_key_confidence(self) -> None:
        with pytest.raises(ValueError, match="key_confidence"):
            SegmentHarmony(key="C major", key_confidence=1.5)

    def test_rejects_chroma_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="chroma_mean"):
            SegmentHarmony(chroma_mean=(0.1, 0.2, 0.3))  # 3 != 12
        with pytest.raises(ValueError, match="chroma_std"):
            SegmentHarmony(chroma_std=(0.1, 0.2, 0.3))  # 3 != 12

    def test_rejects_tonnetz_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="tonnetz_mean"):
            SegmentHarmony(tonnetz_mean=(0.1, 0.2))  # 2 != 6


# ===================================================================
# SegmentTimbre validation
# ===================================================================


class TestSegmentTimbreValidation:
    def test_rejects_negative_rms(self) -> None:
        with pytest.raises(ValueError, match="rms_mean"):
            SegmentTimbre(rms_mean=-0.1)
        with pytest.raises(ValueError, match="rms_std"):
            SegmentTimbre(rms_std=-0.1)

    def test_rejects_negative_centroid(self) -> None:
        with pytest.raises(ValueError, match="spectral_centroid_mean"):
            SegmentTimbre(spectral_centroid_mean=-1.0)

    def test_rejects_invalid_zcr(self) -> None:
        with pytest.raises(ValueError, match="zero_crossing_rate_mean"):
            SegmentTimbre(zero_crossing_rate_mean=1.5)
        with pytest.raises(ValueError, match="zero_crossing_rate_mean"):
            SegmentTimbre(zero_crossing_rate_mean=-0.1)

    def test_rejects_mfcc_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="mfcc_mean"):
            SegmentTimbre(mfcc_mean=(0.1, 0.2))  # 2 != 13
        with pytest.raises(ValueError, match="mfcc_std"):
            SegmentTimbre(mfcc_std=(0.1, 0.2))  # 2 != 13

    def test_rejects_spectral_contrast_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="spectral_contrast_mean"):
            SegmentTimbre(spectral_contrast_mean=(0.1, 0.2))  # 2 != 7


# ===================================================================
# SegmentContext validation
# ===================================================================


class TestSegmentContextValidation:
    def test_rejects_negative_index(self) -> None:
        with pytest.raises(ValueError, match="index"):
            SegmentContext(index=SegmentIndex(-1))

    def test_rejects_invalid_label_confidence(self) -> None:
        with pytest.raises(ValueError, match="label_confidence"):
            SegmentContext(index=SegmentIndex(0), label_confidence=1.5)
        with pytest.raises(ValueError, match="label_confidence"):
            SegmentContext(index=SegmentIndex(0), label_confidence=-0.1)

    def test_rejects_invalid_novelty(self) -> None:
        with pytest.raises(ValueError, match="novelty_score"):
            SegmentContext(index=SegmentIndex(0), novelty_score=1.5)
        with pytest.raises(ValueError, match="novelty_score"):
            SegmentContext(index=SegmentIndex(0), novelty_score=-0.1)

    def test_rejects_invalid_self_similarity(self) -> None:
        with pytest.raises(ValueError, match="self_similarity_score"):
            SegmentContext(index=SegmentIndex(0), self_similarity_score=1.5)


# ===================================================================
# StructureDNA validation
# ===================================================================


class TestStructureDNAValidation:
    def test_rejects_empty_segments(self) -> None:
        with pytest.raises(ValueError, match="segments"):
            StructureDNA(
                segments=(),
                total_duration=0.0,
                boundary_confidences=(),
            )

    def test_rejects_non_contiguous_segments(self) -> None:
        sid = _sid()
        segs = _make_segments_gap(sid)  # gap between segment[0] and segment[1]
        with pytest.raises(ValueError, match="contiguous"):
            StructureDNA(
                segments=segs,
                total_duration=120.0,
                boundary_confidences=_make_boundaries(3),
            )

    def test_rejects_mismatched_duration(self) -> None:
        sid = _sid()
        segs = _make_segments(sid)
        boundaries = _make_boundaries(3)
        # total_duration should be 120.0, passing 100.0
        with pytest.raises(ValueError, match="total_duration"):
            StructureDNA(
                segments=segs,
                total_duration=100.0,
                boundary_confidences=boundaries,
            )

    def test_rejects_wrong_boundary_count(self) -> None:
        sid = _sid()
        segs = _make_segments(sid)  # 4 segments
        # Need 3 boundaries, provide 2
        boundaries = _make_boundaries(2)
        with pytest.raises(ValueError, match="boundary_confidences"):
            StructureDNA(
                segments=segs,
                total_duration=120.0,
                boundary_confidences=boundaries,
            )

    def test_rejects_mismatched_segment_indices(self) -> None:
        sid = _sid()
        # Create segments but force wrong index on segment[1]
        seg0 = _make_segment(sid, index=0, start=0.0, end=30.0)
        seg1 = _make_segment(sid, index=2, start=30.0, end=60.0)  # Wrong index
        seg2 = _make_segment(sid, index=2, start=60.0, end=90.0)
        seg3 = _make_segment(sid, index=3, start=90.0, end=120.0)
        boundaries = _make_boundaries(3)
        with pytest.raises(ValueError, match="context.index"):
            StructureDNA(
                segments=(seg0, seg1, seg2, seg3),
                total_duration=120.0,
                boundary_confidences=boundaries,
            )


# ===================================================================
# SongMetadata validation
# ===================================================================


class TestSongMetadataValidation:
    def test_rejects_empty_song_id(self) -> None:
        with pytest.raises(ValueError, match="song_id"):
            SongMetadata(
                song_id=SongID(""),
                filename="test.mp3",
                duration=120.0,
                sample_rate=44100,
            )

    def test_rejects_empty_filename(self) -> None:
        with pytest.raises(ValueError, match="filename"):
            SongMetadata(
                song_id=_sid(),
                filename="  ",
                duration=120.0,
                sample_rate=44100,
            )

    def test_rejects_non_positive_duration(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            SongMetadata(
                song_id=_sid(),
                filename="test.mp3",
                duration=0.0,
                sample_rate=44100,
            )

    def test_rejects_non_positive_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            SongMetadata(
                song_id=_sid(),
                filename="test.mp3",
                duration=120.0,
                sample_rate=0,
            )

    def test_rejects_invalid_channels(self) -> None:
        with pytest.raises(ValueError, match="channels"):
            SongMetadata(
                song_id=_sid(),
                filename="test.mp3",
                duration=120.0,
                sample_rate=44100,
                channels=3,
            )


# ===================================================================
# SongSummary validation
# ===================================================================


class TestSongSummaryValidation:
    def test_rejects_negative_tempo(self) -> None:
        with pytest.raises(ValueError, match="tempo_estimate"):
            SongSummary(tempo_estimate=-120.0)

    def test_rejects_invalid_tempo_confidence(self) -> None:
        with pytest.raises(ValueError, match="tempo_confidence"):
            SongSummary(tempo_estimate=120.0, tempo_confidence=1.5)

    def test_rejects_invalid_key_confidence(self) -> None:
        with pytest.raises(ValueError, match="key_confidence"):
            SongSummary(key_estimate="C major", key_confidence=1.5)

    def test_rejects_negative_rms(self) -> None:
        with pytest.raises(ValueError, match="rms_energy_mean"):
            SongSummary(rms_energy_mean=-0.1)

    def test_rejects_mfcc_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="mfcc_mean"):
            SongSummary(mfcc_mean=(0.1, 0.2))  # 2 != 13

    def test_rejects_chroma_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="chroma_mean"):
            SongSummary(chroma_mean=(0.1, 0.2))  # 2 != 12

    def test_rejects_invalid_key_alternatives(self) -> None:
        with pytest.raises(ValueError, match="key_alternatives"):
            SongSummary(
                key_alternatives=(("C major",),)  # Wrong tuple length
            )
        with pytest.raises(ValueError, match="key_alternatives"):
            SongSummary(
                key_alternatives=(("C major", 0.8, "extra"),)  # Wrong tuple length
            )
        with pytest.raises(ValueError, match="key_alternatives"):
            SongSummary(
                key_alternatives=(("", 0.8),)  # Empty key name
            )
        with pytest.raises(ValueError, match="key_alternatives"):
            SongSummary(
                key_alternatives=(("C major", 1.5),)  # Confidence > 1
            )


# ===================================================================
# SongDNA validation (cross-field invariants)
# ===================================================================


class TestSongDNAValidation:
    def test_rejects_duration_mismatch(self) -> None:
        """SongDNA should reject when structure duration != metadata duration."""
        sid = _sid()
        metadata = SongMetadata(
            song_id=sid,
            filename="test.mp3",
            duration=120.0,
            sample_rate=44100,
        )
        summary = SongSummary()
        # Structure with total_duration=100.0, not 120.0
        segs = _make_segments(sid, end_time=25.0)  # Each segment 25s, total 100s
        boundaries = tuple(
            BoundaryConfidence(time=t * 25.0, confidence=0.8, method="test")
            for t in range(1, 4)
        )
        structure = StructureDNA(
            segments=segs,
            total_duration=100.0,
            boundary_confidences=boundaries,
        )
        with pytest.raises(ValueError, match="total_duration"):
            SongDNA(metadata=metadata, summary=summary, structure=structure)

    def test_rejects_song_id_mismatch(self) -> None:
        """SongDNA should reject when a segment has a different song_id."""
        sid = _sid()
        other_sid = _sid()
        metadata = SongMetadata(
            song_id=sid,
            filename="test.mp3",
            duration=120.0,
            sample_rate=44100,
        )
        summary = SongSummary()
        # One segment has wrong song_id
        seg0 = _make_segment(sid, index=0, start=0.0, end=30.0)
        seg1 = _make_segment(other_sid, index=1, start=30.0, end=60.0)  # Wrong id
        seg2 = _make_segment(sid, index=2, start=60.0, end=90.0)
        seg3 = _make_segment(sid, index=3, start=90.0, end=120.0)
        boundaries = _make_boundaries(3)
        structure = StructureDNA(
            segments=(seg0, seg1, seg2, seg3),
            total_duration=120.0,
            boundary_confidences=boundaries,
        )
        with pytest.raises(ValueError, match="song_id"):
            SongDNA(metadata=metadata, summary=summary, structure=structure)


# ===================================================================
# BoundaryConfidence validation
# ===================================================================


class TestBoundaryConfidenceValidation:
    def test_rejects_negative_time(self) -> None:
        with pytest.raises(ValueError, match="time"):
            BoundaryConfidence(time=-1.0, confidence=0.8)

    def test_rejects_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            BoundaryConfidence(time=30.0, confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            BoundaryConfidence(time=30.0, confidence=-0.1)

    def test_rejects_empty_method(self) -> None:
        with pytest.raises(ValueError, match="method"):
            BoundaryConfidence(time=30.0, confidence=0.8, method="")


# ===================================================================
# AlternativeSegmentation validation
# ===================================================================


class TestAlternativeSegmentationValidation:
    def test_rejects_empty_label(self) -> None:
        sid = _sid()
        seg = _make_segment(sid)
        with pytest.raises(ValueError, match="label"):
            AlternativeSegmentation(
                label="",
                segments=(seg,),
                confidence=0.8,
            )

    def test_rejects_empty_segments(self) -> None:
        with pytest.raises(ValueError, match="segments"):
            AlternativeSegmentation(
                label="test",
                segments=(),
                confidence=0.8,
            )

    def test_rejects_invalid_confidence(self) -> None:
        sid = _sid()
        seg = _make_segment(sid)
        with pytest.raises(ValueError, match="confidence"):
            AlternativeSegmentation(
                label="test",
                segments=(seg,),
                confidence=1.5,
            )

    def test_rejects_non_contiguous_segments(self) -> None:
        sid = _sid()
        segs = _make_segments_gap(sid)  # gap between segments
        with pytest.raises(ValueError, match="contiguous"):
            AlternativeSegmentation(
                label="test",
                segments=segs,
                confidence=0.8,
            )


# ===================================================================
# Fixture helpers
# ===================================================================


def _make_segment(
    sid: SongID,
    index: int = 0,
    start: float = 0.0,
    end: float = 30.0,
) -> SegmentDNA:
    timing = SegmentTiming(
        start_time=start,
        end_time=end,
        start_beat=0,
        end_beat=64,
        bar_count=16,
    )
    rhythm = SegmentRhythm(tempo=120.0, tempo_confidence=0.9, onset_strength_mean=0.5)
    harmony = SegmentHarmony(
        key="C major",
        key_confidence=0.8,
        chroma_mean=tuple(0.1 * (i % 3) for i in range(12)),
        chroma_std=tuple(0.05 for _ in range(12)),
        tonnetz_mean=tuple(0.0 for _ in range(6)),
    )
    timbre = SegmentTimbre(
        rms_mean=0.5,
        rms_std=0.1,
        spectral_centroid_mean=2000.0,
        mfcc_mean=tuple(float(i) for i in range(13)),
    )
    dsp = SegmentDSP(rhythm=rhythm, harmony=harmony, timbre=timbre)
    context = SegmentContext(
        index=SegmentIndex(index),
        label=LabelType.VERSE,
        label_confidence=0.85,
        repetition_role=RepetitionRole.ORIGINAL,
        repetition_group=f"verse_{index + 1}",
    )
    return SegmentDNA(
        song_id=sid,
        timing=timing,
        dsp=dsp,
        context=context,
    )


def _make_segments(sid: SongID, end_time: float = 30.0) -> tuple[SegmentDNA, ...]:
    """Create 4 contiguous segments, each lasting `end_time` seconds."""
    return (
        _make_segment(sid, index=0, start=0.0, end=end_time),
        _make_segment(sid, index=1, start=end_time, end=2 * end_time),
        _make_segment(sid, index=2, start=2 * end_time, end=3 * end_time),
        _make_segment(sid, index=3, start=3 * end_time, end=4 * end_time),
    )


def _make_segments_gap(sid: SongID) -> tuple[SegmentDNA, ...]:
    """Create 4 segments with a gap between the first two."""
    return (
        _make_segment(sid, index=0, start=0.0, end=30.0),
        _make_segment(sid, index=1, start=35.0, end=65.0),  # Gap at 30-35
        _make_segment(sid, index=2, start=65.0, end=95.0),
        _make_segment(sid, index=3, start=95.0, end=125.0),
    )


def _make_boundaries(count: int) -> tuple[BoundaryConfidence, ...]:
    return tuple(
        BoundaryConfidence(time=float(i + 1) * 30.0, confidence=0.8, method="test")
        for i in range(count)
    )