"""
test_timeline.py — Unit tests for the Timeline domain types.

Covers:
- Timeline.build: downbeat derivation, pickup bar, tail bar
- Bar/Timeline validation invariants
- Timeline.bar_at binary search lookup
- JSON round-trip consistency within SongDNA

Layer rules
-----------
- These tests import **only** from ``src.core``.
"""

from __future__ import annotations

import dataclasses

import pytest

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
from src.core.song_dna import SongDNA, SongMetadata, SongSummary
from src.core.timeline import Bar, Timeline
from src.core.identifiers import SegmentIndex, generate_song_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(song_id: str, index: int, start: float, end: float) -> SegmentDNA:
    timing = SegmentTiming(start_time=start, end_time=end)
    rhythm = SegmentRhythm(tempo=120.0, tempo_confidence=0.9, onset_strength_mean=0.5)
    harmony = SegmentHarmony(
        key="C major",
        key_confidence=0.8,
        chroma_mean=tuple(0.1 for _ in range(12)),
        chroma_std=tuple(0.05 for _ in range(12)),
        tonnetz_mean=tuple(0.0 for _ in range(6)),
    )
    timbre = SegmentTimbre(
        rms_mean=0.5,
        rms_std=0.1,
        spectral_centroid_mean=2000.0,
        spectral_centroid_std=500.0,
        spectral_bandwidth_mean=3000.0,
        mfcc_mean=tuple(float(k) for k in range(13)),
        mfcc_std=tuple(float(k % 3) for k in range(13)),
        zero_crossing_rate_mean=0.1,
        spectral_rolloff_mean=4000.0,
        spectral_contrast_mean=tuple(float(k) for k in range(7)),
        spectral_flux_mean=0.2,
        spectral_flux_std=0.05,
        hpss_envelope_ratio=0.4,
    )
    context = SegmentContext(index=SegmentIndex(index))
    return SegmentDNA(
        song_id=song_id,
        timing=timing,
        dsp=SegmentDSP(rhythm=rhythm, harmony=harmony, timbre=timbre),
        context=context,
    )


def _make_song_dna_with_timeline(duration: float = 120.0) -> SongDNA:
    sid = generate_song_id()
    # Two 60s segments for a simple structure.
    segments = (
        _make_segment(sid, 0, 0.0, 60.0),
        _make_segment(sid, 1, 60.0, 120.0),
    )
    from src.core.structure import BoundaryConfidence

    structure = StructureDNA(
        segments=segments,
        total_duration=duration,
        boundary_confidences=(BoundaryConfidence(time=60.0, confidence=0.9),),
    )
    metadata = SongMetadata(
        song_id=sid,
        filename="test.mp3",
        duration=duration,
        sample_rate=22050,
    )
    summary = SongSummary(tempo_estimate=120.0, tempo_confidence=0.9)
    # Beats every 0.5s (120 BPM in 4/4 → 4 beats per 2s bar).
    beats = tuple(i * 0.5 for i in range(int(duration * 2)))
    timeline = Timeline.build(beats=beats, duration=duration)
    return SongDNA(
        metadata=metadata,
        summary=summary,
        structure=structure,
        timeline=timeline,
    )


# ---------------------------------------------------------------------------
# Timeline.build
# ---------------------------------------------------------------------------


class TestTimelineBuild:
    def test_basic_4_4_grid(self) -> None:
        """Beats every 0.5s → downbeats every 4th beat (every 2.0s)."""
        beats = tuple(i * 0.5 for i in range(41))  # 0.0 .. 20.0
        tl = Timeline.build(beats=beats, duration=20.0)

        assert tl.num_bars == 10
        # 41 beats → 11 downbeats (indices 0,4,8,...,40 → 0,2,...,20);
        # the final downbeat coincides with duration and is absorbed into
        # the last bar boundary.
        assert tl.downbeats == tuple(i * 2.0 for i in range(11))
        assert tl.bars[0] == Bar(number=1, start_time=0.0, end_time=2.0)
        assert tl.bars[1] == Bar(number=2, start_time=2.0, end_time=4.0)
        assert tl.bars[-1] == Bar(number=10, start_time=18.0, end_time=20.0)
        # All beats are present and strictly increasing.
        assert len(tl.beats) == 41
        assert tl.beats[0] == 0.0
        assert tl.beats[-1] == 20.0

    def test_pickup_bar_when_first_downbeat_not_at_zero(self) -> None:
        """When the first beat is not at 0.0, a pickup bar covers [0, first)."""
        beats = (0.75, 1.25, 1.75, 2.25)
        tl = Timeline.build(beats=beats, duration=4.0)

        # Downbeats: every 4th beat → (0.75,)
        assert tl.downbeats == (0.75,)
        # Bars: pickup [0, 0.75) + span [0.75, 4.0) tail
        assert len(tl.bars) == 2
        assert tl.bars[0].start_time == 0.0
        assert tl.bars[0].end_time == 0.75
        assert tl.bars[1].start_time == 0.75
        assert tl.bars[1].end_time == 4.0

    def test_tail_bar_to_duration(self) -> None:
        """The last bar extends to the track duration."""
        beats = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        tl = Timeline.build(beats=beats, duration=10.0)
        # Downbeats at 0.0 and 2.0. Bars: [0,2), [2,10)
        assert tl.num_bars == 2
        assert tl.bars[-1].end_time == 10.0

    def test_last_downbeat_at_duration_no_tail_bar(self) -> None:
        """When the last downbeat coincides with duration, no tail bar exists."""
        beats = tuple(i * 0.5 for i in range(41))  # up to 20.0
        tl = Timeline.build(beats=beats, duration=20.0)
        # Downbeats at 0,2,...,18 → bars [0,2), [2,4), ..., [18,20)
        assert tl.num_bars == 10
        assert tl.bars[-1].end_time == 20.0

    def test_empty_beats_raises(self) -> None:
        with pytest.raises(ValueError, match="beats must be non-empty"):
            Timeline.build(beats=(), duration=10.0)

    def test_nonpositive_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            Timeline.build(beats=(0.0, 0.5), duration=0.0)

    def test_beats_per_bar_validation(self) -> None:
        with pytest.raises(ValueError, match="beats_per_bar must be >= 1"):
            Timeline.build(beats=(0.0, 0.5), duration=10.0, beats_per_bar=0)


# ---------------------------------------------------------------------------
# Validation invariants
# ---------------------------------------------------------------------------


class TestTimelineValidation:
    def test_non_increasing_beats_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            Timeline(
                beats=(0.0, 0.5, 0.5, 1.0),
                downbeats=(0.0,),
                bars=(Bar(1, 0.0, 1.0),),
            )

    def test_negative_beat_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            Timeline(
                beats=(-0.5, 0.5),
                downbeats=(0.5,),
                bars=(Bar(1, 0.0, 0.5), Bar(2, 0.5, 1.0)),
            )

    def test_downbeat_not_a_beat_raises(self) -> None:
        with pytest.raises(ValueError, match="not present in beats"):
            Timeline(
                beats=(0.0, 0.5, 1.0, 1.5),
                downbeats=(0.7,),  # not in beats
                bars=(Bar(1, 0.0, 0.7), Bar(2, 0.7, 1.5)),
            )

    def test_bar_number_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match=r"bars\[0\]\.number"):
            Timeline(
                beats=(0.0, 0.5, 1.0, 1.5),
                downbeats=(0.0,),
                bars=(Bar(number=2, start_time=0.0, end_time=1.5),),
            )

    def test_first_bar_not_at_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="start_time must be 0.0"):
            Timeline(
                beats=(0.5, 1.0, 1.5, 2.0),
                downbeats=(1.0,),
                bars=(Bar(1, 0.5, 1.0), Bar(2, 1.0, 2.0)),
            )

    def test_bar_not_contiguous_raises(self) -> None:
        # The two bars imply boundaries {0.0, 3.0} but the downbeat grid
        # only produces one expected span → the count check raises first.
        with pytest.raises(ValueError, match="Expected 1 bars"):
            Timeline(
                beats=(0.0, 0.5, 1.0, 1.5, 2.0),
                downbeats=(0.0,),
                bars=(Bar(1, 0.0, 1.0), Bar(2, 2.5, 3.0)),  # gap
            )


# ---------------------------------------------------------------------------
# bar_at lookup
# ---------------------------------------------------------------------------


class TestBarAt:
    def test_bar_at_time(self) -> None:
        beats = tuple(i * 0.5 for i in range(41))
        tl = Timeline.build(beats=beats, duration=20.0)

        assert tl.bar_at(0.0).number == 1
        assert tl.bar_at(1.9).number == 1
        assert tl.bar_at(2.0).number == 2
        assert tl.bar_at(19.9).number == 10

    def test_bar_at_out_of_range_raises(self) -> None:
        beats = tuple(i * 0.5 for i in range(41))
        tl = Timeline.build(beats=beats, duration=20.0)
        with pytest.raises(ValueError, match="outside the timeline"):
            tl.bar_at(20.0)
        with pytest.raises(ValueError, match="outside the timeline"):
            tl.bar_at(-0.1)


# ---------------------------------------------------------------------------
# Integration: SongDNA with timeline
# ---------------------------------------------------------------------------


class TestSongDNAIntegration:
    def test_timeline_round_trip(self) -> None:
        """Timeline must survive JSON serialization losslessly."""
        dna = _make_song_dna_with_timeline()
        dna2 = SongDNA.from_json(dna.to_json())

        assert dna2.timeline is not None
        assert dna2.timeline == dna.timeline
        assert dna2.timeline.num_bars == dna.timeline.num_bars
        assert dna2.timeline.bars[0].number == 1

    def test_timeline_duration_match(self) -> None:
        """SongDNA with a timeline whose last bar doesn't reach duration raises."""
        sid = generate_song_id()
        segment = _make_segment(sid, 0, 0.0, 60.0)
        structure = StructureDNA(segments=(segment,), total_duration=60.0)
        metadata = SongMetadata(
            song_id=sid,
            filename="test.mp3",
            duration=60.0,
            sample_rate=22050,
        )
        summary = SongSummary(tempo_estimate=120.0, tempo_confidence=0.9)
        # Timeline covers only the first 10s — mismatch with 60s duration.
        beats = tuple(i * 0.5 for i in range(21))  # 0..10
        timeline = Timeline.build(beats=beats, duration=10.0)

        with pytest.raises(ValueError, match="does not match"):
            SongDNA(
                metadata=metadata,
                summary=summary,
                structure=structure,
                timeline=timeline,
            )

    def test_timeline_defaults_none(self) -> None:
        """Timeline is optional — absent by default."""
        dna = _make_song_dna_with_timeline()
        # Rebuild without timeline to verify it's optional.
        bare = dataclasses.replace(dna, timeline=None)
        assert bare.timeline is None