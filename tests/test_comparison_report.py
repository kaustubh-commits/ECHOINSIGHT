"""
test_comparison_report.py — Unit tests for the raw-metric comparison report.

Covers:
- Camelot wheel key mapping and circular distance
- tempo / duration / energy difference computations
- Full ComparisonReport validation
- Live two-track smoke test (skipped when audio fixtures are absent)
"""

from __future__ import annotations

import pytest

from src.analysis.comparison import (
    ComparisonReport,
    _camelot_distance,
    _camelot_number,
    compare,
)
from src.core.identifiers import SegmentIndex, generate_song_id
from src.core.song_dna import SongDNA, SongMetadata, SongSummary
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(song_id: str) -> SegmentDNA:
    timing = SegmentTiming(start_time=0.0, end_time=60.0)
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
    return SegmentDNA(
        song_id=song_id,
        timing=timing,
        dsp=SegmentDSP(rhythm=rhythm, harmony=harmony, timbre=timbre),
        context=SegmentContext(index=SegmentIndex(0)),
    )


def _make_song(
    tempo: float = 120.0,
    key: str | None = "C major",
    duration: float = 60.0,
    rms: float = 0.5,
    filename: str = "song.mp3",
) -> SongDNA:
    sid = generate_song_id()
    structure = StructureDNA(
        segments=(_make_segment(sid),),
        total_duration=duration,
        boundary_confidences=(),
    )
    metadata = SongMetadata(
        song_id=sid,
        filename=filename,
        duration=duration,
        sample_rate=22050,
    )
    summary = SongSummary(
        tempo_estimate=tempo,
        tempo_confidence=0.9,
        key_estimate=key,
        key_confidence=0.8,
        rms_energy_mean=rms,
    )
    return SongDNA(metadata=metadata, summary=summary, structure=structure)


# ---------------------------------------------------------------------------
# Camelot wheel
# ---------------------------------------------------------------------------


class TestCamelot:
    def test_major_and_minor_keys(self) -> None:
        assert _camelot_number("C major") == 8
        assert _camelot_number("D minor") == 7
        assert _camelot_number("G major") == 9
        assert _camelot_number("B major") == 1
        assert _camelot_number("A minor") == 8

    def test_unknown_key_returns_none(self) -> None:
        assert _camelot_number(None) is None
        assert _camelot_number("") is None

    def test_same_key_distance_zero(self) -> None:
        assert _camelot_distance(8, 8) == 0

    def test_adjacent_wheel_numbers(self) -> None:
        assert _camelot_distance(8, 9) == 1
        assert _camelot_distance(8, 7) == 1

    def test_unknown_distance(self) -> None:
        assert _camelot_distance(None, 8) == 12
        assert _camelot_distance(8, None) == 12
