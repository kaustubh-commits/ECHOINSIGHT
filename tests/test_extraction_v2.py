"""
test_extraction_v2.py — Persistence, round-trip, and regression tests for
the v2 extraction pipeline (SongDNA v2: StructureDNA + SegmentDNA[]).

Covers:
- ``SongDNA.to_json()`` / ``SongDNA.from_json()`` lossless round trip
- FrameReference + FrameStore NPZ round trip
- v2 extraction regression (requires test audio fixtures; skips if absent)
- Evaluation metrics (boundary precision/recall, segment length stats)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.core.enums import LabelType, RepetitionRole
from src.core.identifiers import SegmentIndex, generate_song_id
from src.core.song_dna import SongDNA, SongMetadata, SongSummary
from src.core.structure import (
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

# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------


def _make_sample_song_dna() -> SongDNA:
    """Build a valid, fully-populated v2 SongDNA for round-trip tests."""
    sid = generate_song_id()

    metadata = SongMetadata(
        song_id=sid,
        filename="test_song.mp3",
        duration=120.0,
        sample_rate=44100,
        channels=2,
        bit_depth=16,
        format="mp3",
    )

    summary = SongSummary(
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

    segments = []
    for i, (start, end) in enumerate([(0.0, 30.0), (30.0, 60.0),
                                      (60.0, 90.0), (90.0, 120.0)]):
        timing = SegmentTiming(
            start_time=start,
            end_time=end,
            start_beat=i * 64,
            end_beat=(i + 1) * 64,
            bar_count=16,
        )
        rhythm = SegmentRhythm(tempo=120.0, tempo_confidence=0.9,
                               onset_strength_mean=0.5)
        harmony = SegmentHarmony(
            key="C major",
            key_confidence=0.8,
            chroma_mean=tuple(0.1 * (i % 3) for _ in range(12)),
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
        dsp = SegmentDSP(rhythm=rhythm, harmony=harmony, timbre=timbre)
        context = SegmentContext(
            index=SegmentIndex(i),
            label=LabelType.VERSE,
            label_confidence=0.85,
            repetition_role=RepetitionRole.ORIGINAL,
            repetition_group=f"verse_{i + 1}",
            novelty_score=0.5,
            self_similarity_score=0.3,
        )
        segments.append(
            SegmentDNA(song_id=sid, timing=timing, dsp=dsp, context=context)
        )

    boundaries = tuple(
        BoundaryConfidence(time=30.0 * (i + 1), confidence=0.9,
                           method="novelty_curve")
        for i in range(3)
    )
    structure = StructureDNA(
        segments=tuple(segments),
        total_duration=120.0,
        boundary_confidences=boundaries,
    )

    return SongDNA(metadata=metadata, summary=summary, structure=structure)


# =====================================================================
# SongDNA JSON round-trip
# =====================================================================


def test_song_dna_roundtrip() -> None:
    """JSON round trip must be lossless (dna == dna2)."""
    dna = _make_sample_song_dna()
    json_str = dna.to_json()
    dna2 = SongDNA.from_json(json_str)
    assert dna == dna2


@pytest.mark.parametrize(
    "schema_version",
    ["2.0.0", "2.1.0"],
)
def test_roundtrip_preserves_schema_version(schema_version: str) -> None:
    dna = dataclasses.replace(_make_sample_song_dna(), schema_version=schema_version)
    assert SongDNA.from_json(dna.to_json()).schema_version == schema_version


def test_roundtrip_preserves_enum_values() -> None:
    dna = _make_sample_song_dna()
    dna2 = SongDNA.from_json(dna.to_json())
    assert dna2.structure.segments[0].context.label is LabelType.VERSE
    assert (
        dna2.structure.segments[0].context.repetition_role
        is RepetitionRole.ORIGINAL
    )


def test_roundtrip_reconstructs_frozen_types() -> None:
    """Nested tuples and enums must be restored, not left as lists/strings."""
    dna = _make_sample_song_dna()
    dna2 = SongDNA.from_json(dna.to_json())

    assert isinstance(dna2.structure.segments, tuple)
    seg = dna2.structure.segments[0]
    assert isinstance(seg.dsp.timbre.mfcc_mean, tuple)
    assert isinstance(seg.dsp.harmony.chroma_mean, tuple)
    assert isinstance(seg.dsp.harmony.tonnetz_mean, tuple)
    assert isinstance(seg.context.index, int)
    assert seg.timing.duration == pytest.approx(30.0)


# =====================================================================
# FrameReference + FrameStore NPZ round-trip
# =====================================================================


def test_framereference_npz_roundtrip(tmp_path: Path) -> None:
    """FrameStore.save → load must preserve array values."""
    from src.analysis.frame_store import FrameStore

    n_frames = 100
    features = {
        "rms": np.random.rand(n_frames).astype(np.float32),
        "centroid": np.random.rand(n_frames).astype(np.float32),
        "bandwidth": np.random.rand(n_frames).astype(np.float32),
        "rolloff": np.random.rand(n_frames).astype(np.float32),
        "zcr": np.random.rand(n_frames).astype(np.float32),
        "mfcc": np.random.rand(13, n_frames).astype(np.float32),
        "chroma": np.random.rand(12, n_frames).astype(np.float32),
        "contrast": np.random.rand(7, n_frames).astype(np.float32),
        "tonnetz": np.random.rand(6, n_frames).astype(np.float32),
    }

    ref = FrameStore.build_frame_reference(
        uri="frames/test.npz",
        features=features,
        hop_length=512,
        sample_rate=22050,
        duration=60.0,
    )
    assert ref.frame_count == n_frames

    npz_path = tmp_path / "frames" / "test.npz"
    FrameStore.save(features, npz_path)
    loaded = FrameStore.load(npz_path)
    np.testing.assert_array_equal(loaded["rms"], features["rms"])
    np.testing.assert_array_equal(loaded["mfcc"], features["mfcc"])
    np.testing.assert_array_equal(loaded["chroma"], features["chroma"])


# =====================================================================
# v2 extraction regression (requires fixtures)
# =====================================================================

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_FILES = sorted(FIXTURES_DIR.glob("*.wav")) + sorted(
    FIXTURES_DIR.glob("*.mp3")
)


@pytest.mark.parametrize(
    "audio_file",
    FIXTURE_FILES,
    ids=[p.stem for p in FIXTURE_FILES],
)
def test_extraction_v2_regression(audio_file: Path) -> None:
    """Run v2 extraction on each fixture.

    Requires ``_BUILD_V2 = True`` in ``src.analysis.extractor`` and at
    least one audio fixture in ``tests/fixtures/``.
    """
    import src.analysis.extractor as extractor_mod

    if not extractor_mod._BUILD_V2:
        pytest.skip("_BUILD_V2 is False — v2 branch not enabled in extractor.py")

    dna = extractor_mod.extract_song_dna(str(audio_file))

    # Segment count > 1 for typical tracks (single-segment fallback allowed
    # only for very short clips)
    assert dna.structure.num_segments >= 1

    # Segment coverage matches audio duration
    covered = sum(
        seg.timing.duration for seg in dna.structure.segments
    )
    assert abs(covered - dna.metadata.duration) < 1e-2

    # Contiguous + full coverage invariants
    assert dna.structure.segments[0].timing.start_time == pytest.approx(0.0)
    assert dna.structure.segments[-1].timing.end_time == pytest.approx(
        dna.metadata.duration
    )
    for i in range(1, len(dna.structure.segments)):
        prev_end = dna.structure.segments[i - 1].timing.end_time
        curr_start = dna.structure.segments[i].timing.start_time
        assert abs(prev_end - curr_start) < 1e-6

    # song_id consistency
    sid = dna.metadata.song_id
    for seg in dna.structure.segments:
        assert seg.song_id == sid

    # Serialization round-trip
    assert SongDNA.from_json(dna.to_json()) == dna

    # Manifest is populated for extract_and_save path (tested separately),
    # but extract_song_dna leaves it None.


def test_extraction_manifest_and_frames(tmp_path: Path) -> None:
    """extract_and_save must populate frames + AnalysisManifest."""
    import src.analysis.extractor as extractor_mod

    if not FIXTURE_FILES:
        pytest.skip("No audio fixtures in tests/fixtures/")

    dna = extractor_mod.extract_and_save(str(FIXTURE_FILES[0]), data_root=tmp_path)
    assert dna.frames is not None
    assert dna.manifest is not None
    assert dna.manifest.sample_rate == 22050
    assert dna.manifest.schema_version == "1.0.0"
    assert len(dna.manifest.config_hash) == 64


# =====================================================================
# Evaluation metrics tests
# =====================================================================


class TestBoundaryMetrics:
    def test_boundary_tolerance_hits_perfect(self) -> None:
        from src.analysis.evaluation.metrics import boundary_tolerance_hits

        tp, n_ref = boundary_tolerance_hits([10.0, 30.0], [10.0, 30.0], tolerance=3.0)
        assert tp == 2
        assert n_ref == 2

    def test_boundary_tolerance_hits_miss(self) -> None:
        from src.analysis.evaluation.metrics import boundary_tolerance_hits

        tp, n_ref = boundary_tolerance_hits([10.0], [30.0], tolerance=3.0)
        assert tp == 0
        assert n_ref == 1

    def test_boundary_precision_recall(self) -> None:
        from src.analysis.evaluation.metrics import boundary_precision_recall

        # 35.0 is outside the 3.0s tolerance of 30.0, so only the 10.0
        # boundary matches.  Both precision and recall are 1/2.
        p, r, f1 = boundary_precision_recall([10.0, 35.0], [10.0, 30.0], tolerance=3.0)
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)

    def test_boundary_distance_exact(self) -> None:
        from src.analysis.evaluation.metrics import boundary_distance

        assert boundary_distance([10.0, 30.0], [10.0, 30.0]) == pytest.approx(0.0)

    def test_boundary_distance_offset(self) -> None:
        from src.analysis.evaluation.metrics import boundary_distance

        dist = boundary_distance([11.0, 31.0], [10.0, 30.0])
        assert dist == pytest.approx(1.0)


class TestSegmentLengthMetrics:
    def test_segment_durations(self) -> None:
        from src.analysis.evaluation.metrics import segment_durations

        d = segment_durations([(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)])
        np.testing.assert_array_equal(d, np.array([30.0, 30.0, 30.0]))

    def test_segment_length_stats(self) -> None:
        from src.analysis.evaluation.metrics import segment_length_stats

        stats = segment_length_stats([(0.0, 10.0), (10.0, 30.0), (30.0, 60.0)])
        assert stats["n_segments"] == 3
        assert stats["mean"] == pytest.approx(20.0)
        assert stats["min"] == pytest.approx(10.0)
        assert stats["max"] == pytest.approx(30.0)

    def test_coverage_error_zero(self) -> None:
        from src.analysis.evaluation.metrics import coverage_error

        assert (
            coverage_error([(0.0, 30.0), (30.0, 60.0)], 60.0)
            == pytest.approx(0.0)
        )

    def test_coverage_error_gap(self) -> None:
        from src.analysis.evaluation.metrics import coverage_error

        assert (
            coverage_error([(0.0, 30.0), (35.0, 60.0)], 60.0)
            == pytest.approx(5.0)
        )