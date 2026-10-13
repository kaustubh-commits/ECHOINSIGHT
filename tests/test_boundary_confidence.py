"""Boundary confidence is not computed, so it must be stored as unknown."""

import dataclasses
import json

import pytest

from src.analysis import extractor
from src.analysis.package_store import PackageStore
from src.core.identifiers import SegmentIndex, generate_song_id
from src.core.package import SongPackage
from src.core.song_dna import SongDNA, SongSummary
from src.core.structure import BoundaryConfidence, SegmentContext, SegmentTiming
from tests.test_comparison_report import _make_segment


def _two_segment_dna() -> SongDNA:
    """A SongDNA assembled by the real extractor helper, with one boundary."""
    sid = generate_song_id()
    base = _make_segment(sid)
    first = dataclasses.replace(base, timing=SegmentTiming(0.0, 30.0))
    second = dataclasses.replace(
        base,
        timing=SegmentTiming(30.0, 60.0),
        context=SegmentContext(index=SegmentIndex(1)),
    )
    return extractor._assemble_song_dna_v2(
        song_id=sid,
        filename="two_segments.wav",
        duration=60.0,
        sr=22050,
        segments=[first, second],
        summary=SongSummary(tempo_estimate=120.0),
    )


def test_confidence_defaults_to_unknown_and_is_range_checked():
    assert BoundaryConfidence(time=1.0).confidence is None
    with pytest.raises(ValueError, match="confidence"):
        BoundaryConfidence(time=1.0, confidence=1.5)


def test_extracted_boundaries_store_no_confidence():
    dna = _two_segment_dna()
    (boundary,) = dna.structure.boundary_confidences
    assert boundary.time == 30.0
    assert boundary.confidence is None


def _with_old_placeholder(dna: SongDNA) -> dict:
    data = json.loads(dna.to_json())
    for b in data["structure"]["boundary_confidences"]:
        b["confidence"] = 0.7
        b["method"] = "agglomerative"
    return data


def test_saved_song_dna_with_placeholder_07_loads_as_unknown():
    data = _with_old_placeholder(_two_segment_dna())
    restored = SongDNA.from_json(json.dumps(data))
    (boundary,) = restored.structure.boundary_confidences
    assert boundary.confidence is None
    assert boundary.method == "agglomerative"
    assert boundary.time == 30.0


def test_saved_package_with_placeholder_07_loads_as_unknown(tmp_path):
    dna = _two_segment_dna()
    pkg = PackageStore.to_dict(SongPackage(song_dna=dna))
    pkg["song_dna"] = _with_old_placeholder(dna)
    path = tmp_path / "old_package.json"
    path.write_text(json.dumps(pkg))

    loaded = PackageStore.load(path)
    assert loaded.song_dna.structure.boundary_confidences[0].confidence is None


def test_other_boundary_confidences_are_kept():
    """Only the exact old placeholder pair is cleared."""
    data = json.loads(_two_segment_dna().to_json())
    data["structure"]["boundary_confidences"][0].update(confidence=0.7, method="novelty_curve")
    restored = SongDNA.from_json(json.dumps(data))
    assert restored.structure.boundary_confidences[0].confidence == 0.7
