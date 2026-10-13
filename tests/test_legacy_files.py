"""Files written by the code before PR #2 must still load, honestly.

The fixtures were produced by the pre-PR code itself (see
tests/fixtures/legacy_pre_pr2/README.md), so these tests check real old
output, not hand-edited dictionaries.
"""

import json
from pathlib import Path

import pytest

from src.analysis.package_store import PackageStore
from src.core.song_dna import SongDNA

LEGACY = Path(__file__).resolve().parent / "fixtures" / "legacy_pre_pr2"
PACKAGES = ["package_offline_stub.json", "package_mock_stub.json", "package_realid_stub.json"]


def _raw(name: str) -> dict:
    return json.loads((LEGACY / name).read_text())


def test_fixtures_really_are_old_format():
    """Guard: the files contain the old placeholders this suite is about."""
    dna = _raw("song_dna.json")
    assert dna["schema_version"] == "2.1.0"
    assert "tempo_is_fallback" not in dna["summary"]
    assert {(b["confidence"], b["method"]) for b in dna["structure"]["boundary_confidences"]} == {
        (0.7, "agglomerative")
    }
    for name in PACKAGES:
        pkg = _raw(name)
        assert pkg["schema_version"] == "2.1.0"
        assert "track_info" not in pkg
        assert pkg["spotify_metadata"] is not None


def _assert_migrated(dna: SongDNA) -> None:
    assert dna.summary.tempo_confidence is None
    assert dna.summary.tempo_is_fallback is False  # beats were detected in the clip
    assert dna.structure.num_segments >= 2
    assert all(b.confidence is None for b in dna.structure.boundary_confidences)
    for seg in dna.structure.segments:
        assert seg.dsp.rhythm.tempo_confidence is None
    assert dna.timeline is not None
    assert dna.timeline.downbeats_assumed is True


def test_old_song_dna_json_loads():
    _assert_migrated(SongDNA.from_json((LEGACY / "song_dna.json").read_text()))


@pytest.mark.parametrize("name", PACKAGES)
def test_old_packages_load_and_drop_fake_spotify(name):
    loaded = PackageStore.load(LEGACY / name)

    _assert_migrated(loaded.song_dna)
    assert loaded.spotify is None
    assert not loaded.is_enriched
    assert loaded.track_info is None


@pytest.mark.parametrize("name", PACKAGES)
def test_old_packages_resave_in_current_format(name, tmp_path):
    """Loading then saving upgrades the file and it reloads identically."""
    loaded = PackageStore.load(LEGACY / name)
    out = tmp_path / name
    PackageStore.save(loaded, out)

    resaved = json.loads(out.read_text())
    assert resaved["schema_version"] == "2.2.0"
    assert resaved["spotify_metadata"] is None
    assert PackageStore.load(out).song_dna == loaded.song_dna
