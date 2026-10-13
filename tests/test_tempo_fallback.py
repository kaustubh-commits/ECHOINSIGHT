"""A fallback tempo must be flagged, and no confidence may be invented."""

import copy
import json

import librosa
import numpy as np
import pytest

from src.analysis import extractor, segment_dsp
from src.analysis.package_store import PackageStore
from src.core.song_dna import SongDNA, SongSummary, migrate_song_dna_dict
from src.core.structure import SegmentRhythm
from src.ui.widgets.deck_view import format_tempo

SR = 22050


def _click_track(bpm: float = 120.0, seconds: float = 8.0) -> np.ndarray:
    times = np.arange(0.0, seconds, 60.0 / bpm)
    return librosa.clicks(times=times, sr=SR, length=int(SR * seconds))


# --- Domain types --------------------------------------------------------------


def test_tempo_confidence_defaults_to_unknown():
    assert SongSummary().tempo_confidence is None
    assert SegmentRhythm(tempo=100.0, tempo_confidence=None, onset_strength_mean=0.1).tempo_is_fallback is False


def test_confidence_is_still_range_checked_when_present():
    with pytest.raises(ValueError, match="tempo_confidence"):
        SongSummary(tempo_estimate=100.0, tempo_confidence=1.5)


# --- Extraction ----------------------------------------------------------------


def test_silence_yields_flagged_fallback_tempo():
    tempo, is_fallback, beats = extractor._detect_beats(np.zeros(SR * 4, dtype=np.float32), SR)
    assert is_fallback is True
    assert tempo == extractor.FALLBACK_TEMPO
    assert beats == ()


def test_click_track_yields_detected_tempo():
    tempo, is_fallback, beats = extractor._detect_beats(_click_track(), SR)
    assert is_fallback is False
    assert len(beats) > 0
    assert tempo > 0


def test_silent_segment_rhythm_is_fallback_without_confidence():
    y = np.zeros(SR * 4, dtype=np.float32)
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=segment_dsp.HOP_LENGTH)
    rhythm = segment_dsp._segment_rhythm(y, SR, 0.0, 4.0, onset_env, np.array([]))
    assert rhythm.tempo_is_fallback is True
    assert rhythm.tempo_confidence is None


def test_extracted_song_has_no_invented_confidence(tmp_path):
    import soundfile as sf

    path = tmp_path / "clicks.wav"
    sf.write(str(path), _click_track(seconds=12.0), SR)
    dna = extractor.extract_song_dna(str(path))

    assert dna.summary.tempo_is_fallback is False
    assert dna.summary.tempo_confidence is None
    for seg in dna.structure.segments:
        assert seg.dsp.rhythm.tempo_confidence is None


# --- Migration of files written before schema 2.2.0 ---------------------------


def _old_style(dna: SongDNA, confidence: float) -> dict:
    """Serialise *dna* the way pre-2.2.0 code did (no fallback flag)."""
    data = json.loads(dna.to_json())
    data["schema_version"] = "2.1.0"
    for rhythm_like in [data["summary"]] + [s["dsp"]["rhythm"] for s in data["structure"]["segments"]]:
        del rhythm_like["tempo_is_fallback"]
        rhythm_like["tempo_confidence"] = confidence
    return data


@pytest.fixture
def sample_dna(tmp_path) -> SongDNA:
    import soundfile as sf

    path = tmp_path / "clicks.wav"
    sf.write(str(path), _click_track(seconds=12.0), SR)
    return extractor.extract_song_dna(str(path))


@pytest.mark.parametrize("old_confidence, expect_fallback", [(0.0, True), (1.0, False), (0.8, False)])
def test_old_placeholder_confidences_migrate(sample_dna, old_confidence, expect_fallback):
    data = _old_style(sample_dna, old_confidence)
    restored = SongDNA.from_json(json.dumps(data))

    assert restored.summary.tempo_is_fallback is expect_fallback
    assert restored.summary.tempo_confidence is None
    for seg in restored.structure.segments:
        assert seg.dsp.rhythm.tempo_is_fallback is expect_fallback
        assert seg.dsp.rhythm.tempo_confidence is None


def test_current_data_is_not_changed_by_migration(sample_dna):
    data = json.loads(sample_dna.to_json())
    before = copy.deepcopy(data)
    assert migrate_song_dna_dict(data) == before


def test_package_store_migrates_old_files(sample_dna, tmp_path):
    from src.core.package import SongPackage

    pkg = PackageStore.to_dict(SongPackage(song_dna=sample_dna))
    pkg["song_dna"] = _old_style(sample_dna, 0.0)
    path = tmp_path / "old_pkg.json"
    path.write_text(json.dumps(pkg))

    loaded = PackageStore.load(path)
    assert loaded.song_dna.summary.tempo_is_fallback is True
    assert loaded.song_dna.summary.tempo_confidence is None


# --- Display -------------------------------------------------------------------


def test_format_tempo_marks_fallback():
    assert format_tempo(SongSummary(tempo_estimate=120.0, tempo_is_fallback=True)) == (
        "120 BPM (fallback, not detected)"
    )
    assert format_tempo(SongSummary(tempo_estimate=123.46)) == "123.5 BPM"
