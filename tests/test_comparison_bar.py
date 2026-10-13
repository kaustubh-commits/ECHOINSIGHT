"""The comparison bar shows the engine's raw measurements, honestly labelled."""

import dataclasses

import pytest
from PySide6.QtWidgets import QApplication

from src.analysis.comparison import compare, format_report
from src.core.package import EnrichedSongPackage, SongPackage
from src.ui.widgets.comparison_bar import ComparisonBar, format_comparison
from tests.test_comparison_report import _make_song


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _fallback(song):
    summary = dataclasses.replace(song.summary, tempo_estimate=120.0, tempo_is_fallback=True)
    return dataclasses.replace(song, summary=summary)


def _pkg(song) -> EnrichedSongPackage:
    return EnrichedSongPackage(package=SongPackage(song_dna=song))


# --- Engine --------------------------------------------------------------------


def test_engine_reports_no_tempo_difference_for_fallback_tempo():
    report = compare(_fallback(_make_song(tempo=120.0)), _make_song(tempo=128.0))
    assert report.tempo_difference is None
    assert "n/a (tempo not detected)" in format_report(report)


def test_engine_tempo_difference_when_both_detected():
    report = compare(_make_song(tempo=120.0), _make_song(tempo=128.0))
    assert report.tempo_difference == pytest.approx(8.0)


# --- Card texts ------------------------------------------------------------------


def test_format_comparison_texts():
    a = _make_song(tempo=120.0, key="C major", rms=0.5)
    b = _make_song(tempo=126.0, key="A minor", rms=0.25)
    tempo, harmony, energy = format_comparison(compare(a, b), "C major", "A minor")

    assert tempo == "6.0 BPM apart"
    assert harmony == "C major vs A minor · Camelot distance 0"
    assert energy.startswith("+0.50")
    assert "%" not in energy and "match" not in energy.lower()


def test_format_comparison_unknown_key_and_fallback_tempo():
    a = _fallback(_make_song(key=None))
    b = _make_song(key="D minor")
    tempo, harmony, _ = format_comparison(compare(a, b), None, "D minor")
    assert tempo == "n/a (tempo not detected)"
    assert harmony == "Unknown vs D minor · Camelot n/a"


# --- Widget ----------------------------------------------------------------------


def test_bar_displays_engine_output_and_no_match_wording(qapp):
    bar = ComparisonBar()
    a = _make_song(tempo=120.0, key="C major", rms=0.5)
    b = _make_song(tempo=126.0, key="G major", rms=1.0)
    bar.update_comparison(_pkg(a), _pkg(b))

    expected = format_comparison(compare(a, b), "C major", "G major")
    assert (bar.lbl_tempo.text(), bar.lbl_harmony.text(), bar.lbl_energy.text()) == expected
    assert "compatib" not in bar.title().lower()
    assert "match" not in bar.lbl_energy.text().lower()
