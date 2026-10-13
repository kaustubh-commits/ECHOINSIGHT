"""Downbeats and bars are assumed (every 4th beat), and the data and UI say so."""

import dataclasses
import json

import librosa
import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from src.analysis.extractor import extract_song_dna
from src.core.package import EnrichedSongPackage, SongPackage
from src.core.song_dna import SongDNA
from src.core.timeline import Timeline
from src.ui.widgets.timeline_view import TimelineView

SR = 22050


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def click_dna(tmp_path_factory) -> SongDNA:
    path = tmp_path_factory.mktemp("audio") / "clicks.wav"
    times = np.arange(0.0, 12.0, 0.5)
    sf.write(str(path), librosa.clicks(times=times, sr=SR, length=SR * 12), SR)
    return extract_song_dna(str(path))


def test_built_timeline_records_the_assumption():
    timeline = Timeline.build(beats=[0.5, 1.0, 1.5, 2.0, 2.5], duration=3.0)
    assert timeline.downbeats_assumed is True
    assert timeline.beats_per_bar == 4
    assert timeline.downbeats == (0.5, 2.5)


def test_extracted_timeline_is_marked_assumed(click_dna):
    assert click_dna.timeline is not None
    assert click_dna.timeline.downbeats_assumed is True


def test_old_timeline_without_flag_loads_as_assumed(click_dna):
    data = json.loads(click_dna.to_json())
    del data["timeline"]["downbeats_assumed"]
    del data["timeline"]["beats_per_bar"]
    restored = SongDNA.from_json(json.dumps(data))
    assert restored.timeline.downbeats_assumed is True
    assert restored.timeline.beats_per_bar == 4


def test_invalid_beats_per_bar_rejected():
    with pytest.raises(ValueError, match="beats_per_bar"):
        Timeline(beats=(0.0,), downbeats=(0.0,), bars=(), beats_per_bar=0)


def test_grid_caption_states_what_the_grid_is(click_dna):
    assert TimelineView.grid_caption(click_dna.timeline) == (
        "bars assumed 4/4 from beats; downbeats not detected"
    )
    assert TimelineView.grid_caption(None) == "no beat grid (no beats detected)"
    detected = dataclasses.replace(click_dna.timeline, downbeats_assumed=False)
    assert TimelineView.grid_caption(detected) == ""


@pytest.mark.parametrize("with_timeline", [True, False])
def test_timeline_view_paints_with_and_without_grid(qapp, click_dna, with_timeline):
    dna = click_dna if with_timeline else dataclasses.replace(click_dna, timeline=None)
    view = TimelineView()
    view.resize(600, 150)
    view.set_package(EnrichedSongPackage(package=SongPackage(song_dna=dna)))
    assert not view.grab().isNull()
