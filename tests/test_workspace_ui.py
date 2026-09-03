"""
test_workspace_ui.py — Verification of Phase 3 PySide6 UI components initialization.
"""

import pytest
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.widgets.deck_view import DeckView
from src.ui.widgets.stem_mixer import StemMixer
from src.ui.widgets.timeline_view import TimelineView


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_timeline_view_instantiation(qapp):
    timeline = TimelineView()
    assert timeline.minimumHeight() == 150


def test_stem_mixer_initialization(qapp):
    mixer = StemMixer()
    assert len(mixer.channels) == 4
    assert "vocals" in mixer.channels
    assert "drums" in mixer.channels


def test_deck_view_instantiation(qapp):
    deck = DeckView("A")
    assert deck.deck_label == "A"
    assert deck.btn_play is not None


def test_main_window_instantiation(qapp):
    window = MainWindow()
    assert window.deck_a is not None
    assert window.deck_b is not None
    assert window.comparison_bar is not None