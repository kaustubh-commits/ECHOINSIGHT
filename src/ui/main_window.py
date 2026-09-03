"""
main_window.py — Top-level Workspace Window for the EchoInsight MVP Prototype.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from src.ui.theme import DARK_THEME_QSS
from src.ui.widgets.comparison_bar import ComparisonBar
from src.ui.widgets.deck_view import DeckView


class MainWindow(QMainWindow):
    """Main application window containing Dual Decks and Comparison Panel."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EchoInsight — Creative Music Intelligence Workspace")
        self.resize(1280, 780)
        self.setStyleSheet(DARK_THEME_QSS)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # 1. Dual Decks Layout (Side-by-side Deck A and Deck B)
        decks_layout = QHBoxLayout()
        self.deck_a = DeckView("A")
        self.deck_b = DeckView("B")
        decks_layout.addWidget(self.deck_a)
        decks_layout.addWidget(self.deck_b)
        root_layout.addLayout(decks_layout, stretch=1)

        # 2. Bottom Comparison Bar
        self.comparison_bar = ComparisonBar()
        root_layout.addWidget(self.comparison_bar)

        # 3. Connect signals for real-time comparison updates
        self.deck_a.package_loaded.connect(self._on_track_changed)
        self.deck_b.package_loaded.connect(self._on_track_changed)

    def _on_track_changed(self, _) -> None:
        """Triggered whenever Deck A or Deck B loads a track."""
        self.comparison_bar.update_comparison(self.deck_a.package, self.deck_b.package)