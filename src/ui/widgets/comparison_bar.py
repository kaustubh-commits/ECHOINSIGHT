"""
comparison_bar.py — Raw measurements comparing Deck A and Deck B.

All values come from the comparison engine (``src.analysis.comparison``).
They are simple measurements, not a match or compatibility score.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.analysis.comparison import ComparisonReport, compare
from src.analysis.comparison.engine import _UNKNOWN_KEY_DISTANCE
from src.core.package import EnrichedSongPackage


def format_comparison(
    report: ComparisonReport, key_a: str | None, key_b: str | None
) -> tuple[str, str, str]:
    """Return the (tempo, key, energy) card texts for a comparison report."""
    if report.tempo_difference is None:
        tempo = "n/a (tempo not detected)"
    else:
        tempo = f"{report.tempo_difference:.1f} BPM apart"

    keys = f"{key_a or 'Unknown'} vs {key_b or 'Unknown'}"
    if report.key_relationship == _UNKNOWN_KEY_DISTANCE:
        harmony = f"{keys} · Camelot n/a"
    else:
        harmony = f"{keys} · Camelot distance {report.key_relationship}"

    energy = f"{report.energy_difference:+.2f}  (−1…+1, + = A higher)"
    return tempo, harmony, energy


class ComparisonBar(QGroupBox):
    """Displays raw tempo, key and RMS-energy differences between the decks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Track Comparison (raw measurements)", parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.lbl_tempo = self._create_card(
            "TEMPO DIFFERENCE", "---", "#4299E1", layout,
            "Absolute difference between the two detected tempos.",
        )
        self.lbl_harmony = self._create_card(
            "KEYS", "---", "#9F7AEA", layout,
            "Estimated keys and their Camelot wheel distance.\n"
            "0 = same key or relative major/minor; 6 = furthest apart.",
        )
        self.lbl_energy = self._create_card(
            "RMS ENERGY DIFFERENCE", "---", "#F6AD55", layout,
            "(RMS_A − RMS_B) / max(RMS_A, RMS_B), using each track's mean RMS.\n"
            "A simple level measure. Not loudness (LUFS) and not a\n"
            "match or compatibility score.",
        )

    def _create_card(
        self,
        title: str,
        default_val: str,
        accent_color: str,
        parent_layout: QHBoxLayout,
        tooltip: str,
    ) -> QLabel:
        frame = QFrame()
        frame.setStyleSheet("background-color: #1A202C; border-radius: 6px; padding: 6px;")
        frame.setToolTip(tooltip)
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(4, 4, 4, 4)

        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("font-size: 11px; color: #718096; font-weight: bold;")
        vbox.addWidget(t_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        v_lbl = QLabel(default_val)
        v_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {accent_color};")
        vbox.addWidget(v_lbl, alignment=Qt.AlignmentFlag.AlignCenter)

        parent_layout.addWidget(frame)
        return v_lbl

    def update_comparison(self, pkg_a: EnrichedSongPackage | None, pkg_b: EnrichedSongPackage | None) -> None:
        """Recompute and display the comparison for the loaded decks."""
        if not pkg_a or not pkg_b:
            self.lbl_tempo.setText("Load both tracks")
            self.lbl_harmony.setText("---")
            self.lbl_energy.setText("---")
            return

        report = compare(pkg_a.song_dna, pkg_b.song_dna)
        tempo, harmony, energy = format_comparison(
            report,
            pkg_a.song_dna.summary.key_estimate,
            pkg_b.song_dna.summary.key_estimate,
        )
        self.lbl_tempo.setText(tempo)
        self.lbl_harmony.setText(harmony)
        self.lbl_energy.setText(energy)
