"""
comparison_bar.py — Deterministic comparative analysis strip between Deck A and Deck B.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.core.package import EnrichedSongPackage


class ComparisonBar(QGroupBox):
    """Real-time comparison readout displaying BPM delta, harmonic relationship, and energy delta."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Track Comparison & Harmonic Compatibility", parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Metric 1: Tempo Difference
        self.lbl_tempo = self._create_card("TEMPO DELTA", "---", "#4299E1", layout)
        # Metric 2: Harmonic Match
        self.lbl_harmony = self._create_card("HARMONIC RELATION", "---", "#9F7AEA", layout)
        # Metric 3: Energy Ratio
        self.lbl_energy = self._create_card("ENERGY BALANCE", "---", "#F6AD55", layout)

    def _create_card(self, title: str, default_val: str, accent_color: str, parent_layout: QHBoxLayout) -> QLabel:
        frame = QFrame()
        frame.setStyleSheet("background-color: #1A202C; border-radius: 6px; padding: 6px;")
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
        """Computes and updates side-by-side comparison metrics."""
        if not pkg_a or not pkg_b:
            self.lbl_tempo.setText("Load both tracks")
            self.lbl_harmony.setText("---")
            self.lbl_energy.setText("---")
            return

        sum_a = pkg_a.song_dna.summary
        sum_b = pkg_b.song_dna.summary
        if sum_a.tempo_is_fallback or sum_b.tempo_is_fallback:
            self.lbl_tempo.setText("n/a (tempo not detected)")
        else:
            bpm_a = sum_a.tempo_estimate
            bpm_b = sum_b.tempo_estimate
            diff_bpm = bpm_b - bpm_a
            sign = "+" if diff_bpm > 0 else ""
            self.lbl_tempo.setText(f"{sign}{diff_bpm:.1f} BPM ({bpm_a:.0f} → {bpm_b:.0f})")

        key_a = pkg_a.song_dna.summary.key_estimate or "Unknown"
        key_b = pkg_b.song_dna.summary.key_estimate or "Unknown"
        self.lbl_harmony.setText(f"{key_a} vs {key_b}")

        rms_a = pkg_a.song_dna.summary.rms_energy_mean
        rms_b = pkg_b.song_dna.summary.rms_energy_mean
        ratio = (rms_b / max(1e-5, rms_a)) * 100
        self.lbl_energy.setText(f"{ratio:.0f}% Energy Match")