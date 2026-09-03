"""
stem_mixer.py — 4-channel stem mixer strip with Volume, Mute, and Solo controls.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from src.ui.theme import STEM_COLORS


class StemChannelStrip(QWidget):
    """Single channel strip for an individual audio stem (e.g. Drums, Vocals)."""

    volume_changed = Signal(str, float)  # (stem_name, volume_gain [0.0 - 1.0])
    mute_toggled = Signal(str, bool)     # (stem_name, is_muted)
    solo_toggled = Signal(str, bool)     # (stem_name, is_solo)

    def __init__(self, stem_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stem_name = stem_name

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Stem Title
        self.lbl_title = QLabel(stem_name.upper())
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet(f"color: {STEM_COLORS.get(stem_name, '#E2E8F0')}; font-weight: bold;")
        layout.addWidget(self.lbl_title)

        # Vertical Fader
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(0, 100)
        self.slider.setValue(100)
        self.slider.setMinimumHeight(100)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, alignment=Qt.AlignmentFlag.AlignCenter)

        # Gain Label
        self.lbl_gain = QLabel("100%")
        self.lbl_gain.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_gain)

        # Mute / Solo Buttons
        btn_layout = QHBoxLayout()
        self.btn_mute = QPushButton("M")
        self.btn_mute.setCheckable(True)
        self.btn_mute.setFixedSize(26, 24)
        self.btn_mute.setStyleSheet("QPushButton:checked { background-color: #E53E3E; color: white; font-weight: bold; }")
        self.btn_mute.toggled.connect(lambda chk: self.mute_toggled.emit(self.stem_name, chk))

        self.btn_solo = QPushButton("S")
        self.btn_solo.setCheckable(True)
        self.btn_solo.setFixedSize(26, 24)
        self.btn_solo.setStyleSheet("QPushButton:checked { background-color: #ECC94B; color: black; font-weight: bold; }")
        self.btn_solo.toggled.connect(lambda chk: self.solo_toggled.emit(self.stem_name, chk))

        btn_layout.addWidget(self.btn_mute)
        btn_layout.addWidget(self.btn_solo)
        layout.addLayout(btn_layout)

    def _on_slider_changed(self, val: int) -> None:
        self.lbl_gain.setText(f"{val}%")
        self.volume_changed.emit(self.stem_name, val / 100.0)


class StemMixer(QGroupBox):
    """4-channel horizontal mixer panel for separated tracks."""

    stem_volume_changed = Signal(str, float)
    stem_mute_changed = Signal(str, bool)
    stem_solo_changed = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Stem Mixer", parent)
        self.channels: dict[str, StemChannelStrip] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 12, 6, 6)

        for stem in ["vocals", "drums", "bass", "other"]:
            strip = StemChannelStrip(stem)
            strip.volume_changed.connect(self.stem_volume_changed.emit)
            strip.mute_toggled.connect(self.stem_mute_changed.emit)
            strip.solo_toggled.connect(self.stem_solo_changed.emit)
            self.channels[stem] = strip
            layout.addWidget(strip)