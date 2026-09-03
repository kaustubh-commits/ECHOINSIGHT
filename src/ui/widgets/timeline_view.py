"""
timeline_view.py — Interactive waveform, musical timeline grid, and structural segments.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from src.core.package import EnrichedSongPackage
from src.ui.theme import SEGMENT_COLORS


class TimelineView(QWidget):
    """Custom paint widget rendering waveform, musical grid (beats/bars), and structure."""

    seek_requested = Signal(float)  # Emits target position in seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._package: EnrichedSongPackage | None = None
        self._duration: float = 0.0
        self._playhead_sec: float = 0.0
        self._waveform_peaks: np.ndarray | None = None
        self.setMouseTracking(True)

    def set_package(self, package: EnrichedSongPackage, audio_samples: np.ndarray | None = None) -> None:
        """Configures timeline and computes peak envelope."""
        self._package = package
        self._duration = package.song_dna.metadata.duration
        self._playhead_sec = 0.0

        if audio_samples is not None and len(audio_samples) > 0:
            n_bins = 600
            chunk_size = max(1, len(audio_samples) // n_bins)
            reshaped = audio_samples[: chunk_size * n_bins].reshape(n_bins, chunk_size)
            self._waveform_peaks = np.max(np.abs(reshaped), axis=1)
            max_val = np.max(self._waveform_peaks)
            if max_val > 0:
                self._waveform_peaks /= max_val
        else:
            self._waveform_peaks = np.full(600, 0.35)

        self.update()

    def set_playhead(self, current_sec: float) -> None:
        self._playhead_sec = max(0.0, min(current_sec, self._duration))
        self.update()

    def mousePressEvent(self, event) -> None:
        if self._duration > 0 and self.width() > 0:
            ratio = max(0.0, min(event.position().x() / self.width(), 1.0))
            target_sec = ratio * self._duration
            self.seek_requested.emit(target_sec)
            self.set_playhead(target_sec)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = float(self.width())
        h = float(self.height())

        # 1. Background
        painter.fillRect(self.rect(), QColor("#121318"))

        if self._package is None or self._duration <= 0:
            painter.setPen(QColor("#718096"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Track Loaded")
            return

        # 2. Structural Segment Blocks
        structure = self._package.song_dna.structure
        for seg in structure.segments:
            seg_start = (seg.timing.start_time / self._duration) * w
            seg_w = (seg.timing.duration / self._duration) * w
            label = seg.context.label.value.lower()
            color = QColor(SEGMENT_COLORS.get(label, "#2D3748"))
            color.setAlpha(60)

            rect = QRectF(seg_start, 0, seg_w, h - 22)
            painter.fillRect(rect, color)

            painter.setPen(QPen(QColor("#4A5568"), 1))
            painter.drawLine(int(seg_start), 0, int(seg_start), int(h - 22))

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#CBD5E0"))
            painter.drawText(QRectF(seg_start + 4, 4, seg_w - 8, 16), Qt.AlignmentFlag.AlignLeft, label.upper())

        # 3. Waveform Peaks
        if self._waveform_peaks is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#4299E1")))
            n_bins = len(self._waveform_peaks)
            dx = w / n_bins
            mid_y = (h - 22) / 2.0

            path = QPainterPath()
            for i, peak in enumerate(self._waveform_peaks):
                x = i * dx
                bar_h = peak * ((h - 22) * 0.38)
                path.addRect(QRectF(x, mid_y - bar_h, max(1.0, dx - 0.5), bar_h * 2))
            painter.drawPath(path)

        # 4. Musical Timeline Bar (Beats / Bars Strip at bottom)
        grid_y = h - 22
        painter.fillRect(QRectF(0, grid_y, w, 22), QColor("#1A202C"))
        painter.setPen(QPen(QColor("#2D3748"), 1))
        painter.drawLine(0, int(grid_y), int(w), int(grid_y))

        # Render beat & bar divisions based on tempo
        tempo = self._package.song_dna.summary.tempo_estimate
        if tempo > 0:
            spb = 60.0 / tempo  # seconds per beat
            total_beats = int(self._duration / spb)
            
            painter.setFont(QFont("Segoe UI", 7))
            for b in range(0, total_beats + 1):
                beat_sec = b * spb
                x = (beat_sec / self._duration) * w
                is_bar = (b % 4 == 0)

                if is_bar:
                    bar_num = (b // 4) + 1
                    painter.setPen(QPen(QColor("#A0AEC0"), 1))
                    painter.drawLine(int(x), int(grid_y), int(x), int(h))
                    painter.drawText(QRectF(x + 2, grid_y + 2, 30, 16), Qt.AlignmentFlag.AlignLeft, f"|{bar_num}")
                else:
                    painter.setPen(QPen(QColor("#4A5568"), 1))
                    painter.drawLine(int(x), int(grid_y + 12), int(x), int(h))

        # 5. Playhead
        playhead_x = (self._playhead_sec / self._duration) * w
        painter.setPen(QPen(QColor("#E53E3E"), 2))
        painter.drawLine(int(playhead_x), 0, int(playhead_x), int(h))