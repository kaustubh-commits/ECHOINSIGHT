"""
timeline_view.py — Interactive waveform, musical timeline grid, and structural segments.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QToolTip, QWidget

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
        self._selected_segment: int | None = None
        self._hover_segment: int | None = None

    def set_package(self, package: EnrichedSongPackage, audio_samples: np.ndarray | None = None) -> None:
        """Configures timeline and computes peak envelope."""
        self._package = package
        self._duration = package.song_dna.metadata.duration
        self._playhead_sec = 0.0
        self._selected_segment = None
        self._hover_segment = None

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

    def _segment_at(self, time_seconds: float) -> int | None:
        """Index of the segment whose half-open span contains time_seconds.

        Matches ``start_time <= time_seconds < end_time``.  Returns ``None``
        when the time lies outside every segment or no package is loaded.
        """
        if self._package is None or self._duration <= 0:
            return None
        segments = self._package.song_dna.structure.segments
        for i, seg in enumerate(segments):
            if seg.timing.start_time <= time_seconds < seg.timing.end_time:
                return i
        return None

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format a seconds value as MM:SS for readouts."""
        total = max(0, int(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"

    def set_playhead(self, current_sec: float) -> None:
        self._playhead_sec = max(0.0, min(current_sec, self._duration))
        self.update()

    def mousePressEvent(self, event) -> None:
        if self._duration <= 0 or self.width() <= 0:
            return
        ratio = max(0.0, min(event.position().x() / self.width(), 1.0))
        target_sec = ratio * self._duration

        idx = self._segment_at(target_sec)
        if idx is not None:
            # Clicking inside a structural segment seeks to its start time.
            self._selected_segment = idx
            seek_sec = self._package.song_dna.structure.segments[idx].timing.start_time
        else:
            # No segment under the pointer: keep exact-position seeking.
            self._selected_segment = None
            seek_sec = target_sec

        self.seek_requested.emit(seek_sec)
        self.set_playhead(seek_sec)

    def mouseMoveEvent(self, event) -> None:
        if self._package is None or self._duration <= 0 or self.width() <= 0:
            return
        ratio = max(0.0, min(event.position().x() / self.width(), 1.0))
        hover = self._segment_at(ratio * self._duration)
        if hover != self._hover_segment:
            self._hover_segment = hover
            self.update()

        if hover is not None:
            seg = self._package.song_dna.structure.segments[hover]
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"Segment {hover + 1}\n{self._fmt_time(seg.timing.start_time)} – {self._fmt_time(seg.timing.end_time)}\nDuration: {seg.timing.duration:.1f}s",
                self,
            )

    def leaveEvent(self, event) -> None:
        self._hover_segment = None
        self.update()
        super().leaveEvent(event)

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
        playing_idx = self._segment_at(self._playhead_sec)
        for i, seg in enumerate(structure.segments):
            seg_start = (seg.timing.start_time / self._duration) * w
            seg_w = (seg.timing.duration / self._duration) * w
            rect = QRectF(seg_start, 0, seg_w, h - 22)

            # Display-only neutral tint: the extractor labels every segment
            # UNKNOWN, so semantic captions are never rendered.
            color = QColor(SEGMENT_COLORS.get("unknown", "#2D3748"))
            color.setAlpha(120 if i == self._selected_segment else 60)
            painter.fillRect(rect, color)

            # Hierarchy: selected > currently playing > hovered > normal.
            if i == self._selected_segment:
                painter.setPen(QPen(QColor("#4FD1C5"), 2))
                painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
            elif i == playing_idx:
                painter.setPen(QPen(QColor("#63B3ED"), 1))
                painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
            elif i == self._hover_segment:
                painter.setPen(QPen(QColor("#718096"), 1))
                painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))

            # Boundary separator at every segment start.
            painter.setPen(QPen(QColor("#4A5568"), 1))
            painter.drawLine(int(seg_start), 0, int(seg_start), int(h - 22))

            # Neutral caption only when the segment is wide enough to fit.
            if seg_w >= 30:
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                painter.setPen(QColor("#CBD5E0"))
                painter.drawText(
                    QRectF(seg_start + 4, 4, seg_w - 8, 16),
                    Qt.AlignmentFlag.AlignLeft,
                    f"S{i + 1}",
                )


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

        # Render beat & bar divisions from the real Timeline when available;
        # fall back to a synthetic tempo-derived 4/4 grid only when no
        # Timeline exists (e.g. legacy packages extracted pre-timeline).
        timeline = self._package.song_dna.timeline
        if timeline is not None and timeline.bars:
            bar_start_times = {round(bar.start_time, 3) for bar in timeline.bars}

            painter.setFont(QFont("Segoe UI", 7))

            # Bars: full-height separators; label only bar 1 and every 8th bar.
            for bar in timeline.bars:
                x = (bar.start_time / self._duration) * w
                painter.setPen(QPen(QColor("#A0AEC0"), 1))
                painter.drawLine(int(x), int(grid_y), int(x), int(h))
                if bar.number == 1 or bar.number % 8 == 0:
                    painter.drawText(
                        QRectF(x + 2, grid_y + 2, 30, 16),
                        Qt.AlignmentFlag.AlignLeft,
                        f"|{bar.number}",
                    )

            # Beats: short ticks in the lower strip, skipping bar starts so
            # beat ticks never overwrite the bar separators above.
            painter.setPen(QPen(QColor("#4A5568"), 1))
            for beat in timeline.beats:
                if round(beat, 3) in bar_start_times:
                    continue
                x = (beat / self._duration) * w
                painter.drawLine(int(x), int(grid_y + 12), int(x), int(h))
        else:
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