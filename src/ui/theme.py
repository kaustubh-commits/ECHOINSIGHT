"""
theme.py — Styling constants, color palettes, and stylesheets for EchoInsight UI.
"""

from __future__ import annotations

DARK_THEME_QSS = """
QMainWindow, QWidget {
    background-color: #121318;
    color: #E2E8F0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #2D3748;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 12px;
    font-weight: bold;
    color: #A0AEC0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
}

QPushButton {
    background-color: #2B6CB0;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #3182CE;
}
QPushButton:pressed {
    background-color: #2B6CB0;
}
QPushButton:disabled {
    background-color: #2D3748;
    color: #718096;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #2D3748;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #4299E1;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #E2E8F0;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}

QSlider::groove:vertical {
    width: 6px;
    background: #2D3748;
    border-radius: 3px;
}
QSlider::sub-page:vertical {
    background: #4299E1;
    border-radius: 3px;
}
QSlider::handle:vertical {
    background: #E2E8F0;
    height: 14px;
    margin-left: -4px;
    margin-right: -4px;
    border-radius: 7px;
}

QLabel {
    color: #E2E8F0;
}
"""

STEM_COLORS = {
    "vocals": "#4FD1C5",  # Teal
    "drums": "#F6AD55",   # Orange
    "bass": "#9F7AEA",    # Purple
    "other": "#63B3ED",   # Blue
}

SEGMENT_COLORS = {
    "intro": "#4A5568",
    "verse": "#2B6CB0",
    "chorus": "#C53030",
    "bridge": "#B7791F",
    "outro": "#2D3748",
    "unknown": "#1A202C",
}