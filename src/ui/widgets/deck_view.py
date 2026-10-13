"""
deck_view.py — Resilient Deck widget with explicit audio routing and stem fallback.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import soundfile as sf

from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.analysis.extractor import extract_full_pipeline
from src.analysis.package_store import PackageStore
from src.analysis.separator import load_pre_separated
from src.analysis.track_metadata import build_enriched_package
from src.core.package import EnrichedSongPackage
from src.core.song_dna import SongSummary
from src.ui.widgets.stem_mixer import StemMixer
from src.ui.widgets.timeline_view import TimelineView

logger = logging.getLogger(__name__)

_SOURCE_LABELS = {
    "file_tags": "file tags",
    "spotify": "Spotify",
    "user": "entered by user",
    "unknown": "unknown",
}


def format_tempo(summary: SongSummary) -> str:
    """Tempo for display; a fallback value is labelled as such."""
    bpm = round(summary.tempo_estimate, 1)
    if summary.tempo_is_fallback:
        return f"{bpm:g} BPM (fallback, not detected)"
    return f"{bpm:g} BPM"


def describe_track(package: EnrichedSongPackage) -> tuple[str, str, str]:
    """Return ``(title, artist, tooltip)`` for display, never inventing values.

    A missing title falls back to the file name (shown with its extension so
    it reads as a file name); a missing artist is shown as unknown.  The
    tooltip states the source of every field.
    """
    filename = Path(package.song_dna.metadata.filename).name
    info = package.track_info

    title = info.title.value if info and info.title.value else filename
    artist = info.artist.value if info and info.artist.value else "Unknown artist"

    if info is None:
        tooltip = f"File: {filename}\nNo track metadata available"
    else:
        lines = [f"File: {filename}"]
        for label, field in (
            ("Title", info.title),
            ("Artist", info.artist),
            ("Album", info.album),
            ("Release date", info.release_date),
        ):
            value = field.value if field.value else "unknown"
            lines.append(f"{label}: {value} ({_SOURCE_LABELS[field.source]})")
        tooltip = "\n".join(lines)

    return title, artist, tooltip


class DeckView(QGroupBox):
    """Interactive Deck component with synchronized audio playback and stem mixing."""

    package_loaded = Signal(object)

    STEM_NAMES = ["vocals", "drums", "bass", "other"]

    def __init__(self, deck_label: str, parent: QWidget | None = None) -> None:
        super().__init__(f"Deck {deck_label}", parent)
        self.deck_label = deck_label
        self.package: EnrichedSongPackage | None = None
        self._has_active_stems = False

        # 1. Audio Pipeline
        default_dev = QMediaDevices.defaultAudioOutput()

        self.master_output = QAudioOutput(default_dev, self)
        self.master_output.setVolume(1.0)
        self.master_output.setMuted(False)
        self.master_player = QMediaPlayer(self)
        self.master_player.setAudioOutput(self.master_output)

        self.stem_players: dict[str, QMediaPlayer] = {}
        self.stem_outputs: dict[str, QAudioOutput] = {}
        self.stem_gains: dict[str, float] = {s: 1.0 for s in self.STEM_NAMES}
        self.stem_mutes: dict[str, bool] = {s: False for s in self.STEM_NAMES}
        self.stem_solos: dict[str, bool] = {s: False for s in self.STEM_NAMES}

        for stem in self.STEM_NAMES:
            out = QAudioOutput(default_dev, self)
            out.setVolume(1.0)
            out.setMuted(False)
            player = QMediaPlayer(self)
            player.setAudioOutput(out)
            self.stem_outputs[stem] = out
            self.stem_players[stem] = player

        # 2. Timer for Playhead Tracking
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._sync_playhead)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Header Bar
        hdr_layout = QHBoxLayout()
        self.btn_load = QPushButton(f"Load Track {self.deck_label}")
        self.btn_load.clicked.connect(self._on_load_clicked)
        hdr_layout.addWidget(self.btn_load)

        self.lbl_title = QLabel("No Track Loaded")
        self.lbl_title.setStyleSheet("font-size: 13px; font-weight: bold;")
        hdr_layout.addWidget(self.lbl_title, stretch=1)

        self.lbl_status = QLabel("[IDLE]")
        self.lbl_status.setStyleSheet("color: #718096; font-size: 11px; font-weight: bold;")
        hdr_layout.addWidget(self.lbl_status)

        self.lbl_bpm_key = QLabel("--- BPM | ---")
        self.lbl_bpm_key.setStyleSheet("color: #4FD1C5; font-weight: bold;")
        hdr_layout.addWidget(self.lbl_bpm_key)
        layout.addLayout(hdr_layout)

        # Interactive Timeline
        self.timeline = TimelineView()
        self.timeline.seek_requested.connect(self._on_seek_requested)
        layout.addWidget(self.timeline)

        # Transport Bar
        ctl_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self._toggle_playback)
        self.btn_play.setEnabled(False)
        ctl_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.clicked.connect(self._stop_playback)
        self.btn_stop.setEnabled(False)
        ctl_layout.addWidget(self.btn_stop)

        ctl_layout.addWidget(QLabel("Master:"))
        self.master_vol = QSlider(Qt.Orientation.Horizontal)
        self.master_vol.setRange(0, 100)
        self.master_vol.setValue(100)
        self.master_vol.setFixedWidth(80)
        self.master_vol.valueChanged.connect(self._on_master_vol_changed)
        ctl_layout.addWidget(self.master_vol)

        self.lbl_time = QLabel("00:00 / 00:00")
        ctl_layout.addWidget(self.lbl_time, stretch=1, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(ctl_layout)

        # Stem Mixer
        self.stem_mixer = StemMixer()
        self.stem_mixer.stem_volume_changed.connect(self._on_stem_volume)
        self.stem_mixer.stem_mute_changed.connect(self._on_stem_mute)
        self.stem_mixer.stem_solo_changed.connect(self._on_stem_solo)
        layout.addWidget(self.stem_mixer)

    def load_package(self, package: EnrichedSongPackage, audio_path: Path | None = None) -> None:
        self._stop_playback()
        self.package = package
        dna = package.song_dna

        title, artist, tooltip = describe_track(package)
        self.lbl_title.setText(f"{title} — {artist}")
        self.lbl_title.setToolTip(tooltip)

        key = dna.summary.key_estimate or "Unknown Key"
        self.lbl_bpm_key.setText(f"{format_tempo(dna.summary)}  |  {key}")

        # 1. Resolve Audio File
        resolved_audio: Path | None = None
        if audio_path and audio_path.exists():
            resolved_audio = audio_path
        else:
            candidates = [
                Path(dna.metadata.filename),
                Path("data/raw") / Path(dna.metadata.filename).name,
                Path("data/raw") / dna.metadata.filename,
            ]
            for cand in candidates:
                if cand.exists():
                    resolved_audio = cand.resolve()
                    break

        # If still missing, ask user to select audio source directly
        if resolved_audio is None or not resolved_audio.exists():
            selected_audio, _ = QFileDialog.getOpenFileName(
                self,
                f"Select source audio for '{title}'",
                "",
                "Audio Files (*.wav *.mp3 *.flac *.m4a)",
            )
            if selected_audio:
                resolved_audio = Path(selected_audio).resolve()

        # 2. Check for Stem Waveforms
        self._has_active_stems = False
        stem_map: dict[str, Path] = {}
        if package.has_stems and package.separated_track:
            st = package.separated_track.stems
            stem_map = {
                "vocals": Path(st.vocals),
                "drums": Path(st.drums),
                "bass": Path(st.bass),
                "other": Path(st.other),
            }

        if stem_map and all(p.exists() for p in stem_map.values()):
            self._has_active_stems = True
            for name, path in stem_map.items():
                self.stem_players[name].setSource(QUrl.fromLocalFile(str(path.resolve())))
            self.master_output.setVolume(0.0)
            self.lbl_status.setText("[4-STEMS ACTIVE]")
            self.lbl_status.setStyleSheet("color: #48BB78; font-size: 11px; font-weight: bold;")
        else:
            self._has_active_stems = False
            self.lbl_status.setText("[MASTER STEREO ONLY]")
            self.lbl_status.setStyleSheet("color: #A0AEC0; font-size: 11px; font-weight: bold;")

        # 3. Bind Master Playback & Read Waveform Peaks
        raw_samples = None
        if resolved_audio and resolved_audio.exists():
            try:
                if not self._has_active_stems:
                    self.master_player.setSource(QUrl.fromLocalFile(str(resolved_audio)))
                    self.master_output.setVolume(self.master_vol.value() / 100.0)
                    self.master_output.setMuted(False)

                raw_samples, _ = sf.read(str(resolved_audio), frames=22050 * 300)
                if raw_samples.ndim > 1:
                    raw_samples = raw_samples.mean(axis=1)

                self.btn_play.setEnabled(True)
                self.btn_stop.setEnabled(True)
            except Exception as e:
                logger.error("Failed to load audio for playback: %s", e)
                self.btn_play.setEnabled(False)
                self.btn_stop.setEnabled(False)
        else:
            self.btn_play.setEnabled(False)
            self.btn_stop.setEnabled(False)

        self._apply_mixer_routing()
        self.timeline.set_package(package, raw_samples)
        self.package_loaded.emit(self.package)

    def _apply_mixer_routing(self) -> None:
        if not self._has_active_stems:
            return

        master_gain = self.master_vol.value() / 100.0
        any_solo = any(self.stem_solos.values())

        for stem in self.STEM_NAMES:
            out = self.stem_outputs[stem]
            base_gain = self.stem_gains[stem]
            is_muted = self.stem_mutes[stem]
            is_solo = self.stem_solos[stem]

            active = (is_solo and not is_muted) if any_solo else (not is_muted)
            out.setVolume(base_gain * master_gain if active else 0.0)

    def _on_stem_volume(self, stem_name: str, gain: float) -> None:
        self.stem_gains[stem_name] = gain
        self._apply_mixer_routing()

    def _on_stem_mute(self, stem_name: str, is_muted: bool) -> None:
        self.stem_mutes[stem_name] = is_muted
        self._apply_mixer_routing()

    def _on_stem_solo(self, stem_name: str, is_solo: bool) -> None:
        self.stem_solos[stem_name] = is_solo
        self._apply_mixer_routing()

    def _on_master_vol_changed(self, value: int) -> None:
        if self._has_active_stems:
            self._apply_mixer_routing()
        else:
            self.master_output.setVolume(value / 100.0)

    def _toggle_playback(self) -> None:
        active_player = self.stem_players["vocals"] if self._has_active_stems else self.master_player
        is_playing = active_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

        if is_playing:
            if self._has_active_stems:
                for p in self.stem_players.values():
                    p.pause()
            else:
                self.master_player.pause()
            self.btn_play.setText("▶ Play")
            self.timer.stop()
        else:
            if self._has_active_stems:
                self._apply_mixer_routing()
                for p in self.stem_players.values():
                    p.play()
            else:
                self.master_output.setMuted(False)
                self.master_player.play()
            self.btn_play.setText("⏸ Pause")
            self.timer.start()

    def _stop_playback(self) -> None:
        if self._has_active_stems:
            for p in self.stem_players.values():
                p.stop()
        else:
            self.master_player.stop()
        self.btn_play.setText("▶ Play")
        self.timer.stop()
        self.timeline.set_playhead(0.0)

    def _on_seek_requested(self, sec: float) -> None:
        ms = int(sec * 1000)
        if self._has_active_stems:
            for p in self.stem_players.values():
                p.setPosition(ms)
        else:
            self.master_player.setPosition(ms)

    def _sync_playhead(self) -> None:
        active_player = self.stem_players["vocals"] if self._has_active_stems else self.master_player
        current_sec = active_player.position() / 1000.0
        self.timeline.set_playhead(current_sec)
        dur_sec = self.package.song_dna.metadata.duration if self.package else 0.0
        self.lbl_time.setText(
            f"{int(current_sec // 60):02d}:{int(current_sec % 60):02d} / "
            f"{int(dur_sec // 60):02d}:{int(dur_sec % 60):02d}"
        )

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load Track for Deck {self.deck_label}",
            "",
            "All Supported (*.json *.wav *.mp3 *.flac *.m4a);;Packages (*.json);;Audio Files (*.wav *.mp3 *.flac *.m4a)",
        )
        if not path:
            return

        selected = Path(path)
        try:
            if selected.suffix.lower() == ".json":
                pkg = PackageStore.load(selected)
                self.load_package(pkg)
            else:
                base_pkg = extract_full_pipeline(file_path=selected, separate_stems=False)
                # Attach any pre-separated demo stems that live alongside the
                # chosen audio (e.g. ``data/demo_stems/heart_peripheral/``).
                if base_pkg.separated_track is None:
                    discovered = load_pre_separated(selected)
                    if discovered is not None:
                        base_pkg = dataclasses.replace(base_pkg, separated_track=discovered)
                enriched = build_enriched_package(base_pkg, selected)
                self.load_package(enriched, audio_path=selected)
        except Exception as err:
            logger.exception("Error loading track")
            QMessageBox.critical(self, "Load Error", f"Failed to load track:\n{err}")