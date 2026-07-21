"""
core.py — Immutable application defaults for EchoInsight.

This file defines the frozen dataclasses that hold EchoInsight's
immutable application defaults.  These values are determined at
development time and are never modified at runtime or by the user.

Layer responsibility
--------------------
``src.config.core`` defines *what the defaults are*.
It does not know where they are stored, who reads them, or how
they are overridden.  It imports only Python stdlib types.

Future config groups (AnalysisConfig, ComparisonConfig, etc.)
should be added as frozen dataclasses in this file and composed
into ``EchoInsightConfig``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# AudioSettings
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AudioSettings:
    """Immutable default parameters for audio analysis.

    These values match the defaults used by the librosa-based extraction
    pipeline (``src.analysis.extractor``).  They are the canonical source
    of truth that other modules *should* reference, though existing
    extractor code may currently hardcode its own constants.

    Attributes
    ----------
    sample_rate : int
        Default target sample rate in Hz for audio loading.
        ``22050`` is the standard for music analysis — it captures the
        full frequency range of most instruments while keeping the
        Nyquist frequency (11025 Hz) well above the range of musical
        fundamentals.
    hop_length : int
        Number of samples between successive frames.
        ``512`` at 22050 Hz gives ~23 ms per frame, a balance between
        temporal resolution and computational cost.
    n_mfcc : int
        Number of Mel-Frequency Cepstral Coefficients to retain.
        ``13`` is the industry-standard count; the first coefficient
        carries overall spectral energy and the remaining 12 describe
        the spectral envelope.
    """

    sample_rate: int = 22050
    hop_length: int = 512
    n_mfcc: int = 13

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}")
        if self.n_mfcc <= 0:
            raise ValueError(f"n_mfcc must be positive, got {self.n_mfcc}")


# ---------------------------------------------------------------------------
# PathsConfig
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    """Centralised, immutable storage path layout for EchoInsight.

    All application data directories are derived from a single
    ``data_root`` and exposed as attributes.  This keeps filesystem
    paths in one place as the application grows.

    The layout mirrors the domain model:

    ``data_root`` / ``raw_audio``   — source audio files (MP3, WAV, …)
    ``data_root`` / ``dna``         — extracted SongDNA JSON files
    ``data_root`` / ``frames``      — frame-level feature arrays (NPZ)
    ``data_root`` / ``stems``       — source-separated stem audio (future)
    ``data_root`` / ``cache``       — general-purpose analysis cache
    ``data_root`` / ``exports``     — exported reports, playlists, etc.
    ``data_root`` / ``logs``        — application log files
    ``data_root`` / ``models``      — downloaded ML model weights
    ``data_root`` / ``spotify_cache`` — cached Spotify API responses
    ``data_root`` / ``projects``    — user workspace / project files

    Architectural notes
    -------------------
    -   This is a *computed* configuration object — all paths are
        derived deterministically from ``data_root``.
    -   The config is frozen.  To change paths, construct a new instance
        with a different ``data_root``.
    -   ``ApplicationContext.resolve_data_path()`` provides a URI-based
        lookup mechanism on top of this layout.

    Parameters
    ----------
    data_root : Path
        Root directory for all EchoInsight data.  Must be an absolute path.
    """

    data_root: Path

    # Sub-directories (computed in __post_init__)
    raw_audio: Path = dataclasses.field(init=False)
    dna: Path = dataclasses.field(init=False)
    frames: Path = dataclasses.field(init=False)
    stems: Path = dataclasses.field(init=False)
    cache: Path = dataclasses.field(init=False)
    exports: Path = dataclasses.field(init=False)
    logs: Path = dataclasses.field(init=False)
    models: Path = dataclasses.field(init=False)
    spotify_cache: Path = dataclasses.field(init=False)
    projects: Path = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data_root, Path):
            object.__setattr__(self, "data_root", Path(self.data_root))
        if not self.data_root.is_absolute():
            raise ValueError(
                f"data_root must be an absolute path, got {self.data_root}"
            )

        _sub_dirs: List[str] = [
            "raw_audio",
            "dna",
            "frames",
            "stems",
            "cache",
            "exports",
            "logs",
            "models",
            "spotify_cache",
            "projects",
        ]
        for name in _sub_dirs:
            object.__setattr__(self, name, self.data_root / name)


# ---------------------------------------------------------------------------
# EchoInsightConfig
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EchoInsightConfig:
    """Aggregate of all immutable application defaults for EchoInsight.

    ``EchoInsightConfig`` is the single entry point for reading
    development-time configuration.  Every subsystem that needs default
    parameters should access them through this object.

    Future additions
    ----------------
    -   ``analysis: AnalysisConfig`` — DSP extraction parameters
    -   ``comparison: ComparisonConfig`` — default similarity weights
    -   ``ml: MLConfig`` — model paths, device settings

    Parameters
    ----------
    audio : AudioSettings
        Default audio analysis parameters (sample rate, hop length, etc.)
    """

    audio: AudioSettings = dataclasses.field(default_factory=AudioSettings)

    # Future config groups will be added here:
    # analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    # comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
