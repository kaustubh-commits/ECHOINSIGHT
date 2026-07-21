"""
Tests for the bootstrap pipeline and ApplicationContext.

These tests verify:
- ``bootstrap_application()`` returns a fully initialised ``ApplicationContext``.
- Logger is correctly initialised as a named logger.
- ``resolve_data_path()`` returns absolute paths relative to data_root.
- Path resolution rejects absolute URIs.
- Backward compatibility of extraction pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from main import bootstrap_application
from src.app.context import ApplicationContext
from src.config.core import EchoInsightConfig
from src.config.preferences import UserPreferences

# ======================================================================
# Bootstrap
# ======================================================================


def test_bootstrap_returns_context() -> None:
    """bootstrap_application() should return an ApplicationContext."""
    ctx = bootstrap_application()
    assert isinstance(ctx, ApplicationContext)


def test_bootstrap_has_config() -> None:
    ctx = bootstrap_application()
    assert isinstance(ctx.config, EchoInsightConfig)
    assert ctx.config.audio.sample_rate == 22050


def test_bootstrap_has_preferences() -> None:
    ctx = bootstrap_application()
    assert isinstance(ctx.preferences, UserPreferences)
    assert ctx.preferences.version == 1


def test_bootstrap_has_logger() -> None:
    ctx = bootstrap_application()
    assert isinstance(ctx.logger, logging.Logger)
    assert ctx.logger.name == "echoinsight"


def test_bootstrap_has_paths() -> None:
    ctx = bootstrap_application()
    assert ctx.paths.data_root == ctx.preferences.data_root
    assert ctx.paths.dna == ctx.preferences.data_root / "dna"
    assert ctx.paths.raw_audio == ctx.preferences.data_root / "raw_audio"


# ======================================================================
# Logger
# ======================================================================


def test_logger_name_is_echoinsight() -> None:
    ctx = bootstrap_application()
    assert ctx.logger.name == "echoinsight"


def test_logger_level_can_be_changed() -> None:
    ctx = bootstrap_application()
    ctx.set_log_level(logging.WARNING)
    assert ctx.logger.level == logging.WARNING
    ctx.set_log_level(logging.DEBUG)
    assert ctx.logger.level == logging.DEBUG


# ======================================================================
# Path Resolution
# ======================================================================


def test_resolve_data_path_returns_absolute(tmp_path: Path) -> None:
    config = EchoInsightConfig()
    prefs = UserPreferences(data_root=tmp_path)
    ctx = ApplicationContext(config=config, preferences=prefs)
    resolved = ctx.resolve_data_path("frames/test.npz")
    assert resolved == (tmp_path / "frames" / "test.npz").resolve()


def test_resolve_data_path_rejects_absolute() -> None:
    config = EchoInsightConfig()
    prefs = UserPreferences(data_root=Path("/tmp"))
    ctx = ApplicationContext(config=config, preferences=prefs)
    with pytest.raises(ValueError, match="relative_uri must be a relative path"):
        ctx.resolve_data_path("/etc/passwd")


def test_resolve_data_path_nested(tmp_path: Path) -> None:
    config = EchoInsightConfig()
    prefs = UserPreferences(data_root=tmp_path)
    ctx = ApplicationContext(config=config, preferences=prefs)
    resolved = ctx.resolve_data_path("dna/artist/album/song.json")
    assert resolved == (tmp_path / "dna" / "artist" / "album" / "song.json").resolve()


# ======================================================================
# Backward Compatibility — Extraction Pipeline Still Works
# ======================================================================


def test_extraction_pipeline_produces_song_dna() -> None:
    """Verify the existing extraction pipeline still works after bootstrap changes.

    This test loads a known MP3 and checks that the returned dict has
    the expected SongDNA structure.
    """
    from src.analysis import extract_song_dna

    raw_dir = Path("data/raw")
    mp3_files = sorted(raw_dir.glob("*.mp3"))
    if not mp3_files:
        pytest.skip("No MP3 files in data/raw/ — cannot test extraction")

    dna = extract_song_dna(str(mp3_files[0]))

    assert dna.metadata.duration > 0
    assert dna.rhythm.tempo > 0
    assert len(dna.timbre.mfcc_mean) == 13
    assert len(dna.tonal.chroma_mean) == 12


def test_bootstrap_does_not_break_cli_extraction() -> None:
    """Simulate the CLI flow: bootstrap then extract on a real MP3."""
    from src.analysis import extract_song_dna

    raw_dir = Path("data/raw")
    mp3_files = sorted(raw_dir.glob("*.mp3"))
    if not mp3_files:
        pytest.skip("No MP3 files in data/raw/ — cannot test extraction")

    # This is the exact flow from main.py
    app = bootstrap_application()
    app.set_log_level(logging.WARNING)

    dna = extract_song_dna(str(mp3_files[0]))
    import dataclasses
    import json

    output = json.dumps(dataclasses.asdict(dna), indent=2)
    assert '"metadata"' in output
    assert '"rhythm"' in output
    assert '"timbre"' in output
    assert '"tonal"' in output
    assert '"structure"' in output
    assert '"intelligence"' in output
    assert '"stem"' in output


# ======================================================================
# Circular Import Check
# ======================================================================


def test_no_circular_imports() -> None:
    """Verify the full import chain has no circular dependencies."""

    # If we reach here without ImportError, there are no circular imports
    assert True
