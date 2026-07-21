"""
Tests for the Configuration Layer (``src.config``).

These tests verify:
- Frozen dataclass contract for ``EchoInsightConfig`` and ``AudioSettings``.
- Validation rules in ``AudioSettings.__post_init__``.
- ``PathsConfig`` sub-directory derivation.
- ``PreferenceManager`` file creation, loading, and saving.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.config.core import AudioSettings, EchoInsightConfig, PathsConfig
from src.config.preferences import PreferenceManager

# ======================================================================
# AudioSettings
# ======================================================================


class TestAudioSettings:
    def test_defaults(self) -> None:
        settings = AudioSettings()
        assert settings.sample_rate == 22050
        assert settings.hop_length == 512
        assert settings.n_mfcc == 13

    def test_frozen(self) -> None:
        settings = AudioSettings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.sample_rate = 44100  # type: ignore[misc]

    def test_negative_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            AudioSettings(sample_rate=-1)

    def test_zero_hop_length_raises(self) -> None:
        with pytest.raises(ValueError, match="hop_length must be positive"):
            AudioSettings(hop_length=0)

    def test_zero_n_mfcc_raises(self) -> None:
        with pytest.raises(ValueError, match="n_mfcc must be positive"):
            AudioSettings(n_mfcc=0)


# ======================================================================
# EchoInsightConfig
# ======================================================================


class TestEchoInsightConfig:
    def test_defaults(self) -> None:
        cfg = EchoInsightConfig()
        assert isinstance(cfg.audio, AudioSettings)
        assert cfg.audio.sample_rate == 22050

    def test_frozen(self) -> None:
        cfg = EchoInsightConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.audio = AudioSettings(sample_rate=48000)  # type: ignore[misc]

    def test_custom_audio_settings(self) -> None:
        custom = AudioSettings(sample_rate=48000, hop_length=1024, n_mfcc=20)
        cfg = EchoInsightConfig(audio=custom)
        assert cfg.audio.sample_rate == 48000
        assert cfg.audio.hop_length == 1024
        assert cfg.audio.n_mfcc == 20


# ======================================================================
# PathsConfig
# ======================================================================


class TestPathsConfig:
    def test_sub_dir_derivation(self) -> None:
        root = Path("/tmp/echoinsight_test")
        paths = PathsConfig(data_root=root)
        assert paths.data_root == root
        assert paths.raw_audio == root / "raw_audio"
        assert paths.dna == root / "dna"
        assert paths.frames == root / "frames"
        assert paths.stems == root / "stems"
        assert paths.cache == root / "cache"
        assert paths.exports == root / "exports"
        assert paths.logs == root / "logs"
        assert paths.models == root / "models"
        assert paths.spotify_cache == root / "spotify_cache"
        assert paths.projects == root / "projects"

    def test_relative_path_raises(self) -> None:
        with pytest.raises(ValueError, match="data_root must be an absolute path"):
            PathsConfig(data_root=Path("relative/path"))

    def test_frozen(self) -> None:
        paths = PathsConfig(data_root=Path("/tmp/test"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            paths.data_root = Path("/other")  # type: ignore[misc]


# ======================================================================
# PreferenceManager
# ======================================================================


@pytest.fixture
def temp_preference_manager(tmp_path: Path) -> PreferenceManager:
    """Return a PreferenceManager whose config directory is a temp dir."""
    import src.config.preferences as prefs_mod

    original = prefs_mod._get_preferences_path

    def fake_path() -> Path:
        return tmp_path / "preferences.toml"

    prefs_mod._get_preferences_path = fake_path
    yield PreferenceManager()
    prefs_mod._get_preferences_path = original


class TestPreferenceManager:
    def test_creates_file_on_load(
        self, temp_preference_manager: PreferenceManager
    ) -> None:
        pm = temp_preference_manager
        assert not pm.path.exists()
        _ = pm.load()
        assert pm.path.exists()

    def test_load_returns_defaults(
        self, temp_preference_manager: PreferenceManager
    ) -> None:
        pm = temp_preference_manager
        prefs = pm.load()
        assert prefs.version == 1
        assert prefs.theme == "dark"
        assert isinstance(prefs.data_root, Path)
        assert prefs.data_root.is_absolute()

    def test_load_twice_returns_same_values(
        self, temp_preference_manager: PreferenceManager
    ) -> None:
        pm = temp_preference_manager
        prefs1 = pm.load()
        prefs2 = pm.load()
        assert prefs1.theme == prefs2.theme
        assert prefs1.data_root == prefs2.data_root

    def test_save_persists_changes(
        self, temp_preference_manager: PreferenceManager
    ) -> None:
        pm = temp_preference_manager
        prefs = pm.load()
        prefs.theme = "light"
        pm.save(prefs)

        prefs2 = pm.load()
        assert prefs2.theme == "light"

    def test_save_and_reload_data_root(
        self, temp_preference_manager: PreferenceManager, tmp_path: Path
    ) -> None:
        pm = temp_preference_manager
        prefs = pm.load()
        new_root = tmp_path / "echoinsight_custom"
        prefs.data_root = new_root
        pm.save(prefs)

        prefs2 = pm.load()
        assert prefs2.data_root == new_root

    def test_version_field_present(
        self, temp_preference_manager: PreferenceManager
    ) -> None:
        pm = temp_preference_manager
        prefs = pm.load()
        assert prefs.version == 1

    def test_recent_projects_default_empty(
        self, temp_preference_manager: PreferenceManager
    ) -> None:
        pm = temp_preference_manager
        prefs = pm.load()
        assert prefs.recent_projects == []

    def test_save_recent_projects(
        self, temp_preference_manager: PreferenceManager, tmp_path: Path
    ) -> None:
        pm = temp_preference_manager
        prefs = pm.load()
        proj1 = tmp_path / "proj1"
        proj2 = tmp_path / "proj2"
        prefs.recent_projects = [proj1, proj2]
        pm.save(prefs)

        prefs2 = pm.load()
        assert len(prefs2.recent_projects) == 2
        assert proj1 in prefs2.recent_projects
        assert proj2 in prefs2.recent_projects


# ======================================================================
# Domain purity — core must not import config or app
# ======================================================================


def test_domain_layer_pure() -> None:
    """Verify that src.core does not import from src.config or src.app.

    This test runs in a subprocess to ensure a clean Python environment
    without pre-loaded config/app modules from other tests.
    """
    import subprocess
    import sys

    code = """
import sys
import src.core.dna_schema  # noqa: F401
bad = {n for n in sys.modules if n.startswith("src.config") or n.startswith("src.app")}
if bad:
    print(f"FAIL: domain imported {{bad}}")
    sys.exit(1)
else:
    print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert (
        result.returncode == 0
    ), f"Domain layer imported config/app: {result.stdout}{result.stderr}"
