"""
preferences.py — User-editable preferences for EchoInsight.

This module manages a mutable TOML preferences file stored in the
platform-appropriate configuration directory (via ``platformdirs``).

Layer responsibility
--------------------
``src.config.preferences`` manages persistence of user preferences.
It knows about the filesystem (where to read/write the TOML file)
but contains no business logic and does not import from ``src.core``
or ``src.analysis``.

File format
-----------
The preferences TOML file includes comments explaining every setting,
making it self-documenting for advanced users.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, List

import platformdirs
import tomlkit

logger = logging.getLogger("echoinsight.config")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREFERENCES_VERSION: int = 1
"""Current version of the preferences schema.

Increment when making backward-incompatible changes to the file format.
Migration logic should check this value on load.
"""

_CONFIG_DIR_NAME: str = "EchoInsight"
"""Application sub-directory name inside the platform config root."""

_PREFERENCES_FILENAME: str = "preferences.toml"
"""Filename for the preferences file."""

_DEFAULT_DATA_ROOT_NAME: str = "EchoInsightData"
"""Default name for the data root directory inside XDG_DATA_HOME / equivalent."""

# ---------------------------------------------------------------------------
# UserPreferences
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class UserPreferences:
    """Mutable user-editable preferences for EchoInsight.

    Unlike ``EchoInsightConfig`` (which is frozen), ``UserPreferences``
    is a normal mutable dataclass.  Users and the application are free to
    modify fields at runtime.  Call ``PreferenceManager.save()`` to
    persist changes to disk.

    Attributes
    ----------
    version : int
        Schema version for migration support.  Incremented when the
        file format changes incompatibly.
    data_root : Path
        Absolute path to the root directory where all EchoInsight data
        is stored (raw audio, SongDNA JSON, frames, stems, cache, etc.).
    theme : str
        UI theme identifier (e.g. ``"light"``, ``"dark"``, ``"system"``).
        Currently unused — reserved for the future Workstation frontend.
    recent_projects : list[Path]
        List of recently opened project paths, most recent first.
        Maintained automatically by the Workspace service (future).
    """

    version: int = _PREFERENCES_VERSION
    data_root: Path = Path.home() / _DEFAULT_DATA_ROOT_NAME
    theme: str = "dark"
    recent_projects: List[Path] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# PreferenceManager
# ---------------------------------------------------------------------------


def _get_preferences_path() -> Path:
    """Return the absolute path to the preferences TOML file.

    The path is computed using ``platformdirs.user_config_dir`` so that
    each operating system uses its standard configuration directory:

    - macOS:   ``~/Library/Application Support/EchoInsight/preferences.toml``
    - Linux:   ``~/.config/echoinsight/preferences.toml``
    - Windows: ``%APPDATA%/EchoInsight/preferences.toml``
    """
    config_dir = Path(
        platformdirs.user_config_dir(appname=_CONFIG_DIR_NAME, ensure_exists=True)
    )
    return config_dir / _PREFERENCES_FILENAME


def _generate_default_toml() -> str:
    """Return a human-readable TOML string with default preferences.

    The output includes inline comments explaining every setting.
    """
    doc = tomlkit.document()

    # --- Version -----------------------------------------------------------
    doc.add("version", _PREFERENCES_VERSION)
    doc.add(
        tomlkit.comment(
            "Schema version.  Incremented when the preferences format changes.\n"
            "# EchoInsight checks this on load and migrates automatically."
        )
    )
    doc.add(tomlkit.nl())

    # --- Data root ---------------------------------------------------------
    default_root = str(Path.home() / _DEFAULT_DATA_ROOT_NAME)
    doc.add("data_root", default_root)
    doc.add(
        tomlkit.comment(
            "Absolute path to the root data directory.\n"
            "# All EchoInsight data (raw audio, SongDNA, frames, stems, cache,\n"
            "# exports, logs, models, spotify cache, projects) lives under this\n"
            "# directory in sub-folders.  Use an absolute path; ~ is not expanded."
        )
    )
    doc.add(tomlkit.nl())

    # --- Theme -------------------------------------------------------------
    doc.add("theme", "dark")
    doc.add(
        tomlkit.comment(
            'UI theme.  Valid values: "light", "dark", "system".\n'
            '# "system" follows the OS-level appearance setting.'
        )
    )
    doc.add(tomlkit.nl())

    # --- Recent projects ---------------------------------------------------
    projects = tomlkit.array()
    projects.multiline(True)
    doc.add("recent_projects", projects)
    doc.add(
        tomlkit.comment(
            "List of recently opened project directories, most recent first.\n"
            "# Managed automatically by the Workspace service."
        )
    )

    return tomlkit.dumps(doc)


class PreferenceManager:
    """Reads and writes ``UserPreferences`` to a TOML file on disk.

    The ``PreferenceManager`` is stateless — every call to ``load()``
    reads fresh data from disk, and ``save()`` writes the full state.

    Usage::

        pm = PreferenceManager()
        prefs = pm.load()       # auto-creates file if missing
        prefs.theme = "light"
        pm.save(prefs)

    Architectural note
    ------------------
    The manager does **not** cache preferences in memory.
    Callers (e.g. ``ApplicationContext``) are responsible for holding
    the ``UserPreferences`` instance in memory for the application
    lifetime.
    """

    def __init__(self) -> None:
        self._path: Path = _get_preferences_path()

    @property
    def path(self) -> Path:
        """Absolute path to the preferences TOML file."""
        return self._path

    def load(self) -> UserPreferences:
        """Load preferences from disk, creating a default file if missing.

        Returns
        -------
        UserPreferences
            The preferences read from disk, or the defaults if the file
            did not exist (in which case it is created).

        Raises
        ------
        tomlkit.exceptions.TOMLKitError
            If the file exists but is malformed.
        """
        if not self._path.exists():
            logger.info("Preferences file not found — creating default: %s", self._path)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            default_toml = _generate_default_toml()
            self._path.write_text(default_toml, encoding="utf-8")
            return self._preferences_from_dict(tomlkit.parse(default_toml))

        raw = self._path.read_text(encoding="utf-8")
        data = tomlkit.parse(raw)
        return self._preferences_from_dict(data)

    def save(self, preferences: UserPreferences) -> None:
        """Persist *preferences* to disk as TOML.

        Parameters
        ----------
        preferences : UserPreferences
            The preferences to save.
        """
        doc = tomlkit.document()
        doc["version"] = preferences.version
        doc["data_root"] = str(preferences.data_root)
        doc["theme"] = preferences.theme
        doc["recent_projects"] = tomlkit.array()
        for p in preferences.recent_projects:
            doc["recent_projects"].append(str(p))

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(tomlkit.dumps(doc), encoding="utf-8")
        logger.debug("Preferences saved to %s", self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preferences_from_dict(data: dict[str, Any]) -> UserPreferences:
        """Convert a parsed TOML dict into a ``UserPreferences`` instance.

        Missing keys are replaced with type-appropriate defaults to
        handle partial files (e.g. from older schema versions).
        """
        version = int(data.get("version", _PREFERENCES_VERSION))

        raw_root = data.get("data_root", str(Path.home() / _DEFAULT_DATA_ROOT_NAME))
        data_root = Path(raw_root).expanduser().resolve()

        theme = str(data.get("theme", "dark"))

        raw_projects = data.get("recent_projects", [])
        recent_projects: List[Path] = []
        for item in raw_projects:
            try:
                recent_projects.append(Path(str(item)).expanduser().resolve())
            except Exception:
                logger.warning("Ignoring invalid project path in preferences: %s", item)

        return UserPreferences(
            version=version,
            data_root=data_root,
            theme=theme,
            recent_projects=recent_projects,
        )
