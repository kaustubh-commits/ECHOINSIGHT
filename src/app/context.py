"""
context.py — Runtime application context for EchoInsight.

``ApplicationContext`` is the top-level runtime state container.
It holds configuration, user preferences, the logger, and future
infrastructure services (DatasetManager, Workspace, etc.).

Architectural role
------------------
``ApplicationContext`` sits at the **Application** layer.

- It receives dependencies (config, preferences) via constructor injection.
- It initialises the project logger.
- It exposes convenience methods like ``resolve_data_path()``.
- It acts as a **service container**: future services are composed
  as attributes, not stuffed into a god object.

What this class must NEVER become:
- A place for business logic (extraction, comparison, etc.)
- A global singleton (always constructed and passed explicitly)
- A place that imports domain types (SongDNA, etc.)
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.core import EchoInsightConfig, PathsConfig
from src.config.preferences import UserPreferences


class ApplicationContext:
    """Runtime application context that orchestrates configuration and
    infrastructure services.

    Parameters
    ----------
    config : EchoInsightConfig
        Immutable application defaults.
    preferences : UserPreferences
        Mutable user-editable preferences.
    log_level : int, optional
        Logging level for the ``echoinsight`` logger.
        Defaults to ``logging.INFO``.

    Attributes
    ----------
    config : EchoInsightConfig
        Immutable application defaults.
    preferences : UserPreferences
        Mutable user-editable preferences.
    paths : PathsConfig
        Centralised storage path layout derived from ``preferences.data_root``.
    logger : logging.Logger
        Project logger (name: ``"echoinsight"``).
    """

    def __init__(
        self,
        config: EchoInsightConfig,
        preferences: UserPreferences,
        log_level: int = logging.INFO,
    ) -> None:
        self.config = config
        self.preferences = preferences
        self.paths = PathsConfig(data_root=preferences.data_root)

        # Project logger — named, silent by default
        self._logger: logging.Logger = logging.getLogger("echoinsight")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s :: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self._logger.setLevel(log_level)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def logger(self) -> logging.Logger:
        """Project-scoped logger (``"echoinsight"``).

        Use this instead of ``logging.getLogger(__name__)`` when you
        need EchoInsight-specific logging behaviour.
        """
        return self._logger

    def resolve_data_path(self, relative_uri: str) -> Path:
        """Convert a domain-relative URI into an absolute filesystem path.

        The URI is resolved relative to ``self.preferences.data_root``.
        URIs are forward-slash separated and may include sub-directories.

        Parameters
        ----------
        relative_uri : str
            Domain-relative path, e.g. ``"frames/get_lucky.npz"`` or
            ``"dna/artist/album/song.json"``.

        Returns
        -------
        Path
            Absolute ``Path`` instance.

        Raises
        ------
        ValueError
            If *relative_uri* is absolute (starts with ``/``).

        Examples
        --------
        >>> ctx.resolve_data_path("frames/get_lucky.npz")
        PosixPath('/home/user/EchoInsightData/frames/get_lucky.npz')
        """
        path = Path(relative_uri)
        if path.is_absolute():
            raise ValueError(
                f"relative_uri must be a relative path, got: {relative_uri}"
            )
        return (self.preferences.data_root / path).resolve()

    def set_log_level(self, level: int) -> None:
        """Dynamically change the logger level at runtime.

        Parameters
        ----------
        level : int
            One of ``logging.DEBUG``, ``logging.INFO``, ``logging.WARNING``,
            ``logging.ERROR``, ``logging.CRITICAL``.
        """
        self._logger.setLevel(level)
