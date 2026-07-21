"""
EchoInsight Configuration Layer.

This package provides the immutable application defaults and user-editable
preferences that together form EchoInsight's configuration subsystem.

Layer rules
-----------
- ``src.config`` imports **only** Python stdlib + external packages (tomlkit, platformdirs).
- ``src.config`` must **never** import from ``src.core``, ``src.analysis``, or ``src.app``.
- ``src.config.core`` contains frozen dataclasses — values are never mutated.
- ``src.config.preferences`` manages a mutable user-editable TOML file on disk.
"""

from .core import AudioSettings, EchoInsightConfig, PathsConfig
from .preferences import PreferenceManager, UserPreferences

__all__ = [
    "AudioSettings",
    "EchoInsightConfig",
    "PathsConfig",
    "PreferenceManager",
    "UserPreferences",
]
