"""
EchoInsight Application Layer.

This package provides runtime application state via ``ApplicationContext``.

Layer rules
-----------
- ``src.app`` is allowed to import from ``src.config``, ``src.core``,
  and ``src.analysis``.
- ``src.app`` must **never** be imported by ``src.core`` or ``src.config``.
"""

from .context import ApplicationContext

__all__ = [
    "ApplicationContext",
]
