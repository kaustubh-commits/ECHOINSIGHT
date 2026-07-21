"""
main.py — EchoInsight CLI entry point.

Usage:
    python main.py <path-to-audio-file>

Extracts the SongDNA fingerprint from the given audio file and
prints the result as indented JSON to stdout.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys

from src.analysis import extract_song_dna
from src.app.context import ApplicationContext
from src.config.core import EchoInsightConfig
from src.config.preferences import PreferenceManager


def bootstrap_application() -> ApplicationContext:
    """Initialise and return the EchoInsight runtime context.

    This is the single entry point for bootstrapping the application.
    It composes the three infrastructure layers and returns a fully
    initialised ``ApplicationContext``.

    Returns
    -------
    ApplicationContext
        Runtime context with config, preferences, logger, and path
        resolution wired together.
    """
    core_config = EchoInsightConfig()
    preference_manager = PreferenceManager()
    preferences = preference_manager.load()
    context = ApplicationContext(config=core_config, preferences=preferences)
    return context


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python main.py <path-to-audio-file>", file=sys.stderr)
        sys.exit(1)

    file_path: str = sys.argv[1]

    try:
        dna = extract_song_dna(file_path)
    except Exception as exc:
        print(f"Error extracting SongDNA: {exc}", file=sys.stderr)
        sys.exit(1)

    output: str = json.dumps(dataclasses.asdict(dna), indent=2)
    print(output)


if __name__ == "__main__":
    app = bootstrap_application()
    logging.getLogger("echoinsight").setLevel(logging.WARNING)
    main()
