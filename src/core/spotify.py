"""
spotify.py — Immutable domain models for external Spotify metadata enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SpotifyMetadata:
    """Immutable external metadata fetched from Spotify catalog.

    Only contains catalog information — zero duplicated DSP or acoustic metrics.
    """
    track_id: str
    track_name: str
    artist_name: str
    album_name: str
    album_art_url: Optional[str] = None
    release_date: Optional[str] = None
    genres: Tuple[str, ...] = ()
    spotify_url: Optional[str] = None
    popularity: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.track_id.strip():
            raise ValueError("track_id must be non-empty")
        if not self.track_name.strip():
            raise ValueError("track_name must be non-empty")
        if not self.artist_name.strip():
            raise ValueError("artist_name must be non-empty")
        if self.popularity is not None and not (0 <= self.popularity <= 100):
            raise ValueError(f"popularity must be in [0, 100], got {self.popularity}")