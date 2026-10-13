"""
track_metadata.py — Build honest track metadata for a SongPackage.

Metadata comes from the audio file's own tags.  An optional catalogue
provider (for example a future Spotify client) may fill fields the tags
leave empty.  Nothing is ever invented: a field that no source supplies is
stored as unknown.

No catalogue provider is implemented yet, and nothing in the application
calls Spotify.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

import soundfile as sf

from src.core.package import EnrichedSongPackage, SongPackage
from src.core.spotify import SpotifyMetadata
from src.core.track_info import SourcedText, TrackInfo

logger = logging.getLogger(__name__)


class CatalogueProvider(Protocol):
    """Interface for an external catalogue lookup (e.g. a Spotify client).

    A provider returns basic catalogue metadata for a track, or ``None``
    when it finds no confident match.  It must never return placeholder
    values.

    Note for a future Spotify provider: Spotify restricted its
    audio-features and audio-analysis Web API endpoints in late 2024, so a
    provider should expect only basic catalogue metadata (title, artist,
    album, release date, IDs) — not tempo, key or other acoustic data.
    """

    name: str

    def lookup(self, track: TrackInfo, duration: float) -> Optional[SpotifyMetadata]:
        """Return catalogue metadata for *track*, or ``None`` if not found."""
        ...


def _tag(value: Optional[str]) -> SourcedText:
    """Wrap a raw tag string, treating empty strings as missing."""
    if value is None or not value.strip():
        return SourcedText()
    return SourcedText(value=value.strip(), source="file_tags")


def read_file_tags(audio_path: str | Path) -> TrackInfo:
    """Read title, artist, album and date tags from an audio file.

    Uses libsndfile (via ``soundfile``), which reads tags from WAV, FLAC,
    OGG and MP3 files.  Formats it cannot open (e.g. M4A) and files without
    tags yield all-unknown fields.
    """
    try:
        with sf.SoundFile(str(audio_path)) as f:
            tags = f.copy_metadata()
    except Exception as err:
        logger.info("No readable tags in %s: %s", audio_path, err)
        return TrackInfo()

    return TrackInfo(
        title=_tag(tags.get("title")),
        artist=_tag(tags.get("artist")),
        album=_tag(tags.get("album")),
        release_date=_tag(tags.get("date")),
    )


def _fill(current: SourcedText, candidate: Optional[str]) -> SourcedText:
    """Keep a known value; otherwise take a catalogue value if present."""
    if current.is_known or candidate is None or not candidate.strip():
        return current
    return SourcedText(value=candidate.strip(), source="spotify")


def build_enriched_package(
    package: SongPackage,
    audio_path: str | Path,
    provider: Optional[CatalogueProvider] = None,
) -> EnrichedSongPackage:
    """Attach file-tag metadata, and catalogue metadata if a provider matches.

    File tags take precedence; a provider only fills fields the tags leave
    unknown.  Without a provider, ``spotify`` is ``None``.
    """
    info = read_file_tags(audio_path)
    catalogue: Optional[SpotifyMetadata] = None

    if provider is not None:
        try:
            catalogue = provider.lookup(info, package.song_dna.metadata.duration)
        except Exception as err:
            logger.warning("Catalogue provider %s failed: %s", provider.name, err)
            catalogue = None

    if catalogue is not None:
        info = TrackInfo(
            title=_fill(info.title, catalogue.track_name),
            artist=_fill(info.artist, catalogue.artist_name),
            album=_fill(info.album, catalogue.album_name),
            release_date=_fill(info.release_date, catalogue.release_date),
        )

    return EnrichedSongPackage(package=package, spotify=catalogue, track_info=info)
