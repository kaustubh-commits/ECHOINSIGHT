"""
package.py — SongPackage and EnrichedSongPackage aggregate roots for EchoInsight.

Encapsulates:
- Domain features & structural segments (SongDNA)
- Separated stem waveform references (SeparatedTrack)
- Stem acoustic analysis (StemAnalysisResult)
- External catalog enrichment (SpotifyMetadata)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from src.core.song_dna import SongDNA
from src.core.spotify import SpotifyMetadata
from src.core.stems import SeparatedTrack

if TYPE_CHECKING:
    from src.analysis.stem_analyzer import StemAnalysisResult


@dataclass(frozen=True)
class SongPackage:
    """Unified container for a fully analyzed track."""
    song_dna: SongDNA
    separated_track: Optional[SeparatedTrack] = None
    stem_analysis: Optional[StemAnalysisResult] = None

    @property
    def has_stems(self) -> bool:
        """Indicates whether audio stem paths are populated."""
        return self.separated_track is not None


@dataclass(frozen=True)
class EnrichedSongPackage:
    """Fully analyzed track bundled with optional Spotify catalog metadata."""
    package: SongPackage
    spotify: Optional[SpotifyMetadata] = None

    @property
    def is_enriched(self) -> bool:
        """Indicates whether Spotify catalog metadata is attached."""
        return self.spotify is not None

    @property
    def has_stems(self) -> bool:
        """Delegates stem availability check to underlying SongPackage."""
        return self.package.has_stems

    @property
    def song_dna(self) -> SongDNA:
        """Convenience property delegating to underlying SongDNA."""
        return self.package.song_dna

    @property
    def separated_track(self) -> Optional[SeparatedTrack]:
        """Convenience property delegating to separated stem track."""
        return self.package.separated_track

    @property
    def stem_analysis(self) -> Optional[StemAnalysisResult]:
        """Convenience property delegating to per-stem analysis."""
        return self.package.stem_analysis