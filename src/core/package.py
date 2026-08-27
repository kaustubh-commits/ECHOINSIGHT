"""
package.py — SongPackage aggregate root and data contracts for EchoInsight.

Encapsulates the complete track representation:
- Domain features & structural segments (SongDNA)
- Separated stem waveform references (SeparatedTrack)
- Stem acoustic analysis (StemAnalysisResult)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.analysis.stem_analyzer import StemAnalysisResult
from src.core.song_dna import SongDNA
from src.core.stems import SeparatedTrack


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