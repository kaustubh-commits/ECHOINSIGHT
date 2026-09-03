"""
package_store.py — Persistence manager for SongPackage and EnrichedSongPackage aggregates.

Follows clean storage separation:
- Metadata & Contracts: JSON (data/dna/{song_id}.json)
- Frame Data: NPZ (data/frames/{stem}.npz)
- Stem Waveforms: WAV (data/stems/{track_id}/*.wav)
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.analysis.stem_analyzer import StemAnalysisResult
from src.core.package import EnrichedSongPackage, SongPackage
from src.core.song_dna import SongDNA, _reconstruct_dataclass
from src.core.spotify import SpotifyMetadata
from src.core.stems import SeparatedTrack, StemPaths


class PackageStore:
    """Handles serialization and deserialization of unified SongPackages."""

    @staticmethod
    def to_dict(package: SongPackage | EnrichedSongPackage) -> Dict[str, Any]:
        """Converts a SongPackage or EnrichedSongPackage to a JSON dictionary."""
        base_pkg = package.package if isinstance(package, EnrichedSongPackage) else package
        spotify_meta = package.spotify if isinstance(package, EnrichedSongPackage) else None

        data: Dict[str, Any] = {
            "schema_version": "2.1.0",
            "song_dna": dataclasses.asdict(base_pkg.song_dna),
            "separated_track": None,
            "stem_analysis": None,
            "spotify_metadata": None,
        }

        if base_pkg.separated_track:
            st = base_pkg.separated_track
            data["separated_track"] = {
                "track_id": st.track_id,
                "original_audio": str(st.original_audio),
                "sample_rate": st.sample_rate,
                "duration_seconds": st.duration_seconds,
                "model_name": st.model_name,
                "stems": {
                    "vocals": str(st.stems.vocals),
                    "drums": str(st.stems.drums),
                    "bass": str(st.stems.bass),
                    "other": str(st.stems.other),
                },
            }

        if base_pkg.stem_analysis:
            sa = base_pkg.stem_analysis
            data["stem_analysis"] = {
                "tempo_bpm": sa.tempo_bpm,
                "beat_frames": sa.beat_frames,
                "vocal_energy_ratio": sa.vocal_energy_ratio,
                "bass_energy_ratio": sa.bass_energy_ratio,
                "drums_energy_ratio": sa.drums_energy_ratio,
                "other_energy_ratio": sa.other_energy_ratio,
                "per_stem_rms": sa.per_stem_rms,
            }

        if spotify_meta:
            data["spotify_metadata"] = {
                "track_id": spotify_meta.track_id,
                "track_name": spotify_meta.track_name,
                "artist_name": spotify_meta.artist_name,
                "album_name": spotify_meta.album_name,
                "album_art_url": spotify_meta.album_art_url,
                "release_date": spotify_meta.release_date,
                "genres": list(spotify_meta.genres),
                "spotify_url": spotify_meta.spotify_url,
                "popularity": spotify_meta.popularity,
            }

        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> EnrichedSongPackage:
        """Constructs an EnrichedSongPackage from a parsed dictionary.

        Supports both wrapped package dicts and raw standalone SongDNA JSONs.
        """
        # Determine if payload is wrapped or direct SongDNA
        if "song_dna" in data and isinstance(data["song_dna"], dict):
            dna_dict = data["song_dna"]
        elif "metadata" in data and "structure" in data:
            dna_dict = data
        else:
            raise KeyError("JSON missing 'song_dna' or root 'metadata'/'structure' keys")

        song_dna = _reconstruct_dataclass(SongDNA, dna_dict)

        separated_track: Optional[SeparatedTrack] = None
        if data.get("separated_track"):
            st_data = data["separated_track"]
            stem_paths = StemPaths(
                vocals=Path(st_data["stems"]["vocals"]),
                drums=Path(st_data["stems"]["drums"]),
                bass=Path(st_data["stems"]["bass"]),
                other=Path(st_data["stems"]["other"]),
            )
            separated_track = SeparatedTrack(
                track_id=st_data["track_id"],
                original_audio=Path(st_data["original_audio"]),
                sample_rate=st_data["sample_rate"],
                duration_seconds=st_data["duration_seconds"],
                stems=stem_paths,
                model_name=st_data.get("model_name", "htdemucs"),
            )

        stem_analysis: Optional[StemAnalysisResult] = None
        if data.get("stem_analysis"):
            sa_data = data["stem_analysis"]
            stem_analysis = StemAnalysisResult(
                tempo_bpm=sa_data["tempo_bpm"],
                beat_frames=sa_data["beat_frames"],
                vocal_energy_ratio=sa_data["vocal_energy_ratio"],
                bass_energy_ratio=sa_data["bass_energy_ratio"],
                drums_energy_ratio=sa_data["drums_energy_ratio"],
                other_energy_ratio=sa_data["other_energy_ratio"],
                per_stem_rms=sa_data["per_stem_rms"],
            )

        base_pkg = SongPackage(
            song_dna=song_dna,
            separated_track=separated_track,
            stem_analysis=stem_analysis,
        )

        spotify_meta: Optional[SpotifyMetadata] = None
        if data.get("spotify_metadata"):
            sm = data["spotify_metadata"]
            spotify_meta = SpotifyMetadata(
                track_id=sm["track_id"],
                track_name=sm["track_name"],
                artist_name=sm["artist_name"],
                album_name=sm["album_name"],
                album_art_url=sm.get("album_art_url"),
                release_date=sm.get("release_date"),
                genres=tuple(sm.get("genres", ())),
                spotify_url=sm.get("spotify_url"),
                popularity=sm.get("popularity"),
            )

        return EnrichedSongPackage(
            package=base_pkg,
            spotify=spotify_meta,
        )

    @classmethod
    def save(cls, package: SongPackage | EnrichedSongPackage, output_path: Path) -> None:
        """Persists package metadata to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = cls.to_dict(package)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, json_path: Path) -> EnrichedSongPackage:
        """Loads and reconstructs an EnrichedSongPackage from JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)