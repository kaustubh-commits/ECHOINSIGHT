"""
package_store.py — Persistence manager for SongPackage aggregates.

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
from src.core.package import SongPackage
from src.core.song_dna import SongDNA, _reconstruct_dataclass
from src.core.stems import SeparatedTrack, StemPaths


class PackageStore:
    """Handles serialization and deserialization of unified SongPackages."""

    @staticmethod
    def to_dict(package: SongPackage) -> Dict[str, Any]:
        """Converts a SongPackage to a JSON-serializable dictionary."""
        data: Dict[str, Any] = {
            "schema_version": "2.1.0",
            "song_dna": dataclasses.asdict(package.song_dna),
            "separated_track": None,
            "stem_analysis": None,
        }

        if package.separated_track:
            st = package.separated_track
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

        if package.stem_analysis:
            sa = package.stem_analysis
            data["stem_analysis"] = {
                "tempo_bpm": sa.tempo_bpm,
                "beat_frames": sa.beat_frames,
                "vocal_energy_ratio": sa.vocal_energy_ratio,
                "bass_energy_ratio": sa.bass_energy_ratio,
                "drums_energy_ratio": sa.drums_energy_ratio,
                "other_energy_ratio": sa.other_energy_ratio,
                "per_stem_rms": sa.per_stem_rms,
            }

        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SongPackage:
        """Constructs a SongPackage from a parsed dictionary."""
        song_dna = _reconstruct_dataclass(SongDNA, data["song_dna"])

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

        return SongPackage(
            song_dna=song_dna,
            separated_track=separated_track,
            stem_analysis=stem_analysis,
        )

    @classmethod
    def save(cls, package: SongPackage, output_path: Path) -> None:
        """Persists package metadata to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = cls.to_dict(package)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, json_path: Path) -> SongPackage:
        """Loads and reconstructs a SongPackage from JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)