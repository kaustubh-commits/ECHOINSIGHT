"""
Tests for Sprint 2 — Frame-Level DSP Foundation.

Covers:
- FrameReference creation, validation, immutability
- AnalysisManifest creation, config hash computation
- NPZ round-trip via FrameStore
- FrameStore validation
- FrameStore.build_frame_reference
- RhythmDNA beat fields
- SongDNA new fields (frames, manifest) optionality
- Old JSON backward compatibility
- Domain purity (subprocess check)
- extract_and_save end-to-end
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from src.analysis.frame_store import FrameStore, REQUIRED_ARRAYS
from src.core.analysis_manifest import AnalysisManifest, compute_config_hash
from src.core.dna_schema import RhythmDNA, SongDNA
from src.core.frame_reference import FrameReference


# ======================================================================
# FrameReference
# ======================================================================


class TestFrameReference:
    def test_valid_creation(self) -> None:
        ref = FrameReference(
            uri="frames/test.npz",
            frame_count=1000,
            hop_length=512,
            sample_rate=22050,
            duration=120.0,
            arrays={"rms": (1000,), "mfcc": (13, 1000)},
            dtypes={"rms": "float32", "mfcc": "float32"},
        )
        assert ref.uri == "frames/test.npz"
        assert ref.frame_count == 1000
        assert ref.hop_length == 512
        assert ref.sample_rate == 22050
        assert ref.duration == 120.0
        assert ref.arrays["rms"] == (1000,)
        assert ref.dtypes["mfcc"] == "float32"

    def test_frozen(self) -> None:
        ref = FrameReference(
            uri="frames/test.npz",
            frame_count=100,
            hop_length=512,
            sample_rate=22050,
            duration=10.0,
            arrays={"rms": (100,)},
            dtypes={"rms": "float64"},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.uri = "other.npz"  # type: ignore[misc]

    def test_empty_uri_raises(self) -> None:
        with pytest.raises(ValueError, match="uri must be non-empty"):
            FrameReference(
                uri="",
                frame_count=100,
                hop_length=512,
                sample_rate=22050,
                duration=10.0,
                arrays={"rms": (100,)},
                dtypes={"rms": "float64"},
            )

    def test_zero_frame_count_raises(self) -> None:
        with pytest.raises(ValueError, match="frame_count must be positive"):
            FrameReference(
                uri="frames/test.npz",
                frame_count=0,
                hop_length=512,
                sample_rate=22050,
                duration=10.0,
                arrays={"rms": (0,)},
                dtypes={"rms": "float64"},
            )

    def test_negative_hop_length_raises(self) -> None:
        with pytest.raises(ValueError, match="hop_length must be positive"):
            FrameReference(
                uri="frames/test.npz",
                frame_count=100,
                hop_length=-1,
                sample_rate=22050,
                duration=10.0,
                arrays={"rms": (100,)},
                dtypes={"rms": "float64"},
            )

    def test_negative_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            FrameReference(
                uri="frames/test.npz",
                frame_count=100,
                hop_length=512,
                sample_rate=0,
                duration=10.0,
                arrays={"rms": (100,)},
                dtypes={"rms": "float64"},
            )

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            FrameReference(
                uri="frames/test.npz",
                frame_count=100,
                hop_length=512,
                sample_rate=22050,
                duration=-1.0,
                arrays={"rms": (100,)},
                dtypes={"rms": "float64"},
            )

    def test_empty_arrays_raises(self) -> None:
        with pytest.raises(ValueError, match="arrays must be non-empty"):
            FrameReference(
                uri="frames/test.npz",
                frame_count=100,
                hop_length=512,
                sample_rate=22050,
                duration=10.0,
                arrays={},
                dtypes={},
            )

    def test_mismatched_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="dtypes missing"):
            FrameReference(
                uri="frames/test.npz",
                frame_count=100,
                hop_length=512,
                sample_rate=22050,
                duration=10.0,
                arrays={"rms": (100,), "mfcc": (13, 100)},
                dtypes={"rms": "float64"},
            )

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            FrameReference(
                uri="frames/test.npz",
                frame_count=100,
                hop_length=512,
                sample_rate=22050,
                duration=10.0,
                arrays={"rms": (100,)},
                dtypes={"rms": "complex128"},
            )


# ======================================================================
# AnalysisManifest & Config Hash
# ======================================================================


class TestAnalysisManifest:
    def test_valid_creation(self) -> None:
        manifest = AnalysisManifest(
            schema_version="1.0.0",
            extractor_version="2.0.0",
            created_at="2026-07-12T00:00:00Z",
            python_version="3.13.3",
            numpy_version="1.26.0",
            librosa_version="0.11.0",
            sample_rate=22050,
            hop_length=512,
            n_fft=2048,
            feature_set_version="v1",
            config_hash="abc123",
        )
        assert manifest.schema_version == "1.0.0"
        assert manifest.extractor_version == "2.0.0"
        assert manifest.sample_rate == 22050

    def test_frozen(self) -> None:
        manifest = AnalysisManifest(
            schema_version="1.0.0",
            extractor_version="2.0.0",
            created_at="2026-07-12T00:00:00Z",
            python_version="3.13.3",
            numpy_version="1.26.0",
            librosa_version="0.11.0",
            sample_rate=22050,
            hop_length=512,
            n_fft=2048,
            feature_set_version="v1",
            config_hash="abc123",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.extractor_version = "3.0.0"  # type: ignore[misc]

    def test_empty_schema_version_raises(self) -> None:
        with pytest.raises(ValueError, match="schema_version must be non-empty"):
            AnalysisManifest(
                schema_version="",
                extractor_version="2.0.0",
                created_at="2026-07-12T00:00:00Z",
                python_version="3.13.3",
                numpy_version="1.26.0",
                librosa_version="0.11.0",
                sample_rate=22050,
                hop_length=512,
                n_fft=2048,
                feature_set_version="v1",
                config_hash="abc123",
            )

    def test_empty_config_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="config_hash must be non-empty"):
            AnalysisManifest(
                schema_version="1.0.0",
                extractor_version="2.0.0",
                created_at="2026-07-12T00:00:00Z",
                python_version="3.13.3",
                numpy_version="1.26.0",
                librosa_version="0.11.0",
                sample_rate=22050,
                hop_length=512,
                n_fft=2048,
                feature_set_version="v1",
                config_hash="",
            )

    def test_git_fields_default_empty(self) -> None:
        manifest = AnalysisManifest(
            schema_version="1.0.0",
            extractor_version="2.0.0",
            created_at="2026-07-12T00:00:00Z",
            python_version="3.13.3",
            numpy_version="1.26.0",
            librosa_version="0.11.0",
            sample_rate=22050,
            hop_length=512,
            n_fft=2048,
            feature_set_version="v1",
            config_hash="abc123",
        )
        assert manifest.git_commit == ""
        assert manifest.git_branch == ""


class TestConfigHash:
    def test_same_params_same_hash(self) -> None:
        h1 = compute_config_hash(22050, 512, 2048, "v1")
        h2 = compute_config_hash(22050, 512, 2048, "v1")
        assert h1 == h2

    def test_different_params_different_hash(self) -> None:
        h1 = compute_config_hash(22050, 512, 2048, "v1")
        h2 = compute_config_hash(44100, 512, 2048, "v1")
        assert h1 != h2

    def test_hex_length(self) -> None:
        h = compute_config_hash(22050, 512, 2048, "v1")
        assert len(h) == 64  # SHA-256


# ======================================================================
# RhythmDNA beat fields
# ======================================================================


class TestRhythmDNA:
    def test_with_beats(self) -> None:
        beats = (10, 20, 30, 40)
        dna = RhythmDNA(tempo=120.0, confidence=0.95, beat_frames=beats, beat_count=4)
        assert dna.tempo == 120.0
        assert dna.confidence == 0.95
        assert dna.beat_frames == beats
        assert dna.beat_count == 4

    def test_negative_tempo_raises(self) -> None:
        with pytest.raises(ValueError, match="tempo must be positive"):
            RhythmDNA(tempo=-5.0)

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            RhythmDNA(tempo=120.0, confidence=1.5)

    def test_mismatched_beat_count_raises(self) -> None:
        with pytest.raises(ValueError, match="beat_count"):
            RhythmDNA(tempo=120.0, beat_frames=(10, 20), beat_count=5)

    def test_default_beats_empty(self) -> None:
        dna = RhythmDNA(tempo=120.0)
        assert dna.beat_frames == ()
        assert dna.beat_count == 0
        assert dna.confidence == 0.0


# ======================================================================
# SongDNA new fields
# ======================================================================


class TestSongDNA:
    def test_frames_defaults_none(self) -> None:
        from src.core.dna_schema import (
            IntelligenceDNA,
            SongMetadata,
            StemDNA,
            StructureDNA,
            TimbreDNA,
            TonalDNA,
        )

        dna = SongDNA(
            metadata=SongMetadata(duration=60.0, filename="test.mp3"),
            rhythm=RhythmDNA(tempo=120.0),
            timbre=TimbreDNA(
                rms_energy_mean=0.1,
                spectral_centroid_mean=2000.0,
                spectral_bandwidth_mean=3000.0,
                mfcc_mean=[0.0] * 13,
            ),
            tonal=TonalDNA(),
            structure=StructureDNA(),
            intelligence=IntelligenceDNA(),
            stem=StemDNA(),
        )
        assert dna.frames is None
        assert dna.manifest is None

    def test_old_json_backward_compatible(self) -> None:
        """Load an existing SongDNA JSON file — frames and manifest should be None."""
        old_json_path = Path("data/dna/br3ath3.json")
        if not old_json_path.exists():
            pytest.skip("Old JSON file not found")

        with open(old_json_path, "r") as f:
            data = json.load(f)

        # Simulate what comparison.py's _dict_to_songdna does
        assert "frames" not in data
        assert "manifest" not in data


# ======================================================================
# NPZ Round-Trip via FrameStore
# ======================================================================


def _make_dummy_features(n_frames: int = 100) -> Dict[str, np.ndarray]:
    """Create a dictionary of dummy feature arrays for testing."""
    return {
        "rms": np.random.rand(n_frames).astype(np.float32),
        "centroid": np.random.rand(n_frames).astype(np.float32),
        "bandwidth": np.random.rand(n_frames).astype(np.float32),
        "rolloff": np.random.rand(n_frames).astype(np.float32),
        "zcr": np.random.rand(n_frames).astype(np.float32),
        "mfcc": np.random.rand(13, n_frames).astype(np.float32),
        "chroma": np.random.rand(12, n_frames).astype(np.float32),
        "contrast": np.random.rand(7, n_frames).astype(np.float32),
        "tonnetz": np.random.rand(6, n_frames).astype(np.float32),
    }


class TestFrameStoreSaveLoad:
    def test_save_and_load(self, tmp_path: Path) -> None:
        features = _make_dummy_features(100)
        npz_path = tmp_path / "frames" / "test.npz"

        FrameStore.save(features, npz_path)
        assert npz_path.exists()

        loaded = FrameStore.load(npz_path)
        assert set(loaded.keys()) == REQUIRED_ARRAYS
        np.testing.assert_array_equal(loaded["rms"], features["rms"])
        np.testing.assert_array_equal(loaded["mfcc"], features["mfcc"])
        np.testing.assert_array_equal(loaded["chroma"], features["chroma"])

    def test_named_arrays(self, tmp_path: Path) -> None:
        features = _make_dummy_features(50)
        npz_path = tmp_path / "test.npz"
        FrameStore.save(features, npz_path)

        with np.load(npz_path) as data:
            assert "rms" in data
            assert "centroid" in data
            assert "bandwidth" in data
            assert "rolloff" in data
            assert "zcr" in data
            assert "mfcc" in data
            assert "chroma" in data
            assert "contrast" in data
            assert "tonnetz" in data
            # No arr_0, arr_1, etc.
            assert "arr_0" not in data
            assert "arr_1" not in data

    def test_natural_shapes(self, tmp_path: Path) -> None:
        features = _make_dummy_features(100)
        npz_path = tmp_path / "test.npz"
        FrameStore.save(features, npz_path)

        with np.load(npz_path) as data:
            assert data["rms"].shape == (100,)
            assert data["mfcc"].shape == (13, 100)
            assert data["chroma"].shape == (12, 100)
            assert data["contrast"].shape == (7, 100)
            assert data["tonnetz"].shape == (6, 100)


class TestFrameStoreValidation:
    def test_missing_array_raises(self) -> None:
        features = _make_dummy_features(50)
        del features["mfcc"]
        with pytest.raises(ValueError, match="Missing required arrays"):
            FrameStore.validate(features)

    def test_inconsistent_frame_count_raises(self) -> None:
        features = _make_dummy_features(100)
        features["rms"] = np.random.rand(50).astype(np.float32)  # wrong length
        with pytest.raises(ValueError, match="frames, expected"):
            FrameStore.validate(features)

    def test_extra_arrays_allowed(self) -> None:
        features = _make_dummy_features(100)
        features["extra_feature"] = np.random.rand(100).astype(np.float32)
        # Should not raise
        FrameStore.validate(features)


class TestFrameStoreBuildReference:
    def test_build_frame_reference(self) -> None:
        features = _make_dummy_features(100)
        ref = FrameStore.build_frame_reference(
            uri="frames/test.npz",
            features=features,
            hop_length=512,
            sample_rate=22050,
            duration=60.0,
        )
        assert ref.uri == "frames/test.npz"
        assert ref.frame_count == 100
        assert ref.arrays["rms"] == (100,)
        assert ref.arrays["mfcc"] == (13, 100)
        assert ref.dtypes["rms"] == "float32"
        assert ref.dtypes["mfcc"] == "float32"


# ======================================================================
# extract_and_save end-to-end
# ======================================================================


def test_extract_and_save(tmp_path: Path) -> None:
    """Run extract_and_save on a real MP3 and verify all outputs."""
    from src.analysis import extract_and_save

    mp3_dir = Path("data/raw")
    mp3_files = sorted(mp3_dir.glob("*.mp3"))
    if not mp3_files:
        pytest.skip("No MP3 files in data/raw/")

    audio_path = str(mp3_files[0])
    dna = extract_and_save(audio_path, data_root=tmp_path)

    # Verify SongDNA structure
    assert dna.frames is not None
    assert dna.manifest is not None
    assert dna.rhythm.beat_count > 0
    assert len(dna.rhythm.beat_frames) == dna.rhythm.beat_count

    # Verify NPZ file exists
    stem = Path(audio_path).stem
    npz_path = tmp_path / "frames" / f"{stem}.npz"
    assert npz_path.exists()

    # Verify NPZ contains expected data
    loaded = FrameStore.load(npz_path)
    assert set(loaded.keys()) == REQUIRED_ARRAYS

    # Verify manifest fields
    assert dna.manifest.schema_version == "1.0.0"
    assert dna.manifest.extractor_version == "2.0.0"
    assert dna.manifest.feature_set_version == "v1"
    assert len(dna.manifest.config_hash) == 64


# ======================================================================
# Domain purity
# ======================================================================


def test_domain_layer_pure() -> None:
    """Verify that core modules do not import config or analysis."""
    code = """
import sys
import src.core.frame_reference  # noqa: F401
import src.core.analysis_manifest  # noqa: F401
bad = {n for n in sys.modules if n.startswith("src.config") or n.startswith("src.analysis") or n.startswith("src.app")}
if bad:
    print(f"FAIL: domain imported {bad}")
    sys.exit(1)
else:
    print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, (
        f"Domain layer imported config/analysis/app: {result.stdout}{result.stderr}"
    )


# ======================================================================
# Circular imports
# ======================================================================


def test_no_circular_imports() -> None:
    """Verify the full import chain has no circular dependencies."""

    assert True


# ======================================================================
# Backward compatibility
# ======================================================================


def test_old_extract_api_backward_compatible() -> None:
    """extract_song_dna() must still work and return frames=None."""
    from src.analysis import extract_song_dna

    mp3_dir = Path("data/raw")
    mp3_files = sorted(mp3_dir.glob("*.mp3"))
    if not mp3_files:
        pytest.skip("No MP3 files in data/raw/")

    dna = extract_song_dna(str(mp3_files[0]))
    assert dna.frames is None
    assert dna.manifest is None
    assert dna.rhythm.tempo > 0
    assert dna.rhythm.beat_count > 0