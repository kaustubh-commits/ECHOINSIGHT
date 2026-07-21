"""
frame_store.py — NPZ serialization and deserialization for frame-level data.

Infrastructure layer only.
``FrameStore`` knows about ``numpy``, the filesystem, and domain types
(``FrameReference``).  It never knows about ``SongDNA`` or any higher-level
business object.

Layer rules
-----------
- ``src.analysis.frame_store`` may import from ``src.core`` but **never**
  from ``src.app`` or ``src.config``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.core.frame_reference import FrameReference

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_ARRAYS: frozenset = frozenset({
    "rms",
    "centroid",
    "bandwidth",
    "rolloff",
    "zcr",
    "mfcc",
    "chroma",
    "contrast",
    "tonnetz",
})
"""Canonical set of array names required in every NPZ frame file."""


# ---------------------------------------------------------------------------
# FrameStore
# ---------------------------------------------------------------------------


class FrameStore:
    """Saves, loads, and validates NPZ frame-level feature data.

    All methods are static.  The class serves as a namespace for
    frame-storage logic with no mutable state.

    Usage::

        features = {
            "rms": np.array(...),
            "mfcc": np.array(...),
            ...
        }
        FrameStore.save(features, Path("/data/frames/song.npz"))
        loaded = FrameStore.load(Path("/data/frames/song.npz"))

    Architectural note
    ------------------
    ``FrameStore`` never references ``SongDNA``.  It converts between
    ``dict[str, np.ndarray]`` and ``FrameReference`` — nothing else.
    """

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    @staticmethod
    def save(features: Dict[str, np.ndarray], output_path: Path) -> None:
        """Validate and save feature arrays to a compressed NPZ file.

        Parameters
        ----------
        features : dict[str, np.ndarray]
            Dictionary of named feature arrays.  Must include all arrays
            in ``REQUIRED_ARRAYS`` with consistent frame counts.
        output_path : Path
            Absolute path for the output ``.npz`` file.  Parent directories
            are created automatically.

        Raises
        ------
        ValueError
            If *features* fails validation.
        """
        FrameStore.validate(features)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **features)
        logger.debug("Saved %d arrays to %s", len(features), output_path)

    @staticmethod
    def load(path: Path) -> Dict[str, np.ndarray]:
        """Load feature arrays from a compressed NPZ file.

        Parameters
        ----------
        path : Path
            Absolute path to an existing ``.npz`` file.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of named feature arrays.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the NPZ file is missing required arrays.
        """
        if not path.exists():
            raise FileNotFoundError(f"Frame file not found: {path}")

        with np.load(path) as data:
            # Copy into a plain dict so the file handle can close
            loaded: Dict[str, np.ndarray] = dict(data)

        FrameStore.validate(loaded)
        logger.debug("Loaded %d arrays from %s", len(loaded), path)
        return loaded

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate(features: Dict[str, np.ndarray]) -> None:
        """Verify that *features* contains all required arrays with
        consistent frame counts.

        Parameters
        ----------
        features : dict[str, np.ndarray]
            Dictionary of named feature arrays to validate.

        Raises
        ------
        ValueError
            If required arrays are missing or frame counts are inconsistent.
        """
        missing = REQUIRED_ARRAYS - set(features.keys())
        if missing:
            raise ValueError(
                f"Missing required arrays: {sorted(missing)}"
            )

        # Determine frame count from the RMS array (1-D, length = frames)
        rms_arr = features["rms"]
        if rms_arr.ndim != 1:
            raise ValueError(
                f"Expected rms to be 1-D, got shape {rms_arr.shape}"
            )
        n_frames = rms_arr.shape[0]

        for name, arr in features.items():
            if arr.ndim == 1:
                if arr.shape[0] != n_frames:
                    raise ValueError(
                        f"Array '{name}' has {arr.shape[0]} frames, "
                        f"expected {n_frames}"
                    )
            elif arr.ndim == 2:
                if arr.shape[1] != n_frames:
                    raise ValueError(
                        f"Array '{name}' has {arr.shape[1]} frames "
                        f"(axis=1), expected {n_frames}"
                    )
            else:
                raise ValueError(
                    f"Array '{name}' has unsupported ndim={arr.ndim}; "
                    f"expected 1 or 2"
                )

    # ------------------------------------------------------------------
    # FrameReference construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_frame_reference(
        uri: str,
        features: Dict[str, np.ndarray],
        hop_length: int,
        sample_rate: int,
        duration: float,
    ) -> FrameReference:
        """Construct a ``FrameReference`` from extracted features.

        Parameters
        ----------
        uri : str
            Relative URI for the NPZ file, e.g. ``"frames/song.npz"``.
        features : dict[str, np.ndarray]
            Dictionary of validated feature arrays.
        hop_length : int
            Hop length in samples used during extraction.
        sample_rate : int
            Sample rate in Hz used during extraction.
        duration : float
            Track duration in seconds.

        Returns
        -------
        FrameReference
            Immutable reference with shape and dtype metadata.
        """
        FrameStore.validate(features)
        n_frames = features["rms"].shape[0]

        arrays: Dict[str, Tuple[int, ...]] = {}
        dtypes: Dict[str, str] = {}

        for name, arr in features.items():
            arrays[name] = tuple(arr.shape)
            dtypes[name] = str(arr.dtype)

        return FrameReference(
            uri=uri,
            frame_count=n_frames,
            hop_length=hop_length,
            sample_rate=sample_rate,
            duration=duration,
            arrays=arrays,
            dtypes=dtypes,
        )