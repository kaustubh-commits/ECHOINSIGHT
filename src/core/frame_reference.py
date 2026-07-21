"""
frame_reference.py — Immutable reference to binary frame-level data.

``FrameReference`` lives in the Domain layer.  It stores only structural
metadata about frame-level feature arrays: the relative URI where the
binary data lives, the frame count, feature shapes, and data types.

It never performs I/O and never contains actual array data.

Layer rules
-----------
- ``src.core.frame_reference`` imports **only** Python stdlib + ``src.core.identifiers``.
- It must **never** import ``numpy``, ``pathlib``, ``src.analysis``,
  ``src.config``, or ``src.app``.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

from src.core.identifiers import URI


# ---------------------------------------------------------------------------
# FrameReference
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FrameReference:
    """Immutable reference to binary frame-level feature data.

    The Domain layer stores only a relative URI and structural metadata.
    Absolute path resolution is handled by ``ApplicationContext``.
    Binary I/O is handled by ``FrameStore`` in the Infrastructure layer.

    Attributes
    ----------
    uri : URI
        Relative URI within the data root, e.g. ``"frames/song_id.npz"``.
        Typed as ``URI`` from ``src.core.identifiers``.
    frame_count : int
        Number of time frames in the referenced data.
    hop_length : int
        Hop length in samples used during extraction.
    sample_rate : int
        Sample rate in Hz used during extraction.
    duration : float
        Track duration in seconds.
    arrays : dict[str, tuple[int, ...]]
        Map from array name to its shape.
        Example: ``{"rms": (8421,), "mfcc": (13, 8421)}``.
    dtypes : dict[str, str]
        Map from array name to its NumPy dtype string.
        Example: ``{"rms": "float32", "mfcc": "float32"}``.

    Architectural notes
    -------------------
    -   Every array in *arrays* must have a matching entry in *dtypes*
        (same key set, no missing keys).
    -   ``frame_count`` is derived from the first dimension of the RMS
        array and must be consistent across all arrays.
    """

    uri: URI
    frame_count: int
    hop_length: int
    sample_rate: int
    duration: float
    arrays: Dict[str, Tuple[int, ...]]
    dtypes: Dict[str, str]

    _VALID_DTYPES: Tuple[str, ...] = (
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
    )

    def __post_init__(self) -> None:
        """Validate all fields at construction time."""
        if not self.uri:
            raise ValueError("uri must be non-empty")
        if self.frame_count <= 0:
            raise ValueError(
                f"frame_count must be positive, got {self.frame_count}"
            )
        if self.hop_length <= 0:
            raise ValueError(
                f"hop_length must be positive, got {self.hop_length}"
            )
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )
        if self.duration <= 0:
            raise ValueError(
                f"duration must be positive, got {self.duration}"
            )
        if not self.arrays:
            raise ValueError("arrays must be non-empty")
        if self.arrays.keys() != self.dtypes.keys():
            missing_arrays = set(self.dtypes) - set(self.arrays)
            missing_dtypes = set(self.arrays) - set(self.dtypes)
            msg_parts = []
            if missing_arrays:
                msg_parts.append(f"arrays missing keys: {missing_arrays}")
            if missing_dtypes:
                msg_parts.append(f"dtypes missing keys: {missing_dtypes}")
            raise ValueError("; ".join(msg_parts))

        for name, dtype_str in self.dtypes.items():
            if dtype_str not in self._VALID_DTYPES:
                raise ValueError(
                    f"Unsupported dtype for '{name}': {dtype_str}. "
                    f"Valid: {self._VALID_DTYPES}"
                )