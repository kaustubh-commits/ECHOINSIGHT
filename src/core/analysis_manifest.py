"""
analysis_manifest.py — Record of how a SongDNA extraction was produced.

``AnalysisManifest`` guarantees dataset reproducibility by recording every
parameter and dependency version that influences extraction output.

Layer rules
-----------
- ``src.core.analysis_manifest`` imports **only** Python stdlib.
- It must **never** import ``numpy``, ``librosa``, ``src.analysis``,
  ``src.config``, or ``src.app``.
"""

from __future__ import annotations

import dataclasses
import hashlib


# ---------------------------------------------------------------------------
# AnalysisManifest
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AnalysisManifest:
    """Records the full provenance of a SongDNA extraction.

    Every extraction writes an ``AnalysisManifest`` so that future
    consumers can detect incompatible datasets, reproduce analyses,
    and diagnose regressions.

    The ``config_hash`` field is a deterministic SHA-256 digest of the
    extraction parameters.  Two manifests with the same ``config_hash``
    were produced with identical DSP settings; different hashes indicate
    incomparable feature sets.

    Attributes
    ----------
    schema_version : str
        Version of the ``AnalysisManifest`` schema itself, e.g. ``"1.0.0"``.
    extractor_version : str
        Version of the extraction pipeline that produced this manifest,
        e.g. ``"2.0.0"``.
    created_at : str
        ISO-8601 timestamp of the extraction.
    python_version : str
        Python version string (``sys.version``).
    numpy_version : str
        NumPy version string.
    librosa_version : str
        librosa version string.
    sample_rate : int
        Sample rate used for extraction (Hz).
    hop_length : int
        Hop length used for frame extraction (samples).
    n_fft : int
        FFT window size (samples).
    feature_set_version : str
        Version identifier for the set of features extracted.
        ``"v1"`` = RMS, centroid, bandwidth, rolloff, ZCR, MFCC,
        chroma, contrast, tonnetz.
    config_hash : str
        SHA-256 digest of canonical extraction parameters.
        Computed as ``sha256(f"{sample_rate}:{hop_length}:{n_fft}:{feature_set_version}")``.
    git_commit : str
        Git commit hash at extraction time, if available.
        Empty string if not in a git repository.
    git_branch : str
        Git branch name at extraction time, if available.
        Empty string if not in a git repository.
    """

    schema_version: str
    extractor_version: str
    created_at: str
    python_version: str
    numpy_version: str
    librosa_version: str
    sample_rate: int
    hop_length: int
    n_fft: int
    feature_set_version: str
    config_hash: str
    git_commit: str = ""
    git_branch: str = ""

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate}"
            )
        if self.hop_length <= 0:
            raise ValueError(
                f"hop_length must be positive, got {self.hop_length}"
            )
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if not self.schema_version:
            raise ValueError("schema_version must be non-empty")
        if not self.extractor_version:
            raise ValueError("extractor_version must be non-empty")
        if not self.created_at:
            raise ValueError("created_at must be non-empty")
        if not self.feature_set_version:
            raise ValueError("feature_set_version must be non-empty")
        if not self.config_hash:
            raise ValueError("config_hash must be non-empty")


# ---------------------------------------------------------------------------
# Config hash computation
# ---------------------------------------------------------------------------


def compute_config_hash(
    sample_rate: int, hop_length: int, n_fft: int,
    feature_set_version: str,
) -> str:
    """Deterministic SHA-256 digest of canonical extraction parameters.

    Parameters
    ----------
    sample_rate : int
        Target sample rate in Hz.
    hop_length : int
        Hop length in samples.
    n_fft : int
        FFT window size in samples.
    feature_set_version : str
        Version identifier for the feature set.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest (64 characters).
    """
    canonical = f"{sample_rate}:{hop_length}:{n_fft}:{feature_set_version}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()