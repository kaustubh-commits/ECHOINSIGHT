# Extraction Contract — SongDNA v2

> **This document defines the invariants and guarantees of the v2 extraction pipeline.**

---

## 1. Inputs

| Input | Type | Notes |
|-------|------|-------|
| Audio file | File path | Supported formats: MP3, WAV, FLAC, OGG, M4A |
| Sample rate | `22050 Hz` | **Locked.** `librosa.load(audio_path, sr=22050)`. Do not parameterize. |

Optional future input: 16-bit PCM.

---

## 2. Outputs

| Output | Type |
|--------|------|
| `SongDNA` v2 | `src/core/song_dna.py` aggregate root |
| Structure | `StructureDNA` + `SegmentDNA[]` (`src/core/structure.py`) |
| Frame-level data | NPZ via `FrameStore` (schema-agnostic) — only in `extract_and_save()` |
| Serialization | JSON via `SongDNA.to_json()` / `SongDNA.from_json()` |

---

## 3. Output Schema Version

Current versions (2026-10-13): `SongDNA` **2.2.0**, `SegmentDNA` **2.1.0**, `StructureDNA` **2.1.0**. Each aggregate is versioned separately (ADR-0007).

- 2.2.0 / 2.1.0 added `tempo_is_fallback`, made `tempo_confidence` optional (no confidence is computed, so it is `None`), and recorded `Timeline.downbeats_assumed` / `beats_per_bar` (bars are assumed 4/4, not detected).
- StructureDNA 2.1.0 made `BoundaryConfidence.confidence` optional; segmentation computes no confidence, so it is `None`.
- Older files still load; `migrate_song_dna_dict` converts their placeholder tempo confidences and the placeholder boundary confidence (0.7, "agglomerative") to `None`.

---

## 4. Invariants for `SegmentDNA[]`

| # | Invariant | Violation |
|---|-----------|-----------|
| 1 | Every segment satisfies `start_time < end_time` | `ValueError` at `SegmentTiming.__post_init__` |
| 2 | Segments are temporally contiguous: `segment[i].end_time == segment[i + 1].start_time` | `ValueError` at `StructureDNA.__post_init__` |
| 3 | Segments cover the entire track: `first.start_time == 0.0` and `last.end_time == duration` | `ValueError` at `StructureDNA.__post_init__` / `SongDNA.__post_init__` |
| 4 | `song_id` is consistent across all segments and matches `SongMetadata.song_id` | `ValueError` at `SongDNA.__post_init__` |
| 5 | `tempo > 0.0` | `ValueError` at `SegmentRhythm.__post_init__` |
| 6 | `confidence ∈ [0.0, 1.0]` for all confidence fields | `ValueError` at respective `__post_init__` |
| 7 | `StructureDNA.total_duration` matches `SongMetadata.duration` | `ValueError` at `SongDNA.__post_init__` (tolerance `1e-3`) |

---

## 5. Provenance

`AnalysisManifest` (v2) must record:

| Field | Source |
|-------|--------|
| `schema_version` | `"1.0.0"` (manifest schema) |
| `extractor_version` | extraction pipeline version |
| `created_at` | ISO-8601 UTC timestamp |
| `python_version` | `sys.version` |
| `numpy_version` | `numpy.__version__` |
| `librosa_version` | `librosa.__version__` |
| `sample_rate` | `22050` |
| `hop_length` | hop length used |
| `n_fft` | FFT window size used |
| `feature_set_version` | feature set identifier |
| `config_hash` | SHA-256 of canonical extraction parameters |
| `git_commit` / `git_branch` | git provenance (best-effort) |

---

## 6. Determinism Guarantee

The same input audio file must produce the exact same `SongDNA` output on every extraction run:

- Sample rate locked to `22050 Hz`.
- Deterministic DSP (librosa with fixed parameters, no RNG-dependent stages).
- Identical hop length / FFT configuration across runs.
- Provenance recorded so config drift is detectable via `config_hash`.

**Known deviation (not yet fixed):** `song_id` is a random UUID (`generate_song_id`) and `manifest.created_at` is the run time, so two runs on the same file differ in those fields. All measured values (tempo, key, segments, DSP) are deterministic.

---

## 7. Spotify Exclusion

The extraction engine is **100% offline**. No network calls, no Spotify integration, no external APIs. Spotify enrichment is a separate, optional pass built later and must never be imported by `extractor.py`.