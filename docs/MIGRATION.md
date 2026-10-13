# MIGRATION: v1 → v2 Extraction Pipeline

> This document tracks the temporary compatibility path between the legacy v1 extraction pipeline (`src/core/dna_schema.py`) and the new v2 pipeline (`src/core/song_dna.py` + `src/core/structure.py`).

---

## Current State (checked 2026-10-13)

`_BUILD_V2` is **`True`** in `src/analysis/extractor.py`, so every extraction runs the v2 branch: structural segmentation (`segmentation.py`) and per-segment DSP (`segment_dsp.py`). The v1 branch (`_BUILD_V2 = False`), which builds a single global segment, is still in the file but is not used.

```python
_BUILD_V2 = True  # ← v2 pipeline enabled by default
```

The v1 sample files in `data/dna/` and the v1-only `explorer.py` / `visualization.py` tools were removed on 2026-10-13. They are archived outside the repo and remain in git history.

---

## Removal Criteria

The temporary extraction compatibility path (`_BUILD_V2` flag, the v1 branch, and `src/core/dna_schema.py`) will be removed when **all** of the following are true:

- [x] Workspace consumes SongDNA v2
- [x] Visualization consumes SongDNA v2: n/a, the v1-only `visualization.py` was removed
- [x] Comparison consumes SongDNA v2 (`src/analysis/comparison/engine.py`)
- [ ] All extraction tests pass using v2: they pass, but the two audio-fixture tests in `tests/test_extraction_v2.py` are skipped because `tests/fixtures/` has no audio
- [x] No production code imports `dna_schema.py` (only `tests/test_config.py` and `tests/test_frames.py` still do)

---

## Post-Migration Cleanup

Once the removal criteria are met:

1. Delete the v1 branch in `extractor.py`.
2. Remove the `_BUILD_V2` flag entirely.
3. Rename `build_v2_song_dna` → `build_song_dna`.
4. Delete `src/core/dna_schema.py`.
5. Update `src/core/__init__.py` to drop the v1 deprecation note.
6. Run the full test suite, `ruff check src/`, and `mypy src/` to confirm a clean tree.

---

## Development Notes

- The `_BUILD_V2` flag is **developer-only**. There is no `--v2` / `use_v2` CLI argument exposed to users.
- Do not delete `src/core/dna_schema.py` until the criteria above are met — the v1 branch still relies on its types.
- The v2 branch must satisfy the [Extraction Contract](EXTRACTION_CONTRACT.md) invariants at all times.