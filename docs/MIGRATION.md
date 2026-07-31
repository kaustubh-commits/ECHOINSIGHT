# MIGRATION: v1 → v2 Extraction Pipeline

> This document tracks the temporary compatibility path between the legacy v1 extraction pipeline (`src/core/dna_schema.py`) and the new v2 pipeline (`src/core/song_dna.py` + `src/core/structure.py`).

---

## Current State

The extraction pipeline in `src/analysis/extractor.py` currently supports both branches behind the developer-only `_BUILD_V2` flag:

```python
# MIGRATION: This flag determines which extraction pipeline to use.
_BUILD_V2 = False  # ← Set to True when ready to test v2 pipeline
```

- `_BUILD_V2 = False` — the v1 branch constructs a v2-compatible `SongDNA` (single global segment covering the whole track) from the legacy `dna_schema.py` features. This keeps every consumer on the v2 type without a type duality.
- `_BUILD_V2 = True` — the v2 branch constructs a full v2 `SongDNA` with structural segmentation (`segmentation.py`) and per-segment DSP (`segment_dsp.py`).

---

## Removal Criteria

The temporary extraction compatibility path (`_BUILD_V2` flag, the v1 branch, and `src/core/dna_schema.py`) will be removed when **all** of the following are true:

- [ ] Workspace consumes SongDNA v2
- [ ] Visualization consumes SongDNA v2
- [ ] Comparison consumes SongDNA v2
- [ ] All extraction tests pass using v2
- [ ] No production code imports `dna_schema.py`

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