# Test Audio Fixtures

This directory is for **test audio files** used by the v2 extraction regression tests.

## Adding Fixtures

Add 3 small audio files (10–15 seconds each) in different genres:

- `pop.wav` — a pop track
- `electronic.wav` — an electronic/dance track
- `acoustic.wav` — an acoustic/organic track

Alternatively, any small `.wav` or `.mp3` file works — the fixtures are discovered
automatically via `tests/fixtures/*.wav` and `tests/fixtures/*.mp3`.

## Requirements

- File size should stay small (a few MB per file) to keep the test suite fast.
- Duration should be at least ~10 seconds so the segmentation algorithm has
  enough frames to detect structural boundaries.
- Files must be committed to the repository (do NOT gitignore them) so the
  regression tests run reliably in CI.

## Note

The regression test (`test_extraction_v2_regression`) **skips** when:

1. No fixtures exist in this directory, **or**
2. `_BUILD_V2` is `False` in `src/analysis/extractor.py`.

Once your fixtures are in place and `_BUILD_V2 = True`, the full v2
pipeline (segmentation + per-segment DSP + serialization) will be
exercised on every run.