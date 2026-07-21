# EchoInsight

**A Music Intelligence Platform**

EchoInsight transforms audio into structured, deterministic representations called **SongDNA**. It provides a programmatic foundation for analyzing, comparing, and exploring music at a mathematical level — from frame-level spectral features to macro-scale structural patterns.

---

## Table of Contents

- [Why EchoInsight?](#why-echoinsight)
- [Current Features](#current-features)
- [Planned Features](#planned-features)
- [Architecture Overview](#architecture-overview)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [SongDNA](#songdna)
- [Development](#development)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why EchoInsight?

Most music software treats audio as a waveform to be played, edited, or mixed. Analysis tools typically extract a few high-level properties — tempo, key, energy — and present them as isolated numbers.

EchoInsight treats audio as data to be **modelled**. Every song is transformed into a structured, deterministic representation that captures:

- **Rhythmic structure** — tempo, beat positions, onset strength
- **Harmonic content** — key, chroma profiles, tonal centroid
- **Timbral texture** — spectral centroid, MFCC coefficients, bandwidth
- **Macro-structure** — segment boundaries, section labels, repetition patterns

The system is designed so that every measurement is explainable: each value traces back to a specific DSP computation, and each comparison includes a human-readable rationale for its result.

EchoInsight is not a music player, a DAW, or a visualization tool. It is an engine for building music intelligence applications — comparison, recommendation, search, and analysis — on top of a shared, deterministic representation.

---

## Current Features

**Implemented in v0.1:**

| Feature | Description |
|---------|-------------|
| Domain model | Immutable SongDNA, StructureDNA, and SegmentDNA with typed fields and validation |
| Configuration | Immutable application defaults + user-editable preferences, both injection-friendly |
| Application bootstrap | Dependency injection via `ApplicationContext`, no global state |
| Frame extraction | Librosa-based pipeline for RMS, MFCC, chroma, spectral features, tonnetz, ZCR |
| Song comparison | Deterministic similarity scoring across tempo, timbre, harmonic, and energy dimensions |
| Batch processing | Multi-file DNA generation with progress reporting |
| Dataset exploration | Summary statistics and diagnostic tools for a collection of SongDNA files |
| Validation | Similarity matrix diagnostics for cross-validation |
| Plot generation | Statistical visualizations (tempo distributions, chroma profiles, feature scatter plots) |
| Segment comparison contracts | Typed result types for segment-level and structural comparison (engine implementation pending) |
| Unit tests | 57 tests covering serialization, validation, and invariants |

The **new domain model** (SongDNA v2 with StructureDNA, SegmentDNA, and per-segment DSP features) has been implemented and tested. The extraction pipeline currently produces the v1 schema; migrating it to v2 is the next engineering milestone.

---

## Planned Features

These are **not yet implemented**. They are planned for future versions.

| Version | Feature | Description |
|---------|---------|-------------|
| v0.2 | Extraction Pipeline v2 | Migrate extraction to produce StructureDNA + SegmentDNA |
| v0.2 | Segment-level Comparison | Compare individual sections across songs |
| v0.2 | Structural Flow Comparison | Compare segment sequences and arrangement patterns |
| v0.3 | The Lens | Interactive explainability for comparison results |
| v0.4 | Stem Separation | Per-stem analysis via Demucs integration |
| v0.5 | Recommendation Engine | Segment-based similarity search across libraries |
| v0.6 | Embeddings | ML training on SongDNA representations |
| v0.7 | Dataset Builder | Background collection for large-scale analysis |
| v0.8+ | Visualization | Structural timelines and interactive exploration |
| v1.0 | Full Platform | Integrated comparison, recommendation, visualization |

---

## Architecture Overview

EchoInsight uses a **four-tier layered architecture** with strict dependency direction:

```
┌─────────────────────────────────────────────┐
│  APPLICATION LAYER  (src.app/)              │
│  Runtime context, service container, CLI    │
├─────────────────────────────────────────────┤
│  INFRASTRUCTURE LAYER  (src.analysis/)      │
│  Extraction, comparison, validation, viz    │
├─────────────────────────────────────────────┤
│  CONFIGURATION LAYER  (src.config/)         │
│  Immutable defaults + user preferences      │
├─────────────────────────────────────────────┤
│  DOMAIN LAYER  (src/core/)                  │
│  SongDNA, pure data types, no I/O           │
└─────────────────────────────────────────────┘
```

Dependencies always point downward. The Domain layer imports only the Python standard library. The Infrastructure layer imports from the Domain layer. The Application layer is the composition root and may import everything.

All domain objects are **frozen dataclasses** — immutable, hashable, and directly serializable to JSON.

---

## Pipeline

```
Audio file (MP3, WAV, FLAC, OGG)
       │
       ▼
 Frame extraction (librosa)
       │
       ▼
 Per-segment DSP summaries
       │
       ▼
   SongDNA aggregate
       │
       ├──→ Comparison engine
       ├──→ Search / recommendation
       ├──→ Dataset collection
       └──→ Future: ML, visualization, stem separation
```

Every stage consumes structured data. Downstream consumers never access raw audio directly.

---

## Project Structure

```
src/
├── core/           — Domain layer: pure data types, no external dependencies
│   ├── song_dna.py       # Aggregate root
│   ├── structure.py      # Segment types and structure representation
│   ├── identifiers.py    # Type-safe identifiers (SongID, URI, Estimate)
│   ├── enums.py          # Structural labels and repetition roles
│   ├── frame_reference.py    # NPZ file reference
│   └── analysis_manifest.py  # Extraction provenance
├── analysis/       — Infrastructure: all DSP, comparison, validation, plotting
│   ├── extractor.py       # Audio → SongDNA
│   ├── comparison.py      # Song-level comparison engine (v1)
│   ├── comparison/        # Comparison contracts and future engine v2
│   ├── frame_store.py     # NPZ persistence
│   ├── visualization.py   # Matplotlib plot generation
│   └── ...
├── config/         — Immutable defaults + user-editable preferences
└── app/            — Application context, dependency injection, bootstrap
tests/              — 57 unit tests
docs/               — Architecture guide and ADRs
data/dna/           — SongDNA JSON files (committed)
data/raw/           — Source audio files (gitignored)
```

---

## SongDNA

SongDNA is an immutable, nested representation of a song's acoustic characteristics. It is the core data structure around which the entire platform is built.

```
SongDNA
 ├── Metadata         — song ID, duration, sample rate, format
 ├── Summary          — song-level aggregates for fast filtering
 ├── StructureDNA     — macro-structural organisation
 │    └── SegmentDNA[] — individual sections (verse, chorus, etc.)
 │         ├── Timing      — start/end time, beat positions
 │         ├── DSP         — per-segment rhythm, harmony, timbre
 │         └── Context     — label, repetition role, confidence
 ├── FrameReference   — URI to binary frame-level arrays (NPZ)
 └── AnalysisManifest — extraction provenance record
```

Every estimated value carries a confidence score. Every comparison result includes an explanation. The system is designed so that no number is ever presented without being traceable back to the measurement that produced it.

SongDNA currently exists in two versions:
- **v1** — produced by the current extraction pipeline
- **v2** — defined and tested, with StructureDNA, SegmentDNA, and per-segment DSP features. Extraction migration is the next milestone.

---

## Development

### Requirements

- Python 3.13+
- [librosa](https://librosa.org/) — audio analysis
- [NumPy](https://numpy.org/) — numerical processing
- [Matplotlib](https://matplotlib.org/) — visualization
- [pytest](https://docs.pytest.org/) — testing
- [ruff](https://docs.astral.sh/ruff/) — linting
- [mypy](http://mypy-lang.org/) — type checking

### Installation

```bash
git clone https://github.com/kaustubh-commits/ECHOINSIGHT.git
cd EchoInsight
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running tests

```bash
python -m pytest tests/ -v
```

57 tests should pass with zero failures.

### Linting and type checking

```bash
ruff check src/
ruff format src/ --check
mypy src/
```

### Extracting SongDNA from an audio file

```bash
python main.py path/to/song.mp3
```

Outputs a JSON-formatted SongDNA structure to stdout.

---

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) — full layer responsibilities, dependency rules, anti-patterns
- [Architecture Decision Records](docs/adr/) — engineering rationale for design decisions (7 documents)

---

## Roadmap

### v0.1 (Current) — Domain Foundation
- ✅ Domain model with SongDNA, StructureDNA, SegmentDNA
- ✅ Immutable dataclasses with structural validation
- ✅ Configuration system with dependency injection
- ✅ Extraction pipeline (v1 schema)
- ✅ Song-level comparison engine
- ✅ 57 unit tests
- ✅ ADRs documenting architecture decisions

### v0.2 — Extraction Pipeline v2
- ⬜ Migrate extraction to produce StructureDNA + SegmentDNA
- ⬜ Per-segment DSP feature computation
- ⬜ Segment-level comparison engine
- ⬜ Structural flow comparison

### v0.3 — Comparison and Explainability
- ⬜ The Lens — interactive comparison explanation
- ⬜ Song-to-library comparison
- ⬜ Dataset collector

### v0.4+
- ⬜ Stem separation
- ⬜ Recommendation engine
- ⬜ Embeddings
- ⬜ Visualization
- ⬜ Semantic music understanding

---

## Contributing

Architecture is frozen for the current sprint. If you'd like to contribute:

1. Read the [Architecture Guide](docs/ARCHITECTURE.md) and relevant ADRs
2. Open an issue to discuss proposed changes
3. Follow the existing coding style (frozen dataclasses, strict typing, no clever code)
4. Ensure all tests pass

---

## License

Licensed under the MIT License. See `LICENSE` for details.