# EchoInsight Architecture Guide

> **Maintain this document as the authoritative reference for EchoInsight's architecture.**
> Every architectural decision, layer boundary, and dependency rule must be documented here.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Layer Rules and Dependency Direction](#2-layer-rules-and-dependency-direction)
3. [Domain Layer (`src/core/`)](#3-domain-layer-srccore)
4. [Configuration Layer (`src/config/`)](#4-configuration-layer-srcconfig)
5. [Application Layer (`src/app/`)](#5-application-layer-srcapp)
6. [Infrastructure Layer (`src/analysis/`)](#6-infrastructure-layer-srcanalysis)
7. [Data Flow](#7-data-flow)
8. [Anti-Patterns — What Each Layer Must Never Do](#8-anti-patterns--what-each-layer-must-never-do)
9. [Adding New Services](#9-adding-new-services)

---

## 1. Architecture Overview

EchoInsight uses a **strict four-tier layered architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER  (src.app/)                              │
│  Runtime context, service container, CLI entry points       │
│  Knows about everything.                                    │
├─────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE LAYER  (src.analysis/)                      │
│  Feature extraction, comparison, validation, visualization   │
│  Knows about Domain.                                        │
├─────────────────────────────────────────────────────────────┤
│  CONFIGURATION LAYER  (src.config/)                         │
│  Immutable defaults + user-editable preferences              │
│  Knows about nothing in the project.                        │
├─────────────────────────────────────────────────────────────┤
│  DOMAIN LAYER  (src/core/)                                  │
│  SongDNA schema, pure data types                             │
│  Knows about nothing in the project.                        │
└─────────────────────────────────────────────────────────────┘
```

### Layer Dependency Direction

```
Application → Infrastructure → Domain
Application → Configuration
Configuration → (nothing in project)
Domain → (nothing in project)
```

**The golden rule:** Dependencies always point **downward**. Never upward, never sideways from config/domain to app.

### Architectural Philosophy

EchoInsight's primary optimisation target is **trustworthy musical intelligence**. Correctness is a prerequisite, not the goal. Every architectural decision prioritises:

- **Deterministic analysis** — the same input always produces the same output
- **Explainability** — every result includes a human-readable rationale
- **Reproducibility** — extraction provenance is recorded in every SongDNA
- **Validation** — invalid domain state is unrepresentable by construction
- **Explicit contracts** — typed result objects for all inter-service communication
- **Schema evolution** — backward-compatible data migration paths

These principles are documented in detail in the Architecture Decision Records ([ADR-0001](docs/adr/ADR-0001.md) through [ADR-0007](docs/adr/ADR-0007.md)).

---

## 2. Layer Rules and Dependency Direction

### 2.1 Strict Rules

| Rule | Enforced By |
|------|-------------|
| `src.core/` must never import `src.config/`, `src.app/`, or `src/analysis/` | CI test + code review |
| `src.config/` must never import `src.core/`, `src.app/`, or `src/analysis/` | CI test + code review |
| `src.app/` may import everything (it is the composition root) | Convention |
| `src.analysis/` may import `src.core/` but never `src.app/` or `src/config/` | Code review |

### 2.2 Dependency Injection

- No global singletons.
- No `import`-time state initialization.
- Dependencies are passed explicitly via constructors.
- `ApplicationContext` is the composition root — constructed once at bootstrap, passed explicitly to services that need it.

### 2.3 Path Handling

- Use `pathlib.Path` everywhere.
- Never use `os.path`.
- Domain types receive `FrameReference(uri="relative/path.npz")` — never absolute filesystem paths.
- Absolute path resolution happens in `ApplicationContext.resolve_data_path()`.

---

## 3. Domain Layer (`src/core/`)

### 3.1 Responsibility

Define the pure data models that represent the core business concepts of EchoInsight. The domain layer imports only the Python standard library — no librosa, no numpy, no external packages.

### 3.2 Domain Model Structure

The domain is built around a single aggregate root:

```
SongDNA  (aggregate root)
 ├── SongMetadata     — song ID, duration, sample rate, format
 ├── SongSummary      — song-level aggregates for fast filtering
 ├── StructureDNA     — macro-structural organisation
 │    └── SegmentDNA[] — individual sections (verse, chorus, etc.)
 │         ├── SegmentTiming    — start/end time, beat positions
 │         ├── SegmentDSP       — per-segment rhythm, harmony, timbre
 │         └── SegmentContext   — label, repetition role, confidence
 ├── FrameReference   — URI to binary frame-level arrays (NPZ)
 └── AnalysisManifest — extraction provenance record
```

All domain objects are **frozen dataclasses** — immutable, hashable, and directly serialisable to JSON via `dataclasses.asdict()`. Validation occurs during object construction. Invalid domain objects cannot exist (see [ADR-0002](docs/adr/ADR-0002.md), [ADR-0005](docs/adr/ADR-0005.md)).

Key design decisions in the domain model:

- **SegmentDSP composition** — rhythmic, harmonic, and timbral features are organised into three sub-objects rather than a flat field list. See [ADR-0003](docs/adr/ADR-0003.md).
- **Segment identity** — each `SegmentDNA` carries a `song_id` to enable the comparison engine to return segments without loading their parent aggregate. See [ADR-0004](docs/adr/ADR-0004.md).
- **Per-aggregate schema versioning** — each data type carries an independent `schema_version` string to enable backward-compatible migration. See [ADR-0007](docs/adr/ADR-0007.md).

### 3.3 What Belongs Here

- Data types (`dataclass` or plain classes).
- Value objects (immutable, equality-comparable).
- Validation logic that applies to the type itself (`__post_init__` checks).
- Type aliases, enums, constants that are part of the domain model.

### 3.4 What Must Never Be Here

- I/O of any kind (filesystem, network, database).
- Logging.
- Configuration reading.
- Framework imports (librosa, numpy, matplotlib, etc.).
- Business logic that spans multiple domain objects.
- Path manipulation or path storage.
- Comparison results, recommendation results, or any computation output.

---

## 4. Configuration Layer (`src/config/`)

### 4.1 Responsibility

Provide two kinds of configuration:

1. **Immutable application defaults** (`core.py`) — set at development time, never modified by the user.
2. **User-editable preferences** (`preferences.py`) — persisted as TOML, mutable at runtime, managed by `PreferenceManager`.

### 4.2 Current Contents

| File | Contents | Status |
|------|----------|--------|
| `src/config/__init__.py` | Re-exports public API | Active |
| `src/config/core.py` | `AudioSettings` (frozen), `PathsConfig` (frozen), `EchoInsightConfig` (frozen) | Active |
| `src/config/preferences.py` | `UserPreferences` (mutable), `PreferenceManager` (load/save) | Active |

### 4.3 What Belongs Here

- Frozen dataclasses for immutable defaults.
- Mutable dataclasses for user preferences.
- I/O logic for reading/writing the preferences TOML file.
- Schema versioning for preferences.

### 4.4 What Must Never Be Here

- Domain types (`SongDNA`, `SegmentDNA`, etc.).
- Analysis logic (extraction, comparison, validation).
- Application runtime state.
- Logger configuration (that belongs in `ApplicationContext`).

### 4.5 Adding New Config Groups

To add a new config group (e.g., `AnalysisConfig`):

1. Create a frozen dataclass in `src/config/core.py`.
2. Add it as a field to `EchoInsightConfig`.
3. Do **not** touch `preferences.py` unless users need to override it.

---

## 5. Application Layer (`src/app/`)

### 5.1 Responsibility

Compose and wire together all lower layers. Hold runtime state. Provide the bootstrap entry point.

### 5.2 Current Contents

| File | Contents | Status |
|------|----------|--------|
| `src/app/__init__.py` | Re-exports `ApplicationContext` | Active |
| `src/app/context.py` | `ApplicationContext` (service container, logger, path resolution) | Active |

### 5.3 What Belongs Here

- `ApplicationContext` — the runtime composition root.
- Logger initialization.
- Path resolution utilities.
- Future: service wiring for higher-level capabilities (dataset management, recommendation, etc.).

### 5.4 What Must Never Be Here

- Business logic (extraction, comparison).
- Domain types as fields (they belong in services).
- Configuration defaults (those belong in `src/config/`).
- Global mutable state.

### 5.5 Service Container Pattern

`ApplicationContext` uses a **composition pattern**:

```python
@dataclass
class ApplicationContext:
    config: EchoInsightConfig
    preferences: UserPreferences
    logger: logging.Logger
    paths: PathsConfig
```

New services are added as attributes and wired during construction. The context orchestrates — it does not contain business logic.

---

## 6. Infrastructure Layer (`src/analysis/`)

### 6.1 Responsibility

Implement all DSP, comparison, validation, and visualization logic. This is the "engine room" of EchoInsight.

### 6.2 What Belongs Here

- All DSP and feature extraction logic.
- Comparison algorithms and metrics.
- Comparison result types (response contracts). See [ADR-0006](docs/adr/ADR-0006.md).
- Validation and diagnostic frameworks.
- Visualization generation.
- Batch processing orchestration.

### 6.3 What Must Never Be Here

- Application state or context.
- User preferences reading (receive parameters explicitly).
- Configuration defaults (import from `src.config.core` if needed).
- Database or API access.

### 6.4 Current Modules

The analysis layer contains modules for extraction, comparison (song-level and segment-level), validation, batch processing, and visualization. As the system grows, these concerns may be split into sub-packages following the pattern already established by `src/analysis/comparison/`.

---

## 7. Data Flow

### Current pipeline (v1 schema)

```
Audio File (MP3/WAV/FLAC/OGG/M4A)
    │
    ▼
extractor.py — librosa DSP pipeline
    │
    ▼
SongDNA (v1 schema — dna_schema.py)
    │  Summary statistics + beat grid
    │  Frame-level data stored as NPZ
    ▼
dataclasses.asdict() + json.dumps()
    │
    ▼
SongDNA JSON (data/dna/*.json)
    │
    ├──→ comparison engine
    ├──→ validation diagnostics
    ├──→ dataset exploration
    └──→ plot generation
```

### Planned pipeline (v2 schema)

The v2 domain model (SongDNA with StructureDNA and SegmentDNA) has been implemented and tested. Once the extraction pipeline is migrated, the flow will include structural segmentation and per-segment DSP summaries:

```
Audio File
    │
    ▼
Frame extraction (librosa)
    │
    ▼
Structural segmentation
    │
    ▼
Per-segment DSP computation
    │
    ▼
SongDNA aggregate (v2 — song_dna.py + structure.py)
    │
    ├──→ segment-level comparison
    ├──→ structural flow comparison
    ├──→ search and recommendation
    └──→ visualization
```

Downstream consumers never access raw audio directly. Every stage consumes structured data from the SongDNA representation.

### Application bootstrap flow

```
main.py :: bootstrap_application()
    │  Creates EchoInsightConfig (immutable defaults)
    │  Creates PreferenceManager → loads UserPreferences
    │  Constructs ApplicationContext with config + preferences
    │  Initialises echoinsight logger
    ▼
ApplicationContext
    │
    ├── config         → EchoInsightConfig (immutable)
    ├── preferences    → UserPreferences (mutable)
    ├── logger         → logging.getLogger("echoinsight")
    ├── paths          → PathsConfig (computed from preferences.data_root)
    └── resolve_data_path() → absolute Path from relative URI
```

---

## 8. Anti-Patterns — What Each Layer Must Never Do

### Domain Layer (`src/core/`)

| ❌ Never | ✅ Instead |
|----------|-----------|
| Import from `src.config`, `src.app`, `src.analysis` | Import only stdlib |
| Read files or network | Accept data as constructor arguments |
| Log anything | Raise exceptions for invalid state |
| Store absolute paths | Store domain URIs via `FrameReference` |
| Contain comparison or recommendation results | Those belong in Infrastructure or Services layers |

### Configuration Layer (`src/config/`)

| ❌ Never | ✅ Instead |
|----------|-----------|
| Import from `src.core`, `src.analysis`, `src.app` | Import only stdlib + external packages |
| Hold runtime state | Frozen dataclasses for defaults; `PreferenceManager` for I/O |
| Contain domain logic | Pure configuration and persistence |
| Be a singleton | Constructed at bootstrap, passed via injection |

### Application Layer (`src/app/`)

| ❌ Never | ✅ Instead |
|----------|-----------|
| Contain business logic | Delegate to `src.analysis/` services |
| Be a god object | Be a service container — thin wiring, no logic |
| Store mutable global state | Pass context explicitly |
| Define configuration defaults | Reference `src.config.core` for defaults |

### Infrastructure Layer (`src/analysis/`)

| ❌ Never | ✅ Instead |
|----------|-----------|
| Access `ApplicationContext` directly | Receive parameters explicitly |
| Read user preferences directly | Accept values as function arguments |
| Initialize application state | Pure stateless functions (or receive state) |
| Write to arbitrary paths | Receive output paths as arguments |

---

## 9. Adding New Services

### Pattern for Adding a New Service (e.g., DatasetManager)

1. **Define the service class** in `src/analysis/` or a new `src/services/` package.
   - Receive all dependencies via constructor injection.
   - Accept configuration values, not config objects.
   - Do not import `ApplicationContext`.

2. **Define configuration defaults** in `src/config/core.py` if the service has immutable defaults.
   - Add to `EchoInsightConfig` if needed.

3. **Add user-facing settings** to `src/config/preferences.py` if the service has user-editable preferences.
   - Wire into `UserPreferences` and `PreferenceManager`.

4. **Wire the service** in `src/app/context.py`.
   - Add an optional attribute to `ApplicationContext`.
   - Construct the service in `ApplicationContext.__init__()` or provide a separate `init_service()` method.

5. **Expose via CLI or API** in `main.py` or a future `cli/` module.
   - The CLI receives `ApplicationContext`, extracts needed config/preferences, and calls the service.

---

## Appendix: Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.1 | 2026-07-22 | Updated domain model to v2; added SegmentDSP, segment identity, schema versioning; removed v1 references; added optimisation target |
| 0.1.0 | 2026-07-12 | Initial architecture — four-tier layering, config, context, bootstrap |