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

---

## 2. Layer Rules and Dependency Direction

### 2.1 Strict Rules

| Rule | Enforced By |
|------|-------------|
| `src.core/` must never import `src.config/`, `src.app/`, or `src/analysis/` | CI test + code review |
| `src.config/` must never import `src.core/`, `src.app/`, or `src/analysis/` | CI test + code review |
| `src.app/` may import everything (it is the composition root) | Convention |
| `src/analysis/` may import `src.core/` but never `src.app/` or `src/config/` | Code review |

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

Define the pure data models that represent the core business concepts of EchoInsight.

### 3.2 Current Contents

| File | Contents | Status |
|------|----------|--------|
| `src/core/dna_schema.py` | `SongDNA`, `SongMetadata`, `RhythmDNA`, `TimbreDNA`, `TonalDNA`, `StructureDNA`, `Segment`, `IntelligenceDNA`, `StemDNA` | **Active** |
| `src/core/__init__.py` | Re-exports all schema classes | Active |

### 3.3 What Belongs Here

- Data types (`dataclass` or plain classes).
- Value objects (immutable, equality-comparable).
- Validation logic that applies to the type itself (e.g., `__post_init__` checks).
- Type aliases, enums, constants that are part of the domain model.

### 3.4 What Must Never Be Here

- I/O of any kind (filesystem, network, database).
- Logging.
- Configuration reading.
- Framework imports (librosa, numpy, matplotlib, etc.).
- Business logic that spans multiple domain objects.
- Path manipulation or path storage.

### 3.5 Future Additions

- `FrameReference` — lightweight URI wrapper for frame-level data.
- `BeatGrid` — beat position data type.
- `Project` — user workspace model.

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

- Domain types (`SongDNA`, `RhythmDNA`, etc.).
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
- Future: service wiring (DatasetManager, Workspace, SpotifyClient, etc.).

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

    # Future services are attached as optional attributes:
    # dataset_manager: Optional[DatasetManager] = None
    # workspace: Optional[Workspace] = None
```

New services are added as attributes and wired during construction. The context orchestrates — it does not contain business logic.

---

## 6. Infrastructure Layer (`src/analysis/`)

### 6.1 Responsibility

Implement all DSP, comparison, validation, and visualization logic. This is the "engine room" of EchoInsight.

### 6.2 Current Contents

| File | Contents | Status |
|------|----------|--------|
| `extractor.py` | `extract_song_dna()` — audio → SongDNA pipeline | Active |
| `comparison.py` | `compare_songs()` — similarity engine v1 | Active |
| `batch_generator.py` | `generate_dna_dataset()` — batch processing | Active |
| `explorer.py` | `summarize_dataset()` — dataset statistics | Active |
| `validate_comparison.py` | `run_validation()` — similarity matrix diagnostics | Active |
| `visualization.py` | `generate_all_plots()` — PNG plot generation | Active |

### 6.3 What Belongs Here

- All DSP and feature extraction logic.
- Comparison algorithms and metrics.
- Validation and diagnostic frameworks.
- Visualization generation.
- Batch processing orchestration.

### 6.4 What Must Never Be Here

- Application state or context.
- User preferences reading (receive parameters explicitly).
- Configuration defaults (import from `src.config.core` if needed).
- Database or API access.

### 6.5 Future Additions

- `frame_grid/` — FrameGrid extraction and storage.
- `beat_grid/` — BeatGrid extraction and analysis.
- `structure/` — StructureDNA segmentation.
- `stems/` — Stem separation via Demucs.
- `embeddings/` — Neural embedding extraction.

---

## 7. Data Flow

```
Audio File (MP3/WAV/FLAC/OGG/M4A)
    │
    ▼
extractor.py :: extract_song_dna()
    │  Uses librosa for all DSP
    │  Produces SongDNA dataclass
    ▼
SongDNA (src.core.dna_schema)
    │  Frozen dataclass with validation
    │  6 sub-components (3 populated, 3 placeholder)
    ▼
dataclasses.asdict() + json.dumps()
    │  Serialization to JSON file
    ▼
SongDNA JSON (data/dna/*.json)
    │
    ├──→ comparison.py :: compare_songs_from_dicts()
    │       Loads JSON → reconstructs SongDNA → compares → ComparisonResult
    │
    ├──→ validate_comparison.py :: run_validation()
    │       Builds N×N similarity matrix → diagnostics
    │
    ├──→ explorer.py :: summarize_dataset()
    │       Prints dataset statistics
    │
    └──→ visualization.py :: generate_all_plots()
            Saves PNG plots to data/plots/
```

### Application-Controlled Flow (Sprint 1+)

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
| Store paths | Store domain URIs (handled by `FrameReference` in future) |

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

### Example Future Wiring

```python
# src/app/context.py (future)
class ApplicationContext:
    def __init__(self, config, preferences, log_level=logging.INFO):
        # ... existing init ...
        self.dataset_manager: Optional[DatasetManager] = None
        self.workspace: Optional[Workspace] = None
        self.spotify_client: Optional[SpotifyClient] = None

    def init_dataset_manager(self) -> DatasetManager:
        from src.services.dataset_manager import DatasetManager
        self.dataset_manager = DatasetManager(
            data_root=self.preferences.data_root,
            logger=self.logger.getChild("dataset"),
        )
        return self.dataset_manager
```

---

## Appendix: Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-07-12 | Initial architecture — four-tier layering, config, context, bootstrap |