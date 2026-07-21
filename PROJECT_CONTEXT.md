# EchoInsight — Project Context

## Project

EchoInsight is a music intelligence platform that transforms audio into structured knowledge representations called **SongDNA**.

The long-term goal is to build an explainable system capable of analyzing, comparing, visualizing, and eventually generating music using deterministic DSP techniques combined with machine learning.

Current focus is **backend engineering**, not UI.

---

# Current Version

Version: v0.1 (Domain Foundation)

Status:
- Domain model complete
- Configuration system complete
- Application bootstrap complete
- Tests passing
- Extraction pipeline not yet implemented

---

# Engineering Philosophy

This project values:

- Correctness over speed
- Explicit architecture
- Immutable domain models
- Explainable algorithms
- Strong typing
- Small, reviewable commits
- High test coverage

Avoid clever code.

Prefer readable code.

---

# Architecture

```
Application
      │
      ▼
Analysis
      │
      ▼
Domain
```

Configuration is injected through the application layer.

Dependencies always point downward.

---

# Domain Model

The central aggregate root is:

SongDNA

It contains immutable information describing a song.

Examples include:

- metadata
- rhythm
- timbre
- structure
- segments

Every analysis ultimately produces a SongDNA object.

---

# Current Milestone

Completed

- Domain layer
- Configuration
- Bootstrap
- Tests

Next

Implement deterministic extraction:

Audio
↓

Librosa

↓

Frame Analysis

↓

Segmentation

↓

Feature Extraction

↓

SongDNA

↓

JSON

---

# Development Rules

Never redesign architecture without discussion.

Keep domain objects immutable.

Do not introduce global state.

Do not introduce hidden dependencies.

Prefer composition over inheritance.

Every major architectural decision should receive an ADR.

---

# Repository Structure

src/
    app/
    analysis/
    config/
    core/

tests/

docs/

data/

---

# Documentation

Architecture documentation lives in:

docs/

Architecture decisions live in:

docs/adr/

---

# Long-Term Roadmap

v0.1
Domain Foundation

v0.2
Extraction Pipeline

v0.3
Comparison Engine

v0.4
Visualization

v0.5
Recommendation System

v0.6
Stem Separation

v1.0
Music Intelligence Platform

---

# Important

This file exists to provide project context.

It is **not** the source of architectural truth.

The source of truth is:

- source code
- ADRs
- tests