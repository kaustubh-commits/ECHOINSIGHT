# EchoInsight Vision & Design Decisions

> *This document captures the product philosophy, unresolved implementation choices, and experimental directions. Unlike the Architecture Decision Records (ADRs), which document frozen commitments, this is a living guide for the creative and technical direction of EchoInsight.*

---

## Product Philosophy

**EchoInsight gives you X-ray vision into music.**

Most music software treats audio as a waveform to be played or edited. EchoInsight treats audio as a story to be understood, deconstructed, and reimagined.

### The "Choir Instructor" Metaphor
Imagine standing in front of a choir. You don't just press "play"—you point to the sopranos, ask them to hold a note; you signal the tenors to come in softly; you shape the dynamics with your hands. EchoInsight is that podium.

- You **see** the layers (stems).
- You **direct** the volume, loops, and effects.
- The software **obeys** in real-time.
- The AI **suggests** graceful transitions—but you are always the one giving the final nod.

### The User Journey
We design for the flow of a curious, creative user:

> **Hear** → **See** (waveforms/stems) → **Understand** (SongDNA) → **Experiment** (faders/loops) → **Create** (export the mix)

---

## Unresolved Implementation Decisions

The following technologies are strong candidates, but none are finalized. We will prototype and choose based on performance, developer experience, and user feel.

### 1. Creative Workspace UI Framework
| Candidate | Pros | Cons | Status |
| :--- | :--- | :--- | :--- |
| **PySide6 (Qt)** | Native look, mature, full-featured. | Heavier distribution, steeper learning curve. | **Evaluating** |
| **Gradio** | Rapid prototyping, web-based, hot-reload. | Browser latency, less control over low-level audio. | **Evaluating** |
| **Tkinter** | Built-in, lightweight. | Ugly, poor for real-time visualization. | **Dropped** |

**Decision criteria:** The UI must support smooth, sub-50ms waveform updates and respond to real-time audio controls without stuttering. A prototyping sprint will settle this.

### 2. Real-Time Audio Backend
| Candidate | Pros | Cons | Status |
| :--- | :--- | :--- | :--- |
| **sounddevice** (PortAudio) | Low latency, robust, Python-native. | Blocking callbacks require careful thread handling. | **Preferred** |
| **pygame** mixer | Simple, handles basic playback well. | Limited multi-channel control, not suitable for per-stem real-time gain. | **Limited** |
| **PyAudio** | Very low-level, flexible. | Cumbersome API. | **Low-level fallback** |

### 3. Time-Stretching / Pitch-Shifting
| Candidate | Pros | Cons | Status |
| :--- | :--- | :--- | :--- |
| **pyrubberband** | High-quality, phase-vocoder with transient preservation. | Must call C++ library; dependency on Rubber Band. | **Leading** |
| **librosa** time-stretch | Pure Python, easy to integrate. | Lower quality for aggressive tempo changes; artifacts on drums. | **Fallback** |

---

## Spotify Enrichment Strategy

Spotify metadata is **optional, not required**.

> **Status (2026-10-13):** No Spotify integration exists. Track metadata comes from the audio file's own tags, and each field records its source (`file_tags`, `spotify`, `user`, `unknown`). A future provider can plug into `CatalogueProvider` (`src/analysis/track_metadata.py`). Spotify restricted its audio-features and audio-analysis Web API endpoints in late 2024, so the BPM/Key cross-check below is not possible through that API; a provider should expect only basic catalogue metadata (title, artist, album, release date, IDs).

1. **If a Spotify URI is provided** (planned):
   - **Display**: Show album art and catalogue details in the Creative Workspace UI.
   - **Cross-check**: Compare Librosa's estimated BPM/Key against official values, *if a source for them becomes available* (see the status note above). Flag discrepancies in the UI with a confidence indicator.
   - **Context (Experimental)**: Genre tags may be used for *future* algorithm tuning, but they will **not** influence Section Compatibility in v1.0.

2. **If no URI is provided**:
   - EchoInsight falls back to pure Librosa analysis.
   - All core features (Separation, Workspace, Compatibility) remain fully functional.

---

## Creative Intelligence: The AI Approach

We defined "AI" for this project as **Predictive Smoothing**, not generative music.

### What it does (v1.0):
1. Records your fader, mute, and loop automation over a 30-second window.
2. Analyzes the automation curves (e.g., a sudden volume drop).
3. Suggests an eased, DJ-style transition curve (e.g., exponential fade instead of linear cut).
4. You **accept**, **tweak**, or **reject** the suggestion.
5. Renders the final transition to a polished WAV file.

### What it is NOT (v1.0):
- Not generating new melodies or drum patterns.
- Not "auto-mixing" without user input.
- Not replacing the conductor.

### Section Compatibility (Intelligence without ML):
- The compatibility engine is **deterministic** in v1.0.
- It uses measurable audio features:
  - **BPM Ratio** (must be within ±10%, preferably ±5%).
  - **Key Compatibility** (Camelot wheel).
  - **Chroma Similarity** (harmonic profile).
  - **Energy Trajectory** (is the section building energy or relaxing?).
  - **Beat Alignment** (phase coherence).
- This keeps the system explainable and computationally cheap.
- Neural embeddings are a **future stretch goal**, only if these heuristics prove insufficient.

---

## Design Principles

These are the non-negotiable values that guide every decision:

1. **Offline-First**: The system must work with just a local MP3. No mandatory cloud dependencies.
2. **Explainability**: Every measurement and suggestion must be traceable to a specific computation. No black boxes.
3. **Immutability**: Domain objects (SongDNA) never change after creation. This guarantees deterministic, reproducible analyses.
4. **Validation by Construction**: Invalid data cannot exist. Objects validate themselves at creation time.
5. **Lazy Loading**: Heavy audio data (stems, frame arrays) is loaded only when needed, keeping the SongDNA metadata lightweight.

---

## Open Questions / Future Research

- Can we quantize the fine-tuned Demucs model to run efficiently on a Mac M1 without using Colab?
- Does genre-aware weighting actually improve compatibility suggestions? (This requires a formal user study).
- What is the best format for exporting the AI-smoothed transition? (WAV, FLAC, or MP3).
- Should EchoInsight support real-time MIDI control (external controllers)?

---

*This document will evolve as we prototype and learn. When a decision is solidified and coded into the architecture, it graduates to an ADR.*