"""Key estimation must name the correct tonic for all 24 major and minor keys.

Both estimators (song-level in ``extractor`` and per-segment in
``segment_dsp``) are checked.  No audio fixtures are needed: inputs are
either ideal pitch-class profiles or short synthesized chord cadences.
"""

from typing import Callable, List, Tuple

import librosa
import numpy as np
import pytest

from src.analysis import extractor, segment_dsp
from src.analysis.comparison import _camelot_distance, _camelot_number

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ALL_KEYS: List[Tuple[int, str]] = [
    (tonic, mode) for mode in ("major", "minor") for tonic in range(12)
]
ESTIMATORS = {
    "extractor": extractor._estimate_key,
    "segment_dsp": segment_dsp._estimate_key,
}


def _key_id(param: Tuple[int, str]) -> str:
    tonic, mode = param
    return f"{KEY_NAMES[tonic]}-{mode}"


# --- Ideal pitch-class profiles ----------------------------------------------


@pytest.mark.parametrize("estimator_name", sorted(ESTIMATORS))
@pytest.mark.parametrize("key", ALL_KEYS, ids=_key_id)
def test_ideal_profile_is_detected(estimator_name: str, key: Tuple[int, str]) -> None:
    """The Krumhansl profile of a key, transposed to its tonic, maps back to it."""
    tonic, mode = key
    profile = extractor._MAJOR_PROFILE if mode == "major" else extractor._MINOR_PROFILE
    chroma = np.roll(profile, tonic)  # tonic weight lands on pitch class `tonic`

    estimate: Callable = ESTIMATORS[estimator_name]
    name, confidence = estimate(chroma)

    assert name == f"{KEY_NAMES[tonic]} {mode}"
    assert confidence == pytest.approx(1.0)


# --- Synthesized cadences -----------------------------------------------------

SR = 22050
CHORD_SECONDS = 0.6

# Root-position triads (semitones above the tonic) for I-IV-V-I and i-iv-V-i.
# The minor cadence uses the harmonic-minor dominant (raised leading tone).
CADENCES = {
    "major": [(0, 4, 7), (5, 9, 12), (7, 11, 14), (0, 4, 7)],
    "minor": [(0, 3, 7), (5, 8, 12), (7, 11, 14), (0, 3, 7)],
}


def _tone(midi: float, seconds: float) -> np.ndarray:
    """A harmonic tone (4 partials, decaying) with a short fade in/out."""
    t = np.arange(int(SR * seconds)) / SR
    f0 = librosa.midi_to_hz(midi)
    wave = sum((0.6 ** k) * np.sin(2 * np.pi * f0 * (k + 1) * t) for k in range(4))
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.02)
    return wave * fade


def _cadence(tonic: int, mode: str) -> np.ndarray:
    base = 60 + tonic  # tonic in octave 4 (C4 = MIDI 60)
    chords = []
    for chord in CADENCES[mode]:
        notes = [_tone(base + step, CHORD_SECONDS) for step in chord]
        notes.append(_tone(base - 12 + chord[0], CHORD_SECONDS))  # bass note
        chords.append(np.sum(notes, axis=0))
    y = np.concatenate(chords)
    return (y / np.max(np.abs(y))).astype(np.float32)


def test_every_estimated_key_maps_onto_the_camelot_wheel() -> None:
    """Downstream: each key name the estimator can produce parses in the
    comparison engine, and the 24 keys cover all 24 Camelot positions."""
    positions = set()
    for tonic, mode in ALL_KEYS:
        profile = extractor._MAJOR_PROFILE if mode == "major" else extractor._MINOR_PROFILE
        name, _ = extractor._estimate_key(np.roll(profile, tonic))
        number = _camelot_number(name)
        assert number is not None, name
        positions.add((number, mode))
    assert len(positions) == 24


def test_relative_major_minor_have_camelot_distance_zero() -> None:
    assert _camelot_distance(_camelot_number("C major"), _camelot_number("A minor")) == 0
    assert _camelot_distance(_camelot_number("C major"), _camelot_number("C minor")) == 3


@pytest.mark.parametrize("key", ALL_KEYS, ids=_key_id)
def test_synthesized_cadence_is_detected(key: Tuple[int, str]) -> None:
    """Audio -> chroma_cqt -> estimator recovers the cadence's key."""
    tonic, mode = key
    y = _cadence(tonic, mode)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=extractor.HOP_LENGTH)

    name, _ = extractor._estimate_key(np.mean(chroma, axis=1))

    assert name == f"{KEY_NAMES[tonic]} {mode}"
