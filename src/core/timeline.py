"""
timeline.py — Beat-aligned timeline data types: Bar and Timeline.

``Timeline`` captures the rhythmic grid of a song: the beat positions,
the downbeat positions (bar starts), and the bar spans derived from
them.  It is the **timeline-first** foundation for the Workspace:
everything musical (bar grid rendering, looping, cue points,
transitions) is anchored to bar and beat boundaries.

Layer rules
-----------
- ``src.core.timeline`` imports **only** Python stdlib.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Sequence, Tuple

_BAR_ALIGN_TOLERANCE: float = 1e-6
"""Tolerance (seconds) for float time comparisons."""


@dataclasses.dataclass(frozen=True)
class Bar:
    """A single bar (measure) in the timeline grid.

    Attributes
    ----------
    number : int
        1-based bar number.  Bar 1 is the first bar of the track.
    start_time : float
        Start time in seconds (>= 0).
    end_time : float
        End time in seconds (> start_time).
    """

    number: int
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"number must be >= 1, got {self.number}")
        if self.start_time < 0:
            raise ValueError(
                f"start_time must be >= 0, got {self.start_time}"
            )
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be > "
                f"start_time ({self.start_time})"
            )

    @property
    def duration(self) -> float:
        """Bar duration in seconds (end_time - start_time)."""
        return self.end_time - self.start_time


@dataclasses.dataclass(frozen=True)
class Timeline:
    """Beat-aligned timeline of a song.

    All time values are in **seconds** measured from the start of the
    song — the canonical time representation.

    Attributes
    ----------
    beats : Tuple[float, ...]
        Beat timestamps in seconds, strictly increasing, non-empty.
    downbeats : Tuple[float, ...]
        Downbeat timestamps (bar starts) in seconds, strictly
        increasing, non-empty.  Every downbeat is also a beat.
    bars : Tuple[Bar, ...]
        Contiguous bar spans covering the timeline.  Bar numbers start
        at 1.
    num_bars : int
        Number of bars (convenience, derived from ``len(bars)``).
        Computed automatically — do not pass in constructor.
    """

    beats: Tuple[float, ...]
    downbeats: Tuple[float, ...]
    bars: Tuple[Bar, ...]
    num_bars: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._validate_beats()
        self._validate_downbeats()
        self._validate_bars()
        object.__setattr__(self, "num_bars", len(self.bars))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_beats(self) -> None:
        if not self.beats:
            raise ValueError("beats must be non-empty")
        for i, t in enumerate(self.beats):
            if t < 0:
                raise ValueError(f"beats[{i}] must be >= 0, got {t}")
            if not math.isfinite(t):
                raise ValueError(f"beats[{i}] must be finite, got {t}")
            if i > 0 and t <= self.beats[i - 1]:
                raise ValueError(
                    f"beats must be strictly increasing; beats[{i}] "
                    f"({t}) <= beats[{i - 1}] ({self.beats[i - 1]})"
                )

    def _validate_downbeats(self) -> None:
        if not self.downbeats:
            raise ValueError("downbeats must be non-empty")
        for i, t in enumerate(self.downbeats):
            if t < 0:
                raise ValueError(f"downbeats[{i}] must be >= 0, got {t}")
            if not math.isfinite(t):
                raise ValueError(
                    f"downbeats[{i}] must be finite, got {t}"
                )
            if i > 0 and t <= self.downbeats[i - 1]:
                raise ValueError(
                    f"downbeats must be strictly increasing; "
                    f"downbeats[{i}] ({t}) <= "
                    f"downbeats[{i - 1}] ({self.downbeats[i - 1]})"
                )
        for db in self.downbeats:
            if not any(abs(db - b) <= _BAR_ALIGN_TOLERANCE for b in self.beats):
                raise ValueError(
                    f"downbeat {db} is not present in beats"
                )

    def _validate_bars(self) -> None:
        if not self.bars:
            raise ValueError("bars must be non-empty")

        for i, bar in enumerate(self.bars):
            if bar.number != i + 1:
                raise ValueError(
                    f"bars[{i}].number must be {i + 1}, got {bar.number}"
                )

        if abs(self.bars[0].start_time) > _BAR_ALIGN_TOLERANCE:
            raise ValueError(
                f"bars[0].start_time must be 0.0, "
                f"got {self.bars[0].start_time}"
            )

        # The set of bar boundaries is {0.0} ∪ downbeats ∪ {track end}.
        # When the last downbeat coincides with the track end, the two
        # boundaries dedupe into one (no tail bar).
        boundaries = sorted(
            {0.0, *self.downbeats, self.bars[-1].end_time}
        )
        if len(self.bars) != len(boundaries) - 1:
            raise ValueError(
                f"Expected {len(boundaries) - 1} bars "
                f"(bar boundaries: {boundaries}), "
                f"got {len(self.bars)}"
            )
        for i, bar in enumerate(self.bars):
            if abs(bar.start_time - boundaries[i]) > _BAR_ALIGN_TOLERANCE:
                raise ValueError(
                    f"bars[{i}].start_time ({bar.start_time}) does not "
                    f"match expected bar start ({boundaries[i]})"
                )
            if abs(bar.end_time - boundaries[i + 1]) > _BAR_ALIGN_TOLERANCE:
                raise ValueError(
                    f"bars[{i}].end_time ({bar.end_time}) does not "
                    f"match expected bar end ({boundaries[i + 1]})"
                )

        if self.bars[-1].end_time < self.downbeats[-1]:
            raise ValueError(
                f"bars[-1].end_time ({self.bars[-1].end_time}) must be "
                f">= last downbeat ({self.downbeats[-1]})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bar_at(self, time: float) -> Bar:
        """Return the bar containing *time* (in seconds).

        Parameters
        ----------
        time : float
            Time in seconds within ``[0, bars[-1].end_time)``.

        Returns
        -------
        Bar
            The bar whose span contains *time*.

        Raises
        ------
        ValueError
            If *time* is outside the timeline.
        """
        if time < 0 or time >= self.bars[-1].end_time:
            raise ValueError(
                f"time {time} is outside the timeline "
                f"[0, {self.bars[-1].end_time})"
            )
        lo, hi = 0, len(self.bars)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.bars[mid].start_time <= time:
                lo = mid
            else:
                hi = mid
        return self.bars[lo]

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        beats: Sequence[float],
        duration: float,
        beats_per_bar: int = 4,
    ) -> "Timeline":
        """Build a ``Timeline`` from detected beat times.

        Downbeats are derived by assuming a **4/4 time signature**
        (``beats_per_bar`` beats per bar).  Every ``beats_per_bar``-th
        beat is a downbeat (bar start).

        Bars are constructed to cover the full timeline contiguously:

        - A pickup bar ``[0.0, downbeats[0])`` when the first downbeat
          is not at 0.0.
        - One bar per downbeat span ``[downbeats[i], downbeats[i + 1])``.
        - A tail bar ``[downbeats[-1], duration)`` when the last
          downbeat is before *duration*.

        Parameters
        ----------
        beats : Sequence[float]
            Beat timestamps in seconds, strictly increasing.
        duration : float
            Total track duration in seconds (> 0).  The last bar
            extends to this time.
        beats_per_bar : int
            Number of beats per bar (assumed 4 for 4/4 time).
            Must be >= 1.

        Returns
        -------
        Timeline
            The constructed timeline.

        Raises
        ------
        ValueError
            If ``beats`` is empty or *duration* is not positive.
        """
        if not beats:
            raise ValueError("beats must be non-empty")
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        if beats_per_bar < 1:
            raise ValueError(
                f"beats_per_bar must be >= 1, got {beats_per_bar}"
            )

        beat_times: Tuple[float, ...] = tuple(float(b) for b in beats)
        downbeats: Tuple[float, ...] = beat_times[::beats_per_bar]

        # Bar boundaries: build from {0.0} ∪ downbeats ∪ {duration},
        # avoiding duplicates when the first downbeat is at 0.0 or the
        # last downbeat coincides with the track end.
        boundaries: list[float] = [0.0]
        for db in downbeats:
            if abs(db - boundaries[-1]) > _BAR_ALIGN_TOLERANCE:
                boundaries.append(db)
        if abs(boundaries[-1] - duration) > _BAR_ALIGN_TOLERANCE:
            boundaries.append(duration)

        bars: list[Bar] = []
        for i in range(len(boundaries) - 1):
            bars.append(
                Bar(
                    number=i + 1,
                    start_time=boundaries[i],
                    end_time=boundaries[i + 1],
                )
            )

        return cls(
            beats=beat_times,
            downbeats=downbeats,
            bars=tuple(bars),
        )