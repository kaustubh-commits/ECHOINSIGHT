"""
enums.py — Domain enumerations for structural segment types and repetition
relationships.

These enums live in ``src.core`` and are consumed by ``SegmentDNA`` and
``StructureDNA``.  They are intentionally limited — only enumerations that
are fundamental to the domain model belong here.  Transient or UI-specific
enumerations belong at the application layer.

Layer rules
-----------
- ``src.core.enums`` imports **only** Python stdlib.
- It must **never** import ``src.analysis``, ``src.config``, ``src.app``,
  or any external package.
"""

from __future__ import annotations

from enum import Enum


class LabelType(str, Enum):
    """Standardised structural segment labels.

    These labels describe the *structural role* of a segment — where it
    sits in the arrangement.  The list is based on commonly used labels
    in music production and DJ software.

    ``UNKNOWN`` is the fallback for segments that the segmentation algorithm
    could not confidently classify.  Clients should treat ``UNKNOWN`` as
    "no label available" rather than a bug.

    Usage
    -----
    .. code-block:: python

        if segment.context.label is LabelType.CHORUS:
            print("This is a chorus")
    """

    INTRO = "intro"
    """Opening section, typically before the main groove or vocals begin."""

    VERSE = "verse"
    """A section where the main lyrical content unfolds, typically repeating
    with different lyrics each time."""

    PRE_CHORUS = "pre_chorus"
    """A transitional section that builds tension before the chorus."""

    CHORUS = "chorus"
    """The main recurring section — typically the most memorable, high-energy
    part of the song."""

    POST_CHORUS = "post_chorus"
    """A section that follows the chorus, sometimes an extension or
    instrumental tail."""

    BRIDGE = "bridge"
    """A contrasting section that provides relief from repetition, typically
    occurring once and containing new harmonic or lyrical material."""

    DROP = "drop"
    """A section where the full rhythmic and bass elements return after a
    build — a peak moment common in electronic music."""

    BUILD = "build"
    """A section that increases tension in preparation for a drop or
    chorus — often characterised by risers, snare rolls, or filtering."""

    BREAKDOWN = "breakdown"
    """A stripped-back section that reduces instrumentation to create space
    before rebuilding energy."""

    OUTRO = "outro"
    """The closing section of the song, where elements are gradually
    removed."""

    INTERLUDE = "interlude"
    """A short connecting section between major structural elements."""

    SOLO = "solo"
    """A section featuring a prominent instrumental solo."""

    FILL = "fill"
    """A short, transitional passage — typically 1–4 bars — that bridges
    two structural sections."""

    TRANSITION = "transition"
    """A section whose primary purpose is to transition between two distinct
    musical states (key change, tempo change, genre shift)."""

    SILENCE = "silence"
    """A section of silence or near-silence (ambient noise floor only)."""

    UNKNOWN = "unknown"
    """The segment could not be classified.  This is not an error — it
    indicates uncertainty in the labelling algorithm."""


class RepetitionRole(str, Enum):
    """How a segment relates to other segments in the same song.

    The repetition role captures the *relationship* between a segment and
    the song's larger structural narrative.  It answers the question:
    "Is this segment new material, a repeat, or a variation?"

    Usage
    -----
    .. code-block:: python

        if segment.context.repetition_role is RepetitionRole.VARIATION:
            print("This segment is a variation of a previous one")
    """

    ORIGINAL = "original"
    """The first occurrence of a structural pattern.  Subsequent segments
    that are similar to this one will reference it as their original."""

    REPETITION = "repetition"
    """An exact or near-exact repeat of a previous segment.  The audio
    content is highly similar across all DSP dimensions."""

    VARIATION = "variation"
    """A modified version of a previous segment.  The core structure
    (harmonic progression, length) is preserved, but some elements have
    changed (instrumentation, energy level, vocal delivery)."""

    DEVELOPMENT = "development"
    """An evolved version of a previous segment.  The relationship is
    recognisable, but the segment has been substantially reworked —
    often longer, more intense, or in a different key."""

    CONTRAST = "contrast"
    """A segment that is deliberately different from its surroundings.
    Contrast segments are often unique and serve as structural "shocks"
    to maintain listener attention."""

    UNIQUE = "unique"
    """No known repetition relationship.  The segment occurs exactly once
    and is not structurally similar to any other segment in the song."""