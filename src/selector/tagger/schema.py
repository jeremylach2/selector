"""The vibe tagger's feature schema, split into two groups by how the value
was obtained.

This split is the project's central design claim: **measured** features are
read straight from `data/audio_features.parquet` (Step 9's DSP output) and
are never predicted by any model. **Predicted** features are teacher-labelled
from lyrics and metadata (Step 10), because there is no measured ground
truth for things like valence or lyrical theme in a 30-second instrumental
clip. Step 11's eval table exists specifically to show how well the
predicted half tracks the measured half where they overlap, and to give an
honest ceiling (teacher self-consistency) instead of a synthetic one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Read from data/audio_features.parquet. Listed here for documentation, not
# validated by this model, they're numeric columns produced by DSP, not
# structured output from a model call.
MEASURED_FEATURES = ("tempo", "energy", "acousticness", "instrumentalness", "danceability")

MoodTag = Literal[
    "euphoric",
    "melancholic",
    "aggressive",
    "chill",
    "romantic",
    "nostalgic",
    "anxious",
    "triumphant",
    "playful",
    "somber",
]

Era = Literal["pre-1970", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]


class PredictedLabels(BaseModel):
    """The teacher's (or student's) output for one track: everything that
    has to be inferred from lyrics, metadata, and, where available, the
    measured audio features, rather than read off a sensor."""

    valence: float = Field(ge=0.0, le=1.0, description="Emotional positivity, 0=negative, 1=positive")
    mood_tags: list[MoodTag] = Field(min_length=1, max_length=3)
    era: Era
    lyrical_theme: str = Field(max_length=60, description="Short free-text theme, e.g. 'breakup', 'party'")
    intensity: float = Field(ge=0.0, le=1.0, description="Overall dramatic/emotional intensity")


class TeacherInput(BaseModel):
    """Everything the teacher sees for one track: identity, whatever context
    enrich.py could gather, and the measured features when a preview clip
    was matched, a null lyric or a missing measured feature is a valid
    input, not an error, since coverage is expected to be partial."""

    track_id: str
    track_name: str
    artist_name: str
    album_name: str | None = None
    release_year: int | None = None
    artist_genres: list[str] = Field(default_factory=list)
    lyrics: str | None = None
    measured: dict[str, float] = Field(default_factory=dict)


class LabelRecord(BaseModel):
    """One line of data/labels.jsonl: the input the teacher saw, plus its
    output, plus enough provenance to audit or re-run later."""

    track_id: str
    input: TeacherInput
    labels: PredictedLabels
    teacher_model: str
    prompt_variant: str = "default"
