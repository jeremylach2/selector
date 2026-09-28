"""Stage 1, Brief: read the room and pick a theme.

Inputs are recent plays plus the clock, local hour, weekday vs weekend,
and which artists and moods have dominated the last few dozen plays. Output
is a `Brief`: the chosen `Theme`, the context that chose it, a handful of
seed tracks from recent listening that fit it, and a one-paragraph
rationale for the run log.

Theme choice is deterministic scoring, not a model call, so the same
listening state always produces the same brief and the log can say exactly
why a theme won.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from selector.dj.pool import Crate
from selector.fly.pipeline import MOOD_VOCAB


@dataclass(frozen=True)
class Theme:
    """`energy_floor`/`energy_ceiling` bound the arc (see `selector.dj.arc`):
    a late-night set never peaks where a peak-time set does. `hours` are the
    local hours the theme suits; empty means "any time, but only on request
    or strong mood match"."""

    name: str
    description: str
    mood_tags: tuple[str, ...]
    energy_floor: float
    energy_ceiling: float
    hours: tuple[int, ...] = ()
    weekend_bonus: float = 0.0


THEMES: tuple[Theme, ...] = (
    Theme(
        "slow sunrise",
        "Easing into the morning: warm, unhurried, a gentle lift.",
        ("chill", "nostalgic", "romantic"),
        0.15, 0.6,
        hours=tuple(range(5, 10)),
    ),
    Theme(
        "focus drift",
        "Daytime concentration: steady, low-drama, nothing that grabs the wheel.",
        ("chill", "nostalgic", "somber"),
        0.2, 0.55,
        hours=tuple(range(9, 17)),
    ),
    Theme(
        "golden hour",
        "Late afternoon into evening: bright, playful, building towards the night.",
        ("euphoric", "playful", "romantic"),
        0.3, 0.8,
        hours=tuple(range(15, 20)),
    ),
    Theme(
        "night drive",
        "After dark on an empty road: moody, restless, a slow-burning peak.",
        ("melancholic", "anxious", "somber", "nostalgic"),
        0.3, 0.75,
        hours=(20, 21, 22, 23, 0),
    ),
    Theme(
        "peak time",
        "Weekend night energy: big, triumphant, built to peak hard.",
        ("euphoric", "triumphant", "aggressive", "playful"),
        0.4, 0.95,
        hours=(20, 21, 22, 23, 0, 1),
        weekend_bonus=0.6,
    ),
    Theme(
        "after hours",
        "The small hours: hushed, reflective, winding all the way down.",
        ("chill", "melancholic", "somber"),
        0.1, 0.5,
        hours=(0, 1, 2, 3, 4),
    ),
    Theme(
        "adrenaline",
        "Workout fuel: aggressive, driving, relentless.",
        ("aggressive", "triumphant", "euphoric"),
        0.5, 1.0,
    ),
)

# Plain-English words a caller might use for a theme, mapped onto the
# tagger's mood vocabulary, so `dj_set(theme="sad rainy day")` still works.
MOOD_SYNONYMS = {
    "sad": "melancholic", "rain": "melancholic", "rainy": "melancholic", "blue": "melancholic",
    "happy": "euphoric", "party": "euphoric", "hype": "euphoric", "dance": "euphoric",
    "angry": "aggressive", "gym": "aggressive", "workout": "aggressive", "hard": "aggressive",
    "calm": "chill", "relax": "chill", "relaxing": "chill", "study": "chill", "focus": "chill",
    "love": "romantic", "date": "romantic", "sexy": "romantic",
    "memories": "nostalgic", "throwback": "nostalgic", "retro": "nostalgic",
    "tense": "anxious", "dark": "somber", "moody": "melancholic",
    "win": "triumphant", "epic": "triumphant", "fun": "playful", "silly": "playful",
}

N_SEEDS = 5


@dataclass
class Brief:
    theme: Theme
    local_time: str
    hour: int
    is_weekend: bool
    recent_artists: list[tuple[str, int]]
    recent_moods: list[tuple[str, float]]
    seed_track_ids: list[str]
    rationale: str
    theme_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _mood_vector(tags: list[str] | tuple[str, ...]) -> np.ndarray:
    return np.array([1.0 if m in tags else 0.0 for m in MOOD_VOCAB])


def recent_mood_profile(recent: pd.DataFrame, crate: Crate) -> np.ndarray:
    """Mean multi-hot mood vector over recent plays that are in the crate,
    normalised to sum to 1. Zeros if none are."""
    moods = crate.by_id()["mood_tags"]
    vecs = [_mood_vector(moods[t]) for t in recent["track_id"] if t in moods.index]
    if not vecs:
        return np.zeros(len(MOOD_VOCAB))
    profile = np.mean(vecs, axis=0)
    total = profile.sum()
    return profile / total if total else profile


def score_themes(hour: int, is_weekend: bool, mood_profile: np.ndarray) -> dict[str, float]:
    """Hour fit (1.0 in-window) + weekend bonus + cosine overlap between the
    theme's moods and what's been playing lately."""
    scores = {}
    for theme in THEMES:
        tv = _mood_vector(theme.mood_tags)
        denom = np.linalg.norm(tv) * np.linalg.norm(mood_profile)
        mood_fit = float(tv @ mood_profile / denom) if denom else 0.0
        hour_fit = 1.0 if hour in theme.hours else 0.0
        weekend = theme.weekend_bonus if is_weekend else 0.0
        scores[theme.name] = round(hour_fit + weekend + mood_fit, 3)
    return scores


def resolve_theme(requested: str) -> Theme:
    """A named theme (exact or substring match), else a custom theme built
    from any mood words in `requested`. Raises `ValueError` if neither."""
    text = requested.strip().lower()
    for theme in THEMES:
        if text == theme.name or theme.name in text or (len(text) >= 4 and text in theme.name):
            return theme

    words = [w.strip(".,!?'\"") for w in text.split()]
    moods = []
    for w in words:
        mood = w if w in MOOD_VOCAB else MOOD_SYNONYMS.get(w)
        if mood and mood not in moods:
            moods.append(mood)
    if not moods:
        names = ", ".join(f'"{t.name}"' for t in THEMES)
        raise ValueError(
            f'Couldn\'t read a mood from theme "{requested}". Use one of {names}, '
            f"or describe it with mood words like {', '.join(MOOD_VOCAB)}."
        )

    # Borrow the energy range of the named theme whose moods overlap most,
    # so "sad rainy day" inherits a sensible low-energy arc.
    closest = max(THEMES, key=lambda t: len(set(t.mood_tags) & set(moods)))
    return Theme(
        name=requested.strip(),
        description=f"Custom theme built from moods: {', '.join(moods)}.",
        mood_tags=tuple(moods),
        energy_floor=closest.energy_floor,
        energy_ceiling=closest.energy_ceiling,
    )


def pick_seeds(theme: Theme, recent: pd.DataFrame, crate: Crate, n: int = N_SEEDS) -> list[str]:
    """Recent crate tracks that share a mood with the theme, most recent
    first. Topped up with the fly's highest-taste on-theme tracks if recent
    listening doesn't supply enough, seeds anchor coherence in Select."""
    tracks = crate.by_id()
    on_theme = tracks["mood_tags"].apply(lambda tags: bool(set(tags) & set(theme.mood_tags)))

    seeds: list[str] = []
    for tid in recent["track_id"]:
        if tid in tracks.index and on_theme[tid] and tid not in seeds:
            seeds.append(tid)
        if len(seeds) == n:
            return seeds

    fallback = tracks[on_theme & ~tracks.index.isin(seeds)].sort_values("taste", ascending=False)
    seeds.extend(fallback["track_id"].head(n - len(seeds)))
    return seeds


def build_brief(
    recent: pd.DataFrame,
    crate: Crate,
    now: datetime,
    requested_theme: str | None = None,
) -> Brief:
    """`recent` needs `track_id` and `artist_name` columns, newest first."""
    hour = now.hour
    # Friday evening counts: that's when a weekend night actually starts.
    is_weekend = now.weekday() >= 5 or (now.weekday() == 4 and hour >= 17)
    profile = recent_mood_profile(recent, crate)
    scores = score_themes(hour, is_weekend, profile)

    if requested_theme:
        theme = resolve_theme(requested_theme)
        why = f'Theme "{theme.name}" was requested.'
    else:
        theme = next(t for t in THEMES if t.name == max(scores, key=scores.get))
        why = (
            f'Chose "{theme.name}" (score {scores[theme.name]}) for '
            f"{now:%A} at {now:%H:%M}: "
            + ("it fits this hour" if hour in theme.hours else "it doesn't fit the hour, but")
            + " and its moods overlap what's been playing lately."
        )

    artists = Counter(recent["artist_name"].dropna()).most_common(5)
    moods = sorted(
        ((m, round(float(w), 3)) for m, w in zip(MOOD_VOCAB, profile) if w > 0),
        key=lambda x: -x[1],
    )[:4]
    seeds = pick_seeds(theme, recent, crate)

    rationale = why
    if artists:
        rationale += f" Recent rotation leans on {', '.join(a for a, _ in artists[:3])}"
        rationale += f", mostly {', '.join(m for m, _ in moods[:2])}." if moods else "."

    return Brief(
        theme=theme,
        local_time=now.isoformat(timespec="minutes"),
        hour=hour,
        is_weekend=is_weekend,
        recent_artists=artists,
        recent_moods=moods,
        seed_track_ids=seeds,
        rationale=rationale,
        theme_scores=scores,
    )
