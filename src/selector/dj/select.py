"""Stage 3, Select: fill the arc one track at a time.

Walks forward through the set. At each step every crate track is a
candidate, and each is judged at the point in the set where *it* would
play (its own midpoint, given its own duration), so a long track and a
short one are held to different targets. A candidate must:

- sit inside the arc's energy band at that point (hard; the band is widened
  in steps only if nothing qualifies, and the widening is logged),
- not push its artist past `max_per_artist`,
- be the kind the familiar/fresh ratio currently calls for,
- in `hard_transitions` mode (Critique's revision request), not make a
  transition Critique would call jarring.

When nothing survives, constraints relax in a fixed order, ratio, then
transition rule, then band width, and every relaxation is logged.

Survivors are ranked on five terms:

- **arc fit**: how close the track's measured energy is to the target
  (the band is the hard limit; this pulls picks towards its centre, so the
  realised curve has the arc's shape and not just its bounds),

- **coherence**: fly-brain similarity (Dice overlap of Kenyon-cell tags,
  i.e. normalised Hamming distance) to the previous pick and to the brief's
  seed tracks,
- **taste**: the production mushroom body's valence (percentile over the
  crate), i.e. how much the fly predicts this listener approaches the track,
- **theme**: mood-tag overlap with the theme,
- minus a **transition** penalty for tempo and energy jumps from the
  previous pick (measured features, see `selector.dj.arc`).

Every step logs its shortlist, so the run log can show what was passed over
and why.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from selector.dj.arc import (
    COMBINED_ENERGY_STEP,
    COMBINED_TEMPO_SHIFT,
    HARD_ENERGY_STEP,
    HARD_TEMPO_SHIFT,
    MAX_ENERGY_STEP,
    MAX_TEMPO_SHIFT,
    EnergyArc,
    tempo_shift,
)
from selector.dj.brief import Brief
from selector.dj.crate import Crate

# Band widening steps, as multiples of the arc's tolerance, tried in order
# only when no candidate fits the band at the current width.
BAND_WIDENING = (1.0, 1.5, 2.0)


@dataclass
class SelectConfig:
    familiar_ratio: float = 0.6
    max_per_artist: int = 2
    # Taste is weighted below theme on purpose: at equal weight the fly's
    # favourite few dozen tracks win every slot of every theme, and a
    # morning set and a peak-time set come out half identical.
    w_arc: float = 0.25
    w_coherence: float = 0.25
    w_taste: float = 0.2
    w_theme: float = 0.3
    w_transition: float = 0.15
    hard_transitions: bool = False
    shortlist: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.familiar_ratio <= 1.0:
            raise ValueError("familiar_ratio must be in [0, 1]")
        if self.max_per_artist < 1:
            raise ValueError("max_per_artist must be at least 1")


@dataclass
class Pick:
    track_id: str
    name: str
    artist: str
    start_s: float
    duration_s: float
    t_mid: float
    phase: str
    target_energy: float
    energy: float
    tempo: float | None
    familiar: bool
    fly_valence: float
    hamming_to_prev: int | None
    scores: dict[str, float]


@dataclass
class Selection:
    picks: list[Pick]
    steps: list[dict] = field(default_factory=list)
    config: SelectConfig = field(default_factory=SelectConfig)
    excluded: list[str] = field(default_factory=list)

    @property
    def track_ids(self) -> list[str]:
        return [p.track_id for p in self.picks]

    @property
    def seconds(self) -> float:
        return sum(p.duration_s for p in self.picks)

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "excluded": self.excluded,
            "picks": [asdict(p) for p in self.picks],
            "steps": self.steps,
        }


def _dice(distances: np.ndarray, pop_a: np.ndarray, pop_b: float) -> np.ndarray:
    """Hamming distance -> Dice similarity in [0, 1]: 1 - d / (|a| + |b|)."""
    denom = pop_a + pop_b
    return np.where(denom > 0, 1.0 - distances / np.maximum(denom, 1), 0.0)


def _rank01(x: np.ndarray) -> np.ndarray:
    """Percentile rank within `x`, so terms on different scales combine."""
    if x.size <= 1:
        return np.ones_like(x, dtype=float)
    order = x.argsort().argsort()
    return order / (x.size - 1)


def _scores_at(terms: dict[str, np.ndarray], k: int) -> dict[str, float]:
    return {name: round(float(col[k]), 3) for name, col in terms.items()}


def select(
    crate: Crate,
    brief: Brief,
    arc: EnergyArc,
    config: SelectConfig | None = None,
    exclude: set[str] | None = None,
) -> Selection:
    config = config or SelectConfig()
    exclude = exclude or set()
    tracks = crate.tracks
    n = len(tracks)

    tags = crate.tags.take(tracks["tag_row"].to_numpy())
    popcount = tags.popcount
    energy = tracks["energy"].to_numpy(dtype=float)
    tempo = tracks["tempo"].to_numpy(dtype=float)
    duration_s = tracks["duration_ms"].to_numpy(dtype=float) / 1000
    familiar = tracks["familiar"].to_numpy(dtype=bool)
    artists = tracks["artist"].to_numpy()
    taste = tracks["taste"].to_numpy(dtype=float)
    theme_moods = set(brief.theme.mood_tags)
    theme_fit = np.array(
        [len(set(m) & theme_moods) / len(set(m) | theme_moods) if m else 0.0 for m in tracks["mood_tags"]]
    )

    # Mean similarity to the brief's seeds: the theme's anchor in fly space.
    row_of = {tid: i for i, tid in enumerate(tracks["track_id"])}
    seed_rows = [row_of[s] for s in brief.seed_track_ids if s in row_of]
    seed_sim = np.zeros(n)
    for r in seed_rows:
        seed_sim += _dice(tags.hamming(r), popcount, popcount[r])
    if seed_rows:
        seed_sim /= len(seed_rows)

    available = ~tracks["track_id"].isin(exclude).to_numpy()
    typical_s = float(np.median(duration_s))
    picks: list[Pick] = []
    steps: list[dict] = []
    artist_count: Counter[str] = Counter()
    elapsed = 0.0
    prev: int | None = None

    while arc.seconds - elapsed > typical_s / 2:
        t_mid = np.clip((elapsed + duration_s / 2) / arc.seconds, 0, 1)
        target = np.array([arc.target(t) for t in t_mid])
        artist_ok = np.array([artist_count[a] < config.max_per_artist for a in artists])
        base = available & artist_ok

        # Deficit rule: take a familiar track whenever the running share of
        # familiar picks is below the requested ratio, else a fresh one.
        want_familiar = sum(p.familiar for p in picks) < config.familiar_ratio * (len(picks) + 1)
        kind_ok = familiar == want_familiar

        if prev is None:
            coherence_raw = seed_sim
            transition = np.zeros(n)
            smooth = np.ones(n, dtype=bool)
            prev_dist = None
        else:
            prev_dist = tags.hamming(prev)
            prev_sim = _dice(prev_dist, popcount, popcount[prev])
            coherence_raw = 0.5 * prev_sim + 0.5 * seed_sim if seed_rows else prev_sim
            shifts = np.array([tempo_shift(tempo[prev], b) for b in tempo], dtype=float)
            steps_e = np.abs(energy - energy[prev])
            tempo_pen = np.nan_to_num(shifts / MAX_TEMPO_SHIFT, nan=0.5)
            energy_pen = steps_e / MAX_ENERGY_STEP
            transition = np.minimum(tempo_pen, 2.0) / 2 + np.minimum(energy_pen, 2.0) / 2
            # Vectorised `arc.jarring_reason(...) is None` (NaN shifts compare False).
            smooth = (
                (steps_e <= HARD_ENERGY_STEP)
                & ~(shifts > HARD_TEMPO_SHIFT)
                & ~((shifts > COMBINED_TEMPO_SHIFT) & (steps_e > COMBINED_ENERGY_STEP))
            )
        if not config.hard_transitions:
            smooth = np.ones(n, dtype=bool)

        eligible, width, notes = None, None, []
        for mult in BAND_WIDENING:
            width = arc.tolerance * mult
            in_band = base & (np.abs(energy - target) <= width)
            for mask, note in (
                (in_band & kind_ok & smooth, None),
                (in_band & smooth, "ratio relaxed: no in-band track of the wanted kind"),
                (in_band & kind_ok, "transition rule relaxed: every in-band track would jar"),
                (in_band, "ratio and transition rule relaxed"),
            ):
                if mask.any():
                    eligible = mask
                    if note:
                        notes.append(note)
                    break
            if eligible is not None:
                if mult > 1.0:
                    notes.append(f"band widened to +/-{width:.2f}")
                break
        if eligible is None:
            eligible = base
            notes.append("no track within the widest band; arc constraint broken here")
        if not eligible.any():
            break

        idx = np.flatnonzero(eligible)
        coherence = _rank01(coherence_raw[idx])
        arc_fit = np.clip(1 - np.abs(energy[idx] - target[idx]) / arc.tolerance, 0, 1)
        total = (
            config.w_arc * arc_fit
            + config.w_coherence * coherence
            + config.w_taste * taste[idx]
            + config.w_theme * theme_fit[idx]
            - config.w_transition * transition[idx]
        )
        order = np.argsort(-total)
        best_k = order[0]
        best = idx[best_k]
        # Every score term, aligned to `idx` (position k = candidate idx[k]).
        terms = {
            "arc_fit": arc_fit,
            "coherence": coherence,
            "taste": taste[idx],
            "theme": theme_fit[idx],
            "transition_penalty": transition[idx],
            "total": total,
        }

        steps.append({
            "slot": len(picks),
            "elapsed_s": round(elapsed, 1),
            "phase": arc.phase(float(t_mid[best])),
            "target_energy": round(float(target[best]), 3),
            "band": round(float(width), 3),
            "wanted": "familiar" if want_familiar else "fresh",
            "n_eligible": int(eligible.sum()),
            "notes": notes,
            "shortlist": [
                {
                    "track_id": tracks["track_id"].iat[idx[k]],
                    "name": tracks["name"].iat[idx[k]],
                    "artist": artists[idx[k]],
                    "energy": round(float(energy[idx[k]]), 3),
                    "tempo": None if np.isnan(tempo[idx[k]]) else round(float(tempo[idx[k]]), 1),
                    **_scores_at(terms, k),
                }
                for k in order[: config.shortlist]
            ],
        })

        picks.append(Pick(
            track_id=tracks["track_id"].iat[best],
            name=tracks["name"].iat[best],
            artist=artists[best],
            start_s=round(elapsed, 1),
            duration_s=round(float(duration_s[best]), 1),
            t_mid=round(float(t_mid[best]), 3),
            phase=arc.phase(float(t_mid[best])),
            target_energy=round(float(target[best]), 3),
            energy=round(float(energy[best]), 3),
            tempo=None if np.isnan(tempo[best]) else round(float(tempo[best]), 1),
            familiar=bool(familiar[best]),
            fly_valence=round(float(tracks["fly_valence"].iat[best]), 3),
            hamming_to_prev=None if prev_dist is None else int(prev_dist[best]),
            scores=_scores_at(terms, best_k),
        ))
        available[best] = False
        artist_count[artists[best]] += 1
        elapsed += duration_s[best]
        prev = best

    return Selection(picks=picks, steps=steps, config=config, excluded=sorted(exclude))
