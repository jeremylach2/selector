"""Sequence a list of tracks someone else chose along the DJ's energy arc.

`dj_set` chooses the tracks and orders them. This is the second half on
its own, for a list picked by the user or the model: every given track is
kept, and only the order is decided.

1. **Arc.** The same opener/build/peak/comedown curve (`selector.dj.arc`),
   but fitted to the measured energy range of the given tracks rather than
   a theme's, so the quietest track can open and the loudest can peak.
2. **Assign.** The k-th lowest-energy track goes to the k-th lowest arc
   target. For distance to the curve alone, that is the best order.
3. **Polish.** Swaps within a few positions are kept while they lower the
   total cost: distance from the arc, the transition penalty Select uses,
   a heavy penalty for any transition Critique would call jarring, and a
   bonus for fly-brain similarity between neighbours.
4. **Critique.** The result is read back by the same Critique as a
   `dj_set`, so the report says plainly when a list can't be made to
   flow (no real peak, an unavoidable lurch).

Tracks not in the crate have no measured energy, so there's nothing to
place them by. They are appended after the sequenced tracks, in the order
given, and flagged rather than dropped.

No scipy, so the hosted server runs this on the deploy crate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from selector.dj.arc import (
    COMBINED_ENERGY_STEP,
    COMBINED_TEMPO_SHIFT,
    HARD_ENERGY_STEP,
    HARD_TEMPO_SHIFT,
    MAX_ENERGY_STEP,
    MAX_TEMPO_SHIFT,
    PHASES,
    EnergyArc,
)
from selector.dj.commit import _fly_phrase, _fmt_tempo, _sentence, transition_note
from selector.dj.crate import Crate
from selector.dj.critique import Verdict, critique
from selector.dj.select import Pick, SelectConfig, Selection, _dice

MIN_TRACKS = 2
MAX_TRACKS = 100
SWAP_WINDOW = 3
MAX_PASSES = 8

W_TRANSITION = 0.5
W_JARRING = 2.0
W_COHERENCE = 0.25

# The arc's shape runs between these (see `PHASES`); the fitted arc maps
# them onto the given tracks' lowest and highest energy.
SHAPE_LOW = min(min(p[3], p[4]) for p in PHASES)
SHAPE_HIGH = max(max(p[3], p[4]) for p in PHASES)


@dataclass
class Ordering:
    picks: list[Pick]
    unplaced: list[str]
    arc: EnergyArc
    verdict: Verdict
    duplicates: int

    @property
    def track_ids(self) -> list[str]:
        return [p.track_id for p in self.picks] + self.unplaced

    @property
    def uris(self) -> list[str]:
        return [f"spotify:track:{t}" for t in self.track_ids]


def fit_arc(energy: np.ndarray, minutes: float) -> EnergyArc:
    """An arc whose lowest target is the quietest track and whose peak is
    the loudest."""
    lo, hi = float(energy.min()), float(energy.max())
    span = max(hi - lo, 1e-6) / (SHAPE_HIGH - SHAPE_LOW)
    floor = lo - SHAPE_LOW * span
    return EnergyArc(minutes=minutes, energy_floor=floor, energy_ceiling=floor + span)


def _tempo_shifts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised `selector.dj.arc.tempo_shift`, NaN where either is unmeasured."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([np.abs(b * k - a) / a for k in (0.5, 1.0, 2.0)]).min(axis=0)


def _midpoints(duration: np.ndarray) -> np.ndarray:
    start = np.concatenate([[0.0], np.cumsum(duration)[:-1]])
    return (start + duration / 2) / duration.sum()


class _Cost:
    def __init__(self, energy, tempo, duration, sim, arc: EnergyArc):
        self.energy, self.tempo, self.duration, self.sim, self.arc = energy, tempo, duration, sim, arc

    def __call__(self, order: np.ndarray) -> float:
        e = self.energy[order]
        target = np.array([self.arc.target(t) for t in _midpoints(self.duration[order])])
        arc_cost = np.abs(e - target).sum() / self.arc.tolerance

        a, b = order[:-1], order[1:]
        shifts = _tempo_shifts(self.tempo[a], self.tempo[b])
        steps = np.abs(self.energy[b] - self.energy[a])
        tempo_pen = np.nan_to_num(shifts / MAX_TEMPO_SHIFT, nan=0.5)
        transition = np.minimum(tempo_pen, 2.0) / 2 + np.minimum(steps / MAX_ENERGY_STEP, 2.0) / 2
        # NaN shifts compare False, as in `jarring_reason`.
        jarring = (
            (steps > HARD_ENERGY_STEP)
            | (shifts > HARD_TEMPO_SHIFT)
            | ((shifts > COMBINED_TEMPO_SHIFT) & (steps > COMBINED_ENERGY_STEP))
        )
        return float(
            arc_cost
            + W_TRANSITION * transition.sum()
            + W_JARRING * jarring.sum()
            - W_COHERENCE * self.sim[a, b].sum()
        )


def _assign(energy: np.ndarray, arc: EnergyArc) -> np.ndarray:
    """Sort-match tracks to equal-width slots on the arc."""
    n = len(energy)
    targets = np.array([arc.target((k + 0.5) / n) for k in range(n)])
    order = np.empty(n, dtype=np.int64)
    order[np.argsort(targets, kind="stable")] = np.argsort(energy, kind="stable")
    return order


def _polish(order: np.ndarray, cost: _Cost) -> np.ndarray:
    best = cost(order)
    for _ in range(MAX_PASSES):
        improved = False
        for i in range(len(order)):
            for j in range(i + 1, min(i + SWAP_WINDOW + 1, len(order))):
                order[i], order[j] = order[j], order[i]
                c = cost(order)
                if c < best - 1e-9:
                    best, improved = c, True
                else:
                    order[i], order[j] = order[j], order[i]
        if not improved:
            break
    return order


def order_tracks(crate: Crate, track_ids: list[str]) -> Ordering:
    """Order `track_ids` (bare IDs) along a fitted energy arc. Raises
    `ValueError` for a list too short or too long to sequence."""
    unique = list(dict.fromkeys(track_ids))
    if not MIN_TRACKS <= len(unique) <= MAX_TRACKS:
        raise ValueError(f"Refused: pass {MIN_TRACKS} to {MAX_TRACKS} distinct tracks (got {len(unique)}).")

    by_id = crate.by_id()
    placed = [t for t in unique if t in by_id.index]
    unplaced = [t for t in unique if t not in by_id.index]
    if len(placed) < MIN_TRACKS:
        raise ValueError(
            f"Only {len(placed)} of these {len(unique)} tracks have measured audio, so there's no energy "
            "to order them by."
        )

    rows = by_id.loc[placed]
    energy = rows["energy"].to_numpy(dtype=float)
    tempo = rows["tempo"].to_numpy(dtype=float)
    duration = rows["duration_ms"].to_numpy(dtype=float) / 1000
    tags = crate.tags.take(rows["tag_row"].to_numpy())
    popcount = tags.popcount
    dist = np.stack([tags.hamming(r) for r in range(len(placed))])
    sim = _dice(dist, popcount[None, :], popcount[:, None])

    arc = fit_arc(energy, duration.sum() / 60)
    cost = _Cost(energy, tempo, duration, sim, arc)
    order = _polish(_assign(energy, arc), cost)

    picks: list[Pick] = []
    elapsed = 0.0
    t_mid = _midpoints(duration[order])
    for k, i in enumerate(order):
        row = rows.iloc[i]
        picks.append(Pick(
            track_id=row["track_id"],
            name=row["name"],
            artist=row["artist"],
            start_s=round(elapsed, 1),
            duration_s=round(float(duration[i]), 1),
            t_mid=round(float(t_mid[k]), 3),
            phase=arc.phase(float(t_mid[k])),
            target_energy=round(arc.target(float(t_mid[k])), 3),
            energy=round(float(energy[i]), 3),
            tempo=None if np.isnan(tempo[i]) else round(float(tempo[i]), 1),
            familiar=bool(row["familiar"]),
            fly_valence=round(float(row["fly_valence"]), 3),
            hamming_to_prev=None if k == 0 else int(dist[order[k - 1], i]),
            scores={"taste": round(float(row["taste"]), 3)},
        ))
        elapsed += duration[i]

    # The caller chose these tracks, so no artist cap applies.
    selection = Selection(picks=picks, config=SelectConfig(max_per_artist=len(picks)))
    return Ordering(picks, unplaced, arc, critique(selection, arc), len(track_ids) - len(unique))


def render(ordering: Ordering, labels: dict[str, tuple[str, str]] | None = None) -> str:
    """The order with a note per transition, the critique, and the URIs.
    `labels` maps an unplaced track's ID to `(name, artist)` when known."""
    picks, verdict = ordering.picks, ordering.verdict
    minutes = sum(p.duration_s for p in picks) / 60
    lo, hi = min(p.energy for p in picks), max(p.energy for p in picks)
    lines = [
        (
            f"**{len(picks)} tracks, {minutes:.0f} minutes**, ordered opener -> build -> peak -> comedown "
            f"on an arc fitted to their measured energy ({lo:.2f}-{hi:.2f})."
        ),
        "",
        f"**Flow check.** {verdict.summary}" if verdict.passed else (
            f"**Flow check.** {verdict.summary.removeprefix('Reject: ')} This is still the smoothest order "
            "found: the rough spots come from the tracks given. Off arc means too few tracks at the energy "
            "that part of the curve wants; jarring means no neighbour close enough in tempo or energy. "
            "Swapping tracks in or out fixes them."
        ),
    ]
    for issue in verdict.issues[:6]:
        where = f"track {issue.slot + 1}" if issue.slot >= 0 else "whole set"
        lines.append(f"- {where}: {issue.kind.replace('_', ' ')} - {issue.detail}")
    lines.append("")
    for i, p in enumerate(picks):
        clock = f"{int(p.start_s // 60)}:{int(p.start_s % 60):02d}"
        lines.append(f"{i + 1}. **{p.name}** - {p.artist}  `{clock}` · {p.phase} · {_fmt_tempo(p)} · energy {p.energy:.2f}")
        if i == 0:
            lines.append(f"   Opens at energy {p.energy:.2f}. {_sentence(_fly_phrase(p))}.")
        else:
            lines.append(f"   {transition_note(picks[i - 1], p)}")

    if ordering.unplaced:
        labels = labels or {}
        lines += ["", "**Appended at the end, unordered** (no measured audio, so nothing to place them by):"]
        for t in ordering.unplaced:
            name, artist = labels.get(t, ("(unknown)", ""))
            lines.append(f"- {name}{f' - {artist}' if artist else ''} (`{t}`)")
    if ordering.duplicates:
        lines += ["", f"_{ordering.duplicates} duplicate{'s' if ordering.duplicates != 1 else ''} dropped._"]

    lines += ["", "URIs in this order, ready for `spotify_create_playlist`:", "", json.dumps(ordering.uris)]
    return "\n".join(lines)
