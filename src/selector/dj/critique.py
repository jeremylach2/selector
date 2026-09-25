"""Stage 4, Critique: does the assembled set actually flow?

Reads a finished `Selection` back against the arc it was built for and
checks what a listener would notice:

- **off-arc** -- a track's measured energy sits outside the arc's band at
  the point it plays (Select can be forced there by band widening),
- **jarring transition** -- a tempo or energy lurch between neighbours
  beyond the transition rules in `selector.dj.arc`,
- **flat shape** -- the realised peak isn't clearly above the opener and
  the close, i.e. the set never actually builds and releases,
- **length** -- the set misses the requested running time badly,
- **artist cap** -- more than `max_per_artist` tracks by one artist.

Select is greedy: it only ever looks one track back, so it can paint itself
into a corner that only shows up when the whole set is read at once. That
is what this pass is for. A failed verdict carries a `revision` -- tracks to
exclude and a heavier transition weight -- that the agent hands back to
Select exactly once. Nothing reaches Spotify unless a verdict passes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from selector.dj.arc import EnergyArc, jarring_reason, tempo_shift
from selector.dj.select import SelectConfig, Selection

MIN_PEAK_LIFT = 0.10
LENGTH_TOLERANCE = 0.15
REVISION_TRANSITION_MULTIPLIER = 2.0


@dataclass
class Issue:
    slot: int
    kind: str
    detail: str


@dataclass
class Verdict:
    passed: bool
    issues: list[Issue]
    realised: list[dict]
    summary: str
    revision_exclude: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**asdict(self), "issues": [asdict(i) for i in self.issues]}


def _transition_issue(slot: int, a, b) -> Issue | None:
    reason = jarring_reason(a.tempo, a.energy, b.tempo, b.energy)
    if reason is None:
        return None
    measured = tempo_shift(a.tempo, b.tempo) is not None
    tempo_txt = f"{a.tempo:.0f}->{b.tempo:.0f} BPM" if measured else "tempo unmeasured"
    return Issue(slot, "jarring", f"{reason} ({tempo_txt}, energy {a.energy:.2f}->{b.energy:.2f})")


def critique(selection: Selection, arc: EnergyArc) -> Verdict:
    picks = selection.picks
    issues: list[Issue] = []

    if not picks:
        return Verdict(False, [Issue(0, "empty", "Select produced no tracks")], [], "Empty set.")

    for i, p in enumerate(picks):
        if abs(p.energy - p.target_energy) > arc.tolerance:
            issues.append(Issue(
                i, "off_arc",
                f"energy {p.energy:.2f} vs target {p.target_energy:.2f} in the {p.phase}",
            ))

    for i in range(1, len(picks)):
        issue = _transition_issue(i, picks[i - 1], picks[i])
        if issue:
            issues.append(issue)

    by_phase: dict[str, list[float]] = {}
    for p in picks:
        by_phase.setdefault(p.phase, []).append(p.energy)
    peak = np.mean(by_phase.get("peak", [np.nan]))
    edges = [picks[0].energy, picks[-1].energy]
    if np.isnan(peak) or peak - max(edges) < MIN_PEAK_LIFT:
        issues.append(Issue(
            -1, "flat_shape",
            f"peak phase averages {peak:.2f} against opener {edges[0]:.2f} and close {edges[1]:.2f}; "
            f"needs a lift of {MIN_PEAK_LIFT:.2f}",
        ))

    off_by = selection.seconds / arc.seconds - 1
    if abs(off_by) > LENGTH_TOLERANCE:
        issues.append(Issue(-1, "length", f"runs {selection.seconds / 60:.1f} min for a {arc.minutes:g}-min brief"))

    counts = Counter(p.artist for p in picks)
    for artist, c in counts.items():
        if c > selection.config.max_per_artist:
            issues.append(Issue(-1, "artist_cap", f"{artist} appears {c} times"))

    realised = [
        {"slot": i, "t_mid": p.t_mid, "target": p.target_energy, "energy": p.energy, "tempo": p.tempo}
        for i, p in enumerate(picks)
    ]

    # The track to swap out is the one that caused the problem: the off-arc
    # track itself, or the *incoming* side of a jarring transition.
    exclude = sorted({picks[i.slot].track_id for i in issues if i.slot >= 0})

    passed = not issues
    if passed:
        summary = (
            f"Pass: {len(picks)} tracks, {selection.seconds / 60:.1f} min, every track inside the "
            f"+/-{arc.tolerance:.2f} band, no jarring transitions, peak lifts "
            f"{peak - max(edges):.2f} above the edges."
        )
    else:
        kinds = Counter(i.kind for i in issues)
        summary = "Reject: " + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in kinds.items()) + "."
    return Verdict(passed, issues, realised, summary, revision_exclude=exclude)


def revised_config(config: SelectConfig) -> SelectConfig:
    """What Select runs with on its one revision: the transition rules
    Critique enforces become a hard filter instead of a soft penalty, and
    transitions weigh double in the score, since most rejections are a
    greedy pick that scored well on taste but lurched from its neighbour."""
    return replace(
        config,
        w_transition=config.w_transition * REVISION_TRANSITION_MULTIPLIER,
        hard_transitions=True,
    )
