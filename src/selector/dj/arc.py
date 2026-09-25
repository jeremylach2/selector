"""Stage 2, Arc: the energy curve the set has to follow.

An `EnergyArc` maps elapsed fraction of the set (0 at the first second, 1
at the last) to a target **measured** energy, through four phases: opener,
build, peak, comedown. It is a hard constraint: Select only considers
tracks within `tolerance` of the target at the point in the set where the
track would play, and Critique rejects a set whose realised curve leaves
the band.

Energy here is `selector.dj.pool.measured_energy` -- DSP loudness, onset
density and spectral brightness from the Step 9 audio features, rank-
normalised over the crate. Predicted labels never enter the arc.

The same module owns the transition rules (how far tempo and energy may
jump between neighbouring tracks), since they are the arc's local form:
the curve says where the set should be, the rules say how it may move.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from selector.dj.brief import Theme

# (name, start_frac, end_frac, shape_at_start, shape_at_end). Shape is in
# [0, 1] and is mapped onto the theme's [energy_floor, energy_ceiling], so
# every theme keeps the same opener/build/peak/comedown silhouette at its
# own energy level.
PHASES: tuple[tuple[str, float, float, float, float], ...] = (
    ("opener", 0.00, 0.15, 0.20, 0.20),
    ("build", 0.15, 0.55, 0.20, 0.90),
    ("peak", 0.55, 0.80, 0.95, 0.95),
    ("comedown", 0.80, 1.00, 0.95, 0.25),
)

DEFAULT_TOLERANCE = 0.15

# Transition rules. Tempo shift is octave-folded (see `tempo_shift`), so
# 70 -> 140 BPM counts as a match, the way a DJ would hear it. The limits
# are looser than a beatmatching DJ's ~8% pitch range because this is a
# playlist with gaps between songs, not a continuous mix.
MAX_TEMPO_SHIFT = 0.12
MAX_ENERGY_STEP = 0.25

# A transition is *jarring* (Critique rejects it) past these limits alone,
# or when tempo and energy both move a lot at once: either by itself can be
# a deliberate gear change, both together reads as a different set.
HARD_TEMPO_SHIFT = 2 * MAX_TEMPO_SHIFT
HARD_ENERGY_STEP = MAX_ENERGY_STEP
COMBINED_TEMPO_SHIFT = MAX_TEMPO_SHIFT
COMBINED_ENERGY_STEP = MAX_ENERGY_STEP / 2


@dataclass(frozen=True)
class EnergyArc:
    minutes: float
    energy_floor: float
    energy_ceiling: float
    tolerance: float = DEFAULT_TOLERANCE

    @classmethod
    def for_theme(cls, theme: Theme, minutes: float, tolerance: float = DEFAULT_TOLERANCE) -> EnergyArc:
        return cls(minutes, theme.energy_floor, theme.energy_ceiling, tolerance)

    @property
    def seconds(self) -> float:
        return self.minutes * 60

    def _segment(self, t: float) -> tuple[str, float, float, float, float]:
        t = min(max(t, 0.0), 1.0)
        for phase in PHASES:
            if t < phase[2]:
                return phase
        return PHASES[-1]

    def phase(self, t: float) -> str:
        return self._segment(t)[0]

    def target(self, t: float) -> float:
        """Target measured energy at elapsed fraction `t` of the set."""
        _, start, end, s0, s1 = self._segment(t)
        u = (min(max(t, 0.0), 1.0) - start) / (end - start)
        shape = s0 + (s1 - s0) * u
        return self.energy_floor + shape * (self.energy_ceiling - self.energy_floor)

    def in_band(self, t: float, energy: float) -> bool:
        return abs(energy - self.target(t)) <= self.tolerance

    def sample(self, n: int = 21) -> list[dict]:
        """Evenly spaced points on the curve, for the run log and plots."""
        return [
            {"t": round(i / (n - 1), 3), "phase": self.phase(i / (n - 1)),
             "target_energy": round(self.target(i / (n - 1)), 3)}
            for i in range(n)
        ]

    def to_dict(self) -> dict:
        return {**asdict(self), "phases": [p[0] for p in PHASES], "curve": self.sample()}


def tempo_shift(bpm_a: float | None, bpm_b: float | None) -> float | None:
    """Fractional tempo change from `a` to `b`, folded across octaves (half-
    and double-time count as the same pulse). `None` if either is missing.

    librosa's beat tracker makes octave errors routinely, so without the
    fold a 90 BPM track measured as 180 would look like a 100% jump.
    """
    if not bpm_a or not bpm_b or math.isnan(bpm_a) or math.isnan(bpm_b):
        return None
    return min(abs(bpm_b * k - bpm_a) / bpm_a for k in (0.5, 1.0, 2.0))


def jarring_reason(
    tempo_a: float | None, energy_a: float, tempo_b: float | None, energy_b: float
) -> str | None:
    """Why the move from track a to track b is jarring, or `None` if it isn't."""
    shift = tempo_shift(tempo_a, tempo_b)
    step = abs(energy_b - energy_a)
    if step > HARD_ENERGY_STEP:
        return f"energy jump of {step:.2f}"
    if shift is not None and shift > HARD_TEMPO_SHIFT:
        return f"tempo lurch of {shift:.0%}"
    if shift is not None and shift > COMBINED_TEMPO_SHIFT and step > COMBINED_ENERGY_STEP:
        return f"tempo {shift:.0%} and energy {step:.2f} move together"
    return None
