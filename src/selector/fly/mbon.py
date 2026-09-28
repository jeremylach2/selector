"""Mushroom body plasticity: learning taste from a listener's skips.

Biology: dopaminergic neurons gate depression of Kenyon-cell to mushroom-body
output-neuron (KC->MBON) synapses. A fly learns that an odour predicts
punishment by *weakening* the KC->MBON synapses that were active when the
punishment arrived, not by strengthening anything. This module implements
that rule with two output neurons, approach and avoid, whose balance is the
fly's predicted taste score for a track.

The supervision signal is already in the warehouse: `verdict` in
`data/plays.parquet` is `+1` for a played-out or replayed track, `-1` for a
skip, `0` for neutral (see `selector.ingest.load_history`). Training must
replay plays in chronological order, this is a time series, and shuffling
it would let the model learn from its own future.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = ["MushroomBody"]


def _active_indices(tag) -> np.ndarray:
    """Column indices of the active (firing) Kenyon cells in a single tag row."""
    if sparse.issparse(tag):
        tag = sparse.csr_matrix(tag)
        if tag.shape[0] != 1:
            tag = tag.reshape(1, -1)
        return tag.indices
    tag = np.asarray(tag).ravel()
    return np.flatnonzero(tag)


class MushroomBody:
    """Two mushroom-body output neurons (MBONs), approach and avoid, reading
    off a shared population of Kenyon cells through plastic synapses.

    Parameters
    ----------
    fly_hash:
        A fitted `selector.fly.lsh.FlyHash` (or anything exposing `.n_kc`),
        the KC population this mushroom body reads from.
    n_mbon:
        Fixed at 2 (approach, avoid); accepted as a parameter for interface
        symmetry with the rest of the fly module, but any other value is
        rejected since the valence computation below is specifically an
        approach-minus-avoid contrast.
    lr:
        Fraction each active KC's synapse is depressed by per learning event.
    decay:
        Fraction of the gap back to the naive baseline weight (1.0) that
        recovers after every `learn()` call, modelling gradual forgetting so
        old punishments don't permanently veto a track.
    """

    def __init__(
        self,
        fly_hash,
        n_mbon: int = 2,
        lr: float = 0.05,
        decay: float = 0.0,
    ) -> None:
        if n_mbon != 2:
            raise ValueError("MushroomBody models exactly one approach and one avoid MBON")
        if not 0 < lr <= 1:
            raise ValueError("lr must be in (0, 1]")
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")

        self.lr = lr
        self.decay = decay
        n_kc = fly_hash.n_kc
        self.w_approach = np.ones(n_kc, dtype=np.float64)
        self.w_avoid = np.ones(n_kc, dtype=np.float64)

    def learn(self, tag, verdict: int) -> None:
        """Depress the KC->MBON synapses active in `tag` on the MBON matching
        the punishment's valence. A reward (`verdict > 0`) depresses the
        *avoid* synapses of the active KCs, raising net approach for tags
        like this one in future; a punishment (`verdict < 0`) depresses the
        *approach* synapses instead. `verdict == 0` (neutral/ambiguous plays)
        makes no update, there is no dopaminergic signal to gate anything.
        Only the KCs active in `tag` are touched, which is the whole point:
        this circuit generalises across tracks only through tag overlap.
        """
        active = _active_indices(tag)
        if active.size == 0:
            return

        if verdict > 0:
            self.w_avoid[active] *= 1.0 - self.lr
        elif verdict < 0:
            self.w_approach[active] *= 1.0 - self.lr

        if self.decay:
            self.w_approach += self.decay * (1.0 - self.w_approach)
            self.w_avoid += self.decay * (1.0 - self.w_avoid)

    def valence(self, tag) -> float:
        """Approach minus avoid drive: the fly's predicted taste score for a
        track carrying this tag. Positive means approach, negative avoid.
        """
        active = _active_indices(tag)
        if active.size == 0:
            return 0.0
        return float(self.w_approach[active].sum() - self.w_avoid[active].sum())
