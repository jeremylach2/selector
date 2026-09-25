"""The fruit fly olfactory circuit, implemented as a locality-sensitive hash.

Reference: Dasgupta, Stevens & Navlakha, "A neural algorithm for a fundamental
computing problem", Science 358:6364 (2017). The circuit: a small number of
projection neurons (PNs) fan out to a much larger population of Kenyon cells
(KCs) through a sparse binary random projection, where each KC samples a
handful of PNs. Inhibitory feedback from the APL neuron then applies
winner-take-all so only the top few percent of KCs fire. Similar inputs land
on similar sparse binary tags, which is a locality-sensitive hash.

This module implements the *idealised* circuit with a random PN->KC
projection. Step 6 (`selector.fly.connectome`) swaps that random projection
for the real FlyWire connectome via the `projection_matrix` property, which is
the seam this module exists to provide.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = ["FlyHash", "hamming_distances", "hamming_top_k"]


def hamming_distances(
    query_tag: sparse.spmatrix | np.ndarray,
    corpus: sparse.spmatrix,
) -> np.ndarray:
    """Hamming distance from `query_tag` to every row of `corpus`.

    For two binary vectors, Hamming distance = |a| + |b| - 2*|a & b|, so this
    is a single sparse matrix-vector product rather than any dense arrays.
    """
    corpus = sparse.csr_matrix(corpus).astype(np.float64)

    query = sparse.csr_matrix(query_tag).astype(np.float64)
    if query.shape[0] != 1:
        query = query.reshape(1, -1)

    intersection = np.asarray(corpus.dot(query.T).todense()).ravel()
    corpus_popcount = np.asarray(corpus.sum(axis=1)).ravel()
    return corpus_popcount + query.sum() - 2 * intersection


def hamming_top_k(
    query_tag: sparse.spmatrix | np.ndarray,
    corpus: sparse.spmatrix,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k nearest rows of `corpus` to `query_tag` by Hamming distance.

    Free function (not a `FlyHash` method) so callers who only have a tag
    matrix on hand -- e.g. `selector.fly.pipeline`, which persists tags
    separately from the `FlyHash` that produced them -- don't need to keep a
    fitted `FlyHash` instance around just to call this. `FlyHash.hamming_neighbours`
    below is a thin wrapper kept for backward compatibility.

    Returns `(indices, distances)`, both length `min(k, n_candidates)`,
    sorted nearest first.
    """
    distances = hamming_distances(query_tag, corpus)

    k = min(k, distances.shape[0])
    nearest = np.argpartition(distances, k - 1)[:k]
    order = np.argsort(distances[nearest])
    nearest = nearest[order]
    return nearest, distances[nearest]


class FlyHash:
    """Locality-sensitive hash modelled on the fly olfactory circuit.

    Parameters
    ----------
    d_in:
        Dimensionality of the input feature vectors (the PN count).
    n_kc:
        Number of Kenyon cells. The paper uses roughly 2,000 for ~50 PNs, an
        expansion of about 20x into a higher-dimensional space.
    sample_size:
        Number of PNs each KC samples from. The paper measures about 6 on
        average from the real fly.
    hash_len:
        If set, exactly this many KCs are kept active by winner-take-all,
        overriding `wta_frac`. Useful for comparing hash quality at a fixed
        code length rather than a fixed sparsity fraction.
    wta_frac:
        Fraction of KCs the APL neuron's inhibition leaves firing. The real
        fly keeps roughly 5% of Kenyon cells active.
    seed:
        Seed for the random PN->KC projection. Ignored once
        `projection_matrix` is set explicitly (e.g. from the real connectome).
    """

    def __init__(
        self,
        d_in: int,
        n_kc: int = 2000,
        sample_size: int = 6,
        hash_len: int | None = None,
        wta_frac: float = 0.05,
        seed: int | None = None,
    ) -> None:
        if d_in <= 0:
            raise ValueError("d_in must be positive")
        if not 0 < wta_frac <= 1:
            raise ValueError("wta_frac must be in (0, 1]")

        self.d_in = d_in
        self.sample_size = sample_size
        self.hash_len = hash_len
        self.wta_frac = wta_frac
        self._rng = np.random.default_rng(seed)

        self._projection: sparse.csr_matrix = self._random_projection(n_kc)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.tags_: sparse.csr_matrix | None = None

    # -- projection ---------------------------------------------------

    def _random_projection(self, n_kc: int) -> sparse.csr_matrix:
        """A sparse, unweighted, random PN->KC projection, one row per KC.

        Each KC samples `sample_size` distinct PNs uniformly at random and
        sums their (normalised) activity — connections are binary presence,
        not learned weights, matching the biological circuit.
        """
        sample_size = min(self.sample_size, self.d_in)
        rows = np.repeat(np.arange(n_kc), sample_size)
        cols = np.concatenate(
            [self._rng.choice(self.d_in, size=sample_size, replace=False) for _ in range(n_kc)]
        )
        data = np.ones(rows.shape[0], dtype=np.float64)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_kc, self.d_in))

    @property
    def n_kc(self) -> int:
        return self._projection.shape[0]

    @property
    def projection_matrix(self) -> sparse.csr_matrix:
        """The PN->KC connectivity matrix, shape (n_kc, d_in)."""
        return self._projection

    @projection_matrix.setter
    def projection_matrix(self, matrix) -> None:
        """Replace the random projection — the seam Step 6 plugs the real
        FlyWire connectome into. `matrix` must be (n_kc, d_in) shaped and is
        stored as a sparse CSR matrix regardless of the input format.
        """
        matrix = sparse.csr_matrix(matrix)
        if matrix.shape[1] != self.d_in:
            raise ValueError(
                f"projection matrix has {matrix.shape[1]} columns, expected d_in={self.d_in}"
            )
        self._projection = matrix

    # -- fit / transform ------------------------------------------------

    def fit(self, X: np.ndarray) -> FlyHash:
        """Compute per-dimension centring statistics from `X`.

        This is the paper's divisive normalisation step: real PN firing rates
        are mean-centred and scaled before the projection, which is what
        makes tags comparable across inputs of different overall magnitude
        (a loud, dense track and a quiet, sparse one should not just get
        "more bits on" — they should be normalised onto the same footing
        before the circuit compares them). Skipping this step is the most
        common way to silently break the hash.
        """
        X = np.asarray(X, dtype=np.float64)
        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self._std = std
        return self

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("call fit() before transform()")
        centred = (X - self._mean) / self._std
        # Firing rates are non-negative; rectify after centring.
        return np.clip(centred, 0.0, None)

    def _wta_count(self) -> int:
        if self.hash_len is not None:
            return min(int(self.hash_len), self.n_kc)
        return max(1, round(self.wta_frac * self.n_kc))

    def transform(self, X: np.ndarray) -> sparse.csr_matrix:
        """Project and apply winner-take-all. Returns a sparse boolean matrix
        of shape (n_samples, n_kc), and caches it on `self.tags_` so
        `hamming_neighbours` can be called against it without re-passing it.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        normalised = self._normalise(X)
        # Sparse PN->KC projection times dense input, kept sparse to scale to
        # the real connectome's PN/KC counts without densifying it.
        activations = self._projection.dot(normalised.T).T

        k = self._wta_count()
        n_samples, n_kc = activations.shape
        k = min(k, n_kc)

        if k == n_kc:
            keep = np.ones_like(activations, dtype=bool)
        else:
            threshold_idx = np.argpartition(-activations, k - 1, axis=1)[:, :k]
            rows = np.repeat(np.arange(n_samples), k)
            keep = np.zeros_like(activations, dtype=bool)
            keep[rows, threshold_idx.ravel()] = True

        tags = sparse.csr_matrix(keep)
        self.tags_ = tags
        return tags

    # -- retrieval --------------------------------------------------------

    def hamming_neighbours(
        self,
        query_tag: sparse.spmatrix | np.ndarray,
        k: int,
        tags: sparse.spmatrix | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k nearest tags to `query_tag` by Hamming distance.

        `tags` defaults to the corpus cached by the last `transform()` call.
        Delegates to the free function `hamming_top_k`; see there for the
        distance computation.

        Returns `(indices, distances)`, both length `min(k, n_candidates)`,
        sorted nearest first.
        """
        corpus = tags if tags is not None else self.tags_
        if corpus is None:
            raise RuntimeError("no tag matrix available: call transform() first or pass tags=")
        return hamming_top_k(query_tag, corpus, k)
