"""Shared machinery for comparing FlyHash against classical random-projection
LSH, used by both `tests/test_fly_lsh.py` (a fast synthetic check that runs on
every test invocation) and `notebooks/fly_validation.py` (the slower
reproduction of the paper's claim on real MNIST data).

The comparison follows Dasgupta, Stevens & Navlakha (Science, 2017): at a
fixed hash length `k`, FlyHash should retrieve true nearest neighbours more
accurately than a classical dense random projection of the same length,
especially for small `k`. "Hash length" here means the number of *on* bits
compared: for classical LSH that is the full k-bit code; for FlyHash it is
`hash_len=k` Kenyon cells firing out of a `kc_expansion * k`-cell population,
matching the paper's roughly 20x expansion from PNs into Kenyon cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from selector.fly.lsh import FlyHash

__all__ = [
    "BenchmarkResult",
    "benchmark_hash_lengths",
    "classical_lsh_transform",
    "mean_average_precision",
]


def pool_features(X: np.ndarray, n_out: int) -> np.ndarray:
    """Reduce `X` from `d_in` columns down to `n_out` by grouping columns
    `j -> j % n_out` and summing. Used to bring MNIST's 784 pixels down to
    the real FlyWire PN count for `notebooks/fly_connectome_validation.py`
    without going through PCA, whose signed components would need rectifying
    away half their signal by FlyHash's non-negative firing-rate assumption
    (see `FlyHash._normalise`). Summing raw non-negative pixel intensities
    keeps that assumption valid, at the cost of losing spatial locality
    within a pooled bucket, an honest trade documented here rather than
    in a code comment nobody reads before wondering why mAP looks low.
    """
    n_in = X.shape[1]
    bucket = np.arange(n_in) % n_out
    pooled = np.zeros((X.shape[0], n_out), dtype=X.dtype)
    np.add.at(pooled.T, bucket, X.T)
    return pooled


def classical_lsh_transform(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Dense random-projection LSH ("SimHash"): sign of a projection onto `k`
    random hyperplanes. This is the classical baseline the fly circuit is
    compared against, a k-bit code with no expansion and no sparsification.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    planes = rng.normal(size=(d, k))
    return (X @ planes) > 0


def _hamming_all(tags, query_row_idx: int) -> np.ndarray:
    """Hamming distance from tags[query_row_idx] to every row of `tags`.
    Works for either a dense boolean ndarray or a sparse boolean matrix.
    """
    if sparse.issparse(tags):
        tags = sparse.csr_matrix(tags).astype(np.float64)
        query = tags[query_row_idx]
        intersection = np.asarray(tags.dot(query.T).todense()).ravel()
        popcount = np.asarray(tags.sum(axis=1)).ravel()
        return popcount + popcount[query_row_idx] - 2 * intersection
    return (tags != tags[query_row_idx]).sum(axis=1)


def average_precision(ranked_indices: np.ndarray, relevant: set[int]) -> float:
    """Standard information-retrieval average precision: rank the whole
    corpus by hash distance, and reward relevant items appearing early.
    """
    if not relevant:
        return 0.0
    hits = 0
    precisions = []
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in relevant:
            hits += 1
            precisions.append(hits / rank)
        if hits == len(relevant):
            break
    return sum(precisions) / len(relevant)


def mean_average_precision(
    tags,
    query_indices: np.ndarray,
    true_neighbour_sets: list[set[int]],
) -> float:
    """Mean AP over queries, ranking the corpus by Hamming distance in `tags`."""
    aps = []
    for qi, relevant in zip(query_indices, true_neighbour_sets):
        distances = _hamming_all(tags, qi)
        order = np.argsort(distances, kind="stable")
        order = order[order != qi]
        aps.append(average_precision(order, relevant))
    return float(np.mean(aps))


@dataclass(frozen=True)
class BenchmarkResult:
    hash_lengths: list[int]
    fly_map: list[float]
    lsh_map: list[float]


def benchmark_hash_lengths(
    X: np.ndarray,
    hash_lengths: list[int],
    n_queries: int = 100,
    n_true_neighbours: int = 10,
    kc_expansion: int = 20,
    sample_size: int = 6,
    seed: int = 0,
    projection_matrix: sparse.spmatrix | None = None,
) -> BenchmarkResult:
    """Compare FlyHash mAP against classical LSH mAP across several hash
    lengths on the same dataset `X` (n_samples, n_features).

    `projection_matrix` lets the connectome validation re-run this with the real FlyWire
    connectome in place of the random PN->KC projection: pass a fixed matrix
    and every hash length reuses it (sliced to the first `hash_len` KCs is
    not meaningful for a fixed real matrix, so callers wanting a real
    three-way comparison should call this once per matrix and treat
    `kc_expansion`/`hash_len` as informational rather than controlling KC
    count in that case).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    query_indices = rng.choice(n, size=min(n_queries, n), replace=False)

    neighbours = NearestNeighbors(n_neighbors=n_true_neighbours + 1).fit(X)
    _, true_idx = neighbours.kneighbors(X[query_indices])
    true_neighbour_sets = [
        set(row[row != qi][:n_true_neighbours]) for row, qi in zip(true_idx, query_indices)
    ]

    fly_maps: list[float] = []
    lsh_maps: list[float] = []

    for k in hash_lengths:
        fly = FlyHash(
            d_in=X.shape[1],
            n_kc=kc_expansion * k,
            sample_size=sample_size,
            hash_len=k,
            seed=seed,
        )
        if projection_matrix is not None:
            fly.projection_matrix = projection_matrix
        fly.fit(X)
        fly_tags = fly.transform(X)
        fly_maps.append(mean_average_precision(fly_tags, query_indices, true_neighbour_sets))

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        lsh_tags = classical_lsh_transform((X - mean) / std, k, seed=seed)
        lsh_maps.append(mean_average_precision(lsh_tags, query_indices, true_neighbour_sets))

    return BenchmarkResult(hash_lengths=list(hash_lengths), fly_map=fly_maps, lsh_map=lsh_maps)
