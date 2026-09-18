"""Fast, deterministic reproduction of the paper's core claim, run on every
test invocation. `notebooks/fly_validation.py` reproduces the same claim on
real MNIST data and saves the plot for the README; this test exists so CI
catches a regression (e.g. broken normalisation or WTA ordering) without
paying for a dataset download.
"""

import numpy as np
from sklearn.datasets import make_blobs

from selector.fly.benchmark import benchmark_hash_lengths, pool_features


def test_flyhash_beats_classical_lsh_at_short_hash_lengths():
    X, _ = make_blobs(n_samples=600, n_features=64, centers=25, cluster_std=2.5, random_state=0)

    result = benchmark_hash_lengths(
        X,
        hash_lengths=[4, 8, 16, 32],
        n_queries=80,
        n_true_neighbours=10,
        kc_expansion=20,
        seed=0,
    )

    # The paper's headline result: FlyHash's advantage is largest at short
    # hash lengths. Assert it outright at the shortest length tested.
    assert result.fly_map[0] > result.lsh_map[0], (
        f"FlyHash mAP ({result.fly_map[0]:.3f}) should beat classical LSH mAP "
        f"({result.lsh_map[0]:.3f}) at hash_len={result.hash_lengths[0]}. "
        "If this regresses, check divisive normalisation and WTA ordering first."
    )

    # Both methods should improve as more bits are compared.
    assert result.fly_map[-1] >= result.fly_map[0]
    assert result.lsh_map[-1] >= result.lsh_map[0]


def test_pool_features_preserves_total_and_nonnegativity():
    rng = np.random.default_rng(0)
    X = rng.uniform(low=0.0, high=255.0, size=(10, 7))

    pooled = pool_features(X, n_out=3)

    assert pooled.shape == (10, 3)
    assert np.all(pooled >= 0)
    np.testing.assert_allclose(pooled.sum(axis=1), X.sum(axis=1))


def test_pool_features_groups_columns_by_modulo():
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pooled = pool_features(X, n_out=2)
    # bucket 0 <- columns 0, 2, 4 (1+3+5); bucket 1 <- columns 1, 3 (2+4)
    np.testing.assert_allclose(pooled, [[9.0, 6.0]])
