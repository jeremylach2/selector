import numpy as np
import pytest
from scipy import sparse

from selector.fly.lsh import FlyHash


def _toy_data(n=40, d=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_transform_shape_and_sparsity():
    X = _toy_data(n=30, d=10)
    fly = FlyHash(d_in=10, n_kc=200, sample_size=4, wta_frac=0.05, seed=1)
    fly.fit(X)
    tags = fly.transform(X)

    assert tags.shape == (30, 200)
    assert sparse.issparse(tags)
    active_per_row = np.asarray(tags.sum(axis=1)).ravel()
    # wta_frac=0.05 of 200 KCs -> 10 active per row
    assert np.all(active_per_row == 10)


def test_hash_len_overrides_wta_frac():
    X = _toy_data(n=10, d=10)
    fly = FlyHash(d_in=10, n_kc=200, hash_len=25, wta_frac=0.5, seed=2)
    fly.fit(X)
    tags = fly.transform(X)
    active_per_row = np.asarray(tags.sum(axis=1)).ravel()
    assert np.all(active_per_row == 25)


def test_transform_before_fit_raises():
    fly = FlyHash(d_in=5, n_kc=50, seed=0)
    with pytest.raises(RuntimeError):
        fly.transform(_toy_data(n=3, d=5))


def test_similar_inputs_get_similar_tags():
    rng = np.random.default_rng(3)
    d_in = 16
    base = rng.normal(size=(1, d_in))
    near = base + rng.normal(scale=0.01, size=(1, d_in))
    far = rng.normal(size=(1, d_in)) * 5

    X = np.vstack([base, near, far, _toy_data(n=50, d=d_in, seed=4)])
    fly = FlyHash(d_in=d_in, n_kc=1000, sample_size=6, wta_frac=0.05, seed=5)
    fly.fit(X)
    tags = fly.transform(X)

    base_tag, near_tag, far_tag = tags[0], tags[1], tags[2]
    hamming = lambda a, b: (a != b).sum()

    dist_near = hamming(base_tag.toarray(), near_tag.toarray())
    dist_far = hamming(base_tag.toarray(), far_tag.toarray())
    assert dist_near < dist_far


def test_hamming_neighbours_finds_self_first():
    X = _toy_data(n=25, d=8, seed=6)
    fly = FlyHash(d_in=8, n_kc=300, wta_frac=0.05, seed=7)
    fly.fit(X)
    tags = fly.transform(X)

    query_tag = tags[3]
    indices, distances = fly.hamming_neighbours(query_tag, k=5)

    assert indices[0] == 3
    assert distances[0] == 0
    assert len(indices) == 5
    assert np.all(np.diff(distances) >= 0)


def test_hamming_neighbours_accepts_external_corpus():
    X = _toy_data(n=20, d=6, seed=8)
    fly = FlyHash(d_in=6, n_kc=150, wta_frac=0.1, seed=9)
    fly.fit(X)
    corpus_tags = fly.transform(X)

    query = fly.transform(_toy_data(n=1, d=6, seed=10))
    indices, distances = fly.hamming_neighbours(query, k=3, tags=corpus_tags)

    assert len(indices) == 3
    assert np.all(distances >= 0)


def test_projection_matrix_setter_validates_shape():
    fly = FlyHash(d_in=10, n_kc=50, seed=0)
    good = sparse.random(80, 10, density=0.1, format="csr")
    fly.projection_matrix = good
    assert fly.n_kc == 80

    bad = sparse.random(80, 11, density=0.1, format="csr")
    with pytest.raises(ValueError):
        fly.projection_matrix = bad


def test_projection_matrix_setter_is_the_flywire_seam():
    """The connectome module swaps this property; downstream transform() must honour it."""
    X = _toy_data(n=5, d=4, seed=11)
    fly = FlyHash(d_in=4, n_kc=20, seed=0)
    fly.fit(X)

    custom = np.zeros((6, 4))
    custom[0, 0] = 1
    custom[1, 1] = 1
    custom[2, 2] = 1
    custom[3, 3] = 1
    custom[4, [0, 1]] = 1
    custom[5, [2, 3]] = 1
    fly.projection_matrix = custom

    tags = fly.transform(X)
    assert tags.shape == (5, 6)


def test_hamming_top_k_orders_ties_by_tie_break_across_the_cutoff():
    from selector.fly.lsh import hamming_top_k

    # Rows 1-4 are identical to the query (distance 0); row 5 is further.
    corpus = sparse.csr_matrix(
        np.array([[1, 1, 0, 0]] * 5 + [[0, 0, 1, 1]], dtype=bool)
    )
    query = corpus[0]
    score = np.array([0.0, 0.1, 0.9, 0.2, 0.8, 1.0])

    idx, dist = hamming_top_k(query, corpus, k=3, tie_break=score)

    # All five rows tie at 0, so the best-scored three win, even though
    # row 5 (score 1.0) scores highest overall: it's further away.
    assert list(idx) == [2, 4, 3]
    assert list(dist) == [0, 0, 0]


def test_hamming_top_k_without_tie_break_is_unchanged():
    from selector.fly.lsh import hamming_top_k

    corpus = sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool))
    idx, dist = hamming_top_k(corpus[0], corpus, k=2)
    assert idx[0] == 0 and dist[0] == 0


def test_hamming_top_k_secondary_key_separates_identical_primary_scores():
    # Identical tags get identical tag-derived scores (like fly valence), so
    # only a second, independent key can order them.
    from selector.fly.lsh import hamming_top_k

    corpus = sparse.csr_matrix(np.array([[1, 1, 0, 0]] * 4, dtype=bool))
    taste = np.array([0.5, 0.5, 0.5, 0.5])
    plays = np.array([0.0, 3.0, 9.0, 1.0])

    idx, _ = hamming_top_k(corpus[0], corpus, k=3, tie_break=[taste, plays])

    assert list(idx) == [2, 1, 3]
