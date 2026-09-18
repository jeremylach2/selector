import numpy as np
import pytest

from selector.fly.lsh import FlyHash
from selector.fly.mbon import MushroomBody


def _fitted_fly(d_in=10, n_kc=200, wta_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(30, d_in))
    fly = FlyHash(d_in=d_in, n_kc=n_kc, sample_size=4, wta_frac=wta_frac, seed=seed)
    fly.fit(X)
    return fly, X


def test_rejects_non_two_mbon():
    fly, _ = _fitted_fly()
    with pytest.raises(ValueError):
        MushroomBody(fly, n_mbon=3)


def test_naive_valence_is_zero_before_any_learning():
    fly, X = _fitted_fly()
    tag = fly.transform(X[:1])
    mbon = MushroomBody(fly, lr=0.1)
    # Naive fly: approach and avoid weights start equal, so valence is neutral.
    assert mbon.valence(tag) == pytest.approx(0.0)


def test_reward_raises_valence_for_the_same_tag():
    fly, X = _fitted_fly()
    tag = fly.transform(X[:1])
    mbon = MushroomBody(fly, lr=0.2)

    before = mbon.valence(tag)
    mbon.learn(tag, verdict=1)
    after = mbon.valence(tag)

    assert after > before


def test_punishment_lowers_valence_for_the_same_tag():
    fly, X = _fitted_fly()
    tag = fly.transform(X[:1])
    mbon = MushroomBody(fly, lr=0.2)

    before = mbon.valence(tag)
    mbon.learn(tag, verdict=-1)
    after = mbon.valence(tag)

    assert after < before


def test_neutral_verdict_does_not_change_weights():
    fly, X = _fitted_fly()
    tag = fly.transform(X[:1])
    mbon = MushroomBody(fly, lr=0.2)

    before_approach = mbon.w_approach.copy()
    before_avoid = mbon.w_avoid.copy()
    mbon.learn(tag, verdict=0)

    np.testing.assert_array_equal(mbon.w_approach, before_approach)
    np.testing.assert_array_equal(mbon.w_avoid, before_avoid)


def test_learning_only_touches_active_kenyon_cells():
    fly, X = _fitted_fly(n_kc=500, wta_frac=0.05)
    tags = fly.transform(X)
    tag = tags[0]
    mbon = MushroomBody(fly, lr=0.3)

    active = set(tag.indices)
    mbon.learn(tag, verdict=1)

    inactive_mask = np.ones(fly.n_kc, dtype=bool)
    inactive_mask[list(active)] = False
    assert np.all(mbon.w_avoid[inactive_mask] == 1.0)
    assert np.all(mbon.w_approach[inactive_mask] == 1.0)
    assert np.all(mbon.w_avoid[list(active)] < 1.0)


def test_dissimilar_tags_are_unaffected_by_unrelated_learning():
    rng = np.random.default_rng(1)
    d_in = 16
    base = rng.normal(size=(1, d_in))
    far = rng.normal(size=(1, d_in)) * 10

    fly, _ = _fitted_fly(d_in=d_in, n_kc=1000, wta_frac=0.02, seed=2)
    fly.fit(np.vstack([base, far, rng.normal(size=(50, d_in))]))
    base_tag = fly.transform(base)
    far_tag = fly.transform(far)

    mbon = MushroomBody(fly, lr=0.5)
    far_valence_before = mbon.valence(far_tag)
    mbon.learn(base_tag, verdict=-1)
    far_valence_after = mbon.valence(far_tag)

    # A tag with no overlap in active KCs should be untouched by learning on
    # a completely different tag -- generalisation only happens through
    # shared active cells.
    if not (set(base_tag.indices) & set(far_tag.indices)):
        assert far_valence_after == pytest.approx(far_valence_before)


def test_decay_pulls_weights_back_toward_baseline():
    fly, X = _fitted_fly()
    tag = fly.transform(X[:1])
    mbon = MushroomBody(fly, lr=0.5, decay=0.5)

    mbon.learn(tag, verdict=-1)
    depressed = mbon.w_approach[tag.indices].copy()

    # Learning on an unrelated verdict-0 play still applies decay each call.
    mbon.learn(tag, verdict=0)
    recovered = mbon.w_approach[tag.indices]

    assert np.all(recovered > depressed)
    assert np.all(recovered < 1.0)


def test_invalid_lr_and_decay_rejected():
    fly, _ = _fitted_fly()
    with pytest.raises(ValueError):
        MushroomBody(fly, lr=0.0)
    with pytest.raises(ValueError):
        MushroomBody(fly, lr=1.5)
    with pytest.raises(ValueError):
        MushroomBody(fly, decay=1.0)
