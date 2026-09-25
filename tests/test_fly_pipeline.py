from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from selector.fly import pipeline
from selector.fly.mbon import MushroomBody


def _fake_projection(d_in: int, n_kc: int = 40, seed: int = 0) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    return sparse.random(n_kc, d_in, density=0.2, random_state=rng, format="csr")


@pytest.fixture(autouse=True)
def _no_network_connectome(monkeypatch):
    """Every test in this file fits a fly brain; none of them should reach
    the FlyWire cache or network, so the real connectome loader is replaced
    with a small random projection of matching width."""
    monkeypatch.setattr(pipeline, "load_flywire_projection", _fake_projection)


def _track_features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "t1",
                "valence": 0.9,
                "mood_tags": ["euphoric", "playful"],
                "era": "2010s",
                "lyrical_theme": "party",
                "intensity": 0.8,
                "arm": "C",
                "label_source": "finetuned_gpu",
            },
            {
                "track_id": "t2",
                "valence": 0.1,
                "mood_tags": ["melancholic"],
                "era": "1990s",
                "lyrical_theme": "breakup",
                "intensity": 0.3,
                "arm": "A",
                "label_source": "finetuned_gpu",
            },
            {
                "track_id": "t3",
                "valence": 0.5,
                "mood_tags": ["chill"],
                "era": "2020s",
                "lyrical_theme": "unknown",
                "intensity": 0.4,
                "arm": "A",
                "label_source": "parse_fallback",
            },
        ]
    )


def _audio_features_frame() -> pd.DataFrame:
    # Only t1 has a matched preview clip -- t2/t3 must fall back to zeros
    # plus has_measured=0 in the "full" source.
    return pd.DataFrame(
        [
            {
                "track_id": "t1",
                "tempo_scaled": 0.8,
                "rms_mean_scaled": 0.6,
                "danceability": 0.7,
                "harmonic_percussive_ratio_scaled": 0.5,
            }
        ]
    )


@pytest.fixture
def data_files(tmp_path) -> tuple[Path, Path]:
    track_features_path = tmp_path / "track_features.parquet"
    audio_features_path = tmp_path / "audio_features.parquet"
    _track_features_frame().to_parquet(track_features_path)
    _audio_features_frame().to_parquet(audio_features_path)
    return track_features_path, audio_features_path


def test_predicted_dim_matches_era_and_mood_vocab_size():
    assert pipeline.PREDICTED_DIM == 2 + len(pipeline.ERA_VOCAB) + len(pipeline.MOOD_VOCAB)


def test_full_dim_adds_measured_and_has_measured_flag():
    assert pipeline.FULL_DIM == pipeline.PREDICTED_DIM + len(pipeline.MEASURED_COLUMNS) + 1


def test_build_feature_matrix_text_only_shape(data_files):
    track_features_path, audio_features_path = data_files
    track_ids, X = pipeline.build_feature_matrix("text_only", track_features_path, audio_features_path)

    assert track_ids == ["t1", "t2", "t3"]
    assert X.shape == (3, pipeline.PREDICTED_DIM)


def test_build_feature_matrix_full_flags_measured_tracks(data_files):
    track_features_path, audio_features_path = data_files
    _track_ids, X = pipeline.build_feature_matrix("full", track_features_path, audio_features_path)

    assert X.shape == (3, pipeline.FULL_DIM)
    has_measured = X[:, -1]
    # t1 (index 0) has a matched audio row; t2/t3 don't.
    np.testing.assert_array_equal(has_measured, [1.0, 0.0, 0.0])
    # The measured block for an unmatched track is all zeros, not NaN.
    measured_block = X[1:, pipeline.PREDICTED_DIM : pipeline.PREDICTED_DIM + len(pipeline.MEASURED_COLUMNS)]
    assert np.all(measured_block == 0.0)


def test_build_feature_matrix_full_without_audio_file(tmp_path, data_files):
    track_features_path, _ = data_files
    missing_audio_path = tmp_path / "no_such_audio_features.parquet"

    _track_ids, X = pipeline.build_feature_matrix("full", track_features_path, missing_audio_path)

    assert X.shape == (3, pipeline.FULL_DIM)
    assert np.all(X[:, -1] == 0.0)  # has_measured is false for everyone


def test_build_feature_matrix_placeholder_is_deterministic(data_files):
    track_features_path, audio_features_path = data_files
    ids_a, X_a = pipeline.build_feature_matrix("placeholder", track_features_path, audio_features_path)
    ids_b, X_b = pipeline.build_feature_matrix("placeholder", track_features_path, audio_features_path)

    assert ids_a == ids_b
    np.testing.assert_array_equal(X_a, X_b)


def test_fit_fly_uses_the_flywire_seam(data_files, monkeypatch):
    calls: list[int] = []

    def spy(d_in: int, **kwargs):
        calls.append(d_in)
        return _fake_projection(d_in)

    monkeypatch.setattr(pipeline, "load_flywire_projection", spy)

    track_features_path, audio_features_path = data_files
    track_ids, X = pipeline.build_feature_matrix("full", track_features_path, audio_features_path)
    fly, tags = pipeline.fit_fly(track_ids, X)

    assert calls == [X.shape[1]]
    assert tags.shape[0] == len(track_ids)
    assert fly.n_kc == tags.shape[1]


def test_save_and_load_tags_round_trip(tmp_path, data_files):
    track_features_path, audio_features_path = data_files
    track_ids, X = pipeline.build_feature_matrix("full", track_features_path, audio_features_path)
    _fly, tags = pipeline.fit_fly(track_ids, X)

    out_path = tmp_path / "fly_tags.npz"
    pipeline.save_tags(out_path, track_ids, tags)
    loaded_ids, loaded_tags = pipeline.load_tags(out_path)

    assert loaded_ids == track_ids
    np.testing.assert_array_equal(loaded_tags.toarray(), tags.toarray())


def test_build_and_persist_tags_writes_file(tmp_path, data_files):
    track_features_path, audio_features_path = data_files
    out_path = tmp_path / "fly_tags.npz"

    track_ids, tags = pipeline.build_and_persist_tags(
        source="full",
        output_path=out_path,
        track_features_path=track_features_path,
        audio_features_path=audio_features_path,
    )

    assert out_path.exists()
    loaded_ids, loaded_tags = pipeline.load_tags(out_path)
    assert loaded_ids == track_ids
    np.testing.assert_array_equal(loaded_tags.toarray(), tags.toarray())


def test_train_production_mbon_trains_on_full_chronological_history(tmp_path, data_files):
    track_features_path, audio_features_path = data_files
    track_ids, X = pipeline.build_feature_matrix("full", track_features_path, audio_features_path)
    _fly, tags = pipeline.fit_fly(track_ids, X)

    plays_path = tmp_path / "plays.parquet"
    pd.DataFrame(
        [
            {"track_id": "t1", "ts": "2022-01-01T00:00:00Z", "verdict": 1},
            {"track_id": "t2", "ts": "2022-01-02T00:00:00Z", "verdict": -1},
            {"track_id": "unknown-track", "ts": "2022-01-03T00:00:00Z", "verdict": 1},
        ]
    ).to_parquet(plays_path)

    mbon = pipeline.train_production_mbon(track_ids, tags, plays_path=plays_path)

    assert isinstance(mbon, MushroomBody)
    # A track never seen in `track_ids` must not raise and must not perturb
    # the (fixed at construction) synapse array size.
    assert mbon.w_approach.shape == (tags.shape[1],)


def test_nkc_holder_exposes_only_n_kc():
    holder = pipeline._NKcHolder(n_kc=123)
    assert holder.n_kc == 123
