import pandas as pd

from selector.audio.merge import merge_features


def test_merge_degrades_gracefully_without_essentia(tmp_path):
    librosa_path = tmp_path / "librosa.parquet"
    pd.DataFrame(
        {
            "track_id": ["a", "b"],
            "tempo": [90.0, 140.0],
            "rms_mean": [0.1, 0.3],
        }
    ).to_parquet(librosa_path)

    merged = merge_features(librosa_path, tmp_path / "does_not_exist.parquet")

    assert list(merged["has_essentia"]) == [False, False]
    assert "tempo_scaled" in merged.columns
    assert merged["tempo_scaled"].tolist() == [0.0, 1.0]


def test_merge_joins_essentia_when_present(tmp_path):
    librosa_path = tmp_path / "librosa.parquet"
    essentia_path = tmp_path / "essentia.parquet"
    pd.DataFrame({"track_id": ["a", "b"], "tempo": [90.0, 140.0]}).to_parquet(librosa_path)
    pd.DataFrame({"track_id": ["a"], "danceability": [0.8]}).to_parquet(essentia_path)

    merged = merge_features(librosa_path, essentia_path).set_index("track_id")

    assert bool(merged.loc["a", "has_essentia"]) is True
    assert bool(merged.loc["b", "has_essentia"]) is False
