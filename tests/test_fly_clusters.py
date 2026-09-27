import json

import numpy as np
import pandas as pd
from scipy import sparse

from selector.fly import cluster_names, clusters


def _tags(rows: list[list[int]], n_kc: int = 20) -> sparse.csr_matrix:
    dense = np.zeros((len(rows), n_kc), dtype=bool)
    for i, active in enumerate(rows):
        dense[i, active] = True
    return sparse.csr_matrix(dense)


def test_pairwise_hamming_matches_dense():
    a = _tags([[0, 1, 2], [3, 4]])
    b = _tags([[0, 1, 2], [0, 5], []])
    dense_a, dense_b = a.toarray(), b.toarray()
    expected = np.array([[np.sum(x != y) for y in dense_b] for x in dense_a])
    np.testing.assert_array_equal(clusters.pairwise_hamming(a, b), expected)


def test_normalised_hamming_bounds():
    a = _tags([[0, 1, 2]])
    b = _tags([[0, 1, 2], [5, 6, 7]])
    np.testing.assert_allclose(clusters.normalised_hamming(a, b), [[0.0, 1.0]])


def test_unique_rows_maps_duplicates_together():
    firsts, inverse = clusters._unique_rows(_tags([[0, 1], [2, 3], [0, 1]]))
    assert firsts.tolist() == [0, 1]
    assert inverse.tolist() == [0, 1, 0]


def test_kmedoids_separates_two_blobs():
    rows = [[0, 1, 2, 3]] * 5 + [[0, 1, 2, 4]] * 5 + [[10, 11, 12, 13]] * 5 + [[10, 11, 12, 14]] * 5
    tags = _tags(rows)
    dist = clusters.normalised_hamming(tags, tags)
    _medoids, labels = clusters.kmedoids(dist, 2, np.random.default_rng(0))
    assert len(set(labels[:10])) == 1
    assert len(set(labels[10:])) == 1
    assert labels[0] != labels[10]


def test_fit_clusters_is_deterministic():
    rng = np.random.default_rng(1)
    rows = [sorted(rng.choice(40, size=4, replace=False).tolist()) for _ in range(60)]
    tags = _tags(rows, n_kc=40)
    ids = [f"t{i}" for i in range(60)]
    config = {**clusters.CLUSTER_CONFIG, "k_min": 2, "k_max": 4, "fit_sample": 30}
    a = clusters.fit_clusters(ids, tags, config)
    b = clusters.fit_clusters(ids, tags, config)
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])


def _library() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mood_tags": [["chill"], ["chill"], ["euphoric"], ["euphoric"], ["somber"], ["chill"]],
            "intensity": [0.2, 0.2, 0.9, 0.9, 0.5, 0.3],
            "era": ["2010s", "2010s", "2010s", "1970s", "2010s", "2010s"],
        }
    )


def test_built_name_uses_distinctive_features():
    library = _library()
    members = library.iloc[[2, 3]]
    name = clusters.built_name(members, library)
    assert name.startswith("Euphoric")
    assert "high-energy" in name
    assert "1970s" in name  # half the cluster, 3x the library rate


def test_built_name_omits_undistinctive_era():
    library = _library()
    name = clusters.built_name(library.iloc[[0, 1]], library)
    assert "2010s" not in name  # the library's majority era says nothing
    assert "low-key" in name


def test_llm_name_ignored_when_built_name_changed(tmp_path):
    path = tmp_path / "names.json"
    path.write_text(
        json.dumps({"key1": {"names": {"0": {"name": "Sunday Porch", "built_name": "Chill"}}}}),
        encoding="utf-8",
    )
    assert cluster_names.llm_name_for(0, "Chill", "key1", path) == "Sunday Porch"
    assert cluster_names.llm_name_for(0, "Chill · low-key", "key1", path) is None
    assert cluster_names.llm_name_for(0, "Chill", "other", path) is None
