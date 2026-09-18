import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from selector.fly.connectome import (
    build_projection_matrix,
    download_flywire_data,
    load_cell_ids,
    pool_to_width,
    summarise,
)

# A tiny synthetic slice shaped like the real FlyWire tables: 4 PNs and 3 KCs
# on the right side, one PN on the left (must be excluded by hemisphere
# filtering), plus a non-PN/non-KC neuron that must never appear in either
# id list even though it has a root_id.
ANNOTATION_ROWS = [
    {"root_id": 1, "cell_class": "ALPN", "side": "right"},
    {"root_id": 2, "cell_class": "ALPN", "side": "right"},
    {"root_id": 3, "cell_class": "ALPN", "side": "right"},
    {"root_id": 4, "cell_class": "ALPN", "side": "right"},
    {"root_id": 5, "cell_class": "ALPN", "side": "left"},  # wrong hemisphere
    {"root_id": 10, "cell_class": "Kenyon_Cell", "side": "right"},
    {"root_id": 11, "cell_class": "Kenyon_Cell", "side": "right"},
    {"root_id": 12, "cell_class": "Kenyon_Cell", "side": "right"},
    {"root_id": 13, "cell_class": "Kenyon_Cell", "side": "left"},  # wrong hemisphere
    {"root_id": 99, "cell_class": "visual_projection", "side": "right"},  # irrelevant
]

CONNECTION_ROWS = [
    # KC 10 reads from PNs 1 and 2
    {"pre_pt_root_id": 1, "post_pt_root_id": 10, "syn_count": 5},
    {"pre_pt_root_id": 2, "post_pt_root_id": 10, "syn_count": 3},
    # duplicate pre/post pair in a different neuropil -- should sum
    {"pre_pt_root_id": 2, "post_pt_root_id": 10, "syn_count": 1},
    # KC 11 reads from PN 3 only
    {"pre_pt_root_id": 3, "post_pt_root_id": 11, "syn_count": 7},
    # KC 12 has no PN input at all
    # noise: a left-hemisphere PN onto a right-hemisphere KC, must be dropped
    {"pre_pt_root_id": 5, "post_pt_root_id": 10, "syn_count": 100},
    # noise: an irrelevant cell type onto a KC, must be dropped
    {"pre_pt_root_id": 99, "post_pt_root_id": 11, "syn_count": 100},
]


@pytest.fixture
def synthetic_flywire(tmp_path):
    annotations_path = tmp_path / "neuron_annotations_783.tsv"
    pd.DataFrame(ANNOTATION_ROWS).to_csv(annotations_path, sep="\t", index=False)

    connections_path = tmp_path / "proofread_connections_783.feather"
    pd.DataFrame(CONNECTION_ROWS).to_feather(connections_path)

    return annotations_path, connections_path


def test_load_cell_ids_filters_class_and_hemisphere(synthetic_flywire):
    annotations_path, _ = synthetic_flywire
    pn_ids, kc_ids = load_cell_ids(annotations_path, hemisphere="right")

    assert sorted(pn_ids) == [1, 2, 3, 4]
    assert sorted(kc_ids) == [10, 11, 12]


def test_build_projection_matrix_sums_duplicates_and_drops_noise(synthetic_flywire):
    annotations_path, connections_path = synthetic_flywire
    matrix, pn_ids, kc_ids = build_projection_matrix(annotations_path, connections_path, "right")

    assert matrix.shape == (3, 4)
    dense = matrix.toarray()

    kc10, kc11, kc12 = (np.where(kc_ids == kid)[0][0] for kid in (10, 11, 12))
    pn1, pn2, pn3 = (np.where(pn_ids == pid)[0][0] for pid in (1, 2, 3))

    assert dense[kc10, pn1] == 5
    assert dense[kc10, pn2] == 4  # 3 + 1, summed across the duplicate pre/post pair
    assert dense[kc11, pn3] == 7
    assert dense[kc12].sum() == 0  # no PN input at all
    assert dense.sum() == 5 + 4 + 7  # cross-hemisphere and irrelevant-class noise excluded


def test_summarise_reports_real_numbers(synthetic_flywire):
    annotations_path, connections_path = synthetic_flywire
    matrix, _, _ = build_projection_matrix(annotations_path, connections_path, "right")
    summary = summarise(matrix, "right")

    assert summary.n_pn == 4
    assert summary.n_kc == 3
    assert summary.kc_with_no_pn_input == 1
    # KC10 has 2 inputs, KC11 has 1, KC12 has 0 -> mean = 1.0
    assert summary.mean_pn_inputs_per_kc == pytest.approx(1.0)


def test_pool_to_width_preserves_total_weight_and_shape(synthetic_flywire):
    annotations_path, connections_path = synthetic_flywire
    matrix, _, _ = build_projection_matrix(annotations_path, connections_path, "right")

    pooled = pool_to_width(matrix, d_in=2)

    assert pooled.shape == (3, 2)
    assert sparse.issparse(pooled)
    # Pooling only regroups columns -- no synapse weight is created or lost.
    assert pooled.sum() == pytest.approx(matrix.sum())


def test_download_flywire_data_skips_existing_files(tmp_path, monkeypatch):
    cache_dir = tmp_path / "flywire"
    cache_dir.mkdir()
    (cache_dir / "neuron_annotations_783.tsv").write_text("cached")
    (cache_dir / "proofread_connections_783.feather").write_bytes(b"cached")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not hit the network when files are already cached")

    monkeypatch.setattr("httpx.stream", fail_if_called)

    annotations_path, connections_path = download_flywire_data(cache_dir)
    assert annotations_path.read_text() == "cached"
    assert connections_path.read_bytes() == b"cached"
