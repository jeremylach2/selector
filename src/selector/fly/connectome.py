"""Real FlyWire projection-neuron -> Kenyon-cell connectivity.

Replaces the random projection in `selector.fly.lsh.FlyHash` with real,
synapse-count-weighted wiring from the FlyWire connectome (a fully proofread
adult female Drosophila whole-brain connectome), so a track's fingerprint
reflects an actually measured circuit instead of a random one.

Data source, exact version, and licensing are recorded in `docs/FLYWIRE.md`
-- read that file before changing `DATA_VERSION` or the download URLs below.
This is a wiring diagram, not a trained brain: see the claims-discipline note
in `docs/FLYWIRE.md` before writing anything that could be read as more than
that.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy import sparse

DATA_VERSION = "783"
ZENODO_RECORD = "10676866"  # https://zenodo.org/records/10676866

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
CONNECTIONS_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/"
    f"proofread_connections_{DATA_VERSION}.feather?download=1"
)

DEFAULT_CACHE_DIR = Path("data/flywire")

# Real cell-type labels in the FlyWire annotation table (see docs/FLYWIRE.md
# for how these were identified). ALPN = antennal lobe projection neuron,
# the real olfactory PN population; it includes both the uniglomerular PNs
# that carry the primary per-glomerulus olfactory channel and multiglomerular
# PNs that pool across glomeruli, both are real inputs to the mushroom
# body, so both are kept rather than hand-picking a "purer" subset.
PN_CELL_CLASS = "ALPN"
KC_CELL_CLASS = "Kenyon_Cell"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with open(tmp, "wb") as f:
            f.writelines(response.iter_bytes(1 << 20))
    tmp.rename(dest)


def download_flywire_data(cache_dir: Path = DEFAULT_CACHE_DIR) -> tuple[Path, Path]:
    """Download (once) and cache the FlyWire annotation and connection tables.

    Safe to call every run: both files are skipped if already cached.
    """
    annotations_path = cache_dir / f"neuron_annotations_{DATA_VERSION}.tsv"
    connections_path = cache_dir / f"proofread_connections_{DATA_VERSION}.feather"
    _download(ANNOTATIONS_URL, annotations_path)
    _download(CONNECTIONS_URL, connections_path)
    return annotations_path, connections_path


def load_cell_ids(
    annotations_path: Path, hemisphere: str = "right"
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(pn_root_ids, kc_root_ids)` for one hemisphere.

    The circuit runs largely independently per hemisphere, so this reads one
    side at a time rather than pooling both, pooling would let a PN and a
    KC that never actually meet look connected only because the two
    unrelated per-hemisphere circuits got merged into one index space.
    """
    annotations = pd.read_csv(annotations_path, sep="\t", usecols=["root_id", "cell_class", "side"])
    on_side = annotations["side"] == hemisphere
    pn_ids = annotations.loc[on_side & (annotations["cell_class"] == PN_CELL_CLASS), "root_id"]
    kc_ids = annotations.loc[on_side & (annotations["cell_class"] == KC_CELL_CLASS), "root_id"]
    return pn_ids.to_numpy(), kc_ids.to_numpy()


def build_projection_matrix(
    annotations_path: Path,
    connections_path: Path,
    hemisphere: str = "right",
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """The real PN->KC connectivity matrix, weighted by synapse count.

    Returns `(matrix, pn_ids, kc_ids)`. `matrix` has shape `(n_kc, n_pn)` and
    `matrix[i, j]` is the total synapse count from PN `pn_ids[j]` onto KC
    `kc_ids[i]`, summed across neuropils (overwhelmingly the mushroom body
    calyx, where this synaptic contact actually happens). Pairs that never
    connect are exactly zero, the real circuit's sparsity is preserved,
    never densified.
    """
    pn_ids, kc_ids = load_cell_ids(annotations_path, hemisphere)
    pn_index = pd.Series(np.arange(len(pn_ids)), index=pn_ids)
    kc_index = pd.Series(np.arange(len(kc_ids)), index=kc_ids)

    connections = pd.read_feather(
        connections_path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"]
    )
    connections = connections[
        connections["pre_pt_root_id"].isin(pn_index.index)
        & connections["post_pt_root_id"].isin(kc_index.index)
    ]

    rows = kc_index.loc[connections["post_pt_root_id"]].to_numpy()
    cols = pn_index.loc[connections["pre_pt_root_id"]].to_numpy()
    data = connections["syn_count"].to_numpy(dtype=np.float64)

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(kc_ids), len(pn_ids))).tocsr()
    matrix.sum_duplicates()
    return matrix, pn_ids, kc_ids


@dataclass(frozen=True)
class ConnectomeSummary:
    hemisphere: str
    n_pn: int
    n_kc: int
    mean_pn_inputs_per_kc: float
    kc_with_no_pn_input: int


def summarise(matrix: sparse.csr_matrix, hemisphere: str) -> ConnectomeSummary:
    n_kc, n_pn = matrix.shape
    binary = matrix.copy()
    binary.data[:] = 1.0
    inputs_per_kc = np.asarray(binary.sum(axis=1)).ravel()
    return ConnectomeSummary(
        hemisphere=hemisphere,
        n_pn=n_pn,
        n_kc=n_kc,
        mean_pn_inputs_per_kc=float(inputs_per_kc.mean()),
        kc_with_no_pn_input=int((inputs_per_kc == 0).sum()),
    )


def pool_to_width(matrix: sparse.csr_matrix, d_in: int) -> sparse.csr_matrix:
    """Adapt the real PN population to an arbitrary input width `d_in`.

    The vibe tagger's feature vector width will not equal the real PN count,
    so real PNs are grouped into `d_in` buckets by `pn_index % d_in`, and
    each Kenyon cell's weighted input from the PNs in a bucket is summed.
    Feature dimension `j` is therefore the pooled sum of whichever real PNs
    landed in bucket `j`, an honest lossy compression of the anatomical PN
    population onto the tagger's feature width, not a claim that dimension
    `j` corresponds to one specific real neuron. Padding/tiling a short
    feature vector up to the real PN count was rejected as the alternative:
    it would silently repeat feature values rather than losing information
    in one clearly documented step.
    """
    n_pn = matrix.shape[1]
    bucket_of_pn = np.arange(n_pn) % d_in
    pool = sparse.csr_matrix((np.ones(n_pn), (np.arange(n_pn), bucket_of_pn)), shape=(n_pn, d_in))
    return (matrix @ pool).tocsr()


def load_flywire_projection(
    d_in: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    hemisphere: str = "right",
) -> sparse.csr_matrix:
    """One-call seam for `FlyHash.projection_matrix`: download/cache the
    connectome if needed, build the real weighted PN->KC matrix, and pool it
    to `d_in` columns so the result can be assigned directly.
    """
    annotations_path, connections_path = download_flywire_data(cache_dir)
    matrix, _, _ = build_projection_matrix(annotations_path, connections_path, hemisphere)
    return pool_to_width(matrix, d_in)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hemisphere", choices=["left", "right"], default="right")
    args = parser.parse_args(argv)

    annotations_path, connections_path = download_flywire_data(args.cache_dir)
    matrix, _, _ = build_projection_matrix(annotations_path, connections_path, args.hemisphere)
    summary = summarise(matrix, args.hemisphere)

    print(f"FlyWire v{DATA_VERSION}, {args.hemisphere} hemisphere")
    print(f"  Real PNs (ALPN):          {summary.n_pn:,}   (paper's idealised PN count: 50)")
    print(f"  Real KCs:                 {summary.n_kc:,}   (paper's idealised KC count: 2,000)")
    print(
        f"  Mean PN inputs / KC:      {summary.mean_pn_inputs_per_kc:.2f}   "
        "(paper's idealised sample_size: 6)"
    )
    print(f"  KCs with no ALPN input:   {summary.kc_with_no_pn_input:,}")


if __name__ == "__main__":
    main()
