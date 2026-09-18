"""Three-way reproduction of Step 5's benchmark, with the real FlyWire
connectome added as a third line: random LSH vs. idealised FlyHash (random
PN->KC projection) vs. FlyWire FlyHash (the real, measured PN->KC wiring).

This answers the question Step 6 exists to answer: does the *real* circuit
retrieve nearest neighbours any better or worse than an equally-sized random
one? Report it straight either way.

Run: `uv run python notebooks/fly_connectome_validation.py`
Output: prints a mAP table and writes `docs/img/fly_vs_lsh_connectome.png`.
Downloads ~900 MB of FlyWire data to `data/flywire/` on first run (cached
after that) -- see docs/FLYWIRE.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

from selector.fly.benchmark import benchmark_hash_lengths, pool_features
from selector.fly.connectome import build_projection_matrix, download_flywire_data, summarise

OUTPUT_PATH = Path("docs/img/fly_vs_lsh_connectome.png")

HASH_LENGTHS = [4, 8, 16, 32, 64]
N_SAMPLES = 3000
N_QUERIES = 200
N_TRUE_NEIGHBOURS = 10
SEED = 0
HEMISPHERE = "right"


def load_mnist_subset(n_samples: int, seed: int) -> np.ndarray:
    print("Fetching MNIST (cached after first run outside the repo)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    return X[idx]


def main() -> None:
    print(f"Loading the real FlyWire PN->KC connectome ({HEMISPHERE} hemisphere)...")
    annotations_path, connections_path = download_flywire_data()
    real_matrix, _pn_ids, _kc_ids = build_projection_matrix(
        annotations_path, connections_path, HEMISPHERE
    )
    summary = summarise(real_matrix, HEMISPHERE)
    n_pn = summary.n_pn
    print(
        f"Real circuit: {summary.n_pn} PNs, {summary.n_kc} KCs, "
        f"{summary.mean_pn_inputs_per_kc:.2f} mean PN inputs/KC "
        f"(paper's idealised circuit: 50 PNs, 2000 KCs, 6 inputs/KC)"
    )

    X_raw = load_mnist_subset(N_SAMPLES, SEED)
    print(
        f"Pooling {X_raw.shape[1]}-pixel MNIST down to {n_pn} dimensions (summing raw, "
        f"non-negative pixel intensities, not PCA -- see selector.fly.benchmark.pool_features) "
        f"to match the real PN count exactly, keeping FlyHash's non-negative firing-rate "
        f"assumption intact for all three methods."
    )
    X = pool_features(X_raw, n_pn)

    print("Benchmarking idealised FlyHash + classical LSH...")
    idealised = benchmark_hash_lengths(
        X,
        hash_lengths=HASH_LENGTHS,
        n_queries=N_QUERIES,
        n_true_neighbours=N_TRUE_NEIGHBOURS,
        kc_expansion=20,
        seed=SEED,
    )

    print("Benchmarking FlyWire FlyHash (real connectome)...")
    flywire = benchmark_hash_lengths(
        X,
        hash_lengths=HASH_LENGTHS,
        n_queries=N_QUERIES,
        n_true_neighbours=N_TRUE_NEIGHBOURS,
        seed=SEED,
        projection_matrix=real_matrix,
    )

    print(f"\n{'hash_len':>10} {'classical LSH':>15} {'idealised Fly':>15} {'FlyWire Fly':>15}")
    for i, k in enumerate(HASH_LENGTHS):
        print(
            f"{k:>10} {idealised.lsh_map[i]:>15.3f} "
            f"{idealised.fly_map[i]:>15.3f} {flywire.fly_map[i]:>15.3f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(HASH_LENGTHS, idealised.lsh_map, marker="s", label="Classical random-projection LSH")
    ax.plot(HASH_LENGTHS, idealised.fly_map, marker="o", label="Idealised FlyHash (random PN->KC)")
    ax.plot(HASH_LENGTHS, flywire.fly_map, marker="^", label="FlyWire FlyHash (real PN->KC)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(HASH_LENGTHS)
    ax.set_xticklabels([str(k) for k in HASH_LENGTHS])
    ax.set_xlabel("Hash length (KCs compared)")
    ax.set_ylabel("Mean average precision (10-NN retrieval)")
    ax.set_title(f"Random LSH vs. idealised vs. real FlyWire circuit (MNIST, pooled-{n_pn})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
