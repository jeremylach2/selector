"""Reproduce the paper's core claim: FlyHash beats classical random-projection
LSH at nearest-neighbour retrieval, especially at short hash lengths.

Reference: Dasgupta, Stevens & Navlakha, "A neural algorithm for a
fundamental computing problem", Science 358:6364 (2017).

Dataset: a subsample of MNIST (fetched once via scikit-learn and cached
outside the repo, in the user's default scikit-learn data directory, no
network needed on subsequent runs). `tests/test_fly_benchmark.py` runs the
same comparison on a small synthetic dataset for a fast, offline regression
check; this script is the slower, real-data reproduction that the README's
plot comes from.

Run: `uv run python notebooks/fly_validation.py`
Output: prints a mAP table and writes `docs/img/fly_vs_lsh.png`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

from selector.fly.benchmark import benchmark_hash_lengths

OUTPUT_PATH = Path("docs/img/fly_vs_lsh.png")

HASH_LENGTHS = [4, 8, 16, 32, 64]
N_SAMPLES = 3000  # subsample of MNIST's 70,000 for a tractable brute-force k-NN ground truth
N_QUERIES = 200
N_TRUE_NEIGHBOURS = 10
SEED = 0


def load_mnist_subset(n_samples: int, seed: int) -> np.ndarray:
    print("Fetching MNIST (cached after first run outside the repo)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    return X[idx]


def main() -> None:
    X = load_mnist_subset(N_SAMPLES, SEED)
    print(f"Benchmarking on {X.shape[0]} MNIST digits, {X.shape[1]} pixels each.")

    result = benchmark_hash_lengths(
        X,
        hash_lengths=HASH_LENGTHS,
        n_queries=N_QUERIES,
        n_true_neighbours=N_TRUE_NEIGHBOURS,
        kc_expansion=20,
        sample_size=6,
        seed=SEED,
    )

    print(f"\n{'hash_len':>10} {'FlyHash mAP':>12} {'classical LSH mAP':>18}")
    for k, fly_map, lsh_map in zip(result.hash_lengths, result.fly_map, result.lsh_map):
        print(f"{k:>10} {fly_map:>12.3f} {lsh_map:>18.3f}")

    short_length_win = result.fly_map[0] > result.lsh_map[0]
    if short_length_win:
        print(
            f"\nFlyHash wins at the shortest hash length tested "
            f"({result.hash_lengths[0]} bits): "
            f"{result.fly_map[0]:.3f} vs {result.lsh_map[0]:.3f}. Matches the paper's claim."
        )
    else:
        print(
            "\nFlyHash did NOT beat classical LSH at the shortest hash length. "
            "This does not match the paper's claim — before trusting this result, "
            "check divisive normalisation (selector.fly.lsh.FlyHash._normalise) "
            "and winner-take-all ordering (selector.fly.lsh.FlyHash._wta_count), "
            "which are the two most common places this benchmark silently breaks."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(result.hash_lengths, result.fly_map, marker="o", label="FlyHash (real fan-out + WTA)")
    ax.plot(
        result.hash_lengths, result.lsh_map, marker="s", label="Classical random-projection LSH"
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(result.hash_lengths)
    ax.set_xticklabels([str(k) for k in result.hash_lengths])
    ax.set_xlabel("Hash length (bits compared)")
    ax.set_ylabel("Mean average precision (10-NN retrieval)")
    ax.set_title("FlyHash vs. classical LSH on MNIST")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
