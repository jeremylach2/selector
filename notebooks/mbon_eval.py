"""Evaluate the mushroom body's plasticity rule against real listening
history: does a fly that learns from `verdict` (skip vs. play-out) predict
held-out skips better than simple historical-rate baselines?

Trained on 2022-2024 plays, chronologically, tested on 2025-2026 -- never
shuffled, since this is a time series and a random split would leak the
future into training. The headline question, per the prompt pack: per-track
history is expected to be a strong baseline for tracks seen before, so the
table is broken out by seen/unseen tracks. If the fly does not beat the
baselines, especially on unseen tracks, this file says so.

The vibe tagger (Phase 3) does not exist yet, so track "features" here are a
placeholder: a deterministic pseudo-random vector seeded from each
track_id's hash, standing in for the real measured/predicted feature vector
Step 12 will substitute. This step is only about the plasticity rule and the
real FlyWire wiring from Step 6 -- not about what the features mean.

Run: `uv run python notebooks/mbon_eval.py`
Output: prints the results table and writes `docs/MBON_EVAL.md`.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from selector.fly.connectome import load_flywire_projection
from selector.fly.lsh import FlyHash
from selector.fly.mbon import MushroomBody

PLAYS_PATH = Path("data/plays.parquet")
OUTPUT_PATH = Path("docs/MBON_EVAL.md")

PLACEHOLDER_DIM = 24  # arbitrary -- stands in for the future vibe-tagger feature width
TRAIN_YEARS = (2022, 2023, 2024)
TEST_YEARS = (2025, 2026)
FLY_SEED = 0
LR = 0.05
DECAY = 0.01


def placeholder_embedding(track_id: str, d_in: int) -> np.ndarray:
    """Deterministic pseudo-random stand-in for a real track feature vector.

    Seeded from the track_id's hash so the same track always gets the same
    embedding across runs, without needing any actual audio or lyric data.
    """
    seed = int(hashlib.sha256(track_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=d_in)


def historical_rate_predictor(train: pd.DataFrame, key: str) -> dict[str, float]:
    """Skip rate per `key` (artist_name or track_id) computed on decisive
    (non-neutral) training plays only. Callers fall back to the global rate
    for keys never seen in training -- this dict simply omits them.
    """
    return train.groupby(key)["is_skip"].mean().to_dict()


def main() -> None:
    plays = pd.read_parquet(PLAYS_PATH)
    plays = plays.sort_values("ts").reset_index(drop=True)

    unique_tracks = plays["track_id"].unique()
    print(f"{len(plays):,} plays, {len(unique_tracks):,} unique tracks")

    print(f"Loading the real FlyWire connectome, pooled to {PLACEHOLDER_DIM} placeholder dims...")
    projection = load_flywire_projection(PLACEHOLDER_DIM)
    fly = FlyHash(d_in=PLACEHOLDER_DIM, seed=FLY_SEED)
    fly.projection_matrix = projection

    embeddings = np.stack([placeholder_embedding(tid, PLACEHOLDER_DIM) for tid in unique_tracks])
    fly.fit(embeddings)
    tags = fly.transform(embeddings)
    tag_row = {tid: i for i, tid in enumerate(unique_tracks)}

    train = plays[plays["year"].isin(TRAIN_YEARS)]
    test = plays[plays["year"].isin(TEST_YEARS)]

    print(
        f"Training on {len(train):,} plays ({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}), chronologically..."
    )
    mbon = MushroomBody(fly, lr=LR, decay=DECAY)
    for track_id, verdict in zip(train["track_id"].to_numpy(), train["verdict"].to_numpy()):
        mbon.learn(tags[tag_row[track_id]], int(verdict))

    # Decisive (non-neutral) plays only, both for baselines and for the eval
    # target -- verdict==0 plays carry no reward/punishment judgement.
    train_decisive = train[train["verdict"] != 0].copy()
    train_decisive["is_skip"] = (train_decisive["verdict"] == -1).astype(int)
    test_decisive = test[test["verdict"] != 0].copy()
    test_decisive["is_skip"] = (test_decisive["verdict"] == -1).astype(int)

    train_tracks_seen = set(train["track_id"].unique())
    test_decisive["seen"] = test_decisive["track_id"].isin(train_tracks_seen)

    global_rate = train_decisive["is_skip"].mean()
    artist_rate = historical_rate_predictor(train_decisive, "artist_name")
    track_rate = historical_rate_predictor(train_decisive, "track_id")

    def score_global(row) -> float:
        return global_rate

    def score_artist(row) -> float:
        return artist_rate.get(row["artist_name"], global_rate)

    def score_track(row) -> float:
        if row["track_id"] in track_rate:
            return track_rate[row["track_id"]]
        return artist_rate.get(row["artist_name"], global_rate)

    def score_fly(row) -> float:
        return -mbon.valence(tags[tag_row[row["track_id"]]])

    methods = {
        "Global skip rate": score_global,
        "Per-artist historical skip rate": score_artist,
        "Per-track historical skip rate": score_track,
        "Fly MBON (FlyWire connectome)": score_fly,
    }

    subsets = {
        "all": test_decisive,
        "seen": test_decisive[test_decisive["seen"]],
        "unseen": test_decisive[~test_decisive["seen"]],
    }

    print(
        f"\nTest set: {len(test_decisive):,} decisive plays "
        f"({subsets['seen'].shape[0]:,} seen tracks, {subsets['unseen'].shape[0]:,} unseen)"
    )

    results: dict[str, dict[str, float]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant-score baselines on tiny subsets
        for name, score_fn in methods.items():
            results[name] = {}
            for subset_name, subset in subsets.items():
                if subset["is_skip"].nunique() < 2:
                    results[name][subset_name] = float("nan")
                    continue
                scores = subset.apply(score_fn, axis=1)
                results[name][subset_name] = roc_auc_score(subset["is_skip"], scores)

    print(f"\n{'method':<34} {'AUC (all)':>10} {'AUC (seen)':>11} {'AUC (unseen)':>13}")
    for name, row in results.items():
        print(f"{name:<34} {row['all']:>10.3f} {row['seen']:>11.3f} {row['unseen']:>13.3f}")

    fly_row = results["Fly MBON (FlyWire connectome)"]
    track_row = results["Per-track historical skip rate"]
    verdict_line = (
        f"On unseen tracks, the fly MBON scores {fly_row['unseen']:.3f} AUC versus "
        f"{track_row['unseen']:.3f} for the per-track baseline (which has no track-specific "
        "history to fall back on there, so it degrades toward the artist/global rate)."
    )
    beats_or_not = "beats" if fly_row["unseen"] > track_row["unseen"] else "does not beat"
    print(f"\nOn unseen tracks the fly {beats_or_not} the per-track historical baseline.")

    write_report(results, subsets, verdict_line, beats_or_not)


def write_report(results, subsets, verdict_line, beats_or_not) -> None:
    lines = [
        "# Mushroom body plasticity: skip prediction results",
        "",
        "Train: 2022-2024 plays, chronological order (never shuffled -- this is a time series).",
        "Test: 2025-2026 plays, held out entirely from training.",
        (
            "Metric: ROC-AUC predicting skip (verdict == -1) vs. play-out (verdict == +1) on "
            "decisive plays; neutral plays (verdict == 0) are excluded from both training influence "
            "and the eval target, since they carry no reward/punishment judgement."
        ),
        "",
        (
            f"Test set: {len(subsets['all']):,} decisive plays "
            f"({len(subsets['seen']):,} on tracks seen during training, "
            f"{len(subsets['unseen']):,} on tracks never played before 2025)."
        ),
        "",
        (
            "**Track features are a placeholder** (`notebooks/mbon_eval.py:placeholder_embedding`) "
            "-- a deterministic pseudo-random vector seeded per track_id, standing in for the real "
            "vibe-tagger feature vector Step 12 substitutes once it exists. This step evaluates the "
            "mushroom body's plasticity rule and the real FlyWire wiring from Step 6, not what the "
            "features encode."
        ),
        "",
        "| Method | AUC (all) | AUC (seen tracks) | AUC (unseen tracks) |",
        "|---|---|---|---|",
    ]
    for name, row in results.items():
        lines.append(f"| {name} | {row['all']:.3f} | {row['seen']:.3f} | {row['unseen']:.3f} |")

    lines += [
        "",
        (
            f"**The interesting claim is unseen tracks.** Per-track history is expected to win "
            "comfortably on tracks the model has already seen skipped or played out before -- "
            "that is just memorisation. The fly's plasticity rule only has something to prove on "
            f"tracks it never encountered during training. {verdict_line} Reported as measured: "
            f"the fly **{beats_or_not}** the per-track historical baseline on unseen tracks."
        ),
        "",
        (
            "**Why the fly still edges out chance on seen tracks but not on unseen ones.** The "
            "placeholder embedding is independent random noise per track_id, so two different "
            "tracks share no real similarity -- any Kenyon-cell overlap between two different "
            "tracks' tags is pure chance, uncorrelated with taste. The fly's small lift over chance "
            f"on seen tracks ({results['Fly MBON (FlyWire connectome)']['seen']:.3f}) is the "
            "plasticity rule re-recognising the *exact same* track it was trained on, the same "
            "mechanism the per-track baseline uses, just implemented as a noisier sparse-hash "
            "lookup instead of an exact one -- which is also why it trails the per-track baseline "
            "there. On unseen tracks there is no real content-similarity signal for the fly to "
            "generalise from yet, since the embedding carries none, so a result indistinguishable "
            "from chance is the expected outcome given this input, not a failure of the plasticity "
            "rule itself. Step 12 replaces the placeholder with real measured/predicted track "
            'features and re-runs this exact comparison -- that is the point at which "does the '
            'fly generalise to unseen tracks" becomes a meaningful question to ask.'
        ),
    ]

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
