"""Wire the vibe tagger's output into the fly brain.

Step 7 trained and evaluated the mushroom body against a placeholder
embedding -- deterministic noise per `track_id`, standing in for a feature
vector that didn't exist yet. This module builds the real one, from two
files Phase 3 produces:

- `data/track_features.parquet` (`selector.tagger.infer`): the fine-tuned
  student's **predicted** labels for every one of the 19,386 warehouse
  tracks -- valence, intensity, era, mood_tags.
- `data/audio_features.parquet` (`selector.audio.merge`): **measured** DSP
  features for the ~3,198 tracks a preview clip could be matched to.

Three feature sources are supported, all built by `build_feature_matrix`,
so `notebooks/mbon_eval.py` can re-run Step 7's comparison across them
side by side:

- `"placeholder"` -- Step 7's original noise embedding, kept only as the
  zero baseline for that comparison.
- `"text_only"` -- the predicted labels alone (valence, intensity, era,
  mood_tags), available for every track regardless of audio match.
- `"full"` -- text features plus the measured audio features where a match
  exists, flagged with a `has_measured` bit. This is the production source:
  `build_and_persist_tags` (source="full") is what `data/fly_tags.npz` and
  the MCP tools below it are built from.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd
from scipy import sparse

from selector.fly.connectome import load_flywire_projection
from selector.fly.lsh import FlyHash
from selector.fly.mbon import MushroomBody
from selector.tagger.schema import Era, MoodTag

TRACK_FEATURES_PATH = Path("data/track_features.parquet")
AUDIO_FEATURES_PATH = Path("data/audio_features.parquet")
PLAYS_PATH = Path("data/plays.parquet")
FLY_TAGS_PATH = Path("data/fly_tags.npz")

# Same four scaled columns `selector.tagger.enrich` gives the teacher/student
# as "measured" context -- reused here so the fly sees the same measured
# signal the tagger was scored against, not a different slice of the 94
# DSP columns in `audio_features.parquet`.
MEASURED_COLUMNS = ["tempo_scaled", "rms_mean_scaled", "danceability", "harmonic_percussive_ratio_scaled"]

MOOD_VOCAB = list(get_args(MoodTag))
ERA_VOCAB = list(get_args(Era))

FeatureSource = Literal["placeholder", "text_only", "full"]

FLY_SEED = 0
MBON_LR = 0.05
MBON_DECAY = 0.01


# -- feature vectors ------------------------------------------------------


def placeholder_embedding(track_id: str, d_in: int) -> np.ndarray:
    """Step 7's noise embedding: deterministic pseudo-random per `track_id`,
    carrying no real similarity signal. Kept only so the Step 12 comparison
    has the original zero baseline to compare against, not as a feature
    source anything should actually be built on.
    """
    seed = int(hashlib.sha256(track_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=d_in)


def _predicted_vector(row: pd.Series) -> np.ndarray:
    """valence + intensity + one-hot era + multi-hot mood_tags, in that
    fixed column order. Every warehouse track has a row in
    `track_features.parquet` (arm A at worst), so this is always available
    -- `lyrical_theme` is deliberately left out: it's free text with no
    controlled vocabulary, so there's no honest fixed-width numeric encoding
    for it here.
    """
    era_one_hot = np.array([1.0 if row["era"] == era else 0.0 for era in ERA_VOCAB])
    tags = set(row["mood_tags"]) if row["mood_tags"] is not None else set()
    mood_multi_hot = np.array([1.0 if tag in tags else 0.0 for tag in MOOD_VOCAB])
    return np.concatenate([[row["valence"], row["intensity"]], era_one_hot, mood_multi_hot])


PREDICTED_DIM = 2 + len(ERA_VOCAB) + len(MOOD_VOCAB)
FULL_DIM = PREDICTED_DIM + len(MEASURED_COLUMNS) + 1  # + has_measured flag


def build_feature_matrix(
    source: FeatureSource,
    track_features_path: Path = TRACK_FEATURES_PATH,
    audio_features_path: Path = AUDIO_FEATURES_PATH,
) -> tuple[list[str], np.ndarray]:
    """Build `(track_ids, X)` for one feature source, `X` shape `(n, d_in)`.

    `track_ids` is every track in `track_features.parquet` (the full
    19,386-track warehouse), in that file's row order.
    """
    track_features = pd.read_parquet(track_features_path)
    track_ids = track_features["track_id"].tolist()

    if source == "placeholder":
        d_in = PREDICTED_DIM
        X = np.stack([placeholder_embedding(tid, d_in) for tid in track_ids])
        return track_ids, X

    predicted = np.stack([_predicted_vector(row) for _, row in track_features.iterrows()])
    if source == "text_only":
        return track_ids, predicted

    if source != "full":
        raise ValueError(f"unknown feature source: {source!r}")

    measured_by_track: dict[str, np.ndarray] = {}
    if audio_features_path.exists():
        audio_features = pd.read_parquet(audio_features_path)
        for c in MEASURED_COLUMNS:
            if c not in audio_features.columns:
                audio_features[c] = np.nan
        for _, row in audio_features.iterrows():
            measured_by_track[row["track_id"]] = row[MEASURED_COLUMNS].to_numpy(dtype=np.float64)

    measured_rows = []
    has_measured_col = []
    for tid in track_ids:
        vec = measured_by_track.get(tid)
        if vec is None:
            measured_rows.append(np.zeros(len(MEASURED_COLUMNS)))
            has_measured_col.append(0.0)
        else:
            measured_rows.append(np.nan_to_num(vec, nan=0.0))
            has_measured_col.append(1.0)
    measured = np.stack(measured_rows)
    has_measured = np.array(has_measured_col).reshape(-1, 1)

    X = np.concatenate([predicted, measured, has_measured], axis=1)
    return track_ids, X


# -- fly fitting and persistence ------------------------------------------


def fit_fly(track_ids: list[str], X: np.ndarray, seed: int = FLY_SEED) -> tuple[FlyHash, sparse.csr_matrix]:
    """Fit a `FlyHash` wired with the real FlyWire connectome (Step 6) over
    `X`, and return `(fly, tags)`. `fly.fit(X)` computes normalisation
    statistics from the same data being tagged -- fine here since the fly
    brain is a fixed hash, not a model being evaluated for generalisation
    (that check happens downstream, in the MBON eval)."""
    d_in = X.shape[1]
    fly = FlyHash(d_in=d_in, seed=seed)
    fly.projection_matrix = load_flywire_projection(d_in)
    fly.fit(X)
    tags = fly.transform(X)
    return fly, tags


def save_tags(path: Path, track_ids: list[str], tags: sparse.csr_matrix) -> None:
    """Persist a sparse tag matrix plus its row order to one `.npz` file."""
    tags = sparse.csr_matrix(tags)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        track_ids=np.array(track_ids),
        data=tags.data,
        indices=tags.indices,
        indptr=tags.indptr,
        shape=np.array(tags.shape),
    )


def load_tags(path: Path = FLY_TAGS_PATH) -> tuple[list[str], sparse.csr_matrix]:
    """Inverse of `save_tags`."""
    with np.load(path, allow_pickle=False) as f:
        tags = sparse.csr_matrix((f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"]))
        track_ids = f["track_ids"].tolist()
    return track_ids, tags


def build_and_persist_tags(
    source: FeatureSource = "full",
    output_path: Path = FLY_TAGS_PATH,
    track_features_path: Path = TRACK_FEATURES_PATH,
    audio_features_path: Path = AUDIO_FEATURES_PATH,
) -> tuple[list[str], sparse.csr_matrix]:
    """The item-1 pipeline: build feature vectors, fit the real-connectome
    fly, and persist every track's fingerprint to `output_path`."""
    track_ids, X = build_feature_matrix(source, track_features_path, audio_features_path)
    _fly, tags = fit_fly(track_ids, X)
    save_tags(output_path, track_ids, tags)
    return track_ids, tags


# -- production MBON --------------------------------------------------------


def train_production_mbon(
    track_ids: list[str],
    tags: sparse.csr_matrix,
    plays_path: Path = PLAYS_PATH,
    lr: float = MBON_LR,
    decay: float = MBON_DECAY,
) -> MushroomBody:
    """Train a mushroom body on the *entire* chronological play history.

    This is deliberately not the train/test split `notebooks/mbon_eval.py`
    uses -- that split exists to measure whether the plasticity rule
    generalises to unseen tracks honestly, by holding out 2025-2026. This
    function is for the MCP tools below, which want the best taste model
    available and have no reason to withhold real listening history from
    it. `tags.shape[1]` (not a fitted `FlyHash`) is all `MushroomBody` needs
    to size its synapse arrays, so no `FlyHash` instance has to be
    reconstructed just to read `.n_kc` off it.
    """
    tag_row = {tid: i for i, tid in enumerate(track_ids)}
    plays = pd.read_parquet(plays_path, columns=["track_id", "ts", "verdict"]).sort_values("ts")

    n_kc = tags.shape[1]
    mbon = MushroomBody(_NKcHolder(n_kc), lr=lr, decay=decay)
    for track_id, verdict in zip(plays["track_id"].to_numpy(), plays["verdict"].to_numpy()):
        row = tag_row.get(track_id)
        if row is not None:
            mbon.learn(tags[row], int(verdict))
    return mbon


class _NKcHolder:
    """Minimal stand-in for a `FlyHash`, exposing only the `.n_kc` attribute
    `MushroomBody.__init__` reads -- see `train_production_mbon`."""

    def __init__(self, n_kc: int) -> None:
        self.n_kc = n_kc


def main() -> None:
    print("Building feature vectors (source=full) and fitting the fly...")
    track_ids, tags = build_and_persist_tags(source="full")
    active_per_row = np.asarray(tags.sum(axis=1)).ravel()
    print(
        f"Tagged {len(track_ids):,} tracks -> {tags.shape[1]:,} Kenyon cells, "
        f"{active_per_row.mean():.1f} active on average "
        f"({active_per_row.mean() / tags.shape[1]:.1%}). Wrote {FLY_TAGS_PATH}."
    )


if __name__ == "__main__":
    main()
