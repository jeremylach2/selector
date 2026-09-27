"""Taste clusters over the fly brain's Kenyon-cell fingerprints.

Stage 5 of the Wrapped extension plan (`Selector - Project Extension
Plan.md`): cluster every track's fingerprint (`data/fly_tags.npz`) by
Hamming distance with seeded k-medoids, then give each cluster a name built
from its features. No model is involved in the name, so the same logic
ports to the browser demo; `selector.fly.cluster_names` optionally layers a
cached frontier-model headline on top for your own report.

A full 19,386 x 19,386 distance matrix is ~375M entries, so the fit runs on
a seeded sample of the *unique* fingerprints (40% of tracks share an exact
fingerprint with another track), and every track is then assigned to its
nearest medoid. `k` is chosen within `CLUSTER_CONFIG`'s bounds by silhouette
score on that sample.

Fitted results are cached to disk keyed by a hash of the config plus the
fingerprint file's contents, so re-running the report with unchanged inputs
returns the identical clusters without refitting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import silhouette_score

from selector.fly.pipeline import AUDIO_FEATURES_PATH, FLY_TAGS_PATH, TRACK_FEATURES_PATH, load_tags

CLUSTERS_CACHE_DIR = Path("data/taste_clusters")

CLUSTER_CONFIG = {
    "seed": 0,
    "k_min": 6,
    "k_max": 12,
    "fit_sample": 4000,
    "max_iter": 50,
}

# Built-name thresholds. A mood tag names a cluster only if at least this
# share of the cluster's tracks carry it. An era names it only if it holds at
# least NAME_MIN_ERA_SHARE of the cluster *and* is over-represented by
# NAME_MIN_ERA_LIFT versus the library -- 2010s is ~2/3 of the library, so a
# 2010s majority alone says nothing. Energy is named only when the cluster's
# mean intensity sits this many library standard deviations from the mean.
NAME_MIN_TAG_RATE = 0.25
NAME_MIN_ERA_SHARE = 0.25
NAME_MIN_ERA_LIFT = 1.5
NAME_ENERGY_Z = 0.3


@dataclass
class TasteClusters:
    track_ids: list[str]
    labels: np.ndarray  # (n_tracks,) cluster index per track
    distances: np.ndarray  # (n_tracks, k) normalised Hamming distance to each medoid, in [0, 1]
    medoid_track_ids: list[str]
    silhouette: float
    cache_key: str

    @property
    def k(self) -> int:
        return len(self.medoid_track_ids)


# -- distances and k-medoids ----------------------------------------------


def pairwise_hamming(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    """Dense Hamming distances between every row of `a` and every row of
    `b`, via |x| + |y| - 2|x & y| as one sparse product."""
    a = sparse.csr_matrix(a, dtype=np.float32)
    b = sparse.csr_matrix(b, dtype=np.float32)
    inter = np.asarray((a @ b.T).todense(), dtype=np.float32)
    pa = np.asarray(a.sum(axis=1), dtype=np.float32).reshape(-1, 1)
    pb = np.asarray(b.sum(axis=1), dtype=np.float32).reshape(1, -1)
    return pa + pb - 2 * inter


def normalised_hamming(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    """Hamming distance divided by the two tags' combined popcount: 0 for
    identical tags, 1 for tags with no active cell in common."""
    a = sparse.csr_matrix(a, dtype=np.float32)
    b = sparse.csr_matrix(b, dtype=np.float32)
    pa = np.asarray(a.sum(axis=1), dtype=np.float32).reshape(-1, 1)
    pb = np.asarray(b.sum(axis=1), dtype=np.float32).reshape(1, -1)
    denom = np.maximum(pa + pb, 1.0)
    return pairwise_hamming(a, b) / denom


def kmedoids(
    dist: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating k-medoids over a precomputed distance matrix, seeded with
    k-medoids++ (each new medoid drawn with probability proportional to its
    distance from the nearest one already chosen). Returns
    `(medoid_indices, labels)`."""
    n = dist.shape[0]
    medoids = [int(rng.integers(n))]
    nearest = dist[medoids[0]].astype(np.float64)
    for _ in range(1, k):
        probs = nearest / nearest.sum() if nearest.sum() > 0 else None
        nxt = int(rng.choice(n, p=probs))
        medoids.append(nxt)
        nearest = np.minimum(nearest, dist[nxt])
    medoids = np.array(medoids)

    labels = np.argmin(dist[:, medoids], axis=1)
    for _ in range(max_iter):
        new_medoids = medoids.copy()
        for c in range(k):
            members = np.flatnonzero(labels == c)
            if members.size == 0:
                continue
            within = dist[np.ix_(members, members)].sum(axis=1)
            new_medoids[c] = members[np.argmin(within)]
        new_labels = np.argmin(dist[:, new_medoids], axis=1)
        if np.array_equal(new_medoids, medoids):
            break
        medoids, labels = new_medoids, new_labels
    return medoids, labels


# -- fitting and caching ----------------------------------------------------


def _unique_rows(tags: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """`(first_row_of_each_unique_fingerprint, inverse)`, where `inverse[i]`
    is the unique-fingerprint index of track `i`."""
    keys = [tags.indices[tags.indptr[i] : tags.indptr[i + 1]].tobytes() for i in range(tags.shape[0])]
    index: dict[bytes, int] = {}
    firsts: list[int] = []
    inverse = np.empty(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        if key not in index:
            index[key] = len(firsts)
            firsts.append(i)
        inverse[i] = index[key]
    return np.array(firsts), inverse


def cache_key(tags_path: Path, config: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(Path(tags_path).read_bytes())
    return digest.hexdigest()[:12]


def fit_clusters(
    track_ids: list[str], tags: sparse.csr_matrix, config: dict[str, Any] = CLUSTER_CONFIG
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Fit k-medoids on a seeded sample of unique fingerprints, choosing `k`
    by silhouette, then assign every track. Returns
    `(medoid_track_ids, labels, distances, silhouette)`."""
    tags = sparse.csr_matrix(tags)
    rng = np.random.default_rng(config["seed"])
    firsts, _inverse = _unique_rows(tags)
    sample = firsts
    if len(firsts) > config["fit_sample"]:
        sample = np.sort(rng.choice(firsts, size=config["fit_sample"], replace=False))

    dist = normalised_hamming(tags[sample], tags[sample])
    best: tuple[float, np.ndarray] | None = None
    for k in range(config["k_min"], config["k_max"] + 1):
        medoids, labels = kmedoids(dist, k, np.random.default_rng(config["seed"] + k), config["max_iter"])
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(dist, labels, metric="precomputed"))
        if best is None or score > best[0]:
            best = (score, sample[medoids])
    if best is None:
        raise ValueError("k-medoids produced no usable clustering")

    score, medoid_rows = best
    distances = normalised_hamming(tags, tags[medoid_rows])
    labels = np.argmin(distances, axis=1)
    return [track_ids[r] for r in medoid_rows], labels, distances, score


def load_or_fit_clusters(
    tags_path: Path = FLY_TAGS_PATH,
    config: dict[str, Any] = CLUSTER_CONFIG,
    cache_dir: Path = CLUSTERS_CACHE_DIR,
) -> TasteClusters:
    """Cached `fit_clusters`: identical config and fingerprints return the
    identical clustering from disk."""
    key = cache_key(tags_path, config)
    path = Path(cache_dir) / f"{key}.npz"
    track_ids, tags = load_tags(tags_path)
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            return TasteClusters(
                track_ids=track_ids,
                labels=f["labels"],
                distances=f["distances"],
                medoid_track_ids=f["medoid_track_ids"].tolist(),
                silhouette=float(f["silhouette"]),
                cache_key=key,
            )

    medoid_ids, labels, distances, score = fit_clusters(track_ids, tags, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        labels=labels,
        distances=distances.astype(np.float32),
        medoid_track_ids=np.array(medoid_ids),
        silhouette=np.array(score),
    )
    return TasteClusters(track_ids, labels, distances, medoid_ids, score, key)


# -- profiles and built names -------------------------------------------------


def track_feature_frame(
    track_features_path: Path = TRACK_FEATURES_PATH,
    audio_features_path: Path = AUDIO_FEATURES_PATH,
) -> pd.DataFrame:
    """Per-track predicted features plus a `has_measured` flag, indexed by
    `track_id`."""
    feats = pd.read_parquet(
        track_features_path, columns=["track_id", "valence", "intensity", "era", "mood_tags"]
    ).set_index("track_id")
    measured: set[str] = set()
    if Path(audio_features_path).exists():
        measured = set(pd.read_parquet(audio_features_path, columns=["track_id"])["track_id"])
    feats["has_measured"] = feats.index.isin(measured)
    return feats


def _tag_rates(mood_tags: pd.Series) -> pd.Series:
    exploded = mood_tags.dropna().explode()
    return exploded.value_counts() / max(len(mood_tags), 1)


def built_name(members: pd.DataFrame, library: pd.DataFrame) -> str:
    """Name a cluster from the features that most set it apart from the
    library: up to two mood tags by lift over the library rate, an energy
    word if mean intensity is far enough from the library's, and the era if
    one is strongly over-represented. e.g. "Melancholic · nostalgic ·
    low-key · 1970s".
    """
    parts: list[str] = []
    rates = _tag_rates(members["mood_tags"])
    lib_rates = _tag_rates(library["mood_tags"])
    eligible = rates[rates >= NAME_MIN_TAG_RATE]
    if eligible.empty and not rates.empty:
        eligible = rates.head(1)
    lift = (eligible / lib_rates.reindex(eligible.index)).sort_values(ascending=False)
    parts.extend(lift.head(2).index.tolist())

    lib_std = float(library["intensity"].std()) or 1.0
    z = (float(members["intensity"].mean()) - float(library["intensity"].mean())) / lib_std
    if z >= NAME_ENERGY_Z:
        parts.append("high-energy")
    elif z <= -NAME_ENERGY_Z:
        parts.append("low-key")

    eras = members["era"].value_counts(normalize=True)
    eras = eras[eras >= NAME_MIN_ERA_SHARE]
    if not eras.empty:
        era_lift = eras / library["era"].value_counts(normalize=True).reindex(eras.index)
        if era_lift.max() >= NAME_MIN_ERA_LIFT:
            parts.append(era_lift.idxmax())

    if parts:
        parts[0] = parts[0].capitalize()
    return " · ".join(parts)


def cluster_summary(
    clusters: TasteClusters, tracks: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """One row per cluster: built name, track and play counts, play share,
    share of tracks with measured audio, top artists, and exemplar track IDs
    (most-played first). `tracks` is the warehouse tracks table (`track_id`,
    `name`, `artist`, `play_count`).
    """
    frame = pd.DataFrame({"track_id": clusters.track_ids, "cluster": clusters.labels})
    frame = frame.merge(tracks, on="track_id", how="left").fillna({"play_count": 0})
    frame = frame.join(features, on="track_id")
    total_plays = frame["play_count"].sum()

    rows = []
    for c in range(clusters.k):
        members = frame[frame["cluster"] == c]
        by_plays = members.sort_values("play_count", ascending=False)
        top_artists = (
            members.groupby("artist")["play_count"].sum().sort_values(ascending=False).head(3)
        )
        rows.append(
            {
                "cluster": c,
                "built_name": built_name(members, frame),
                "track_count": len(members),
                "play_count": int(members["play_count"].sum()),
                "play_share": float(members["play_count"].sum() / total_plays) if total_plays else 0.0,
                "measured_share": float(members["has_measured"].mean()) if len(members) else 0.0,
                "top_artists": top_artists.index.tolist(),
                "exemplars": by_plays["track_id"].head(10).tolist(),
                "medoid_track_id": clusters.medoid_track_ids[c],
            }
        )

    summary = pd.DataFrame(rows)
    # Two clusters can land on the same built name; disambiguate by each
    # one's top artist rather than an opaque number.
    dupes = summary["built_name"].duplicated(keep=False)
    summary.loc[dupes, "built_name"] = [
        f"{name} ({artists[0]})" if artists else name
        for name, artists in zip(summary.loc[dupes, "built_name"], summary.loc[dupes, "top_artists"])
    ]
    return summary.sort_values("play_share", ascending=False).reset_index(drop=True)


def main() -> None:
    from selector.warehouse import queries
    from selector.warehouse.build import DEFAULT_DB_PATH

    clusters = load_or_fit_clusters()
    with queries._connect(DEFAULT_DB_PATH) as con:
        tracks = con.execute("SELECT track_id, name, artist, play_count FROM tracks").df()
    summary = cluster_summary(clusters, tracks, track_feature_frame())
    print(f"k={clusters.k}, silhouette={clusters.silhouette:.3f}, cache key {clusters.cache_key}")
    for r in summary.itertuples():
        print(
            f"  {r.play_share:5.1%}  {r.track_count:5,} tracks  {r.built_name}"
            f"  [{', '.join(r.top_artists)}]"
        )


if __name__ == "__main__":
    main()
