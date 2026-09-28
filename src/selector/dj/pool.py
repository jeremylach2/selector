"""The DJ's crate: every track it is allowed to play, with everything the
five stages need attached to one row.

A track is only in the crate if it has **measured** audio features
(`data/audio_features.parquet`, Step 9), about 3,200 of the 19,386
warehouse tracks. That restriction is deliberate: the energy arc is a hard
constraint, and a hard constraint checked against a model's *guess* at
energy would be a constraint on nothing. Everything else on the row mixes
three provenances, and the column names keep them apart:

- measured (DSP on a 30-second preview): `tempo`, `energy`
- predicted (fine-tuned tagger, Step 11): `mood_tags`, `pred_valence`
- learned (fly brain, Step 12): `tag_row` into the fly tag matrix,
  `fly_valence` from the production mushroom body
- observed (the warehouse): play counts, recency, durations
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

from selector.fly import pipeline as fly_pipeline
from selector.warehouse.build import DEFAULT_DB_PATH

# Perceived energy is loudness, activity and brightness together, the same
# ingredients Spotify's own (now removed) `energy` feature was documented as
# combining. Each column is already min-max scaled to [0, 1] by
# `selector.audio.merge`; the weighted sum is then rank-normalised over the
# crate so arc targets like "0.8" mean "louder than 80% of the crate",
# independent of how skewed any one DSP column happens to be.
ENERGY_WEIGHTS = {
    "rms_mean_scaled": 0.5,
    "onset_density_scaled": 0.25,
    "spectral_centroid_mean_scaled": 0.25,
}

# librosa returns 0 BPM when it can't find a beat grid at all (ambient
# intros, spoken word). Treated as "no tempo" rather than a real value.
MIN_VALID_BPM = 40.0

# Duration of a track isn't in the export directly; the longest play that
# ended with `trackdone` is. Tracks never played to the end, or whose
# "trackdone" play is implausibly short (a glitch, not a song), fall back to
# the crate median. Tracks over the cap are left out of the crate: one
# 25-minute live cut would eat half a 45-minute set.
FALLBACK_DURATION_MS = 210_000
MIN_DURATION_MS = 60_000
MAX_DURATION_MS = 600_000

# "Familiar" means in current rotation: played within this many days of the
# newest play in the warehouse. Every crate track has been played at some
# point (the crate is drawn from the listening history), so the other side
# of the ratio is "not heard lately", i.e. rediscovery, not literally new.
FAMILIAR_WINDOW_DAYS = 90


@dataclass
class Crate:
    """`tracks` is one row per playable track (see module docstring for the
    columns); `tags` is the full fly tag matrix, indexed by `tracks.tag_row`."""

    tracks: pd.DataFrame
    tags: sparse.csr_matrix
    as_of: datetime

    def by_id(self) -> pd.DataFrame:
        return self.tracks.set_index("track_id", drop=False)


def measured_energy(audio: pd.DataFrame) -> pd.Series:
    """Composite perceived energy, rank-normalised to [0, 1] over `audio`."""
    raw = sum(audio[col].fillna(audio[col].median()) * w for col, w in ENERGY_WEIGHTS.items())
    return raw.rank(pct=True, method="average")


def _warehouse_rows(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(
            """
            SELECT
                t.track_id, t.name, t.artist, t.album, t.play_count,
                t.skip_rate, t.last_played,
                MAX(CASE WHEN p.reason_end = 'trackdone' THEN p.ms_played END) AS duration_ms
            FROM tracks t
            LEFT JOIN plays p USING (track_id)
            GROUP BY ALL
            """
        ).df()


def build_crate(
    db_path: Path = DEFAULT_DB_PATH,
    audio_features_path: Path = fly_pipeline.AUDIO_FEATURES_PATH,
    track_features_path: Path = fly_pipeline.TRACK_FEATURES_PATH,
    fly_tags_path: Path = fly_pipeline.FLY_TAGS_PATH,
    familiar_window_days: int = FAMILIAR_WINDOW_DAYS,
) -> Crate:
    """Join measured audio, predicted labels, fly tags and warehouse stats
    into one crate, and train the production mushroom body to score it."""
    audio = pd.read_parquet(audio_features_path)
    audio = audio.assign(energy=measured_energy(audio))
    audio["tempo"] = audio["tempo"].where(audio["tempo"] >= MIN_VALID_BPM)
    audio = audio[["track_id", "tempo", "energy"]]

    predicted = pd.read_parquet(track_features_path, columns=["track_id", "valence", "mood_tags"])
    predicted = predicted.rename(columns={"valence": "pred_valence"})

    track_ids, tags = fly_pipeline.load_tags(fly_tags_path)
    tag_rows = pd.DataFrame({"track_id": track_ids, "tag_row": np.arange(len(track_ids))})

    crate = (
        audio.merge(_warehouse_rows(db_path), on="track_id")
        .merge(predicted, on="track_id")
        .merge(tag_rows, on="track_id")
    )
    duration = crate["duration_ms"].where(crate["duration_ms"] >= MIN_DURATION_MS)
    crate["duration_ms"] = duration.fillna(FALLBACK_DURATION_MS).astype(int)
    crate = crate[crate["duration_ms"] <= MAX_DURATION_MS]
    crate["mood_tags"] = crate["mood_tags"].apply(lambda t: list(t) if t is not None else [])

    mbon = fly_pipeline.train_production_mbon(track_ids, tags)
    crate["fly_valence"] = [mbon.valence(tags[r]) for r in crate["tag_row"]]
    # Raw MBON valence is a sum over ~130 active KCs, so its scale depends on
    # tag density, and because play-outs outnumber skips ~4:1 in the history
    # it is positive for essentially every track. Only its *rank* carries
    # information, so the percentile is what Select and the notes use.
    crate["taste"] = crate["fly_valence"].rank(pct=True)

    as_of = pd.Timestamp(crate["last_played"].max()).to_pydatetime()
    cutoff = as_of - timedelta(days=familiar_window_days)
    crate["familiar"] = crate["last_played"] >= cutoff

    return Crate(tracks=crate.reset_index(drop=True), tags=tags, as_of=as_of)


def recent_plays(db_path: Path = DEFAULT_DB_PATH, limit: int = 50) -> pd.DataFrame:
    """The newest `limit` plays in the warehouse, newest first."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(
            """
            SELECT ts, track_id, track_name, artist_name, verdict
            FROM plays WHERE track_id IS NOT NULL
            ORDER BY ts DESC LIMIT ?
            """,
            [limit],
        ).df()
