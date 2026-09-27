"""Build the static assets the public demo (`web/`) runs on.

The demo has no backend: the visitor's export is parsed in their browser and
the fly brain runs client-side. What it needs from this repo is shipped as
static files under `web/public/`:

- `fly/catalog.json.gz` -- one row per tagged track: name, artist,
  predicted mood tags and era, and (for the ~3,200 tracks with a matched
  preview) the **measured** tempo, energy and duration the DJ's arc runs on.
  No play counts, skip rates or timestamps: nothing about how the author
  listened, only what the tracks are. Rows are sorted by track id so the
  file order carries no ranking either.
- `fly/tags.bin.gz` -- every track's fly-brain fingerprint (the 130 active
  Kenyon cells of 2,597) as delta-encoded uint8 gaps, row order matching the
  catalog. The browser rebuilds a bitset from it for Hamming search.
- `fly/circuit.json.gz` -- what the `/watch` visualiser needs to rerun the
  hash live in the browser rather than read the finished tags: the pooled
  FlyWire PN->KC projection exactly as `FlyHash` stored it, the `fit()`
  normalisation statistics, and every catalog track's 24-number input
  vector (predicted valence, intensity, era, moods; measured audio where a
  preview matched). The build recomputes every tag from this file the way
  the browser will and refuses to write it if any differs from
  `data/fly_tags.npz` outside ties at the winner-take-all cutoff.
- `sample/sample_spotify_data.zip` -- a **synthetic** Extended Streaming History
  export in Spotify's real schema, for visitors with no export of their own.
  The listener is invented: tracks are drawn from the catalog around three
  fly-brain neighbourhoods (one per "era"), timestamps come from a made-up
  daily rhythm, and skips from per-track dice rolls. No `ip_addr`, no real
  timestamps, no real play history.

Audio is never shipped, only derived numbers.

Usage: uv run python scripts/build_demo_assets.py
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from selector.dj.pool import (
    FALLBACK_DURATION_MS,
    MAX_DURATION_MS,
    MIN_DURATION_MS,
    MIN_VALID_BPM,
    measured_energy,
)
from selector.fly import pipeline as fly_pipeline
from selector.fly.connectome import (
    DATA_VERSION,
    build_projection_matrix,
    download_flywire_data,
)
from selector.fly.lsh import hamming_distances
from selector.warehouse.build import DEFAULT_DB_PATH

WEB_PUBLIC = Path("web/public")
CATALOG_PATH = WEB_PUBLIC / "fly/catalog.json.gz"
TAGS_PATH = WEB_PUBLIC / "fly/tags.bin.gz"
CIRCUIT_PATH = WEB_PUBLIC / "fly/circuit.json.gz"
SAMPLE_PATH = WEB_PUBLIC / "sample/sample_spotify_data.zip"

SEED = 7

# The invented listener's three eras, each anchored on the catalog tracks
# closest (in fly space) to a mood profile. Their libraries overlap a little
# at the seams, so taste drift reads as a drift rather than three strangers.
ERAS = (
    {"start": "2023-01-01", "end": "2023-12-31", "moods": ("chill", "nostalgic", "romantic")},
    {"start": "2024-01-01", "end": "2024-12-31", "moods": ("euphoric", "playful", "triumphant")},
    {"start": "2025-01-01", "end": "2025-11-30", "moods": ("melancholic", "anxious", "somber")},
)
ERA_LIBRARY_SIZE = 260
MAX_TRACKS_PER_ARTIST = 6
SAMPLE_TZ_OFFSET_H = -5  # the invented listener lives on US Central-ish time


def _encode_tags(tags) -> bytes:
    """Delta-encode each row's sorted active indices as uint8 gaps. A gap
    over 255 is written as 0 followed by the gap as little-endian uint16 --
    the first gap is from -1, so a real gap is never 0."""
    out = bytearray()
    for r in range(tags.shape[0]):
        prev = -1
        for idx in tags.indices[tags.indptr[r] : tags.indptr[r + 1]]:
            gap = int(idx) - prev
            if gap < 256:
                out.append(gap)
            else:
                out.append(0)
                out += gap.to_bytes(2, "little")
            prev = int(idx)
    return bytes(out)


def build_catalog() -> tuple[pd.DataFrame, object]:
    track_ids, tags = fly_pipeline.load_tags()
    tags.sort_indices()
    active = np.diff(tags.indptr)
    if not (active == active[0]).all():
        raise ValueError("expected a fixed number of active Kenyon cells per track")

    features = pd.read_parquet(fly_pipeline.TRACK_FEATURES_PATH, columns=["track_id", "mood_tags", "era"])

    audio = pd.read_parquet(fly_pipeline.AUDIO_FEATURES_PATH)
    audio = audio.assign(energy=measured_energy(audio))
    audio["tempo"] = audio["tempo"].where(audio["tempo"] >= MIN_VALID_BPM)
    audio = audio[["track_id", "tempo", "energy"]]

    with duckdb.connect(str(DEFAULT_DB_PATH), read_only=True) as con:
        names = con.execute(
            """
            SELECT t.track_id, t.name, t.artist, t.album,
                   MAX(CASE WHEN p.reason_end = 'trackdone' THEN p.ms_played END) AS duration_ms
            FROM tracks t LEFT JOIN plays p USING (track_id)
            GROUP BY ALL
            """
        ).df()

    cat = (
        pd.DataFrame({"track_id": track_ids, "tag_row": np.arange(len(track_ids))})
        .merge(names, on="track_id")
        .merge(features, on="track_id")
        .merge(audio, on="track_id", how="left")
    )
    duration_ms = cat["duration_ms"].where(cat["duration_ms"] >= MIN_DURATION_MS)
    duration_ms = duration_ms.fillna(FALLBACK_DURATION_MS)
    cat["duration_s"] = (duration_ms / 1000).round()
    # Same rule as the DJ's crate (`build_crate` in `selector/dj/pool.py`): a
    # track with no completed play falls back to a 210s guess rather than
    # being dropped, so the demo and the MCP tool draw from the same track
    # set. Only an implausibly long measured duration (a 10-minute-plus live
    # cut) excludes a track from the arc.
    in_crate = cat["energy"].notna() & (duration_ms <= MAX_DURATION_MS)
    cat.loc[~in_crate, ["tempo", "energy"]] = np.nan
    cat = cat.sort_values("track_id").reset_index(drop=True)
    return cat, tags


def write_catalog(cat: pd.DataFrame, tags) -> None:
    mood_vocab = fly_pipeline.MOOD_VOCAB
    era_vocab = fly_pipeline.ERA_VOCAB
    mood_bits = [
        sum(1 << mood_vocab.index(m) for m in (tags_ if tags_ is not None else []) if m in mood_vocab)
        for tags_ in cat["mood_tags"]
    ]

    def col(values, digits=None):
        return [None if pd.isna(v) else (round(float(v), digits) if digits is not None else v) for v in values]

    payload = {
        "version": 1,
        "n_kc": int(tags.shape[1]),
        "k_active": int(np.diff(tags.indptr)[0]),
        "mood_vocab": mood_vocab,
        "era_vocab": era_vocab,
        "id": cat["track_id"].tolist(),
        "name": cat["name"].tolist(),
        "artist": cat["artist"].tolist(),
        "mood": mood_bits,
        "era": [era_vocab.index(e) if e in era_vocab else -1 for e in cat["era"]],
        "tempo": col(cat["tempo"], 1),
        "energy": col(cat["energy"], 3),
        "duration": [None if pd.isna(e) else int(d) for e, d in zip(cat["energy"], cat["duration_s"].fillna(0))],
    }
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), 9))

    reordered = tags[cat["tag_row"].to_numpy()]
    reordered.sort_indices()
    TAGS_PATH.write_bytes(gzip.compress(_encode_tags(reordered), 9))
    print(
        f"catalog: {len(cat):,} tracks ({cat['energy'].notna().sum():,} in the DJ crate), "
        f"{CATALOG_PATH.stat().st_size / 1e6:.2f} MB; tags {TAGS_PATH.stat().st_size / 1e6:.2f} MB"
    )


def write_circuit(cat: pd.DataFrame, tags) -> None:
    """The live-circuit asset for `/watch`: projection, fit statistics and
    input vectors, in catalog row order. See the module docstring."""
    track_ids, X = fly_pipeline.build_feature_matrix("full")
    fly, _ = fly_pipeline.fit_fly(track_ids, X)
    X = X[cat["tag_row"].to_numpy()]
    proj = fly.projection_matrix
    raw, _, _ = build_projection_matrix(*download_flywire_data())

    # Recompute the tags the way `web/lib/circuit.ts` does, row by row in
    # stored CSR order, and compare with the shipped fingerprints.
    norm = np.clip((X - fly._mean) / fly._std, 0.0, None)
    act = np.zeros((len(X), proj.shape[0]))
    for i in range(proj.shape[0]):
        for jj in range(proj.indptr[i], proj.indptr[i + 1]):
            act[:, i] += proj.data[jj] * norm[:, proj.indices[jj]]
    k = int(np.diff(tags.indptr)[0])
    reordered = tags[cat["tag_row"].to_numpy()]
    reordered.sort_indices()
    cutoff = -np.sort(-act, axis=1)[:, k - 1 : k]
    above = act > cutoff
    for r in range(len(X)):
        shipped = np.zeros(proj.shape[0], dtype=bool)
        shipped[reordered.indices[reordered.indptr[r] : reordered.indptr[r + 1]]] = True
        tied = act[r] == cutoff[r, 0]
        if (above[r] & ~shipped).any() or (shipped & ~(above[r] | tied)).any():
            raise ValueError(f"recomputed tag for {cat['track_id'].iat[r]} differs from fly_tags.npz")

    columns = (
        [("valence", "predicted"), ("intensity", "predicted")]
        + [(f"era {e}", "predicted") for e in fly_pipeline.ERA_VOCAB]
        + [(m, "predicted") for m in fly_pipeline.MOOD_VOCAB]
        + [(c.removesuffix("_scaled").replace("_", " "), "measured") for c in fly_pipeline.MEASURED_COLUMNS]
        + [("has audio", "measured")]
    )
    measured = np.flatnonzero(X[:, -1] > 0)
    payload = {
        "version": 1,
        "flywire": DATA_VERSION,
        "hemisphere": "right",
        "n_pn_real": int(raw.shape[1]),
        "raw_synapses": int(raw.data.sum()),
        "columns": [{"name": n, "kind": kind} for n, kind in columns],
        "mean": fly._mean.tolist(),
        "std": fly._std.tolist(),
        "indptr": proj.indptr.tolist(),
        "indices": proj.indices.tolist(),
        "data": [int(v) for v in proj.data],
        "valence": X[:, 0].tolist(),
        "intensity": X[:, 1].tolist(),
        "measured_rows": measured.tolist(),
        "measured": X[measured][:, -1 - len(fly_pipeline.MEASURED_COLUMNS) : -1].tolist(),
    }
    CIRCUIT_PATH.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), 9))
    ties = int((np.abs(act - cutoff) == 0).sum(axis=1).__gt__(1).sum())
    print(f"circuit: {CIRCUIT_PATH.stat().st_size / 1e6:.2f} MB, every tag reproduced ({ties:,} rows tie at the cutoff)")


# -- the synthetic listener -------------------------------------------------


def _era_library(cat: pd.DataFrame, tags, moods: tuple[str, ...], rng, taken: set[int]) -> np.ndarray:
    """Catalog rows around one mood profile: pick an anchor among tracks
    tagged with those moods, then take its nearest fly-brain neighbours, with
    a bias towards the DJ crate so the sample exercises the DJ too."""
    has = cat["mood_tags"].apply(lambda t: t is not None and len(set(t) & set(moods)) >= 2)
    in_crate = cat["energy"].notna()
    anchors = np.flatnonzero((has & in_crate).to_numpy())
    rows = cat["tag_row"].to_numpy()
    anchor = anchors[rng.integers(len(anchors))]
    dist = hamming_distances(tags[rows[anchor]], tags[rows])
    # Favour on-mood and in-crate tracks, then fly distance.
    score = dist - 40 * has.to_numpy() - 25 * in_crate.to_numpy()
    # Each era gets its own artists, at most a handful of tracks apiece, so
    # one prolific catalog artist can't swamp a whole era.
    artists = cat["artist"].to_numpy()
    used_artists = {artists[i] for i in taken}
    per_artist: dict[str, int] = {}
    library = []
    for i in np.argsort(score):
        a = artists[i]
        if i in taken or a in used_artists or per_artist.get(a, 0) >= MAX_TRACKS_PER_ARTIST:
            continue
        per_artist[a] = per_artist.get(a, 0) + 1
        library.append(i)
        if len(library) == ERA_LIBRARY_SIZE:
            break
    # Shuffle so the Zipf weights in `build_sample` don't all land on the
    # anchor's closest neighbours.
    return rng.permutation(np.array(library))


def _session_starts(day: datetime, rng) -> list[datetime]:
    """A made-up daily rhythm in the listener's local time: a commute block
    on weekdays, afternoons at weekends, and most evenings."""
    weekend = day.weekday() >= 5
    starts = []
    if not weekend and rng.random() < 0.7:
        starts.append(day.replace(hour=7) + timedelta(minutes=int(rng.integers(30, 110))))
    if not weekend and rng.random() < 0.45:
        starts.append(day.replace(hour=12) + timedelta(minutes=int(rng.integers(0, 60))))
    if weekend and rng.random() < 0.75:
        starts.append(day.replace(hour=13) + timedelta(minutes=int(rng.integers(0, 180))))
    if rng.random() < (0.8 if day.weekday() in (4, 5) else 0.55):
        starts.append(day.replace(hour=19) + timedelta(minutes=int(rng.integers(0, 240))))
    return starts


def build_sample(cat: pd.DataFrame, tags) -> None:
    rng = np.random.default_rng(SEED)
    duration = cat["duration_s"].fillna(pd.Series(rng.integers(150, 270, len(cat)), index=cat.index))

    taken: set[int] = set()
    libraries = []
    for era in ERAS:
        lib = _era_library(cat, tags, era["moods"], rng, taken)
        taken.update(lib.tolist())
        libraries.append(lib)

    # A dozen tracks the listener keeps queuing and bailing on.
    skip_prone = {int(lib[i]) for lib in libraries for i in rng.choice(40, 4, replace=False)}

    tz = timezone(timedelta(hours=SAMPLE_TZ_OFFSET_H))
    records = []
    for e, era in enumerate(ERAS):
        lib = libraries[e]
        # Zipf-ish weights, so a few artists and tracks dominate each era.
        weights = 1.0 / np.arange(1, len(lib) + 1) ** 0.8
        weights /= weights.sum()
        prev_lib = libraries[e - 1] if e else None
        day = datetime.fromisoformat(era["start"]).replace(tzinfo=tz)
        end = datetime.fromisoformat(era["end"]).replace(tzinfo=tz)
        span = (end - day).days or 1
        while day <= end:
            progress = (day - datetime.fromisoformat(era["start"]).replace(tzinfo=tz)).days / span
            for start in _session_starts(day, rng):
                t = start
                for _ in range(int(rng.integers(3, 12))):
                    # Early in an era the old library still turns up.
                    if prev_lib is not None and rng.random() < 0.35 * (1 - progress) ** 2:
                        row = int(prev_lib[rng.integers(40)])
                    else:
                        row = int(lib[rng.choice(len(lib), p=weights)])
                    dur_ms = int(duration.iat[row] * 1000)
                    skip_p = 0.7 if row in skip_prone else 0.12
                    if rng.random() < skip_p:
                        reason_end, ms = "fwdbtn", int(dur_ms * rng.uniform(0.03, 0.4))
                    elif rng.random() < 0.08:
                        reason_end, ms = "endplay", int(dur_ms * rng.uniform(0.1, 0.9))
                    else:
                        reason_end, ms = "trackdone", dur_ms
                    t += timedelta(milliseconds=ms)
                    records.append({
                        "ts": t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "platform": "android" if start.hour < 17 else "windows",
                        "ms_played": ms,
                        "conn_country": "ZZ",
                        "master_metadata_track_name": cat["name"].iat[row],
                        "master_metadata_album_artist_name": cat["artist"].iat[row],
                        "master_metadata_album_album_name": cat["album"].iat[row],
                        "spotify_track_uri": f"spotify:track:{cat['track_id'].iat[row]}",
                        "episode_name": None,
                        "episode_show_name": None,
                        "spotify_episode_uri": None,
                        "reason_start": "trackdone" if records and records[-1]["reason_end"] == "trackdone" else "clickrow",
                        "reason_end": reason_end,
                        "shuffle": bool(rng.random() < 0.4),
                        "skipped": reason_end == "fwdbtn",
                        "offline": False,
                    })
            day += timedelta(days=1)

    records.sort(key=lambda r: r["ts"])
    by_year: dict[str, list[dict]] = {}
    for r in records:
        by_year.setdefault(r["ts"][:4], []).append(r)

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for year, rows in sorted(by_year.items()):
            zf.writestr(f"Spotify Extended Streaming History/Streaming_History_Audio_{year}.json", json.dumps(rows))
        zf.writestr(
            "Spotify Extended Streaming History/README.txt",
            "Synthetic sample export for the Selector demo. The listener is invented: tracks are real "
            "catalog entries, but every timestamp, skip and play count was generated.\n",
        )
    SAMPLE_PATH.write_bytes(buf.getvalue())
    print(f"sample: {len(records):,} synthetic plays, {SAMPLE_PATH.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    cat, tags = build_catalog()
    write_catalog(cat, tags)
    write_circuit(cat, tags)
    build_sample(cat, tags)


if __name__ == "__main__":
    main()
