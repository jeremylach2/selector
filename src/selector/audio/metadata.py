"""Resolve album-level release years for the tracks `selector.audio.resolve`
already matched to an iTunes or Deezer preview.

This is stage 2 ("Resolve metadata") of the Wrapped extension plan — see
`Selector - Project Extension Plan.md` — and it's what the `listening_age`
and decade-histogram cards (v1.5) need. Both iTunes and Deezer return the
parent album's release date on the *track* lookup itself, so this makes one
call per unique `(artist, album)` among the matched tracks rather than one
per track: about 1,800 albums instead of ~3,200 matched tracks.

Checkpointed the same way `resolve.py` is. The iTunes Search API has
previously started 403-ing partway through a several-thousand-call run (see
docs/AUDIO_MATCHING.md's "operational finding"); this pass makes far fewer
calls but still uses a longer delay for iTunes specifically, and any lookup
failure (rate limit, missing track, network error) just leaves that album
without a release year rather than aborting the run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import pandas as pd

from selector.audio.resolve import DEFAULT_OUTPUT_PATH as MATCHES_PATH
from selector.warehouse.build import DEFAULT_DB_PATH
from selector.warehouse.queries import _connect, normalize_album_key

DEFAULT_CACHE_PATH = Path("data/.album_release_years_checkpoint.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/album_release_years.parquet")

ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"
DEEZER_TRACK_URL = "https://api.deezer.com/track"

USER_AGENT = "Selector/0.1 (portfolio project, keyless public API usage)"

# iTunes has previously started 403-ing partway through a several-thousand
# call run at the resolve.py default of 0.2s (see docs/AUDIO_MATCHING.md);
# this pass makes far fewer iTunes calls but still leans slower for it.
ITUNES_DELAY_SECONDS = 0.3
DEEZER_DELAY_SECONDS = 0.15

BATCH_SIZE = 50


@dataclass(frozen=True)
class AlbumRelease:
    album_key: str
    artist: str
    album: str
    release_date: str | None
    release_year: int | None
    source: str | None  # "itunes" | "deezer" | None if the lookup failed


def _itunes_release(client: httpx.Client, source_id: str) -> tuple[str | None, str | None]:
    """Returns (album_name, release_date) from an iTunes track lookup, or
    (None, None) on any failure (HTTP error, no result, rate limit)."""
    try:
        resp = client.get(ITUNES_LOOKUP_URL, params={"id": source_id})
        resp.raise_for_status()
    except httpx.HTTPError:
        return None, None
    results = resp.json().get("results", [])
    if not results:
        return None, None
    result = results[0]
    return result.get("collectionName"), result.get("releaseDate")


def _deezer_release(client: httpx.Client, source_id: str) -> tuple[str | None, str | None]:
    """Returns (album_name, release_date) from a Deezer track lookup, or
    (None, None) on any failure."""
    try:
        resp = client.get(f"{DEEZER_TRACK_URL}/{source_id}")
        resp.raise_for_status()
    except httpx.HTTPError:
        return None, None
    album = resp.json().get("album") or {}
    return album.get("title"), album.get("release_date")


def _fetch_release(
    client: httpx.Client, source: str, source_id: str
) -> tuple[str | None, str | None]:
    if source == "itunes":
        time.sleep(ITUNES_DELAY_SECONDS)
        return _itunes_release(client, source_id)
    if source == "deezer":
        time.sleep(DEEZER_DELAY_SECONDS)
        return _deezer_release(client, source_id)
    return None, None


def _load_checkpoint(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    done: dict[str, dict] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done[row["album_key"]] = row
    return done


def _append_checkpoint(cache_path: Path, rows: list[AlbumRelease]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def representative_albums(matches_path: Path, db_path: Path) -> pd.DataFrame:
    """One row per unique `(artist, album)` among matched tracks: the
    normalised album key, display artist/album, and the source/source_id of
    one representative matched track to resolve a release date through.
    """
    matches = pd.read_parquet(matches_path)
    matched = matches[matches["local_path"].notna() & matches["source_id"].notna()]

    with _connect(db_path) as con:
        tracks = con.execute("SELECT track_id, artist, album FROM tracks").df()

    merged = matched.merge(tracks, on="track_id", how="left")
    merged = merged[merged["album"].notna() & (merged["album"] != "")]
    if merged.empty:
        return pd.DataFrame(columns=["album_key", "artist", "album", "match_source", "source_id"])

    merged = merged.copy()
    merged["album_key"] = [
        normalize_album_key(artist, album) for artist, album in zip(merged["artist"], merged["album"])
    ]
    reps = (
        merged.sort_values(["album_key", "track_id"])
        .groupby("album_key", as_index=False)
        .first()
    )
    return reps[["album_key", "artist", "album", "match_source", "source_id"]]


def resolve_album_release_years(
    matches_path: Path = MATCHES_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    cache_path: Path = DEFAULT_CACHE_PATH,
    limit: int | None = None,
) -> pd.DataFrame:
    """Fetch album-level release dates for every unique album among the
    already-matched tracks. Idempotent: albums already in the cache are
    skipped on re-run, same as `resolve.py`.
    """
    reps = representative_albums(matches_path, db_path)
    if limit is not None:
        reps = reps.head(limit)

    done = _load_checkpoint(cache_path)
    to_fetch = reps[~reps["album_key"].isin(done.keys())]
    print(f"{len(done)} albums already cached, {len(to_fetch)} to go")

    batch: list[AlbumRelease] = []
    with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
        for _, row in to_fetch.iterrows():
            album_name, release_date = _fetch_release(client, row["match_source"], row["source_id"])
            release_year = int(release_date[:4]) if release_date else None
            result = AlbumRelease(
                album_key=row["album_key"],
                artist=row["artist"],
                album=album_name or row["album"],
                release_date=release_date[:10] if release_date else None,
                release_year=release_year,
                source=row["match_source"] if release_date else None,
            )
            batch.append(result)

            if len(batch) >= BATCH_SIZE:
                _append_checkpoint(cache_path, batch)
                print(f"  checkpointed {len(batch)} albums ({row['album']!r} just done)")
                batch = []

    if batch:
        _append_checkpoint(cache_path, batch)

    all_rows = list(_load_checkpoint(cache_path).values())
    return pd.DataFrame(all_rows)


def coverage_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "No albums to report on."
    found = df["release_year"].notna()
    lines = [f"Release year found for {found.mean():.1%} ({found.sum()}/{len(df)}) of albums"]
    for source, count in df.loc[found, "source"].value_counts().items():
        lines.append(f"  {source}: {count}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-path", type=Path, default=MATCHES_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the number of unique albums processed"
    )
    args = parser.parse_args(argv)

    df = resolve_album_release_years(
        matches_path=args.matches_path,
        db_path=args.db_path,
        cache_path=args.cache_path,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print()
    print(coverage_report(df))


if __name__ == "__main__":
    main()
