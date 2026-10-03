"""The hosted `dj_set`: the DJ agent on the Vercel deployment.

It runs the same five stages as the local tool (`selector.mcp.server.dj_set`),
Brief, Arc, Select, Critique and Commit, on the deploy crate instead of a
crate built from the local pipeline. The deploy crate (`selector.dj.crate`)
carries the fly brain's output as data: each track's Kenyon-cell tag for
coherence and its mushroom-body valence for taste. So the hosted set is
picked with the fly brain, and nothing here imports scipy or trains
anything. Given the same crate, recent plays and clock, both tools pick the
same set.

What differs from the local tool:

- **Data.** The crate is a private Blob (`mcp/dj_crate_deploy.parquet`),
  downloaded to `/tmp` by the first `dj_set` call on an instance, so cold
  starts for every other tool stay as they were.
- **Recent plays.** From the hosted Spotify login, falling back to the
  newest hourly buckets of the deploy warehouse.
- **Clock.** Vercel runs in UTC, and the Brief picks themes by local hour.
  `SELECTOR_TIMEZONE` (an IANA name like `America/Chicago`) sets the local
  time. Without it the run plans in UTC and says so.
- **The write.** `guarded_create_playlist`, as for `spotify_create_playlist`:
  the daily cap, rollback, description tag and audit log, on top of the DJ's
  own two gates (`dry_run` off, critique passed).
- **No run log on disk.** The tool's answer carries the critique chain and
  the liner notes.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import duckdb
import httpx
import pandas as pd

from selector.mcp import deploy_data, spotify_tools
from selector.mcp.warehouse_tools import _db_path
from selector.spotify.auth import SpotifyAuthError
from selector.spotify.client import SpotifyAPIError, SpotifyClient
from selector.spotify.remote_store import RedisError, RemoteStoreNotConfigured

CRATE_ENV_VAR = "SELECTOR_DJ_CRATE"
TIMEZONE_ENV_VAR = "SELECTOR_TIMEZONE"
# Bounds the work per call: Vercel's function limit is 60s (vercel.json),
# and a 180-minute set takes a few seconds locally.
MIN_MINUTES = 10
MAX_MINUTES = 180
RECENT_LIMIT = 50

# The DJ modules (and pyarrow, via the crate) are imported on the first
# `dj_set` call, not at cold start, so the other tools don't pay for them.
_crate_cache: tuple[Path, float, object] | None = None


def _crate_path() -> Path:
    from selector.dj.crate import DEFAULT_DEPLOY_CRATE_PATH

    return Path(os.environ.get(CRATE_ENV_VAR, str(DEFAULT_DEPLOY_CRATE_PATH)))


def _missing_crate_message(path: Path, why: str = "") -> str:
    return (
        f"No DJ crate at `{path}`{f' ({why})' if why else ''}. Build it locally and "
        "upload it to the Blob store:\n\n"
        "```\n"
        "uv run python -m selector.dj.pool --deploy\n"
        "uv run python -m selector.mcp.deploy_data upload-crate\n"
        "```"
    )


def _ensure_crate(path: Path) -> str | None:
    """Download the crate on first use if a Blob token is set. Returns a
    message for the model if there's still no crate."""
    if path.exists():
        return None
    if not os.environ.get(deploy_data.TOKEN_ENV_VAR):
        return _missing_crate_message(path)
    try:
        deploy_data.fetch(path, deploy_data.CRATE_BLOB_PATHNAME)
    except Exception as exc:  # noqa: BLE001 - shown to the model, not a crash
        return _missing_crate_message(path, f"download failed: {exc}")
    return None


def _load_crate(path: Path):
    from selector.dj.crate import load_deploy_crate

    global _crate_cache
    mtime = path.stat().st_mtime
    if _crate_cache is None or _crate_cache[0] != path or _crate_cache[1] != mtime:
        _crate_cache = (path, mtime, load_deploy_crate(path))
    return _crate_cache[2]


def _now() -> tuple[datetime, str | None]:
    """Local time for the Brief, and a note when it had to fall back to UTC."""
    name = os.environ.get(TIMEZONE_ENV_VAR, "").strip()
    if not name:
        return datetime.now(UTC), (
            f"Planned in UTC: set `{TIMEZONE_ENV_VAR}` on the deployment (e.g. "
            "`America/Chicago`) so the brief reads your local hour."
        )
    try:
        return datetime.now(ZoneInfo(name)), None
    except (ZoneInfoNotFoundError, ValueError):
        return datetime.now(UTC), f"`{TIMEZONE_ENV_VAR}={name}` isn't a known time zone, so this was planned in UTC."


def _live_recent(client: SpotifyClient) -> pd.DataFrame:
    items = client.recently_played(limit=RECENT_LIMIT).get("items", [])
    rows = [
        {
            "track_id": (it.get("track") or {}).get("id"),
            "artist_name": ((it.get("track") or {}).get("artists") or [{}])[0].get("name"),
        }
        for it in items
    ]
    return pd.DataFrame(rows, columns=["track_id", "artist_name"]).dropna(subset=["track_id"])


def _warehouse_recent(db_path: Path) -> pd.DataFrame:
    """The newest hourly buckets of the deploy warehouse, newest first. It
    has no per-play order within an hour, which the Brief doesn't need."""
    empty = pd.DataFrame(columns=["track_id", "artist_name"])
    if not db_path.exists():
        return empty
    try:
        with duckdb.connect(str(db_path), read_only=True) as con:
            return con.execute(
                """
                SELECT track_id, artist AS artist_name FROM plays_hourly
                WHERE track_id IS NOT NULL
                ORDER BY date DESC, hour_utc DESC LIMIT ?
                """,
                [RECENT_LIMIT],
            ).df()
    except duckdb.Error:
        return empty


def _recent(client: SpotifyClient | None) -> tuple[pd.DataFrame, str]:
    source = "the deploy warehouse"
    if client is not None:
        try:
            live = _live_recent(client)
            if not live.empty:
                return live, "live Spotify"
        except (SpotifyAuthError, SpotifyAPIError, RedisError, httpx.HTTPError) as exc:
            source = f"the deploy warehouse (live Spotify failed: {exc})"
    return _warehouse_recent(_db_path()), source


def dj_set(
    theme: str | None = None,
    minutes: int = 45,
    dry_run: bool = True,
    familiar_ratio: float = 0.6,
    seed_tracks: list[str] | None = None,
    exclude_tracks: list[str] | None = None,
    exclude_artists: list[str] | None = None,
) -> str:
    """Plan a themed DJ set from this person's own library and, only if
    `dry_run` is False, create it as a Spotify playlist kept off the
    profile. The Web API can't make a playlist private by link, so the app
    still shows it as Public until the user picks "Make private" there.

    Runs five explicit stages: Brief (reads recent plays and the clock,
    picks a theme), Arc (an opener/build/peak/comedown energy curve over
    `minutes`, a hard constraint on *measured* audio energy), Select (fills
    the arc using fly-brain similarity for coherence and mushroom-body
    valence for taste, max two tracks per artist), Critique (rejects sets
    with off-arc tracks, jarring tempo/energy transitions, or no real peak,
    and sends them back to Select once), and Commit (liner notes citing
    measured tempo and energy per transition; the playlist write).

    `theme` is optional: omit it to let the brief choose from the time of
    day and recent listening, or pass a named theme ("night drive", "peak
    time", "slow sunrise", "focus drift", "golden hour", "after hours",
    "adrenaline") or mood words ("sad rainy day"). If the user describes an
    occasion instead ("background music for a talk"), pick the named theme
    that fits it rather than leaving it to the clock. `minutes` is 10 to
    180. `familiar_ratio` is the share of tracks from current rotation
    (played in the last 90 days) versus rediscoveries from further back.

    To steer it: `seed_tracks` (`track_id`s or `spotify:track:` URIs) pull
    the set's sound towards those tracks in place of recent listening; they
    anchor similarity and aren't forced into the set. `exclude_tracks` and
    `exclude_artists` (exact names, any case) are never played. To keep a
    list of your own picks and only order them, use `order_tracks`.

    `dry_run` defaults to True and never touches the account. Only pass
    `dry_run=False` once the user has explicitly asked for the playlist to
    be created; a set that fails critique twice is never written. The same
    inputs give the same set, so a dry run is an honest preview. Created
    playlists count towards the hosted server's 20 playlists per day.
    """
    from selector.dj.agent import render, run_dj

    if not MIN_MINUTES <= minutes <= MAX_MINUTES:
        return f"Refused: `minutes` must be between {MIN_MINUTES} and {MAX_MINUTES} (got {minutes})."

    path = _crate_path()
    problem = _ensure_crate(path)
    if problem:
        return problem

    try:
        client: SpotifyClient | None = spotify_tools.remote_client()
    except (spotify_tools.SpotifyNotConfigured, RemoteStoreNotConfigured) as exc:
        if not dry_run:
            return f"Can't create the playlist: {exc}"
        client = None

    try:
        crate = _load_crate(path)
        now, clock_note = _now()
        recent, source = _recent(client)
        writer = None if dry_run else partial(spotify_tools.guarded_create_playlist, client)
        run = run_dj(
            crate,
            theme=theme,
            minutes=minutes,
            dry_run=dry_run,
            familiar_ratio=familiar_ratio,
            now=now,
            recent=recent,
            log_dir=None,
            writer=writer,
            seed_track_ids=seed_tracks,
            exclude_track_ids=exclude_tracks,
            exclude_artists=exclude_artists,
        )
    except ValueError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text, not a crash
        return f"DJ run failed: {exc}"

    context = (
        f"_Recent plays from {source}. Planned for {now:%A %H:%M} ({now.tzname()}). "
        f"Crate of {len(crate.tracks):,} tracks, as of {crate.as_of:%Y-%m-%d}._"
    )
    if clock_note:
        context += f"\n\n_{clock_note}_"
    return f"{context}\n\n{render(run)}"
