"""Selector MCP server: exposes the local taste warehouse to any MCP client.

Run with `uv run selector-mcp` (stdio transport). Point Claude Desktop at it
per `docs/INSTALL_MCP.md`.

Every tool below wraps one query from `selector.warehouse.queries` and
formats the result as a compact markdown table, since that reads far better
in a chat transcript than a raw JSON dump. The warehouse path comes from the
`SELECTOR_DB` env var, defaulting to `./data/selector.duckdb`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from selector.fly import pipeline as fly_pipeline
from selector.fly.lsh import hamming_top_k
from selector.spotify.auth import SpotifyAuthError
from selector.spotify.client import SpotifyAPIError, SpotifyClient
from selector.spotify.reconcile import reconcile_library as _reconcile_library
from selector.warehouse import queries, wrapped

# Loaded here, at import time, since Claude Desktop launches this process
# directly and never sources a shell profile — SPOTIFY_CLIENT_ID etc. would
# otherwise never reach os.environ.
load_dotenv()

DB_ENV_VAR = "SELECTOR_DB"
DEFAULT_DB_PATH = Path("data/selector.duckdb")
MAX_ROWS = 40

server = MCPServer(
    name="selector",
    instructions=(
        "Selector exposes a personal Spotify listening-history warehouse. "
        "Call `warehouse_summary` first to orient yourself on the date range "
        "and scale of the data before running more specific queries."
    ),
)


def _db_path() -> Path:
    return Path(os.environ.get(DB_ENV_VAR, str(DEFAULT_DB_PATH)))


def _missing_db_message(db_path: Path) -> str:
    return (
        f"No warehouse found at `{db_path}`. It hasn't been built yet — run:\n\n"
        "```\n"
        "uv run python -m selector.ingest.load_history\n"
        "uv run python -m selector.warehouse.build\n"
        "```\n\n"
        "then retry this call."
    )


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        rounded = round(value, 3)
        if rounded == int(rounded):
            return f"{int(rounded):,}"
        return f"{rounded:,.3f}".rstrip("0").rstrip(".")
    if hasattr(value, "isoformat"):
        return str(value)[:19]
    return str(value)


def _df_to_markdown(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> str:
    if df.empty:
        return "_No rows matched._"
    truncated = len(df) > max_rows
    shown = df.head(max_rows)
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(_format_cell(v) for v in row) + " |")
    table = "\n".join(lines)
    if truncated:
        table += f"\n\n… {len(df) - max_rows} more rows"
    return table


def _run(fn: Callable[..., pd.DataFrame], **kwargs: Any) -> str:
    """Call a warehouse query function against the configured db, as markdown.

    Missing warehouse and query errors both come back as a plain message
    instead of a stack trace, since the caller is a model that should be
    able to recover (or tell the user what to run) rather than fail loudly.
    """
    db_path = _db_path()
    if not db_path.exists():
        return _missing_db_message(db_path)
    try:
        df = fn(db_path=db_path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: surface any query
        # failure to the calling model as text instead of letting it become a
        # stack trace on the MCP transport.
        return f"Query failed: {exc}"
    return _df_to_markdown(df)


@server.tool()
def warehouse_summary() -> str:
    """Get the shape of the listening-history warehouse: date range, total
    plays, unique tracks and artists, and total hours listened.

    Call this first when starting a conversation about this person's
    listening history, before running any more specific query, so you know
    the scale and time span of the data you're working with.
    """
    return _run(queries.warehouse_summary)


@server.tool()
def search_library(query: str, limit: int = 20) -> str:
    """Search the listening history for tracks or artists whose name
    contains `query` (case-insensitive substring match), ranked by play
    count. Use this to find a `track_id` before calling `track_detail`, or
    to check whether an artist appears in this person's history at all.
    """
    return _run(queries.search_library, query=query, limit=limit)


@server.tool()
def track_detail(track_id_or_name: str) -> str:
    """Get full stats for one track: play count, skip rate, mean completion,
    net verdict (reward minus punishment across all plays), first and last
    played. Accepts either an exact `track_id` (from `search_library`) or a
    track name substring, in which case the highest-play-count match is
    returned.
    """
    return _run(queries.track_detail, track_id_or_name=track_id_or_name)


@server.tool()
def top_artists(start: str | None = None, end: str | None = None, limit: int = 20) -> str:
    """List the most-played artists by play count and total hours, over an
    optional date range. `start` and `end` are ISO date strings like
    "2024-01-01"; omit either for an open-ended range. Use this to answer
    "who did I listen to most" for a period, or with no range at all for
    all-time favourites.
    """
    return _run(queries.top_artists, start=start, end=end, limit=limit)


@server.tool()
def binged_then_abandoned(min_plays: int = 15, window_months: int = 3) -> str:
    """Find artists with a sharp one-month listening spike (at least
    `min_plays` plays in a single calendar month) followed by near-silence
    for the next `window_months` months. This answers questions shaped like
    "what did I binge and then abandon" or "what phases did my taste go
    through that didn't last".
    """
    return _run(queries.binged_then_abandoned, min_plays=min_plays, window_months=window_months)


@server.tool()
def skip_offenders(min_plays: int = 5) -> str:
    """Find tracks that get played often but also skipped often — songs this
    person keeps queuing up and then bailing on. `min_plays` filters out
    tracks with too few plays to have a meaningful skip rate.
    """
    return _run(queries.skip_offenders, min_plays=min_plays)


@server.tool()
def listening_clock() -> str:
    """Get play counts broken down by hour of day and day of week (UTC), to
    answer questions about when listening actually happens — late-night
    habits, weekday-vs-weekend patterns, commute-time spikes.
    """
    return _run(queries.listening_clock)


@server.tool()
def taste_drift(granularity: str = "quarter", top_n: int = 5) -> str:
    """Get the top artists per time period, to show how taste evolved.
    `granularity` is "month", "quarter", or "year". Use a coarser
    granularity (quarter or year) for a broad multi-year overview, and
    "month" only for a short, recent window — otherwise the table gets long.
    """
    return _run(queries.taste_drift, granularity=granularity, top_n=top_n)


@server.tool()
def rediscovery_candidates(dormant_months: int = 6, min_past_plays: int = 10) -> str:
    """Find tracks that were played often in the past (at least
    `min_past_plays` times) but haven't been played in at least
    `dormant_months` months — songs this person used to love and forgot
    about. Good for "what should I revisit" style questions.
    """
    return _run(
        queries.rediscovery_candidates,
        dormant_months=dormant_months,
        min_past_plays=min_past_plays,
    )


@server.tool()
def wrapped_report(top_n: int = 5, save_html: bool = False) -> str:
    """Get a Spotify-Wrapped-style report over the full listening history:
    total hours, top artists/tracks/albums, an artist "sprint" (monthly
    play counts for every artist who ever cracked a top spot), peak
    listening hour, and skip offenders. `top_n` controls how many entries
    each ranked card keeps. This is the warehouse-only slice of the report —
    it doesn't need audio features or the fly brain, so it always reflects
    the full library. Set `save_html` to also write a standalone HTML story
    to `data/wrapped_report.html` alongside the JSON at
    `data/wrapped_report.json`.
    """
    db_path = _db_path()
    if not db_path.exists():
        return _missing_db_message(db_path)
    try:
        report = wrapped.build_report(top_n=top_n, db_path=db_path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text, not a crash
        return f"Report failed: {exc}"

    out_dir = db_path.parent
    (out_dir / "wrapped_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    note = f"\n_Report JSON saved to `{out_dir / 'wrapped_report.json'}`._"
    if save_html:
        html_path = out_dir / "wrapped_report.html"
        html_path.write_text(wrapped.render_html(report), encoding="utf-8")
        note += f" _HTML story saved to `{html_path}`._"

    return wrapped.render_markdown(report) + note


# -- live Spotify API tools ---------------------------------------------
#
# Everything below calls the live Spotify Web API rather than the local
# warehouse. Only endpoints that survived the November 2024 deprecation are
# used — no audio-features, audio-analysis, recommendations, related-artists,
# or preview URLs.

_spotify_client: SpotifyClient | None = None


def _track_row(track: dict) -> dict:
    artists = track.get("artists") or []
    album = track.get("album") or {}
    return {
        "name": track.get("name", ""),
        "artist": artists[0]["name"] if artists else "",
        "album": album.get("name", ""),
        "uri": track.get("uri", ""),
    }


def _artist_row(artist: dict) -> dict:
    return {
        "name": artist.get("name", ""),
        "genres": ", ".join((artist.get("genres") or [])[:3]),
        "popularity": artist.get("popularity"),
        "uri": artist.get("uri", ""),
    }


def _run_spotify(fn: Callable[[SpotifyClient], str]) -> str:
    """Call `fn` with a live Spotify client, keeping auth/API failures as
    plain text for the model rather than exceptions — same rationale as
    `_run` for the warehouse queries.
    """
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    if not client_id:
        return (
            "SPOTIFY_CLIENT_ID is not set. Register a Spotify developer app "
            "and add it to `.env` — see `docs/OAUTH_NOTES.md` for the exact "
            "steps and the redirect URI to register."
        )

    global _spotify_client
    if _spotify_client is None or _spotify_client.client_id != client_id:
        _spotify_client = SpotifyClient(client_id=client_id)

    try:
        return fn(_spotify_client)
    except SpotifyAuthError as exc:
        return f"Spotify authorization failed: {exc}"
    except SpotifyAPIError as exc:
        return f"Spotify API error: {exc}"
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text, not a crash
        return f"Unexpected error calling Spotify: {exc}"


@server.tool()
def spotify_search(query: str, item_type: str = "track", limit: int = 10) -> str:
    """Search Spotify's full catalogue (not just this person's library) for
    tracks, artists, albums, or playlists. `item_type` is one of "track",
    "artist", "album", "playlist", or a comma-separated combination like
    "track,artist". The `uri` field in the results is what
    `spotify_create_playlist` expects for `track_uris`.
    """

    def _call(client: SpotifyClient) -> str:
        data = client.search(query, types=item_type, limit=limit)
        sections = []
        for key, row_fn in (
            ("tracks", _track_row),
            ("artists", _artist_row),
        ):
            items = (data.get(key) or {}).get("items") or []
            if items:
                rows = [row_fn(item) for item in items if item]
                sections.append(f"**{key}**\n\n{_df_to_markdown(pd.DataFrame(rows))}")
        for key in ("albums", "playlists"):
            items = (data.get(key) or {}).get("items") or []
            if items:
                rows = [{"name": i.get("name", ""), "uri": i.get("uri", "")} for i in items if i]
                sections.append(f"**{key}**\n\n{_df_to_markdown(pd.DataFrame(rows))}")
        return "\n\n".join(sections) if sections else "_No results._"

    return _run_spotify(_call)


@server.tool()
def spotify_saved_tracks(limit: int = 20) -> str:
    """List this person's saved ("liked") tracks from their live Spotify
    library, most recently saved first. `limit` is capped at 50 (one API
    page); for a full reconciliation against listening history, use
    `reconcile_library` instead.
    """

    def _call(client: SpotifyClient) -> str:
        data = client.saved_tracks(limit=limit)
        rows = [_track_row(item["track"]) for item in data.get("items", []) if item.get("track")]
        return _df_to_markdown(pd.DataFrame(rows))

    return _run_spotify(_call)


@server.tool()
def spotify_top_artists(time_range: str = "medium_term", limit: int = 20) -> str:
    """List this person's top artists by Spotify's own listening algorithm —
    distinct from the historical warehouse's `top_artists`, which counts raw
    plays from the export. `time_range` is "short_term" (~4 weeks),
    "medium_term" (~6 months), or "long_term" (years).
    """

    def _call(client: SpotifyClient) -> str:
        data = client.top_items("artists", time_range=time_range, limit=limit)
        rows = [_artist_row(a) for a in data.get("items", [])]
        return _df_to_markdown(pd.DataFrame(rows))

    return _run_spotify(_call)


@server.tool()
def spotify_top_tracks(time_range: str = "medium_term", limit: int = 20) -> str:
    """List this person's top tracks by Spotify's own listening algorithm —
    distinct from the historical warehouse's play-count-based queries.
    `time_range` is "short_term" (~4 weeks), "medium_term" (~6 months), or
    "long_term" (years).
    """

    def _call(client: SpotifyClient) -> str:
        data = client.top_items("tracks", time_range=time_range, limit=limit)
        rows = [_track_row(t) for t in data.get("items", [])]
        return _df_to_markdown(pd.DataFrame(rows))

    return _run_spotify(_call)


@server.tool()
def spotify_recently_played(limit: int = 20) -> str:
    """List the most recently played tracks straight from Spotify's live
    playback history. This only covers roughly the last 50 plays — for
    anything further back, use the historical warehouse tools instead
    (`top_artists`, `taste_drift`, etc.), which cover the full export.
    """

    def _call(client: SpotifyClient) -> str:
        data = client.recently_played(limit=limit)
        rows = []
        for item in data.get("items", []):
            row = _track_row(item.get("track") or {})
            row["played_at"] = item.get("played_at", "")
            rows.append(row)
        return _df_to_markdown(pd.DataFrame(rows))

    return _run_spotify(_call)


@server.tool()
def spotify_create_playlist(
    name: str,
    description: str = "",
    track_uris: list[str] | None = None,
    public: bool = False,
) -> str:
    """Create a new playlist in this person's Spotify account, optionally
    pre-filled with `track_uris` (values like "spotify:track:...", from
    `spotify_search` results or elsewhere). This performs a real, immediate
    write to the user's account with no dry-run mode — only call it once the
    user has clearly asked for a playlist to be created, not speculatively.
    """

    def _call(client: SpotifyClient) -> str:
        playlist = client.create_playlist(
            name, description=description, public=public, track_uris=track_uris
        )
        n = len(track_uris) if track_uris else 0
        url = (playlist.get("external_urls") or {}).get("spotify", "(no url returned)")
        return (
            f"Created playlist **{playlist.get('name', name)}** "
            f"with {n} track{'s' if n != 1 else ''}. Open it: {url}"
        )

    return _run_spotify(_call)


@server.tool()
def reconcile_library() -> str:
    """Compare this person's live Spotify saved-tracks library against the
    historical listening warehouse. Reports three things: tracks saved but
    apparently never played, tracks played heavily but never saved, and
    tracks that were saved and played a lot but haven't been touched
    recently. Good for "what have I saved that I never actually listen to"
    or "what should I clean out of my library" questions.
    """

    def _call(client: SpotifyClient) -> str:
        db_path = _db_path()
        if not db_path.exists():
            return _missing_db_message(db_path)
        result = _reconcile_library(client, db_path=db_path)
        sections = [
            ("Saved but never played", result.saved_never_played),
            ("Played heavily but never saved", result.played_never_saved),
            ("Saved, then abandoned", result.saved_then_abandoned),
        ]
        return "\n\n".join(f"**{title}**\n\n{_df_to_markdown(df)}" for title, df in sections)

    return _run_spotify(_call)


# -- fly brain tools ------------------------------------------------------
#
# Both tools below read `data/fly_tags.npz`, the fingerprints
# `selector.fly.pipeline` (Step 12) computes over the fine-tuned vibe
# tagger's output plus measured audio features, wired through the real
# FlyWire connectome. Loaded and cached lazily on first use, since fitting
# nothing here is free but training the production mushroom body does a
# pass over the full play history.

_fly_track_ids: list[str] | None = None
_fly_tags = None
_fly_track_index: dict[str, int] | None = None
_fly_mbon = None


def _missing_fly_tags_message(path: Path) -> str:
    return (
        f"No fly-brain fingerprints found at `{path}`. Run the Step 12 pipeline first:\n\n"
        "```\n"
        "uv run python -m selector.fly.pipeline\n"
        "```\n\n"
        "(needs `data/track_features.parquet` from Step 11 first) then retry this call."
    )


def _load_fly_state() -> tuple[list[str], Any, dict[str, int]] | None:
    """Returns `(track_ids, tags, track_id -> row index)`, loading and
    caching from disk once. `None` if the fingerprints haven't been built
    yet."""
    global _fly_track_ids, _fly_tags, _fly_track_index
    if _fly_track_ids is None:
        path = fly_pipeline.FLY_TAGS_PATH
        if not path.exists():
            return None
        _fly_track_ids, _fly_tags = fly_pipeline.load_tags(path)
        _fly_track_index = {tid: i for i, tid in enumerate(_fly_track_ids)}
    return _fly_track_ids, _fly_tags, _fly_track_index


def _get_production_mbon(track_ids: list[str], tags) -> Any:
    """Trains once, on the full chronological play history (see
    `selector.fly.pipeline.train_production_mbon`), and caches — this is a
    single pass over `data/plays.parquet`, not per-call work."""
    global _fly_mbon
    if _fly_mbon is None:
        _fly_mbon = fly_pipeline.train_production_mbon(track_ids, tags)
    return _fly_mbon


def _resolve_track(track: str) -> tuple[str, str, str] | None:
    """Resolve a `track_id` or name substring to `(track_id, name, artist)`
    via the warehouse, the same resolution `track_detail` uses. `None` if
    nothing matches."""
    db_path = _db_path()
    if not db_path.exists():
        return None
    match = queries.track_detail(track, db_path=db_path)
    if match.empty:
        return None
    row = match.iloc[0]
    return row["track_id"], row["name"], row["artist"]


@server.tool()
def more_like_this(track: str, k: int = 10) -> str:
    """Find tracks whose fly-brain fingerprint is nearest by Hamming
    distance to `track`'s — the fly-brain equivalent of Spotify's dead
    "related tracks" endpoint. `track` accepts an exact `track_id` (from
    `search_library`) or a name substring, in which case the highest-play-count
    match is used. This is content/vibe similarity (shared Kenyon-cell
    activity from the vibe tagger's features), not taste — use `fly_score`
    for whether this person is predicted to actually like a track.
    """
    state = _load_fly_state()
    if state is None:
        return _missing_fly_tags_message(fly_pipeline.FLY_TAGS_PATH)
    track_ids, tags, track_index = state

    resolved = _resolve_track(track)
    if resolved is None:
        return f'No track in the warehouse matches "{track}".'
    track_id, name, artist = resolved

    idx = track_index.get(track_id)
    if idx is None:
        return f'"{name}" by {artist} has no fly-brain fingerprint yet (not in `track_features.parquet`).'

    neighbour_idx, distances = hamming_top_k(tags[idx], tags, k=k + 1)

    rows = []
    for i, dist in zip(neighbour_idx, distances):
        if track_ids[i] == track_id:
            continue
        rows.append({"track_id": track_ids[i], "hamming_distance": int(dist)})
    rows = rows[:k]

    labels = queries.tracks_by_ids([r["track_id"] for r in rows], db_path=_db_path())
    labels = labels.set_index("track_id")
    for row in rows:
        info = labels.loc[row["track_id"]] if row["track_id"] in labels.index else None
        row["name"] = info["name"] if info is not None else "(unknown)"
        row["artist"] = info["artist"] if info is not None else "(unknown)"

    df = pd.DataFrame(rows)[["name", "artist", "hamming_distance"]] if rows else pd.DataFrame()
    header = f"Nearest to **{name}** by {artist}:\n\n"
    return header + _df_to_markdown(df)


@server.tool()
def fly_score(track: str) -> str:
    """Get the fly brain's predicted taste score for `track`: net approach
    minus avoid drive from the mushroom body's KC->MBON synapses, trained
    chronologically on this person's actual skip/play-out history (see
    `selector.fly.mbon`). Positive means the fly predicts this person
    approaches this track; negative means it predicts avoidance. `track`
    accepts an exact `track_id` or a name substring.
    """
    state = _load_fly_state()
    if state is None:
        return _missing_fly_tags_message(fly_pipeline.FLY_TAGS_PATH)
    track_ids, tags, track_index = state

    resolved = _resolve_track(track)
    if resolved is None:
        return f'No track in the warehouse matches "{track}".'
    track_id, name, artist = resolved

    idx = track_index.get(track_id)
    if idx is None:
        return f'"{name}" by {artist} has no fly-brain fingerprint yet (not in `track_features.parquet`).'

    mbon = _get_production_mbon(track_ids, tags)
    score = mbon.valence(tags[idx])
    verdict = "approach" if score > 0 else "avoid" if score < 0 else "neutral"
    return f"**{name}** by {artist}: fly valence = {score:.3f} ({verdict})."


# -- DJ agent -------------------------------------------------------------
#
# The crate (measured audio + predicted labels + fly tags + a freshly
# trained production mushroom body) takes a few seconds to build, so it's
# built once per server process and reused across `dj_set` calls.

_dj_crate = None


@server.tool()
def dj_set(
    theme: str | None = None,
    minutes: int = 45,
    dry_run: bool = True,
    familiar_ratio: float = 0.6,
) -> str:
    """Plan a themed DJ set from this person's own library and, only if
    `dry_run` is False, create it as a private Spotify playlist.

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
    "adrenaline") or mood words ("sad rainy day"). `familiar_ratio` is the
    share of tracks from current rotation (played in the last 90 days)
    versus rediscoveries from further back.

    `dry_run` defaults to True and never touches the account. Only pass
    `dry_run=False` once the user has explicitly asked for the playlist to
    be created; a set that fails critique twice is never written.
    """
    from selector.dj.agent import render, run_dj
    from selector.dj.pool import build_crate

    global _dj_crate
    db_path = _db_path()
    if not db_path.exists():
        return _missing_db_message(db_path)
    if not fly_pipeline.FLY_TAGS_PATH.exists():
        return _missing_fly_tags_message(fly_pipeline.FLY_TAGS_PATH)

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    global _spotify_client
    if client_id and (_spotify_client is None or _spotify_client.client_id != client_id):
        _spotify_client = SpotifyClient(client_id=client_id)
    client = _spotify_client if client_id else None

    try:
        if _dj_crate is None:
            _dj_crate = build_crate(db_path=db_path)
        run = run_dj(
            _dj_crate,
            theme=theme,
            minutes=minutes,
            dry_run=dry_run,
            familiar_ratio=familiar_ratio,
            client=client,
            db_path=db_path,
        )
    except ValueError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text, not a crash
        return f"DJ run failed: {exc}"
    return render(run)


def main() -> None:
    server.run(transport='stdio')


if __name__ == "__main__":
    main()
