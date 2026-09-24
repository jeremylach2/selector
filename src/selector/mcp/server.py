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


def main() -> None:
    server.run(transport='stdio')


if __name__ == "__main__":
    main()
