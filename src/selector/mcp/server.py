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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from selector.fly import pipeline as fly_pipeline
from selector.fly.lsh import hamming_distances, hamming_top_k
from selector.mcp.crate_tools import tie_note
from selector.mcp.spotify_tools import (
    LOCAL_READ_TOOLS,
    READ_ANNOTATIONS,
    WRITE_ANNOTATIONS,
    SpotifyNotConfigured,
    add_spotify_tools,
    local_client,
    run_spotify,
)
from selector.mcp.warehouse_tools import (  # noqa: F401 - re-exported for callers and tests
    DB_ENV_VAR,
    DEFAULT_DB_PATH,
    INSTRUCTIONS,
    MAX_ROWS,
    WAREHOUSE_TOOLS,
    _db_path,
    _df_to_markdown,
    _missing_db_message,
    binged_then_abandoned,
    listening_clock,
    rediscovery_candidates,
    search_library,
    skip_offenders,
    taste_drift,
    top_artists,
    track_detail,
    warehouse_summary,
)
from selector.spotify.client import SpotifyClient
from selector.spotify.reconcile import reconcile_library as _reconcile_library
from selector.warehouse import queries, wrapped

# Loaded here, at import time, since Claude Desktop launches this process
# directly and never sources a shell profile, so SPOTIFY_CLIENT_ID etc. would
# otherwise never reach os.environ.
load_dotenv()

server = MCPServer(name="selector", instructions=INSTRUCTIONS)
for _tool in WAREHOUSE_TOOLS:
    server.add_tool(_tool)


@server.tool()
def wrapped_report(top_n: int = 5, save_html: bool = False, year: int | None = None) -> str:
    """Get a Spotify-Wrapped-style report: total hours, top
    artists/tracks/albums, the artist race, peak listening hour, skip
    offenders, listening age and decade mix, fly-brain taste clusters,
    hidden gems, and a listening archetype. Covers the full history by
    default, or one calendar year (UTC) if `year` is given. `top_n` controls
    how many entries each ranked card keeps. Cards whose inputs are missing
    (release years, fly-brain fingerprints) are left out rather than
    failing the report. Set `save_html` to also write a standalone HTML story
    to `data/wrapped_report.html` alongside the JSON at
    `data/wrapped_report.json`.
    """
    db_path = _db_path()
    if not db_path.exists():
        return _missing_db_message(db_path)
    window = wrapped.Window.year(year) if year is not None else wrapped.ALL_TIME
    try:
        report = wrapped.build_report(top_n=top_n, db_path=db_path, window=window)
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
# The read tools are shared with the hosted server (`spotify_tools.py`).
# This server's `spotify_create_playlist` (with a `public` flag and no
# daily cap) and `reconcile_library` (which needs the full warehouse) stay
# local-only.

add_spotify_tools(server, LOCAL_READ_TOOLS)


@server.tool(annotations=WRITE_ANNOTATIONS)
def spotify_create_playlist(
    name: str,
    description: str = "",
    track_uris: list[str] | None = None,
    public: bool = False,
) -> str:
    """Create a new playlist in this person's Spotify account, optionally
    pre-filled with `track_uris` (values like "spotify:track:...", from
    `resolve_tracks`, `spotify_search`, or a warehouse `track_id` as
    `spotify:track:<track_id>`). This performs a real, immediate
    write to the user's account with no dry-run mode, only call it once the
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

    return run_spotify(local_client, _call)


@server.tool(annotations=READ_ANNOTATIONS)
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

    return run_spotify(local_client, _call)


# -- fly brain tools ------------------------------------------------------
#
# Both tools below read `data/fly_tags.npz`, the fingerprints
# `selector.fly.pipeline` computes over the fine-tuned vibe
# tagger's output plus measured audio features, wired through the real
# FlyWire connectome. Loaded and cached lazily on first use, since fitting
# nothing here is free but training the production mushroom body does a
# pass over the full play history.

_fly_track_ids: list[str] | None = None
_fly_tags = None
_fly_track_index: dict[str, int] | None = None
_fly_mbon = None
_fly_taste = None
_fly_plays = None


def _missing_fly_tags_message(path: Path) -> str:
    return (
        f"No fly-brain fingerprints found at `{path}`. Run the fly pipeline first:\n\n"
        "```\n"
        "uv run python -m selector.fly.pipeline\n"
        "```\n\n"
        "(needs `data/track_features.parquet` from the tagger first) then retry this call."
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
    `selector.fly.pipeline.train_production_mbon`), and caches. This is a
    single pass over `data/plays.parquet`, not per-call work."""
    global _fly_mbon
    if _fly_mbon is None:
        _fly_mbon = fly_pipeline.train_production_mbon(track_ids, tags)
    return _fly_mbon


def _get_fly_taste(track_ids: list[str], tags) -> np.ndarray:
    """Every track's fly valence, one sparse product over the tag matrix.
    Used to order tracks the hash can't tell apart, never as similarity."""
    global _fly_taste
    if _fly_taste is None:
        mbon = _get_production_mbon(track_ids, tags)
        _fly_taste = tags.astype(np.float64) @ (mbon.w_approach - mbon.w_avoid)
    return _fly_taste


def _get_play_counts(track_ids: list[str]) -> np.ndarray:
    """Warehouse play count per fingerprint row. The second tie-break key:
    identical fingerprints get identical taste scores, so only something
    outside the fingerprint can order them."""
    global _fly_plays
    if _fly_plays is None:
        with queries._connect(_db_path()) as con:
            counts = dict(con.execute("SELECT track_id, play_count FROM tracks").fetchall())
        _fly_plays = np.array([counts.get(t, 0) for t in track_ids], dtype=np.float64)
    return _fly_plays


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
    distance to `track`'s, the fly-brain equivalent of Spotify's dead
    "related tracks" endpoint. `track` accepts an exact `track_id` (from
    `search_library`) or a name substring, in which case the highest-play-count
    match is used. This is content/vibe similarity (shared Kenyon-cell
    activity from the vibe tagger's features), not taste. Use `fly_score`
    for whether this person is predicted to actually like a track.

    Many tracks share an identical fingerprint (mostly tracks with no
    matched audio), so neighbours often tie. Tied tracks are ordered by the
    fly's predicted taste, then by play count (identical fingerprints share
    one taste score), and the output says how many tie: tell the user that
    those tracks are equally similar, not ranked by similarity.
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

    tie_break = [_get_fly_taste(track_ids, tags), _get_play_counts(track_ids)]
    neighbour_idx, distances = hamming_top_k(tags[idx], tags, k=k + 1, tie_break=tie_break)

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
    shown = [r["hamming_distance"] for r in rows]
    return header + _df_to_markdown(df) + tie_note(hamming_distances(tags[idx], tags), shown)


@server.tool()
def fly_score(track: str) -> str:
    """Get the fly brain's predicted taste score for `track`: net approach
    minus avoid drive from the mushroom body's KC->MBON synapses, trained
    chronologically on this person's actual skip/play-out history (see
    `selector.fly.mbon`). Positive means the fly predicts this person
    approaches this track; negative means it predicts avoidance. Raw
    valence is positive for nearly every track (play-outs outnumber skips
    in the history), so its rank among all tracks is the part to go by.
    `track` accepts an exact `track_id` or a name substring.
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
    rank = int((_get_fly_taste(track_ids, tags) > score).sum()) + 1
    return (
        f"**{name}** by {artist}: fly valence = {score:.3f} ({verdict}), "
        f"ranked {rank:,} of the {len(track_ids):,} fingerprinted tracks."
    )


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
    seed_tracks: list[str] | None = None,
    exclude_tracks: list[str] | None = None,
    exclude_artists: list[str] | None = None,
) -> str:
    """Plan a themed DJ set from this person's own library and, only if
    `dry_run` is False, create it as a Spotify playlist kept off the
    profile. The Web API can't make a playlist private by link (see
    `docs/OAUTH_NOTES.md`), so the app still shows it as Public until the
    user picks "Make private" there.

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

    To steer it: `seed_tracks` (`track_id`s or `spotify:track:` URIs) pull
    the set's sound towards those tracks in place of recent listening; they
    anchor similarity and aren't forced into the set. `exclude_tracks` and
    `exclude_artists` (exact names, any case) are never played. To keep a
    list of your own picks and only order them, use `order_tracks`.

    `dry_run` defaults to True and never touches the account. Only pass
    `dry_run=False` once the user has explicitly asked for the playlist to
    be created; a set that fails critique twice is never written.
    """
    from selector.dj.agent import render, run_dj

    db_path = _db_path()
    problem = _dj_crate_problem(db_path)
    if problem:
        return problem

    try:
        client = local_client()
    except SpotifyNotConfigured:
        client = None

    try:
        run = run_dj(
            _get_dj_crate(db_path),
            theme=theme,
            minutes=minutes,
            dry_run=dry_run,
            familiar_ratio=familiar_ratio,
            client=client,
            db_path=db_path,
            seed_track_ids=seed_tracks,
            exclude_track_ids=exclude_tracks,
            exclude_artists=exclude_artists,
        )
    except ValueError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text, not a crash
        return f"DJ run failed: {exc}"
    return render(run)


@server.tool()
def order_tracks(tracks: list[str]) -> str:
    """Put a list of tracks in DJ order, keeping every one: the quietest
    opens, energy builds to a peak and comes down, with tempo and energy
    jumps between neighbours kept small and fly-brain neighbours placed
    together. Use it to sequence a set the user or you picked (`dj_set`
    picks its own tracks). `tracks` are `track_id`s or `spotify:track:`
    URIs, 2 to 100. Energy and tempo are measured from audio, so tracks
    without measured audio are appended at the end and flagged. Read-only:
    the answer ends with the URIs in order, for `spotify_create_playlist`.
    """
    from selector.dj.agent import track_id_of
    from selector.dj.order import order_tracks as _order
    from selector.dj.order import render

    db_path = _db_path()
    problem = _dj_crate_problem(db_path)
    if problem:
        return problem
    try:
        ordering = _order(_get_dj_crate(db_path), [track_id_of(t) for t in tracks or [] if t.strip()])
    except ValueError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text, not a crash
        return f"Ordering failed: {exc}"
    labels = queries.tracks_by_ids(ordering.unplaced, db_path=db_path)
    return render(ordering, {r.track_id: (r.name, r.artist) for r in labels.itertuples()})


def _dj_crate_problem(db_path: Path) -> str | None:
    if not db_path.exists():
        return _missing_db_message(db_path)
    if not fly_pipeline.FLY_TAGS_PATH.exists():
        return _missing_fly_tags_message(fly_pipeline.FLY_TAGS_PATH)
    return None


def _get_dj_crate(db_path: Path):
    from selector.dj.pool import build_crate

    global _dj_crate
    if _dj_crate is None:
        _dj_crate = build_crate(db_path=db_path)
    return _dj_crate


def main() -> None:
    server.run(transport='stdio')


if __name__ == "__main__":
    main()
