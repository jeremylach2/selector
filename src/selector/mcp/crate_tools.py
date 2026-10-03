"""The hosted fly-brain tools: `more_like_this`, `fly_score` and
`order_tracks`, read from the DJ's deploy crate.

Locally these read `data/fly_tags.npz` and train the mushroom body at
startup (`selector.mcp.server`), which needs scipy and the full play
history. The deploy crate (`selector.dj.crate`) already carries what they
need: each track's Kenyon-cell tag and its trained mushroom-body valence.
So the answers match the local tools for every track in the crate. The
difference is coverage: the crate holds only tracks with measured audio
(16,700 of 19,386 in the current build), and a track outside it gets a
message saying so, not a guess.

The crate is fetched by the first call that needs it, shared with `dj_set`
(`selector.mcp.dj_tools`).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from selector.mcp.dj_tools import _crate_path, _ensure_crate, _load_crate
from selector.mcp.warehouse_tools import _db_path, _df_to_markdown
from selector.warehouse import queries

_play_count_cache: tuple[Path, float, dict[str, int]] | None = None


def tie_note(all_distances: np.ndarray, shown: list[int]) -> str:
    """Say how many tracks tie at each distance that appears more than once
    among the neighbours shown, so ties aren't read as a ranking.
    `all_distances` includes the seed itself, at distance 0."""
    notes = []
    for d in sorted(set(shown)):
        if shown.count(d) < 2:
            continue
        tied = int((all_distances == d).sum()) - (1 if d == 0 else 0)
        notes.append(f"{tied:,} tracks tie at distance {d}")
    if not notes:
        return ""
    how = (
        "Tracks at distance 0 have identical fingerprints, so they share one taste score too and are "
        "ordered by play count."
        if 0 in shown and shown.count(0) > 1
        else "They're ordered by the fly's predicted taste (`fly_score`), then play count."
    )
    return "\n\n" + "; ".join(notes) + ". The fly can't tell tied tracks apart. " + how + " Neither is similarity."


def _crate():
    """The deploy crate, or a message for the model if there isn't one."""
    path = _crate_path()
    problem = _ensure_crate(path)
    if problem:
        return None, problem
    return _load_crate(path), None


def _play_counts() -> dict[str, int]:
    global _play_count_cache
    path = _db_path()
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    if _play_count_cache is None or _play_count_cache[:2] != (path, mtime):
        with duckdb.connect(str(path), read_only=True) as con:
            counts = dict(con.execute("SELECT track_id, play_count FROM tracks").fetchall())
        _play_count_cache = (path, mtime, counts)
    return _play_count_cache[2]


def _resolve(crate, track: str) -> tuple[int | None, str, str]:
    """`(crate row, name, artist)` for `track`, an exact `track_id` or a
    name substring, with an empty name if nothing matches. Names resolve
    through the warehouse (highest play count, as the local tools do),
    falling back to the crate's own names. The row is `None` when the
    track exists but isn't in the crate."""
    track = track.strip().removeprefix("spotify:track:")
    ids = crate.tracks["track_id"]
    match = pd.DataFrame()

    db_path = _db_path()
    if db_path.exists():
        match = queries.track_detail(track, db_path=db_path)
    if match.empty:
        exact = crate.tracks[ids == track]
        named = crate.tracks[crate.tracks["name"].str.lower().str.contains(track.lower(), regex=False)]
        match = exact if not exact.empty else named.sort_values("taste", ascending=False)
    if match.empty:
        return None, "", ""

    hit = match.iloc[0]
    rows = np.flatnonzero(ids.to_numpy() == hit["track_id"])
    return (int(rows[0]) if len(rows) else None), hit["name"], hit["artist"]


def _not_in_crate(name: str, artist: str, crate) -> str:
    return (
        f'"{name}" by {artist} isn\'t in the hosted crate, which holds only the {len(crate.tracks):,} '
        "tracks with measured audio. The local server's fly tools cover the whole library."
    )


def more_like_this(track: str, k: int = 10) -> str:
    """Find tracks whose fly-brain fingerprint is nearest by Hamming
    distance to `track`'s, the fly-brain equivalent of Spotify's dead
    "related tracks" endpoint. `track` accepts an exact `track_id` (from
    `search_library`) or a name substring, in which case the highest-play-count
    match is used. This is content/vibe similarity (shared Kenyon-cell
    activity from the vibe tagger's features), not taste. Use `fly_score`
    for whether this person is predicted to actually like a track. On the
    hosted server this searches the tracks with measured audio, not the
    whole library.

    Neighbours can tie on distance. Tied tracks are ordered by the fly's
    predicted taste, then by play count, and the output says how many tie:
    tell the user that those tracks are equally similar, not ranked by
    similarity.
    """
    crate, problem = _crate()
    if problem:
        return problem
    row, name, artist = _resolve(crate, track)
    if not name:
        return f'No track matches "{track}".'
    if row is None:
        return _not_in_crate(name, artist, crate)

    tracks = crate.tracks
    tag_rows = tracks["tag_row"].to_numpy()
    distances = crate.tags.hamming(int(tag_rows[row]))[tag_rows]
    plays = _play_counts()
    play_count = tracks["track_id"].map(plays).fillna(0).to_numpy()
    # `np.lexsort` sorts by its last key first.
    order = np.lexsort((-play_count, -tracks["fly_valence"].to_numpy(), distances))
    neighbours = [i for i in order if i != row][:k]

    df = pd.DataFrame({
        "name": tracks["name"].to_numpy()[neighbours],
        "artist": tracks["artist"].to_numpy()[neighbours],
        "hamming_distance": distances[neighbours].astype(int),
    })
    shown = [int(d) for d in df["hamming_distance"]]
    return f"Nearest to **{name}** by {artist}:\n\n" + _df_to_markdown(df) + tie_note(distances, shown)


def fly_score(track: str) -> str:
    """Get the fly brain's predicted taste score for `track`: net approach
    minus avoid drive from the mushroom body's KC->MBON synapses, trained
    chronologically on this person's actual skip/play-out history. Raw
    valence is positive for nearly every track (play-outs outnumber skips
    in the history), so its rank among the crate's tracks is the part to
    go by. `track` accepts an exact `track_id` or a name substring.
    """
    crate, problem = _crate()
    if problem:
        return problem
    row, name, artist = _resolve(crate, track)
    if not name:
        return f'No track matches "{track}".'
    if row is None:
        return _not_in_crate(name, artist, crate)

    all_valence = crate.tracks["fly_valence"].to_numpy()
    valence = float(all_valence[row])
    rank = int((all_valence > valence).sum()) + 1
    verdict = "approach" if valence > 0 else "avoid" if valence < 0 else "neutral"
    return (
        f"**{name}** by {artist}: fly valence = {valence:.3f} ({verdict}), "
        f"ranked {rank:,} of the {len(all_valence):,} tracks in the crate."
    )


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

    crate, problem = _crate()
    if problem:
        return problem
    try:
        ordering = _order(crate, [track_id_of(t) for t in tracks or [] if t.strip()])
    except ValueError as exc:
        return str(exc)
    return render(ordering, _labels(ordering.unplaced))


def _labels(track_ids: list[str]) -> dict[str, tuple[str, str]]:
    db_path = _db_path()
    if not track_ids or not db_path.exists():
        return {}
    try:
        rows = queries.tracks_by_ids(track_ids, db_path=db_path)
    except duckdb.Error:
        return {}
    return {r.track_id: (r.name, r.artist) for r in rows.itertuples()}


CRATE_TOOLS = (more_like_this, fly_score, order_tracks)
