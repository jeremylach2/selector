"""Resolve loose "Artist - Title" strings to Spotify track URIs, and say
which ones it couldn't.

Each item is tried against the listening-history warehouse first: a track
this person has played is already a known Spotify ID, so it costs no API
call and can't land on a stranger's cover. Only what the warehouse can't
match goes to Spotify search. Every item comes back with a status:

- **matched**: scored at or above `selector.matching.DEFAULT_MATCH_THRESHOLD`,
  the same reject-over-guess bar as the audio matcher,
- **best guess**: the closest result scored below that bar, or the query
  had no artist to check against. Its URI is returned, flagged, so the
  caller can confirm it rather than trust it,
- **missed**: nothing close enough,
- **error**: Spotify search failed for this item, so there's no verdict
  yet (an outage isn't a miss).

Spotify writes version info as a " - " suffix ("Let It Be - Remastered
2009"). It is folded into parentheses before scoring, so the version words
count as modifiers rather than as part of the title, and "remaster" is
ignored as a modifier: a remaster is the same recording for a playlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from selector.matching import DEFAULT_MATCH_THRESHOLD, match_score, normalize, similarity

GUESS_THRESHOLD = 0.5
SEARCH_LIMIT = 5
IGNORED_MODIFIERS = frozenset({"remaster"})
# Shortest title that may match by substring rather than exactly, so "Go"
# doesn't pull in every track with "go" in its name.
MIN_SUBSTRING_TITLE = 4

MATCHED, GUESS, MISSED, ERROR = "matched", "best guess", "missed", "error"


@dataclass
class Resolution:
    query: str
    status: str
    name: str = ""
    artist: str = ""
    source: str = ""  # "library" | "catalog"
    confidence: float = 0.0
    uri: str = ""
    note: str = ""


def parse_query(text: str) -> tuple[str, str]:
    """`"Artist - Title"` -> `(artist, title)`. Split on the first " - ",
    since titles carry version suffixes ("Song - Live") far more often than
    artist names contain the separator. No separator: `("", text)`."""
    artist, sep, title = text.strip().partition(" - ")
    if not sep:
        return "", text.strip()
    return artist.strip(), title.strip()


def _fold_suffix(title: str) -> str:
    head, sep, tail = title.partition(" - ")
    return f"{head} ({tail})" if sep else title


def score(query_title: str, query_artist: str, cand_title: str, cand_artists: list[str]) -> float:
    """Best match score over the candidate's credited artists, so a query
    naming the featured artist still matches."""
    q_title, c_title = _fold_suffix(query_title), _fold_suffix(cand_title)
    if not query_artist:
        # No artist to check: title similarity alone, which can at best be
        # a guess (see `_status`).
        return similarity(normalize(q_title)[0], normalize(c_title)[0])
    return max(
        (match_score(q_title, query_artist, c_title, a, IGNORED_MODIFIERS) for a in cand_artists or [""]),
        default=0.0,
    )


def _status(confidence: float, has_artist: bool) -> str:
    if confidence >= DEFAULT_MATCH_THRESHOLD and has_artist:
        return MATCHED
    if confidence >= GUESS_THRESHOLD:
        return GUESS
    return MISSED


# -- the warehouse ----------------------------------------------------------


@dataclass
class Library:
    """The warehouse's tracks with their titles pre-normalised, so each
    query only scores the few rows whose title could plausibly match."""

    tracks: pd.DataFrame

    @classmethod
    def load(cls, db_path: Path) -> Library:
        with duckdb.connect(str(db_path), read_only=True) as con:
            tracks = con.execute(
                "SELECT track_id, name, artist, play_count FROM tracks WHERE track_id IS NOT NULL"
            ).df()
        tracks["core"] = [normalize(_fold_suffix(n or ""))[0] for n in tracks["name"]]
        return cls(tracks)

    def best(self, artist: str, title: str) -> tuple[pd.Series, float] | None:
        core = normalize(_fold_suffix(title))[0]
        if not core:
            return None
        cores = self.tracks["core"]
        hits = cores == core
        if len(core) >= MIN_SUBSTRING_TITLE:
            hits |= cores.str.contains(core, regex=False)
            hits |= cores.map(lambda c: len(c) >= MIN_SUBSTRING_TITLE and c in core)
        candidates = self.tracks[hits]
        if candidates.empty:
            return None
        scores = [score(title, artist, n, [a]) for n, a in zip(candidates["name"], candidates["artist"])]
        top = candidates.assign(score=scores).sort_values(["score", "play_count"], ascending=False).iloc[0]
        return top, float(top["score"])


# -- resolving ---------------------------------------------------------------


def _catalog_best(client, artist: str, title: str) -> tuple[dict, float] | None:
    query = f"track:{title} artist:{artist}" if artist else title
    items = (client.search(query, types="track", limit=SEARCH_LIMIT).get("tracks") or {}).get("items") or []
    if not items and artist:
        # Field filters are strict about spelling; a plain query is not.
        items = (client.search(f"{artist} {title}", types="track", limit=SEARCH_LIMIT).get("tracks") or {}).get(
            "items"
        ) or []
    best: tuple[dict, float] | None = None
    for item in items:
        if not item:
            continue
        s = score(title, artist, item.get("name", ""), [a.get("name", "") for a in item.get("artists") or []])
        if best is None or s > best[1]:
            best = (item, s)
    return best


def resolve(
    queries: list[str],
    library: Library | None,
    client=None,
    search_errors: tuple[type[BaseException], ...] = (Exception,),
) -> list[Resolution]:
    """Resolve each of `queries` in order, library first, then (if
    `client` is given) Spotify search. `search_errors` are the exceptions a
    failed search may raise; they mark that item as an error, not a miss."""
    results: list[Resolution] = []
    for query in queries:
        artist, title = parse_query(query)
        has_artist = bool(artist)
        res = Resolution(query=query, status=MISSED)
        if not title:
            res.note = "empty query"
            results.append(res)
            continue

        local = library.best(artist, title) if library is not None else None
        if local is not None:
            row, conf = local
            res = Resolution(
                query, _status(conf, has_artist), row["name"], row["artist"], "library",
                round(conf, 2), f"spotify:track:{row['track_id']}",
            )

        if res.status != MATCHED and client is not None:
            try:
                remote = _catalog_best(client, artist, title)
            except search_errors as exc:
                if res.status == MISSED:
                    res = Resolution(query, ERROR, note=f"Spotify search failed: {exc}")
                else:
                    res.note = f"Spotify search failed, kept the library's best guess: {exc}"
                remote = None
            if remote is not None and remote[1] > res.confidence:
                item, conf = remote
                artists = item.get("artists") or []
                res = Resolution(
                    query, _status(conf, has_artist), item.get("name", ""),
                    artists[0].get("name", "") if artists else "", "catalog", round(conf, 2), item.get("uri", ""),
                )

        if not has_artist and res.status == GUESS:
            res.note = res.note or 'no artist given; write it as "Artist - Title" to confirm the match'
        if res.status == MISSED:
            res.uri = ""
        results.append(res)
    return results
