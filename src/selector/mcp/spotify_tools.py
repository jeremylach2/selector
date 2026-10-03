"""The live Spotify tools, shared by both MCP entrypoints.

`server.py` (stdio) registers the read tools built on `local_client`,
which caches its token in `~/.selector/token.json` and may open a browser
the first time. `http_server.py` (the Vercel deployment) registers
`REMOTE_SPOTIFY_TOOLS`, built on `remote_client`, which keeps an encrypted
token in Redis and never opens a browser (see `docs/DEPLOY_MCP.md`).

The remote `spotify_create_playlist` is a separate function rather than a
flag on the local one, so its schema can't offer what the hosted server
shouldn't do: it has no `public` parameter, and it checks its input, caps
playlists per day, deletes a playlist again if adding its tracks fails, and
logs what it created.

Like `warehouse_tools.py`, this imports nothing beyond `requirements.txt`.
Only endpoints that survived the November 2024 deprecation are used.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from selector.mcp.warehouse_tools import _db_path, _df_to_markdown
from selector.spotify import resolve as track_resolve
from selector.spotify.auth import SpotifyAuthError
from selector.spotify.client import SpotifyAPIError, SpotifyClient
from selector.spotify.remote_store import RedisError, RedisTokenStore, RemoteStoreNotConfigured

CLIENT_ID_ENV_VAR = "SPOTIFY_CLIENT_ID"
# Set to "off" on Vercel to disable every remote Spotify tool without a
# code change. Deleting the `spotify:token` Redis key has the same effect.
REMOTE_SWITCH_ENV_VAR = "SELECTOR_REMOTE_SPOTIFY"
# Vercel's function limit is 60s (vercel.json); a longer 429 wait raises.
REMOTE_MAX_RETRY_AFTER_SECONDS = 10.0

MAX_REMOTE_PLAYLISTS_PER_DAY = 20
MAX_PLAYLIST_TRACKS = 500
MAX_PLAYLIST_NAME_CHARS = 100
SPOTIFY_DESCRIPTION_LIMIT = 300
DESCRIPTION_TAG = " · made with Selector"
TRACK_URI = re.compile(r"spotify:track:[A-Za-z0-9]{22}")
PLAYLIST_COUNT_KEY = "spotify:playlists:{day}"
AUDIT_KEY = "spotify:audit"
AUDIT_KEEP = 200

READ_ANNOTATIONS = ToolAnnotations(read_only_hint=True, open_world_hint=True)
WRITE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True
)
WRITE_TOOL_NAMES = {"spotify_create_playlist"}


class SpotifyNotConfigured(RuntimeError):
    """Raised by a client factory when it can't build a client. The message
    is shown to the model as is."""


# -- clients ----------------------------------------------------------------

_local_client: SpotifyClient | None = None
_remote_client: SpotifyClient | None = None


def _client_id() -> str:
    client_id = os.environ.get(CLIENT_ID_ENV_VAR)
    if not client_id:
        raise SpotifyNotConfigured(
            "SPOTIFY_CLIENT_ID is not set. Register a Spotify developer app "
            "and add it to `.env`. See `docs/OAUTH_NOTES.md` for the exact "
            "steps and the redirect URI to register."
        )
    return client_id


def local_client() -> SpotifyClient:
    """The stdio server's client: file token cache, browser login allowed."""
    global _local_client
    client_id = _client_id()
    if _local_client is None or _local_client.client_id != client_id:
        _local_client = SpotifyClient(client_id=client_id)
    return _local_client


def remote_client() -> SpotifyClient:
    """The hosted server's client: encrypted Redis token, no browser."""
    global _remote_client
    if os.environ.get(REMOTE_SWITCH_ENV_VAR, "").strip().lower() in ("off", "0", "false"):
        raise SpotifyNotConfigured(
            f"Remote Spotify access is switched off ({REMOTE_SWITCH_ENV_VAR}). "
            "Use the local Selector server instead."
        )
    client_id = _client_id()
    if _remote_client is None or _remote_client.client_id != client_id:
        _remote_client = SpotifyClient(
            client_id=client_id,
            store=RedisTokenStore.from_env(),
            interactive=False,
            max_retry_after=REMOTE_MAX_RETRY_AFTER_SECONDS,
        )
    return _remote_client


def run_spotify(get_client: Callable[[], SpotifyClient], fn: Callable[[SpotifyClient], str]) -> str:
    """Call `fn` with a live Spotify client, keeping config, auth and API
    failures as plain text for the model rather than exceptions."""
    try:
        client = get_client()
    except (SpotifyNotConfigured, RemoteStoreNotConfigured) as exc:
        return str(exc)
    try:
        return fn(client)
    except SpotifyAuthError as exc:
        return f"Spotify authorization failed: {exc}"
    except SpotifyAPIError as exc:
        return f"Spotify API error: {exc}"
    except RedisError as exc:
        return f"Token store error: {exc}"
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text, not a crash
        return f"Unexpected error calling Spotify: {exc}"


# -- formatting -------------------------------------------------------------


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


def _playlist_url(playlist: dict) -> str:
    return (playlist.get("external_urls") or {}).get("spotify", "(no url returned)")


# -- read tools -------------------------------------------------------------


def make_read_tools(get_client: Callable[[], SpotifyClient]) -> tuple[Callable[..., str], ...]:
    """The five Spotify reads plus `resolve_tracks`, bound to one client
    factory. Each transport calls this once, so the definitions never
    drift."""

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
            for key, row_fn in (("tracks", _track_row), ("artists", _artist_row)):
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

        return run_spotify(get_client, _call)

    def spotify_saved_tracks(limit: int = 20) -> str:
        """List this person's saved ("liked") tracks from their live Spotify
        library, most recently saved first. `limit` is capped at 50 (one API
        page); for a full reconciliation against listening history, use
        `reconcile_library` instead (local server only).
        """

        def _call(client: SpotifyClient) -> str:
            data = client.saved_tracks(limit=limit)
            rows = [_track_row(item["track"]) for item in data.get("items", []) if item.get("track")]
            return _df_to_markdown(pd.DataFrame(rows))

        return run_spotify(get_client, _call)

    def spotify_top_artists(time_range: str = "medium_term", limit: int = 20) -> str:
        """List this person's top artists by Spotify's own listening algorithm,
        distinct from the historical warehouse's `top_artists`, which counts raw
        plays from the export. `time_range` is "short_term" (~4 weeks),
        "medium_term" (~6 months), or "long_term" (years).
        """

        def _call(client: SpotifyClient) -> str:
            data = client.top_items("artists", time_range=time_range, limit=limit)
            return _df_to_markdown(pd.DataFrame([_artist_row(a) for a in data.get("items", [])]))

        return run_spotify(get_client, _call)

    def spotify_top_tracks(time_range: str = "medium_term", limit: int = 20) -> str:
        """List this person's top tracks by Spotify's own listening algorithm,
        distinct from the historical warehouse's play-count-based queries.
        `time_range` is "short_term" (~4 weeks), "medium_term" (~6 months), or
        "long_term" (years).
        """

        def _call(client: SpotifyClient) -> str:
            data = client.top_items("tracks", time_range=time_range, limit=limit)
            return _df_to_markdown(pd.DataFrame([_track_row(t) for t in data.get("items", [])]))

        return run_spotify(get_client, _call)

    def spotify_recently_played(limit: int = 20) -> str:
        """List the most recently played tracks straight from Spotify's live
        playback history. This only covers roughly the last 50 plays. For
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

        return run_spotify(get_client, _call)

    return (
        spotify_search,
        spotify_saved_tracks,
        spotify_top_artists,
        spotify_top_tracks,
        spotify_recently_played,
        make_resolve_tool(get_client),
    )


# -- batch resolution -------------------------------------------------------

MAX_RESOLVE_ITEMS = 50
_library_cache: tuple[Path, float, track_resolve.Library] | None = None


def _library() -> tuple[track_resolve.Library | None, str | None]:
    """The warehouse's tracks for matching, cached until the file changes,
    or a note on why there are none."""
    global _library_cache
    path = _db_path()
    if not path.exists():
        return None, "No listening-history warehouse here, so nothing was matched from the library."
    mtime = path.stat().st_mtime
    if _library_cache is None or _library_cache[:2] != (path, mtime):
        _library_cache = (path, mtime, track_resolve.Library.load(path))
    return _library_cache[2], None


def _resolution_report(results: list[track_resolve.Resolution], notes: list[str]) -> str:
    counts = Counter(r.status for r in results)
    sources = Counter(r.source for r in results if r.status == track_resolve.MATCHED)
    order = (track_resolve.MATCHED, track_resolve.GUESS, track_resolve.MISSED, track_resolve.ERROR)
    head = ", ".join(f"{counts[s]} {s}" for s in order if counts[s])
    if sources:
        head += " (matched: " + ", ".join(f"{n} from the {src}" for src, n in sources.items()) + ")"

    rows = [
        {
            "#": i + 1,
            "query": r.query,
            "status": r.status,
            "match": f"{r.name} - {r.artist}" if r.name else "",
            "from": r.source,
            "confidence": r.confidence if r.name else None,
            "uri": r.uri,
            "note": r.note,
        }
        for i, r in enumerate(results)
    ]
    out = [f"**{head}.**", *notes, _df_to_markdown(pd.DataFrame(rows), max_rows=MAX_RESOLVE_ITEMS)]

    matched = [r.uri for r in results if r.status == track_resolve.MATCHED]
    ready = list(dict.fromkeys(matched))
    if ready:
        dupes = len(matched) - len(ready)
        out.append(
            "Matched URIs in order, ready for `spotify_create_playlist`"
            + (f" ({dupes} duplicate{'s' if dupes != 1 else ''} dropped)" if dupes else "")
            + ":\n\n" + json.dumps(ready)
        )
    if counts[track_resolve.GUESS]:
        out.append("Best guesses are left out of that list. Confirm them with the user before adding them.")
    return "\n\n".join(out)


def make_resolve_tool(get_client: Callable[[], SpotifyClient]) -> Callable[..., str]:
    def resolve_tracks(tracks: list[str]) -> str:
        """Resolve a list of songs written as "Artist - Title" to Spotify
        track URIs in one call, instead of one `spotify_search` per song.
        Each is matched against this person's listening history first (no
        API call, and the version they actually play), then Spotify's
        catalogue. Every item gets a status: "matched" (safe to use),
        "best guess" (URI given, but confirm it), "missed", or "error"
        (search failed; retry later). Only matched URIs go in the
        ready-to-use list at the end. At most 50 songs per call.
        """
        items = [t for t in (tracks or []) if t and t.strip()]
        if not items:
            return 'Refused: pass at least one song, written as "Artist - Title".'
        if len(items) > MAX_RESOLVE_ITEMS:
            return f"Refused: at most {MAX_RESOLVE_ITEMS} songs per call (got {len(items)}). Split the list."

        notes: list[str] = []
        try:
            library, why = _library()
        except Exception as exc:  # noqa: BLE001 - the catalogue can still answer
            library, why = None, f"The warehouse couldn't be read ({exc}), so nothing was matched from the library."
        if why:
            notes.append(f"_{why}_")
        try:
            client: SpotifyClient | None = get_client()
        except (SpotifyNotConfigured, RemoteStoreNotConfigured) as exc:
            client = None
            notes.append(f"_Spotify search unavailable, library matches only: {exc}_")

        results = track_resolve.resolve(
            items,
            library,
            client,
            search_errors=(SpotifyAuthError, SpotifyAPIError, RedisError, httpx.HTTPError),
        )
        return _resolution_report(results, notes)

    return resolve_tracks


# -- remote playlist creation -----------------------------------------------


def _check_playlist_request(name: str, uris: list[str]) -> str | None:
    if not name:
        return "Refused: the playlist needs a name."
    if len(name) > MAX_PLAYLIST_NAME_CHARS:
        return f"Refused: playlist names are capped at {MAX_PLAYLIST_NAME_CHARS} characters."
    if len(uris) > MAX_PLAYLIST_TRACKS:
        return f"Refused: at most {MAX_PLAYLIST_TRACKS} tracks per playlist (got {len(uris)})."
    bad = [u for u in uris if not TRACK_URI.fullmatch(u)]
    if bad:
        shown = ", ".join(f"`{u}`" for u in bad[:5])
        return (
            f"Refused: {len(bad)} value(s) aren't Spotify track URIs like "
            f"`spotify:track:<22 characters>`: {shown}. Get them from `spotify_search`."
        )
    return None


def _tagged_description(description: str) -> str:
    room = SPOTIFY_DESCRIPTION_LIMIT - len(DESCRIPTION_TAG)
    return description.strip()[:room] + DESCRIPTION_TAG


def _audit(redis, playlist: dict, name: str, n_tracks: int) -> None:
    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "id": playlist.get("id"),
        "name": name,
        "tracks": n_tracks,
    }
    try:
        redis.command("LPUSH", AUDIT_KEY, json.dumps(entry))
        redis.command("LTRIM", AUDIT_KEY, 0, AUDIT_KEEP - 1)
    except RedisError:
        pass  # the playlist exists either way; a missed log line isn't worth failing the call


class PlaylistRefused(RuntimeError):
    """A remote playlist write that was refused or undone. The message is
    shown to the model as is."""


def guarded_create_playlist(client: SpotifyClient, name: str, description: str, uris: list[str]) -> dict:
    """Every remote playlist write goes through here, `spotify_create_playlist`
    and the hosted `dj_set` alike: input checks, the daily cap, create then
    add (deleting the playlist again if the add fails), the description tag
    and the audit log. Returns the created playlist; raises `PlaylistRefused`."""
    name = name.strip()
    problem = _check_playlist_request(name, uris)
    if problem:
        raise PlaylistRefused(problem)

    redis = client.store.redis
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    key = PLAYLIST_COUNT_KEY.format(day=day)
    count = int(redis.command("INCR", key))
    if count == 1:
        redis.command("EXPIRE", key, 2 * 86_400)
    if count > MAX_REMOTE_PLAYLISTS_PER_DAY:
        raise PlaylistRefused(
            f"Refused: the hosted server has already created {MAX_REMOTE_PLAYLISTS_PER_DAY} "
            "playlists today (UTC). Try again tomorrow, or use the local Selector server."
        )

    playlist = client.create_playlist(name, description=_tagged_description(description), public=False)
    if uris:
        try:
            client.add_tracks_to_playlist(playlist["id"], uris)
        except (SpotifyAPIError, SpotifyAuthError, httpx.HTTPError) as exc:
            try:
                client.unfollow_playlist(playlist["id"])
                cleanup = "The empty playlist was deleted again."
            except (SpotifyAPIError, SpotifyAuthError, httpx.HTTPError):
                cleanup = f"Deleting the empty playlist also failed; remove it by hand: {_playlist_url(playlist)}"
            raise PlaylistRefused(f"Created the playlist but adding its tracks failed ({exc}). {cleanup}") from exc

    _audit(redis, playlist, name, len(uris))
    return playlist


def spotify_create_playlist(
    name: str,
    description: str = "",
    track_uris: list[str] | None = None,
) -> str:
    """Create a new playlist in this person's Spotify account, kept off
    their profile, optionally pre-filled with `track_uris` (values like
    "spotify:track:...", from `resolve_tracks`, `spotify_search`, or a
    warehouse `track_id` as `spotify:track:<track_id>`). This performs a
    real, immediate write to the user's account with no dry-run mode, only
    call it once the user has clearly asked for a playlist to be created,
    not speculatively. At most 500 tracks per playlist and 20 playlists per
    day. The Spotify app will still list it as Public until the user picks
    "Make private" there; the Web API can't change that.
    """
    name = name.strip()
    uris = list(track_uris or [])
    # Checked here too, so a bad request never even builds a client.
    problem = _check_playlist_request(name, uris)
    if problem:
        return problem

    def _call(client: SpotifyClient) -> str:
        try:
            playlist = guarded_create_playlist(client, name, description, uris)
        except PlaylistRefused as exc:
            return str(exc)
        n = len(uris)
        return (
            f"Created playlist **{playlist.get('name', name)}** "
            f"with {n} track{'s' if n != 1 else ''}. Open it: {_playlist_url(playlist)}"
        )

    return run_spotify(_remote, _call)


def _remote() -> SpotifyClient:
    # Looked up at call time, so tests can swap `remote_client`.
    return remote_client()


def _local() -> SpotifyClient:
    return local_client()


LOCAL_READ_TOOLS = make_read_tools(_local)
REMOTE_SPOTIFY_TOOLS = (*make_read_tools(_remote), spotify_create_playlist)


def add_spotify_tools(server: MCPServer, tools: Iterable[Callable[..., str]]) -> None:
    """Register `tools` on `server`, marking the write tool as a write so
    clients ask before running it."""
    for fn in tools:
        annotations = WRITE_ANNOTATIONS if fn.__name__ in WRITE_TOOL_NAMES else READ_ANNOTATIONS
        server.add_tool(fn, annotations=annotations)
