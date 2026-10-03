"""`resolve_tracks`: loose "Artist - Title" strings to Spotify URIs,
library first, then the catalogue, with every miss reported."""

import json

import pytest

from selector.mcp import spotify_tools
from selector.spotify import resolve
from selector.spotify.client import SpotifyAPIError

ID = "a" * 22


def _item(name, artist, uri_id=ID):
    return {"name": name, "artists": [{"name": artist}], "uri": f"spotify:track:{uri_id}"}


class FakeSearch:
    def __init__(self, items=(), fail=False):
        self.items, self.fail, self.queries = list(items), fail, []

    def search(self, query, types="track", limit=10):
        self.queries.append(query)
        if self.fail:
            raise SpotifyAPIError(503, "down")
        return {"tracks": {"items": self.items[:limit]}}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Alpha - Song", ("Alpha", "Song")),
        ("  Alpha -  Song - Live  ", ("Alpha", "Song - Live")),
        ("Just A Title", ("", "Just A Title")),
    ],
)
def test_parse_query(text, expected):
    assert resolve.parse_query(text) == expected


def test_a_remaster_is_the_same_recording_but_live_is_not():
    assert resolve.score("Let It Be", "The Beatles", "Let It Be - Remastered 2009", ["The Beatles"]) > 0.9
    assert resolve.score("Let It Be", "The Beatles", "Let It Be - Live", ["The Beatles"]) < 0.72


def test_a_featured_artist_still_matches():
    assert resolve.score("Song", "Guest", "Song", ["Lead", "Guest"]) == pytest.approx(1.0)


def test_library_match_needs_no_search(warehouses):
    library = resolve.Library.load(warehouses[0])
    client = FakeSearch()
    (r,) = resolve.resolve(["Alpha - Song t1"], library, client)
    assert (r.status, r.source, r.uri) == ("matched", "library", "spotify:track:t1")
    assert client.queries == []


def test_deploy_warehouse_works_as_the_library(warehouses):
    (r,) = resolve.resolve(["Beta - Song t2"], resolve.Library.load(warehouses[1]))
    assert r.status == "matched" and r.uri == "spotify:track:t2"


def test_catalogue_fallback_and_statuses(warehouses):
    library = resolve.Library.load(warehouses[0])
    client = FakeSearch([_item("Midnight City", "M83")])
    matched, guess, missed = resolve.resolve(
        ["M83 - Midnight City", "M83 - Midnight Citee Remix", "Nobody - Nothing At All"], library, client
    )
    assert (matched.status, matched.source) == ("matched", "catalog")
    assert guess.status == "best guess" and guess.uri
    assert missed.status == "missed" and missed.uri == ""


def test_title_only_is_never_more_than_a_guess():
    (r,) = resolve.resolve(["Midnight City"], None, FakeSearch([_item("Midnight City", "M83")]))
    assert r.status == "best guess" and "Artist - Title" in r.note


def test_a_failed_search_is_an_error_not_a_miss():
    (r,) = resolve.resolve(["M83 - Midnight City"], None, FakeSearch(fail=True), search_errors=(SpotifyAPIError,))
    assert r.status == "error" and "503" in r.note


# -- the tool ---------------------------------------------------------------------


@pytest.fixture
def tool(warehouses, monkeypatch):
    monkeypatch.setenv("SELECTOR_DB", str(warehouses[1]))
    monkeypatch.setattr(spotify_tools, "_library_cache", None)
    client = FakeSearch([_item("Midnight City", "M83")])
    return spotify_tools.make_resolve_tool(lambda: client), client


def test_tool_lists_only_matched_uris_once(tool):
    resolve_tracks, _ = tool
    out = resolve_tracks(["Alpha - Song t1", "M83 - Midnight City", "Alpha - Song t1", "M83 - Midnight Citee Remix"])
    assert out.startswith("**3 matched, 1 best guess")
    ready = json.loads(out.split("dropped):\n\n", 1)[1].split("\n", 1)[0])
    assert ready == ["spotify:track:t1", f"spotify:track:{ID}"]
    assert "Confirm them with the user" in out


def test_tool_without_spotify_still_matches_the_library(warehouses, monkeypatch):
    monkeypatch.setenv("SELECTOR_DB", str(warehouses[1]))
    monkeypatch.setattr(spotify_tools, "_library_cache", None)

    def _unconfigured():
        raise spotify_tools.SpotifyNotConfigured("SPOTIFY_CLIENT_ID is not set.")

    out = spotify_tools.make_resolve_tool(_unconfigured)(["Alpha - Song t1", "M83 - Midnight City"])
    assert "library matches only" in out and "spotify:track:t1" in out and "1 missed" in out


def test_tool_refuses_empty_and_oversized_lists(tool):
    resolve_tracks, client = tool
    assert resolve_tracks([]).startswith("Refused")
    assert resolve_tracks(["A - B"] * 51).startswith("Refused")
    assert client.queries == []
