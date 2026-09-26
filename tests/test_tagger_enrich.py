from __future__ import annotations

import httpx
import pytest
import respx

from selector.tagger import enrich as enrich_module
from selector.tagger.enrich import LyricsFetchError, fetch_lyrics, lyrics_status


@respx.mock
def test_fetch_lyrics_returns_plain_lyrics(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    monkeypatch.setattr(enrich_module.time, "sleep", lambda seconds: None)
    respx.get("https://lrclib.net/api/search").mock(
        return_value=httpx.Response(200, json=[{"plainLyrics": "la la la"}])
    )

    with httpx.Client() as client:
        lyrics = fetch_lyrics(client, "track1", "Some Song", "Some Artist")

    assert lyrics == "la la la"
    assert (tmp_path / "track1.txt").read_text(encoding="utf-8") == "la la la"


@respx.mock
def test_fetch_lyrics_returns_none_when_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    monkeypatch.setattr(enrich_module.time, "sleep", lambda seconds: None)
    respx.get("https://lrclib.net/api/search").mock(return_value=httpx.Response(200, json=[]))

    with httpx.Client() as client:
        lyrics = fetch_lyrics(client, "track2", "Unknown Song", "Unknown Artist")

    assert lyrics is None
    assert (tmp_path / "track2.txt").read_text(encoding="utf-8") == ""


@respx.mock
def test_fetch_lyrics_uses_cache_without_a_network_call(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    (tmp_path / "track3.txt").write_text("cached lyrics", encoding="utf-8")
    # No route registered - respx raises if fetch_lyrics tries a real call,
    # so a cache hit is the only way this test can pass.

    with httpx.Client() as client:
        lyrics = fetch_lyrics(client, "track3", "irrelevant", "irrelevant")

    assert lyrics == "cached lyrics"


@respx.mock
def test_fetch_lyrics_does_not_cache_a_failed_request(tmp_path, monkeypatch):
    # An outage must not be remembered as "this song has no lyrics".
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    monkeypatch.setattr(enrich_module.time, "sleep", lambda seconds: None)
    route = respx.get("https://lrclib.net/api/search").mock(return_value=httpx.Response(503))

    with httpx.Client() as client, pytest.raises(LyricsFetchError):
        fetch_lyrics(client, "track4", "Some Song", "Some Artist")

    assert route.call_count == enrich_module.MAX_ATTEMPTS
    assert not (tmp_path / "track4.txt").exists()
    assert lyrics_status("track4") is None


@respx.mock
def test_fetch_lyrics_retries_a_transient_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    monkeypatch.setattr(enrich_module.time, "sleep", lambda seconds: None)
    respx.get("https://lrclib.net/api/search").mock(
        side_effect=[httpx.ConnectTimeout("timeout"), httpx.Response(200, json=[{"plainLyrics": "hey"}])]
    )

    with httpx.Client() as client:
        assert fetch_lyrics(client, "track5", "Some Song", "Some Artist") == "hey"
    assert lyrics_status("track5") == "lyrics"


@respx.mock
def test_fetch_lyrics_records_lrclib_instrumental_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    monkeypatch.setattr(enrich_module.time, "sleep", lambda seconds: None)
    respx.get("https://lrclib.net/api/search").mock(
        return_value=httpx.Response(200, json=[{"instrumental": True, "plainLyrics": None}])
    )

    with httpx.Client() as client:
        assert fetch_lyrics(client, "track6", "Some Song", "Some Artist") is None
    assert lyrics_status("track6") == "instrumental"


@respx.mock
def test_empty_cache_is_unknown_not_instrumental(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    (tmp_path / "track7.txt").write_text("", encoding="utf-8")
    assert lyrics_status("track7") == "unknown"


@respx.mock
def test_refresh_requeries_an_empty_cache_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_module, "LYRICS_DIR", tmp_path)
    monkeypatch.setattr(enrich_module.time, "sleep", lambda seconds: None)
    (tmp_path / "track8.txt").write_text("", encoding="utf-8")
    respx.get("https://lrclib.net/api/search").mock(
        return_value=httpx.Response(200, json=[{"plainLyrics": "found it"}])
    )

    with httpx.Client() as client:
        assert fetch_lyrics(client, "track8", "Some Song", "Some Artist") is None
        assert fetch_lyrics(client, "track8", "Some Song", "Some Artist", refresh=True) == "found it"
