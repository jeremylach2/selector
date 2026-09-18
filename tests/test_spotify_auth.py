"""Unit tests for the PKCE auth flow. No network calls and no browser opens —
every Spotify-facing function is monkeypatched at the module boundary.
"""

from __future__ import annotations

import base64
import hashlib
import time

from selector.spotify import auth


def test_generate_pkce_pair_shape():
    verifier, challenge = auth._generate_pkce_pair()

    assert 43 <= len(verifier) <= 128
    assert "=" not in verifier  # unpadded base64url

    expected_digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected_challenge = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
    assert challenge == expected_challenge


def test_generate_pkce_pair_is_random():
    v1, _ = auth._generate_pkce_pair()
    v2, _ = auth._generate_pkce_pair()
    assert v1 != v2


def test_token_set_is_expired_true_within_safety_margin():
    token = auth.TokenSet(
        access_token="a", refresh_token="r", expires_at=time.time() + 10, scope=""
    )
    assert token.is_expired() is True  # inside the 60s safety margin


def test_token_set_is_expired_false_well_in_future():
    token = auth.TokenSet(
        access_token="a", refresh_token="r", expires_at=time.time() + 3600, scope=""
    )
    assert token.is_expired() is False


def test_save_and_load_token_round_trip(tmp_path):
    token_path = tmp_path / "token.json"
    token = auth.TokenSet(
        access_token="access-123", refresh_token="refresh-456", expires_at=1234.5, scope="a b"
    )

    auth._save_token(token_path, token)
    assert token_path.exists()

    loaded = auth._load_cached_token(token_path)
    assert loaded == token


def test_load_cached_token_missing_file(tmp_path):
    assert auth._load_cached_token(tmp_path / "nope.json") is None


def test_load_cached_token_corrupt_file(tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text("not json")
    assert auth._load_cached_token(token_path) is None


def test_get_valid_token_returns_cached_without_network(tmp_path, monkeypatch):
    token_path = tmp_path / "token.json"
    fresh = auth.TokenSet(
        access_token="cached", refresh_token="r", expires_at=time.time() + 3600, scope=""
    )
    auth._save_token(token_path, fresh)

    def _boom(*args, **kwargs):
        raise AssertionError("should not hit the network for a fresh cached token")

    monkeypatch.setattr(auth, "_run_authorize_flow", _boom)
    monkeypatch.setattr(auth, "_refresh_token", _boom)

    result = auth.get_valid_token("client-id", token_path)
    assert result.access_token == "cached"


def test_get_valid_token_refreshes_expired(tmp_path, monkeypatch):
    token_path = tmp_path / "token.json"
    expired = auth.TokenSet(
        access_token="old", refresh_token="refresh-tok", expires_at=time.time() - 10, scope=""
    )
    auth._save_token(token_path, expired)

    refreshed = auth.TokenSet(
        access_token="new", refresh_token="refresh-tok", expires_at=time.time() + 3600, scope=""
    )

    def _fake_refresh(client_id, refresh_token):
        assert refresh_token == "refresh-tok"
        return refreshed

    def _boom(*args, **kwargs):
        raise AssertionError("should not run the full browser flow when refresh succeeds")

    monkeypatch.setattr(auth, "_refresh_token", _fake_refresh)
    monkeypatch.setattr(auth, "_run_authorize_flow", _boom)

    result = auth.get_valid_token("client-id", token_path)
    assert result.access_token == "new"
    assert auth._load_cached_token(token_path).access_token == "new"


def test_get_valid_token_falls_back_to_full_flow_when_refresh_fails(tmp_path, monkeypatch):
    token_path = tmp_path / "token.json"
    expired = auth.TokenSet(
        access_token="old", refresh_token="dead", expires_at=time.time() - 10, scope=""
    )
    auth._save_token(token_path, expired)

    def _fail_refresh(client_id, refresh_token):
        raise auth.SpotifyAuthError("refresh token revoked")

    def _fake_authorize(client_id):
        return {"code": "auth-code", "verifier": "verifier"}

    def _fake_exchange(client_id, code, verifier):
        assert code == "auth-code"
        return auth.TokenSet(
            access_token="brand-new", refresh_token="new-refresh", expires_at=time.time() + 3600, scope=""
        )

    monkeypatch.setattr(auth, "_refresh_token", _fail_refresh)
    monkeypatch.setattr(auth, "_run_authorize_flow", _fake_authorize)
    monkeypatch.setattr(auth, "_exchange_code_for_token", _fake_exchange)

    result = auth.get_valid_token("client-id", token_path)
    assert result.access_token == "brand-new"


def test_get_valid_token_no_cache_runs_full_flow(tmp_path, monkeypatch):
    token_path = tmp_path / "token.json"  # does not exist yet

    def _fake_authorize(client_id):
        return {"code": "auth-code", "verifier": "verifier"}

    def _fake_exchange(client_id, code, verifier):
        return auth.TokenSet(
            access_token="first-token", refresh_token="r", expires_at=time.time() + 3600, scope=""
        )

    monkeypatch.setattr(auth, "_run_authorize_flow", _fake_authorize)
    monkeypatch.setattr(auth, "_exchange_code_for_token", _fake_exchange)

    result = auth.get_valid_token("client-id", token_path)
    assert result.access_token == "first-token"
    assert token_path.exists()
