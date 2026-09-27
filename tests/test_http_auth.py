import pytest
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

from selector.mcp.http_server import ALLOW_NO_AUTH_ENV_VAR, TOKEN_ENV_VAR, BearerAuthMiddleware


async def _inner(scope, receive, send):
    await PlainTextResponse("through")(scope, receive, send)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    monkeypatch.delenv(ALLOW_NO_AUTH_ENV_VAR, raising=False)
    return TestClient(BearerAuthMiddleware(_inner))


def test_unset_token_refuses_everything(client):
    r = client.get("/mcp", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 503


def test_explicit_opt_out_passes(client, monkeypatch):
    monkeypatch.setenv(ALLOW_NO_AUTH_ENV_VAR, "1")
    assert client.get("/mcp").text == "through"


def test_opt_out_is_ignored_when_a_token_is_set(client, monkeypatch):
    monkeypatch.setenv(ALLOW_NO_AUTH_ENV_VAR, "1")
    monkeypatch.setenv(TOKEN_ENV_VAR, "s3cret")
    assert client.get("/mcp").status_code == 401


def test_wrong_or_missing_token_is_401(client, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, "s3cret")
    assert client.get("/mcp").status_code == 401
    assert client.get("/mcp", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_right_token_goes_through(client, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, "s3cret")
    r = client.get("/mcp", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200 and r.text == "through"
