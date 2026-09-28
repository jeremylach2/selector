"""Blob transfer of the deploy warehouse, and the Git-deploy trigger."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from selector.mcp import deploy_data

REPO = Path(__file__).resolve().parent.parent
FAKE_TOKEN = "vercel_blob_rw_AbC123store_s3cretpart"


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv(deploy_data.TOKEN_ENV_VAR, FAKE_TOKEN)


def test_url_points_at_the_private_store():
    assert deploy_data.blob_url(FAKE_TOKEN) == (
        "https://AbC123store.private.blob.vercel-storage.com/mcp/selector_deploy.duckdb"
    )


def test_malformed_token_is_rejected():
    with pytest.raises(RuntimeError):
        deploy_data.blob_url("not-a-blob-token")


def test_upload_refuses_the_full_warehouse(warehouses, monkeypatch):
    full, _ = warehouses
    monkeypatch.setattr(deploy_data.httpx, "put", lambda *a, **k: pytest.fail("uploaded"))
    with pytest.raises(RuntimeError, match="table plays"):
        deploy_data.upload(full)


def test_upload_sends_a_private_overwrite(warehouses, monkeypatch):
    _, deploy = warehouses
    sent = {}

    class Ok:
        def raise_for_status(self):
            pass

    def put(url, content, headers, timeout):
        sent.update(url=url, size=len(content), headers=headers)
        return Ok()

    monkeypatch.setattr(deploy_data.httpx, "put", put)
    deploy_data.upload(deploy)
    assert sent["url"].endswith("?pathname=mcp/selector_deploy.duckdb")
    assert sent["size"] == deploy.stat().st_size
    assert sent["headers"]["x-vercel-blob-access"] == "private"
    assert sent["headers"]["x-add-random-suffix"] == "0"
    assert sent["headers"]["authorization"] == f"Bearer {FAKE_TOKEN}"


def test_fetch_downloads_once(tmp_path, monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield b"duck"
            yield b"db"

    @contextmanager
    def stream(method, url, headers, timeout):
        calls.append((method, url, headers["authorization"]))
        yield Response()

    monkeypatch.setattr(deploy_data.httpx, "stream", stream)
    dest = tmp_path / "w.duckdb"
    deploy_data.fetch(dest)
    deploy_data.fetch(dest)
    assert dest.read_bytes() == b"duckdb"
    assert calls == [("GET", deploy_data.blob_url(FAKE_TOKEN), f"Bearer {FAKE_TOKEN}")]
    assert not (tmp_path / "w.duckdb.part").exists()


def test_fetch_without_a_token_fails(tmp_path, monkeypatch):
    monkeypatch.delenv(deploy_data.TOKEN_ENV_VAR)
    with pytest.raises(RuntimeError):
        deploy_data.fetch(tmp_path / "w.duckdb")


def test_git_deploys_watch_everything_the_deployment_ships():
    """A change to any allowlisted path must trigger a build on main."""
    shipped = [
        line[2:]
        for line in (REPO / ".vercelignore").read_text(encoding="utf-8").splitlines()
        if line.startswith("!/")
    ]
    # .git ships only so the ignoreCommand can diff; it isn't code.
    assert ".git" in shipped
    shipped.remove(".git")
    command = json.loads((REPO / "vercel.json").read_text(encoding="utf-8"))["ignoreCommand"]
    watched = command.split(" -- ", 1)[1].split()
    assert shipped and set(shipped) <= set(watched), (shipped, watched)
    assert '"$VERCEL_GIT_COMMIT_REF" != "main"' in command
