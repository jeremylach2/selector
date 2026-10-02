"""Where the hosted MCP server gets its data: a private Vercel Blob store.

The deploy warehouse is personal data, so it's gitignored, and a Git-built
deployment can't bundle it. Instead the code deploys from Git, and the data
lives in a private Blob store connected only to the MCP project:

- `upload` (run locally) puts `data/selector_deploy.duckdb` there, after
  re-checking that it's the coarse copy.
- `upload-crate` (run locally) puts the DJ's deploy crate,
  `data/dj_crate_deploy.parquet`, there too, after re-checking its columns
  (`selector.dj.crate`).
- `fetch` downloads either into `/tmp`: the warehouse on a cold start
  (`api/index.py`), the crate on the first hosted `dj_set` call.

Refreshing the data needs no deploy, and deploying code needs no data.
Auth is the store's read-write token, `BLOB_READ_WRITE_TOKEN`, which
Vercel sets when the store is connected to the project. This speaks the
Blob HTTP API directly (what `@vercel/blob`'s `get` and `put` send), so the
function needs nothing beyond `httpx`.

Usage, from the repo root (the directory linked to the MCP project):

    vercel env pull .env.local --environment=production   # brings BLOB_READ_WRITE_TOKEN
    uv run python -m selector.warehouse.build --deploy
    uv run python -m selector.mcp.deploy_data upload
    uv run python -m selector.dj.pool --deploy
    uv run python -m selector.mcp.deploy_data upload-crate
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import quote

import duckdb
import httpx
from dotenv import load_dotenv

from selector.warehouse.build import DEFAULT_DEPLOY_DB_PATH, deploy_violations

TOKEN_ENV_VAR = "BLOB_READ_WRITE_TOKEN"
BLOB_PATHNAME = "mcp/selector_deploy.duckdb"
CRATE_BLOB_PATHNAME = "mcp/dj_crate_deploy.parquet"
BLOB_API_URL = "https://vercel.com/api/blob/"
# The version `@vercel/blob` sends. The API rejects requests without one.
BLOB_API_VERSION = "12"


def _token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR)
    if not token:
        raise RuntimeError(f"{TOKEN_ENV_VAR} is not set; connect the Blob store to the project")
    return token


def _store_id(token: str) -> str:
    # vercel_blob_rw_<storeId>_<secret>
    parts = token.split("_")
    if len(parts) < 5 or not parts[3]:
        raise RuntimeError(f"{TOKEN_ENV_VAR} doesn't look like a Blob read-write token")
    return parts[3]


def blob_url(token: str, pathname: str = BLOB_PATHNAME) -> str:
    return f"https://{_store_id(token)}.private.blob.vercel-storage.com/{pathname}"


def fetch(dest: Path, pathname: str = BLOB_PATHNAME) -> Path:
    """Download the deploy warehouse to `dest`, unless a previous call in
    this instance already did. Written to a temp name and renamed, so a
    half-finished download is never opened."""
    if dest.exists():
        return dest
    token = _token()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream(
        "GET", blob_url(token, pathname), headers={"authorization": f"Bearer {token}"}, timeout=30.0
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    partial.replace(dest)
    return dest


def upload(path: Path = DEFAULT_DEPLOY_DB_PATH, pathname: str = BLOB_PATHNAME) -> None:
    """Put the deploy warehouse in the store, overwriting the previous one.
    Refuses anything that isn't the coarse copy, so the full warehouse can't
    be uploaded by mistake."""
    with duckdb.connect(str(path), read_only=True) as con:
        problems = deploy_violations(con)
    if problems:
        raise RuntimeError(f"{path} is not a deploy warehouse: {problems}")
    _put(path, pathname)


def upload_crate(path: Path | None = None, pathname: str = CRATE_BLOB_PATHNAME) -> None:
    """Put the DJ's deploy crate in the store, overwriting the previous one.
    Refuses a file with any column the hosted DJ doesn't read, or any
    date/time column."""
    import pyarrow.parquet as pq

    from selector.dj.crate import DEFAULT_DEPLOY_CRATE_PATH, deploy_crate_violations

    path = path or DEFAULT_DEPLOY_CRATE_PATH
    problems = deploy_crate_violations(pq.read_schema(path))
    if problems:
        raise RuntimeError(f"{path} is not a deploy crate: {problems}")
    _put(path, pathname)


def _put(path: Path, pathname: str) -> None:
    token = _token()
    response = httpx.put(
        f"{BLOB_API_URL}?pathname={quote(pathname)}",
        content=path.read_bytes(),
        headers={
            "authorization": f"Bearer {token}",
            "x-api-version": BLOB_API_VERSION,
            "x-vercel-blob-store-id": _store_id(token),
            "x-vercel-blob-access": "private",
            "x-add-random-suffix": "0",
            "x-allow-overwrite": "1",
            "x-content-type": "application/octet-stream",
        },
        timeout=120.0,
    )
    response.raise_for_status()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["upload", "upload-crate"])
    parser.add_argument("--path", type=Path, help="defaults to the deploy warehouse or the deploy crate")
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    args = parser.parse_args(argv)
    load_dotenv(args.env_file)
    if args.command == "upload-crate":
        upload_crate(args.path)
        print(f"uploaded {args.path or 'the deploy crate'} to the Blob store as {CRATE_BLOB_PATHNAME}")
    else:
        path = args.path or DEFAULT_DEPLOY_DB_PATH
        upload(path)
        print(f"uploaded {path} to the Blob store as {BLOB_PATHNAME}")


if __name__ == "__main__":
    main()
