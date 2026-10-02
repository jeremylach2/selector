"""Vercel Python function entrypoint. Resolves to a single serverless
function (per `vercel.json`'s catch-all rewrite) exposing the Selector MCP
server over Streamable HTTP at /mcp. See `docs/DEPLOY_MCP.md`.

`src/` isn't installed as a package in the Vercel build (only
`requirements.txt` is pip-installed), so it's added to `sys.path` directly
before importing.

The tools read the coarse deploy warehouse (no per-play timestamps). It's
personal data, so it isn't in git or the deployment: a cold start
downloads it from the project's private Blob store into `/tmp`
(`selector.mcp.deploy_data`). Set here rather than in `http_server.py` so
importing that module in tests or local `uvicorn` runs doesn't redirect
the stdio server's default too.

The DJ's crate comes from the same store but isn't fetched here: the
hosted `dj_set` downloads it to `SELECTOR_DJ_CRATE` on its first call
(`selector.mcp.dj_tools`), so cold starts for the other tools don't wait
on it.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selector.mcp import deploy_data

WAREHOUSE = Path("/tmp/selector_deploy.duckdb")
DJ_CRATE = Path("/tmp/selector_dj_crate.parquet")
log = logging.getLogger(__name__)

os.environ.setdefault("SELECTOR_DJ_CRATE", str(DJ_CRATE))

if "SELECTOR_DB" not in os.environ:
    os.environ["SELECTOR_DB"] = str(WAREHOUSE)
    try:
        deploy_data.fetch(WAREHOUSE)
    except Exception:
        # The server still starts; every tool answers that the warehouse is
        # missing rather than the function crashing on every request.
        log.exception("could not fetch the deploy warehouse from Blob")

from selector.mcp.http_server import app

__all__ = ["app"]
