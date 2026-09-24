"""Vercel Python function entrypoint. Resolves to a single serverless
function (per `vercel.json`'s catch-all rewrite) exposing the Selector MCP
server over Streamable HTTP at /mcp. See `docs/DEPLOY_MCP.md`.

`src/` isn't installed as a package in the Vercel build (only
`requirements.txt` is pip-installed), so it's added to `sys.path` directly
before importing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selector.mcp.http_server import app  # noqa: E402

__all__ = ["app"]
