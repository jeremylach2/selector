"""ASGI entrypoint for running the Selector MCP server over Streamable HTTP.

`server.py` defines every tool and stays stdio-only for local Claude Code /
Desktop use per `docs/INSTALL_MCP.md`. This module reuses that same `server`
object (so tool definitions never drift between the two transports) and
exposes it as a plain ASGI `app`, suitable for a Vercel Python function or
any other ASGI host. See `docs/DEPLOY_MCP.md`.

Two deliberate choices, both driven by this being a single-user deployment:

- `stateless_http=True` — each request is handled independently with no
  server-side session state, since a serverless function gets a fresh
  instance on every cold start and there's no session store (e.g. Redis)
  wired up. This is fine for the request/response tool calls exposed here.
- Auth is a single shared-secret bearer token (`SELECTOR_MCP_TOKEN`), not the
  SDK's full OAuth provider machinery, since there's exactly one legitimate
  caller (the deployer) rather than a population of distinct users.
"""

from __future__ import annotations

import os

from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from selector.mcp.server import server

TOKEN_ENV_VAR = "SELECTOR_MCP_TOKEN"


class BearerAuthMiddleware:
    """Pure-ASGI (not BaseHTTPMiddleware) so the Streamable HTTP transport's
    SSE responses pass through unbuffered.

    If `SELECTOR_MCP_TOKEN` isn't set, auth is skipped entirely — that's the
    right behavior for local `uvicorn` testing, and deploy docs make setting
    the token on Vercel a required step before the endpoint is reachable.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        expected = os.environ.get(TOKEN_ENV_VAR)
        if expected:
            headers = dict(scope.get("headers") or [])
            got = headers.get(b"authorization", b"").decode("latin-1")
            if got != f"Bearer {expected}":
                response = PlainTextResponse("Unauthorized", status_code=401)
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


# The SDK's DNS-rebinding protection matches the `Host` header against an
# allowlist meant for a server bound to localhost — it would reject every
# real request here, since Vercel's Host header is the deployment domain,
# not "localhost". That protection defends against a browser being tricked
# into hitting a *local* MCP server; it doesn't apply to a public endpoint
# gated by its own bearer token, which is the actual access control here.
_transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

app = BearerAuthMiddleware(
    server.streamable_http_app(stateless_http=True, transport_security=_transport_security)
)
