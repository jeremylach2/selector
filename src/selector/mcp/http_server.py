"""ASGI entrypoint for running the Selector MCP server over Streamable HTTP.

`server.py` defines every tool and stays stdio-only for local Claude Code /
Desktop use per `docs/INSTALL_MCP.md`. This module registers only the
read-only warehouse tools (`warehouse_tools.WAREHOUSE_TOOLS`, the same
functions `server.py` registers, so definitions never drift between the
two transports) on its own `deploy_server`, and exposes it as a plain ASGI
`app`, suitable for a Vercel Python function or any other ASGI host. See
`docs/DEPLOY_MCP.md`.

It deliberately never imports `server.py`, which pulls in scipy and the
Spotify client. The live Spotify tools need a local OAuth flow and could
write to the account, the fly-brain and DJ tools need files the
deployment doesn't ship (and the mushroom body trains on per-play
history), and `wrapped_report` writes to disk.

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

import hmac
import os

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from selector.mcp.warehouse_tools import INSTRUCTIONS, WAREHOUSE_TOOLS

DEPLOY_TOOLS = WAREHOUSE_TOOLS

deploy_server = MCPServer(name="selector", instructions=INSTRUCTIONS)
for _tool in DEPLOY_TOOLS:
    deploy_server.add_tool(_tool)

TOKEN_ENV_VAR = "SELECTOR_MCP_TOKEN"
# Local `uvicorn` testing only. Never set on Vercel: without it, a missing
# token refuses every request instead of serving the warehouse to anyone.
ALLOW_NO_AUTH_ENV_VAR = "SELECTOR_MCP_ALLOW_NO_AUTH"


class BearerAuthMiddleware:
    """Pure-ASGI (not BaseHTTPMiddleware) so the Streamable HTTP transport's
    SSE responses pass through unbuffered.

    Fails closed: if `SELECTOR_MCP_TOKEN` isn't set, every request gets 503,
    so a deployment that forgot the env var serves nothing rather than the
    whole warehouse. Local testing without a token needs the explicit opt-out
    `SELECTOR_MCP_ALLOW_NO_AUTH=1`.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        expected = os.environ.get(TOKEN_ENV_VAR)
        if not expected:
            if os.environ.get(ALLOW_NO_AUTH_ENV_VAR) != "1":
                response = PlainTextResponse("Server not configured", status_code=503)
                await response(scope, receive, send)
                return
        else:
            headers = dict(scope.get("headers") or [])
            got = headers.get(b"authorization", b"")
            if not hmac.compare_digest(got, f"Bearer {expected}".encode("latin-1")):
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
    deploy_server.streamable_http_app(stateless_http=True, transport_security=_transport_security)
)
