"""ASGI entrypoint for running the Selector MCP server over Streamable HTTP.

`server.py` defines every tool and stays stdio-only for local Claude Code /
Desktop use per `docs/INSTALL_MCP.md`. This module registers the read-only
warehouse tools (`warehouse_tools.WAREHOUSE_TOOLS`), the remote Spotify
tools (`spotify_tools.REMOTE_SPOTIFY_TOOLS`: five reads plus a guarded
`spotify_create_playlist`) and the hosted DJ (`dj_tools.dj_set`) on its own
`deploy_server`, and exposes it as a plain ASGI `app`, suitable for a
Vercel Python function or any other ASGI host. The warehouse and Spotify
lists come from modules `server.py` also uses, so definitions never drift
between the two transports. See `docs/DEPLOY_MCP.md`.

It deliberately never imports `server.py`, which pulls in scipy. The DJ
runs here on a precomputed crate (`selector.dj.crate`) rather than one
built by the fly pipeline. `more_like_this` and `fly_score` stay local, as
does `reconcile_library` (it needs the full warehouse) and `wrapped_report`
(it writes to disk). The Spotify tools read their token from Redis rather
than running a browser login (see `selector.spotify.remote_store`).

Auth is MCP OAuth with Spotify as the login (`selector.mcp.oauth`): only
the owner's Spotify account gets a token. While `SELECTOR_MCP_TOKEN` is
still set, that static bearer is accepted too, for clients not yet moved
over.

It fails closed. If Redis, `SELECTOR_TOKEN_KEY`,
`SELECTOR_OWNER_SPOTIFY_ID`, `SPOTIFY_CLIENT_ID` or the public URL is
missing, every request gets 503. Local testing without any of that needs
the explicit opt-out `SELECTOR_MCP_ALLOW_NO_AUTH=1`, which serves with no
auth at all.

`stateless_http=True`: each request is handled independently with no MCP
session state, since a serverless function gets a fresh instance on every
cold start. That's fine for the request/response tool calls exposed here.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from selector.mcp import oauth
from selector.mcp.dj_tools import dj_set
from selector.mcp.spotify_tools import REMOTE_SPOTIFY_TOOLS, WRITE_ANNOTATIONS, add_spotify_tools
from selector.mcp.warehouse_tools import INSTRUCTIONS, WAREHOUSE_TOOLS
from selector.spotify.remote_store import RemoteStoreNotConfigured

DEPLOY_TOOLS = (*WAREHOUSE_TOOLS, *REMOTE_SPOTIFY_TOOLS, dj_set)

# Local `uvicorn` testing only. Never set on Vercel.
ALLOW_NO_AUTH_ENV_VAR = "SELECTOR_MCP_ALLOW_NO_AUTH"

log = logging.getLogger(__name__)

# The SDK's DNS-rebinding protection matches the `Host` header against an
# allowlist meant for a server bound to localhost. It would reject every
# real request here, since Vercel's Host header is the deployment domain,
# not "localhost". That protection defends against a browser being tricked
# into hitting a *local* MCP server; it doesn't apply to a public endpoint
# gated by its own OAuth tokens, which are the actual access control here.
_transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)


def build_server(provider: oauth.SpotifyOAuthProvider | None = None) -> MCPServer:
    """The deploy tool set, with OAuth when `provider` is given."""
    kwargs: dict[str, Any] = {}
    if provider is not None:
        kwargs = {"auth_server_provider": provider, "auth": provider.config.auth_settings()}
    server = MCPServer(name="selector", instructions=INSTRUCTIONS, **kwargs)
    for tool in WAREHOUSE_TOOLS:
        server.add_tool(tool)
    add_spotify_tools(server, REMOTE_SPOTIFY_TOOLS)
    # A write whenever `dry_run` is off, so clients that honour annotations ask first.
    server.add_tool(dj_set, annotations=WRITE_ANNOTATIONS)
    if provider is not None:
        server.custom_route(oauth.CALLBACK_PATH, methods=["GET"])(provider.handle_spotify_callback)
    return server


async def _not_configured(scope: Scope, receive: Receive, send: Send) -> None:
    if scope["type"] == "http":
        await PlainTextResponse("Server not configured", status_code=503)(scope, receive, send)
    elif scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return


def create_app(provider: oauth.SpotifyOAuthProvider | None = None) -> tuple[MCPServer, ASGIApp]:
    """Build the server and its ASGI app from the environment (or the given
    provider). A missing setting gives a 503-only app, never an open one."""
    if provider is None:
        try:
            provider = oauth.SpotifyOAuthProvider.from_env()
        except RemoteStoreNotConfigured as exc:
            server = build_server()
            if os.environ.get(ALLOW_NO_AUTH_ENV_VAR) == "1":
                return server, server.streamable_http_app(
                    stateless_http=True, transport_security=_transport_security
                )
            log.error("refusing every request: %s", exc)
            return server, _not_configured
    server = build_server(provider)
    return server, server.streamable_http_app(stateless_http=True, transport_security=_transport_security)


deploy_server, app = create_app()
