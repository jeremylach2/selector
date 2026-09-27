# Deploying the Selector MCP server to Vercel

`docs/INSTALL_MCP.md` covers the stdio server for local Claude Code/Desktop
use. This covers a second, independent entrypoint — `api/index.py` — that
exposes the same warehouse tools over Streamable HTTP so the server is
reachable from anywhere, not just this machine.

## Scope

Only the read-only warehouse tools (`warehouse_summary`, `search_library`,
`top_artists`, `binged_then_abandoned`, `skip_offenders`, `listening_clock`,
`taste_drift`, `rediscovery_candidates`, `track_detail`) are exposed here.
The live Spotify tools are deliberately left out of this deployment — their
OAuth flow opens a local browser and catches the redirect on
`127.0.0.1:8899` (see `src/selector/spotify/auth.py`), which has no
equivalent on a stateless serverless function. Wiring those up remotely
would mean replacing that flow with a real HTTP callback route and storing
the resulting token somewhere durable (e.g. the Redis instance already on
hand) instead of `~/.selector/token.json` — worth doing later, not needed
for the warehouse tools.

## Why no Redis (yet)

Two things made the "hosting the data" problem smaller than it looked:

- The tools above only ever read `data/selector.duckdb`, which is **14MB**
  — not the 1.2GB `data/` directory as a whole (that's dominated by raw
  audio and the flywire model, neither of which the MCP server touches).
  14MB fits comfortably in a Vercel Python function's bundle (500MB
  standard limit), so it's just shipped as part of the deployment rather
  than fetched from anywhere at request time.
- The server runs in `stateless_http=True` mode — every request is
  independent, with no server-side session to persist across the cold
  starts a serverless function is subject to. Redis becomes relevant again
  the moment this needs to remember something between requests (an
  authenticated Spotify token being the obvious case).

## What was added

- `src/selector/mcp/http_server.py` — imports the same `server` object
  `server.py` defines (so tool definitions never drift between the stdio
  and HTTP entrypoints), exposes it as an ASGI `app` via
  `server.streamable_http_app(stateless_http=True)`, and wraps it in a
  bearer-token auth check.
- `api/index.py` — the Vercel Python function entrypoint; puts `src/` on
  `sys.path` and re-exports that `app`.
- `requirements.txt` (repo root) — a deliberately minimal dependency list
  for this one function. `pyproject.toml`'s full dependency set includes
  librosa/torch/scikit-learn/matplotlib for local ingestion and tagging
  work the deployed tools never touch; pulling those into the Vercel bundle
  would cost bundle size and cold-start time for nothing.
- `vercel.json` — routes every path to `api/index.py`, since the MCP
  Starlette app handles its own internal routing (`/mcp`).
- `.vercelignore` — controls what `vercel deploy` actually uploads,
  separately from `.gitignore`. `.gitignore` keeps `data/selector.duckdb`
  out of git ("Personal data — never committed"); `.vercelignore`
  deliberately does *not* exclude that one file, since the deployment needs
  it, while still excluding everything else large (audio, flywire,
  essentia models, lyrics, `.venv/`, etc.).

## Auth

There's no per-user auth system here, just a single shared secret. Set
`SELECTOR_MCP_TOKEN` as a Vercel environment variable. Every request must
send `Authorization: Bearer <that value>` (compared in constant time), or the
middleware in `http_server.py` returns 401 before the request reaches the MCP
session manager.

The middleware fails closed. If `SELECTOR_MCP_TOKEN` is unset, every request
gets 503 ("Server not configured"), so a deploy that forgot the variable
serves nothing. For local testing without a token, set the explicit opt-out
`SELECTOR_MCP_ALLOW_NO_AUTH=1` (`uvicorn selector.mcp.http_server:app
--reload` with `src/` on `PYTHONPATH`). Never set it on Vercel.

Rotate the token by replacing the env var and redeploying. Old clients get
401 until they're given the new value.

The SDK's built-in DNS-rebinding protection is explicitly disabled
(`enable_dns_rebinding_protection=False` in `http_server.py`) — it checks
the `Host` header against an allowlist meant for a server bound to
`localhost`, and would reject every real request against a Vercel domain.
That protection defends against a browser being tricked into hitting a
*local* MCP server from a malicious page; it doesn't apply here, where the
bearer token is the actual access control.

## Deploying

```
vercel link            # first time only, creates/links the Vercel project
vercel env add SELECTOR_MCP_TOKEN production   # paste a long random secret
vercel deploy --prod
```

Then point any Streamable-HTTP-capable MCP client at
`https://<your-deployment>.vercel.app/mcp` with the header
`Authorization: Bearer <token>`.

## Keeping the warehouse current

The deployed `data/selector.duckdb` is a snapshot from whenever you last
ran `vercel deploy` — updating your local warehouse
(`uv run python -m selector.warehouse.build`) doesn't change what's
deployed until you redeploy. For a personal tool refreshed occasionally,
redeploying is the simplest way to pick up new listening history; if that
becomes annoying, the next step would be uploading the duckdb file to
Vercel Blob separately and having `http_server.py` fetch it into `/tmp` on
cold start, decoupling data refreshes from code deploys.
