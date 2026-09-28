# Deploying the Selector MCP server to Vercel

`docs/INSTALL_MCP.md` covers the stdio server for local Claude Code/Desktop
use. This covers a second, independent entrypoint — `api/index.py` — that
exposes the same warehouse tools over Streamable HTTP so the server is
reachable from anywhere, not just this machine.

## Scope

Only the read-only warehouse tools (`warehouse_summary`, `search_library`,
`top_artists`, `binged_then_abandoned`, `skip_offenders`, `listening_clock`,
`taste_drift`, `rediscovery_candidates`, `track_detail`) are exposed here.
`DEPLOY_TOOLS` in `http_server.py` is the list, and
`tests/test_deploy_warehouse.py` fails if it grows. Everything else stays
local-only:

- **Live Spotify tools** (`spotify_*`, `reconcile_library`). Their OAuth
  flow opens a local browser and catches the redirect on `127.0.0.1:8899`
  (see `src/selector/spotify/auth.py`), which has no equivalent on a
  stateless serverless function, and `spotify_create_playlist` writes to
  the account. Keeping Web API results off the remote server also keeps the
  Spotify-content boundary in `docs/PRIVACY.md` simple.
- **Fly-brain and DJ tools** (`more_like_this`, `fly_score`, `dj_set`).
  They need `data/fly_tags.npz` and train the mushroom body on the per-play
  history, neither of which is shipped.
- **`wrapped_report`**, which writes files next to the warehouse.

Before this list existed, the HTTP app served the stdio server's full tool
set, so the Spotify tools were listed remotely even though they couldn't
authenticate.

## What data is served

The server reads `data/selector_deploy.duckdb`, a coarse copy of the local
warehouse built by `uv run python -m selector.warehouse.build --deploy`,
never the full `data/selector.duckdb`. It isn't in git or in the
deployment. It lives in a private Blob store connected only to this
project, and a cold start downloads it into `/tmp` (see
[Keeping the warehouse current](#keeping-the-warehouse-current)). It holds
no per-play timestamps:

| Table | What it keeps |
| --- | --- |
| `plays_hourly` | Plays, listening time and forward skips per (UTC day, UTC hour, day of week, track). Replaces `plays`. |
| `tracks`, `artists` | Unchanged aggregates, with `first_played`/`last_played` cut to the UTC day |
| `artist_months` | Unchanged, `month` stored as a date |

Exact timestamps, session order, platform, connection country and the
start/end reason codes are gone, and `sessions` is dropped. The build
refuses to write the file if any time or timestamp column survives. The
warehouse queries read `plays_hourly` whenever `plays` is absent, and the
copy is built in UTC, the zone Vercel runs in, so every deployed tool gives
the answer it gave on the full warehouse. The exceptions are
`warehouse_summary`'s date range and `top_artists`' `start`/`end`, which
are now whole days.

It is still personal: the hour buckets barely merge anything (about 41k rows
for 44k plays), so it says roughly what was played in which hour. The
bearer token is what keeps it private.

## Why no Redis (yet)

Two things made the "hosting the data" problem smaller than it looked:

- The tools above only ever read the deploy warehouse, which is **about
  4MB**, not the 1.2GB `data/` directory as a whole (that's dominated by
  raw audio and the flywire model, neither of which the MCP server
  touches). One download per cold start from a same-platform Blob store is
  cheap, and the file then sits in `/tmp` for as long as the instance
  lives.
- The server runs in `stateless_http=True` mode — every request is
  independent, with no server-side session to persist across the cold
  starts a serverless function is subject to. Redis becomes relevant again
  the moment this needs to remember something between requests (an
  authenticated Spotify token being the obvious case).

## What was added

- `src/selector/mcp/warehouse_tools.py` — the nine read-only warehouse tools
  and their formatting helpers, importing nothing outside
  `requirements.txt`. `server.py` and `http_server.py` both register these
  same functions, so tool definitions never drift between the stdio and
  HTTP entrypoints. `tests/test_deploy_imports.py` walks the import graph
  from `api/index.py` and fails if anything it can reach needs a package
  `requirements.txt` doesn't install. That's how a deploy once crashed on
  every request with `No module named 'scipy'`.
- `src/selector/mcp/http_server.py` — registers those tools (and never
  imports `server.py`) on its own `deploy_server`,
  exposes it as an ASGI `app` via
  `deploy_server.streamable_http_app(stateless_http=True)`, and wraps it in
  a bearer-token auth check.
- `api/index.py` — the Vercel Python function entrypoint; puts `src/` on
  `sys.path`, downloads the deploy warehouse to `/tmp` on cold start,
  points `SELECTOR_DB` at it, and re-exports that `app`. If the download
  fails the server still starts, and every tool answers that the warehouse
  is missing.
- `src/selector/mcp/deploy_data.py` — the Blob side: `fetch` (used by the
  function) and `upload` (run locally), speaking the Blob HTTP API with
  `httpx` so the function needs no Node package. `upload` refuses any file
  that has a `plays` or `sessions` table or a time column, so the full
  warehouse can't be pushed by mistake.
- `requirements.txt` (repo root) — a deliberately minimal dependency list
  for this one function. `pyproject.toml`'s full dependency set includes
  librosa/torch/scikit-learn/matplotlib for local ingestion and tagging
  work the deployed tools never touch; pulling those into the Vercel bundle
  would cost bundle size and cold-start time for nothing.
- `vercel.json` — routes every path to `api/index.py`, since the MCP
  Starlette app handles its own internal routing (`/mcp`). Its
  `ignoreCommand` decides which Git pushes build (see
  [Deploying](#deploying)).
- `.vercelignore` — controls what a CLI deploy uploads, separately from
  `.gitignore`. It's an allowlist: everything is ignored except `api/`,
  `src/`, `requirements.txt` and `vercel.json`, and no data at all. It used
  to be a blocklist, which let `web/.env.local`, `data/wrapped_private/`
  and `data/wrapped_report.json` into the upload. The server never served
  them, but they had no business being on Vercel.

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

Rotate the token with `uv run python scripts/rotate_mcp_token.py` from the
repo root. It generates the new value and hands it to `vercel env add` on
stdin, so it never appears on screen, in shell history or in a chat
transcript. It saves the value to `~/.selector/mcp_token` and copies it to
the clipboard. The running deployment keeps the old token until the next
production deploy. After that, old clients get 401 until they're given the
new value.

The SDK's built-in DNS-rebinding protection is explicitly disabled
(`enable_dns_rebinding_protection=False` in `http_server.py`) — it checks
the `Host` header against an allowlist meant for a server bound to
`localhost`, and would reject every real request against a Vercel domain.
That protection defends against a browser being tricked into hitting a
*local* MCP server from a malicious page; it doesn't apply here, where the
bearer token is the actual access control.

## Deploying

Code deploys from Git. The project is connected to the GitHub repo with
the repo root as its root directory, and `vercel.json`'s `ignoreCommand`
builds only a push to `main` that changes a path the function ships
(`api`, `src`, `requirements.txt`, `vercel.json`, `.vercelignore`). Every
other push is skipped, including every non-`main` branch, so there are no
preview deployments. `tests/test_deploy_data.py` fails if `.vercelignore`
ever ships a path the trigger doesn't watch.

One-time setup, from the repo root:

```
vercel link                                   # links the MCP project
vercel git connect                            # connects it to the GitHub repo
vercel blob create-store selector-mcp-data --access private   # connect it to this project only
uv run python scripts/rotate_mcp_token.py     # sets SELECTOR_MCP_TOKEN, never printed
vercel env pull .env.local --environment=production   # brings BLOB_READ_WRITE_TOKEN (gitignored)
```

The Blob store is separate from the web project's, so neither project's
token can read the other's private data. `vercel deploy --prod` still
works for a one-off deploy and produces the same thing, since no data is
bundled either way.

Then point any Streamable-HTTP-capable MCP client at
`https://<your-deployment>.vercel.app/mcp` with the header
`Authorization: Bearer <token>`.

## Keeping the warehouse current

Data and code are separate. To refresh the data, with no deploy:

```
uv run python -m selector.warehouse.build            # after a new export
uv run python -m selector.warehouse.build --deploy   # the coarse copy
uv run python -m selector.mcp.deploy_data upload     # overwrites the blob
```

Running instances keep the copy they downloaded until they're recycled.
A new cold start picks up the new one.
