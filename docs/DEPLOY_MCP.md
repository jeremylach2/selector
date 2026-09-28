# Deploying the Selector MCP server to Vercel

`docs/INSTALL_MCP.md` covers the stdio server for local Claude Code/Desktop
use. This covers a second, independent entrypoint — `api/index.py` — that
exposes the warehouse tools and the live Spotify tools, including playlist
creation, over Streamable HTTP so the server is reachable from anywhere,
not just this machine.

## Scope

The hosted server exposes two sets of tools. `DEPLOY_TOOLS` in
`http_server.py` is the list, and `tests/test_deploy_warehouse.py` fails if
it changes:

- **Warehouse tools**, read-only: `warehouse_summary`, `search_library`,
  `top_artists`, `binged_then_abandoned`, `skip_offenders`,
  `listening_clock`, `taste_drift`, `rediscovery_candidates`,
  `track_detail`.
- **Live Spotify tools**: `spotify_search`, `spotify_saved_tracks`,
  `spotify_top_artists`, `spotify_top_tracks`, `spotify_recently_played`,
  and a guarded `spotify_create_playlist` (see
  [Live Spotify tools](#live-spotify-tools)).

Everything else stays local-only:

- **`reconcile_library`**, which hasn't been checked against the coarse
  deploy warehouse.
- **Fly-brain and DJ tools** (`more_like_this`, `fly_score`, `dj_set`).
  They need `data/fly_tags.npz` and train the mushroom body on the per-play
  history, neither of which is shipped.
- **`wrapped_report`**, which writes files next to the warehouse.

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
OAuth login (see [Auth](#auth)) is what keeps it private.

## Why Redis

The warehouse needs no server-side state: it's about 4MB, read-only, and
downloaded once per cold start from Blob. The server runs in
`stateless_http=True` mode, so no MCP session needs persisting either.

The Spotify tools do need state that outlives an instance and can be
written: the Spotify token, which refreshes every hour and can come back
with a new refresh token. That lives in Upstash Redis (Vercel
Marketplace), connected only to this project, along with a refresh lock,
the daily playlist counter, the playlist audit log and the OAuth state
(clients, codes and tokens). Redis rather than
the existing Blob store because it has atomic `SET NX` and `INCR`.

## What was added

- `src/selector/mcp/warehouse_tools.py` — the nine read-only warehouse tools
  and their formatting helpers, importing nothing outside
  `requirements.txt`. `server.py` and `http_server.py` both register these
  same functions, so tool definitions never drift between the stdio and
  HTTP entrypoints. `tests/test_deploy_imports.py` walks the import graph
  from `api/index.py` and fails if anything it can reach needs a package
  `requirements.txt` doesn't install. That's how a deploy once crashed on
  every request with `No module named 'scipy'`.
- `src/selector/mcp/spotify_tools.py` — the live Spotify tools. The five
  read tools are built once per transport from the same definitions: the
  stdio server binds them to the local token file, the hosted server to
  the Redis store. The hosted `spotify_create_playlist` is its own
  function with the guardrails below.
- `src/selector/spotify/remote_store.py` — `RedisTokenStore`, the
  encrypted Redis token store, and a minimal Upstash REST client over
  `httpx`.
- `src/selector/mcp/http_server.py` — registers those tools (and never
  imports `server.py`) on its own `deploy_server`,
  exposes it as an ASGI `app` via
  `deploy_server.streamable_http_app(stateless_http=True)`, with the SDK's
  OAuth routes and token check wired to `oauth.py`.
- `src/selector/mcp/oauth.py` — the OAuth provider: client registration,
  the Spotify login leg and its callback, and the MCP codes and tokens in
  Redis. See [Auth](#auth).
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

MCP OAuth, with Spotify as the login (`src/selector/mcp/oauth.py`, on top
of the SDK's `mcp.server.auth` handlers). The server is its own
authorization server, and the only account that can get a token is the
owner's Spotify account, `SELECTOR_OWNER_SPOTIFY_ID`.

1. A request to `/mcp` without a token gets a 401 whose
   `WWW-Authenticate` header points at
   `/.well-known/oauth-protected-resource/mcp`, which names this server as
   its authorization server (`/.well-known/oauth-authorization-server`).
2. The client registers itself (`/register`) and sends the browser to
   `/authorize` with PKCE.
3. The server parks that request in Redis and redirects to Spotify with
   the hosted scopes. That leg is PKCE too, so there's still no Spotify
   client secret anywhere.
4. Spotify comes back to `/oauth/spotify/callback`. The server exchanges
   the code, calls `GET /me`, and refuses unless it's the owner. Someone
   else's Spotify token is dropped, never stored. The owner's replaces the
   hosted Spotify token, so logging in also repairs a dead one.
5. The client gets an authorization code (single use, 5 minutes) and
   swaps it at `/token` for an access token (1 hour) and a refresh token
   (30 days, rotated on every use). `/revoke` revokes both.

Clients, pending logins, codes and tokens live in the same Redis as the
Spotify token, under `mcp:*` keys with TTLs. Keys are SHA-256 hashes of
the tokens and every value is Fernet-encrypted, so Redis holds no usable
token.

Two guards, since `/register` is open to anyone:

- **Redirect URI allowlist.** A client can only register claude.ai's
  callback (`https://claude.ai/api/mcp/auth_callback`, and the
  `claude.com` one) or a loopback `http://localhost`/`127.0.0.1` URI
  (Claude Code, MCP Inspector). Add exact URIs with
  `SELECTOR_OAUTH_REDIRECT_URIS` (space-separated). Without this, anyone
  could register their own redirect URI and send the owner a login link:
  Spotify skips its consent page for an app already approved, so one
  click would hand them a token.
- **20 registrations per UTC day**, so nobody can fill Redis through it.

It fails closed. If Redis, `SELECTOR_TOKEN_KEY`,
`SELECTOR_OWNER_SPOTIFY_ID`, `SPOTIFY_CLIENT_ID` or the public URL
(`SELECTOR_MCP_PUBLIC_URL`, else Vercel's `VERCEL_PROJECT_PRODUCTION_URL`)
is missing, every request gets 503 ("Server not configured"). For local
testing without any of it, set the explicit opt-out
`SELECTOR_MCP_ALLOW_NO_AUTH=1` (`uvicorn selector.mcp.http_server:app
--reload` with `src/` on `PYTHONPATH`). Never set it on Vercel.

**Logging everyone out:** `uv run python scripts/revoke_mcp_sessions.py`
deletes every MCP token, code and pending login, immediately.
`--clients` also drops the registered clients, `--spotify` the hosted
Spotify token. Rotating `SELECTOR_TOKEN_KEY` makes everything in Redis
unreadable at once, and forces a fresh login.

**The old static bearer.** Until `SELECTOR_MCP_TOKEN` is removed, that
value is accepted too (compared in constant time), so clients can move
over one at a time. Once every client logs in with OAuth, run
`vercel env rm SELECTOR_MCP_TOKEN production` and redeploy.

The SDK's built-in DNS-rebinding protection is explicitly disabled
(`enable_dns_rebinding_protection=False` in `http_server.py`) — it checks
the `Host` header against an allowlist meant for a server bound to
`localhost`, and would reject every real request against a Vercel domain.
That protection defends against a browser being tricked into hitting a
*local* MCP server from a malicious page; it doesn't apply here, where the
OAuth tokens are the actual access control.

## Live Spotify tools

The hosted server has its own Spotify login, separate from the local
`~/.selector/token.json`. A PKCE refresh can hand back a new refresh token,
so if the two shared one, a refresh on either side could break the other.

- **Login.** Every OAuth login stores a fresh token set, with narrower
  scopes (`REMOTE_SCOPES` in `auth.py`: library, top items, recently
  played, `playlist-modify-private`). `scripts/seed_remote_spotify_token.py`
  runs the same login on this machine and writes the result to Redis, as
  a fallback.
- **At rest.** The token set is encrypted with Fernet under
  `SELECTOR_TOKEN_KEY` (a sensitive Vercel env var), so the Redis data
  alone is useless. `scripts/setup_remote_spotify.py` generates the key and
  sets it without printing it.
- **No browser, ever.** The hosted client runs with `interactive=False`: a
  missing token or a refresh token Spotify rejects is an error telling you
  to reconnect the client (or re-seed).
- **Refreshes.** Held under a Redis lock (`SET NX PX`), and the token is
  reloaded once the lock is held, so concurrent requests on a reused
  instance refresh once.
- **Timeouts.** A 429 with a `Retry-After` over 10 seconds raises instead
  of sleeping past the function's 60 second limit.

`spotify_create_playlist` on the hosted server differs from the local one:

- There is no `public` parameter: every playlist is kept off the profile.
  The Spotify app still lists it as Public until you pick "Make private"
  there (see `docs/OAUTH_NOTES.md`).
- Names are 1 to 100 characters, at most 500 tracks, and every URI must be
  a `spotify:track:` URI. Anything else is refused before Spotify is called.
- At most 20 playlists per UTC day, counted in Redis.
- Tracks are added after the playlist is created. If that fails, the empty
  playlist is deleted again (unfollowed, which is how the API deletes your
  own playlist).
- The description gets " · made with Selector" appended, so remote
  creations are easy to find.
- Each creation is logged to the `spotify:audit` Redis list (time, id,
  name, track count), trimmed to the last 200.
- It's annotated as a write (`readOnlyHint: false`) and the read tools as
  reads, so clients that honour annotations ask before creating.

To switch the Spotify tools off, set `SELECTOR_REMOTE_SPOTIFY=off` and
redeploy, or delete the `spotify:token` key in Redis for an immediate
stop. To revoke the login entirely, remove the app at spotify.com →
Account → Apps. That revokes the local login too.

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
vercel env pull .env.local --environment=production   # brings BLOB_READ_WRITE_TOKEN (gitignored)
```

For the Spotify tools, also:

```
vercel integration add upstash                # Redis; connect it to the MCP project only
uv run python scripts/setup_remote_spotify.py # sets SELECTOR_TOKEN_KEY (never printed) and SPOTIFY_CLIENT_ID
vercel env pull .env.local --environment=production   # brings the Redis URL and token
uv run python scripts/seed_remote_spotify_token.py    # browser login; writes the encrypted token and owner id to Redis
uv run python scripts/setup_mcp_oauth.py      # sets SELECTOR_OWNER_SPOTIFY_ID and SELECTOR_MCP_PUBLIC_URL
```

`setup_mcp_oauth.py` prints the callback URL
(`https://selector-mcp.vercel.app/oauth/spotify/callback`) to add as a
second redirect URI on the Spotify app, next to the loopback one.

The Blob store is separate from the web project's, so neither project's
token can read the other's private data. `vercel deploy --prod` still
works for a one-off deploy and produces the same thing, since no data is
bundled either way.

Then point any MCP client that speaks Streamable HTTP and MCP OAuth at
`https://selector-mcp.vercel.app/mcp`, with no token. It logs in through
Spotify in the browser.

`uv run python scripts/check_mcp_deploy.py` checks the OAuth discovery
endpoints, then calls the server with a five-minute access token it writes
straight to Redis and deletes afterwards: the warehouse and a Spotify
search through the hosted token. Add `--write` to also create a
one-track playlist through the hosted server and delete it again.

## Keeping the warehouse current

Data and code are separate. To refresh the data, with no deploy:

```
uv run python -m selector.warehouse.build            # after a new export
uv run python -m selector.warehouse.build --deploy   # the coarse copy
uv run python -m selector.mcp.deploy_data upload     # overwrites the blob
```

Running instances keep the copy they downloaded until they're recycled.
A new cold start picks up the new one.
