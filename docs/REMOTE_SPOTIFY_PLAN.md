# Remote Spotify tools plan

A runbook for exposing the live Spotify tools (`spotify_*`) on the hosted
MCP server, with **creating playlists remotely** as the main goal. Written
so a session with no prior context can pick it up. Background is in
`docs/DEPLOY_MCP.md` (the hosted server), `docs/OAUTH_NOTES.md` (the
Spotify auth flow) and `docs/PRIVACY.md` (what may leave this machine).

## Progress log (2026-09-28)

- **Phase 1 code: done.** Steps 2, 3, 5, 6 (code and tests), 7 (script)
  and 9 are in the working tree. 307 tests pass, including the 30 in
  `tests/test_remote_spotify.py`. The token store is in
  `selector.spotify.remote_store`, the shared tools in
  `selector.mcp.spotify_tools`, and the scripts are
  `scripts/setup_remote_spotify.py` (Step 1's key) and
  `scripts/seed_remote_spotify_token.py` (Step 4).
- **Phase 1 operations: Steps 1 and 4 done.** Upstash for Redis is
  connected to the MCP project (it sets `KV_REST_API_URL` /
  `KV_REST_API_TOKEN` on every environment, so the development pull in
  `.env.local` points at the same store). `SELECTOR_TOKEN_KEY` and
  `SPOTIFY_CLIENT_ID` are set on production. The remote token is seeded
  with all four remote scopes. Left: rotate the bearer, push to `main`,
  run `check_mcp_deploy.py --write`. Note that `uv run` fails to sync while
  the local MCP server is running (it holds `selector-mcp.exe`), so the
  scripts were run with `uv run --no-sync`.
- **Open question 2 answered:** a second grant leaves the first working.
  After seeding, the local refresh token still refreshed.
- **Open question 3 answered:** the claude.ai connector sends the static
  bearer token today, so Phase 1 works from claude.ai as is.

## Where things stood before Phase 1 (2026-09-28)

- The hosted server (`api/index.py` → `src/selector/mcp/http_server.py`)
  serves only the nine read-only warehouse tools. `DEPLOY_TOOLS` is the
  list. `tests/test_deploy_warehouse.py` fails if it grows, and
  `tests/test_deploy_imports.py` fails if the deploy can import anything
  under `selector.spotify`.
- The live Spotify tools live in `src/selector/mcp/server.py`, which also
  imports scipy and the fly brain, so the deploy can't import it.
- `src/selector/spotify/auth.py` gets a token by opening a browser and
  catching the redirect on `127.0.0.1:8899`, then caches it in
  `~/.selector/token.json`. Neither of those exists on a Vercel function.
  Worse, if the deploy ever reached `_run_authorize_flow`, it would bind
  port 8899 and block until the function timed out.
- Callers authenticate with a single static bearer token,
  `SELECTOR_MCP_TOKEN`, which never expires.
- The claude.ai "Spotify Selector" connector still lists `spotify_*`
  tools. That list is stale (from before `DEPLOY_TOOLS` was cut down). The
  current deploy doesn't register them. Reconnecting the connector after
  Phase 1 ships will refresh it.

## The two auth problems

1. **Server → Spotify.** The function needs a Spotify credential it can
   use without a browser, and somewhere writable to keep it.
2. **Caller → server.** Once the server can write to the Spotify account,
   the MCP credential is worth more. A leaked static bearer would let
   anyone create playlists on the account, not just read coarse listening
   stats.

Phase 1 solves (1) and adds guardrails around (2). Phase 2 replaces the
static bearer with MCP OAuth that logs in through Spotify, which solves
(2) and makes seeding (1) part of the login.

## Decisions

| Question | Choice | Why |
| --- | --- | --- |
| Who can use it | Single user (the owner) | Spotify Development Mode apps are limited to allowlisted users anyway. Per-user tokens aren't worth the complexity. |
| Where the Spotify token lives | Upstash Redis (Vercel Marketplace), connected to the MCP project only | Writable, has atomic `SET NX` for a refresh lock and `INCR` for rate limits, and Phase 2 needs a store for OAuth codes and clients anyway. Spoken over its REST API with `httpx`, so no new client package. The existing private Blob store would work for a single token but has no locking. |
| Token at rest | Encrypted with Fernet (`cryptography`), key in `SELECTOR_TOKEN_KEY` | A Redis dump or leaked Redis token alone can't be used against Spotify. `cryptography` already comes with `mcp` (via `pyjwt[crypto]`); list it in `requirements.txt` explicitly anyway. |
| Spotify grant type | Keep Authorization Code with PKCE | The existing flow and tests stay. A client-secret (confidential) flow would mean a stolen refresh token alone isn't enough, but it would need a second code path. Revisit in Phase 2, where the server does the auth code exchange itself and could hold the secret. |
| Local and remote tokens | Separate grants | A PKCE refresh can return a new refresh token. If local and remote shared one refresh token, whichever refreshed first could invalidate the other. The remote gets its own authorization, seeded once. |
| Remote scopes | `user-library-read user-top-read user-read-recently-played playlist-modify-private` | Enough for search, library, top items, recently played and creating non-profile playlists. No `playlist-modify-public`, no `playlist-read-private`. |
| Playlist visibility | Remote tool always passes `public=False` (no parameter) | Keeps remote playlists off the profile and out of search. The app's Public/Private toggle isn't reachable through the API (see `OAUTH_NOTES.md`), so that stays a manual step. |

## Phase 1: Spotify tools on the hosted server (bearer auth)

### Step 1. Provision storage and secrets

```
vercel integration add upstash            # Redis, connect to the MCP project only
vercel env ls production                  # confirm the injected vars (currently KV_REST_API_URL / KV_REST_API_TOKEN)
vercel env add SPOTIFY_CLIENT_ID production
```

Add `SELECTOR_TOKEN_KEY` with a script in the style of
`scripts/rotate_mcp_token.py`, so the value is generated locally and piped
to `vercel env add` on stdin, never printed. Save a copy to
`~/.selector/token_key`, since the seeding script (Step 4) needs it to
encrypt.

Done when: `vercel env ls production` shows the Redis vars,
`SPOTIFY_CLIENT_ID` and `SELECTOR_TOKEN_KEY`, and `vercel env pull
.env.local --environment=production` brings the Redis vars locally.

### Step 2. A pluggable token store

New `src/selector/spotify/token_store.py`:

- `TokenStore` protocol with `load() -> TokenSet | None`,
  `save(TokenSet)`, and a `refresh_lock()` context manager.
- `FileTokenStore(path)`, the current behavior (`~/.selector/token.json`,
  `0600`), and a no-op lock.
- `RedisTokenStore(url, token, key)`, which talks to Upstash's REST API with
  `httpx` and Fernet-encrypts the JSON before writing it to one key
  (`spotify:token`). Its lock is `SET spotify:refresh-lock <nonce> NX PX
  10000`, released only if the nonce still matches.

Change `auth.get_valid_token(client_id, store, interactive=True)`:

- A fresh cached token is returned as is.
- An expired one is refreshed under `store.refresh_lock()`. After the lock
  is acquired, reload from the store first, since another concurrent
  request (Fluid Compute reuses instances) may have already refreshed.
- If there's no token or the refresh fails, the **full browser flow runs
  only when `interactive=True`**. Otherwise it raises `SpotifyAuthError`
  with a message saying to re-run the seeding script. The deploy always
  passes `interactive=False`.

`SpotifyClient` takes a `store` instead of `token_path`. Stdio callers
(`server.py`, `dj_set`, the DJ cron) pass `FileTokenStore`, so local
behavior doesn't change.

Also cap `client.py`'s 429 sleep (e.g. `min(Retry-After, 10)`) with a
flag the deploy sets, so one throttled call can't burn through the
function's 60s `maxDuration`.

Tests (`tests/test_spotify_token_store.py`, extend
`tests/test_spotify_auth.py`), all against `httpx.MockTransport`:
encrypt/decrypt round trip, and a wrong key fails loudly. Refresh saves the
new refresh token when Spotify returns one and keeps the old one when it
doesn't. A second request that loses the lock reloads instead of
refreshing. `interactive=False` never calls `webbrowser.open` or binds a
port.

### Step 3. Share the tool definitions between transports

Move the Spotify tools out of `server.py` into
`src/selector/mcp/spotify_tools.py`, the same pattern as
`warehouse_tools.py`, so the stdio and HTTP servers register the same
functions. The module imports only `selector.spotify.client`,
`selector.spotify.auth` and `selector.spotify.token_store`, nothing heavy.
A small factory decides the store: `FileTokenStore` for stdio,
`RedisTokenStore` when the Redis env vars are set.

Two tools get a separate remote version rather than a flag, so the remote
tool's schema can't offer something the remote shouldn't do:

- `spotify_create_playlist` (remote) has no `public` parameter and gets the
  guardrails in Step 5.
- `reconcile_library` stays local-only for now. It reads the warehouse
  through pandas with day-level `last_played`, which the deploy copy
  supports, but the "abandoned" logic hasn't been checked against
  `plays_hourly`. Add it in Step 8 once `test_deploy_warehouse.py`
  compares its output on both warehouses.

Remote tool set: `spotify_search`, `spotify_saved_tracks`,
`spotify_top_artists`, `spotify_top_tracks`, `spotify_recently_played`,
`spotify_create_playlist`.

Add MCP tool annotations: `readOnlyHint=True, openWorldHint=True` on the
five read tools, and `readOnlyHint=False, destructiveHint=False,
idempotentHint=False, openWorldHint=True` on create. Clients such as
claude.ai use these to ask before running a write.

### Step 4. Seed the remote token

New `scripts/seed_remote_spotify_token.py`:

1. Runs the existing PKCE browser flow locally with the **remote scope
   set**.
2. Encrypts the result with `SELECTOR_TOKEN_KEY` and writes it to Redis.
   It never touches `~/.selector/token.json`, and never prints the token.
3. Calls `GET /me` with the new access token and saves the Spotify user id
   as `SELECTOR_OWNER_SPOTIFY_ID` (used by Step 5's audit log now and by
   Phase 2's login check).

To verify during this step: after seeding, confirm the local tools still
work with the old `~/.selector/token.json` (i.e. a second grant didn't
revoke the first). If it did, the plan changes to one shared grant stored
only in Redis, with the local server also reading from it.

Re-running this script is also the recovery path whenever the remote says
Spotify authorization failed.

### Step 5. Guardrails on remote playlist creation

In the remote `spotify_create_playlist`:

- **Input checks.** `name` 1-100 characters. Each URI must match
  `spotify:track:[A-Za-z0-9]{22}`. At most 500 tracks per playlist (Spotify
  allows more, but nothing Claude builds needs it).
- **Rate limit.** At most 20 playlists per UTC day, counted with Redis
  `INCR` + `EXPIRE` on `spotify:playlists:<date>`. Enough for real use,
  small enough that a leaked bearer can't spam the account.
- **No orphaned playlists.** If adding tracks fails after the playlist was
  created, unfollow it (`DELETE /playlists/{id}/followers`, which is how
  the API deletes your own playlist) and return the error.
- **Tag it.** Append " · made with Selector" to the description, so remote
  creations are easy to find and clean up in the app.
- **Audit log.** `LPUSH` `{time, playlist id, name, track count}` onto
  `spotify:audit`, trimmed to 200 entries. No track lists, no listening
  data.
- **Kill switch.** `SELECTOR_REMOTE_SPOTIFY=off` (env var) makes every
  Spotify tool answer that remote Spotify access is disabled. Deleting the
  `spotify:token` key has the same effect without a redeploy.

Tests: each rejection path, the counter, the unfollow-on-failure path, and
that the remote tool's input schema has no `public` field.

### Step 6. Wire it into the deploy

- `http_server.py`: `DEPLOY_TOOLS = WAREHOUSE_TOOLS + REMOTE_SPOTIFY_TOOLS`.
  Update its module docstring, which currently explains why the Spotify
  tools are excluded.
- `requirements.txt`: add `cryptography`.
- `tests/test_deploy_warehouse.py`: add the six names to
  `DEPLOYED_TOOL_NAMES`.
- `tests/test_deploy_imports.py`: the forbidden list becomes
  `selector.mcp.server`, `selector.fly`, `selector.dj`, `selector.tagger`
  and `selector.spotify.reconcile`. Keep the runtime check that nothing
  heavy (scipy, torch, sklearn, librosa) gets loaded.
- Rotate `SELECTOR_MCP_TOKEN` (`scripts/rotate_mcp_token.py`) before the
  deploy that turns this on, so any copy of the old read-only token can't
  create playlists.

### Step 7. Verify end to end

Extend `scripts/check_mcp_deploy.py`:

- `tools/list` includes the six Spotify tools.
- `spotify_search` for a known track returns a URI.
- With `--write`, creates a playlist called `selector-check-<date>` with
  that one track, confirms the returned URL, then unfollows it. Prints only
  yes/no and status codes, as the script does today.

Then, from a real client, ask it to "make a playlist of 20 upbeat tracks
from my rediscovery candidates". That uses `rediscovery_candidates` →
`spotify_search` → `spotify_create_playlist` entirely on the hosted
server. Confirm the playlist shows up in the Spotify app, off the profile.

### Step 8 (optional). `reconcile_library` remotely

Add a deploy-vs-full comparison for it in `test_deploy_warehouse.py`. If
the answers match, add it to the remote set.

### Step 9. Docs

- `docs/DEPLOY_MCP.md`: Scope, Auth and "Why no Redis (yet)" sections.
- `docs/PRIVACY.md`: the hosted server now passes Spotify Web API results
  through to the caller. It stores none of them, only the encrypted token,
  the daily counter and the playlist audit log.
- `docs/OAUTH_NOTES.md`: the remote grant, its scopes, the seeding script,
  and how to revoke (spotify.com → Account → Apps removes every grant for
  the app, local included).
- `.env.example`: `SELECTOR_TOKEN_KEY`, the Redis vars,
  `SELECTOR_OWNER_SPOTIFY_ID`, `SELECTOR_REMOTE_SPOTIFY`.
- `docs/INSTALL_MCP.md`: reconnect the claude.ai connector to refresh its
  tool list.

## Phase 2: MCP OAuth with Spotify login (replaces the static bearer)

Goal: the MCP client logs in through a browser, the server accepts only
the owner's Spotify account, and callers hold short-lived, revocable
tokens instead of a permanent shared secret. MCP clients (claude.ai
connectors, Claude Code) support this flow natively. The installed `mcp`
SDK has the pieces under `mcp.server.auth` (an
`OAuthAuthorizationServerProvider` interface plus handlers for metadata,
dynamic client registration, authorize and token).

### Flow

1. The client hits `/mcp` without a token and gets a 401 pointing at
   `/.well-known/oauth-protected-resource`, which names this server as its
   own authorization server.
2. The client registers itself (`/register`, dynamic client registration)
   and sends the user to `/authorize` with PKCE.
3. `/authorize` saves the MCP request in Redis and redirects to Spotify's
   authorize page with the remote scopes.
4. Spotify redirects to `/oauth/spotify/callback`. The server exchanges the
   code, calls `GET /me`, and **refuses unless the id equals
   `SELECTOR_OWNER_SPOTIFY_ID`**.
5. On success it stores the Spotify token set through `RedisTokenStore`
   (replacing the seeding script), issues an MCP authorization code, and
   redirects back to the client.
6. `/token` swaps that code for an MCP access token (1 hour) and refresh
   token (30 days). Both are stored hashed in Redis, never in plain text.

### Steps

1. Register `https://<production domain>/oauth/spotify/callback` as a
   second redirect URI on the Spotify app (HTTPS is required for
   non-loopback URIs). Use the stable production domain, not a deployment
   URL.
2. Implement the provider in `src/selector/mcp/oauth.py` on top of
   `RedisTokenStore`'s Redis client: clients, pending authorizations, auth
   codes (single use, 5 minute TTL), access and refresh tokens.
3. Decide whether the server should now use the client secret for the
   Spotify code exchange (see Decisions). If yes, store it as
   `SPOTIFY_CLIENT_SECRET` on Vercel.
4. Mount the auth routes in `http_server.py` and replace
   `BearerAuthMiddleware` with the SDK's token verification. Keep the
   fail-closed behavior: missing `SELECTOR_TOKEN_KEY`,
   `SELECTOR_OWNER_SPOTIFY_ID` or Redis config means 503 on everything.
5. Migration: accept both the static bearer and OAuth tokens for one
   deploy, move each client over, then remove `SELECTOR_MCP_TOKEN` and
   `rotate_mcp_token.py`.
6. Revocation: a `scripts/revoke_mcp_sessions.py` that deletes every MCP
   token in Redis. Rotating `SELECTOR_TOKEN_KEY` also invalidates the
   stored Spotify token and forces a fresh login.
7. Tests: someone else's Spotify id is refused, codes can't be reused,
   state mismatch is rejected, expired tokens get 401, `.well-known`
   metadata is correct.
8. Update `DEPLOY_MCP.md`'s Auth section and `PRIVACY.md`.

## Open questions to settle while building

- Does Spotify invalidate the old refresh token when a PKCE refresh returns
  a new one? The store saves whatever comes back either way, but the answer
  decides whether the lock in Step 2 is required or only nice to have.
- ~~Does a second authorization for the same app and user leave the first
  grant working (Step 4)?~~ Yes.
- ~~How is the claude.ai connector authenticating today?~~ With the static
  bearer token, so Phase 1 needs no client changes.
