# Spotify OAuth notes

Selector authenticates against the live Spotify Web API with **Authorization
Code with PKCE** — the flow Spotify recommends for an app that can't keep a
secret safe (a CLI/MCP server running on someone's own machine qualifies).
No client secret is required or used anywhere in this code, even though the
Spotify dashboard hands you one when you register an app.

## Registering the dev app (one-time, ~5 minutes)

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and create an app.
2. **Redirect URI — register exactly this:**
   ```
   http://127.0.0.1:8899/callback
   ```
   This has to match byte-for-byte, port included, or Spotify rejects the
   authorize request with `INVALID_CLIENT: Invalid redirect URI`. It's
   `127.0.0.1`, not `localhost` — Spotify treats those as different strings
   even though they resolve the same way locally.
3. Copy the **Client ID** into `.env` as `SPOTIFY_CLIENT_ID`. The **Client
   Secret** isn't needed (see above) but `.env.example` has a slot for it
   anyway, since the dashboard gives you both at once.
4. Under the app's API settings, make sure the Web API is enabled — new apps
   get it by default, but it's worth a glance since the whole project
   depends on it.

## Scopes requested

```
user-library-read user-top-read playlist-modify-private playlist-modify-public playlist-read-private user-read-recently-played
```

Covers `saved_tracks`, `top_artists`/`top_tracks`, `create_playlist`, and
`recently_played`/`reconcile_library`. `playlist-modify-public` is requested
even though playlists are created private by default (`public=False`) —
see the migration note below for why it turned out not to matter, but it's
harmless to request and removes one variable when debugging. No
playback-control scopes, since nothing in Phase 1 controls playback.

## How the flow works end to end

1. `get_valid_token()` checks `~/.selector/token.json`. If there's a valid,
   non-expired token, that's it — no network call at all.
2. If the cached token is expired, it POSTs to the token endpoint with
   `grant_type=refresh_token`. This is silent — no browser involved.
3. If there's no cached token, or the refresh token itself was revoked, it
   runs the full interactive flow: generate a PKCE verifier/challenge pair,
   open the system browser to Spotify's authorize page, and start a
   one-shot `http.server` on `127.0.0.1:8899` to catch the redirect. The
   moment Spotify redirects back with a `code`, the local server captures it
   and shuts down — `httpd.handle_request()` blocks for exactly one request,
   so there's no lingering local server process.
4. The `code` (plus the original PKCE verifier — never the challenge) is
   exchanged for an access token and refresh token, which get written to
   `~/.selector/token.json` with `0600` permissions where the OS supports it
   (POSIX; best-effort no-op on Windows, where NTFS ACLs are the real
   mechanism and out of scope here).

State on the redirect is checked against what was sent in step 3 before
anything else happens, so a stray or forged callback can't be exchanged for
a token.

## What's been verified in this repo, and what's still a manual step

Everything in `selector/spotify/auth.py` and `client.py` is covered by unit
tests (`tests/test_spotify_auth.py`, `tests/test_spotify_client.py`) that
mock the network layer directly — PKCE pair generation and shape, the
token-cache read/write round trip, the "cached token is fresh → no network
call at all" path, the "expired → silent refresh" path, the "refresh token
revoked → falls back to full flow" path, and the client's 429/5xx
retry-and-backoff behaviour. None of that requires a real Spotify account.

**The literal browser round trip does need a real, registered Spotify app**,
which only the project owner can create (it's tied to a Spotify account).
To verify it for real:

```
uv run python -c "from selector.spotify.auth import get_valid_token; print(get_valid_token('<your client id>'))"
```

with `SPOTIFY_CLIENT_ID` set in `.env` (or passed directly as above). First
run opens a browser tab, asks you to approve the five scopes above, and
should land on "Selector is connected" before printing a `TokenSet`.
Deleting `~/.selector/token.json` and re-running exercises the cold-start
path again; editing `expires_at` in that file to a past timestamp and
re-running (without deleting `refresh_token`) exercises the silent-refresh
path against the real API instead of a mock.

## Migration note: `/users/{user_id}/playlists` is dead

Spotify's February 2026 Web API migration removed
`POST /users/{user_id}/playlists` for Development Mode apps (deadline March
9, 2026); it now returns a bare `403 {"error": {"status": 403, "message":
"Forbidden"}}` for every caller, regardless of scopes or account. `client.py`
originally called `current_user()` to get an id for that path — that's why
`create_playlist` broke while every read endpoint (`/me`, `/me/tracks`,
`/me/top/*`, `/me/player/recently-played`) kept working fine, which is what
made this look like a scope or permissions problem at first. It wasn't.

The fix was switching to `POST /me/playlists`, the documented replacement,
which needs no separate user-id lookup and no new scopes. Diagnosed by
isolating every Spotify tool individually against the live API (each read
endpoint returning 200, only the old playlist endpoint returning 403) and
confirmed by testing with `playlist-modify-public` added to the requested
scopes — that changed nothing, which ruled out scope as the cause before
searching turned up the actual migration.

**The same migration also renamed the add-tracks endpoint**:
`POST /playlists/{id}/tracks` → `POST /playlists/{id}/items` (same method,
same `{"uris": [...]}` body). It broke the same way — playlist creation via
`/me/playlists` would succeed, then adding tracks would 403, leaving an
empty orphaned playlist behind. `add_tracks_to_playlist` now posts to
`/items`. A `GET /me/playlists` or `GET /playlists/{id}` response also now
nests track count under `items.total` rather than a top-level `tracks` key,
in case anything ever needs to read it back.

**Known unresolved quirk:** setting `public=False` on create, or PUTting
`{"public": false}` to `/playlists/{id}` afterward, does not appear to take
effect — `GET /playlists/{id}` keeps reporting `"public": true` either way.
This only affects whether the playlist is listed on the account's public
profile page, not link access (an unlisted-but-unshared playlist is not
discoverable regardless), so it's a cosmetic gap rather than a privacy leak,
but it's real and unresolved — don't claim private playlist creation works
until Spotify's behavior here is understood better.

## Things that cost time, for the next person

- **`127.0.0.1` vs `localhost`** in the redirect URI is a common first
  failure — they look interchangeable but Spotify does an exact string
  match against the registered value.
- **Fixed loopback port (8899)**, not an ephemeral one. An ephemeral port is
  more "correct" in the abstract, but it means updating the registered
  redirect URI on every run, which defeats the point. A fixed port only
  needs registering once. If 8899 is already taken on someone's machine,
  `LOOPBACK_PORT` in `auth.py` is the one place to change it — and the
  dashboard redirect URI has to change to match.
- **`Retry-After` is a plain integer of seconds** on Spotify's 429s, not an
  HTTP-date — `client.py` assumes that and would need adjusting if that ever
  changes.
