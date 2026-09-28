"""Give the hosted MCP server its own Spotify login, from this machine.

Runs the usual PKCE browser flow (`selector.spotify.auth`), but with the
narrower `REMOTE_SCOPES` and without touching ~/.selector/token.json: the
result is encrypted with the key from `setup_remote_spotify.py` and written
to the MCP project's Redis. The hosted server refreshes it in place after
that. It's a separate grant from the local one, so a refresh on either side
can't invalidate the other's refresh token.

Afterwards it refreshes the local token once, to confirm the new grant
didn't revoke it. Prints only your Spotify display name, the scopes granted
and yes/no results, never a token.

Needs, from the repo root:
- `.env`: SPOTIFY_CLIENT_ID
- `.env.local`: the Redis URL and token (`vercel env pull .env.local
  --environment=production` after connecting Upstash), or
  UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN from the Upstash console
- ~/.selector/token_key (or SELECTOR_TOKEN_KEY in the environment)

    uv run python scripts/seed_remote_spotify_token.py

Re-run it whenever the hosted server says the Spotify token needs
re-seeding.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from selector.spotify import auth, remote_store

KEY_FILE = Path.home() / ".selector" / "token_key"


def _check_local_token(client_id: str) -> str:
    local = auth.FileTokenStore()
    cached = local.load()
    if cached is None:
        return "no local token to check"
    try:
        local.save(auth._refresh_token(client_id, cached.refresh_token))
    except auth.SpotifyAuthError:
        return "NO: the local refresh token no longer works (the local server will ask you to log in again)"
    return "yes"


def main() -> None:
    load_dotenv(".env")
    load_dotenv(".env.local")
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    if not client_id:
        sys.exit("SPOTIFY_CLIENT_ID isn't set in .env")
    if not os.environ.get(remote_store.KEY_ENV_VAR):
        if not KEY_FILE.exists():
            sys.exit(f"no {remote_store.KEY_ENV_VAR} and no {KEY_FILE}; run setup_remote_spotify.py first")
        os.environ[remote_store.KEY_ENV_VAR] = KEY_FILE.read_text(encoding="utf-8").strip()

    try:
        store = remote_store.RedisTokenStore.from_env()
    except remote_store.RemoteStoreNotConfigured as exc:
        sys.exit(str(exc))

    print("Opening Spotify in the browser. Approve the hosted server's scopes:")
    print(f"  {auth.REMOTE_SCOPES}")
    result = auth._run_authorize_flow(client_id, scopes=auth.REMOTE_SCOPES)
    token = auth._exchange_code_for_token(client_id, result["code"], result["verifier"])
    store.save(token)

    me = httpx.get(
        "https://api.spotify.com/v1/me",
        headers={"Authorization": f"Bearer {token.access_token}"},
        timeout=15,
    )
    me.raise_for_status()
    user = me.json()
    store.redis.command("SET", remote_store.OWNER_KEY, user["id"])

    print(f"\nSeeded the hosted server's Spotify token for {user.get('display_name') or user['id']}.")
    print(f"  scopes granted: {token.scope}")
    print(f"  stored encrypted, read back OK: {'yes' if store.load() == token else 'NO'}")
    print(f"  local token still works: {_check_local_token(client_id)}")


if __name__ == "__main__":
    main()
