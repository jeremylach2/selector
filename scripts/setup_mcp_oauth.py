"""Set the hosted MCP server's OAuth settings on Vercel.

- `SELECTOR_OWNER_SPOTIFY_ID`: the only Spotify account allowed to log in.
  Read from the `spotify:owner` key that `seed_remote_spotify_token.py`
  wrote to Redis, so it's the same account the hosted token belongs to.
- `SELECTOR_MCP_PUBLIC_URL`: the stable production URL, used as the OAuth
  issuer and to build the Spotify callback URL. Defaults to
  https://selector-mcp.vercel.app.

Neither is a secret, so both are printed. Also prints the callback URL to
add as a redirect URI on the Spotify app (developer.spotify.com →
Dashboard → the app → Settings → Redirect URIs). Like every env change,
this takes effect on the next production deploy. Run from the repo root,
which is linked to the MCP project (`.vercel/project.json`), after
`vercel env pull .env.local --environment=production`:

    uv run python scripts/setup_mcp_oauth.py [--url https://...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from setup_remote_spotify import _set

from selector.mcp import oauth
from selector.spotify import remote_store

DEFAULT_URL = "https://selector-mcp.vercel.app"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="the production URL, no trailing path")
    args = parser.parse_args()

    if not Path(".vercel/project.json").exists():
        sys.exit("run from the repo root, which is linked to the MCP project")
    load_dotenv(".env.local")
    try:
        redis = remote_store.UpstashRedis.from_env()
    except remote_store.RemoteStoreNotConfigured as exc:
        sys.exit(f"{exc} (pull .env.local first)")
    owner = redis.command("GET", remote_store.OWNER_KEY)
    if not owner:
        sys.exit("no spotify:owner in Redis; run seed_remote_spotify_token.py first")

    url = args.url.rstrip("/")
    _set(oauth.OWNER_ENV_VAR, owner, sensitive=False)
    _set(oauth.PUBLIC_URL_ENV_VAR, url, sensitive=False)

    print(f"\n{oauth.OWNER_ENV_VAR}={owner} and {oauth.PUBLIC_URL_ENV_VAR}={url} set on production.")
    print("  add this redirect URI to the Spotify app, next to the loopback one:")
    print(f"    {url}{oauth.CALLBACK_PATH}")
    print("  takes effect on the next production deploy of the MCP project")


if __name__ == "__main__":
    main()
