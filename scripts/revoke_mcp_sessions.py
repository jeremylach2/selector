"""Log every MCP client out of the hosted server.

Deletes every MCP access token, refresh token, authorization code and
pending login from Redis, so each client has to log in through Spotify
again. Takes effect immediately, no deploy needed.

- `--clients` also deletes the registered clients, so they re-register.
- `--spotify` also deletes the hosted server's Spotify token, which stops
  the Spotify tools until the next login.

The static `SELECTOR_MCP_TOKEN`, while it's still set, isn't in Redis:
remove it with `vercel env rm SELECTOR_MCP_TOKEN production` and redeploy.
Rotating `SELECTOR_TOKEN_KEY` (`setup_remote_spotify.py --rotate`) has the
same effect as `--clients --spotify` once deployed. Needs the Redis pair in
`.env.local`. Prints only counts.

    uv run python scripts/revoke_mcp_sessions.py [--clients] [--spotify]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from selector.mcp import oauth
from selector.spotify import remote_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clients", action="store_true", help="also delete registered clients")
    parser.add_argument("--spotify", action="store_true", help="also delete the hosted Spotify token")
    args = parser.parse_args()

    load_dotenv(".env.local")
    try:
        redis = remote_store.UpstashRedis.from_env()
    except remote_store.RemoteStoreNotConfigured as exc:
        sys.exit(str(exc))

    deleted = oauth.revoke_all_sessions(redis, include_clients=args.clients)
    print(f"deleted {deleted} MCP session keys{' (clients included)' if args.clients else ''}")
    if args.spotify:
        gone = int(redis.command("DEL", remote_store.TOKEN_KEY))
        print(f"hosted Spotify token deleted: {'yes' if gone else 'there was none'}")


if __name__ == "__main__":
    main()
