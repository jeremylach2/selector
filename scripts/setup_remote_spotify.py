"""Set the hosted MCP server's Spotify secrets on Vercel without printing them.

- `SELECTOR_TOKEN_KEY`: a new Fernet key that encrypts the Spotify token in
  Redis. Generated here, handed to `vercel env add` on stdin, and saved to
  ~/.selector/token_key so `seed_remote_spotify_token.py` can encrypt with
  it. Nothing is printed or copied to the clipboard.
- `SPOTIFY_CLIENT_ID`: copied from `.env`. Not a secret, but the function
  needs it to refresh the token.

Refuses to replace an existing key unless `--rotate` is passed, since a
new key makes the stored token unreadable (re-run the seeding script
after rotating). Like every env change, this takes effect on the next
production deploy. Run from the repo root, which is linked to the MCP
project (`.vercel/project.json`):

    uv run python scripts/setup_remote_spotify.py [--rotate]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from cryptography.fernet import Fernet
from dotenv import dotenv_values

KEY_VAR = "SELECTOR_TOKEN_KEY"
KEY_FILE = Path.home() / ".selector" / "token_key"


def _vercel(*args: str, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    npx = shutil.which("npx")
    if npx is None:
        sys.exit("npx not found; install Node.js")
    return subprocess.run([npx, "--yes", "vercel", *args], input=stdin, text=True, check=check)


def _set(var: str, value: str, sensitive: bool) -> None:
    # Absent is fine (first run). If `add` then fails, the variable is
    # missing and the Spotify tools answer "not configured", never open.
    _vercel("env", "rm", var, "production", "--yes", check=False)
    _vercel("env", "add", var, "production", *(["--sensitive"] if sensitive else []), stdin=value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rotate", action="store_true", help="replace an existing token key")
    args = parser.parse_args()

    if not Path(".vercel/project.json").exists():
        sys.exit("run from the repo root, which is linked to the MCP project")
    client_id = dotenv_values(".env").get("SPOTIFY_CLIENT_ID")
    if not client_id:
        sys.exit("SPOTIFY_CLIENT_ID isn't set in .env")
    if KEY_FILE.exists() and not args.rotate:
        sys.exit(f"{KEY_FILE} already exists; pass --rotate to replace it (then re-seed the token)")

    key = Fernet.generate_key().decode("ascii")
    _set(KEY_VAR, key, sensitive=True)
    _set("SPOTIFY_CLIENT_ID", client_id, sensitive=False)

    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key, encoding="utf-8")
    os.chmod(KEY_FILE, 0o600)

    print(f"\n{KEY_VAR} and SPOTIFY_CLIENT_ID set on production. The key was not printed.")
    print(f"  key saved to {KEY_FILE}")
    print("  next: uv run python scripts/seed_remote_spotify_token.py")
    print("  takes effect on the next production deploy of the MCP project")


if __name__ == "__main__":
    main()
