"""Rotate SELECTOR_MCP_TOKEN on the Vercel MCP project without printing it.

The new token is generated here and handed to `vercel env add` on stdin,
so it never appears in a command line, shell history, terminal output or
a chat transcript. It's saved to ~/.selector/mcp_token (next to the
Spotify OAuth token) and copied to the clipboard, for pasting into the MCP
client's `Authorization: Bearer ...` header.

The running deployment keeps the old token until the next production
deploy; env changes only apply to new deployments. Run from the repo root,
which is linked to the MCP project (`.vercel/project.json`):

    uv run python scripts/rotate_mcp_token.py
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

VAR = "SELECTOR_MCP_TOKEN"
TOKEN_FILE = Path.home() / ".selector" / "mcp_token"


def _vercel(*args: str, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    npx = shutil.which("npx")
    if npx is None:
        sys.exit("npx not found; install Node.js")
    return subprocess.run(
        [npx, "--yes", "vercel", *args], input=stdin, text=True, check=check
    )


def _copy_to_clipboard(value: str) -> bool:
    for cmd in (["clip"], ["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"]):
        if shutil.which(cmd[0]):
            return subprocess.run(cmd, input=value, text=True, check=False).returncode == 0
    return False


def main() -> None:
    if not Path(".vercel/project.json").exists():
        sys.exit("run from the repo root, which is linked to the MCP project")
    token = secrets.token_urlsafe(32)

    # Absent is fine (first run). If `add` then fails, the variable is
    # missing and the next deploy fails closed with 503s, never open.
    _vercel("env", "rm", VAR, "production", "--yes", check=False)
    _vercel("env", "add", VAR, "production", "--sensitive", stdin=token)

    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    os.chmod(TOKEN_FILE, 0o600)
    copied = _copy_to_clipboard(token)

    print(f"\n{VAR} rotated. The value was not printed.")
    print(f"  saved to {TOKEN_FILE}")
    print("  copied to the clipboard" if copied else "  (no clipboard tool found)")
    print("  takes effect on the next production deploy of the MCP project")


if __name__ == "__main__":
    main()
