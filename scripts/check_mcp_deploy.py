"""Smoke-check the hosted MCP server without printing any secret.

Reads the Blob token from .env.local (and web/.env.local for comparison)
and the bearer token from ~/.selector/mcp_token (written by
rotate_mcp_token.py). Prints only yes/no results, status codes and tool
names.

    uv run python scripts/check_mcp_deploy.py [https://selector-mcp.vercel.app]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from selector.mcp import deploy_data

TOKEN_FILE = Path.home() / ".selector" / "mcp_token"
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "check", "version": "0"}},
}


def _rpc_result(response: httpx.Response) -> dict:
    text = response.text
    return json.loads(text[text.find("{"):])["result"]


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "https://selector-mcp.vercel.app").rstrip("/")

    mcp_blob = dotenv_values(".env.local").get(deploy_data.TOKEN_ENV_VAR)
    web_blob = dotenv_values("web/.env.local").get(deploy_data.TOKEN_ENV_VAR)
    if not mcp_blob:
        sys.exit("no BLOB_READ_WRITE_TOKEN in .env.local")
    if web_blob:
        separate = deploy_data._store_id(mcp_blob) != deploy_data._store_id(web_blob)
        print(f"MCP and web use separate Blob stores: {'yes' if separate else 'NO, same store'}")

    with httpx.stream(
        "GET", deploy_data.blob_url(mcp_blob), headers={"authorization": f"Bearer {mcp_blob}"}, timeout=30
    ) as r:
        size = sum(len(c) for c in r.iter_bytes()) if r.status_code == 200 else 0
    print(f"warehouse blob: HTTP {r.status_code}, {size / 1e6:.1f} MB")

    url = f"{base}/mcp"
    print(f"no token -> HTTP {httpx.post(url, headers=HEADERS, json=INIT, timeout=60).status_code} (want 401)")
    wrong = {**HEADERS, "Authorization": "Bearer not-the-token"}
    print(f"wrong token -> HTTP {httpx.post(url, headers=wrong, json=INIT, timeout=60).status_code} (want 401)")

    if not TOKEN_FILE.exists():
        sys.exit(f"{TOKEN_FILE} missing; run rotate_mcp_token.py")
    authed = {**HEADERS, "Authorization": f"Bearer {TOKEN_FILE.read_text(encoding='utf-8').strip()}"}
    r = httpx.post(url, headers=authed, json=INIT, timeout=60)
    print(f"new token -> HTTP {r.status_code} (want 200)")
    if r.status_code != 200:
        sys.exit("the new token isn't live yet: has the push to main deployed?")
    tools = _rpc_result(httpx.post(url, headers=authed, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, timeout=60))
    names = sorted(t["name"] for t in tools["tools"])
    print(f"tools ({len(names)}): {', '.join(names)}")
    call = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "warehouse_summary", "arguments": {}}}
    text = _rpc_result(httpx.post(url, headers=authed, json=call, timeout=60))["content"][0]["text"]
    ok = text.startswith("| earliest_play")
    print(f"warehouse_summary answers from the Blob data: {'yes' if ok else 'NO: ' + text[:80]}")
    if ok:
        first_row = text.splitlines()[2]
        print(f"  day-granular dates: {'yes' if '00:00:00' in first_row else 'NO'}")


if __name__ == "__main__":
    main()
