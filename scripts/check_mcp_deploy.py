"""Smoke-check the hosted MCP server without printing any secret.

Reads the Blob token and the Redis pair from .env.local (and
web/.env.local for comparison), and SELECTOR_TOKEN_KEY from
~/.selector/token_key. Prints only yes/no results, status codes and tool
names.

It checks the OAuth discovery endpoints, then calls the server with a
five-minute access token written straight to Redis (deleted again at the
end), so no browser login is needed. If ~/.selector/mcp_token exists, it
also reports whether the old static bearer is still accepted.

    uv run python scripts/check_mcp_deploy.py [https://selector-mcp.vercel.app] [--write]

It also checks the remote Spotify tools: a catalogue search through the
hosted server. `--write` additionally creates a one-track playlist through
the hosted server, then deletes it again with the local Spotify login.

Last, a 20-minute `dj_set` dry run, which never writes to Spotify: it
proves the DJ crate downloads from Blob and the five stages run there.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import dotenv_values, load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from selector.mcp import deploy_data, oauth
from selector.spotify import remote_store
from selector.spotify.client import SpotifyClient

REMOTE_SPOTIFY_TOOLS = {
    "spotify_search",
    "spotify_saved_tracks",
    "spotify_top_artists",
    "spotify_top_tracks",
    "spotify_recently_played",
    "spotify_create_playlist",
}

TOKEN_FILE = Path.home() / ".selector" / "mcp_token"
KEY_FILE = Path.home() / ".selector" / "token_key"
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


def _call_tool(url: str, headers: dict, name: str, arguments: dict) -> str:
    call = {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    return _rpc_result(httpx.post(url, headers=headers, json=call, timeout=60))["content"][0]["text"]


def _check_spotify(url: str, headers: dict, names: list[str], write: bool) -> None:
    missing = REMOTE_SPOTIFY_TOOLS - set(names)
    print(f"remote Spotify tools listed: {'yes' if not missing else 'NO, missing ' + ', '.join(sorted(missing))}")
    if missing:
        return
    text = _call_tool(url, headers, "spotify_search", {"query": "Daft Punk One More Time", "limit": 1})
    uri = re.search(r"spotify:track:[A-Za-z0-9]{22}", text)
    print(f"spotify_search through the hosted token: {'yes' if uri else 'NO: ' + text[:120]}")
    if not (write and uri):
        return
    name = f"selector-check-{datetime.now(UTC).date().isoformat()}"
    text = _call_tool(url, headers, "spotify_create_playlist", {"name": name, "track_uris": [uri.group(0)]})
    link = re.search(r"open\.spotify\.com/playlist/([A-Za-z0-9]+)", text)
    print(f"spotify_create_playlist: {'yes' if link else 'NO: ' + text[:120]}")
    if link:
        load_dotenv(".env")
        local = SpotifyClient(client_id=os.environ["SPOTIFY_CLIENT_ID"], interactive=False)
        local.unfollow_playlist(link.group(1))
        print(f"  deleted {name} again with the local login")


def _check_dj(url: str, headers: dict, names: list[str]) -> None:
    if "dj_set" not in names:
        print("dj_set listed: NO")
        return
    text = _call_tool(url, headers, "dj_set", {"minutes": 20, "dry_run": True})
    planned = "## Critique chain" in text and "Dry run" in text
    print(f"dj_set dry run on the hosted crate: {'yes' if planned else 'NO: ' + text[:160]}")
    if planned:
        print(f"  {text.splitlines()[0].strip('_')}")
        if "Planned in UTC" in text or "isn't a known time zone" in text:
            print("  SELECTOR_TIMEZONE isn't set (or isn't valid) on the deployment")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv[1:]
    base = (args[0] if args else "https://selector-mcp.vercel.app").rstrip("/")

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
    crate_url = deploy_data.blob_url(mcp_blob, deploy_data.CRATE_BLOB_PATHNAME)
    with httpx.stream("GET", crate_url, headers={"authorization": f"Bearer {mcp_blob}"}, timeout=30) as r:
        size = sum(len(c) for c in r.iter_bytes()) if r.status_code == 200 else 0
    print(f"DJ crate blob: HTTP {r.status_code}, {size / 1e6:.1f} MB")

    url = f"{base}/mcp"
    r = httpx.post(url, headers=HEADERS, json=INIT, timeout=60)
    points = f'resource_metadata="{base}/.well-known/oauth-protected-resource/mcp"' in r.headers.get("www-authenticate", "")
    print(f"no token -> HTTP {r.status_code} (want 401), points at resource metadata: {'yes' if points else 'NO'}")
    wrong = {**HEADERS, "Authorization": "Bearer not-the-token"}
    print(f"wrong token -> HTTP {httpx.post(url, headers=wrong, json=INIT, timeout=60).status_code} (want 401)")
    resource = httpx.get(f"{base}/.well-known/oauth-protected-resource/mcp", timeout=30)
    ok = resource.status_code == 200 and resource.json().get("authorization_servers") == [base]
    print(f"protected resource metadata names this server: {'yes' if ok else f'NO (HTTP {resource.status_code})'}")
    server = httpx.get(f"{base}/.well-known/oauth-authorization-server", timeout=30)
    ok = server.status_code == 200 and server.json().get("registration_endpoint") == f"{base}/register"
    print(f"authorization server metadata with registration: {'yes' if ok else f'NO (HTTP {server.status_code})'}")

    if TOKEN_FILE.exists():
        legacy = {**HEADERS, "Authorization": f"Bearer {TOKEN_FILE.read_text(encoding='utf-8').strip()}"}
        code = httpx.post(url, headers=legacy, json=INIT, timeout=60).status_code
        print(f"static bearer -> HTTP {code} ({'still accepted' if code == 200 else 'retired'})")

    load_dotenv(".env.local")
    key = os.environ.get(remote_store.KEY_ENV_VAR) or (KEY_FILE.read_text(encoding="utf-8").strip() if KEY_FILE.exists() else None)
    if not key:
        sys.exit(f"no {remote_store.KEY_ENV_VAR} and no {KEY_FILE}")
    redis = remote_store.UpstashRedis.from_env()
    owner = redis.command("GET", remote_store.OWNER_KEY) or "owner"
    token, token_key = oauth.mint_access_token(redis, key, owner)
    try:
        authed = {**HEADERS, "Authorization": f"Bearer {token}"}
        r = httpx.post(url, headers=authed, json=INIT, timeout=60)
        print(f"OAuth access token -> HTTP {r.status_code} (want 200)")
        if r.status_code != 200:
            sys.exit("the OAuth server isn't live yet: has the push to main deployed, with its env vars?")
        _check_tools(url, authed, write)
    finally:
        redis.command("DEL", token_key)


def _check_tools(url: str, authed: dict, write: bool) -> None:
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
    _check_spotify(url, authed, names, write)
    _check_dj(url, authed, names)


if __name__ == "__main__":
    main()
