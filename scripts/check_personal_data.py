"""Refuse to commit personal data.

Everything derived from the Spotify export lives under `data/` and is
gitignored, but `.gitignore` only protects against `git add .`, not against
`git add -f` or a file saved somewhere unexpected. This is the second line:
run by the pre-commit hook on staged files and by CI on every tracked file.

It blocks, by path: the export zip, anything under `data/`, raw
`Streaming_History_*.json`, audio files, the warehouse, `.env` files, and the
OAuth token cache. By content (including inside zip files, so the demo's
sample export is checked, not just named): real `ip_addr` values, and
Anthropic, Spotify and PEM secrets.

Rewind reports (see docs/PRIVACY.md): every JSON under `web/public/rewind/`
must be marked audience=synthetic (the index also carries a `privacy`
note), any file marked audience=private is blocked wherever it sits,
and a public asset holding a real clock (`America/Chicago` next to hourly
counts) without the synthetic marker is blocked. Nothing under `web/` may
carry lyrics text, including inside `.gz` assets.

Usage: python scripts/check_personal_data.py [files...]
       (no arguments: every file `git ls-files` reports)
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

BLOCKED_NAMES = {"my_spotify_data.zip", "token.json"}
BLOCKED_SUFFIXES = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".duckdb", ".parquet"}
BLOCKED_NAME_PATTERNS = [
    re.compile(r"^Streaming_History_.*\.json$"),
    re.compile(r"^\.env(\..+)?$"),
]
ALLOWED_NAMES = {".env.example"}

CONTENT_PATTERNS = {
    "Anthropic API key": re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),
    "OAuth refresh token": re.compile(rb'"refresh_token"\s*:\s*"[A-Za-z0-9_-]{20,}"'),
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

# A real address, not a schema example like "ip_addr": "..." or null.
IP_ADDR = re.compile(rb'"ip_addr"\s*:\s*"([0-9a-fA-F.:]{7,})"')
# RFC 5737 / RFC 3849 documentation ranges, safe in test fixtures.
DOCUMENTATION_IPS = (b"192.0.2.", b"198.51.100.", b"203.0.113.", b"2001:db8:", b"2001:DB8:")

MAX_SCAN_BYTES = 50_000_000

# Public report folders: the current one, and the pre-rename one so a stale
# export can't slip back in under the old name.
PUBLIC_REPORT_DIRS = ("web/public/rewind", "web/public/wrapped")
SYNTHETIC_MARKER = re.compile(rb'"audience"\s*:\s*"synthetic"')
PRIVATE_MARKER = re.compile(rb'"audience"\s*:\s*"private"')
REAL_CLOCK = (b"America/Chicago", re.compile(rb'"(by_hour|hour_local)"'))
LYRICS = re.compile(rb'"(plainLyrics|syncedLyrics)"\s*:\s*"(?:[^"\\]|\\.){200,}"')
PROFILE_HINT = "export with `python -m selector.warehouse.wrapped --profile synthetic`"


def path_problem(path: str) -> str | None:
    p = PurePosixPath(path.replace("\\", "/"))
    if p.name in ALLOWED_NAMES:
        return None
    if p.parts and p.parts[0] == "data":
        return "files under data/ are derived from personal data"
    if p.name in BLOCKED_NAMES:
        return f"{p.name} is personal data"
    if p.suffix.lower() in BLOCKED_SUFFIXES:
        return f"{p.suffix} files are never committed (audio or warehouse data)"
    if any(pat.match(p.name) for pat in BLOCKED_NAME_PATTERNS):
        return f"{p.name} looks like a raw export file or secrets file"
    return None


def content_problems(data: bytes) -> list[str]:
    problems = [label for label, pat in CONTENT_PATTERNS.items() if pat.search(data)]
    if any(not m.group(1).startswith(DOCUMENTATION_IPS) for m in IP_ADDR.finditer(data)):
        problems.append("ip_addr value")
    return problems


def _posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def report_problems(path: str, data: bytes) -> list[str]:
    """Rewind-specific checks for one file's (decompressed) bytes."""
    posix = _posix(path)
    problems = []
    if PRIVATE_MARKER.search(data):
        problems.append("private Rewind report (marked audience=private)")
    in_report_dir = any(posix.startswith(d + "/") for d in PUBLIC_REPORT_DIRS)
    if in_report_dir and posix.endswith(".json"):
        try:
            doc = json.loads(data)
        except ValueError:
            doc = None
        if not isinstance(doc, dict) or doc.get("audience") != "synthetic":
            problems.append(f"public Rewind report is not marked audience=synthetic; {PROFILE_HINT}")
        elif PurePosixPath(posix).name == "index.json" and not doc.get("privacy"):
            problems.append("public Rewind index.json is missing its privacy note")
    if posix.startswith("web/public/") and not SYNTHETIC_MARKER.search(data):
        tz, hourly = REAL_CLOCK
        if tz in data and hourly.search(data):
            problems.append("public asset has hourly listening counts on a real clock (America/Chicago)")
    if posix.startswith("web/") and LYRICS.search(data):
        problems.append("lyrics text under web/")
    return problems


def file_problems(path: Path) -> list[str]:
    if not path.is_file() or path.stat().st_size > MAX_SCAN_BYTES:
        return []
    data = path.read_bytes()
    problems = content_problems(data)
    scanned = data
    if path.suffix.lower() == ".gz" and _posix(path).startswith("web/"):
        try:
            scanned = gzip.decompress(data)[:MAX_SCAN_BYTES]
            problems += content_problems(scanned)
        except (OSError, EOFError):
            pass
    problems += report_problems(_posix(path).removesuffix(".gz"), scanned)
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    if (reason := path_problem(member)) and "Streaming_History" not in member:
                        problems.append(f"{member}: {reason}")
                    problems += [f"{member}: {p}" for p in content_problems(zf.read(member))]
        except zipfile.BadZipFile:
            pass
    return problems


def main(argv: list[str]) -> int:
    files = argv or subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    failures = []
    for f in files:
        if reason := path_problem(f):
            failures.append(f"{f}: {reason}")
        failures += [f"{f}: {p}" for p in file_problems(Path(f))]

    if failures:
        print("Personal data or secrets would be committed:")
        for line in failures:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
