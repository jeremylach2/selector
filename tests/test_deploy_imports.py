"""The Vercel function installs only `requirements.txt`, so everything
`api/index.py` can import must be in it.

A deploy once crashed on every request with `No module named 'scipy'`:
`http_server.py` imported `server.py`, which imports the fly brain. Local
tests never noticed, because the dev environment has everything. This walks
the import graph statically (including imports inside functions) from the
entrypoint, so it catches that without a Vercel build.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
ENTRYPOINT = REPO / "api" / "index.py"

# Distribution name -> import name, where they differ.
IMPORT_NAMES = {"python-dotenv": "dotenv"}
# Installed as dependencies of a listed package, and imported directly.
TRANSITIVE = {"starlette": "mcp"}


def _allowed() -> set[str]:
    names = set()
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            dist = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower()
            names.add(IMPORT_NAMES.get(dist, dist.replace("-", "_")))
    return names | {mod for mod, parent in TRANSITIVE.items() if parent in names}


def _repo_module(name: str) -> Path | None:
    base = SRC.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.exists():
            return path
    return None


def _walk() -> tuple[set[str], set[str]]:
    """(in-repo modules reached, third-party top-level names imported)."""
    seen: set[str] = set()
    third_party: set[str] = set()
    stack = [ENTRYPOINT]
    visited: set[Path] = set()
    while stack:
        path = stack.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top == "selector":
                    parts = name.split(".")
                    for i in range(1, len(parts) + 1):
                        module = ".".join(parts[:i])
                        found = _repo_module(module)
                        if found:
                            seen.add(module)
                            stack.append(found)
                elif top not in sys.stdlib_module_names and top != "__future__":
                    third_party.add(top)
    return seen, third_party


def test_every_import_is_in_requirements():
    _, third_party = _walk()
    missing = third_party - _allowed()
    assert not missing, f"api/index.py can import {sorted(missing)}, which requirements.txt doesn't install"


def test_the_deployment_never_reaches_heavy_or_local_only_code():
    reached, _ = _walk()
    assert "selector.mcp.http_server" in reached
    assert "selector.spotify.remote_store" in reached
    local_only = (
        "selector.mcp.server",
        "selector.fly",
        "selector.spotify.reconcile",
        "selector.dj",
        "selector.tagger",
    )
    for forbidden in local_only:
        assert not [m for m in reached if m == forbidden or m.startswith(forbidden + ".")], forbidden


def test_importing_the_entrypoint_loads_nothing_heavy(tmp_path):
    """The runtime view of the same thing, in a clean interpreter with the
    warehouse path set so no Blob fetch is attempted."""
    code = (
        "import runpy, sys; runpy.run_path(sys.argv[1]); "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'scipy', 'torch', 'sklearn', 'librosa'} or m.startswith(('selector.fly', 'selector.spotify.reconcile'))); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    env_db = str(tmp_path / "none.duckdb")
    result = subprocess.run(
        [sys.executable, "-c", code, str(ENTRYPOINT)],
        env={**os.environ, "SELECTOR_DB": env_db},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
