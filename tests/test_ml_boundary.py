"""The training boundary in docs/PRIVACY.md: nothing that builds a model's
inputs or trains one may reach the Spotify Web API client.

Spotify's Developer Policy bars using the Spotify Platform or Spotify
Content to train a model. The tagger, the fly brain, and the audio and
ingest steps feeding them run on the GDPR export, preview clips and lrclib
lyrics only. This walks the import graph statically (including imports
inside functions), so it needs none of the heavy ML dependencies.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
FORBIDDEN = "selector.spotify"
TRAINING_PACKAGES = ["selector.tagger", "selector.fly", "selector.audio", "selector.ingest"]


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


MODULES = {_module_name(p): p for p in SRC.rglob("*.py")}


def _imports(module: str) -> set[str]:
    tree = ast.parse(MODULES[module].read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            # `from selector.warehouse import queries` imports a submodule.
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in found if name in MODULES}


def _reachable(module: str) -> dict[str, str]:
    """Every in-repo module `module` imports, directly or not, mapped to the
    module that imported it (for a readable failure)."""
    seen = {module: ""}
    stack = [module]
    while stack:
        current = stack.pop()
        for dep in _imports(current):
            # Importing `a.b.c` runs `a` and `a.b` first.
            parts = dep.split(".")
            for i in range(2, len(parts) + 1):
                name = ".".join(parts[:i])
                if name in MODULES and name not in seen:
                    seen[name] = current
                    stack.append(name)
    return seen


def _chain(seen: dict[str, str], name: str) -> str:
    chain = [name]
    while seen[chain[-1]]:
        chain.append(seen[chain[-1]])
    return " <- ".join(chain)


TRAINING_MODULES = sorted(
    m for m in MODULES if any(m == p or m.startswith(p + ".") for p in TRAINING_PACKAGES)
)


def test_training_modules_were_found():
    assert "selector.tagger.train" in TRAINING_MODULES
    assert "selector.fly.mbon" in TRAINING_MODULES


@pytest.mark.parametrize("module", TRAINING_MODULES)
def test_training_code_never_reaches_the_spotify_client(module):
    seen = _reachable(module)
    leaks = [name for name in seen if name == FORBIDDEN or name.startswith(FORBIDDEN + ".")]
    assert not leaks, _chain(seen, leaks[0])


def test_the_check_would_catch_a_leak():
    seen = _reachable("selector.mcp.server")
    assert "selector.spotify.client" in seen
