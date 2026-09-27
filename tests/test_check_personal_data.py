"""The personal-data guard's Rewind, clock and lyrics rules. Marker strings
are built at runtime so this file doesn't trip the guard it tests."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_personal_data", Path(__file__).parent.parent / "scripts" / "check_personal_data.py"
)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

PRIVATE = "pri" + "vate"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(rel: str, payload: dict | bytes) -> Path:
    path = Path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    path.write_bytes(data)
    return path


def _fails(rel: str) -> bool:
    return guard.main([rel]) == 1


def test_synthetic_report_passes(repo):
    _write("web/public/rewind/all.json", {"audience": "synthetic", "cards": []})
    assert not _fails("web/public/rewind/all.json")


@pytest.mark.parametrize("folder", ["rewind", "wrapped"])
def test_unmarked_public_report_fails(repo, folder):
    _write(f"web/public/{folder}/2024.json", {"schema_version": "2.1", "cards": []})
    assert _fails(f"web/public/{folder}/2024.json")


def test_index_needs_privacy_note(repo):
    _write("web/public/rewind/index.json", {"audience": "synthetic", "windows": []})
    assert _fails("web/public/rewind/index.json")
    _write("web/public/rewind/index.json", {"audience": "synthetic", "privacy": "invented", "windows": []})
    assert not _fails("web/public/rewind/index.json")


def test_private_report_fails_anywhere(repo):
    _write("notes/report.json", {"audience": PRIVATE, "cards": []})
    assert _fails("notes/report.json")


def test_private_export_dir_is_blocked():
    assert guard.path_problem("data/wrapped_private/all.json")


def test_deploy_warehouse_is_blocked():
    assert guard.path_problem("data/selector_deploy.duckdb")
    assert guard.path_problem("selector_deploy.duckdb")


def test_real_clock_in_public_asset_fails(repo):
    card = {"timezone": "America/Chicago", "by_hour": [0] * 24}
    _write("web/public/stats/clock.json", {"cards": [card]})
    assert _fails("web/public/stats/clock.json")
    _write("web/public/stats/clock.json", {"audience": "synthetic", "cards": [card]})
    assert not _fails("web/public/stats/clock.json")


def test_lyrics_under_web_fail_even_gzipped(repo):
    body = json.dumps({"plainLyrics": "la " * 100}).encode()
    _write("web/public/fly/extra.json.gz", gzip.compress(body))
    assert _fails("web/public/fly/extra.json.gz")
    _write("web/public/fly/short.json", {"plainLyrics": "la la"})
    assert not _fails("web/public/fly/short.json")


def test_audio_under_web_fails():
    assert guard.path_problem("web/public/clips/preview.mp3")
