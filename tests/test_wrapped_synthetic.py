"""The synthetic Rewind profile: the public reports must come from the
invented listener in the sample export alone.

Needs the fly artefacts (fingerprints, track features) for the v2 cards, like
the rest of the Rewind tests, but never the real plays or warehouse.
"""

import dataclasses
import json
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from selector.fly.pipeline import FLY_TAGS_PATH
from selector.warehouse import wrapped
from selector.warehouse.build import DEFAULT_DB_PATH, DEFAULT_PLAYS_PATH

pytestmark = pytest.mark.skipif(
    not (FLY_TAGS_PATH.exists() and wrapped.SAMPLE_ZIP_PATH.exists()),
    reason="fly fingerprints or sample export missing; run scripts/build_demo_assets.py",
)

REAL = {DEFAULT_PLAYS_PATH.resolve(), DEFAULT_DB_PATH.resolve()}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("sample")
    profile = dataclasses.replace(
        wrapped.PROFILES["synthetic"],
        plays_path=root / "plays.parquet",
        db_path=root / "selector.duckdb",
        window_db_dir=root / "windows",
        out_dir=root / "out",
    )
    touched: list[Path] = []
    real_read_parquet, real_connect = pd.read_parquet, duckdb.connect

    def read_parquet(path, *args, **kwargs):
        touched.append(Path(path).resolve())
        return real_read_parquet(path, *args, **kwargs)

    def connect(database=":memory:", *args, **kwargs):
        touched.append(Path(str(database)).resolve())
        return real_connect(database, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd, "read_parquet", read_parquet)
        mp.setattr(duckdb, "connect", connect)
        wrapped.build_sample_warehouse(profile)
        wrapped.export_reports(profile)

    reports = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in profile.out_dir.glob("*.json")}
    return {"reports": reports, "touched": touched}


def _sample_track_ids() -> set[str]:
    ids = set()
    with zipfile.ZipFile(wrapped.SAMPLE_ZIP_PATH) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                ids |= {r["spotify_track_uri"].removeprefix("spotify:track:") for r in json.loads(zf.read(name))}
    return ids


def test_never_reads_real_plays_or_warehouse(exported):
    assert exported["touched"], "the spies saw no reads at all"
    assert not REAL & set(exported["touched"])


def test_every_file_is_marked_synthetic(exported):
    reports = exported["reports"]
    assert {"index", "all"} <= set(reports)
    assert all(r["audience"] == "synthetic" for r in reports.values())
    assert reports["index"]["privacy"]


def test_every_card_computes(exported):
    report = exported["reports"]["all"]
    assert [c["id"] for c in report["cards"]] == wrapped.CARD_ORDER


def test_no_track_outside_the_sample(exported):
    sample = _sample_track_ids()
    for report in (r for k, r in exported["reports"].items() if k != "index"):
        for card in report["cards"]:
            if card["id"] == "hidden_gems":
                assert {g["track_id"] for g in card["value"]} <= sample


def test_cluster_names_are_built_names(exported):
    for report in (r for k, r in exported["reports"].items() if k != "index"):
        clusters = next((c for c in report["cards"] if c["id"] == "taste_clusters"), None)
        if clusters is not None:
            assert all(c["name"] == c["built_name"] for c in clusters["value"])
