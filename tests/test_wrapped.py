"""Tests for the v1 (warehouse-only) Wrapped report slice.

Assumes the real warehouse has already been built, matching the convention
in `test_queries.py`.
"""

import pytest

from selector.warehouse import wrapped
from selector.warehouse.build import DEFAULT_DB_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_DB_PATH.exists(), reason="warehouse not built; run selector.warehouse.build first"
)


def test_build_report_shape():
    report = wrapped.build_report(top_n=5)
    assert report["schema_version"] == wrapped.SCHEMA_VERSION
    assert set(report) == {
        "schema_version",
        "generated_at",
        "config_hash",
        "window",
        "totals",
        "cards",
    }
    assert report["totals"]["plays"] > 0
    assert report["cards"]  # every v1 card should compute on a real warehouse


def test_build_report_card_order_is_fixed():
    report = wrapped.build_report(top_n=5)
    ids = [c["id"] for c in report["cards"]]
    expected_order = [cid for cid in wrapped.CARD_ORDER if cid in ids]
    assert ids == expected_order


def test_every_card_has_headline_and_evidence():
    report = wrapped.build_report(top_n=5)
    for card in report["cards"]:
        assert card["headline"]
        assert 1 <= len(card["evidence"]) <= 3


def test_build_report_missing_warehouse(tmp_path):
    with pytest.raises(FileNotFoundError):
        wrapped.build_report(db_path=tmp_path / "nope.duckdb")


def test_render_markdown_contains_every_headline():
    report = wrapped.build_report(top_n=5)
    md = wrapped.render_markdown(report)
    for card in report["cards"]:
        assert card["headline"] in md


def test_render_html_contains_every_headline():
    report = wrapped.build_report(top_n=5)
    html = wrapped.render_html(report)
    assert "<html" in html
    for card in report["cards"]:
        assert card["headline"] in html


def test_freeze_config_is_deterministic():
    a = wrapped.freeze_config(top_n=5)
    b = wrapped.freeze_config(top_n=5)
    assert a["config_hash"] == b["config_hash"]
    c = wrapped.freeze_config(top_n=10)
    assert a["config_hash"] != c["config_hash"]
