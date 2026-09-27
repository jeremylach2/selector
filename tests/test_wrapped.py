"""Tests for the Wrapped report (v1 warehouse, v1.5 metadata, v2 fly-brain).

Assumes the real warehouse has already been built, matching the convention
in `test_queries.py`.
"""

import pandas as pd
import pytest

from selector.warehouse import wrapped
from selector.warehouse.build import DEFAULT_DB_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_DB_PATH.exists(), reason="warehouse not built; run selector.warehouse.build first"
)


@pytest.fixture(scope="module")
def report():
    return wrapped.build_report(top_n=5)


def test_build_report_shape(report):
    assert report["schema_version"] == wrapped.SCHEMA_VERSION
    assert set(report) == {
        "schema_version",
        "audience",
        "generated_at",
        "config_hash",
        "window",
        "totals",
        "cards",
    }
    assert report["audience"] == "private"  # the default profile is the real history
    assert report["totals"]["plays"] > 0
    assert report["cards"]  # every v1 card should compute on a real warehouse


def test_build_report_card_order_is_fixed(report):
    ids = [c["id"] for c in report["cards"]]
    expected_order = [cid for cid in wrapped.CARD_ORDER if cid in ids]
    assert ids == expected_order


def test_every_card_has_headline_and_evidence(report):
    for card in report["cards"]:
        assert card["headline"]
        assert 1 <= len(card["evidence"]) <= 3


def test_build_report_missing_warehouse(tmp_path):
    with pytest.raises(FileNotFoundError):
        wrapped.build_report(db_path=tmp_path / "nope.duckdb")


def test_render_markdown_contains_every_headline(report):
    md = wrapped.render_markdown(report)
    for card in report["cards"]:
        assert card["headline"] in md


def test_render_html_contains_every_headline(report):
    html = wrapped.render_html(report)
    assert "<html" in html
    for card in report["cards"]:
        assert card["headline"] in html


def test_tier_b_cards_carry_coverage(report):
    tier_b = [c for c in report["cards"] if c["id"] in {"listening_age", "decade_histogram"}]
    for card in tier_b:
        cov = card["coverage"]
        assert cov["tier"] == "B"
        assert 0 < cov["tracks_used"] <= cov["of"]
        assert 0 < cov["plays_used"] <= cov["plays_of"]


def test_missing_release_years_drops_only_v15_cards(tmp_path):
    report = wrapped.build_report(
        top_n=5, release_years_path=tmp_path / "nope.parquet", include_fly=False
    )
    ids = {c["id"] for c in report["cards"]}
    assert "listening_age" not in ids
    assert "decade_histogram" not in ids
    assert "total_hours" in ids


def test_v2_cards_present_with_tier_a_coverage(report):
    by_id = {c["id"]: c for c in report["cards"]}
    for cid in ("taste_clusters", "hidden_gems"):
        cov = by_id[cid]["coverage"]
        assert cov["tier"] == "A"
        assert 0 < cov["measured"] <= cov["tracks_used"]
    shares = [c["play_share"] for c in by_id["taste_clusters"]["value"]]
    assert shares == sorted(shares, reverse=True)
    assert abs(sum(shares) - 1) < 0.01


def test_hidden_gems_are_rarely_played_and_one_per_artist(report):
    gems = next(c for c in report["cards"] if c["id"] == "hidden_gems")["value"]
    assert all(1 <= g["play_count"] <= wrapped.GEM_MAX_PLAYS for g in gems)
    assert len({g["artist"] for g in gems}) == len(gems)


def test_without_fly_archetype_survives(report):
    no_fly = wrapped.build_report(top_n=5, include_fly=False)
    ids = {c["id"] for c in no_fly["cards"]}
    assert "taste_clusters" not in ids
    assert "hidden_gems" not in ids
    assert "archetype" in ids
    assert "cluster_evenness" not in no_fly["cards"][-1]["metrics"]


def test_available_windows_all_time_then_years():
    windows = wrapped.available_windows()
    assert windows[0] == wrapped.ALL_TIME
    years = [int(w.id) for w in windows[1:]]
    assert years == sorted(years) and years


def test_yearly_reports_partition_all_time(report):
    windows = wrapped.available_windows()[1:]
    yearly = [wrapped.build_report(window=w, include_fly=False) for w in windows]
    assert sum(r["totals"]["plays"] for r in yearly) == report["totals"]["plays"]
    for w, r in zip(windows, yearly):
        assert r["window"]["id"] == w.id
        assert r["window"]["from"].startswith(w.id)


def test_artist_sprint_is_dense_matrix(report):
    sprint = next(c for c in report["cards"] if c["id"] == "artist_sprint")["value"]
    assert len(sprint["cumulative"]) == len(sprint["months"])
    assert all(len(row) == len(sprint["artists"]) for row in sprint["cumulative"])
    # running totals never go down
    for a in range(len(sprint["artists"])):
        col = [row[a] for row in sprint["cumulative"]]
        assert col == sorted(col)


def test_window_contains_is_half_open_in_report_tz():
    # 2024-01-01 00:00 Central is 06:00 UTC; a 03:00 UTC play on New Year's
    # Day is still New Year's Eve in Chicago.
    ts = pd.Series(
        pd.to_datetime(
            ["2024-01-01 03:00", "2024-01-01 06:00", "2025-01-01 05:59", "2025-01-01 06:00"], utc=True
        )
    )
    assert wrapped.Window.year(2024).contains(ts).tolist() == [False, True, True, False]


def test_time_of_day_is_local(report):
    card = next(c for c in report["cards"] if c["id"] == "time_of_day")
    assert card["value"]["timezone"] == wrapped.REPORT_TZ
    assert len(card["value"]["by_hour"]) == 24
    assert sum(card["value"]["by_hour"]) == report["totals"]["plays"]
    assert "UTC" not in card["headline"]


def test_clock_label():
    assert wrapped._clock_label(0) == "12 AM"
    assert wrapped._clock_label(12) == "12 PM"
    assert wrapped._clock_label(15) == "3 PM"


def test_archetype_picks_widest_margin():
    card = wrapped._card_archetype({"discovery_ratio": 0.44, "skip_rate": 0.40}, None)
    # 0.40 / 0.25 = 1.6 beats 0.44 / 0.40 = 1.1
    assert card["value"] == "restless"
    assert card["headline"] == "You're The Restless"


def test_archetype_falls_back_to_all_rounder():
    card = wrapped._card_archetype({"discovery_ratio": 0.1, "skip_rate": 0.05}, None)
    assert card["value"] == "all_rounder"


def test_archetype_lower_is_better_rule():
    card = wrapped._card_archetype({"cluster_evenness": 0.5}, None)
    assert card["value"] == "specialist"


def _dated_tracks():
    return pd.DataFrame(
        {
            "track_id": ["a", "b", "c", "d"],
            "name": ["A", "B", "C", "D"],
            "artist": ["X", "X", "Y", "Z"],
            "album": ["Old", "Old", "New", "Unknown"],
            "play_count": [1, 1, 5, 100],
            "release_year": pd.array([1965, 1965, 2021, None], dtype="Int64"),
        }
    )


def test_listening_age_is_play_weighted_median_over_dated_tracks():
    card = wrapped._card_listening_age(_dated_tracks(), "2026-09-01")
    # 2 plays in 1965, 5 in 2021; the undated 100-play track is excluded.
    assert card["value"] == 2021
    assert card["coverage"]["tracks_used"] == 3
    assert card["coverage"]["plays_used"] == 7
    assert card["coverage"]["plays_of"] == 107
    assert "1965" in card["evidence"][1]


def test_decade_histogram_shares_sum_to_one():
    card = wrapped._card_decade_histogram(_dated_tracks())
    decades = {row["decade"]: row for row in card["value"]}
    assert set(decades) == {1960, 2020}
    assert decades[2020]["play_count"] == 5
    assert abs(sum(r["share"] for r in card["value"]) - 1) < 0.01
    assert card["headline"].startswith("The 2020s")


def test_tier_b_cards_none_without_dated_tracks():
    df = _dated_tracks().assign(release_year=pd.array([None] * 4, dtype="Int64"))
    assert wrapped._card_listening_age(df, "2026-09-01") is None
    assert wrapped._card_decade_histogram(df) is None


def test_freeze_config_is_deterministic():
    a = wrapped.freeze_config(top_n=5)
    b = wrapped.freeze_config(top_n=5)
    assert a["config_hash"] == b["config_hash"]
    c = wrapped.freeze_config(top_n=10)
    assert a["config_hash"] != c["config_hash"]
