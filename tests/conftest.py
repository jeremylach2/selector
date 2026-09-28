import pandas as pd
import pytest

from selector.warehouse.build import build_deploy_warehouse, build_warehouse


def _plays() -> pd.DataFrame:
    rows = []
    # Midday UTC, mid-month, so no query's bucket edge depends on the
    # machine's zone. No two artists tie in any period, so rankings are
    # deterministic.
    for day, hour, minute, track, artist, reason_end in [
        ("2024-03-14", 12, 5, "t1", "Alpha", "trackdone"),
        ("2024-03-14", 12, 9, "t1", "Alpha", "fwdbtn"),
        ("2024-03-14", 13, 0, "t2", "Beta", "trackdone"),
        ("2024-03-14", 13, 20, "t2", "Beta", "trackdone"),
        ("2024-06-15", 15, 30, "t1", "Alpha", "trackdone"),
        ("2024-06-15", 16, 5, "t1", "Alpha", "trackdone"),
        ("2024-06-15", 15, 41, "t3", "Gamma", "fwdbtn"),
        ("2025-01-15", 18, 2, "t2", "Beta", "trackdone"),
    ]:
        ts = pd.Timestamp(f"{day} {hour:02d}:{minute:02d}:17", tz="UTC")
        rows.append(
            {
                "ts": ts,
                "platform": "android",
                "ms_played": 180_000,
                "conn_country": "US",
                "track_name": f"Song {track}",
                "artist_name": artist,
                "album_name": f"Album {artist}",
                "reason_start": "clickrow",
                "reason_end": reason_end,
                "hour_utc": ts.hour,
                "dow": ts.dayofweek,
                "track_id": track,
                "completion": 1.0 if reason_end == "trackdone" else 0.2,
                "verdict": 1 if reason_end == "trackdone" else -1,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def warehouses(tmp_path):
    """`(full, deploy)` warehouses built from a few invented plays, so the
    deploy tests run in CI, where the real warehouse doesn't exist."""
    plays_path = tmp_path / "plays.parquet"
    _plays().to_parquet(plays_path)
    full = tmp_path / "selector.duckdb"
    build_warehouse(plays_path, full)
    deploy = build_deploy_warehouse(full, tmp_path / "selector_deploy.duckdb")
    return full, deploy
