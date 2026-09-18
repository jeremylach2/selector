"""Parse a Spotify Extended Streaming History export into a normalised Parquet table.

Reads ``my_spotify_data.zip`` directly, no manual extraction step. Only the audio
history files are used; video/podcast history is out of scope for this project
and is skipped.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_ZIP_PATH = Path("my_spotify_data.zip")
DEFAULT_OUTPUT_PATH = Path("data/plays.parquet")

# Personal or non-analytical fields dropped at ingest. ip_addr in particular must
# never survive into any downstream artifact, fixture, log, or the public demo.
DROPPED_FIELDS = ["ip_addr", "offline_timestamp", "incognito_mode"]

RENAMED_COLUMNS = {
    "master_metadata_track_name": "track_name",
    "master_metadata_album_artist_name": "artist_name",
    "master_metadata_album_album_name": "album_name",
}

# Verdict thresholds, tunable in one place. `reason_end` is the fly brain's
# supervision signal: this rule turns it into a scalar reward/punishment/neutral
# label per play.
VERDICT_RULES = {
    "fwdbtn_skip_completion_max": 0.8,  # below this completion, a fwdbtn is a real skip
    "endplay_punish_completion_max": 0.3,  # below this, an endplay reads as abandonment
}


@dataclass(frozen=True)
class IngestResult:
    frame: pd.DataFrame
    row_count: int
    unique_tracks: int
    verdict_counts: pd.Series


def _read_audio_records(zip_path: Path) -> list[dict]:
    records: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            n
            for n in zf.namelist()
            if Path(n).name.startswith("Streaming_History_Audio_") and n.endswith(".json")
        )
        if not names:
            raise ValueError(f"No Streaming_History_Audio_*.json files found in {zip_path}")
        for name in names:
            with zf.open(name) as f:
                records.extend(json.load(f))
    return records


def _completion(df: pd.DataFrame) -> pd.Series:
    # The export gives no track duration, so the observed max ms_played for a
    # track across the whole history stands in for it.
    max_ms_played = df.groupby("track_id")["ms_played"].transform("max")
    completion = (df["ms_played"] / max_ms_played).clip(upper=1.0)
    return completion.fillna(0.0)


def _verdict(reason_end: pd.Series, completion: pd.Series) -> pd.Series:
    verdict = pd.Series(0, index=reason_end.index, dtype="int8")

    is_fwdbtn_skip = (reason_end == "fwdbtn") & (
        completion < VERDICT_RULES["fwdbtn_skip_completion_max"]
    )
    verdict.loc[is_fwdbtn_skip] = -1

    verdict.loc[reason_end == "trackdone"] = 1
    verdict.loc[reason_end == "backbtn"] = 1

    is_endplay_punish = (reason_end == "endplay") & (
        completion < VERDICT_RULES["endplay_punish_completion_max"]
    )
    verdict.loc[is_endplay_punish] = -1

    return verdict


def _records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Normalise raw export records into the plays schema. Split out for testing."""
    raw = pd.DataFrame(records)

    raw = raw.drop(columns=DROPPED_FIELDS, errors="ignore")

    # Podcast/audiobook rows leak into the audio history files. Anything with
    # no track URI isn't a music play.
    raw = raw[raw["spotify_track_uri"].notna()].copy()

    raw = raw.rename(columns=RENAMED_COLUMNS)

    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw["year"] = raw["ts"].dt.year
    raw["month"] = raw["ts"].dt.month
    raw["hour_utc"] = raw["ts"].dt.hour
    raw["dow"] = raw["ts"].dt.dayofweek

    raw["track_id"] = raw["spotify_track_uri"].str.removeprefix("spotify:track:")

    raw["completion"] = _completion(raw)
    raw["verdict"] = _verdict(raw["reason_end"], raw["completion"])

    raw = raw.reset_index(drop=True)
    return raw


def load_history(zip_path: Path = DEFAULT_ZIP_PATH) -> IngestResult:
    """Parse the export zip into a normalised, privacy-scrubbed plays table."""
    frame = _records_to_frame(_read_audio_records(zip_path))

    return IngestResult(
        frame=frame,
        row_count=len(frame),
        unique_tracks=frame["track_id"].nunique(),
        verdict_counts=frame["verdict"].value_counts().sort_index(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    result = load_history(args.zip_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_parquet(args.output, index=False)

    print(f"Wrote {result.row_count:,} plays to {args.output}")
    print(f"Unique tracks: {result.unique_tracks:,}")
    print("Verdict distribution:")
    for verdict, count in result.verdict_counts.items():
        print(f"  {verdict:+d}: {count:,}")


if __name__ == "__main__":
    main()
