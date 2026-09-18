"""Merge the librosa and Essentia feature tables into one row per track.

Degrades gracefully: if ``data/features_essentia.parquet`` doesn't exist
(no WSL2 available, or the Essentia half hasn't been run yet), this proceeds
with librosa features alone and sets ``has_essentia = False`` for every row,
rather than failing. The pipeline should be runnable end-to-end on a machine
with no WSL2 at all, just with a narrower feature set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_LIBROSA_PATH = Path("data/features_librosa.parquet")
DEFAULT_ESSENTIA_PATH = Path("data/features_essentia.parquet")
DEFAULT_OUTPUT_PATH = Path("data/audio_features.parquet")

# Bounded features scaled to 0-1. Essentia's model outputs are already
# probabilities in [0, 1]; only librosa's unbounded/differently-scaled
# columns need min-max scaling here, fit per-column over the resolved set.
LIBROSA_BOUNDED_ALREADY = {"key_confidence"}
LIBROSA_MINMAX_COLUMNS = [
    "tempo",
    "beat_strength_mean",
    "onset_density",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "harmonic_percussive_ratio",
]


def _minmax_scale(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        lo, hi = df[col].min(), df[col].max()
        df[f"{col}_scaled"] = 0.5 if hi == lo else (df[col] - lo) / (hi - lo)
    return df


def merge_features(
    librosa_path: Path = DEFAULT_LIBROSA_PATH,
    essentia_path: Path = DEFAULT_ESSENTIA_PATH,
) -> pd.DataFrame:
    librosa_df = pd.read_parquet(librosa_path)
    librosa_df = _minmax_scale(librosa_df, LIBROSA_MINMAX_COLUMNS)

    if essentia_path.exists():
        essentia_df = pd.read_parquet(essentia_path)
        merged = librosa_df.merge(essentia_df, on="track_id", how="left")
        merged["has_essentia"] = merged["track_id"].isin(essentia_df["track_id"])
    else:
        print(f"{essentia_path} not found - proceeding with librosa features only.")
        merged = librosa_df.copy()
        merged["has_essentia"] = False

    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librosa", type=Path, default=DEFAULT_LIBROSA_PATH)
    parser.add_argument("--essentia", type=Path, default=DEFAULT_ESSENTIA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    merged = merge_features(args.librosa, args.essentia)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    print(f"Wrote {len(merged):,} rows to {args.output} ({merged['has_essentia'].mean():.1%} with Essentia features)")


if __name__ == "__main__":
    main()
