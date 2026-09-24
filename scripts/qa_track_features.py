"""Quick quality check on data/track_features.parquet - not a formal eval
(that's docs/EVAL.md), just sanity checks on the full-warehouse inference
run: does it agree with the teacher on tracks it has ground truth for, is
the output diverse (not collapsed to one mode), and do a few recognisable
tracks look right by eye.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from selector.tagger.dataset import load_labels
from selector.warehouse.build import DEFAULT_DB_PATH

feats = pd.read_parquet("data/track_features.parquet")
print(f"Total rows: {len(feats):,}\n")

# 1. Agreement with teacher ground truth, where we have it (labels.jsonl).
# Most of these tracks were in TRAIN for at least one arm, so this is not
# an eval-quality holdout check - it's a "did something break" check: if a
# model can't even track the label it was trained on, something is wrong.
records = load_labels(Path("data/labels.jsonl"))
truth = pd.DataFrame(
    [
        {
            "track_id": r.track_id,
            "true_valence": r.labels.valence,
            "true_intensity": r.labels.intensity,
            "true_era": r.labels.era,
            "true_mood_tags": tuple(sorted(r.labels.mood_tags)),
        }
        for r in records
    ]
)
joined = feats.merge(truth, on="track_id", how="inner")
print(f"Tracks with teacher ground truth: {len(joined):,} / {len(truth):,} labelled tracks\n")

joined["mood_tags_tuple"] = joined["mood_tags"].apply(lambda x: tuple(sorted(x)))
valence_mae = (joined["valence"] - joined["true_valence"]).abs().mean()
intensity_mae = (joined["intensity"] - joined["true_intensity"]).abs().mean()
era_match = (joined["era"] == joined["true_era"]).mean()
mood_match = (joined["mood_tags_tuple"] == joined["true_mood_tags"]).mean()
print("Agreement with teacher labels (includes train/val/test tracks, not a holdout eval):")
print(f"  valence MAE={valence_mae:.3f}  intensity MAE={intensity_mae:.3f}  era_match={era_match:.1%}  mood_match={mood_match:.1%}")
print("  (compare to docs/EVAL.md's held-out test numbers: arm C valence MAE=0.096, intensity MAE=0.073,")
print("   era_match=61.8%, mood_match=18.9% - this row should look at least as good since parts of it are train data)\n")

# 2. Diversity check - did the model collapse to a small number of modes?
print("mood_tags cardinality (top 10 of", feats["mood_tags"].apply(tuple).nunique(), "distinct sets):")
print(feats["mood_tags"].apply(lambda x: tuple(sorted(x))).value_counts().head(10))
print()
print("valence/intensity spread (should not be a spike at one value):")
print(feats[["valence", "intensity"]].describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]])
print()

# 3. Parse-fallback concentration check.
fallback_rate_by_arm = feats.groupby("arm")["label_source"].apply(lambda s: (s == "parse_fallback").mean())
print("parse_fallback rate by arm:")
print(fallback_rate_by_arm)
print()

# 4. Eyeball check on well-known, high-play-count tracks.
with duckdb.connect(str(DEFAULT_DB_PATH), read_only=True) as con:
    top_tracks = con.execute(
        "SELECT track_id, name, artist, play_count FROM tracks ORDER BY play_count DESC LIMIT 15"
    ).df()
eyeball = top_tracks.merge(feats, on="track_id", how="left")
print("Top 15 most-played tracks and their predicted vibe:")
for _, row in eyeball.iterrows():
    tags = ", ".join(row["mood_tags"]) if isinstance(row["mood_tags"], (list, tuple)) else row["mood_tags"]
    print(
        f"  [{row['arm']}] {row['name']!r} - {row['artist']!r} (played {row['play_count']}x): "
        f"valence={row['valence']:.2f} intensity={row['intensity']:.2f} era={row['era']} "
        f"tags=[{tags}] theme={row['lyrical_theme']!r}"
    )

# 5. For arm C tracks, sanity-check intensity vs measured tempo/energy correlation.
audio = pd.read_parquet("data/audio_features.parquet")
armc = feats[feats["arm"] == "C"].merge(audio, on="track_id", how="inner")
if "tempo_scaled" in armc.columns:
    corr = armc[["intensity", "tempo_scaled"]].corr().iloc[0, 1]
    print(f"\nArm C sanity: corr(predicted intensity, measured tempo_scaled) = {corr:.3f} (n={len(armc)})")
if "rms_mean_scaled" in armc.columns:
    corr = armc[["intensity", "rms_mean_scaled"]].corr().iloc[0, 1]
    print(f"Arm C sanity: corr(predicted intensity, measured rms_mean_scaled) = {corr:.3f} (n={len(armc)})")
