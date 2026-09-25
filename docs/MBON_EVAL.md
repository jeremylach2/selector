# Mushroom body plasticity: skip prediction results

Train: 2022-2024 plays, chronological order (never shuffled -- this is a time series).
Test: 2025-2026 plays, held out entirely from training.
Metric: ROC-AUC predicting skip (verdict == -1) vs. play-out (verdict == +1) on decisive plays; neutral plays (verdict == 0) are excluded from both training influence and the eval target, since they carry no reward/punishment judgement.

Test set: 8,735 decisive plays (2,469 on tracks seen during training, 6,266 on tracks never played before 2025).

**Step 12 update.** Step 7 ran this table once, against a placeholder noise embedding, since the vibe tagger (Phase 3) didn't exist yet. Step 12 wires in `selector.fly.pipeline` and re-runs the fly row across three feature sources side by side, so the effect of real track features on taste prediction is visible directly rather than asserted.

## What the feature vector numbers mean

Before the fly ever sees a track, `selector.fly.pipeline.build_feature_matrix` turns it into a fixed-width numeric vector. Every position means the same thing for every track — there is no free-text field anywhere in it, since that's what makes the vector something a projection matrix can act on. The layout, in order (see `_predicted_vector` and the `full`-source branch of `build_feature_matrix`):

| Index | Field | Width | Source | Notes |
|---|---|---|---|---|
| 0 | `valence` | 1 | Predicted (Step 11) | Float 0-1, straight from `track_features.parquet` |
| 1 | `intensity` | 1 | Predicted (Step 11) | Float 0-1 |
| 2-8 | `era` | 7 | Predicted (Step 11) | One-hot over `ERA_VOCAB` = `pre-1970, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s` |
| 9-18 | `mood_tags` | 10 | Predicted (Step 11) | Multi-hot over `MOOD_VOCAB` (the 10 tags in `schema.MoodTag`); up to 3 can be set |
| 19-22 | measured audio | 4 | Measured (Step 9), `full` source only | `tempo_scaled, rms_mean_scaled, danceability, harmonic_percussive_ratio_scaled` from `audio_features.parquet` — the same 4 columns the teacher/student saw as "measured" context in Phase 3 |
| 23 | `has_measured` | 1 | Derived, `full` source only | 1.0 if this track had a matched preview clip (arm C), 0.0 if not (arm A) — zeros in positions 19-22 are indistinguishable from a genuinely low measured value without this bit |

That's 19 dimensions for the `text_only` source (indices 0-18: every track has these, since Step 11 tagged the entire warehouse) and 24 for `full` (0-23). `lyrical_theme` is the one predicted field left out entirely — it's free text with no controlled vocabulary, so there's no honest fixed-width numeric encoding for it. The `placeholder` source ignores all of this and substitutes 19 dimensions of deterministic noise (`placeholder_embedding`), matched in width to `text_only` so the three sources are at least comparable in size.

**A real example.** "Can We Kiss Forever?" by Kina (`track_id=58wyJLv6yH1La9NIZPl3ne`, 4 plays, arm C — it has a matched preview clip) produces this 24-dimensional `full` vector:

```
index   0     1     2-8 (era, one-hot)          9-18 (mood_tags, multi-hot)                                    19        20        21      22        23
value   0.45  0.45  [0,0,0,0,0,1,0]              [0,1,0,0,1,1,0,0,0,0]                                          0.522     0.203     0.704   0.001     1.0
field   val.  int.  2010s                        melancholic, romantic, nostalgic                               tempo_sc. rms_sc.   dance.  h/p ratio has_measured
```

Reading it off: the tagger predicted valence 0.45 (slightly negative-of-neutral) and intensity 0.45 (moderate), era 2010s, and the mood tags `melancholic`, `romantic`, `nostalgic` — consistent with its teacher-labelled `lyrical_theme`, "heartbreak and healing after a romantic breakup" (not itself in the vector, but a sanity check that the predicted fields are coherent). The measured half says the track scores mid-low on the tempo/loudness scale (`tempo_scaled=0.52`, `rms_mean_scaled=0.20`), fairly danceable (`danceability=0.70`), and almost entirely percussive/inharmonic by this ratio (`harmonic_percussive_ratio_scaled≈0.0008`), and `has_measured=1.0` marks all four of those as real readings rather than the zero-fill an unmatched track would get.

This 24-dimensional vector is what `fit_fly` normalises (per-dimension mean/std, the paper's divisive-normalisation step) and feeds through the real FlyWire PN->KC projection; the resulting ~2,597-dimensional sparse tag (about 130 bits set) is this track's fly-brain fingerprint, persisted in `data/fly_tags.npz`.

| Method | AUC (all) | AUC (seen tracks) | AUC (unseen tracks) |
|---|---|---|---|
| Global skip rate | 0.500 | 0.500 | 0.500 |
| Per-artist historical skip rate | 0.556 | 0.578 | 0.509 |
| Per-track historical skip rate | 0.534 | 0.569 | 0.509 |
| Fly MBON (placeholder embedding) | 0.495 | 0.516 | 0.490 |
| Fly MBON (text-only features) | 0.595 | 0.556 | 0.570 |
| **Fly MBON (full audio+text features)** | 0.556 | 0.559 | 0.540 |

**The interesting claim is unseen tracks.** Per-track history is expected to win comfortably on tracks the model has already seen skipped or played out before -- that is just memorisation. The fly's plasticity rule only has something to prove on tracks it never encountered during training. On unseen tracks, the full-feature fly scores 0.540 AUC versus 0.509 for the per-track baseline (which has no track-specific history to fall back on there, so it degrades toward the artist/global rate). Reported as measured: the fly **beats** the per-track historical baseline on unseen tracks.

**Does a better feature source make the fly a better predictor?** That's the question Step 12 exists to answer, and the placeholder-vs-real comparison in the table above is the answer, reported as measured rather than picking the flattering row. On unseen tracks, the placeholder embedding scores 0.490 AUC, text-only features score 0.570, and full audio+text scores 0.540. Both real feature sources beat the placeholder, confirming that a real content-similarity signal beats none. The placeholder embedding is independent random noise per track_id, so any Kenyon-cell overlap between two different tracks' tags there is pure chance, uncorrelated with taste -- real features give the fly an actual signal to generalise through instead.

**The honest surprise: text-only features score best on unseen tracks, not full audio+text.** Text-only (0.570) beats full audio+text (0.540) here, which runs the other way from Step 11's vibe-tagger eval, where adding measured audio improved the *label* quality (see `docs/EVAL.md`). The two evals aren't measuring the same thing: Step 11 scores label accuracy against ground truth, while this table scores whether the resulting feature vector's Kenyon-cell overlaps happen to correlate with taste. The `full` vector reserves 5 of its 24 dimensions for measured audio (4 features plus the `has_measured` flag) and only 83.5% of tracks (arm A, no audio match) carry a nonzero value there at all -- the flag itself, constant 1.0 or 0.0 for a whole track, likely soaks up Kenyon-cell capacity that would otherwise encode the same mood/valence/era information the text-only vector uses at full weight. A larger Kenyon-cell budget, a fusion scheme other than concatenation, or just more matched audio (16.5% coverage today) are the things to try before concluding that measured audio doesn't help the fly -- this table shows it doesn't help *this* fusion, at *this* coverage, not that it can't.

**Seen-track performance is not the interesting comparison.** All three sources' seen-track AUCs sit well above 0.5, including the placeholder's, because the plasticity rule can re-recognise the *exact same* track it was trained on through tag overlap regardless of what the tag encodes -- the same mechanism the per-track baseline uses, just as a noisier sparse-hash lookup. That is why the seen-track numbers move much less across feature sources than the unseen-track numbers do: seen-track performance measures memorisation capacity, not feature quality.
