# Mushroom body plasticity: skip prediction results

Train: 2022-2024 plays, chronological order (never shuffled -- this is a time series).
Test: 2025-2026 plays, held out entirely from training.
Metric: ROC-AUC predicting skip (verdict == -1) vs. play-out (verdict == +1) on decisive plays; neutral plays (verdict == 0) are excluded from both training influence and the eval target, since they carry no reward/punishment judgement.

Test set: 8,735 decisive plays (2,469 on tracks seen during training, 6,266 on tracks never played before 2025).

**Track features are a placeholder** (`notebooks/mbon_eval.py:placeholder_embedding`) -- a deterministic pseudo-random vector seeded per track_id, standing in for the real vibe-tagger feature vector Step 12 substitutes once it exists. This step evaluates the mushroom body's plasticity rule and the real FlyWire wiring from Step 6, not what the features encode.

| Method | AUC (all) | AUC (seen tracks) | AUC (unseen tracks) |
|---|---|---|---|
| Global skip rate | 0.500 | 0.500 | 0.500 |
| Per-artist historical skip rate | 0.556 | 0.578 | 0.509 |
| Per-track historical skip rate | 0.534 | 0.569 | 0.509 |
| Fly MBON (FlyWire connectome) | 0.508 | 0.539 | 0.486 |

**The interesting claim is unseen tracks.** Per-track history is expected to win comfortably on tracks the model has already seen skipped or played out before -- that is just memorisation. The fly's plasticity rule only has something to prove on tracks it never encountered during training. On unseen tracks, the fly MBON scores 0.486 AUC versus 0.509 for the per-track baseline (which has no track-specific history to fall back on there, so it degrades toward the artist/global rate). Reported as measured: the fly **does not beat** the per-track historical baseline on unseen tracks.

**Why the fly still edges out chance on seen tracks but not on unseen ones.** The placeholder embedding is independent random noise per track_id, so two different tracks share no real similarity -- any Kenyon-cell overlap between two different tracks' tags is pure chance, uncorrelated with taste. The fly's small lift over chance on seen tracks (0.539) is the plasticity rule re-recognising the *exact same* track it was trained on, the same mechanism the per-track baseline uses, just implemented as a noisier sparse-hash lookup instead of an exact one -- which is also why it trails the per-track baseline there. On unseen tracks there is no real content-similarity signal for the fly to generalise from yet, since the embedding carries none, so a result indistinguishable from chance is the expected outcome given this input, not a failure of the plasticity rule itself. Step 12 replaces the placeholder with real measured/predicted track features and re-runs this exact comparison -- that is the point at which "does the fly generalise to unseen tracks" becomes a meaningful question to ask.
