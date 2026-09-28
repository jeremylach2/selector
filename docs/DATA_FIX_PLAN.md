# Vibe tagger data-fix plan

A self-contained runbook for fixing the vibe tagger's training data and
finishing Step 11/12's eval table. Written so a session with no prior
context can execute it end to end, background docs are linked, not
assumed to be already read, but each step below has the exact command to
run and how to tell it worked.

## Progress log (as of 2026-09-23, resume point)

- **Step 1 (backup pilot runs):** done. `data/runs` moved to
  `data/runs_pilot_audio_v1`.
- **Step 2 (scale resolve+fetch to 3,000):** done. `data/audio_matches.parquet`
  has 3,000 rows, 91.5% match rate (2,745/3,000). `docs/AUDIO_MATCHING.md`
  updated with the full-scope numbers, including an operational finding
  that the iTunes Search API started returning HTTP 403 partway through
  the run (rate-limited at this request volume), Deezer picked up the
  slack via the existing fallback logic, so the overall match rate wasn't
  hurt, but iTunes's share of matches dropped from ~60% (pilot) to ~20%
  (full run). See that doc's "Matched source split" section for detail.
- **Step 3 (feature extraction + merge):** done. librosa half: 2,745 rows,
  0 decode failures. Essentia half (WSL, rerun after the Step 3 interruption
  noted below): 2,745/2,745 clips, 0 failures. `merge.py`:
  `data/audio_features.parquet` has 2,745 rows, 94 columns, 100% with
  Essentia features. Note for any future WSL invocation: from this repo's
  Bash tool (Git Bash/MSYS), a `/mnt/d/...` path argument gets mangled by
  MSYS path translation before reaching `wsl.exe`, run via PowerShell
  instead, or set `MSYS_NO_PATHCONV=1` first if using Git Bash.
- **Step 4 (supplemental tail sample):** done.
  - Added `track_ids`/`--track-ids-file` support to `selector/audio/resolve.py`
    (mirrors the existing pattern in `enrich.py:_load_tracks`).
  - Sampled 250 skipped-once + 250 completed-once tracks from the excluded
    tail (`play_count <= 2`), written to `data/supplemental_track_ids.txt`.
  - Resolved audio for those 500 (90.6% match rate, 453/500), bringing the
    combined pool to 3,500 tracks / 3,198 matched.
  - Reran librosa + Essentia + merge over the full combined 3,198-track pool
    (required by the Step 3 rescaling gotcha, not incremental).
  - **Bug found and fixed along the way:** `features_essentia.py`'s
    `extract_features` only caught `RuntimeError` from the audio loader, not
    exceptions from model inference. One clip in the supplemental batch
    (`1EJIcDYXwSqipW5dFe4uJz`) made `TensorflowPredictMusiCNN` return a
    plain list instead of an ndarray, causing an unhandled `TypeError` that
    crashed the whole (uncheckpointed) run ~3,100 tracks in. Fixed by
    wrapping the inference call in a try/except that returns `None` for
    that one clip, matching the "one bad file shouldn't kill the batch"
    pattern already used elsewhere in the codebase (`resolve.py`,
    `features_librosa.py`). Rerun after the fix: 3,197/3,198 succeeded,
    1 clip legitimately skipped.
  - `data/audio_features.parquet`: 3,198 rows, 3,197 with Essentia features.
- **Step 5 (regenerate labels.jsonl):** done. Old file backed up to
  `data/labels_pilot_audio_v1.jsonl`. Added `scripts/label_full_with_supplemental.py`
  (label.py's CLI only supports `--limit N`, not an explicit id list) to
  build the combined top-3,000 + 500-supplemental id set and run
  `enrich_tracks`/`run_labelling` directly. Result: `data/labels.jsonl` has
  3,492 records (8 failures out of 3,500, all schema-validation misses, not
  investigated further), **91.4% (3,190/3,492) now have measured audio
  features**, up from 6.6% (198/3,000) before this plan. 6.0M input /
  558K output tokens total. Note: `label.py`'s `DEFAULT_MODEL` is
  `claude-sonnet-5`, not Haiku as this doc's original cost estimate assumed.
  Actual cost wasn't verified against the $4-9 estimate, so flag it for Step 8.
- **Step 6 (retrain arms A/B/C): done.** Pre-flight check caught that
  `data/splits/*.jsonl` were stale (built from the old pilot labels, dated
  before Step 5's regeneration), `train.py` only reads pre-built split
  files, it doesn't call `dataset.write_split_files` itself, so this would
  have silently trained on old data. Regenerated all three arms' splits
  from the new `data/labels.jsonl` (2,359 train / 646 val / 487 test each,
  790/169/170 artists, no artist crosses a split). Verified Arm B's train
  split: only 8.6% (202/2,359) of prompts now say "none available" (was
  ~93% before the fix). All train completions validate against
  `PredictedLabels`.

  Launched all three arms sequentially in one background chain
  (`scripts/train_arm_{a,b,c}.ps1`, A -> B -> C). Arm A was interrupted
  twice by Claude Code's low-memory task reaper (not a training failure)
  and finished via a manual `--resume-from-checkpoint data\runs\A\checkpoint-2310`
  run outside Claude Code. Arms B and C each ran to completion in a single
  unattended pass (`resumed_from: null` in both `run_info.json`s). All
  three adapters verified healthy: no `eval_loss: NaN` in any epoch,
  monotonically decreasing loss, LoRA config consistent across arms
  (r=8, alpha=16, target_modules=["q_proj","v_proj"], base
  `Qwen/Qwen3-0.6B`).

  | Arm | n_train/n_val used | eval_loss (epoch 1 -> 2 -> 3) |
  |---|---|---|
  | A (lyrics + metadata) | 2,310 / 636 | 0.5784 -> 0.5608 -> **0.5565** |
  | B (measured audio + metadata) | 2,359 / 646 | 0.6678 -> 0.6408 -> **0.6319** |
  | C (lyrics + audio + metadata) | 2,277 / 630 | 0.5851 -> 0.5485 -> **0.5431** |

  A's and C's `n_train`/`n_val` are below the 2,359/646 split-file counts
  because `_drop_fully_masked` drops examples whose completion is fully
  truncated at `max_length=768`, expected for the two arms with the
  longest prompts (lyrics-bearing). B's short audio-feature prompts never
  hit the cap. eval_loss ordering (C lowest, B highest) matches the
  original pilot's ordering and is a reasonable prior, but it's next-token
  perplexity on the label JSON, not a task-level score, Step 7's
  MAE/Spearman/exact-match table is the real answer to "does audio help."
- **Step 7 (implement `evaluate_model`, run the real eval table): done
  2026-09-23.** Results are in `docs/EVAL.md`, raw logs in
  `data/step7_eval_table.log` and `data/step7_zero_shot.log`. Fine-tuned
  arm C is best (valence MAE 0.096, intensity MAE 0.073 vs baseline
  0.163/0.120). Instructed zero-shot is below the baseline on every metric.
  The main run took 4h04m, not the ~15h estimated below, because each
  fine-tuned row ran at ~7s/example. Details of how it was built follow. Implemented in `src/selector/tagger/eval.py`:
  `_load_model_and_tokenizer` (base model, optional `PeftModel` adapter),
  `_generate_completion` (free-generation, not teacher-forced),
  `_parse_prediction` (extracts the first balanced JSON object via
  `json.JSONDecoder.raw_decode`, `rfind("}")` was tried first and is
  wrong, see below), and `evaluate_model` (wires it all into
  `evaluate_predictions`, falling back to the train-mean baseline on a
  parse failure rather than dropping it, tracked via
  `parse_failure_rate`). `build_table` now calls `evaluate_model` for the
  untuned-base and fine-tuned rows of all three arms instead of printing
  `None`.

  **Two real bugs found and fixed before launching the full run:**
  1. `_parse_prediction` originally spanned `find("{")` to `rfind("}")`.
     `train.py` never appends an EOS token to a training completion (see
     `_format_example`), so at inference the fine-tuned model has no
     learned stop signal and keeps generating past a complete, valid JSON
     object, in practice trailing into a second lookalike object. That
     made every fine-tuned prediction unparseable (100% `parse_failure_rate`
     in a smoke test) even though the model's actual output was correct.
     Fixed by using `json.JSONDecoder().raw_decode` to grab just the first
     complete JSON value.
  2. Generation was far slower than the plan's "~1-2s/example" estimate:
     measured ~0.285s/token on this CPU, and with no stop signal the model
     was burning the full `max_new_tokens=160` budget on every example
     (~43s/example actual vs. assumed). Added
     `_JSONCompleteStoppingCriteria` (halts as soon as the generated text
     contains one balanced `{...}`) and tightened `max_new_tokens` to 80
     (verified against the true completion-length distribution across all
     3,492 labels: min 48, p99 67, max 70 tokens, 80 is a safe margin,
     not a loose one). Cut fine-tuned-row generation to ~14s/example.
     Investigated GPU offload (this machine's AMD RX 5600 XT is RDNA1, no
     ROCm on Windows or Linux), `torch-directml` does see the card, but
     force-downgrades `torch` to 2.4.1, which is incompatible with the
     `transformers` version needed for Qwen3 support. Reverted via
     `uv sync --extra finetune`, no lasting effect on the environment.
     Not pursued further this session (llama.cpp + Vulkan would likely
     work but needs a real rewrite of the generation path, flagged as a
     future option, not attempted).

  **Realistic runtime with fixes applied: ~15h**, not the plan's original
  1.5-2h estimate, because the untuned-base-model row never triggers the
  early-stop (it never emits valid JSON at all, see
  `_JSONCompleteStoppingCriteria`'s docstring) and always burns the full
  80-token budget: ~9.3h for the 3 untuned rows + ~5.7h for the 3
  fine-tuned rows.

  **Update 2026-09-23:** a 5-example probe confirmed the untuned base
  model never emits JSON (0/5 parsed: it continues the lyrics, echoes the
  prompt, or loops, always using all 80 tokens), since `build_prompt`
  carries no instruction or schema. `build_table` now scores the untuned
  rows on a seeded 50-example subsample (`--untuned-limit`, default 50,
  `0` = full split), cutting total runtime to ~6.5-7h. The fine-tuned
  rows still use the full 487-example test split.

  Because that raw-prompt row only shows "can't produce the format", an
  instructed zero-shot row was added as the fair base-model baseline:
  same per-arm inputs, wrapped in Qwen3's chat template (thinking off)
  with the schema and allowed `Era`/`MoodTag` values in a system prompt.
  A 5-example probe parsed 5/5 (~20s/example, output wrapped in a
  ```json fence, which `_parse_prediction` handles). Run it after the
  main table finishes, not alongside it:
  ```powershell
  uv run python -m selector.tagger.eval --zero-shot-only | Tee-Object -FilePath data\step7_zero_shot.log
  ```
  (`--zero-shot-limit N` for a seeded subsample; default is the full split.)
  The main table was run with:
  ```powershell
  uv run python -m selector.tagger.eval | Tee-Object -FilePath data\step7_eval_table.log
  ```
  Both logs are UTF-16LE (written by `Tee-Object`); decode with
  `iconv -f UTF-16LE -t UTF-8` outside PowerShell. The main run predates
  the per-example predictions output and the Spearman column, so only the
  zero-shot rows have `data/eval_predictions/*.jsonl` files and printed
  Spearman values. `--tuned-only` regenerates the fine-tuned ones (~3-4h).
- **Step 8 (docs): done 2026-09-23.** `docs/EVAL.md` rewritten with the
  full table, findings, zero-shot failure analysis, setup, the Step 4
  supplemental-sample write-up (its label distribution by group and its
  small net effect) and the pre-fix history. `docs/AUDIO_MATCHING.md`
  already had the full-3,000 numbers; it now also has the supplemental
  batch's match rate (453/500, 90.6%). `docs/futureIdeas.txt` was
  deliberately left as is.

**Optional, not run:** `uv run python -m selector.tagger.eval --tuned-only`
(~3–4h, overnight). It regenerates the fine-tuned rows with per-example
predictions and Spearman, which a C vs A paired bootstrap and a
tail-vs-top error comparison both need. Listed under "Still open" in
`docs/EVAL.md`.

Next action on resume: none required for this plan. The optional run
above is the only remaining item.

## Why this plan exists

`docs/EVAL.md` documents three LoRA fine-tunes (arms A/B/C) that were run
against `data/labels.jsonl`, a 3,000-track teacher-labelled set. An
investigation (see git history / session notes around 2026-09-19) found
two problems with that run:

1. **`data/audio_features.parquet` has exactly 198 rows.** `docs/AUDIO_MATCHING.md`
   documents this as a 200-track *pilot*, explicitly not yet scaled to the
   full top-3,000 scope in `Selector — Project Plan.md`. That scale-up
   never happened, but Step 10 (labelling) and Step 11 (fine-tuning) ran
   on the full 3,000 tracks anyway. Result: only 198 of 3,000 labelled
   tracks (6.6%) have *any* measured audio feature — the `measured` dict
   in `TeacherInput` is all-or-nothing per track (see `enrich.py`), so
   `harmonic_percussive_ratio_scaled`'s sparsity is just one symptom of
   this, not a separate bug.

   This cripples **Arm B** specifically (measured-audio-only input): 2,802
   of its 3,000 training prompts are literally `Measured audio features:
   none available` (see `dataset.py:build_prompt`), meaning the arm
   comparison in EVAL.md can't yet answer the plan's headline question
   ("does measured audio beat lyrics alone?").

2. **Arm A's epoch 1/2 eval_loss is lost, but recoverable.** `train.py`
   gained `_drop_fully_masked` (drops examples whose completion was fully
   truncated, which otherwise NaNs the whole split's averaged eval loss)
   *after* arm A's checkpoints were saved. The final epoch was backfilled
   by rerunning eval only against the saved adapter
   (`data/runs/A/eval_loss_backfill.json`); epochs 1 and 2
   (`checkpoint-2068`, `checkpoint-4136`) were never backfilled the same
   way, even though the checkpoints are still on disk.

A follow-up question raised a **class-imbalance concern**:
the label set is the top 3,000 tracks *by play count*, which the
warehouse data shows is effectively "played 3+ times" — 84.5% of the
19,386-track library (15,561 tracks, 80% of the whole library) has only
1–2 plays and is entirely excluded from every label. Skip-heavy tracks
*within* the top 3,000 are present in reasonable numbers (485 tracks with
skip_rate ≥ 30%), so this isn't a "no bad songs at all" problem — but mood
diversity from tracks the listener sampled once and dropped is
structurally absent from the label set. This plan includes an **optional
Phase 2** to address it.

Read `Selector — Project Plan.md`, `docs/EVAL.md`, `docs/AUDIO_MATCHING.md`,
and `docs/TEACHER.md` for full background before making judgment calls not
covered explicitly below.

## Prerequisites

- `uv` installed, `uv sync` run at repo root (or equivalent venv with
  project deps installed).
- `.env` with `ANTHROPIC_API_KEY` (and `ANTHROPIC_WORKSPACE_ID` if the key
  isn't workspace-scoped) — needed for Step 10 relabelling.
- WSL2 Ubuntu with the Essentia venv already set up per
  `docs/WSL_SETUP.md` (this was done for the pilot and should not need
  redoing — verify with `wsl -d Ubuntu -- bash -c "source ~/essentia_venv/bin/activate && python3 -c 'import essentia'"`
  before assuming it's still good).
- `$env:HF_HOME` pointed at wherever the Qwen3-0.6B weights are already
  cached (`D:\hf_cache` per `scripts/train_arm_*.ps1`) so retraining
  doesn't re-download the base model.
- This machine is CPU-only for training (no usable GPU — see
  `train.py`'s module docstring). All time estimates below assume that.

## Step 0 — Backfill Arm A's epoch 1/2 eval_loss (independent, do any time)

Not a prerequisite for anything else — the adapters this produces numbers
for will likely be superseded by Step 6's retrain. Worth doing anyway
since it's cheap and completes the documented training-dynamics record in
EVAL.md.

Write a small script (there is no checked-in one — the epoch-3 backfill
in `data/runs/A/eval_loss_backfill.json` was produced ad hoc) that:

1. Loads `Qwen/Qwen3-0.6B` + the adapter at `data/runs/A/checkpoint-2068`
   (then repeat for `checkpoint-4136`).
2. Rebuilds the val dataset from `data/splits/val_A.jsonl` using
   `train._format_example` and `train._drop_fully_masked` (import from
   `selector.tagger.train` — don't reimplement the masking logic).
3. Runs `Trainer.evaluate()` (or equivalent) and writes the result next to
   the existing backfill file, e.g.
   `data/runs/A/eval_loss_backfill_epoch1.json` /
   `..._epoch2.json`, matching the shape of the existing
   `eval_loss_backfill.json`.

**Expect:** ~15–20 min CPU per checkpoint (matches the existing
backfill's `eval_runtime` of 824s). Update the epoch table in
`docs/EVAL.md` once both numbers exist.

## Step 1 — Back up the current runs before overwriting them

`scripts/train_arm_{a,b,c}.ps1` hardcode `data\runs\{A,B,C}` as their
output dir, and `train.py` will happily write into an existing dir. The
current adapters are the only record of the pilot-data run described in
EVAL.md, so preserve them first:

```powershell
Move-Item data\runs data\runs_pilot_audio_v1
```

Keep `data\runs_pilot_audio_v1` until the new eval table (Step 7) is
written up — it's the "before" side of the data-fix story and worth
citing in the final README as evidence the fix mattered.

## Step 2 — Scale audio resolve + fetch to the full top-3,000

```powershell
uv run python -m selector.audio.resolve --limit 3000
```

`resolve.py` already defaults `--limit` to 3000 — the pilot was a manual
`--limit 200` override, so this is not a new code path, just the intended
one. It's checkpointed (`data/.audio_resolve_checkpoint.jsonl`) and
idempotent — the 200 pilot tracks are skipped, so this only processes the
~2,800 new ones. It prints a match-rate report at the end
(`match_rate_report`); expect the overall rate to land well below the
pilot's 99% (that number was explicitly flagged in `AUDIO_MATCHING.md` as
inflated by only sampling the most mainstream 200 tracks) — the plan's own
estimate of 75–90% on mainstream, worse on the tail, is the number to
compare against.

**Expect:** ~1–2h, network-bound (iTunes + Deezer search and download per
track, sequential with a 0.2s delay between API calls).

**Acceptance check:** `data/audio_matches.parquet` has ~3,000 rows;
`(matches['local_path'].notna()).mean()` is meaningfully above the current
6.6% (198/3000).

## Step 3 — Scale feature extraction (librosa + Essentia) and merge

```powershell
uv run python -m selector.audio.features_librosa
wsl -d Ubuntu -- bash /mnt/d/codeprojects/spotifyProject/scripts/run_essentia.sh
uv run python -m selector.audio.merge
```

`features_librosa.py` reads `data/audio_matches.parquet` and runs
natively on Windows (multi-process via `--workers`, defaults to CPU
count). `run_essentia.sh` must run inside WSL2 because
`essentia-tensorflow` has no Windows wheels — see `docs/WSL_SETUP.md` if
the venv needs rebuilding. `merge.py` combines both into
`data/audio_features.parquet` and degrades gracefully (sets
`has_essentia=False`) if the Essentia half didn't run.

**Important gotcha:** `merge.py`'s `_minmax_scale` fits min/max **over
whatever set is passed in** — it is not a fixed scaler. Once this rerun
happens, the `_scaled` columns for the *original 198 tracks* will also
change (different min/max now that ~2,000+ more tracks are in the pool).
This means Step 5 (relabelling) must be a full 3,000-track regeneration,
not an incremental "just label the new ones" pass — the old 198 tracks'
`measured` values in `labels.jsonl` are now stale too.

**Expect:** ~1–1.5h CPU-bound for librosa (parallelized), plus WSL/Essentia
inference time (roughly comparable order of magnitude, was fast enough in
the 198-track pilot to not be separately called out — budget another
30–60 min to be safe).

**Acceptance check:** `data/audio_features.parquet` row count matches (or
is close to) Step 2's matched-track count; `merge.py`'s printed
`has_essentia` percentage is high (WSL half succeeded).

## Step 4 (optional, recommended) — Supplemental sample from the excluded tail

Addresses the class-imbalance question: the top-3,000-by-play-count set
structurally excludes the 84.5% of the library played only 1–2 times,
which is where most of the mood/genre diversity the listener *didn't*
keep replaying lives. This step pulls in a small stratified sample from
that tail so the label set isn't only "songs this person listens to
repeatedly."

Query to build the candidate list (adjust sample sizes as desired — 250 +
250 is a reasonable starting point, small enough not to distort the
existing 3,000-track scope much, large enough to matter):

```python
import duckdb
con = duckdb.connect("data/selector.duckdb", read_only=True)

skipped_once = con.execute("""
    SELECT track_id FROM tracks
    WHERE play_count <= 2 AND skip_rate >= 0.99
    ORDER BY random() LIMIT 250
""").df()["track_id"].tolist()

completed_once = con.execute("""
    SELECT track_id FROM tracks
    WHERE play_count <= 2 AND skip_rate = 0.0
    ORDER BY random() LIMIT 250
""").df()["track_id"].tolist()

supplemental_ids = skipped_once + completed_once
```

This needs two small code additions, since neither `resolve.py` nor the
CLI in `enrich.py`/`label.py` currently accepts an explicit track-ID list
alongside the top-N-by-play-count query:

1. **`selector/audio/resolve.py`**: give `resolve_tracks`/`_top_tracks` an
   optional `track_ids: list[str] | None` parameter, mirroring the
   pattern already used in `enrich.py:_load_tracks` (`WHERE track_id =
   ANY(?)` when ids are given, else the existing `ORDER BY play_count
   DESC LIMIT ?`). Add a `--track-ids-file` CLI flag that reads a
   newline-delimited file of ids.
2. **Step 5's labelling call**: build the combined id list (top-3,000 by
   play count, queried directly, plus `supplemental_ids` above) in Python
   and pass it to `enrich_tracks(track_ids=combined_ids)` instead of using
   `--limit 3000`, since `enrich_tracks` already supports an explicit
   `track_ids` argument today.

Then re-run Steps 2–3 for just this supplemental set (or fold it into the
Step 2/3 commands above via the new flag) before moving to Step 5.

**Expect:** ~30–45 min total (500 tracks is a small fraction of Step 2/3's
workload). **Skip this step entirely if time-constrained** — it improves
mood/valence coverage but doesn't fix the audio-coverage bug, which Steps
1–3 already handle. If skipped, proceed with the existing top-3,000 id
list only.

## Step 5 — Regenerate `data/labels.jsonl` from scratch

Because of Step 3's rescaling gotcha, this must be a full regeneration,
not an incremental append. `label.py`'s `_load_cached` skips any
`track_id` already present in the output file regardless of whether its
input changed — so the old file must be moved aside first, or every
existing record deleted, or the teacher will silently keep the stale
pre-audio-fix labels for all 3,000 tracks.

```powershell
Move-Item data\labels.jsonl data\labels_pilot_audio_v1.jsonl
uv run python -m selector.tagger.label --limit 3000
```

(If Step 4 was done, replace the plain `--limit 3000` invocation with a
short script that calls `enrich_tracks(track_ids=combined_ids)` +
`run_labelling(...)` directly, per Step 4's note above, so the
supplemental tracks get labelled too.)

**Expect:** ~1–2h, sequential Haiku API calls (not batched — see
`docs/TEACHER.md`). Cost should be similar to the original run's $6.32
(same token volume per track, same track count), maybe $4–9 depending on
whether Step 4's extra 500 tracks are included.

**Acceptance check:** `data/labels.jsonl` has ~3,000 (or ~3,500 with Step
4) lines; rerun the field-presence count from the investigation —
`sum(1 for r in records if r["input"]["measured"])` — and confirm it's
now a large majority of rows, not 198.

## Step 6 — Retrain arms A/B/C

```powershell
uv run pwsh scripts\train_arm_a.ps1
uv run pwsh scripts\train_arm_b.ps1
uv run pwsh scripts\train_arm_c.ps1
```

(Or invoke `selector.tagger.train` directly per arm if the scripts'
hardcoded paths need adjusting.) These scripts write to `data\runs\{A,B,C}`,
which Step 1 already cleared out. Regenerate the split files first if
`dataset.write_split_files` isn't already wired into the scripts — check
`data/splits/{train,val,test}_{A,B,C}.jsonl` timestamps against the new
`labels.jsonl` before trusting them.

**Expect:** the big one — ~15–18h CPU total if run sequentially (actual
prior run: A 6h32m, B 2h27m, C 6h46m). Arm B's prompts are no longer
mostly "none available" now that most tracks have real measured features,
so its per-example prompt length (and thus runtime) may increase somewhat
versus the prior run. Best run overnight/unattended; `--resume-from-checkpoint`
exists in `train.py` if a run gets killed (arm C needed this last time due
to an OS low-memory kill, not a code bug).

**Acceptance check:** `data/runs/{A,B,C}/adapter/` exists for all three
arms with a valid `run_info.json`; no `eval_loss: NaN` in the final
epoch's `trainer_state.json` (would indicate the masking bug resurfaced,
e.g. from a new fully-truncated example in the enlarged label set).

## Step 7 — Implement `evaluate_model` and run the real eval table

This was already the stated gap in `eval.py` before this data fix — it's
the actual deliverable per `Selector — Project Plan.md` ("the deliverable
that matters is not the model but the eval table"). Implement:

- `evaluate_model(adapter_path, test_records, arm)` — loads the base model
  + adapter, generates a completion per test-split prompt
  (`dataset.build_prompt`), parses it into `PredictedLabels` (handle
  parse failures explicitly, don't silently drop them), and scores with
  the existing `evaluate_predictions`.
- An "untuned base model" variant that skips the adapter load — same
  prompts, base `Qwen/Qwen3-0.6B` only.
- Wire both into `eval.py:build_table`'s currently-`None` rows for arms
  A/B/C (untuned and fine-tuned).

```powershell
uv run python -m selector.tagger.eval
```

**Expect:** ~1–2h to implement, then ~1.5–2h unattended (486-track test
split × 6 rows — untuned + tuned × 3 arms — at roughly the per-example
generation latency already seen in eval runs, ~1–2s/example on CPU).

**Acceptance check:** all nine rows in `build_table`'s printed output have
real numbers, no `"not run this session"` placeholders remaining.

## Step 8 — Update the docs

- `docs/EVAL.md`: replace the "what's still not run" / "honest headline
  finding, still deferred" sections with the real table from Step 7, and
  add a short note on what changed in the data (link back to this plan or
  summarize the audio-coverage fix) so the before/after is auditable.
- `docs/AUDIO_MATCHING.md`: add the full-3,000-scope match-rate numbers
  from Step 2 alongside the existing 200-track pilot numbers.
- If Step 4 was done, document the supplemental-sample methodology and
  its effect (or lack of one) on the mood_tags distribution — this is a
  good thing to be able to say plainly in the README either way.
- Delete or archive `docs/futureIdeas.txt`'s contents once addressed, or
  fold any remaining open threads from it into this plan.

## Time and cost summary

| Step | Agent time | Wall-clock | Cost |
|---|---|---|---|
| 0. Arm A epoch backfill | ~10 min | ~30–40 min | $0 |
| 1. Back up old runs | ~2 min | instant | $0 |
| 2. Resolve + fetch, full 3,000 | ~15 min | ~1–2h | $0 (keyless APIs) |
| 3. Feature extraction + merge | ~15 min | ~1–1.5h | $0 |
| 4. Supplemental sample (optional) | ~30 min incl. small code change | ~30–45 min | ~$1 |
| 5. Regenerate labels.jsonl | ~10 min | ~1–2h | ~$4–9 |
| 6. Retrain arms A/B/C | ~30 min to launch all three | ~15–18h | $0 (local CPU) |
| 7. Implement + run real eval table | ~1–2h coding | ~1.5–2h | $0 |
| 8. Update docs | ~30–45 min | — | $0 |

**Total: roughly 4–6h of active/agent time, spread across ~20–26h of wall
clock**, dominated by Step 6 (best left running overnight). Steps 2–5 can
likely all complete in one sitting if started in the morning; Step 6
starts that evening; Steps 7–8 wrap up the next day. Step 0 is independent
and can run in parallel with anything else, any time.
