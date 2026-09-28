"""GPU parity check: re-score the 487-track held-out test split
through llama.cpp's Vulkan `llama-server` and confirm the metrics match the
CPU rows already in docs/EVAL.md (identical or within noise), before
trusting the GPU path for infer.py's full 19,386-track run.

Requires a llama-server instance already running per arm (see
docs/GPU_INFERENCE.md):
  arm A -> http://127.0.0.1:8711  (data/runs/A/merged/model-f16.gguf)
  arm C -> http://127.0.0.1:8712  (data/runs/C/merged/model-f16.gguf)

Runs requests concurrently (matching each server's --parallel slot count)
since a single-threaded loop over 487 examples per arm is the whole point
of the GPU move but still adds up serially.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from selector.tagger.dataset import build_prompt, load_labels, split_by_artist
from selector.tagger.eval import (
    _parse_prediction,
    evaluate_predictions,
    train_mean_baseline_predictions,
)
from selector.tagger.gpu_infer import SERVER_URLS, generate_completion_gpu

N_WORKERS = 4

# From docs/EVAL.md's main table, for comparison.
CPU_ROWS = {
    "A": {"valence_mae": 0.106, "intensity_mae": 0.094, "era_exact_match": 0.614, "mood_tags_exact_match": 0.185, "parse_failure_rate": 0.006},
    "C": {"valence_mae": 0.096, "intensity_mae": 0.073, "era_exact_match": 0.618, "mood_tags_exact_match": 0.189, "parse_failure_rate": 0.008},
}


def _score_one(record, arm: str, fallback, client: httpx.Client):
    prompt = build_prompt(record, arm) + "\n"
    raw = generate_completion_gpu(prompt, client, server_url=SERVER_URLS[arm])
    parsed = _parse_prediction(raw)
    return {
        "track_id": record.track_id,
        "true": record.labels.model_dump(),
        "pred": (parsed or fallback).model_dump(),
        "parsed": parsed is not None,
        "raw": raw,
    }, (parsed or fallback), parsed is not None


def run_arm(arm: str, test_records, fallback) -> dict:
    client = httpx.Client(timeout=60.0)
    rows = [None] * len(test_records)
    n_failures = 0
    predictions = [None] * len(test_records)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_score_one, record, arm, fallback, client): i for i, record in enumerate(test_records)
        }
        for future, i in futures.items():
            row, pred, ok = future.result()
            rows[i] = row
            predictions[i] = pred
            if not ok:
                n_failures += 1
    elapsed = time.monotonic() - start
    client.close()

    out_path = Path("data/eval_predictions") / f"gpu_parity_{arm}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    true_labels = [r.labels for r in test_records]
    metrics = evaluate_predictions(true_labels, predictions)
    metrics["parse_failure_rate"] = n_failures / len(test_records)
    metrics["elapsed_seconds"] = elapsed
    return metrics


def main() -> None:
    records = load_labels(Path("data/labels.jsonl"))
    train, _val, test = split_by_artist(records)
    fallback = train_mean_baseline_predictions(train, 1)[0]

    print(f"Parity check: n_test={len(test)}, {N_WORKERS} concurrent requests per arm\n")

    for arm in ("A", "C"):
        print(f"[{arm}] scoring {len(test)} examples via {SERVER_URLS[arm]} ...")
        gpu_metrics = run_arm(arm, test, fallback)
        cpu = CPU_ROWS[arm]
        print(f"  GPU: valence_mae={gpu_metrics['valence_mae']:.3f}  intensity_mae={gpu_metrics['intensity_mae']:.3f}  "
              f"era_match={gpu_metrics['era_exact_match']:.1%}  mood_match={gpu_metrics['mood_tags_exact_match']:.1%}  "
              f"parse_fail={gpu_metrics['parse_failure_rate']:.1%}  ({gpu_metrics['elapsed_seconds']:.1f}s, "
              f"{gpu_metrics['elapsed_seconds'] / len(test):.2f}s/example wall)")
        print(f"  CPU (docs/EVAL.md): valence_mae={cpu['valence_mae']:.3f}  intensity_mae={cpu['intensity_mae']:.3f}  "
              f"era_match={cpu['era_exact_match']:.1%}  mood_match={cpu['mood_tags_exact_match']:.1%}  "
              f"parse_fail={cpu['parse_failure_rate']:.1%}")
        print(f"  Delta: valence_mae={gpu_metrics['valence_mae'] - cpu['valence_mae']:+.4f}  "
              f"intensity_mae={gpu_metrics['intensity_mae'] - cpu['intensity_mae']:+.4f}  "
              f"era_match={gpu_metrics['era_exact_match'] - cpu['era_exact_match']:+.1%}  "
              f"mood_match={gpu_metrics['mood_tags_exact_match'] - cpu['mood_tags_exact_match']:+.1%}\n")


if __name__ == "__main__":
    main()
