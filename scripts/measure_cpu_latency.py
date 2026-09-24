"""Single-request median generation latency on CPU, for comparison against
scripts/measure_gpu_latency.py in docs/EVAL.md's latency row. Loads the
real arm C adapter (same one behind the GPU path) and times
selector.tagger.eval's actual generation function on a handful of real
test-split examples - not a synthetic prompt - since CPU load time is the
expensive part and this only needs to happen once.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from selector.tagger.dataset import build_prompt, load_labels, split_by_artist
from selector.tagger.eval import DEFAULT_MODEL_NAME, _generate_completion, _load_model_and_tokenizer

N_SAMPLES = 8


def main() -> None:
    records = load_labels(Path("data/labels.jsonl"))
    _train, _val, test = split_by_artist(records)
    sample = test[:N_SAMPLES]

    print(f"Loading {DEFAULT_MODEL_NAME} + arm C adapter on CPU...")
    model, tokenizer = _load_model_and_tokenizer(DEFAULT_MODEL_NAME, Path("data/runs/C/adapter"))

    latencies = []
    for record in sample:
        prompt = build_prompt(record, "C") + "\n"
        start = time.monotonic()
        _generate_completion(model, tokenizer, prompt)
        latencies.append(time.monotonic() - start)
    latencies.sort()
    print(f"arm C (CPU): median={statistics.median(latencies):.2f}s  min={latencies[0]:.2f}s  max={latencies[-1]:.2f}s  (n={N_SAMPLES})")


if __name__ == "__main__":
    main()
