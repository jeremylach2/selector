"""Single-request (no concurrency) median generation latency per arm on the
GPU path, for docs/EVAL.md and docs/GPU_INFERENCE.md's latency row. The
parity check's 1.1-1.3s/example figures are throughput under 4-way
concurrency, not single-request latency - this measures the latter
directly against the already-running llama-server instances.
"""

from __future__ import annotations

import statistics
import time

import httpx

from selector.tagger.dataset import build_prompt_from_input
from selector.tagger.gpu_infer import SERVER_URLS, generate_completion_gpu
from selector.tagger.schema import TeacherInput

N_SAMPLES = 15

SAMPLE_ITEM = TeacherInput(
    track_id="latency_probe",
    track_name="Latency Probe Track",
    artist_name="Test Artist",
    album_name="Test Album",
    lyrics="This is a short sample lyric line for timing purposes only.",
    measured={"tempo_scaled": 0.6, "rms_mean_scaled": 0.5, "danceability": 0.7, "harmonic_percussive_ratio_scaled": 0.4},
)


def main() -> None:
    client = httpx.Client(timeout=60.0)
    for arm in ("A", "C"):
        prompt = build_prompt_from_input(SAMPLE_ITEM, arm) + "\n"
        latencies = []
        for _ in range(N_SAMPLES):
            start = time.monotonic()
            generate_completion_gpu(prompt, client, server_url=SERVER_URLS[arm])
            latencies.append(time.monotonic() - start)
        latencies.sort()
        print(
            f"arm {arm}: median={statistics.median(latencies):.3f}s  "
            f"min={latencies[0]:.3f}s  max={latencies[-1]:.3f}s  (n={N_SAMPLES})"
        )
    client.close()


if __name__ == "__main__":
    main()
