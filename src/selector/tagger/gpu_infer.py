"""GPU-backed generation client for llama.cpp's Vulkan `llama-server`.

CPU generation on this dev machine's AMD Radeon RX 5600 XT (RDNA1, no ROCm
support on Windows) would take on the order of a day across 19,386 tracks -
see docs/GPU_INFERENCE.md for why llama.cpp's Vulkan backend was chosen
over ROCm and torch-directml, and the measured parity check against
`selector.tagger.eval`'s CPU rows.

Uses the exact same raw prompt framing as
`selector.tagger.dataset.build_prompt` + "\\n" (no chat template - the
fine-tuned adapters were trained on a raw continuation, not an instructed
turn) and greedy decoding (temperature=0, top_k=1, matching
`do_sample=False` on the CPU path). The response is parsed with
`selector.tagger.eval._parse_prediction`, which extracts only the first
complete JSON object, so a generation that runs past the closing brace (the
model has no learned EOS - see docs/EVAL.md "Bugs found") parses
identically whether or not the server stops early.
"""

from __future__ import annotations

import httpx

DEFAULT_SERVER_URL = "http://127.0.0.1:8712"

# One llama-server instance per arm actually used by infer.py (arm B has no
# trained GGUF export - it's never the winning or fallback arm).
SERVER_URLS = {"A": "http://127.0.0.1:8711", "C": "http://127.0.0.1:8712"}


def generate_completion_gpu(
    prompt: str,
    client: httpx.Client,
    server_url: str = DEFAULT_SERVER_URL,
    max_new_tokens: int = 80,
) -> str:
    """One greedy completion from a running llama-server instance."""
    resp = client.post(
        f"{server_url}/completion",
        json={
            "prompt": prompt,
            "n_predict": max_new_tokens,
            "temperature": 0,
            "top_k": 1,
            "cache_prompt": False,
        },
    )
    resp.raise_for_status()
    return resp.json()["content"]


def server_healthy(server_url: str = DEFAULT_SERVER_URL, client: httpx.Client | None = None) -> bool:
    owns_client = client is None
    client = client or httpx.Client(timeout=5.0)
    try:
        resp = client.get(f"{server_url}/health")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
    finally:
        if owns_client:
            client.close()
