"""Merge each arm's LoRA adapter into the Qwen3-0.6B base model and save the
merged weights, so they can be converted to GGUF for GPU inference via
llama.cpp (see docs/EVAL.md "Still open" / docs/GPU_INFERENCE.md).

`infer.py` needs to run generation over 19,386 tracks, which at the CPU
eval run's measured rate (~5s/example after the JSON-complete stopping
criterion) would take on the order of a day; llama.cpp's Vulkan backend
gives this AMD RDNA1 card a real inference path, but llama.cpp only loads
merged (non-PEFT) weights in GGUF form.
"""

from __future__ import annotations

from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-0.6B"
ARMS = ("A", "B", "C")
RUNS_DIR = Path("data/runs")


def merge_arm(arm: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = RUNS_DIR / arm / "adapter"
    merged_path = RUNS_DIR / arm / "merged"

    print(f"[{arm}] loading base model + adapter from {adapter_path}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cpu", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"[{arm}] merged model saved to {merged_path}")
    return merged_path


def main() -> None:
    for arm in ARMS:
        merge_arm(arm)


if __name__ == "__main__":
    main()
