"""LoRA fine-tune a sub-1B causal LM on one arm's train split, produced by
`selector.tagger.dataset`.

CPU-only by design: this project's dev machine has an AMD GPU (no CUDA,
and ROCm doesn't support that card on Windows), so `device_map` is pinned
to `"cpu"` rather than left to auto-detect. Runtime scales roughly linearly
with example count and epochs — a few hundred examples for a couple of
epochs is a tens-of-minutes job on a 6-core CPU; the full ~2,000+ track
training split, once labelled at scale, should be assumed to take several
hours CPU-only. See docs/EVAL.md for the actual numbers run in this project.

No external tracking service — everything logs to a local run directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_MODELS = {
    "qwen": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.2-1B",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _format_example(tokenizer, example: dict, max_length: int) -> dict:
    """Prompt + completion concatenated as one causal-LM sequence, with the
    prompt portion's labels masked to -100 so loss only trains the model to
    produce the completion, not to reproduce the prompt it was given."""
    prompt_ids = tokenizer(example["prompt"] + "\n", add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def _drop_fully_masked(examples: list[dict], split_name: str) -> list[dict]:
    """Examples whose prompt alone reaches max_length truncate the
    completion entirely, leaving every label -100. Such an example
    contributes no training signal, and in eval it turns the mean loss for
    the whole split into NaN (one all-masked sequence has no valid target
    token to average over), so it's dropped rather than fed to the model."""
    kept = [ex for ex in examples if any(label != -100 for label in ex["labels"])]
    dropped = len(examples) - len(kept)
    if dropped:
        print(f"{split_name}: dropping {dropped}/{len(examples)} examples with a fully-truncated completion")
    return kept


def train(
    model_name: str,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    lora_r: int = 8,
    max_length: int = 768,
    resume_from_checkpoint: Path | None = None,
) -> None:
    # Imported lazily: torch/transformers/peft are heavy dependencies only
    # needed for this one command, not for the rest of the package.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32)
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_examples = _drop_fully_masked(
        [_format_example(tokenizer, ex, max_length) for ex in _read_jsonl(train_path)], "train"
    )
    val_examples = _drop_fully_masked(
        [_format_example(tokenizer, ex, max_length) for ex in _read_jsonl(val_path)], "val"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="epoch" if val_examples else "no",
        save_strategy="epoch",
        logging_steps=1,
        report_to=[],  # no external tracking service
    )

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor(
                [b["input_ids"] + [pad_id] * (max_len - len(b["input_ids"])) for b in batch]
            ),
            "labels": torch.tensor([b["labels"] + [-100] * (max_len - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor(
                [b["attention_mask"] + [0] * (max_len - len(b["attention_mask"])) for b in batch]
            ),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_examples),
        eval_dataset=Dataset.from_list(val_examples) if val_examples else None,
        data_collator=collate,
    )

    start = time.monotonic()
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    elapsed = time.monotonic() - start

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")

    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "n_train": len(train_examples),
                "n_val": len(val_examples),
                "elapsed_seconds": elapsed,
                "resumed_from": str(resume_from_checkpoint) if resume_from_checkpoint else None,
            },
            f,
            indent=2,
        )
    print(f"Trained in {elapsed:.1f}s. Adapter saved to {output_dir / 'adapter'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen", choices=list(DEFAULT_MODELS) + ["custom"])
    parser.add_argument("--model-name", default=None, help="overrides --model with an exact HF model id")
    parser.add_argument("--arm", default="C", choices=["A", "B", "C"])
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/runs/latest"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="path to a checkpoint dir under --output-dir (e.g. data/runs/C/checkpoint-4070) to continue from",
    )
    args = parser.parse_args(argv)

    model_name = args.model_name or DEFAULT_MODELS[args.model]
    train(
        model_name=model_name,
        train_path=args.splits_dir / f"train_{args.arm}.jsonl",
        val_path=args.splits_dir / f"val_{args.arm}.jsonl",
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
