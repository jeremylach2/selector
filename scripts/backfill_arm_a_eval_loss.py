"""Backfill arm A's epoch 1/2 eval_loss, matching the ad hoc process that
produced data/runs/A/eval_loss_backfill.json for epoch 3.

train.py gained `_drop_fully_masked` after arm A's checkpoint-2068 (epoch 1)
and checkpoint-4136 (epoch 2) were saved, so those epochs' eval_loss was
never recomputed with the masking fix. This loads each checkpoint's adapter
on top of the base model, rebuilds the val split the same way `train.train`
does, and runs `Trainer.evaluate()` against it. Read-only with respect to
the checkpoints themselves: only writes new
data/runs/A/eval_loss_backfill_epoch{1,2}.json files.

CPU-only, ~15-20 min per checkpoint. See docs/DATA_FIX_PLAN.md Step 0.
"""

from __future__ import annotations

import json
from pathlib import Path

from selector.tagger.train import _drop_fully_masked, _format_example, _read_jsonl

MODEL_NAME = "Qwen/Qwen3-0.6B"
VAL_PATH = Path("data/splits/val_A.jsonl")
MAX_LENGTH = 768

CHECKPOINTS = {
    1: Path("data/runs/A/checkpoint-2068"),
    2: Path("data/runs/A/checkpoint-4136"),
}


def backfill_epoch(epoch: int, checkpoint_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    out_path = Path(f"data/runs/A/eval_loss_backfill_epoch{epoch}.json")
    print(f"\n=== Epoch {epoch}: {checkpoint_dir} -> {out_path} ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cpu", torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))

    val_examples = _drop_fully_masked(
        [_format_example(tokenizer, ex, MAX_LENGTH) for ex in _read_jsonl(VAL_PATH)], "val"
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

    args = TrainingArguments(
        output_dir=str(Path("data/runs/A/_reeval_scratch") / f"epoch{epoch}"),
        per_device_eval_batch_size=1,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=Dataset.from_list(val_examples),
        data_collator=collate,
    )

    metrics = trainer.evaluate()
    result = {
        "n_val_filtered": len(val_examples),
        "eval_loss": metrics["eval_loss"],
        "eval_model_preparation_time": metrics.get("eval_model_preparation_time"),
        "eval_runtime": metrics["eval_runtime"],
        "eval_samples_per_second": metrics["eval_samples_per_second"],
        "eval_steps_per_second": metrics["eval_steps_per_second"],
        "epoch": epoch,
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}: eval_loss={result['eval_loss']:.4f} (n={result['n_val_filtered']})")


def main() -> None:
    for epoch, checkpoint_dir in CHECKPOINTS.items():
        backfill_epoch(epoch, checkpoint_dir)


if __name__ == "__main__":
    main()
