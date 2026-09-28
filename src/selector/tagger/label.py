"""The teacher labelling run: calls an Anthropic model with structured
output to produce the subjective (predicted) half of the vibe tagger's
schema for each track.

Batches, caches by track_id (so a re-run only pays for tracks not already
labelled), checkpoints to JSONL after every batch, and logs running token
spend. Structured output is enforced by forcing a single tool call whose
input schema is `PredictedLabels.model_json_schema()`. Anthropic's API has
no separate "JSON mode", so a forced tool call is the standard way to get
schema-validated output back.

Requires `ANTHROPIC_API_KEY` in the environment. See docs/TEACHER.md for
cost, self-consistency, and how the pilot's labels were actually produced in
a session with no separate API key configured.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from selector.tagger.enrich import enrich_tracks

load_dotenv()


def _build_client() -> Anthropic:
    """Some API keys aren't scoped to a single workspace, in which case the
    API requires the workspace to be named explicitly via this header."""
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace_id:
        return Anthropic(default_headers={"anthropic-workspace-id": workspace_id})
    return Anthropic()
from selector.tagger.schema import LabelRecord, PredictedLabels, TeacherInput

DEFAULT_OUTPUT_PATH = Path("data/labels.jsonl")
DEFAULT_MODEL = "claude-sonnet-5"
BATCH_SIZE = 20

SYSTEM_PROMPT = """You are labelling music tracks for a recommendation system's training data.
For each track you are given its metadata, lyrics (if available), and — where available —
MEASURED audio features (tempo, energy, danceability, etc). Reason about valence and intensity
knowing the measured tempo/energy where given; don't guess at them from text alone if a measured
value is present. mood_tags must contain between 1 and 3 tags, picking the ones that fit best —
never more than 3. lyrical_theme must be a short phrase, at most 60 characters — a few words,
not a sentence. valence and intensity are each a float between 0.0 and 1.0 inclusive — never a
1-10 scale or an integer outside that range. Respond only by calling the emit_labels tool."""

TOOL_NAME = "emit_labels"


def _strip_unsupported_strict_keywords(node: object) -> None:
    """Strict tool use validates type/enum/required structurally but rejects
    numeric range keywords outright (400: "properties maximum, minimum are
    not supported"). Range checking still happens afterwards when the
    response is re-validated against the pydantic model."""
    unsupported_keys = ("minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength")
    if isinstance(node, dict):
        for key in unsupported_keys:
            node.pop(key, None)
        for value in node.values():
            _strip_unsupported_strict_keywords(value)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported_strict_keywords(item)


def _build_tool() -> dict:
    schema = PredictedLabels.model_json_schema()
    schema.pop("title", None)
    schema["additionalProperties"] = False
    _strip_unsupported_strict_keywords(schema)
    return {
        "name": TOOL_NAME,
        "description": "Emit the predicted (subjective) labels for one track.",
        "input_schema": schema,
        "strict": True,
    }


def _track_prompt(item: TeacherInput, variant: str) -> str:
    lines = [
        f"Track: {item.track_name!r} by {item.artist_name!r}",
        f"Album: {item.album_name!r}" if item.album_name else "Album: unknown",
    ]
    if item.measured:
        lines.append(f"Measured audio features: {item.measured}")
    else:
        lines.append("Measured audio features: none available for this track")
    lines.append(f"Lyrics: {item.lyrics}" if item.lyrics else "Lyrics: not available")
    if variant == "alt_phrasing":
        lines.append(
            "\nDescribe how this song feels to a listener and what it's about, "
            "then call emit_labels with your assessment."
        )
    else:
        lines.append("\nCall emit_labels with your assessment of this track.")
    return "\n".join(lines)


def _load_cached(output_path: Path) -> dict[str, LabelRecord]:
    if not output_path.exists():
        return {}
    cached = {}
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = LabelRecord.model_validate_json(line)
            cached[record.track_id] = record
    return cached


def label_one(
    client: Anthropic, item: TeacherInput, model: str = DEFAULT_MODEL, variant: str = "default"
) -> tuple[PredictedLabels | None, int, int]:
    """Returns (labels_or_None, input_tokens, output_tokens). None means the
    model's tool call didn't validate against the schema on either attempt."""
    tool = _build_tool()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[{"role": "user", "content": _track_prompt(item, variant)}],
    )
    usage = response.usage
    for block in response.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            try:
                return PredictedLabels.model_validate(block.input), usage.input_tokens, usage.output_tokens
            except ValidationError:
                return None, usage.input_tokens, usage.output_tokens
    return None, usage.input_tokens, usage.output_tokens


def run_labelling(
    inputs: list[TeacherInput],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model: str = DEFAULT_MODEL,
    variant: str = "default",
) -> None:
    client = _build_client()
    cached = _load_cached(output_path)
    total_input_tokens = 0
    total_output_tokens = 0
    failures: list[str] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for i, item in enumerate(inputs):
            if item.track_id in cached:
                continue

            labels, in_tok, out_tok = label_one(client, item, model=model, variant=variant)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            if labels is None:
                failures.append(item.track_id)
                continue

            record = LabelRecord(
                track_id=item.track_id,
                input=item,
                labels=labels,
                teacher_model=model,
                prompt_variant=variant,
            )
            f.write(record.model_dump_json() + "\n")
            f.flush()

            if (i + 1) % BATCH_SIZE == 0:
                print(
                    f"  {i + 1}/{len(inputs)} labelled - running spend: "
                    f"{total_input_tokens:,} in / {total_output_tokens:,} out tokens"
                )

    print(f"Done. {len(failures)} failures: {failures[:10]}")
    print(f"Total: {total_input_tokens:,} input tokens, {total_output_tokens:,} output tokens")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--variant", type=str, default="default")
    args = parser.parse_args(argv)

    inputs = enrich_tracks(limit=args.limit)
    run_labelling(inputs, output_path=args.output, model=args.model, variant=args.variant)


if __name__ == "__main__":
    main()
