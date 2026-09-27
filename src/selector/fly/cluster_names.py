"""Optional frontier-model names for taste clusters, layered over the built
names from `selector.fly.clusters`.

The hybrid naming decision in the Wrapped extension plan: every cluster
always has a name built from its features, which is what the browser demo
shows. For your own report, this module makes one call per cluster, given
the built name plus the cluster's ten most-played tracks, and caches the
result in `data/cluster_names.json` keyed by the clustering's cache key.

`selector.warehouse.wrapped` only ever *reads* that cache, so building a
report never touches the network. A cached name is used only while the
built name it was written from is unchanged; if the naming rules or the
clusters move, the stale entry is ignored until this is re-run.

Requires `ANTHROPIC_API_KEY`. About 12 calls of ~500 input and ~10 output
tokens, i.e. well under $0.05 per run at Sonnet 5 rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selector.fly.clusters import cluster_summary, load_or_fit_clusters, track_feature_frame
from selector.tagger.label import DEFAULT_MODEL, _build_client
from selector.warehouse import queries
from selector.warehouse.build import DEFAULT_DB_PATH

NAMES_CACHE_PATH = Path("data/cluster_names.json")

SYSTEM_PROMPT = """You name clusters of songs from one person's listening history, for a
Spotify-Wrapped-style report. Each cluster comes with a feature summary (mood tags,
energy, era) and its most-played tracks. Reply with only the name: 2 to 5 words,
evocative but faithful to the features and the tracks listed. No quotes, no emoji,
no genre the listed artists don't support."""


def load_cached_names(cache_key: str, path: Path = NAMES_CACHE_PATH) -> dict[str, dict[str, str]]:
    """`{cluster_index: {"name", "built_name"}}` for this clustering, or `{}`."""
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8")).get(cache_key, {}).get("names", {})


def llm_name_for(cluster: int, built: str, cache_key: str, path: Path = NAMES_CACHE_PATH) -> str | None:
    """The cached model-written name for one cluster, if it was written from
    the same built name the cluster has now."""
    entry = load_cached_names(cache_key, path).get(str(cluster))
    if entry and entry.get("built_name") == built:
        return entry["name"]
    return None


def _prompt(row: pd.Series, tracks: pd.DataFrame) -> str:
    by_id = tracks.set_index("track_id")
    lines = [
        f"Feature summary: {row['built_name']}",
        f"Share of listening: {row['play_share']:.0%}",
        "Most-played tracks:",
    ]
    for tid in row["exemplars"]:
        if tid in by_id.index:
            t = by_id.loc[tid]
            lines.append(f"- {t['name']} — {t['artist']}")
    return "\n".join(lines)


def name_clusters(model: str = DEFAULT_MODEL, path: Path = NAMES_CACHE_PATH) -> dict[str, dict[str, str]]:
    clusters = load_or_fit_clusters()
    with queries._connect(DEFAULT_DB_PATH) as con:
        tracks = con.execute("SELECT track_id, name, artist, play_count FROM tracks").df()
    summary = cluster_summary(clusters, tracks, track_feature_frame())

    client = _build_client()
    names: dict[str, dict[str, str]] = {}
    in_tok = out_tok = 0
    for _, row in summary.iterrows():
        response = client.messages.create(
            model=model,
            max_tokens=40,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _prompt(row, tracks)}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip().strip('"')
        in_tok += response.usage.input_tokens
        out_tok += response.usage.output_tokens
        names[str(row["cluster"])] = {"name": text, "built_name": row["built_name"]}
        print(f"  {row['built_name']}  ->  {text}")

    cache = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    cache[clusters.cache_key] = {"model": model, "names": names}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{in_tok:,} input + {out_tok:,} output tokens. Wrote {path}.")
    return names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    name_clusters(model=args.model)


if __name__ == "__main__":
    main()
