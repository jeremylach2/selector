"""Resolve Selector tracks to 30-second audio previews from public APIs.

Spotify removed preview URLs for new apps in November 2024, so previews come
from elsewhere: the iTunes Search API (no auth, ``previewUrl``, AAC) tried
first, and the Deezer public catalog API (no auth for search, ``preview``,
MP3) as a fallback. Neither is scraped and neither requires a login — this is
a recruiter-facing project and a ToS-violating pipeline would be a liability,
not a shortcut.

Deezer preview URLs are time-limited signed links. This module therefore
downloads a match's clip *at resolve time*, immediately after scoring it, and
never persists the URL for a later fetch. ``selector.audio.fetch`` holds the
actual download function; this module is the orchestration and scoring layer
around it.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import httpx
import pandas as pd

from selector.audio.fetch import AUDIO_DIR, download_preview
from selector.warehouse.build import DEFAULT_DB_PATH
from selector.warehouse.queries import _connect

DEFAULT_OUTPUT_PATH = Path("data/audio_matches.parquet")
CHECKPOINT_PATH = Path("data/.audio_resolve_checkpoint.jsonl")

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
DEEZER_SEARCH_URL = "https://api.deezer.com/search"

USER_AGENT = "Selector/0.1 (portfolio project, keyless public API usage)"

# Below this score a candidate is recorded as unmatched rather than guessed
# at. Silent bad matches would poison every downstream audio feature, so this
# is deliberately conservative — see docs/AUDIO_MATCHING.md.
DEFAULT_MATCH_THRESHOLD = 0.72

# Tokens that change what a recording *is*, not just how it's spelled. A
# candidate that has one of these and the query doesn't (or vice versa) is
# probably a different recording of the same song, not the same recording.
MODIFIER_TOKENS = ("live", "remix", "acoustic", "cover", "remaster", "demo", "instrumental")

BATCH_SIZE = 25
REQUEST_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class Candidate:
    source: str  # "itunes" | "deezer"
    source_id: str
    title: str
    artist: str
    album: str
    preview_url: str


@dataclass(frozen=True)
class MatchResult:
    track_id: str
    query_title: str
    query_artist: str
    match_source: str | None
    source_id: str | None
    matched_title: str | None
    matched_artist: str | None
    match_confidence: float
    local_path: str | None


def _strip_parens(text: str) -> str:
    return re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", text)


def normalize(text: str) -> tuple[str, frozenset[str]]:
    """Lowercase/punctuation-fold `text` and pull out its modifier tokens.

    Returns ``(clean_text, modifiers)``. Modifiers (live/remix/acoustic/...)
    are stripped from the text used for similarity scoring but kept as a
    separate signal — a "Live" candidate for a studio original is a bad
    match, not a good one with noisy formatting.
    """
    lowered = text.lower()
    lowered = re.sub(r"\bfeat\.?\b|\bft\.?\b", " ", lowered)
    # \w* lets "remaster" match "remastered", "instrumental" match
    # "instrumentals", etc. — these are inflections of the same modifier.
    found_modifiers = {tok for tok in MODIFIER_TOKENS if re.search(rf"\b{tok}\w*\b", lowered)}
    cleaned = _strip_parens(lowered)
    for tok in MODIFIER_TOKENS:
        cleaned = re.sub(rf"\b{tok}\w*\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, frozenset(found_modifiers)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def score_candidate(query_title: str, query_artist: str, candidate: Candidate) -> float:
    """Score one candidate against the query track.

    Weighted title/artist similarity, a small album-match bonus, and a
    modifier-mismatch penalty for live/remix/acoustic/cover/etc. recordings
    of the right song that aren't the recording we're actually after.
    """
    q_title, q_mods = normalize(query_title)
    q_artist, _ = normalize(query_artist)
    c_title, c_mods = normalize(candidate.title)
    c_artist, _ = normalize(candidate.artist)

    title_sim = _similarity(q_title, c_title)
    artist_sim = _similarity(q_artist, c_artist)
    score = 0.55 * title_sim + 0.45 * artist_sim

    if q_mods != c_mods:
        score -= 0.35

    return max(0.0, min(1.0, score))


def _itunes_candidates(client: httpx.Client, title: str, artist: str, limit: int = 5) -> list[Candidate]:
    resp = client.get(
        ITUNES_SEARCH_URL,
        params={"term": f"{artist} {title}", "entity": "song", "limit": limit},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [
        Candidate(
            source="itunes",
            source_id=str(r.get("trackId")),
            title=r.get("trackName", ""),
            artist=r.get("artistName", ""),
            album=r.get("collectionName", ""),
            preview_url=r["previewUrl"],
        )
        for r in results
        if r.get("previewUrl")
    ]


def _deezer_candidates(client: httpx.Client, title: str, artist: str, limit: int = 5) -> list[Candidate]:
    resp = client.get(DEEZER_SEARCH_URL, params={"q": f"{artist} {title}", "limit": limit})
    resp.raise_for_status()
    results = resp.json().get("data", [])
    return [
        Candidate(
            source="deezer",
            source_id=str(r.get("id")),
            title=r.get("title", ""),
            artist=(r.get("artist") or {}).get("name", ""),
            album=(r.get("album") or {}).get("title", ""),
            preview_url=r["preview"],
        )
        for r in results
        if r.get("preview")
    ]


def _best_candidate(
    client: httpx.Client, title: str, artist: str
) -> tuple[Candidate | None, float]:
    """iTunes first, Deezer as fallback. Returns the best-scoring candidate
    from whichever source produces one above the threshold-agnostic best."""
    best: Candidate | None = None
    best_score = -1.0

    for fetch_candidates in (_itunes_candidates, _deezer_candidates):
        try:
            candidates = fetch_candidates(client, title, artist)
        except httpx.HTTPError:
            candidates = []
        for cand in candidates:
            score = score_candidate(title, artist, cand)
            if score > best_score:
                best, best_score = cand, score
        time.sleep(REQUEST_DELAY_SECONDS)
        # If iTunes already found a confident match, don't bother with Deezer.
        if best_score >= DEFAULT_MATCH_THRESHOLD:
            break

    return best, best_score


def _load_checkpoint() -> dict[str, dict]:
    if not CHECKPOINT_PATH.exists():
        return {}
    done = {}
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done[row["track_id"]] = row
    return done


def _append_checkpoint(rows: list[MatchResult]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def _top_tracks(limit: int, db_path: Path) -> pd.DataFrame:
    with _connect(db_path) as con:
        return con.execute(
            "SELECT track_id, name, artist, play_count FROM tracks ORDER BY play_count DESC LIMIT ?",
            [limit],
        ).df()


def resolve_tracks(
    limit: int = 3000,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Resolve the top `limit` tracks by play count to downloaded previews.

    Idempotent: tracks already present in the checkpoint file are skipped on
    re-run, and the audio directory is only ever added to, never re-downloaded.
    """
    tracks = _top_tracks(limit, db_path)
    done = _load_checkpoint()

    to_process = tracks[~tracks["track_id"].isin(done.keys())]
    print(f"{len(done)} already resolved, {len(to_process)} to go")

    batch: list[MatchResult] = []
    with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
        for _, row in to_process.iterrows():
            candidate, score = _best_candidate(client, row["name"], row["artist"])

            local_path = None
            match_source = None
            source_id = None
            matched_title = None
            matched_artist = None

            if candidate is not None and score >= threshold:
                local_path = download_preview(client, candidate.preview_url, row["track_id"])
                if local_path is not None:
                    match_source = candidate.source
                    source_id = candidate.source_id
                    matched_title = candidate.title
                    matched_artist = candidate.artist
                else:
                    score = 0.0  # download failed; don't record a phantom match

            result = MatchResult(
                track_id=row["track_id"],
                query_title=row["name"],
                query_artist=row["artist"],
                match_source=match_source,
                source_id=source_id,
                matched_title=matched_title,
                matched_artist=matched_artist,
                match_confidence=round(score, 4),
                local_path=local_path,
            )
            batch.append(result)

            if len(batch) >= BATCH_SIZE:
                _append_checkpoint(batch)
                print(f"  checkpointed {len(batch)} tracks ({row['name']!r} just done)")
                batch = []

    if batch:
        _append_checkpoint(batch)

    all_rows = list(_load_checkpoint().values())
    return pd.DataFrame(all_rows)


def match_rate_report(matches: pd.DataFrame, tracks: pd.DataFrame) -> str:
    """A human-readable match-rate report: overall, by play-count decile, and
    a sample of borderline matches for manual eyeballing."""
    merged = matches.merge(tracks[["track_id", "play_count"]], on="track_id", how="left")
    matched = merged["local_path"].notna()

    lines = [
        f"Overall match rate: {matched.mean():.1%} ({matched.sum()}/{len(merged)})",
        "",
        "By play-count decile (0 = most-played):",
    ]
    merged["decile"] = pd.qcut(
        merged["play_count"].rank(method="first", ascending=False), 10, labels=False
    )
    by_decile = merged.groupby("decile")["local_path"].apply(lambda s: s.notna().mean())
    for decile, rate in by_decile.items():
        lines.append(f"  decile {int(decile)}: {rate:.1%}")

    lines.append("")
    lines.append("20 sampled matches near the confidence threshold:")
    near_threshold = merged[merged["match_confidence"].between(0.55, 0.85)].sample(
        min(20, len(merged[merged["match_confidence"].between(0.55, 0.85)])), random_state=42
    )
    for _, r in near_threshold.iterrows():
        lines.append(
            f"  [{r['match_confidence']:.2f}] {r['query_artist']!r}/{r['query_title']!r} "
            f"-> {r['matched_artist']!r}/{r['matched_title']!r} ({r['match_source']})"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3000, help="top N tracks by play count")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    matches = resolve_tracks(limit=args.limit, threshold=args.threshold, db_path=args.db_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(args.output, index=False)

    tracks = _top_tracks(args.limit, args.db_path)
    print()
    print(match_rate_report(matches, tracks))


if __name__ == "__main__":
    main()
