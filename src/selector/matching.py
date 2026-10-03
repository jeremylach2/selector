"""Title/artist match scoring, shared by the audio-preview matcher
(`selector.audio.resolve`) and the playlist track resolver
(`selector.spotify.resolve`).

Standard library only, so the hosted MCP server can use it without pulling
in the audio pipeline.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Below this score a candidate is recorded as unmatched rather than guessed
# at. Silent bad matches would poison every downstream audio feature, so this
# is deliberately conservative, see docs/AUDIO_MATCHING.md.
DEFAULT_MATCH_THRESHOLD = 0.72

# Tokens that change what a recording *is*, not just how it's spelled. A
# candidate that has one of these and the query doesn't (or vice versa) is
# probably a different recording of the same song, not the same recording.
MODIFIER_TOKENS = ("live", "remix", "acoustic", "cover", "remaster", "demo", "instrumental")

MODIFIER_PENALTY = 0.35


def _strip_parens(text: str) -> str:
    return re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", text)


def normalize(text: str) -> tuple[str, frozenset[str]]:
    """Lowercase/punctuation-fold `text` and pull out its modifier tokens.

    Returns ``(clean_text, modifiers)``. Modifiers (live/remix/acoustic/...)
    are stripped from the text used for similarity scoring but kept as a
    separate signal: a "Live" candidate for a studio original is a bad
    match, not a good one with noisy formatting.
    """
    lowered = text.lower()
    lowered = re.sub(r"\bfeat\.?\b|\bft\.?\b", " ", lowered)
    # \w* lets "remaster" match "remastered", "instrumental" match
    # "instrumentals", etc. These are inflections of the same modifier.
    found_modifiers = {tok for tok in MODIFIER_TOKENS if re.search(rf"\b{tok}\w*\b", lowered)}
    cleaned = _strip_parens(lowered)
    for tok in MODIFIER_TOKENS:
        cleaned = re.sub(rf"\b{tok}\w*\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, frozenset(found_modifiers)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_score(
    query_title: str,
    query_artist: str,
    cand_title: str,
    cand_artist: str,
    ignore_modifiers: frozenset[str] = frozenset(),
) -> float:
    """Weighted title/artist similarity, minus a penalty when the two sides
    disagree on a modifier (live/remix/acoustic/cover/...): the right song
    in a recording that isn't the one asked for. Modifiers in
    `ignore_modifiers` don't count towards that disagreement."""
    q_title, q_mods = normalize(query_title)
    q_artist, _ = normalize(query_artist)
    c_title, c_mods = normalize(cand_title)
    c_artist, _ = normalize(cand_artist)

    score = 0.55 * similarity(q_title, c_title) + 0.45 * similarity(q_artist, c_artist)
    if q_mods - ignore_modifiers != c_mods - ignore_modifiers:
        score -= MODIFIER_PENALTY
    return max(0.0, min(1.0, score))
