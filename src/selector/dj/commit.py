"""Stage 5, Commit: liner notes, then (only if asked) the real playlist.

Liner notes are written for every run, dry or not. Each transition cites
the **measured** tempo and energy that justify it -- "tempo locks in at
122 BPM while energy lifts 0.48 -> 0.61 into the peak" -- plus the fly
brain's view (Kenyon-cell distance from the previous track, and whether
the mushroom body predicts approach).

The playlist write is gated twice: `dry_run` must be off, and the critique
verdict must have passed. A set that failed critique twice is never
committed, whatever the caller asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from selector.dj.arc import tempo_shift
from selector.dj.brief import Brief
from selector.dj.critique import Verdict
from selector.dj.select import Pick, Selection
from selector.spotify.client import SpotifyClient

# Spotify rejects playlist descriptions over 300 characters.
MAX_DESCRIPTION_CHARS = 300

# Fly tags have 130 active Kenyon cells; across the crate the median
# pairwise Hamming distance is ~186 and the 5th percentile ~114. Under this
# counts as a genuine fly-brain neighbour.
CLOSE_HAMMING = 120


class CommitRefused(RuntimeError):
    """Raised when asked to write a set whose critique did not pass."""


@dataclass
class CommitResult:
    dry_run: bool
    title: str
    description: str
    playlist_id: str | None = None
    playlist_url: str | None = None


def playlist_title(brief: Brief, now: datetime) -> str:
    return f"Selector DJ · {brief.theme.name.title()} · {now:%b %d}"


def playlist_description(brief: Brief) -> str:
    text = (
        f"{brief.theme.description} Picked with a fruit fly's olfactory circuit, wired from the "
        "FlyWire connectome, as a similarity hash over my listening history, shaped to a measured energy arc: opener, build, peak, comedown."
    )
    return text[:MAX_DESCRIPTION_CHARS]


def _tempo_phrase(a: Pick, b: Pick) -> str:
    shift = tempo_shift(a.tempo, b.tempo)
    if shift is None:
        return "tempo unmeasured on one side"
    # `tempo_shift` folds octaves; if the direct change is larger than the
    # folded one, the two tracks relate through half or double time.
    octave = abs(b.tempo - a.tempo) / a.tempo > shift + 0.01
    if octave:
        relation = "double" if b.tempo > a.tempo else "half"
        if shift < 0.04:
            return f"{b.tempo:.0f} BPM runs at {relation} time against {a.tempo:.0f}"
        return f"tempo moves {a.tempo:.0f} -> {b.tempo:.0f} BPM, near {relation} time ({shift:.0%} off)"
    if shift < 0.04:
        return f"tempo locks in at ~{b.tempo:.0f} BPM"
    verb = "pushes" if b.tempo > a.tempo else "drops"
    return f"tempo {verb} {a.tempo:.0f} -> {b.tempo:.0f} BPM"


def _energy_phrase(a: Pick, b: Pick) -> str:
    step = b.energy - a.energy
    if step > 0.05:
        return f"energy lifts {a.energy:.2f} -> {b.energy:.2f}"
    if step < -0.05:
        return f"energy eases {a.energy:.2f} -> {b.energy:.2f}"
    return f"energy holds at {b.energy:.2f}"


def _fly_phrase(b: Pick) -> str:
    parts = []
    if b.hamming_to_prev is not None and b.hamming_to_prev < CLOSE_HAMMING:
        parts.append(f"fly-brain neighbour of the last track (Hamming {b.hamming_to_prev})")
    taste = b.scores["taste"]
    if taste >= 0.5:
        parts.append(f"fly taste score in the top {max(1, round((1 - taste) * 100))}% of the crate")
    else:
        parts.append(f"a taste risk: fly ranks it in the bottom {max(1, round(taste * 100))}%")
    parts.append("in current rotation" if b.familiar else "back from the vault")
    return "; ".join(parts)


def _sentence(text: str) -> str:
    """Upper-case the first letter only (`str.capitalize` would also lower-
    case "BPM")."""
    return text[:1].upper() + text[1:]


def transition_note(a: Pick, b: Pick) -> str:
    into = f", into the {b.phase}" if b.phase != a.phase else ""
    return f"{_sentence(_tempo_phrase(a, b))} while {_energy_phrase(a, b)}{into}. {_sentence(_fly_phrase(b))}."


def _fmt_tempo(p: Pick) -> str:
    return f"{p.tempo:.0f} BPM" if p.tempo is not None else "tempo n/a"


def liner_notes(brief: Brief, selection: Selection, verdict: Verdict, title: str) -> str:
    picks = selection.picks
    lines = [
        f"# {title}",
        "",
        f"_{brief.theme.description}_",
        "",
        f"**Brief.** {brief.rationale}",
        "",
        (
            f"**Arc.** {selection.seconds / 60:.0f} minutes, opener -> build -> peak -> comedown, "
            f"energy {brief.theme.energy_floor:.2f}-{brief.theme.energy_ceiling:.2f}. "
            "Energy and tempo are measured from 30-second audio previews, not predicted."
        ),
        "",
        f"**Critique.** {verdict.summary}",
        "",
    ]
    for i, p in enumerate(picks):
        clock = f"{int(p.start_s // 60)}:{int(p.start_s % 60):02d}"
        lines.append(
            f"{i + 1}. **{p.name}** - {p.artist}  `{clock}` · {p.phase} · "
            f"{_fmt_tempo(p)} · energy {p.energy:.2f}"
        )
        if i == 0:
            lines.append(
                f"   Opens at energy {p.energy:.2f} (target {p.target_energy:.2f}), "
                f"{_fmt_tempo(p)}. {_sentence(_fly_phrase(p))}."
            )
        else:
            lines.append(f"   {transition_note(picks[i - 1], p)}")
    return "\n".join(lines) + "\n"


def commit(
    brief: Brief,
    selection: Selection,
    verdict: Verdict,
    now: datetime,
    client: SpotifyClient | None,
    dry_run: bool = True,
) -> CommitResult:
    title = playlist_title(brief, now)
    description = playlist_description(brief)
    if dry_run:
        return CommitResult(dry_run=True, title=title, description=description)
    if not verdict.passed:
        raise CommitRefused(f"Critique did not pass, so nothing was written to Spotify. {verdict.summary}")
    if client is None:
        raise CommitRefused("No Spotify client configured (SPOTIFY_CLIENT_ID unset).")

    uris = [f"spotify:track:{tid}" for tid in selection.track_ids]
    playlist = client.create_playlist(title, description=description, public=False, track_uris=uris)
    return CommitResult(
        dry_run=False,
        title=title,
        description=description,
        playlist_id=playlist.get("id"),
        playlist_url=(playlist.get("external_urls") or {}).get("spotify"),
    )
