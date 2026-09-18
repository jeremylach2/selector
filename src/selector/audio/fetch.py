"""Download and cache audio preview clips.

Called by ``selector.audio.resolve`` immediately after a match is scored —
Deezer preview URLs are signed and time-limited, so a clip is downloaded the
moment it's found, never stored as a URL for a later fetch. Caching by
``track_id`` means a crash or a re-run only ever costs the batch in flight;
already-downloaded files are never re-fetched.
"""

from __future__ import annotations

from pathlib import Path

import httpx

AUDIO_DIR = Path("data/audio")

_EXTENSION_BY_CONTENT_TYPE = {
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
}


def _extension_for(url: str, content_type: str | None) -> str:
    if content_type:
        for prefix, ext in _EXTENSION_BY_CONTENT_TYPE.items():
            if content_type.startswith(prefix):
                return ext
    suffix = Path(url.split("?")[0]).suffix
    return suffix if suffix else ".mp3"


def existing_preview_path(track_id: str) -> Path | None:
    """Return the cached clip for `track_id`, if one is already on disk."""
    for ext in (".m4a", ".mp3"):
        candidate = AUDIO_DIR / f"{track_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def download_preview(client: httpx.Client, url: str, track_id: str) -> str | None:
    """Download `url` to `data/audio/{track_id}.{ext}`, skipping if cached.

    Returns the local path as a string, or None if the download failed —
    callers treat a failed download as "no match", not a fatal error, since
    one bad preview shouldn't stop resolution of the other ~3000 tracks.
    """
    cached = existing_preview_path(track_id)
    if cached is not None:
        return str(cached)

    try:
        with client.stream("GET", url, timeout=20.0) as response:
            response.raise_for_status()
            ext = _extension_for(url, response.headers.get("Content-Type"))
            dest = AUDIO_DIR / f"{track_id}{ext}"
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
            tmp.rename(dest)
            return str(dest)
    except httpx.HTTPError:
        return None
