"""Extract audio DSP features from downloaded preview clips, using librosa.

Runs natively on Windows — the other half of the feature stack
(``features_essentia``) needs WSL2 because ``essentia-tensorflow`` ships no
Windows wheels.

Honesty constraint: every feature here is computed on a **30-second
excerpt**, not the full track. Tempo, spectral descriptors, and MFCCs are
reliable measurements of a clip. Song structure and arrangement are not
observable from 30 seconds and are never claimed here or downstream.
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import librosa
import numpy as np
import pandas as pd

DEFAULT_MATCHES_PATH = Path("data/audio_matches.parquet")
DEFAULT_OUTPUT_PATH = Path("data/features_librosa.parquet")

SAMPLE_RATE = 22050  # librosa default; plenty for a 30s excerpt's spectral content


def _mean_std(values: np.ndarray, name: str) -> dict[str, float]:
    return {f"{name}_mean": float(np.mean(values)), f"{name}_std": float(np.std(values))}


def _load_via_pyav(path: str, sr: int) -> np.ndarray:
    """Decode with PyAV's bundled ffmpeg libraries.

    libsndfile (soundfile's backend, and librosa's default loader) has no
    AAC/M4A decoder at all — a system ffmpeg install or a wheel that bundles
    one is required. PyAV ships static ffmpeg libs, so this needs no
    external binary on the machine running it. iTunes previews are AAC-in-M4A;
    Deezer's are plain MP3, which libsndfile decodes natively.
    """
    container = av.open(path)
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sr)
    chunks = [
        rframe.to_ndarray()
        for frame in container.decode(container.streams.audio[0])
        for rframe in resampler.resample(frame)
    ]
    container.close()
    if not chunks:
        return np.array([], dtype="float32")
    return np.concatenate(chunks, axis=1).flatten().astype("float32")


def _load_clip(path: str, sr: int) -> np.ndarray:
    """Load an audio clip to a mono float array at `sr`, trying librosa's
    native (libsndfile) path first and falling back to PyAV for formats
    libsndfile can't decode (notably M4A/AAC)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # librosa is chatty about short/quiet clips
        try:
            y, _ = librosa.load(path, sr=sr, mono=True)
            return y
        except Exception:  # noqa: BLE001 - soundfile raises its own error type per backend; any failure means "try PyAV next"
            return _load_via_pyav(path, sr)


def extract_features(path: str) -> dict[str, float] | None:
    """Compute one row of DSP features for a single clip. Returns None on
    a decode failure so one bad file doesn't kill the whole batch."""
    try:
        y = _load_clip(path, SAMPLE_RATE)
    except Exception:  # noqa: BLE001 - PyAV can raise its own decode errors on a genuinely corrupt file
        return None
    sr = SAMPLE_RATE

    if y.size == 0:
        return None

    features: dict[str, float] = {}

    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(np.asarray(tempo).item())
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features["beat_strength_mean"] = float(np.mean(onset_env)) if onset_env.size else 0.0
    features["onset_density"] = float(len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    features.update(_mean_std(spectral_centroid, "spectral_centroid"))
    features.update(_mean_std(spectral_rolloff, "spectral_rolloff"))
    features.update(_mean_std(spectral_bandwidth, "spectral_bandwidth"))
    features.update(_mean_std(spectral_flatness, "spectral_flatness"))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(13):
        features.update(_mean_std(mfcc[i], f"mfcc{i}"))
        features.update(_mean_std(mfcc_delta[i], f"mfcc{i}_delta"))

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    features["estimated_key"] = int(np.argmax(chroma_mean))
    features["key_confidence"] = float(chroma_mean.max() / (chroma_mean.sum() + 1e-9))

    rms = librosa.feature.rms(y=y)[0]
    features.update(_mean_std(rms, "rms"))
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features.update(_mean_std(zcr, "zcr"))

    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.sum(harmonic**2))
    percussive_energy = float(np.sum(percussive**2))
    features["harmonic_percussive_ratio"] = harmonic_energy / (percussive_energy + 1e-9)

    return features


def _extract_one(track_id: str, path: str) -> tuple[str, dict[str, float] | None]:
    return track_id, extract_features(path)


def extract_all(matches: pd.DataFrame, max_workers: int | None = None) -> pd.DataFrame:
    """Extract features for every matched track, in parallel across cores.

    Embarrassingly parallel — one clip's decode + feature extraction doesn't
    depend on any other's — so this uses a process pool and should take
    minutes, not hours, even at the full ~3,000-track scope.
    """
    matched = matches[matches["local_path"].notna()]
    rows: list[dict] = []
    failures: list[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_extract_one, r["track_id"], r["local_path"]): r["track_id"]
            for _, r in matched.iterrows()
        }
        for future in as_completed(futures):
            track_id, feats = future.result()
            if feats is None:
                failures.append(track_id)
                continue
            feats["track_id"] = track_id
            rows.append(feats)

    if failures:
        print(f"{len(failures)} clips failed to decode: {failures[:10]}{'...' if len(failures) > 10 else ''}")

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    matches = pd.read_parquet(args.matches)
    features = extract_all(matches, max_workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(f"Wrote {len(features):,} rows of librosa features to {args.output}")


if __name__ == "__main__":
    main()
