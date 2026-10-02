"""The DJ's crate as data, kept free of heavy imports so the hosted MCP
server can run the DJ too.

`selector.dj.pool.build_crate` assembles a crate from the local pipeline
outputs (measured audio, tagger labels, fly tags, warehouse stats) and
trains the mushroom body that scores it. That needs scipy, the fly brain
and the full per-play history, none of which the Vercel function has. So
the hosted server gets the *result* instead: `save_deploy_crate` writes
the finished crate to one Parquet file, the Blob store holds it next to
the deploy warehouse, and `load_deploy_crate` reads it back. Brief, Arc,
Select, Critique and Commit then run unchanged on either side.

Fly tags are binary and sparse (130 active Kenyon cells of 2,597), so they
are held as CSR index arrays and compared with numpy alone. `KCTags` gives
the same Hamming distances as `selector.fly.lsh.hamming_distances`.

The deploy file keeps only what the DJ reads, and no timestamps: recency
is already folded into the boolean `familiar` column. `deploy_crate_violations`
is checked on write and again before upload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DEPLOY_CRATE_PATH = Path("data/dj_crate_deploy.parquet")

# Everything Brief, Select and Commit read from a crate row, and nothing
# else. `kc` holds the track's active Kenyon cells; `tag_row` is implied by
# row order in the file.
DEPLOY_COLUMNS = (
    "track_id",
    "name",
    "artist",
    "tempo",
    "energy",
    "duration_ms",
    "familiar",
    "mood_tags",
    "fly_valence",
    "taste",
    "kc",
)
_METADATA_KEY = b"selector_dj_crate"


@dataclass(frozen=True)
class KCTags:
    """Binary Kenyon-cell tags as CSR index arrays: row `r` is active at
    `indices[indptr[r]:indptr[r + 1]]`."""

    indptr: np.ndarray
    indices: np.ndarray
    n_kc: int

    @classmethod
    def from_sparse(cls, matrix: Any) -> KCTags:
        """From a scipy sparse matrix, read by attribute so scipy itself is
        never imported here. Stored values are taken as 1."""
        csr = matrix.tocsr() if hasattr(matrix, "tocsr") else matrix
        return cls(np.asarray(csr.indptr, dtype=np.int64), np.asarray(csr.indices, dtype=np.int32), int(csr.shape[1]))

    @classmethod
    def from_rows(cls, rows: list[np.ndarray] | pd.Series, n_kc: int) -> KCTags:
        lengths = np.array([len(r) for r in rows], dtype=np.int64)
        indptr = np.concatenate([[0], np.cumsum(lengths)])
        indices = np.concatenate([np.asarray(r, dtype=np.int32) for r in rows]) if len(rows) else np.array([], np.int32)
        return cls(indptr, indices, n_kc)

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    @property
    def popcount(self) -> np.ndarray:
        return np.diff(self.indptr)

    def row(self, r: int) -> np.ndarray:
        return self.indices[self.indptr[r] : self.indptr[r + 1]]

    def take(self, rows: np.ndarray) -> KCTags:
        """A new tag set holding `rows`, in that order."""
        rows = np.asarray(rows, dtype=np.int64)
        starts = self.indptr[rows]
        lengths = self.indptr[rows + 1] - starts
        indptr = np.concatenate([[0], np.cumsum(lengths)])
        offsets = np.repeat(starts - indptr[:-1], lengths)
        return KCTags(indptr, self.indices[np.arange(indptr[-1]) + offsets], self.n_kc)

    def hamming(self, r: int) -> np.ndarray:
        """Hamming distance from row `r` to every row: |a| + |b| - 2|a & b|."""
        active = np.zeros(self.n_kc, dtype=bool)
        active[self.row(r)] = True
        hits = np.concatenate([[0], np.cumsum(active[self.indices], dtype=np.int64)])
        overlap = hits[self.indptr[1:]] - hits[self.indptr[:-1]]
        pop = self.popcount
        return pop + pop[r] - 2 * overlap


@dataclass
class Crate:
    """`tracks` is one row per playable track (see `selector.dj.pool`);
    `tags` holds the fly tags, indexed by `tracks.tag_row`. A scipy sparse
    matrix is accepted and converted."""

    tracks: pd.DataFrame
    tags: KCTags
    as_of: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.tags, KCTags):
            self.tags = KCTags.from_sparse(self.tags)

    def by_id(self) -> pd.DataFrame:
        return self.tracks.set_index("track_id", drop=False)


# -- the deploy file ----------------------------------------------------------


def _deploy_frame(crate: Crate) -> pd.DataFrame:
    rows = crate.tracks["tag_row"].to_numpy()
    frame = crate.tracks[[c for c in DEPLOY_COLUMNS if c != "kc"]].copy()
    frame["mood_tags"] = frame["mood_tags"].apply(list)
    frame["kc"] = [crate.tags.row(int(r)).astype(np.int16) for r in rows]
    return frame.reset_index(drop=True)


def deploy_crate_violations(schema: pa.Schema) -> list[str]:
    """Why a file with `schema` may not be shipped as the deploy crate:
    missing or unexpected columns, or any date/time column. Empty means
    it's safe."""
    names = set(schema.names)
    problems = [f"missing column {c}" for c in DEPLOY_COLUMNS if c not in names]
    problems += [f"unexpected column {c}" for c in sorted(names - set(DEPLOY_COLUMNS))]
    problems += [
        f"time column {field.name} ({field.type})"
        for field in schema
        if pa.types.is_temporal(field.type)
    ]
    return problems


def save_deploy_crate(crate: Crate, path: Path = DEFAULT_DEPLOY_CRATE_PATH) -> Path:
    table = pa.Table.from_pandas(_deploy_frame(crate), preserve_index=False)
    problems = deploy_crate_violations(table.schema)
    if problems:
        raise RuntimeError(f"refusing to write the deploy crate: {problems}")
    meta = {"as_of": crate.as_of.date().isoformat(), "n_kc": crate.tags.n_kc}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), _METADATA_KEY: json.dumps(meta)})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def load_deploy_crate(path: Path = DEFAULT_DEPLOY_CRATE_PATH) -> Crate:
    table = pq.read_table(path)
    problems = deploy_crate_violations(table.schema)
    if problems:
        raise RuntimeError(f"{path} is not a deploy crate: {problems}")
    meta = json.loads(table.schema.metadata[_METADATA_KEY])
    tracks = table.to_pandas()
    tags = KCTags.from_rows(tracks.pop("kc"), int(meta["n_kc"]))
    tracks["mood_tags"] = tracks["mood_tags"].apply(list)
    tracks["tag_row"] = np.arange(len(tracks))
    return Crate(tracks=tracks, tags=tags, as_of=datetime.fromisoformat(meta["as_of"]))
