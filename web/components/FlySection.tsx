"use client";

import { useMemo, useState } from "react";
import { type Catalog, moodsOf } from "@/lib/catalog";
import { type MushroomBody, moreLikeThis, type Overlap, percentileRank } from "@/lib/fly";
import type { History } from "@/lib/types";

type Props = {
  catalog: Catalog | null;
  catalogError: string | null;
  history: History;
  fly: { overlap: Overlap; mb: MushroomBody } | null;
};

const spotifyUrl = (id: string) => `https://open.spotify.com/track/${id}`;
const ordinal = (n: number) => {
  const s = n % 100 >= 11 && n % 100 <= 13 ? "th" : (["th", "st", "nd", "rd"][n % 10] ?? "th");
  return `${n}${s}`;
};

export default function FlySection({ catalog, catalogError, history, fly }: Props) {
  const [picked, setPicked] = useState<number | null>(null);
  const [query, setQuery] = useState("");

  // The visitor's most-played tracks that the fly has a fingerprint for.
  const seeds = useMemo(() => {
    if (!catalog || !fly) return [];
    return Array.from(history.plays.keys())
      .filter((t) => fly.overlap.catalogRow[t] >= 0)
      .sort((a, b) => history.plays[b] - history.plays[a])
      .slice(0, 8)
      .map((t) => fly.overlap.catalogRow[t]);
  }, [catalog, fly, history]);

  const playedRows = useMemo(() => {
    const m = new Map<number, number>();
    if (fly) fly.overlap.catalogRow.forEach((r, t) => r >= 0 && m.set(r, history.plays[t]));
    return m;
  }, [fly, history]);

  // The mushroom body's valence for every catalog track, as a percentile.
  const taste = useMemo(() => {
    if (!catalog || !fly || fly.mb.lessons === 0) return null;
    return percentileRank(Float64Array.from({ length: catalog.size }, (_, r) => fly.mb.valence(catalog, r)));
  }, [catalog, fly]);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!catalog || q.length < 2) return [];
    const out: number[] = [];
    for (let r = 0; r < catalog.size && out.length < 8; r++) {
      if (catalog.name[r].toLowerCase().includes(q) || catalog.artist[r].toLowerCase().includes(q)) out.push(r);
    }
    return out;
  }, [catalog, query]);

  const seed = picked ?? seeds[0] ?? null;
  const neighbours = useMemo(() => (catalog && seed !== null ? moreLikeThis(catalog, seed, 10) : []), [catalog, seed]);

  const o = fly?.overlap;
  return (
    <section className="block" aria-labelledby="fly-h">
      <h2 id="fly-h">More like this, chosen by a fly</h2>
      <p className="explain">
        Each track is described by 24 numbers (predicted mood and era, plus measured tempo, loudness and rhythm where a
        preview was found). They drive the fly&apos;s projection neurons, fan out through FlyWire&apos;s measured
        projection-neuron to Kenyon-cell wiring to 2,597 Kenyon cells, and winner-take-all keeps only the 130 that
        fire hardest. That sparse pattern is the track&apos;s fingerprint.
        &ldquo;More like this&rdquo; is the tracks whose fingerprints differ in the fewest cells: Hamming distance over
        biologically wired hash codes.
      </p>

      {!catalog && !catalogError && <p className="note">Loading the fly&apos;s catalog (19,386 fingerprints, ~1.3 MB)…</p>}
      {catalogError && <div className="error">Couldn&apos;t load the fly catalog: {catalogError}</div>}

      {catalog && o && (
        <>
          <p className="sub">
            {o.tracksMatched > 0 ? (
              <>
                The fly knows <b>{o.tracksMatched.toLocaleString()}</b> of these{" "}
                {history.trackIds.length.toLocaleString()} tracks ({Math.round((o.playsMatched / o.playsTotal) * 100)}% of
                plays), and replayed those {o.playsMatched.toLocaleString()} plays through its plasticity rule: every
                play-out depressed avoid synapses, every skip depressed approach synapses. It learned from{" "}
                {fly!.mb.lessons.toLocaleString()} of them.
              </>
            ) : (
              <>
                None of these tracks are in the fly&apos;s 19,386-track catalog, so it has nothing of yours to learn
                from. Search the catalog below to try it anyway.
              </>
            )}
          </p>

          {seeds.length > 0 && (
            <div className="chips" role="group" aria-label="Seed track">
              {seeds.map((r) => (
                <button key={r} className="chip" aria-pressed={r === seed} onClick={() => setPicked(r)} title={`${catalog.name[r]} · ${catalog.artist[r]}`}>
                  {catalog.name[r]}
                </button>
              ))}
            </div>
          )}
          <label className="visually-hidden" htmlFor="fly-search">
            Search the catalog for a seed track
          </label>
          <input
            id="fly-search"
            className="search"
            type="search"
            placeholder="Or search the catalog: a track or artist"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {results.length > 0 && (
            <div className="chips" role="group" aria-label="Search results">
              {results.map((r) => (
                <button
                  key={r}
                  className="chip"
                  aria-pressed={r === seed}
                  onClick={() => {
                    setPicked(r);
                    setQuery("");
                  }}
                >
                  {catalog.name[r]} · {catalog.artist[r]}
                </button>
              ))}
            </div>
          )}

          {seed !== null && (
            <div className="card">
              <h3>
                Nearest to &ldquo;{catalog.name[seed]}&rdquo; · {catalog.artist[seed]}
              </h3>
              <p className="note">
                {[
                  moodsOf(catalog, seed).join(", "),
                  catalog.era[seed] >= 0 ? catalog.eraVocab[catalog.era[seed]] : null,
                  catalog.tempo[seed] ? `${Math.round(catalog.tempo[seed]!)} BPM measured` : null,
                ]
                  .filter(Boolean)
                  .join(" · ")}
                . Shared cells out of {catalog.kActive}; lower Hamming distance is closer.
              </p>
              <ol className="neighbours">
                {neighbours.map((nb, i) => (
                  <li key={nb.row}>
                    <span className="n">{i + 1}</span>
                    <span className="t">
                      <a href={spotifyUrl(catalog.id[nb.row])} target="_blank" rel="noreferrer">
                        {catalog.name[nb.row]}
                        {playedRows.has(nb.row) ? (
                          <span className="badge">you&apos;ve played it</span>
                        ) : (
                          <span className="badge">new to you</span>
                        )}
                      </a>
                      <span>
                        {catalog.artist[nb.row]} · {moodsOf(catalog, nb.row).join(", ")}
                        {taste && ` · fly taste ${ordinal(Math.round(taste[nb.row] * 100))} pct`}
                      </span>
                    </span>
                    <span className="d">
                      Hamming {nb.distance}
                      <span className="overlap" title={`${nb.shared} of ${catalog.kActive} cells shared`}>
                        <i style={{ width: `${(nb.shared / catalog.kActive) * 100}%` }} />
                      </span>
                      <span style={{ fontSize: 12, color: "var(--muted)" }}>
                        {nb.shared}/{catalog.kActive} shared
                      </span>
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </>
      )}
    </section>
  );
}
