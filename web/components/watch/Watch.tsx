"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type Catalog, loadCatalog, moodsOf } from "@/lib/catalog";
import { type Circuit, type Firing, fire, loadCircuit } from "@/lib/circuit";
import { tempoShift } from "@/lib/dj";
import { cloneMushroomBody, matchHistory, moreLikeThis, type MushroomBody, type Neighbour, trainMushroomBody } from "@/lib/fly";
import { getSessionHistory } from "@/lib/session";
import type { History, WorkerOut } from "@/lib/types";
import { type Palette, usePalette, useReducedMotion } from "./palette";
import Stage from "./Stage";
import Synapses from "./Synapses";
import TagChip from "./TagChip";
import { BEATS, clamp01, ease, MATCH, position, STARTS, TOTAL, VERDICT } from "./timeline";

const SAMPLE_URL = "/sample/sample_spotify_data.zip";
const SAMPLE_TZ = "America/Chicago";
const SHORTLIST = 8;
const N_SEEDS = 6;
const ROW_H = 68;
const LESSON_MS = 1600;

type Candidate = Neighbour & { sharedCells: Uint16Array };
type Score = { approach: number; avoid: number; valence: number };
type Round = { scores: Score[]; winner: number };
type Lesson = { row: number; mbon: "approach" | "avoid"; before: Map<number, number>; progress: number };
type Taught = { text: string; from: number; changed: boolean | null };
type Listener = { history: History; source: "sample" | "own" };

function cellsOf(cat: Catalog, row: number): Uint16Array {
  return cat.active.subarray(row * cat.kActive, row * cat.kActive + cat.kActive);
}

function shortlist(cat: Catalog, seed: number): Candidate[] {
  const q = new Set(cellsOf(cat, seed));
  return moreLikeThis(cat, seed, SHORTLIST).map((nb) => ({ ...nb, sharedCells: cellsOf(cat, nb.row).filter((i) => q.has(i)) }));
}

// The MBON readout for every shortlisted track, exactly `MushroomBody.valence`
// split into its two halves. Highest net valence (approach minus avoid)
// wins; a tie goes to the closer track.
function judge(cat: Catalog, mb: MushroomBody, cands: Candidate[]): Round {
  const scores = cands.map((c) => {
    let approach = 0;
    let avoid = 0;
    for (const i of cellsOf(cat, c.row)) {
      approach += mb.wApproach[i];
      avoid += mb.wAvoid[i];
    }
    return { approach, avoid, valence: approach - avoid };
  });
  let winner = 0;
  scores.forEach((s, i) => {
    if (s.valence > scores[winner].valence) winner = i;
  });
  return { scores, winner };
}

function linerNote(cat: Catalog, seed: number, cands: Candidate[], round: Round): string {
  const c = cands[round.winner];
  const parts = [`Follows “${cat.name[seed]}” at Hamming ${c.distance}: ${c.shared} of ${cat.kActive} Kenyon cells in common.`];
  const [ta, tb, ea, eb] = [cat.tempo[seed], cat.tempo[c.row], cat.energy[seed], cat.energy[c.row]];
  const shift = tempoShift(ta, tb);
  const measured: string[] = [];
  if (shift !== null && ta && tb) measured.push(shift < 0.04 ? `tempo locks in at ~${tb.toFixed(0)} BPM` : `tempo moves ${ta.toFixed(0)} → ${tb.toFixed(0)} BPM`);
  if (ea !== null && eb !== null) measured.push(`energy ${eb > ea + 0.05 ? "lifts" : eb < ea - 0.05 ? "eases" : "holds"} ${ea.toFixed(2)} → ${eb.toFixed(2)}`);
  if (measured.length) parts.push(`Measured from the previews: ${measured.join(", ")}.`);
  const closer = cands.filter((x) => x.distance < c.distance).length;
  const s = round.scores[round.winner];
  parts.push(
    `The approach MBON outvotes avoid by ${s.valence.toFixed(1)}, the strongest net valence on the shortlist` +
      (closer ? `, enough to beat ${closer} closer ${closer === 1 ? "match" : "matches"}.` : "."),
  );
  return parts.join(" ");
}

export default function Watch() {
  const palette = usePalette();
  const reduced = useReducedMotion();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [circuit, setCircuit] = useState<Circuit | null>(null);
  const [listener, setListener] = useState<Listener | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([loadCatalog(), loadCircuit()]).then(
      ([cat, circ]) => {
        setCatalog(cat);
        setCircuit(circ);
      },
      (e) => setError(String(e?.message ?? e)),
    );
    // A visitor who already dropped their export on the landing page gets
    // their own fly; everyone else gets the sample listener.
    const own = getSessionHistory();
    if (own) {
      setListener(own);
      return;
    }
    const worker = new Worker(new URL("../../lib/parse.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<WorkerOut>) => {
      if (e.data.type === "done") {
        setListener({ history: e.data.history, source: "sample" });
        worker.terminate();
      } else if (e.data.type === "error") {
        setError(e.data.message);
        worker.terminate();
      }
    };
    worker.postMessage({ source: { kind: "url", url: SAMPLE_URL }, timeZone: SAMPLE_TZ });
    return () => worker.terminate();
  }, []);

  // The mushroom body, trained in this tab on every play, in order.
  const trained = useMemo(() => {
    if (!catalog || !listener) return null;
    const overlap = matchHistory(catalog, listener.history);
    return { overlap, mb: trainMushroomBody(catalog, listener.history, overlap) };
  }, [catalog, listener]);

  // Seeds: the listener's most-played tracks the fly has fingerprints for.
  const seeds = useMemo(() => {
    if (!trained || !listener) return [];
    const h = listener.history;
    return Array.from(h.plays.keys())
      .filter((t) => trained.overlap.catalogRow[t] >= 0)
      .sort((a, b) => h.plays[b] - h.plays[a])
      .slice(0, N_SEEDS)
      .map((t) => trained.overlap.catalogRow[t]);
  }, [trained, listener]);

  // Open on the first seed where one skip is enough to change the fly's
  // mind, found by running the real rule on a scratch copy. On other seeds
  // it can take two or three skips, and the page says so when it does.
  const defaultSeed = useMemo(() => {
    if (!catalog || !trained || !seeds.length) return null;
    for (const s of seeds) {
      const cands = shortlist(catalog, s);
      const before = judge(catalog, trained.mb, cands);
      const mb = cloneMushroomBody(trained.mb);
      mb.learn(catalog, cands[before.winner].row, -1);
      if (judge(catalog, mb, cands).winner !== before.winner) return s;
    }
    return seeds[0];
  }, [catalog, trained, seeds]);

  const [picked, setPicked] = useState<number | null>(null);
  const seed = picked ?? defaultSeed;

  if (error)
    return (
      <div className="error" role="alert">
        Couldn&apos;t start the circuit: {error}
      </div>
    );
  if (!catalog || !circuit || !palette || !trained || !listener || seed === null)
    return (
      <div className="theatre loading" aria-live="polite">
        <span className="spinner" aria-hidden="true" />
        <p>
          Loading the FlyWire wiring{circuit ? " ✓" : "…"} · 19,386 fingerprints{catalog ? " ✓" : "…"} ·{" "}
          {listener?.source === "own" ? "your export" : "the sample listener's plays"}
          {listener ? " ✓" : "…"}
        </p>
      </div>
    );

  return (
    <Theatre
      key={seed}
      cat={catalog}
      circuit={circuit}
      palette={palette}
      reduced={reduced}
      seed={seed}
      seeds={seeds}
      onSeed={setPicked}
      baseMb={trained.mb}
      source={listener.source}
      plays={listener.history.seqTrack.length}
    />
  );
}

type TheatreProps = {
  cat: Catalog;
  circuit: Circuit;
  palette: Palette;
  reduced: boolean;
  seed: number;
  seeds: number[];
  onSeed: (row: number) => void;
  baseMb: MushroomBody;
  source: "sample" | "own";
  plays: number;
};

function Theatre({ cat, circuit, palette, reduced, seed, seeds, onSeed, baseMb, source, plays }: TheatreProps) {
  const firing = useMemo(() => fire(cat, circuit, seed), [cat, circuit, seed]);
  const cands = useMemo(() => shortlist(cat, seed), [cat, seed]);
  // The working copy the visitor teaches; the trained one stays pristine
  // for the reset button.
  const mb = useRef<MushroomBody>(cloneMushroomBody(baseMb));
  const [round, setRound] = useState<Round>(() => judge(cat, mb.current, cands));
  const [taught, setTaught] = useState<Taught[]>([]);
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [teaching, setTeaching] = useState(false);

  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.25, (now - last) / 1000);
      last = now;
      setT((x) => Math.min(TOTAL, x + dt));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);
  useEffect(() => {
    if (t >= TOTAL && playing) setPlaying(false);
  }, [t, playing]);

  // Reduced motion: the same sequence and timing, but each beat is drawn
  // only in its finished state, so it advances in steps instead of moving.
  const pos = position(t);
  const shownT = reduced ? Math.min(TOTAL, STARTS[pos.beat] + BEATS[pos.beat].dur) - 1e-6 : t;
  const { beat, u, local } = position(shownT);
  const done = t >= TOTAL;

  const step = () => {
    setPlaying(false);
    setT(pos.beat + 1 < BEATS.length ? STARTS[pos.beat + 1] : TOTAL);
  };

  const teach = useCallback(
    (verdict: 1 | -1) => {
      const row = cands[round.winner].row;
      const which = verdict < 0 ? "approach" : "avoid";
      const w = which === "approach" ? mb.current.wApproach : mb.current.wAvoid;
      const cells = cellsOf(cat, row);
      const before = new Map<number, number>();
      let driveBefore = 0;
      for (const i of cells) {
        before.set(i, w[i]);
        driveBefore += w[i];
      }
      // The real rule, `MushroomBody.learn`: depress this track's synapses
      // onto one MBON, then let every synapse decay towards naive.
      mb.current.learn(cat, row, verdict);
      let driveAfter = 0;
      for (const i of cells) driveAfter += w[i];

      setLesson({ row, mbon: which, before, progress: reduced ? 1 : 0 });
      setTeaching(true);
      setTaught((h) => [
        ...h,
        {
          from: row,
          changed: null,
          text:
            `${verdict < 0 ? "Skipped" : "Played out"} “${cat.name[row]}”: its ${cells.length} ${which} synapses depressed ` +
            `${Math.round(mb.current.lr * 100)}%, then every synapse relaxed ${Math.round(mb.current.decay * 100)}% back ` +
            `toward naive. ${which === "approach" ? "Approach" : "Avoid"} drive for it: ${driveBefore.toFixed(1)} → ${driveAfter.toFixed(1)}.`,
        },
      ]);

      const finish = () => {
        const next = judge(cat, mb.current, cands);
        setRound(next);
        setTaught((h) => h.map((e, i) => (i === h.length - 1 ? { ...e, changed: cands[next.winner].row !== row } : e)));
        setTeaching(false);
        setT(STARTS[MATCH]);
        setPlaying(true);
      };
      if (reduced) {
        setTimeout(finish, 800);
        return;
      }
      const started = performance.now();
      const tick = (now: number) => {
        const p = clamp01((now - started) / LESSON_MS);
        setLesson((x) => x && { ...x, progress: ease(p) });
        if (p < 1) requestAnimationFrame(tick);
        else setTimeout(finish, 400);
      };
      requestAnimationFrame(tick);
    },
    [cat, cands, round.winner, reduced],
  );

  const reset = () => {
    mb.current = cloneMushroomBody(baseMb);
    setRound(judge(cat, mb.current, cands));
    setTaught([]);
    setLesson(null);
    setT(STARTS[MATCH]);
    setPlaying(true);
  };

  const k = cat.kActive;
  const nPred = circuit.columns.filter((c) => c.kind === "predicted").length;
  const hasAudio = circuit.measured.has(seed);
  const activePn = firing.pn.filter((v) => v > 0).length;
  const edgesShown = useMemo(() => {
    let n = 0;
    for (let jj = 0; jj < circuit.indices.length; jj++) if (firing.pn[circuit.indices[jj]] > 0) n++;
    return n;
  }, [circuit, firing]);
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

  const explain = [
    <>
      “{cat.name[seed]}” enters the circuit as {circuit.dIn} numbers: {nPred} predicted by the fine-tuned vibe tagger and{" "}
      {circuit.dIn - nPred} measured from a 30-second preview
      {hasAudio ? "." : ". No preview matched this track, so its measured channels are zero."}
    </>,
    <>
      Each number is centred and scaled by the catalog&apos;s statistics, then rectified: {activePn} of {circuit.dIn} channels fire.
      In the fly these are the {circuit.nPnReal} antennal-lobe projection neurons of the {circuit.hemisphere} hemisphere, pooled
      here into {circuit.dIn} channels, one per input number.
    </>,
    <>
      Only wiring that leaves a firing channel is drawn: {edgesShown.toLocaleString()} of {circuit.indices.length.toLocaleString()}{" "}
      connections, each weighted by synapse counts measured in the FlyWire connectome (v{circuit.flywire}). Not a random matrix.
    </>,
    <>
      {firing.driven.toLocaleString()} of {circuit.nKc.toLocaleString()} Kenyon cells get some drive ({pct(firing.driven / circuit.nKc)}).
      Far too many to tell one song from another.
    </>,
    <>
      The APL neuron inhibits every Kenyon cell at once, so only the most strongly driven keep firing. The bar rises until {k}{" "}
      are left: the top {pct(k / circuit.nKc)}.
    </>,
    <>
      What survives is the track&apos;s fingerprint: {k} bits on out of {circuit.nKc.toLocaleString()}.{" "}
      {firing.matchesShipped
        ? "Recomputed here from the wiring, it matches the precomputed tag cell for cell."
        : "Recomputed here, it does not match the precomputed tag."}
      {firing.tied > 0 &&
        ` ${firing.tied} cells tied exactly at the cutoff for the last ${firing.tiedSlots} places; numpy settled that tie offline, so it is taken from the precomputed tag.`}
    </>,
    <>
      Similar tracks light similar cells. Hamming distance counts the cells two fingerprints don&apos;t share, 2 × ({k} − shared), and
      these are the {SHORTLIST} nearest of {cat.size.toLocaleString()} catalog tracks.
    </>,
    <>
      Two mushroom-body output neurons, approach and avoid, read each fingerprint through plastic synapses trained in this tab on{" "}
      {source === "sample" ? "the sample listener's" : "your"} {plays.toLocaleString()} plays: play-outs depress avoid synapses, skips
      depress approach. Highest net valence wins.
    </>,
  ];

  // Beat 7: shared cells light one by one, distances count down from
  // 2 × k, and the rows re-sort as they resolve.
  const matchLocal = beat === MATCH ? local : beat > MATCH ? Infinity : -Infinity;
  const reveal = (i: number) => (reduced ? (beat >= MATCH ? 1 : 0) : clamp01((matchLocal - 0.3 * i - 0.6) / 3));
  const live = cands.map((c, i) => {
    const shown = Math.floor(c.shared * reveal(i));
    return { i, shown, distance: 2 * (k - shown) };
  });
  const order = [...live].sort((a, b) => a.distance - b.distance || cands[a.i].distance - cands[b.i].distance || a.i - b.i);
  const rankOf = new Map(order.map((r, rank) => [r.i, rank]));

  const verdictU = beat === VERDICT ? u : beat > VERDICT ? 1 : 0;
  const winner = cands[round.winner];
  const note = linerNote(cat, seed, cands, round);
  const typed = reduced || done ? note.length : Math.floor(note.length * clamp01((verdictU - 0.55) / 0.4));
  // After a lesson the deck stays put while the query re-runs, so the page
  // doesn't jump, but it doesn't name the new pick until the MBONs have.
  const resolved = verdictU > 0.5 || done;
  const deckVisible = resolved || taught.length > 0;
  const maxDrive = Math.max(...round.scores.flatMap((s) => [s.approach, s.avoid]), 1);
  const last = taught.length ? taught[taught.length - 1] : null;

  return (
    <>
      <div className="theatre">
        <div className="seed">
          <TagChip active={cellsOf(cat, seed)} nKc={circuit.nKc} palette={palette} size={48} label="The seed track's fingerprint" />
          <div>
            <p className="eyebrow">Seed · {source === "sample" ? "the sample listener's" : "your"} heavy rotation</p>
            <h2>{cat.name[seed]}</h2>
            <p className="meta">
              {cat.artist[seed]} · {[...moodsOf(cat, seed), cat.era[seed] >= 0 ? cat.eraVocab[cat.era[seed]] : null].filter(Boolean).join(", ")}
            </p>
          </div>
        </div>

        <div className="beat-caption" aria-live="polite">
          <span className="beat-n">
            {beat + 1}/{BEATS.length}
          </span>
          <b>{BEATS[beat].title}</b>
          <span className="alias">/ {BEATS[beat].alias}</span>
        </div>
        <p className="beat-explain">{explain[beat]}</p>

        {beat <= 2 && <FeatureBars circuit={circuit} firing={firing} lit={beat === 0 ? ease(u * 1.3) : 1} />}
        {beat <= 5 && <Stage circuit={circuit} firing={firing} kActive={k} beat={beat} u={u} palette={palette} />}

        {beat >= MATCH && (
          <ol className="matches" style={{ height: cands.length * ROW_H }} aria-label="Shortlist, nearest first">
            {cands.map((c, i) => {
              const r = live[i];
              const s = round.scores[i];
              const bars = clamp01(verdictU / 0.5);
              const isWinner = i === round.winner && verdictU > 0.5;
              const visible = reduced || matchLocal > 0.3 * i;
              return (
                <li
                  key={c.row}
                  className={`match${isWinner ? " winner" : ""}${reduced ? " still" : ""}`}
                  style={{ top: rankOf.get(i)! * ROW_H, opacity: visible ? 1 : 0, transform: visible ? "none" : "translateX(24px)" }}
                >
                  <TagChip
                    active={cellsOf(cat, c.row)}
                    nKc={circuit.nKc}
                    palette={palette}
                    size={52}
                    shared={c.sharedCells}
                    sharedShown={r.shown}
                    label={`${r.shown} cells shared with the seed`}
                  />
                  <div className="m-text">
                    <b>{cat.name[c.row]}</b>
                    <span>{cat.artist[c.row]}</span>
                    {bars > 0 && (
                      <span className="mbon" title={`approach ${s.approach.toFixed(1)}, avoid ${s.avoid.toFixed(1)}`}>
                        <i className="ap" style={{ width: `${((s.approach / maxDrive) * 50 * bars).toFixed(1)}%` }} />
                        <i className="av" style={{ width: `${((s.avoid / maxDrive) * 50 * bars).toFixed(1)}%` }} />
                      </span>
                    )}
                  </div>
                  <div className="m-num">
                    <span className="big">{r.distance}</span>
                    <span>
                      {r.shown}/{k} shared
                    </span>
                    {bars >= 1 && (
                      <span className="val">
                        valence {s.valence >= 0 ? "+" : "−"}
                        {Math.abs(s.valence).toFixed(1)}
                      </span>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
        )}

        {beat >= MATCH && (
          <p className="legend-row">
            <span>
              <i style={{ background: palette.active }} /> active cell
            </span>
            <span>
              <i style={{ background: palette.shared }} /> shared with the seed
            </span>
            <span>big number: Hamming distance</span>
            {verdictU > 0 && (
              <>
                <span>
                  <i style={{ background: palette.approach }} /> approach drive
                </span>
                <span>
                  <i style={{ background: palette.avoid }} /> avoid drive
                </span>
              </>
            )}
          </p>
        )}

        {deckVisible && (
          <div className="deck">
            <div className="deck-head">
              <FlyAtTheDecks />
              <div>
                <p className="eyebrow">
                  Next up
                  {taught.length > 0 && ` · after ${taught.length} ${taught.length === 1 ? "lesson" : "lessons"} from you`}
                </p>
                {resolved && !teaching ? (
                  <h3>
                    {cat.name[winner.row]} <span className="meta">· {cat.artist[winner.row]}</span>
                  </h3>
                ) : (
                  <h3 className="meta">{teaching ? "Rewiring the mushroom body…" : "Re-running the same query with the retrained fly…"}</h3>
                )}
              </div>
            </div>
            {resolved && last && last.changed !== null && (
              <p className={`changed${last.changed ? " yes" : ""}`}>
                {last.changed
                  ? `The fly changed its mind. It was going to play “${cat.name[last.from]}”.`
                  : `Still “${cat.name[last.from]}”. Its neighbours share many of its cells, so the lesson dented them too. Tell it again.`}
              </p>
            )}
            <p className="liner">
              {resolved && !teaching && note.slice(0, typed)}
              {resolved && !teaching && typed < note.length && <span className="caret" aria-hidden="true" />}
            </p>
            {(done || taught.length > 0) && (
              <>
                <Synapses
                  cells={cellsOf(cat, lesson && (teaching || !resolved) ? lesson.row : winner.row)}
                  wApproach={mb.current.wApproach}
                  wAvoid={mb.current.wAvoid}
                  lesson={lesson}
                  palette={palette}
                />
                <p className="note">
                  The pick&apos;s {k} active Kenyon cells and their synapses onto the two output neurons; line width is synaptic
                  weight.
                  {lesson && (
                    <>
                      {" "}
                      <span className="inline-swatch" style={{ background: palette.depressed }} aria-hidden="true" /> Orange marks the {lesson.mbon} synapses your last
                      lesson depressed{lesson.row !== winner.row ? ", on the cells this track shares with the one you taught it about" : ""}.
                    </>
                  )}
                </p>
                <div className="teach">
                  <button className="btn" onClick={() => teach(1)} disabled={teaching || !done}>
                    ▶ Play it out
                  </button>
                  <button className="btn skip" onClick={() => teach(-1)} disabled={teaching || !done}>
                    ⏭ Skip it
                  </button>
                  <button className="btn ghost" onClick={reset} disabled={teaching || !done || !taught.length}>
                    Reset the fly
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {taught.length > 0 && (
          <ol className="lessons" aria-label="What you taught the fly">
            {taught.map((h, i) => (
              <li key={i}>{h.text}</li>
            ))}
          </ol>
        )}
      </div>

      <div className="transport">
        <button className="btn" onClick={() => (done ? (setT(0), setPlaying(true)) : setPlaying((p) => !p))} disabled={teaching}>
          {done ? "↺ Replay" : playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button className="btn ghost" onClick={step} disabled={teaching || done}>
          Step ›
        </button>
        <label className="scrub">
          <span className="visually-hidden">Scrub through the sequence</span>
          <input
            type="range"
            min={0}
            max={TOTAL}
            step={0.05}
            value={t}
            disabled={teaching}
            onChange={(e) => {
              setPlaying(false);
              setT(Number(e.target.value));
            }}
          />
        </label>
        <span className="time">
          {done ? Math.round(TOTAL) : Math.floor(t)}/{Math.round(TOTAL)} s
        </span>
      </div>

      <div className="seed-pick">
        <p className="note">Try another seed:</p>
        <div className="chips" role="group" aria-label="Choose a seed track">
          {seeds.map((r) => (
            <button key={r} className="chip" aria-pressed={r === seed} onClick={() => onSeed(r)} disabled={teaching}>
              {cat.name[r]}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

function FeatureBars({ circuit, firing, lit }: { circuit: Circuit; firing: Firing; lit: number }) {
  return (
    <figure className="features">
      <div className="fbars" role="img" aria-label="The seed track's input vector, one bar per number">
        {circuit.columns.map((c, j) => {
          const v = clamp01(firing.x[j]);
          return (
            <div key={j} className={`fbar ${c.kind}`} title={`${c.name}: ${firing.x[j].toFixed(3)} (${c.kind})`}>
              <span className="track">
                <i style={{ height: `${v * 100 * lit}%` }} />
              </span>
              <span className="kind">{c.kind === "measured" ? "M" : "P"}</span>
              <span className="name">{c.name}</span>
            </div>
          );
        })}
      </div>
      <figcaption>
        <b>P</b> predicted by the tagger · <b>M</b> measured from the audio
      </figcaption>
    </figure>
  );
}

// The one bit of personality: a fly at the decks.
function FlyAtTheDecks() {
  return (
    <svg className="fly" width="44" height="36" viewBox="0 0 44 36" aria-hidden="true">
      <ellipse cx="14" cy="10" rx="9" ry="4.5" fill="var(--accent-wash)" stroke="var(--muted)" transform="rotate(-20 14 10)" />
      <ellipse cx="30" cy="10" rx="9" ry="4.5" fill="var(--accent-wash)" stroke="var(--muted)" transform="rotate(20 30 10)" />
      <ellipse cx="22" cy="16" rx="4.5" ry="6.5" fill="var(--ink-2)" />
      <circle cx="22" cy="8.5" r="3.5" fill="var(--ink-2)" />
      <circle cx="20.5" cy="7.8" r="1.4" fill="var(--s2)" />
      <circle cx="23.5" cy="7.8" r="1.4" fill="var(--s2)" />
      <path d="M18.5 20 L13 27 M25.5 20 L31 27" stroke="var(--ink-2)" strokeWidth="1.2" fill="none" />
      <rect x="3" y="26" width="38" height="9" rx="2" fill="var(--surface-2)" stroke="var(--axis)" />
      <circle cx="13" cy="30.5" r="2.6" fill="none" stroke="var(--muted)" />
      <circle cx="31" cy="30.5" r="2.6" fill="none" stroke="var(--muted)" />
    </svg>
  );
}
