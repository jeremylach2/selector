"use client";

import type { ReactNode } from "react";
import ChartRace from "@/components/rewind/ChartRace";
import { type AnyCard, type Card, type Report, coverageNote } from "@/lib/rewind";

const pct = (x: number) => `${Math.round(100 * x)}%`;
const int = (x: number) => Math.round(x).toLocaleString();

function Shell({ card, children }: { card: AnyCard; children?: ReactNode }) {
  const note = coverageNote(card.coverage);
  return (
    <div className="slide-body">
      <h2 className="slide-headline">{card.headline}</h2>
      {card.sublabel && <p className="slide-sub">{card.sublabel}</p>}
      {children}
      {note && <p className="slide-cov">{note}</p>}
    </div>
  );
}

function Evidence({ lines }: { lines: string[] }) {
  return (
    <ul className="slide-evidence">
      {lines.map((l) => (
        <li key={l}>{l}</li>
      ))}
    </ul>
  );
}

function Big({ children }: { children: ReactNode }) {
  return <p className="slide-big">{children}</p>;
}

// A ranked list with bars scaled to the leader.
function Ranked({ rows }: { rows: { title: string; sub?: string; value: number; label: string }[] }) {
  const max = Math.max(...rows.map((r) => r.value), 1);
  return (
    <ol className="ranked">
      {rows.map((r, i) => (
        <li key={`${r.title}-${i}`}>
          <span className="ranked-n">{i + 1}</span>
          <span className="ranked-text">
            <span className="ranked-title">{r.title}</span>
            {r.sub && <span className="ranked-sub">{r.sub}</span>}
            <span className="ranked-bar" style={{ width: `${(100 * r.value) / max}%` }} />
          </span>
          <span className="ranked-val">{r.label}</span>
        </li>
      ))}
    </ol>
  );
}

function Columns({ bars, highlight }: { bars: { label: string; value: number; tick?: string }[]; highlight?: number }) {
  const max = Math.max(...bars.map((b) => b.value), 1);
  return (
    <div className="cols" role="img" aria-label={bars.map((b) => `${b.label}: ${int(b.value)}`).join(", ")}>
      {bars.map((b, i) => (
        <div key={b.label} className="col">
          <div className={i === highlight ? "col-bar on" : "col-bar"} style={{ height: `${(100 * b.value) / max}%` }} />
          <span className="col-tick">{b.tick ?? ""}</span>
        </div>
      ))}
    </div>
  );
}

export function IntroSlide({ report }: { report: Report }) {
  const { totals, window: w } = report;
  return (
    <div className="slide-body intro">
      <p className="eyebrow">Selector Rewind</p>
      <h2 className="slide-big">{w.label}</h2>
      <p className="slide-sub">
        {w.from} → {w.to}
      </p>
      {report.audience === "synthetic" && <p className="slide-cov">An invented listener, not a real person.</p>}
      <dl className="intro-totals">
        <div>
          <dt>plays</dt>
          <dd>{int(totals.plays)}</dd>
        </div>
        <div>
          <dt>hours</dt>
          <dd>{int(totals.hours)}</dd>
        </div>
        <div>
          <dt>tracks</dt>
          <dd>{int(totals.tracks)}</dd>
        </div>
        <div>
          <dt>artists</dt>
          <dd>{int(totals.artists)}</dd>
        </div>
      </dl>
      <p className="slide-cov">Tap the right side or press → to continue.</p>
    </div>
  );
}

export function CardSlide({ card, active }: { card: AnyCard; active: boolean }) {
  const c = card as Card;
  switch (c.id) {
    case "total_hours":
      return (
        <Shell card={c}>
          <Big>{int(c.value)}</Big>
          <Evidence lines={c.evidence} />
        </Shell>
      );
    case "listening_age":
      return (
        <Shell card={c}>
          <Big>{c.value}</Big>
          <Evidence lines={c.evidence.slice(0, 2)} />
        </Shell>
      );
    case "top_artists":
      return (
        <Shell card={c}>
          <Ranked rows={c.value.map((r) => ({ title: r.artist, value: r.play_count, label: int(r.play_count) }))} />
        </Shell>
      );
    case "top_tracks":
      return (
        <Shell card={c}>
          <Ranked rows={c.value.map((r) => ({ title: r.name, sub: r.artist, value: r.play_count, label: int(r.play_count) }))} />
        </Shell>
      );
    case "top_albums":
      return (
        <Shell card={c}>
          <Ranked rows={c.value.map((r) => ({ title: r.album, sub: r.artist, value: r.play_count, label: int(r.play_count) }))} />
        </Shell>
      );
    case "artist_sprint":
      return (
        <Shell card={c}>
          <ChartRace {...c.value} active={active} />
        </Shell>
      );
    case "time_of_day": {
      const clock = (h: number) => `${h % 12 || 12}${h < 12 ? "a" : "p"}`;
      const bars = c.value.by_hour.map((v, h) => ({ label: clock(h), value: v, tick: h % 6 === 0 ? clock(h) : "" }));
      return (
        <Shell card={c}>
          <Columns bars={bars} highlight={c.value.peak_hour} />
          <p className="slide-cov">Hour of day, {c.value.timezone_label} time.</p>
        </Shell>
      );
    }
    case "skip_offenders":
      return (
        <Shell card={c}>
          <Ranked
            rows={c.value.slice(0, 5).map((r) => ({ title: r.name, sub: r.artist, value: r.skip_rate, label: `${pct(r.skip_rate)} skipped` }))}
          />
        </Shell>
      );
    case "decade_histogram":
      return (
        <Shell card={c}>
          <Columns bars={c.value.map((d) => ({ label: `${d.decade}s`, value: d.play_count, tick: `'${String(d.decade).slice(2)}` }))} />
          <Evidence lines={c.evidence} />
        </Shell>
      );
    case "taste_clusters":
      return (
        <Shell card={c}>
          <Ranked
            rows={c.value.slice(0, 6).map((r) => ({
              title: r.name,
              sub: r.name === r.built_name ? r.top_artists.slice(0, 2).join(", ") : r.built_name,
              value: r.play_share,
              label: pct(r.play_share),
            }))}
          />
        </Shell>
      );
    case "hidden_gems":
      return (
        <Shell card={c}>
          <ol className="gems">
            {c.value.map((g) => (
              <li key={g.track_id}>
                <span className="ranked-title">{g.name}</span>
                <span className="ranked-sub">
                  {g.artist} · fly rank #{int(g.fly_score_rank)} · played {g.play_count === 1 ? "once" : `${g.play_count}×`}
                </span>
              </li>
            ))}
          </ol>
        </Shell>
      );
    case "archetype":
      return (
        <Shell card={c}>
          <Evidence lines={c.evidence} />
        </Shell>
      );
    default:
      return (
        <Shell card={card}>
          <Evidence lines={card.evidence} />
        </Shell>
      );
  }
}
