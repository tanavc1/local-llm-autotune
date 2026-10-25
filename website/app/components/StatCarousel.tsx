"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// Real benchmark data (proof_suite v2, Apple M2 16 GB, Ollama Go timers)
type Slide = {
  eyebrow: string;
  value: string;
  label: string;
  sub: string;
  visual: React.ReactNode;
};

function Bars({ rows, unit }: { rows: { name: string; raw: number; tuned: number }[]; unit: string }) {
  const max = Math.max(...rows.map((r) => r.raw));
  return (
    <div className="carousel-bars">
      {rows.map((r) => (
        <div key={r.name} className="carousel-bar-row">
          <span className="carousel-bar-name">{r.name}</span>
          <div className="carousel-bar-track">
            <div className="carousel-bar raw" style={{ width: `${(r.raw / max) * 100}%` }} />
            <div className="carousel-bar tuned" style={{ width: `${(r.tuned / max) * 100}%` }} />
          </div>
          <span className="carousel-bar-val">
            {r.raw}→{r.tuned}
            {unit}
          </span>
        </div>
      ))}
    </div>
  );
}

const slides: Slide[] = [
  {
    eyebrow: "RAM freed per request",
    value: "381 MB",
    label: "returned to your system on every chat request",
    sub: "qwen3:8b · Apple M2 16 GB",
    visual: (
      <div className="carousel-memory">
        <div className="memory-bar raw"><span>Raw Ollama</span><strong>576 <small>MB</small></strong></div>
        <div className="memory-bar tuned"><span>With autotune</span><strong>195 <small>MB</small></strong></div>
      </div>
    ),
  },
  {
    eyebrow: "Faster first word",
    value: "53%",
    label: "faster time-to-first-token on qwen3:8b",
    sub: "39% average across 3 models",
    visual: (
      <div className="carousel-list">
        {[
          ["qwen3:8b", "−53%", 53],
          ["llama3.2:3b", "−35%", 35],
          ["gemma3n:e4b", "−29%", 29],
        ].map(([name, label, pct]) => (
          <div key={name as string} className="carousel-bar-row">
            <span className="carousel-bar-name">{name}</span>
            <div className="carousel-bar-track">
              <div className="carousel-bar tuned" style={{ width: `${((pct as number) / 53) * 100}%` }} />
            </div>
            <span className="carousel-bar-val accent">{label}</span>
          </div>
        ))}
      </div>
    ),
  },
  {
    eyebrow: "Less KV cache",
    value: "67%",
    label: "less memory reserved by Ollama — about 3× smaller",
    sub: "KV cache, raw vs tuned",
    visual: (
      <Bars
        unit=" MB"
        rows={[
          { name: "qwen3:8b", raw: 576, tuned: 195 },
          { name: "llama3.2:3b", raw: 448, tuned: 155 },
          { name: "gemma3n:e4b", raw: 96, tuned: 30 },
        ]}
      />
    ),
  },
  {
    eyebrow: "Swap events",
    value: "0",
    label: "disk swaps across all 45 benchmark runs",
    sub: "Live RAM tiers + NoSwapGuard keep headroom",
    visual: (
      <div className="carousel-grid" aria-hidden="true">
        {Array.from({ length: 45 }, (_, i) => (
          <span key={i} style={{ animationDelay: `${i * 12}ms` }} />
        ))}
      </div>
    ),
  },
  {
    eyebrow: "Agent sessions",
    value: "46%",
    label: "lower wall time on a 10-turn coding agent task",
    sub: "74 s → 40 s · llama3.2:3b",
    visual: (
      <Bars unit=" s" rows={[{ name: "session", raw: 74, tuned: 40 }]} />
    ),
  },
];

const INTERVAL_MS = 5000;

export function StatCarousel() {
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const touchX = useRef<number | null>(null);
  const count = slides.length;

  const go = useCallback((i: number) => setIndex(((i % count) + count) % count), [count]);

  useEffect(() => {
    if (paused) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const t = setTimeout(() => go(index + 1), INTERVAL_MS);
    return () => clearTimeout(t);
  }, [index, paused, go]);

  return (
    <aside
      className="hero-measurement stat-carousel"
      aria-roledescription="carousel"
      aria-label="autotune benchmark results"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") go(index + 1);
        if (e.key === "ArrowLeft") go(index - 1);
      }}
      onTouchStart={(e) => (touchX.current = e.touches[0].clientX)}
      onTouchEnd={(e) => {
        if (touchX.current === null) return;
        const dx = e.changedTouches[0].clientX - touchX.current;
        if (Math.abs(dx) > 40) go(index + (dx < 0 ? 1 : -1));
        touchX.current = null;
      }}
    >
      <div className="measurement-heading">
        <span>Measured · Apple M2 16 GB</span>
        <span>
          {String(index + 1).padStart(2, "0")} / {String(count).padStart(2, "0")}
        </span>
      </div>

      <div className="carousel-viewport">
        {slides.map((s, i) => (
          <div
            key={s.eyebrow}
            className={`carousel-slide ${i === index ? "is-active" : ""}`}
            role="group"
            aria-roledescription="slide"
            aria-label={`${i + 1} of ${count}: ${s.eyebrow}`}
            aria-hidden={i !== index}
          >
            <p className="carousel-eyebrow">{s.eyebrow}</p>
            <div className="carousel-value">{s.value}</div>
            <p className="carousel-label">{s.label}</p>
            <div className="carousel-visual">{s.visual}</div>
            <p className="carousel-sub">{s.sub}</p>
          </div>
        ))}
      </div>

      <div className="carousel-controls">
        <div className="carousel-dots" role="tablist" aria-label="Choose a stat">
          {slides.map((s, i) => (
            <button
              key={s.eyebrow}
              role="tab"
              aria-selected={i === index}
              aria-label={s.eyebrow}
              className={i === index ? "is-active" : ""}
              onClick={() => go(i)}
            >
              <span
                className={i === index && !paused ? "is-running" : undefined}
                style={{ animationDuration: `${INTERVAL_MS}ms` }}
              />
            </button>
          ))}
        </div>
        <div className="carousel-arrows">
          <button aria-label="Previous stat" onClick={() => go(index - 1)}>←</button>
          <button aria-label="Next stat" onClick={() => go(index + 1)}>→</button>
        </div>
      </div>
    </aside>
  );
}
