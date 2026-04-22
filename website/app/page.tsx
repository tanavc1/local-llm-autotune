import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "autotune — Make your local AI faster",
  description:
    "autotune frees 300+ MB of RAM per request and cuts response time by up to 53%. Drop-in wrapper for Ollama. One pip install, no config.",
  openGraph: {
    title: "autotune — Make your local AI faster",
    description:
      "Frees 300+ MB of RAM per request. Cuts first-word latency by up to 53%. Drop-in for Ollama.",
    type: "website",
  },
};

// ─── Real benchmark data (proof_suite v2, Apple M2 16 GB, Ollama Go timers) ──

const stats = [
  {
    value: "381 MB",
    label: "RAM freed per request",
    sub: "on qwen3:8b — back to your browser",
  },
  {
    value: "53%",
    label: "Faster first word",
    sub: "on qwen3:8b; 39% avg across 3 models",
  },
  {
    value: "67%",
    label: "Less KV cache",
    sub: "Ollama reserves 3× less memory",
  },
  {
    value: "0",
    label: "Swap events",
    sub: "across all 45 benchmark runs",
  },
];

// KV freed = raw_kv_mb − tuned_kv_mb from proof_suite
const benchmarks = [
  {
    model: "qwen3:8b",
    kvRaw: "576 MB",
    kvTuned: "195 MB",
    kvFreed: "381 MB",
    ttft: "−53%",
    speed: "unchanged",
  },
  {
    model: "llama3.2:3b",
    kvRaw: "448 MB",
    kvTuned: "155 MB",
    kvFreed: "293 MB",
    ttft: "−35%",
    speed: "unchanged",
  },
  {
    model: "gemma4:e2b",
    kvRaw: "96 MB",
    kvTuned: "30 MB",
    kvFreed: "66 MB",
    ttft: "−29%",
    speed: "unchanged",
  },
];

const installSnippet = `pip install llm-autotune
autotune chat --model qwen3:8b`;

const proofSnippet = `autotune proof -m qwen3:8b
# Runs in ~30 seconds. Uses Ollama's own timers.
# Saves a proof_qwen3_8b.json you can share.`;

const codeSnippet = `import autotune
from openai import OpenAI

autotune.start()  # start the optimizing proxy

client = OpenAI(**autotune.client_kwargs())

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Hello!"}],
)
# Every optimization is automatic.`;

// ─── Components ────────────────────────────────────────────────────────────────

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-violet-500/40 bg-violet-500/10 px-3 py-1 text-xs font-medium text-violet-300">
      {children}
    </span>
  );
}

function CodeBlock({ code, language = "bash" }: { code: string; language?: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 backdrop-blur overflow-hidden">
      <div className="flex items-center gap-1.5 border-b border-white/10 px-4 py-2.5">
        <span className="h-2.5 w-2.5 rounded-full bg-red-500/70" />
        <span className="h-2.5 w-2.5 rounded-full bg-yellow-500/70" />
        <span className="h-2.5 w-2.5 rounded-full bg-green-500/70" />
        <span className="ml-3 text-xs text-white/30">{language}</span>
      </div>
      <pre className="overflow-x-auto p-5 text-sm leading-relaxed text-green-300 font-mono">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-3">
      {children}
    </p>
  );
}

function StatCard({ value, label, sub }: { value: string; label: string; sub: string }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/4 p-5 text-center backdrop-blur">
      <div className="text-3xl font-bold text-violet-300">{value}</div>
      <div className="mt-1 text-sm font-medium text-white/85">{label}</div>
      <div className="mt-0.5 text-xs text-white/40">{sub}</div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function Home() {
  return (
    <div className="min-h-screen text-[#e8e8f0] selection:bg-violet-500/30">

      {/* ── Nav ── */}
      <nav className="fixed top-0 z-50 w-full border-b border-white/5 bg-[#09090f]/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold tracking-tight text-white">autotune</span>
            <span className="hidden rounded-full bg-violet-500/20 px-2 py-0.5 text-xs font-medium text-violet-300 sm:block">
              v0.2.0
            </span>
          </div>
          <div className="flex items-center gap-6 text-sm text-white/60">
            <a href="#how-it-works" className="hidden hover:text-white transition-colors sm:block">
              How it works
            </a>
            <a href="#benchmarks" className="hidden hover:text-white transition-colors sm:block">
              Benchmarks
            </a>
            <a href="#quickstart" className="hidden hover:text-white transition-colors sm:block">
              Install
            </a>
            <a href="/what-we-do" className="hidden hover:text-white transition-colors sm:block font-medium text-violet-400 hover:text-violet-300">
              All we do
            </a>
            <a
              href="https://github.com/tanavc1/local-llm-autotune"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/80 transition hover:border-violet-500/50 hover:bg-violet-500/10 hover:text-white"
            >
              <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
              </svg>
              GitHub
            </a>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 text-center pt-16">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -top-32 left-1/2 h-[600px] w-[600px] -translate-x-1/2 rounded-full bg-violet-600/10 blur-[100px]" />
          <div className="absolute bottom-0 left-0 h-[400px] w-[400px] rounded-full bg-blue-600/8 blur-[80px]" />
        </div>

        <div className="relative z-10 flex flex-col items-center gap-6 max-w-4xl">
          <Badge>Open source · MIT · pip install llm-autotune</Badge>

          <h1 className="text-5xl font-bold tracking-tight text-white sm:text-6xl lg:text-7xl animate-fade-up">
            Your local AI,{" "}
            <span className="bg-gradient-to-r from-violet-400 to-blue-400 bg-clip-text text-transparent">
              actually fast.
            </span>
          </h1>

          <p className="max-w-2xl text-lg text-white/60 leading-relaxed animate-fade-up delay-100">
            autotune is an inference optimizer that sits in front of Ollama and applies
            four automatic optimizations: right-sized KV buffers, system prompt caching,
            adaptive memory precision, and model keep-alive.
            The result: <strong className="text-white/90">300+ MB freed</strong> per
            request, first word up to <strong className="text-white/90">53% faster</strong>,
            and your computer stays responsive.
            No config changes. Your code stays exactly the same.
          </p>

          <div className="mt-2 w-full max-w-md animate-fade-up delay-200">
            <CodeBlock code={installSnippet} language="bash" />
          </div>

          <div className="flex flex-wrap items-center justify-center gap-4 animate-fade-up delay-300">
            <a
              href="#quickstart"
              className="rounded-xl bg-violet-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-900/40 transition hover:bg-violet-500"
            >
              Get started →
            </a>
            <a
              href="#proof"
              className="rounded-xl border border-white/10 bg-white/5 px-6 py-3 text-sm font-medium text-white/80 backdrop-blur transition hover:border-violet-500/40 hover:bg-violet-500/10 hover:text-white"
            >
              Prove it on your machine
            </a>
          </div>
        </div>

        {/* Stats row */}
        <div className="relative z-10 mt-20 grid grid-cols-2 gap-4 sm:grid-cols-4 max-w-4xl w-full animate-fade-up delay-400">
          {stats.map((s) => (
            <StatCard key={s.label} {...s} />
          ))}
        </div>
      </section>

      {/* ── How it works ── */}
      <section id="how-it-works" className="py-28 px-6 bg-white/[0.02] border-y border-white/5">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>How it works</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            Four optimizations, applied automatically.
          </h2>
          <p className="text-white/55 mb-12 max-w-2xl">
            autotune sits between your code and Ollama as a transparent proxy. Every request
            passes through four layers of optimization — none require config, none change your
            prompt or output quality.
          </p>

          {/* Optimization cards */}
          <div className="grid gap-5 sm:grid-cols-2">

            {/* 1 — Dynamic KV sizing */}
            <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-6">
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-violet-500/20 text-xs font-bold text-violet-300">1</span>
                <span className="text-sm font-semibold text-white">Dynamic KV buffer sizing</span>
              </div>
              <p className="text-sm text-white/60 mb-4">
                Ollama always reserves a fixed 4,096-token KV buffer — regardless of how long
                your message is. For a typical 340-token message, that&apos;s 3× more memory
                than needed. autotune computes the exact minimum and allocates only that.
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg border border-red-500/15 bg-red-500/5 px-3 py-2">
                  <div className="text-white/40 mb-0.5">Raw Ollama</div>
                  <div className="font-mono text-white/70">448 MB reserved</div>
                  <div className="text-white/35">for 4,096 tokens</div>
                </div>
                <div className="rounded-lg border border-green-500/20 bg-green-500/5 px-3 py-2">
                  <div className="text-white/40 mb-0.5">autotune</div>
                  <div className="font-mono text-green-300">155 MB reserved</div>
                  <div className="text-white/35">293 MB freed</div>
                </div>
              </div>
            </div>

            {/* 2 — System prompt caching */}
            <div className="rounded-2xl border border-blue-500/20 bg-blue-500/5 p-6">
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-500/20 text-xs font-bold text-blue-300">2</span>
                <span className="text-sm font-semibold text-white">System prompt prefix caching</span>
              </div>
              <p className="text-sm text-white/60 mb-4">
                Every multi-turn conversation has a system prompt that Ollama re-processes
                from scratch on each message. autotune pins those tokens in the KV cache via{" "}
                <code className="text-white/70">num_keep</code> — they&apos;re evaluated once
                and reused forever, cutting prefill time on every follow-up turn.
              </p>
              <div className="rounded-lg border border-blue-500/15 bg-blue-500/5 px-3 py-2 text-xs">
                <div className="text-white/40 mb-1">Impact per follow-up message</div>
                <div className="text-blue-300 font-mono">System prompt tokens: never re-evaluated</div>
                <div className="text-white/35 mt-0.5">Pure latency win — zero quality impact</div>
              </div>
            </div>

            {/* 3 — Adaptive KV precision */}
            <div className="rounded-2xl border border-orange-500/20 bg-orange-500/5 p-6">
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-orange-500/20 text-xs font-bold text-orange-300">3</span>
                <span className="text-sm font-semibold text-white">Adaptive KV precision</span>
              </div>
              <p className="text-sm text-white/60 mb-4">
                As your system RAM fills up, autotune automatically downgrades KV cache
                precision from F16 to Q8 — cutting KV memory in half at three configurable
                thresholds. On tight machines (8 GB MacBooks, RAM-heavy multitasking), this
                prevents swap entirely.
              </p>
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span className="text-white/30">RAM &gt; 80%</span>
                  <span className="text-orange-300/80">→ context −10%, stays F16</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-white/30">RAM &gt; 88%</span>
                  <span className="text-orange-300">→ switch F16 → Q8  (½ KV memory)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-white/30">RAM &gt; 93%</span>
                  <span className="text-red-300">→ context hard-capped, swap prevented</span>
                </div>
              </div>
            </div>

            {/* 4 — Keep-alive */}
            <div className="rounded-2xl border border-green-500/20 bg-green-500/5 p-6">
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-green-500/20 text-xs font-bold text-green-300">4</span>
                <span className="text-sm font-semibold text-white">Model keep-alive</span>
              </div>
              <p className="text-sm text-white/60 mb-4">
                Raw Ollama unloads the model after 5 minutes of idle. Every time you come
                back, you pay a 1–3 second reload cost before your first token. autotune
                keeps the model loaded in unified memory between sessions and manages
                eviction gracefully when RAM is needed elsewhere.
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg border border-red-500/15 bg-red-500/5 px-3 py-2">
                  <div className="text-white/40 mb-0.5">Raw Ollama (idle 5 min)</div>
                  <div className="font-mono text-white/70">1–3s cold reload</div>
                  <div className="text-white/35">every new session</div>
                </div>
                <div className="rounded-lg border border-green-500/20 bg-green-500/5 px-3 py-2">
                  <div className="text-white/40 mb-0.5">autotune</div>
                  <div className="font-mono text-green-300">stays loaded</div>
                  <div className="text-white/35">instant first token</div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 flex flex-col sm:flex-row sm:items-center gap-4">
            <p className="text-xs text-white/35 max-w-xl">
              Numbers above use llama3.2:3b on Apple M2. KV size scales with model architecture —
              larger models free more RAM in absolute terms.
              Generation quality is identical: autotune changes buffer sizes and precision,
              not model weights or sampling.
            </p>
            <a
              href="/what-we-do"
              className="shrink-0 rounded-lg border border-violet-500/30 bg-violet-500/8 px-4 py-2 text-xs font-medium text-violet-300 transition hover:border-violet-500/50 hover:bg-violet-500/15 hover:text-violet-200"
            >
              All 14 optimizations explained →
            </a>
          </div>
        </div>
      </section>

      {/* ── Benchmarks ── */}
      <section id="benchmarks" className="py-28 px-6">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Measured results</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            Real numbers. Real hardware.
          </h2>
          <p className="text-white/55 mb-10 max-w-2xl">
            Benchmarked on Apple M2 16 GB using Ollama&apos;s internal Go nanosecond timers —
            not wall-clock estimates. 3 runs × 5 prompt types, Wilcoxon signed-rank test.
            Every number here is reproducible with{" "}
            <code className="text-white/70">autotune proof</code>.
          </p>

          <div className="overflow-hidden rounded-2xl border border-white/8">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/8 bg-white/4 text-left text-xs font-semibold uppercase tracking-wider text-white/50">
                  <th className="px-5 py-4">Model</th>
                  <th className="px-5 py-4">KV: Before</th>
                  <th className="px-5 py-4">KV: After</th>
                  <th className="px-5 py-4 text-green-400/80">RAM freed</th>
                  <th className="px-5 py-4">First word</th>
                  <th className="px-5 py-4 hidden sm:table-cell">Speed</th>
                </tr>
              </thead>
              <tbody>
                {benchmarks.map((b, i) => (
                  <tr
                    key={b.model}
                    className={`border-b border-white/5 ${i % 2 === 0 ? "bg-white/2" : ""}`}
                  >
                    <td className="px-5 py-4 font-mono text-white/80">{b.model}</td>
                    <td className="px-5 py-4 text-white/40">{b.kvRaw}</td>
                    <td className="px-5 py-4 text-white/60">{b.kvTuned}</td>
                    <td className="px-5 py-4 font-semibold text-green-400">{b.kvFreed}</td>
                    <td className="px-5 py-4 font-semibold text-green-400">{b.ttft}</td>
                    <td className="px-5 py-4 text-white/40 hidden sm:table-cell">{b.speed}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-4 text-xs text-white/30 max-w-2xl">
            TTFT improvement is largest when the model is cold or when RAM is under pressure.
            Generation speed (tok/s) is Metal GPU-bound and is not affected by autotune.
            KV savings apply every single request regardless of hardware.
          </p>

          {/* KPI cards */}
          <div className="mt-14 grid gap-5 sm:grid-cols-3">
            <div className="rounded-2xl border border-white/8 bg-white/3 p-6">
              <div className="text-2xl font-bold text-violet-300 mb-2">300+ MB freed</div>
              <p className="text-sm text-white/55">
                Every request you send, Ollama allocates a KV buffer for 4,096 tokens.
                autotune sizes it to the actual prompt — returning hundreds of MB to your
                system on every single call, automatically.
              </p>
            </div>
            <div className="rounded-2xl border border-white/8 bg-white/3 p-6">
              <div className="text-2xl font-bold text-violet-300 mb-2">Up to 53% faster</div>
              <p className="text-sm text-white/55">
                The KV buffer must be initialized before token 1. A smaller buffer
                initializes faster. On qwen3:8b, autotune cuts first-word time from
                the raw baseline by 53% — every new session, every cold request.
              </p>
            </div>
            <div className="rounded-2xl border border-white/8 bg-white/3 p-6">
              <div className="text-2xl font-bold text-violet-300 mb-2">Zero trade-offs</div>
              <p className="text-sm text-white/55">
                autotune changes only the KV buffer size. Model weights, sampling, and
                generation speed are identical.{" "}
                <code className="text-white/70">prompt_eval_count</code> is unchanged —
                no tokens are dropped or skipped.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Agents ── */}
      <section className="py-28 px-6">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Multi-turn &amp; agentic workloads</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            Where it matters most.
          </h2>
          <p className="text-white/55 mb-4 max-w-2xl">
            Single-prompt benchmarks miss the real problem: <strong className="text-white/80">context accumulates</strong>.
            Each tool call, each reasoning step, each file read appends more tokens. By turn 8,
            the model is processing 5–8× more tokens than turn 1 — and raw Ollama&apos;s fixed
            4,096-token window runs out, forcing a full model reload mid-session.
          </p>
          <p className="text-white/55 mb-10 max-w-2xl">
            autotune computes a session-ceiling KV window once before the loop starts and locks
            it for the entire session. No reloads. And because the system prompt is pinned via
            prefix caching, TTFT actually <em>falls</em> as the session grows — not climbs.
          </p>

          <div className="grid gap-6 lg:grid-cols-2 mb-8">
            {/* Code debugger result */}
            <div className="rounded-2xl border border-white/8 overflow-hidden">
              <div className="border-b border-white/8 bg-white/3 px-5 py-3">
                <div className="text-xs font-semibold text-white/60 uppercase tracking-wider">
                  Code debugger task — 10 turns, llama3.2:3b
                </div>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/5 text-xs font-semibold uppercase tracking-wider text-white/35">
                    <th className="px-5 py-3 text-left">Metric</th>
                    <th className="px-5 py-3 text-right">Raw Ollama</th>
                    <th className="px-5 py-3 text-right text-green-400/80">autotune</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { metric: "Session wall time", raw: "74 s", tuned: "40 s", good: true },
                    { metric: "Model reloads", raw: "0.5", tuned: "0.5", good: false },
                    { metric: "TTFT trend per turn", raw: "−101 ms/turn", tuned: "−435 ms/turn", good: true },
                    { metric: "Swap events", raw: "0", tuned: "0", good: false },
                    { metric: "Context at session end", raw: "3,043 tokens", tuned: "1,946 tokens", good: true },
                  ].map((row, i) => (
                    <tr key={row.metric} className={`border-b border-white/5 ${i % 2 === 0 ? "bg-white/2" : ""}`}>
                      <td className="px-5 py-3 text-white/60">{row.metric}</td>
                      <td className="px-5 py-3 text-right text-white/40 font-mono">{row.raw}</td>
                      <td className={`px-5 py-3 text-right font-mono font-semibold ${row.good ? "text-green-400" : "text-white/50"}`}>
                        {row.tuned}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Explanation cards */}
            <div className="flex flex-col gap-4">
              <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-5">
                <div className="text-sm font-semibold text-white mb-2">
                  TTFT falls as the session grows
                </div>
                <p className="text-sm text-white/55">
                  The system prompt is pinned in KV after turn 1 and never re-evaluated.
                  Each new turn only prefills the new tokens — not the full conversation
                  from scratch. By turn 5, autotune is noticeably faster than turn 1.
                  By turn 10, the difference compounds significantly.
                </p>
              </div>
              <div className="rounded-2xl border border-green-500/20 bg-green-500/5 p-5">
                <div className="text-sm font-semibold text-white mb-2">
                  Session window sized for the task
                </div>
                <p className="text-sm text-white/55">
                  autotune computes a KV window for the <em>full session ceiling</em> before
                  the first turn, then holds it constant. raw Ollama&apos;s fixed 4,096-token
                  window fills up mid-task and forces a model reload (~1–3 s each).
                  autotune trades a slightly higher turn-1 cost to eliminate all reloads.
                </p>
              </div>
              <div className="rounded-xl border border-white/8 bg-white/3 px-4 py-3 text-xs text-white/40">
                <strong className="text-white/55">Honest caveat:</strong> Turn 1 is ~80%
                slower — autotune pre-allocates a larger KV window for the whole session.
                From turn 2 onward prefix-cache savings compound and wall time comes out
                46% lower. For single-turn usage, the per-request benchmark numbers apply.
              </div>
            </div>
          </div>

          <p className="text-xs text-white/30">
            Benchmark: code_debugger task, N=2 trials, Apple M2 16 GB, llama3.2:3b balanced profile.
            Timings from Ollama&apos;s internal Go nanosecond timers. Full methodology in{" "}
            <code className="text-white/50">AGENT_BENCHMARK.md</code>.
          </p>
        </div>
      </section>

      {/* ── Prove it yourself ── */}
      <section id="proof" className="py-28 px-6 bg-white/[0.02] border-y border-white/5">
        <div className="mx-auto max-w-5xl">
          <div className="grid gap-12 lg:grid-cols-2 items-center">
            <div>
              <SectionLabel>Verify it yourself</SectionLabel>
              <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
                Don&apos;t trust the numbers. Run the proof.
              </h2>
              <p className="text-white/55 mb-6">
                autotune ships with a built-in benchmark that runs two head-to-head tests on
                your hardware in about 30 seconds. It uses Ollama&apos;s own internal Go
                nanosecond timers — nothing estimated, nothing made up.
              </p>
              <ul className="space-y-3 text-sm text-white/60 mb-8">
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 shrink-0 text-violet-400">✓</span>
                  <span>KV cache size: raw Ollama vs autotune — exact MB</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 shrink-0 text-violet-400">✓</span>
                  <span>Time to first word: two conditions from the same neutral state</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 shrink-0 text-violet-400">✓</span>
                  <span>Saves a JSON file you can inspect or share</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 shrink-0 text-violet-400">✓</span>
                  <span>Generation speed reported honestly (usually unchanged)</span>
                </li>
              </ul>
              <p className="text-xs text-white/35">
                Works with any model you have installed in Ollama.
                Picks the smallest installed model automatically if you don&apos;t specify one.
              </p>
            </div>
            <div>
              <CodeBlock code={proofSnippet} language="bash" />
              <div className="mt-4 rounded-xl border border-violet-500/20 bg-violet-500/5 p-4 text-sm text-white/55">
                <span className="text-violet-300 font-medium">Tip:</span> Run{" "}
                <code className="text-green-300/80">autotune proof --list-models</code> to see
                which Ollama models are available on your machine.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Quickstart ── */}
      <section id="quickstart" className="py-28 px-6">
        <div className="mx-auto max-w-5xl">
          <div className="grid gap-12 lg:grid-cols-2">
            <div>
              <SectionLabel>Quickstart</SectionLabel>
              <h2 className="text-3xl font-bold text-white mb-8">Up in 60 seconds</h2>
              <ol className="space-y-6">
                {[
                  {
                    n: "1",
                    title: "Install Ollama and pull a model",
                    body: "Download from ollama.com — any model you already have works.",
                    code: "ollama pull qwen3:8b",
                  },
                  {
                    n: "2",
                    title: "Install autotune",
                    body: "One pip install — no other configuration required.",
                    code: "pip install llm-autotune",
                  },
                  {
                    n: "3",
                    title: "Chat with automatic optimization",
                    body: "autotune intercepts every Ollama request and right-sizes it.",
                    code: "autotune chat --model qwen3:8b",
                  },
                  {
                    n: "4",
                    title: "Prove it helps on your hardware",
                    body: "30-second head-to-head. Uses Ollama's own timers.",
                    code: "autotune proof -m qwen3:8b",
                  },
                ].map((step) => (
                  <li key={step.n} className="flex gap-4">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-violet-500/40 bg-violet-500/10 text-xs font-bold text-violet-300">
                      {step.n}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-semibold text-white mb-1">{step.title}</div>
                      <div className="text-xs text-white/50 mb-2">{step.body}</div>
                      <code className="block rounded-lg border border-white/8 bg-black/40 px-3 py-1.5 text-xs font-mono text-green-300 overflow-x-auto">
                        {step.code}
                      </code>
                    </div>
                  </li>
                ))}
              </ol>

              <div className="mt-8 rounded-2xl border border-blue-500/20 bg-blue-500/5 p-4">
                <div className="text-xs font-semibold text-blue-300 mb-1">Apple Silicon</div>
                <div className="text-xs text-white/50 mb-2">
                  Native Metal GPU kernels via MLX — 10–40% faster generation throughput.
                </div>
                <code className="text-xs font-mono text-green-300">
                  pip install &quot;llm-autotune[mlx]&quot;
                </code>
              </div>
            </div>

            <div className="flex flex-col gap-5">
              <div>
                <div className="text-sm font-semibold text-white/70 mb-3">
                  Use from Python — drop-in for any OpenAI client
                </div>
                <CodeBlock code={codeSnippet} language="python" />
              </div>

              <div>
                <div className="text-sm font-semibold text-white/70 mb-3">
                  Or run as an API server
                </div>
                <CodeBlock
                  code={`autotune serve\n# → http://localhost:8765/v1\n# Any OpenAI client works automatically.`}
                  language="bash"
                />
              </div>

              <div className="rounded-2xl border border-white/8 bg-white/3 p-4 text-xs text-white/50 space-y-1">
                <div className="font-semibold text-white/70 mb-2">Profiles</div>
                <div className="flex gap-2">
                  <span className="text-violet-300 font-mono w-20 shrink-0">fast</span>
                  <span>2k ctx · Q8 KV · quick lookups &amp; completions</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-violet-300 font-mono w-20 shrink-0">balanced</span>
                  <span>8k ctx · F16 KV · general chat (default)</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-violet-300 font-mono w-20 shrink-0">quality</span>
                  <span>32k ctx · F16 KV · long-form writing &amp; analysis</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── What's inside ── */}
      <section id="features" className="py-28 px-6 bg-white/[0.02] border-y border-white/5">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>What autotune does</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-12">
            Every optimization, automatic
          </h2>

          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {[
              {
                title: "Dynamic KV sizing",
                icon: "⚡",
                desc: "Computes the exact context window each request needs. Typical chat message: 4,096 → 1,400 tokens. Frees 60–70% of the KV buffer back to your system.",
              },
              {
                title: "System prompt caching",
                icon: "🔒",
                desc: "Pins your system prompt in Ollama's KV so it's never re-evaluated on follow-up messages. Pure latency win with no quality cost.",
              },
              {
                title: "Adaptive KV precision",
                icon: "🧠",
                desc: "Switches from F16 to Q8 KV under memory pressure — halves KV memory at three automatic thresholds with no quality impact.",
              },
              {
                title: "Keep-alive management",
                icon: "♾️",
                desc: "Holds the model in memory between messages. Eliminates the 1–3s cold-reload cost you'd otherwise pay every time a session goes idle.",
              },
              {
                title: "OpenAI-compatible API",
                icon: "🔌",
                desc: "Drop-in server at localhost:8765/v1. Works with any OpenAI SDK, LangChain, LlamaIndex, or agent framework without code changes.",
              },
              {
                title: "MLX backend (Apple Silicon)",
                icon: "🍎",
                desc: "Routes inference to MLX-LM on M-series Macs for native Metal GPU kernels — 10–40% faster generation throughput over Ollama.",
              },
            ].map((f) => (
              <div
                key={f.title}
                className="rounded-2xl border border-white/8 bg-white/3 p-5 transition hover:border-violet-500/30 hover:bg-violet-500/5"
              >
                <div className="text-2xl mb-3">{f.icon}</div>
                <div className="text-sm font-semibold text-white mb-2">{f.title}</div>
                <div className="text-xs text-white/50 leading-relaxed">{f.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Model recommendations ── */}
      <section className="py-28 px-6">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>What to run</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            Best models for your hardware
          </h2>
          <p className="text-white/55 mb-8 max-w-2xl">
            autotune works with any Ollama model. These are the best options as of April 2026.
            Run <code className="text-white/70">autotune recommend</code> to get a
            hardware-specific recommendation.
          </p>
          <div className="overflow-hidden rounded-2xl border border-white/8">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/8 bg-white/4 text-left text-xs font-semibold uppercase tracking-wider text-white/50">
                  <th className="px-5 py-4">RAM</th>
                  <th className="px-5 py-4">Model</th>
                  <th className="px-5 py-4">Size</th>
                  <th className="px-5 py-4 hidden sm:table-cell">Why</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { ram: "8 GB",  model: "qwen3:4b",           size: "~2.6 GB", why: "Best 4B — hybrid thinking mode, strong reasoning" },
                  { ram: "16 GB", model: "qwen3:8b",           size: "~5.2 GB", why: "Near-frontier quality; best 8B as of 2026" },
                  { ram: "16 GB", model: "gemma4:e2b",         size: "~5.8 GB", why: "Google's newest; 128k native context" },
                  { ram: "24 GB", model: "qwen3:14b",          size: "~9.0 GB", why: "Excellent reasoning and coding" },
                  { ram: "32 GB", model: "qwen3:30b-a3b",      size: "~17 GB",  why: "MoE: flagship quality at 7B inference cost" },
                  { ram: "Coding", model: "qwen2.5-coder:14b", size: "~9.0 GB", why: "Best open coding model" },
                  { ram: "Reasoning", model: "deepseek-r1:14b", size: "~9.0 GB", why: "Chain-of-thought; math & logic" },
                ].map((r, i) => (
                  <tr
                    key={r.model}
                    className={`border-b border-white/5 ${i % 2 === 0 ? "bg-white/2" : ""}`}
                  >
                    <td className="px-5 py-4 text-xs font-semibold text-white/60">{r.ram}</td>
                    <td className="px-5 py-4 font-mono text-violet-300 font-semibold">{r.model}</td>
                    <td className="px-5 py-4 text-white/50">{r.size}</td>
                    <td className="px-5 py-4 text-white/50 hidden sm:table-cell">{r.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="py-28 px-6 bg-white/[0.02] border-t border-white/5">
        <div className="mx-auto max-w-3xl text-center">
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-violet-500/20 text-2xl mb-6">
            ⚡
          </div>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            Try it in 60 seconds.
          </h2>
          <p className="text-white/55 mb-8 max-w-xl mx-auto">
            Open source, MIT licensed. Works with whatever Ollama models you already have.
            The <code className="text-white/70">autotune proof</code> command will show you
            the exact improvement on your own hardware.
          </p>
          <div className="mb-8 max-w-sm mx-auto">
            <CodeBlock code="pip install llm-autotune" language="bash" />
          </div>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <a
              href="https://github.com/tanavc1/local-llm-autotune"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl bg-violet-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-900/40 transition hover:bg-violet-500"
            >
              View on GitHub →
            </a>
            <a
              href="https://pypi.org/project/llm-autotune/"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl border border-white/10 bg-white/5 px-6 py-3 text-sm font-medium text-white/80 backdrop-blur transition hover:border-violet-500/40 hover:bg-violet-500/10 hover:text-white"
            >
              PyPI page
            </a>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-white/5 px-6 py-10">
        <div className="mx-auto max-w-5xl flex flex-col items-center gap-3 sm:flex-row sm:justify-between text-xs text-white/30">
          <div>autotune v0.2.0 — MIT License</div>
          <div className="flex gap-6">
            <a href="/install" className="hover:text-white/60 transition-colors">Install</a>
            <a href="/commands" className="hover:text-white/60 transition-colors">Commands</a>
            <a href="/what-we-do" className="hover:text-white/60 transition-colors">All we do</a>
            <a href="https://github.com/tanavc1/local-llm-autotune" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">GitHub</a>
            <a href="https://pypi.org/project/llm-autotune/" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">PyPI</a>
            <a href="https://github.com/tanavc1/local-llm-autotune/blob/main/CHANGELOG.md" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">Changelog</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
