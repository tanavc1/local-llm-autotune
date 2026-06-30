import type { Metadata } from "next";
import { CopyButton } from "./components/CopyButton";

export const dynamic = "force-static";

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
    model: "gemma3n:e4b",
    kvRaw: "96 MB",
    kvTuned: "30 MB",
    kvFreed: "66 MB",
    ttft: "−29%",
    speed: "unchanged",
  },
];

const installSnippet = `pip install llm-autotune
autotune start`;

const dockerSnippet = `# Build once
docker build -t autotune .

# Run — autotune on :8765, models cached in a volume
docker run -p 8765:8765 \\
  -v ollama_models:/root/.ollama \\
  -e OLLAMA_MODEL=qwen3:8b \\
  autotune`;

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
        <CopyButton text={code} />
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
          <a href="/" className="flex items-center gap-2">
            <span className="text-lg font-bold tracking-tight text-white">autotune</span>
            <span className="hidden rounded-full bg-violet-500/20 px-2 py-0.5 text-xs font-medium text-violet-300 sm:block">
              v1.6.0
            </span>
          </a>
          <div className="flex items-center gap-6 text-sm text-white/60">
            <a href="#how-it-works" className="hidden hover:text-white transition-colors sm:block">
              How it works
            </a>
            <a href="#benchmarks" className="hidden hover:text-white transition-colors sm:block">
              Benchmarks
            </a>
            <a href="#dashboard" className="hidden hover:text-white transition-colors sm:block">
              Dashboard
            </a>
            <a href="#quickstart" className="hidden hover:text-white transition-colors sm:block">
              Install
            </a>
            <a href="#docker" className="hidden hover:text-white transition-colors sm:block">
              Docker
            </a>
            <a href="/what-we-do" className="hidden hover:text-white transition-colors sm:block">
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
      <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 text-center pt-16 pb-20">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -top-32 left-1/2 h-[600px] w-[600px] -translate-x-1/2 rounded-full bg-violet-600/10 blur-[100px]" />
          <div className="absolute bottom-0 left-0 h-[400px] w-[400px] rounded-full bg-blue-600/8 blur-[80px]" />
        </div>

        <div className="relative z-10 flex flex-col items-center gap-6 max-w-4xl mt-10 sm:mt-16">
          <Badge>Open source · MIT · pip install llm-autotune</Badge>

          <h1 className="text-5xl font-bold tracking-tight text-white sm:text-6xl lg:text-7xl animate-fade-up">
            Your local AI,{" "}
            <span className="bg-gradient-to-r from-violet-400 to-blue-400 bg-clip-text text-transparent">
              actually fast.
            </span>
          </h1>

          <p className="max-w-2xl text-lg text-white/60 leading-relaxed animate-fade-up delay-100">
            autotune sits between your code and Ollama and applies automatic optimizations:
            right-sized KV buffers, KV precision tuning, system prompt caching,
            intelligent context management, and model keep-alive.
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

          <div className="animate-fade-up delay-400">
            <a
              href="https://www.producthunt.com/products/autotune-llm?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-autotune-2"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                alt="Autotune - Allows local LLMs to run faster and smoother on your device. | Product Hunt"
                width="250"
                height="54"
                src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1134681&theme=light&t=1777947695225"
              />
            </a>
          </div>
        </div>

        {/* Stats row */}
        <div className="relative z-10 mt-32 grid grid-cols-2 gap-4 sm:grid-cols-4 max-w-4xl w-full animate-fade-up delay-400">
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
            Every request, sized exactly right.
          </h2>
          <p className="text-white/55 mb-12 max-w-2xl">
            autotune sits between your code and Ollama as a transparent proxy. Before each
            request reaches Ollama, autotune calculates the exact memory it needs, watches
            live RAM usage from your other apps, and adjusts automatically. No config. No
            changes to your code or output quality.
          </p>

          {/* 1 — Precise KV cache allocation */}
          <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-6 mb-5">
            <div className="flex items-center gap-2 mb-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-violet-500/20 text-xs font-bold text-violet-300">1</span>
              <span className="text-sm font-semibold text-white">Precise KV cache allocation — every single request</span>
            </div>
            <p className="text-sm text-white/60 mb-5">
              Every time Ollama runs your prompt, it must first allocate a block of RAM called
              the KV cache — it&apos;s where it stores the attention state for every token in the
              context window. By default, Ollama always allocates for 4,096 tokens. For a typical
              50-word message, that&apos;s allocating 12× more RAM than the message actually needs.
              autotune measures the real token count, adds a safe headroom buffer, and tells Ollama
              the exact minimum. That freed RAM goes back to your browser, your apps, your system.
            </p>
            <div className="grid gap-4 sm:grid-cols-3 mb-4">
              <div className="rounded-xl border border-white/8 bg-black/20 p-4 text-xs">
                <div className="text-white/35 text-[10px] uppercase tracking-wider mb-2">Formula autotune uses</div>
                <div className="font-mono text-violet-300 leading-relaxed">
                  ctx = input_tokens<br />
                  &nbsp;&nbsp;&nbsp;&nbsp;+ max_reply<br />
                  &nbsp;&nbsp;&nbsp;&nbsp;+ 256 (buffer)<br />
                  <span className="text-white/35">→ rounded to nearest bucket</span>
                </div>
              </div>
              <div className="rounded-xl border border-red-500/15 bg-red-500/5 p-4 text-xs">
                <div className="text-white/35 text-[10px] uppercase tracking-wider mb-2">Raw Ollama — qwen3:8b</div>
                <div className="font-mono text-white/60 text-sm font-bold">576 MB</div>
                <div className="text-white/40 mt-1">always allocated, every request</div>
                <div className="text-white/30 mt-0.5">for 4,096 tokens</div>
              </div>
              <div className="rounded-xl border border-green-500/20 bg-green-500/5 p-4 text-xs">
                <div className="text-white/35 text-[10px] uppercase tracking-wider mb-2">autotune — qwen3:8b</div>
                <div className="font-mono text-green-300 text-sm font-bold">195 MB</div>
                <div className="text-white/40 mt-1">381 MB returned to your system</div>
                <div className="text-white/30 mt-0.5">per typical chat request</div>
              </div>
            </div>
            <p className="text-xs text-white/35">
              Buckets (512, 768, 1024, 1536, 2048…) prevent Ollama from reallocating the Metal
              buffer on every call — requests with similar lengths reuse the same pre-allocated
              buffer, eliminating 100–300 ms of KV thrashing overhead per request.
            </p>
          </div>

          {/* 2 — Live memory pressure management */}
          <div className="rounded-2xl border border-orange-500/20 bg-orange-500/5 p-6 mb-5">
            <div className="flex items-center gap-2 mb-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-orange-500/20 text-xs font-bold text-orange-300">2</span>
              <span className="text-sm font-semibold text-white">Live pressure management — proactive RAM tier system</span>
            </div>
            <p className="text-sm text-white/60 mb-5">
              Right-sizing the KV cache at request time is the foundation. But RAM usage on your
              machine is dynamic: Chrome opens a tab, Xcode compiles, a background process wakes
              up. autotune reads the OS&apos;s RAM utilization percentage before every single request
              and applies two independent levers — context window size and KV precision — across
              four fixed tiers, maintaining headroom well before any swap risk develops.
            </p>

            {/* RAM pressure visualization */}
            <div className="rounded-xl border border-white/8 bg-black/25 p-4 mb-4">
              <div className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-4">
                Live RAM thresholds — checked before every request
              </div>
              <div className="space-y-2.5">
                <div className="flex items-stretch gap-3">
                  <div className="flex items-center">
                    <div className="h-full w-1 rounded-full bg-green-500/50 min-h-[36px]" />
                  </div>
                  <div className="flex-1 rounded-lg border border-green-500/15 bg-green-500/5 px-4 py-2.5 text-xs">
                    <div className="flex items-center justify-between gap-4 flex-wrap">
                      <div className="font-mono text-green-300 font-semibold">RAM &lt; 80%</div>
                      <div className="text-white/50">Full context window · KV at profile default (F16 or Q8)</div>
                    </div>
                  </div>
                </div>
                <div className="flex items-stretch gap-3">
                  <div className="flex items-center">
                    <div className="h-full w-1 rounded-full bg-yellow-500/50 min-h-[36px]" />
                  </div>
                  <div className="flex-1 rounded-lg border border-yellow-500/15 bg-yellow-500/5 px-4 py-2.5 text-xs">
                    <div className="flex items-center justify-between gap-4 flex-wrap">
                      <div className="font-mono text-yellow-300 font-semibold">RAM 80–88%</div>
                      <div className="text-white/50">Context trimmed −10% · KV precision unchanged</div>
                    </div>
                  </div>
                </div>
                <div className="flex items-stretch gap-3">
                  <div className="flex items-center">
                    <div className="h-full w-1 rounded-full bg-orange-500/50 min-h-[36px]" />
                  </div>
                  <div className="flex-1 rounded-lg border border-orange-500/15 bg-orange-500/5 px-4 py-2.5 text-xs">
                    <div className="flex items-center justify-between gap-4 flex-wrap">
                      <div className="font-mono text-orange-300 font-semibold">RAM 88–93%</div>
                      <div className="text-white/50">Context −25% · KV switches F16 → Q8 (halves KV memory)</div>
                    </div>
                  </div>
                </div>
                <div className="flex items-stretch gap-3">
                  <div className="flex items-center">
                    <div className="h-full w-1 rounded-full bg-red-500/50 min-h-[36px]" />
                  </div>
                  <div className="flex-1 rounded-lg border border-red-500/15 bg-red-500/5 px-4 py-2.5 text-xs">
                    <div className="flex items-center justify-between gap-4 flex-wrap">
                      <div className="font-mono text-red-300 font-semibold">RAM &gt; 93%</div>
                      <div className="text-white/50">Context halved · KV forced Q8 · prevents disk swap</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <p className="text-xs text-white/40">
              KV precision switching (F16 → Q8) cuts the KV cache&apos;s RAM footprint in half
              instantly — with no meaningful quality impact. Q8 stores each attention value in 1
              byte instead of 2; the difference in model output is undetectable in practice.
              These adjustments happen automatically — you see a brief note in the chat UI when one fires.
              This is a heuristic tier system based on RAM percentage. autotune also runs a separate
              exact-math pre-flight check (NoSwapGuard) that computes precise KV bytes using your
              model&apos;s architecture — that system only fires when swap is mathematically certain.
            </p>
          </div>

          {/* 3 & 4 side by side */}
          <div className="grid gap-5 sm:grid-cols-2">

            {/* 3 — System prompt caching */}
            <div className="rounded-2xl border border-blue-500/20 bg-blue-500/5 p-6">
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-500/20 text-xs font-bold text-blue-300">3</span>
                <span className="text-sm font-semibold text-white">System prompt prefix caching</span>
              </div>
              <p className="text-sm text-white/60 mb-4">
                In any multi-turn chat, Ollama re-processes your entire system prompt from scratch
                on every message. autotune pins those tokens in the KV cache so they&apos;re
                only ever evaluated once — at the start. Every follow-up turn gets faster because
                fewer tokens need processing. The savings compound with every turn.
              </p>
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex gap-2 items-baseline">
                  <span className="text-red-300/70 w-14 shrink-0">Turn 1</span>
                  <span className="text-white/40">system prompt + message evaluated</span>
                </div>
                <div className="flex gap-2 items-baseline">
                  <span className="text-blue-300 w-14 shrink-0">Turn 2+</span>
                  <span className="text-blue-300/80">system prompt skipped — new tokens only</span>
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
                Ollama unloads the model after 5 minutes idle — a 1–4 second reload every time
                you come back to it. autotune keeps the model resident in memory between sessions.
                The weights were already using that RAM; keeping them there costs nothing extra
                and eliminates the cold-start delay entirely.
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg border border-red-500/15 bg-red-500/5 px-3 py-2">
                  <div className="text-white/40 mb-0.5">Raw Ollama</div>
                  <div className="font-mono text-white/70">1–4s reload</div>
                  <div className="text-white/35">after 5 min idle</div>
                </div>
                <div className="rounded-lg border border-green-500/20 bg-green-500/5 px-3 py-2">
                  <div className="text-white/40 mb-0.5">autotune</div>
                  <div className="font-mono text-green-300">stays loaded</div>
                  <div className="text-white/35">instant first token</div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 flex flex-col sm:flex-row sm:items-center gap-4">
            <p className="text-xs text-white/35 max-w-xl">
              Benchmark numbers use qwen3:8b / llama3.2:3b on Apple M2 16 GB.
              KV savings scale with model size — larger models free more RAM in absolute terms.
              Generation speed and output quality are unchanged: autotune touches only buffer
              sizes, precision, and scheduling — never model weights or sampling.
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

      {/* ── Dashboard ── */}
      <section id="dashboard" className="py-28 px-6">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Built-in dashboard</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            See every optimization, live.
          </h2>
          <p className="text-white/55 mb-10 max-w-2xl">
            autotune ships a full monitoring and control dashboard — no extra install, no
            external service, nothing sent to the cloud. Run{" "}
            <code className="text-violet-300">autotune serve</code> and open{" "}
            <code className="text-violet-300">localhost:8765/dashboard</code> in any browser.
            It auto-refreshes every 10 seconds and shows exactly what autotune is doing to your
            requests in real time.
          </p>

          {/* Browser-chrome mockup */}
          <div className="rounded-2xl border border-white/10 bg-black/40 overflow-hidden shadow-2xl shadow-violet-950/40">
            {/* fake browser bar */}
            <div className="flex items-center gap-2 border-b border-white/8 bg-white/[0.03] px-4 py-2.5">
              <div className="flex gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-red-500/50" />
                <span className="h-2.5 w-2.5 rounded-full bg-yellow-500/50" />
                <span className="h-2.5 w-2.5 rounded-full bg-green-500/50" />
              </div>
              <div className="ml-3 flex-1 rounded-md bg-black/40 px-3 py-1 text-[11px] font-mono text-white/40">
                localhost:8765/dashboard
              </div>
              <span className="flex items-center gap-1.5 text-[11px] text-green-300">
                <span className="h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" /> live
              </span>
            </div>

            {/* dashboard body */}
            <div className="p-5 sm:p-6">
              {/* KPI row */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-4">
                {[
                  { label: "RAM used", val: "61%", tone: "text-green-300" },
                  { label: "Running models", val: "2", tone: "text-white" },
                  { label: "Requests today", val: "1,284", tone: "text-white" },
                  { label: "Avg TTFT", val: "0.42s", tone: "text-violet-300" },
                  { label: "Avg tok/s", val: "47.3", tone: "text-white" },
                  { label: "KV saved", val: "67%", tone: "text-green-300" },
                ].map((k) => (
                  <div key={k.label} className="rounded-xl border border-white/8 bg-white/[0.02] p-3">
                    <div className="text-[10px] uppercase tracking-wider text-white/35 mb-1">{k.label}</div>
                    <div className={`font-mono text-lg font-bold ${k.tone}`}>{k.val}</div>
                  </div>
                ))}
              </div>

              {/* panels */}
              <div className="grid gap-3 lg:grid-cols-3">
                {/* Requests — last 24h */}
                <div className="rounded-xl border border-white/8 bg-white/[0.02] p-4">
                  <div className="text-[10px] uppercase tracking-wider text-white/35 mb-3">Requests · last 24h</div>
                  <div className="flex h-20 items-end gap-1">
                    {[18, 26, 31, 22, 40, 55, 48, 62, 71, 58, 80, 67, 90, 76, 61, 49, 57, 44, 33, 41, 52, 38, 29, 24].map((h, i) => (
                      <div key={i} className="flex-1 rounded-sm bg-violet-500/40" style={{ height: `${h}%` }} />
                    ))}
                  </div>
                </div>

                {/* TTFT sparkline */}
                <div className="rounded-xl border border-white/8 bg-white/[0.02] p-4">
                  <div className="text-[10px] uppercase tracking-wider text-white/35 mb-3">TTFT · last 100 requests</div>
                  <div className="flex h-20 items-end gap-[3px]">
                    {[
                      30, 22, 41, 28, 35, 52, 26, 19, 44, 31, 24, 38, 60, 29, 21, 33, 47, 25,
                      18, 40, 27, 36, 55, 23, 30, 42, 20, 34, 48, 26,
                    ].map((h, i) => (
                      <div
                        key={i}
                        className={`flex-1 rounded-sm ${h > 50 ? "bg-yellow-400/60" : h > 35 ? "bg-blue-400/55" : "bg-green-400/55"}`}
                        style={{ height: `${h}%` }}
                      />
                    ))}
                  </div>
                  <div className="mt-2 text-[10px] text-white/30">green &lt; 0.5s · blue &lt; 1s · yellow slower</div>
                </div>

                {/* Raw vs Tuned */}
                <div className="rounded-xl border border-white/8 bg-white/[0.02] p-4">
                  <div className="text-[10px] uppercase tracking-wider text-white/35 mb-3">Raw vs Tuned · qwen3:8b</div>
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between rounded-lg border border-red-500/15 bg-red-500/5 px-3 py-2">
                      <span className="text-[11px] text-white/50">Ollama default</span>
                      <span className="font-mono text-sm font-bold text-white/60">576 MB</span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-green-500/20 bg-green-500/5 px-3 py-2">
                      <span className="text-[11px] text-white/50">autotune</span>
                      <span className="font-mono text-sm font-bold text-green-300">195 MB</span>
                    </div>
                    <div className="text-[11px] text-white/35">−66% KV cache · 381 MB freed per request</div>
                  </div>
                </div>
              </div>

              {/* Per-model breakdown */}
              <div className="mt-3 rounded-xl border border-white/8 bg-white/[0.02] p-4">
                <div className="text-[10px] uppercase tracking-wider text-white/35 mb-3">Per-model breakdown</div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-[12px]">
                    <thead className="text-[10px] uppercase tracking-wider text-white/30">
                      <tr>
                        <th className="pb-2 pr-4 font-medium">Model</th>
                        <th className="pb-2 pr-4 font-medium">Requests</th>
                        <th className="pb-2 pr-4 font-medium">Avg TTFT</th>
                        <th className="pb-2 pr-4 font-medium">Avg tok/s</th>
                        <th className="pb-2 font-medium">Avg context</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono text-white/60">
                      <tr className="border-t border-white/5">
                        <td className="py-1.5 pr-4 text-white/80">qwen3:8b</td>
                        <td className="py-1.5 pr-4">912</td>
                        <td className="py-1.5 pr-4 text-violet-300">0.39s</td>
                        <td className="py-1.5 pr-4">46.1</td>
                        <td className="py-1.5">1,536</td>
                      </tr>
                      <tr className="border-t border-white/5">
                        <td className="py-1.5 pr-4 text-white/80">qwen2.5-coder:7b</td>
                        <td className="py-1.5 pr-4">372</td>
                        <td className="py-1.5 pr-4 text-violet-300">0.48s</td>
                        <td className="py-1.5 pr-4">51.7</td>
                        <td className="py-1.5">2,048</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Panels included */}
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[
              { title: "Overview KPIs", body: "RAM pressure, running models, requests today, avg TTFT, avg tok/s, and KV savings vs Ollama's 4,096-token default." },
              { title: "Requests & TTFT charts", body: "24-hour request volume plus a 100-request TTFT sparkline coloured by latency tier — spot regressions at a glance." },
              { title: "Raw vs Tuned", body: "autotune's average dynamic context against Ollama's fixed default, with live context-reduction and KV-memory savings." },
              { title: "Per-model breakdown", body: "Requests, avg/min/max TTFT, tok/s, context and total tokens for every model routed through autotune." },
              { title: "API keys & slow requests", body: "Per-key usage today, and a feed of recent requests over 5 s with model, context, and profile." },
              { title: "Catalog & live Settings", body: "The full 43-model catalog scored for your machine, plus a read/write Settings panel for context, KV precision, keep-alive and more." },
            ].map((p) => (
              <div key={p.title} className="rounded-2xl border border-white/8 bg-white/[0.02] p-5">
                <div className="text-sm font-semibold text-white mb-1.5">{p.title}</div>
                <div className="text-xs text-white/50 leading-relaxed">{p.body}</div>
              </div>
            ))}
          </div>

          {/* Launch + security */}
          <div className="mt-8 grid gap-4 lg:grid-cols-2">
            <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-5">
              <div className="text-xs font-semibold text-violet-300 mb-2">Launch it in one command</div>
              <CodeBlock code={`export AUTOTUNE_ADMIN_KEY="your-secret-key"
autotune serve
# → open http://localhost:8765/dashboard`} language="bash" />
            </div>
            <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-5">
              <div className="text-xs font-semibold text-white mb-2">Secure by default</div>
              <ul className="flex flex-col gap-1.5 text-xs text-white/55">
                <li>· Login gated by <code className="text-white/70">AUTOTUNE_ADMIN_KEY</code> — no key, no dashboard.</li>
                <li>· HMAC-signed session cookies with server-side revocation on logout.</li>
                <li>· Sliding-window rate limits and CSP headers on every response.</li>
                <li>· 100% local — all metrics come from your own SQLite, nothing leaves your machine.</li>
              </ul>
            </div>
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
              <h2 className="text-3xl font-bold text-white mb-3">Up in 60 seconds</h2>
              <p className="text-sm text-white/50 mb-8">
                Two commands. No Ollama setup, no config — autotune handles everything.
              </p>
              <ol className="space-y-6">
                {[
                  {
                    n: "1",
                    title: "Install autotune",
                    body: "One pip install. Nothing else to configure.",
                    code: "pip install llm-autotune",
                  },
                  {
                    n: "2",
                    title: "Run the guided setup",
                    body: "Run this first. It verifies Ollama, picks and pulls the best model for your hardware, and proves the speedup — about 2 minutes.",
                    code: "autotune start",
                  },
                  {
                    n: "3",
                    title: "Start chatting with optimization",
                    body: "Every request is automatically right-sized. No flags, no config.",
                    code: "autotune chat --model qwen3:8b",
                  },
                  {
                    n: "4",
                    title: "Prove it on your own hardware",
                    body: "30-second benchmark using Ollama's own nanosecond timers. Saves a JSON you can share.",
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
                      <div className="flex items-center justify-between rounded-lg border border-white/8 bg-black/40 px-3 py-1.5 gap-2">
                        <code className="text-xs font-mono text-green-300 overflow-x-auto">{step.code}</code>
                        <CopyButton text={step.code} />
                      </div>
                    </div>
                  </li>
                ))}
              </ol>

              <div className="mt-8 rounded-2xl border border-blue-500/20 bg-blue-500/5 p-4">
                <div className="text-xs font-semibold text-blue-300 mb-1">Apple Silicon (M1/M2/M3/M4)</div>
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

      {/* ── Docker ── */}
      <section id="docker" className="py-28 px-6 bg-white/[0.02] border-y border-white/5">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Docker</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            Ollama + autotune, bundled.
          </h2>
          <p className="text-white/55 mb-10 max-w-2xl">
            The Docker image bundles Ollama and autotune in a single container.
            No local install needed — just pull the image, mount a volume for
            model storage, and your OpenAI-compatible endpoint is ready on
            port 8765.
          </p>

          <div className="grid gap-8 lg:grid-cols-2 items-start">
            <div>
              <CodeBlock code={dockerSnippet} language="bash" />
              <p className="mt-4 text-xs text-white/40">
                <code className="text-white/55">OLLAMA_MODEL</code> auto-pulls the
                model on first start. Models are cached in the named volume and
                persist across restarts.
              </p>
            </div>

            <div className="flex flex-col gap-4">
              <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-5">
                <div className="text-sm font-semibold text-white mb-2">docker-compose — two options</div>
                <div className="space-y-2 text-xs text-white/60">
                  <div className="flex gap-2">
                    <code className="text-violet-300 shrink-0">--profile single</code>
                    <span>Ollama + autotune in one container. Simplest setup.</span>
                  </div>
                  <div className="flex gap-2">
                    <code className="text-violet-300 shrink-0">--profile multi</code>
                    <span>Separate services. Lighter autotune image (~200 MB). Set <code className="text-white/60">AUTOTUNE_OLLAMA_URL=http://ollama:11434</code>.</span>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-white/8 bg-white/3 p-5">
                <div className="text-sm font-semibold text-white mb-3">Environment variables</div>
                <div className="space-y-2 text-xs font-mono">
                  {[
                    { key: "OLLAMA_MODEL", val: "qwen3:8b", desc: "auto-pull on first boot" },
                    { key: "AUTOTUNE_PORT", val: "8765", desc: "autotune bind port" },
                    { key: "AUTOTUNE_OLLAMA_URL", val: "http://ollama:11434", desc: "remote/multi-container Ollama" },
                  ].map((v) => (
                    <div key={v.key} className="flex flex-col gap-0.5">
                      <div className="flex gap-2 items-baseline">
                        <span className="text-violet-300 shrink-0">{v.key}</span>
                        <span className="text-white/30">= {v.val}</span>
                      </div>
                      <div className="text-white/35 pl-0 text-[11px]">{v.desc}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-green-500/20 bg-green-500/5 p-4 text-xs text-white/55">
                <span className="text-green-300 font-medium">GPU support:</span>{" "}
                Built on <code className="text-white/65">ollama/ollama:latest</code> — includes CUDA
                and ROCm layers. Add <code className="text-white/65">--gpus all</code> for NVIDIA,
                or mount <code className="text-white/65">/dev/kfd</code> for AMD.
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
              {
                title: "Docker — Ollama bundled",
                icon: "🐳",
                desc: "Single container with Ollama + autotune. Mount a volume for models, set OLLAMA_MODEL to auto-pull on first boot, and your OpenAI-compatible API is ready on :8765.",
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
            autotune works with any Ollama model. These are the best options as of June 2026.
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
                  { ram: "8 GB",   model: "qwen3.5:4b",       size: "~2.6 GB", why: "Newest small Qwen — hybrid thinking, 256k context" },
                  { ram: "16 GB",  model: "qwen3.5:9b",       size: "~5.6 GB", why: "Best overall 8–9B model (June 2026)" },
                  { ram: "16 GB",  model: "gpt-oss:20b",      size: "~14 GB",  why: "OpenAI MoE — ~o3-mini reasoning, fits 16 GB" },
                  { ram: "16 GB",  model: "gemma4:12b",       size: "~8.1 GB", why: "Multimodal with native audio" },
                  { ram: "24 GB",  model: "qwen3.6:27b",      size: "~17 GB",  why: "Best overall on consumer hardware — 77.2% SWE-bench" },
                  { ram: "32 GB",  model: "qwen3-coder:30b",  size: "~19 GB",  why: "MoE: best open coder, agentic, 256k context" },
                  { ram: "48 GB+", model: "gpt-oss:120b",     size: "~65 GB",  why: "Near-o3 reasoning you can self-host" },
                  { ram: "Coding", model: "devstral:24b",     size: "~14 GB",  why: "Agentic coding — multi-file edits & tool use" },
                  { ram: "Reasoning", model: "deepseek-r1:32b", size: "~20 GB", why: "Chain-of-thought; math & logic" },
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
        <div className="mx-auto max-w-5xl flex flex-col items-center gap-4 sm:flex-row sm:justify-between text-xs text-white/30">
          <div className="flex flex-col items-center sm:items-start gap-1">
            <span>autotune v1.6.0 — MIT License</span>
            <a href="mailto:autotunellm@gmail.com" className="hover:text-white/60 transition-colors">autotunellm@gmail.com</a>
          </div>
          <div className="flex flex-wrap justify-center gap-6">
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
