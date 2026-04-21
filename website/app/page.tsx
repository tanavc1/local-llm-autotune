import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "autotune — Local LLM Inference Optimizer",
};

// ─── Data ─────────────────────────────────────────────────────────────────────

const stats = [
  { value: "−39%", label: "Time to first word", sub: "up to −53% on qwen3:8b" },
  { value: "−67%", label: "KV cache size", sub: "3× less memory allocated" },
  { value: "122k", label: "KV slots freed", sub: "across benchmark runs" },
  { value: "0", label: "Swap events", sub: "across all models tested" },
];

const benchmarks = [
  { model: "llama3.2:3b", ttft: "−35%", kv: "−66%", ram: "−11%" },
  { model: "gemma4:e2b",  ttft: "−29%", kv: "−69%", ram: "−0%" },
  { model: "qwen3:8b",   ttft: "−53%", kv: "−66%", ram: "−7%" },
];

const features = [
  {
    icon: "⚡",
    title: "Dynamic KV Sizing",
    desc: "Computes the exact num_ctx each request needs — 4–8× less KV cache than Ollama's fixed 4096-token buffer.",
  },
  {
    icon: "🔒",
    title: "KV Prefix Caching",
    desc: "Pins system-prompt tokens via num_keep so they're never re-evaluated on subsequent turns.",
  },
  {
    icon: "🧠",
    title: "Adaptive KV Precision",
    desc: "Downgrades F16 → Q8 under memory pressure — three thresholds (80%, 88%, 93%) with automatic context reduction.",
  },
  {
    icon: "♾️",
    title: "Model Keep-Alive",
    desc: "Holds the model in unified memory between turns — eliminates cold-start reload latency on every request.",
  },
  {
    icon: "🤖",
    title: "Agentic Stability",
    desc: "Sizes KV for the full session ceiling before the loop starts. Prefix caching compounds across turns — TTFT falls as context grows.",
  },
  {
    icon: "🔌",
    title: "OpenAI-Compatible API",
    desc: "Drop-in server at localhost:8765/v1 — works with any OpenAI SDK, LangChain, or agent framework.",
  },
  {
    icon: "📊",
    title: "4-Tier Context Management",
    desc: "FULL → RECENT+FACTS → COMPRESSED → EMERGENCY. Deterministic fact extraction. Never truncates mid-sentence.",
  },
  {
    icon: "🍎",
    title: "MLX Backend",
    desc: "On M-series Macs, routes inference to MLX-LM for native Metal GPU kernels — 10–40% faster throughput.",
  },
];

const agentMetrics = [
  { metric: "Model reloads per session", raw: "0–1", tuned: "~0" },
  { metric: "Swap events", raw: "1 of 3 trials", tuned: "0" },
  { metric: "TTFT trend per turn", raw: "−101 ms/turn", tuned: "−435 ms/turn" },
  { metric: "Tool call errors", raw: "1 avg", tuned: "0" },
  { metric: "Context at session end", raw: "3,043 tokens", tuned: "1,946 (−36%)" },
];

const codeSnippet = `import autotune
from openai import OpenAI

# Start autotune server (safe to call on every launch)
autotune.start()

# Standard OpenAI client — all optimization is automatic
client = OpenAI(**autotune.client_kwargs())

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Hello!"}],
)`;

const installSnippet = `pip install llm-autotune
autotune chat --model qwen3:8b`;

// ─── Components ───────────────────────────────────────────────────────────────

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

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function Home() {
  return (
    <div className="min-h-screen text-[#e8e8f0] selection:bg-violet-500/30">
      {/* Nav */}
      <nav className="fixed top-0 z-50 w-full border-b border-white/5 bg-[#09090f]/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold tracking-tight text-white">autotune</span>
            <span className="hidden rounded-full bg-violet-500/20 px-2 py-0.5 text-xs font-medium text-violet-300 sm:block">
              v0.2.0
            </span>
          </div>
          <div className="flex items-center gap-6 text-sm text-white/60">
            <a href="#benchmarks" className="hidden hover:text-white transition-colors sm:block">
              Benchmarks
            </a>
            <a href="#features" className="hidden hover:text-white transition-colors sm:block">
              Features
            </a>
            <a href="#quickstart" className="hidden hover:text-white transition-colors sm:block">
              Quickstart
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

      {/* Hero */}
      <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 text-center pt-16">
        {/* Background gradient */}
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -top-32 left-1/2 h-[600px] w-[600px] -translate-x-1/2 rounded-full bg-violet-600/10 blur-[100px]" />
          <div className="absolute bottom-0 left-0 h-[400px] w-[400px] rounded-full bg-blue-600/8 blur-[80px]" />
        </div>

        <div className="relative z-10 flex flex-col items-center gap-6 max-w-4xl">
          <Badge>Open Source · MIT · Python 3.10+</Badge>

          <h1 className="text-5xl font-bold tracking-tight text-white sm:text-6xl lg:text-7xl animate-fade-up">
            Local LLMs,{" "}
            <span className="bg-gradient-to-r from-violet-400 to-blue-400 bg-clip-text text-transparent">
              actually fast
            </span>
          </h1>

          <p className="max-w-2xl text-lg text-white/60 leading-relaxed animate-fade-up delay-100">
            autotune cuts time-to-first-word by <strong className="text-white/90">39%</strong> and
            KV cache by <strong className="text-white/90">67%</strong> — without touching your existing
            code. Drop it in front of Ollama, LM Studio, or MLX and everything gets faster.
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
              href="https://github.com/tanavc1/local-llm-autotune"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl border border-white/10 bg-white/5 px-6 py-3 text-sm font-medium text-white/80 backdrop-blur transition hover:border-violet-500/40 hover:bg-violet-500/10 hover:text-white"
            >
              View on GitHub
            </a>
          </div>
        </div>

        {/* Stats row */}
        <div className="relative z-10 mt-20 grid grid-cols-2 gap-4 sm:grid-cols-4 max-w-4xl w-full animate-fade-up delay-400">
          {stats.map((s) => (
            <div
              key={s.label}
              className="rounded-2xl border border-white/8 bg-white/4 p-5 text-center backdrop-blur"
            >
              <div className="text-3xl font-bold text-violet-300">{s.value}</div>
              <div className="mt-1 text-sm font-medium text-white/80">{s.label}</div>
              <div className="mt-0.5 text-xs text-white/40">{s.sub}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Benchmarks */}
      <section id="benchmarks" className="py-28 px-6">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Measured results</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            Real numbers, real hardware
          </h2>
          <p className="text-white/55 mb-3 max-w-2xl">
            Benchmarked on Apple M2 16 GB using Ollama&apos;s internal Go nanosecond timers —
            not wall-clock estimates. Wilcoxon signed-rank testing with Cohen&apos;s d effect size
            across 3 runs × 5 prompt types.
          </p>
          <p className="text-white/40 text-sm mb-10">
            Honest caveat: Turn 1 is ~80% slower because autotune pre-allocates a larger KV window
            for the full session. From turn 2 onward prefix caching compounds — TTFT falls instead of growing.
          </p>

          {/* Benchmark table */}
          <div className="overflow-hidden rounded-2xl border border-white/8">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/8 bg-white/4 text-left text-xs font-semibold uppercase tracking-wider text-white/50">
                  <th className="px-6 py-4">Model</th>
                  <th className="px-6 py-4">TTFT</th>
                  <th className="px-6 py-4">KV cache</th>
                  <th className="px-6 py-4">Peak RAM</th>
                </tr>
              </thead>
              <tbody>
                {benchmarks.map((b, i) => (
                  <tr
                    key={b.model}
                    className={`border-b border-white/5 ${i % 2 === 0 ? "bg-white/2" : ""}`}
                  >
                    <td className="px-6 py-4 font-mono text-white/80">{b.model}</td>
                    <td className="px-6 py-4 font-semibold text-green-400">{b.ttft}</td>
                    <td className="px-6 py-4 font-semibold text-green-400">{b.kv}</td>
                    <td className="px-6 py-4 font-semibold text-green-400">{b.ram}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-6 text-xs text-white/30">
            Generation speed (tok/s) unchanged — Metal GPU-bound, not affected by software layers above it.
            Run <code className="text-white/50">python scripts/proof_suite.py</code> to reproduce on your hardware.
          </p>

          {/* KPI breakdown */}
          <div className="mt-16 grid gap-6 sm:grid-cols-3">
            <div className="rounded-2xl border border-white/8 bg-white/3 p-6">
              <div className="text-2xl font-bold text-violet-300 mb-2">−39% TTFT</div>
              <p className="text-sm text-white/55">
                Raw Ollama allocates a fixed 4096-token KV buffer — zeros it, initialises it — before generating token 1.
                autotune computes the minimum buffer each request needs. That&apos;s what you&apos;re waiting for.
              </p>
            </div>
            <div className="rounded-2xl border border-white/8 bg-white/3 p-6">
              <div className="text-2xl font-bold text-violet-300 mb-2">3× less KV</div>
              <p className="text-sm text-white/55">
                For a typical chat message: 4,096 → 1,302 tokens = ~224 MB of KV cache never allocated on qwen3:8b.
                No prompt tokens dropped — <code className="text-white/70">prompt_eval_count</code> is identical.
              </p>
            </div>
            <div className="rounded-2xl border border-white/8 bg-white/3 p-6">
              <div className="text-2xl font-bold text-violet-300 mb-2">Zero swap</div>
              <p className="text-sm text-white/55">
                Zero swap events across all 3 models × 5 prompt types × 3 runs.
                The <code className="text-white/70">--no-swap</code> flag guarantees this even on tight hardware.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Agentic section */}
      <section className="py-28 px-6 bg-white/[0.02] border-y border-white/5">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Multi-turn & agent workloads</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            Where it matters most
          </h2>
          <p className="text-white/55 mb-10 max-w-2xl">
            Raw Ollama&apos;s fixed <code className="text-white/80">num_ctx=4096</code> causes model
            reloads when agent context grows past the KV window. autotune computes a single session
            ceiling before the loop starts, then holds it constant. Prefix caching compounds — TTFT
            falls as the session grows, not climbs.
          </p>

          <div className="overflow-hidden rounded-2xl border border-white/8 mb-8">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/8 bg-white/4 text-left text-xs font-semibold uppercase tracking-wider text-white/50">
                  <th className="px-6 py-4">Metric</th>
                  <th className="px-6 py-4">Raw Ollama</th>
                  <th className="px-6 py-4">autotune</th>
                </tr>
              </thead>
              <tbody>
                {agentMetrics.map((m, i) => (
                  <tr
                    key={m.metric}
                    className={`border-b border-white/5 ${i % 2 === 0 ? "bg-white/2" : ""}`}
                  >
                    <td className="px-6 py-4 text-white/70">{m.metric}</td>
                    <td className="px-6 py-4 text-red-400/80 font-mono">{m.raw}</td>
                    <td className="px-6 py-4 text-green-400 font-mono font-semibold">{m.tuned}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-xs text-white/35">
            Measured on llama3.2:3b, multi-turn tool-calling agent task. Turn 1 TTFT is 80% slower (expected —
            larger KV pre-allocated). From turn 2 onward prefix cache compounds.
            Full data: <code className="text-white/50">AGENT_BENCHMARK.md</code>
          </p>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-28 px-6">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>What&apos;s inside</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-12">
            Every optimization, automatic
          </h2>

          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {features.map((f) => (
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

      {/* Quickstart + Code */}
      <section id="quickstart" className="py-28 px-6 bg-white/[0.02] border-y border-white/5">
        <div className="mx-auto max-w-5xl">
          <div className="grid gap-12 lg:grid-cols-2">
            {/* Left: steps */}
            <div>
              <SectionLabel>Quickstart</SectionLabel>
              <h2 className="text-3xl font-bold text-white mb-8">Up in 60 seconds</h2>
              <ol className="space-y-6">
                {[
                  {
                    n: "1",
                    title: "Install Ollama",
                    body: "Download from ollama.com, then pull a model.",
                    code: "ollama pull qwen3:8b",
                  },
                  {
                    n: "2",
                    title: "Install autotune",
                    body: "One pip install — no other dependencies required.",
                    code: "pip install llm-autotune",
                  },
                  {
                    n: "3",
                    title: "Check your hardware",
                    body: "autotune profiles your machine and recommends the right settings.",
                    code: "autotune hardware",
                  },
                  {
                    n: "4",
                    title: "Start chatting",
                    body: "All optimizations are automatic. No config files to edit.",
                    code: "autotune chat --model qwen3:8b",
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
                  Native Metal GPU kernels via MLX — 10–40% faster throughput.
                </div>
                <code className="text-xs font-mono text-green-300">pip install &quot;llm-autotune[mlx]&quot;</code>
              </div>
            </div>

            {/* Right: code */}
            <div className="flex flex-col gap-5">
              <div>
                <div className="text-sm font-semibold text-white/70 mb-3">Embed in your application</div>
                <CodeBlock code={codeSnippet} language="python" />
              </div>

              <div>
                <div className="text-sm font-semibold text-white/70 mb-3">Or run as a server</div>
                <CodeBlock
                  code={`autotune serve\n# → http://localhost:8765/v1\n# Any OpenAI client works. All optimizations automatic.`}
                  language="bash"
                />
              </div>

              <div className="rounded-2xl border border-white/8 bg-white/3 p-4 text-xs text-white/50 space-y-1">
                <div className="font-semibold text-white/70 mb-2">Profiles</div>
                <div className="flex gap-2"><span className="text-violet-300 font-mono">fast</span> <span>2k ctx · Q8 KV · temp 0.1 · quick lookups</span></div>
                <div className="flex gap-2"><span className="text-violet-300 font-mono">balanced</span> <span>8k ctx · F16 KV · temp 0.7 · general use (default)</span></div>
                <div className="flex gap-2"><span className="text-violet-300 font-mono">quality</span> <span>32k ctx · F16 KV · temp 0.8 · long-form writing</span></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Models section */}
      <section className="py-28 px-6">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Model recommendations</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-10">
            What to run on your hardware
          </h2>
          <div className="overflow-hidden rounded-2xl border border-white/8">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/8 bg-white/4 text-left text-xs font-semibold uppercase tracking-wider text-white/50">
                  <th className="px-6 py-4">RAM</th>
                  <th className="px-6 py-4">Model</th>
                  <th className="px-6 py-4">Size</th>
                  <th className="px-6 py-4 hidden sm:table-cell">Why</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { ram: "8 GB",  model: "qwen3:4b",           size: "~2.6 GB", why: "Best 4B; hybrid thinking mode" },
                  { ram: "16 GB", model: "qwen3:8b",           size: "~5.2 GB", why: "Near-frontier; best 8B as of 2026" },
                  { ram: "16 GB", model: "gemma4",             size: "~5.8 GB", why: "Google's newest; 128k context" },
                  { ram: "24 GB", model: "qwen3:14b",          size: "~9.0 GB", why: "Excellent reasoning" },
                  { ram: "32 GB", model: "qwen3:30b-a3b",      size: "~17 GB",  why: "MoE: flagship quality at 7B cost" },
                  { ram: "Coding", model: "qwen2.5-coder:14b", size: "~9.0 GB", why: "Best open coding model" },
                  { ram: "Reasoning", model: "deepseek-r1:14b", size: "~9.0 GB", why: "Chain-of-thought; math & logic" },
                ].map((r, i) => (
                  <tr
                    key={r.model}
                    className={`border-b border-white/5 ${i % 2 === 0 ? "bg-white/2" : ""}`}
                  >
                    <td className="px-6 py-4 text-xs font-semibold text-white/60">{r.ram}</td>
                    <td className="px-6 py-4 font-mono text-violet-300 font-semibold">{r.model}</td>
                    <td className="px-6 py-4 text-white/50">{r.size}</td>
                    <td className="px-6 py-4 text-white/50 hidden sm:table-cell">{r.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-xs text-white/30">
            Run <code className="text-white/50">autotune ls</code> to see how each installed model scores against your specific hardware.
          </p>
        </div>
      </section>

      {/* CTA */}
      <section className="py-28 px-6">
        <div className="mx-auto max-w-3xl text-center">
          <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-violet-500/20 text-2xl mb-6">
            ⚡
          </div>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-4">
            Ready to try it?
          </h2>
          <p className="text-white/55 mb-8">
            Open source, MIT licensed. Works with whatever Ollama models you already have.
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

      {/* Footer */}
      <footer className="border-t border-white/5 px-6 py-10">
        <div className="mx-auto max-w-5xl flex flex-col items-center gap-3 sm:flex-row sm:justify-between text-xs text-white/30">
          <div>autotune v0.2.0 — MIT License</div>
          <div className="flex gap-6">
            <a href="https://github.com/tanavc1/local-llm-autotune" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">GitHub</a>
            <a href="https://pypi.org/project/llm-autotune/" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">PyPI</a>
            <a href="https://github.com/tanavc1/local-llm-autotune/blob/main/CHANGELOG.md" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">Changelog</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
