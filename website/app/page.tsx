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
            autotune wraps Ollama in a smart proxy that right-sizes the KV buffer for each
            request — freeing{" "}
            <strong className="text-white/90">300+ MB of RAM</strong> back to your browser
            and cutting time-to-first-word by up to{" "}
            <strong className="text-white/90">53%</strong>.
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
            The problem: Ollama over-allocates memory.
          </h2>
          <p className="text-white/55 mb-12 max-w-2xl">
            Every request you send, Ollama reserves a fixed 4,096-token KV buffer — regardless
            of how long your actual message is. For a typical 100-word message, that&apos;s
            about 10× more memory than needed. autotune measures the actual prompt, allocates
            only what&apos;s required, and returns the rest to your system.
          </p>

          <div className="grid gap-6 sm:grid-cols-2">
            {/* Without */}
            <div className="rounded-2xl border border-red-500/15 bg-red-500/5 p-6">
              <div className="text-xs font-semibold uppercase tracking-wider text-red-400/80 mb-4">
                Without autotune
              </div>
              <div className="space-y-3 text-sm text-white/70">
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 shrink-0 text-red-400/60">→</span>
                  <span>You send a message (100 words ≈ 340 tokens)</span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 shrink-0 text-red-400/60">→</span>
                  <span>
                    Ollama reserves a fixed <strong className="text-white/85">448 MB</strong>{" "}
                    KV buffer for 4,096 tokens
                  </span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 shrink-0 text-red-400/60">→</span>
                  <span>Initializes the full buffer before generating token 1</span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 shrink-0 text-red-400/60">→</span>
                  <span>Chrome, Slack, VS Code fight for the remaining RAM</span>
                </div>
              </div>
            </div>

            {/* With */}
            <div className="rounded-2xl border border-green-500/20 bg-green-500/5 p-6">
              <div className="text-xs font-semibold uppercase tracking-wider text-green-400/80 mb-4">
                With autotune
              </div>
              <div className="space-y-3 text-sm text-white/70">
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 shrink-0 text-green-400/60">→</span>
                  <span>You send the same message (340 tokens)</span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 shrink-0 text-green-400/60">→</span>
                  <span>
                    autotune computes what&apos;s actually needed:{" "}
                    <strong className="text-white/85">155 MB</strong> for 1,405 tokens
                  </span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 shrink-0 text-green-400/60">→</span>
                  <span>Smaller buffer initializes faster — first word arrives sooner</span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 shrink-0 text-green-400/60">→</span>
                  <span>
                    <strong className="text-green-300">293 MB freed</strong> — back to your other
                    apps, every single request
                  </span>
                </div>
              </div>
            </div>
          </div>

          <p className="mt-6 text-xs text-white/35 max-w-2xl">
            Numbers above are for llama3.2:3b on Apple M2. KV buffer size scales with model
            architecture — larger models free more RAM in absolute terms.
            Generation quality and speed are identical: autotune changes only the buffer size,
            not the model weights or sampling.
          </p>
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
            <a href="https://github.com/tanavc1/local-llm-autotune" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">GitHub</a>
            <a href="https://pypi.org/project/llm-autotune/" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">PyPI</a>
            <a href="https://github.com/tanavc1/local-llm-autotune/blob/main/CHANGELOG.md" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">Changelog</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
