import type { Metadata } from "next";
import Link from "next/link";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "All we do — autotune",
  description:
    "Every optimization autotune applies — explained plainly. KV cache sizing, swap prevention, prefix caching, hardware tuning, conversation memory, and more.",
  openGraph: {
    title: "All we do — autotune",
    description:
      "Every optimization autotune applies, explained in plain language. No jargon.",
    type: "website",
  },
};

// ─── Shared components ────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-3">
      {children}
    </p>
  );
}

function Callout({
  color = "violet",
  children,
}: {
  color?: "violet" | "blue" | "green" | "orange" | "red";
  children: React.ReactNode;
}) {
  const styles: Record<string, string> = {
    violet: "border-violet-500/25 bg-violet-500/8",
    blue:   "border-blue-500/25 bg-blue-500/8",
    green:  "border-green-500/25 bg-green-500/8",
    orange: "border-orange-500/25 bg-orange-500/8",
    red:    "border-red-500/25 bg-red-500/8",
  };
  return (
    <div className={`rounded-xl border px-5 py-4 text-sm text-white/65 leading-relaxed ${styles[color]}`}>
      {children}
    </div>
  );
}

function Formula({ children }: { children: string }) {
  return (
    <div className="my-3 rounded-lg border border-white/10 bg-black/40 px-4 py-3 font-mono text-sm text-green-300 overflow-x-auto">
      {children}
    </div>
  );
}

function Compare({
  label,
  before,
  after,
  afterColor = "green",
}: {
  label: string;
  before: string;
  after: string;
  afterColor?: "green" | "blue" | "violet";
}) {
  const colors: Record<string, string> = {
    green:  "text-green-300",
    blue:   "text-blue-300",
    violet: "text-violet-300",
  };
  return (
    <div className="grid grid-cols-3 gap-2 text-xs items-center">
      <div className="text-white/40">{label}</div>
      <div className="rounded border border-red-500/15 bg-red-500/5 px-2 py-1.5 font-mono text-white/55 text-center">
        {before}
      </div>
      <div className={`rounded border border-green-500/20 bg-green-500/5 px-2 py-1.5 font-mono font-semibold text-center ${colors[afterColor]}`}>
        {after}
      </div>
    </div>
  );
}

function OptimizationCard({
  number,
  title,
  tag,
  tagColor = "violet",
  icon,
  children,
}: {
  number: string;
  title: string;
  tag: string;
  tagColor?: "violet" | "blue" | "green" | "orange" | "red";
  icon: string;
  children: React.ReactNode;
}) {
  const tagStyles: Record<string, string> = {
    violet: "border-violet-500/30 bg-violet-500/10 text-violet-300",
    blue:   "border-blue-500/30 bg-blue-500/10 text-blue-300",
    green:  "border-green-500/30 bg-green-500/10 text-green-300",
    orange: "border-orange-500/30 bg-orange-500/10 text-orange-300",
    red:    "border-red-500/30 bg-red-500/10 text-red-300",
  };
  return (
    <div className="rounded-2xl border border-white/8 bg-white/3 overflow-hidden">
      <div className="flex items-start gap-4 p-6 pb-0">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-xl">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <span className="text-xs font-bold text-white/30">#{number}</span>
            <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${tagStyles[tagColor]}`}>
              {tag}
            </span>
          </div>
          <h3 className="text-base font-bold text-white">{title}</h3>
        </div>
      </div>
      <div className="p-6 pt-4 text-sm text-white/60 leading-relaxed space-y-4">
        {children}
      </div>
    </div>
  );
}

// ─── Summary data ─────────────────────────────────────────────────────────────

const ALL_OPTIMIZATIONS = [
  { n: "01", name: "Precise KV allocation",        cat: "Memory",       impact: "Highest",  desc: "Right-sizes the KV cache to each request — frees 300+ MB per call",              color: "violet" },
  { n: "02", name: "Live pressure response",        cat: "Memory",       impact: "Highest",  desc: "Heuristic RAM % tiers — proactively trims context + KV precision long before swap risk",  color: "orange" },
  { n: "03", name: "System prompt caching",         cat: "Speed",        impact: "High",     desc: "Pins system prompt in KV — never re-evaluated after turn 1",                    color: "blue" },
  { n: "04", name: "Keep-alive",                    cat: "Speed",        impact: "High",     desc: "Model stays in RAM — eliminates 1–4s cold-reload between sessions",             color: "green" },
  { n: "05", name: "Bucket snapping",               cat: "Speed",        impact: "High",     desc: "Snaps context to stable sizes so Ollama reuses Metal buffers — no thrashing",   color: "violet" },
  { n: "06", name: "KV precision control",          cat: "Memory",       impact: "High",     desc: "F16 → Q8 under pressure — halves KV footprint with negligible quality impact",  color: "blue" },
  { n: "07", name: "Flash attention",               cat: "Speed",        impact: "Medium",   desc: "Reduces peak activation memory during prefill — zero quality impact",            color: "violet" },
  { n: "08", name: "NoSwapGuard",                   cat: "Safety",       impact: "Medium",   desc: "Exact-math hard guarantee — computes precise KV bytes from model arch; only fires when swap is certain", color: "orange" },
  { n: "09", name: "Larger prefill batch",          cat: "Speed",        impact: "Medium",   desc: "num_batch=1024 → fewer GPU passes for long prompts",                            color: "blue" },
  { n: "10", name: "Pre-flight model analysis",     cat: "Safety",       impact: "Medium",   desc: "Checks model fits before loading; suggests lighter quants if not",              color: "orange" },
  { n: "11", name: "Hardware tuner",                cat: "Intelligence", impact: "Moderate", desc: "QOS class, process priority, GC disable, CPU governor around inference",        color: "orange" },
  { n: "12", name: "Adaptive session advisor",      cat: "Intelligence", impact: "Moderate", desc: "Watches live metrics; takes graduated action before performance degrades",       color: "red" },
  { n: "13", name: "Context compressor",            cat: "Context",      impact: "Moderate", desc: "Compresses old messages in tiers when approaching context ceiling",              color: "violet" },
  { n: "14", name: "Conversation memory & recall",  cat: "Context",      impact: "Selective", desc: "Saves sessions locally; injects semantically relevant past context",            color: "blue" },
] as const;

const impactBadge: Record<string, string> = {
  "Highest":   "border-violet-500/40 bg-violet-500/15 text-violet-300",
  "High":      "border-green-500/30 bg-green-500/10 text-green-300",
  "Medium":    "border-blue-500/30 bg-blue-500/10 text-blue-300",
  "Moderate":  "border-white/15 bg-white/5 text-white/50",
  "Selective": "border-white/15 bg-white/5 text-white/50",
};

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function WhatWeDo() {
  return (
    <div className="min-h-screen text-[#e8e8f0]">

      {/* ── Nav ── */}
      <nav className="sticky top-0 z-50 w-full border-b border-white/5 bg-[#09090f]/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <Link href="/" className="flex items-center gap-2 text-sm font-bold text-white hover:text-violet-300 transition-colors">
            <svg className="h-4 w-4 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            autotune
          </Link>
          <div className="flex items-center gap-5 text-xs text-white/50">
            <a href="#all" className="hidden hover:text-white transition-colors sm:block">All 14</a>
            <a href="#memory" className="hidden hover:text-white transition-colors sm:block">Memory</a>
            <a href="#speed" className="hidden hover:text-white transition-colors sm:block">Speed</a>
            <a href="#intelligence" className="hidden hover:text-white transition-colors sm:block">Intelligence</a>
            <a href="#context" className="hidden hover:text-white transition-colors sm:block">Context</a>
            <a
              href="https://github.com/tanavc1/local-llm-autotune"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 font-medium text-white/70 transition hover:border-violet-500/40 hover:text-white"
            >
              <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
              </svg>
              GitHub
            </a>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="relative overflow-hidden px-6 pt-24 pb-12 text-center">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -top-32 left-1/2 h-[500px] w-[700px] -translate-x-1/2 rounded-full bg-violet-600/8 blur-[120px]" />
        </div>
        <div className="relative z-10 mx-auto max-w-3xl">
          <p className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-4">
            Full transparency
          </p>
          <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl mb-5">
            Everything autotune does,<br />
            <span className="bg-gradient-to-r from-violet-400 to-blue-400 bg-clip-text text-transparent">
              explained plainly.
            </span>
          </h1>
          <p className="text-lg text-white/55 leading-relaxed max-w-2xl mx-auto">
            No marketing copy. No jargon. Every single optimization — what it is, why it matters,
            and exactly how it works. Ordered by how much it actually helps.
          </p>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/* ── ALL 14 OPTIMIZATIONS AT A GLANCE ── */}
      {/* ══════════════════════════════════════════════════════════════════════ */}

      <section id="all" className="px-6 pb-16">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>All 14 optimizations</SectionLabel>
          <h2 className="text-2xl font-bold text-white mb-2">At a glance — ordered by impact</h2>
          <p className="text-white/45 text-sm mb-8 max-w-xl">
            Click any item below to jump to the full explanation.
          </p>

          <div className="grid gap-2.5 sm:grid-cols-2">
            {ALL_OPTIMIZATIONS.map((opt, i) => {
              const borderColors: Record<string, string> = {
                violet: "hover:border-violet-500/30",
                blue:   "hover:border-blue-500/25",
                green:  "hover:border-green-500/25",
                orange: "hover:border-orange-500/25",
                red:    "hover:border-red-500/25",
              };
              const dotColors: Record<string, string> = {
                violet: "bg-violet-400",
                blue:   "bg-blue-400",
                green:  "bg-green-400",
                orange: "bg-orange-400",
                red:    "bg-red-400",
              };
              const sectionMap: Record<number, string> = {
                0: "memory", 1: "memory", 2: "speed", 3: "speed", 4: "speed",
                5: "memory", 6: "speed", 7: "safety", 8: "speed", 9: "safety",
                10: "intelligence", 11: "intelligence", 12: "context", 13: "context",
              };
              return (
                <a
                  key={opt.n}
                  href={`#${sectionMap[i] ?? "memory"}`}
                  className={`flex items-start gap-3 rounded-xl border border-white/6 bg-white/3 px-4 py-3 transition-colors ${borderColors[opt.color]} hover:bg-white/5 group`}
                >
                  <div className="flex items-center gap-2 shrink-0 mt-0.5">
                    <span className="text-[10px] font-mono text-white/25">#{opt.n}</span>
                    <div className={`h-1.5 w-1.5 rounded-full ${dotColors[opt.color]}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-2 mb-0.5">
                      <span className="text-sm font-semibold text-white/85 group-hover:text-white transition-colors">{opt.name}</span>
                      <span className={`inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] font-medium ${impactBadge[opt.impact]}`}>
                        {opt.impact}
                      </span>
                    </div>
                    <p className="text-xs text-white/40 leading-relaxed">{opt.desc}</p>
                  </div>
                  <span className="text-white/20 group-hover:text-white/50 transition-colors text-xs mt-0.5 shrink-0">→</span>
                </a>
              );
            })}
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/* ── THE KV CACHE — FOUNDATION ── */}
      {/* ══════════════════════════════════════════════════════════════════════ */}

      <section className="px-6 py-20 bg-white/[0.02] border-y border-white/5">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Foundation</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-3">
            What is the KV cache?
          </h2>
          <p className="text-white/50 mb-10 max-w-2xl text-sm">
            Almost every optimization in autotune touches the KV cache in some way. Understanding it
            takes two minutes and makes everything else make sense.
          </p>

          <div className="grid gap-6 lg:grid-cols-2 mb-8">
            <div className="space-y-4 text-sm text-white/65 leading-relaxed">
              <p>
                When a language model generates text, it doesn&apos;t work one word at a time in isolation.
                Every new word it produces requires it to &ldquo;attend to&rdquo; — look back at — every
                previous word in the conversation. That backward look is the attention mechanism, and it&apos;s
                what makes LLMs coherent and context-aware.
              </p>
              <p>
                The problem: that backward look is expensive. Producing word #500 would require
                re-processing words #1 through #499 from scratch — hundreds of matrix multiplications,
                repeated, for every single token. A 1,000-word reply would require a billion redundant
                calculations.
              </p>
              <p>
                The solution is caching. When the model processes token #1, it computes two tables of
                numbers that represent &ldquo;what this token contributes to future attention.&rdquo; These
                tables are called <strong className="text-white/85">K (keys)</strong> and{" "}
                <strong className="text-white/85">V (values)</strong>. By storing — caching — them in RAM,
                the model can skip recomputing them for every future token. Token #500 just reads
                from the cache instead of redoing all that work.
              </p>
              <p>
                That cache is the <strong className="text-white/85">KV cache</strong>. It lives in your
                computer&apos;s RAM (or GPU VRAM — on Apple Silicon, they&apos;re the same pool). Its size
                is mathematically predictable:
              </p>
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border border-white/8 bg-white/3 p-5">
                <p className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-3">KV cache size formula</p>
                <Formula>2 × n_layers × kv_heads × head_dim × num_ctx × bytes</Formula>
                <div className="space-y-2 text-xs text-white/50 mt-4">
                  <div className="flex gap-3">
                    <span className="font-mono text-violet-300 w-24 shrink-0">n_layers</span>
                    <span>How many transformer layers the model has (e.g. 32 for a 7B model)</span>
                  </div>
                  <div className="flex gap-3">
                    <span className="font-mono text-violet-300 w-24 shrink-0">kv_heads</span>
                    <span>Number of KV attention heads (often fewer than total heads, via GQA)</span>
                  </div>
                  <div className="flex gap-3">
                    <span className="font-mono text-violet-300 w-24 shrink-0">head_dim</span>
                    <span>Dimension of each attention head (embedding size ÷ total heads)</span>
                  </div>
                  <div className="flex gap-3">
                    <span className="font-mono text-violet-300 w-24 shrink-0">num_ctx</span>
                    <span>How many tokens the context window is sized for — this is the big lever</span>
                  </div>
                  <div className="flex gap-3">
                    <span className="font-mono text-violet-300 w-24 shrink-0">bytes</span>
                    <span>F16 = 2 bytes per element, Q8 = 1 byte — halving this halves the cache</span>
                  </div>
                </div>
              </div>

              <Callout color="orange">
                <strong className="text-orange-300">The key insight:</strong> The KV cache scales
                linearly with <code className="text-white/70">num_ctx</code>. If you allocate
                a 4,096-token context when your prompt is only 200 tokens, you&apos;re wasting 95%
                of that memory. Ollama does this by default. autotune fixes it.
              </Callout>
            </div>
          </div>

          <div className="rounded-2xl border border-white/8 bg-black/20 overflow-hidden">
            <div className="border-b border-white/8 bg-white/3 px-5 py-3">
              <p className="text-xs font-semibold text-white/50 uppercase tracking-wider">
                Real example: qwen3:8b (48 layers, 8 KV heads, 128 head_dim)
              </p>
            </div>
            <div className="p-5 space-y-2">
              <Compare label="4,096 ctx (Ollama default)" before="576 MB KV" after="576 MB allocated" afterColor="violet" />
              <Compare label="2,048 ctx (autotune, typical)" before="576 MB KV" after="288 MB — 288 MB freed" />
              <Compare label="1,536 ctx (short message)" before="576 MB KV" after="216 MB — 360 MB freed" />
              <p className="text-xs text-white/30 pt-2">
                Every freed megabyte goes back to your system&apos;s available pool — your browser,
                other apps, and macOS all benefit. The model weights themselves don&apos;t change in size.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/* ── MEMORY OPTIMIZATIONS ── */}
      {/* ══════════════════════════════════════════════════════════════════════ */}

      <section id="memory" className="px-6 py-20">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Memory — highest impact</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-3">
            How autotune manages RAM.
          </h2>
          <p className="text-white/50 mb-12 max-w-2xl text-sm">
            RAM is the single most important resource for local LLM inference. Running out means
            your OS starts writing to your SSD (swap), which drops generation speed from 30+ tok/s
            to under 5 tok/s and makes your whole computer sluggish. autotune has four independent
            systems for keeping memory under control — ordered here from most to least impactful.
          </p>

          <div className="space-y-6">

            <OptimizationCard
              number="01"
              title="Precise KV allocation — dynamic context sizing"
              tag="Highest impact · Every request"
              tagColor="violet"
              icon="📐"
            >
              <p>
                Ollama allocates the full KV cache before generating the first token. With the
                default <code className="text-white/70">num_ctx=4096</code>, it zeros and initializes
                a 4,096-token buffer even if your prompt is 50 words. That initialization is part
                of what you wait for before the first token appears.
              </p>
              <p>
                autotune computes the minimum context that actually fits this specific request:
              </p>
              <Formula>num_ctx = clamp(input_tokens + max_new_tokens + 256, 512, profile_max)</Formula>
              <p>
                For a typical balanced-profile chat message: ~22-token prompt + 1024 max reply +
                256 buffer = 1,302 tokens. That maps to the 1,536 bucket (see optimization #05 below).
                On qwen3:8b, this frees 381 MB before a single token is generated — on every single request.
              </p>
              <div className="rounded-xl border border-white/8 bg-black/30 p-4 space-y-2">
                <p className="text-xs font-semibold text-white/40 uppercase tracking-wider">RAM freed per request — qwen3:8b</p>
                <Compare label="Short message" before="576 MB (4k default)" after="197 MB → 379 MB freed" />
                <Compare label="Medium message" before="576 MB (4k default)" after="288 MB → 288 MB freed" />
                <Compare label="Long document" before="576 MB (4k default)" after="576 MB → 0 MB freed" />
              </div>
              <p className="text-xs text-white/40">
                As conversations grow longer, the needed context grows too — the math always reflects
                the actual history. No tokens are ever dropped. The full context window expands
                organically as you chat.
              </p>
            </OptimizationCard>

            <OptimizationCard
              number="02"
              title="Live memory pressure response — proactive RAM tier system"
              tag="Highest impact · Every request"
              tagColor="orange"
              icon="📊"
            >
              <p>
                Live pressure is autotune&apos;s proactive, heuristic RAM management system. Before every
                request, it reads the OS&apos;s RAM utilization percentage and adjusts context window
                size and KV precision according to four fixed tiers — firing well before any swap risk
                exists. At 80% utilization there is still 20% RAM free and no danger of swap, but
                autotune starts backing off preemptively to stay that way as conditions change.
              </p>
              <div className="rounded-xl border border-white/8 bg-black/30 p-4">
                <p className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-3">
                  Automatic adjustments by RAM tier — checked live, every request
                </p>
                <div className="space-y-2 text-xs">
                  <div className="grid grid-cols-3 gap-2">
                    <div className="font-semibold text-white/40">RAM usage</div>
                    <div className="font-semibold text-white/40">Context window</div>
                    <div className="font-semibold text-white/40">KV precision</div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 font-mono border-t border-white/5 pt-2">
                    <div className="text-green-300">under 80%</div>
                    <div className="text-white/60">full size</div>
                    <div className="text-white/60">profile default</div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 font-mono">
                    <div className="text-yellow-300">80–88%</div>
                    <div className="text-yellow-300">−10%</div>
                    <div className="text-white/60">unchanged</div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 font-mono">
                    <div className="text-orange-300">88–93%</div>
                    <div className="text-orange-300">−25%</div>
                    <div className="text-orange-300">F16 → Q8 (−50% KV RAM)</div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 font-mono">
                    <div className="text-red-400">over 93%</div>
                    <div className="text-red-400">halved</div>
                    <div className="text-red-400">forced Q8</div>
                  </div>
                </div>
              </div>
              <p>
                KV precision switching (F16 → Q8) cuts the KV cache&apos;s RAM footprint in half at
                the cost of negligible quality degradation — each attention value goes from 2 bytes to
                1 byte. The difference in model output is undetectable in practice. You get a notice in
                the chat interface when an adjustment fires: &ldquo;RAM 88% — context 8,192→6,144 tokens, KV F16→Q8&rdquo;.
              </p>
              <p className="text-xs text-white/40 border-t border-white/8 pt-3 mt-1">
                <strong className="text-white/55">How this differs from NoSwapGuard (#08):</strong>{" "}
                Live pressure fires on RAM percentage alone — no model architecture math, no precise byte
                calculation. It is a proactive heuristic. NoSwapGuard is the separate exact-math check
                that only activates when the precise KV byte calculation says swap will actually occur.
                If Live pressure has already trimmed enough headroom, NoSwapGuard may not need to fire at all.
              </p>
            </OptimizationCard>

            <OptimizationCard
              number="06"
              title="KV cache precision control"
              tag="High impact · Profile-aware"
              tagColor="blue"
              icon="🎚️"
            >
              <p>
                Beyond the live pressure response above, each profile has a deliberate default KV
                precision setting. F16 (16-bit float) uses 2 bytes per KV element. Q8 (8-bit quantized)
                uses 1 byte — <strong className="text-white/85">half the KV memory at the same context size</strong>,
                with negligible quality impact.
              </p>
              <p>
                This is separate from model quantization (Q4_K_M, Q5_K_M, etc.), which applies
                to the model&apos;s weights. KV precision only affects the temporary computation
                cache, not the model itself.
              </p>
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-xl border border-white/8 bg-white/3 p-3 text-xs">
                  <div className="font-semibold text-violet-300 mb-1">fast profile</div>
                  <div className="text-white/50">Always Q8</div>
                  <div className="text-white/35 mt-1">Priority: lowest latency</div>
                </div>
                <div className="rounded-xl border border-white/8 bg-white/3 p-3 text-xs">
                  <div className="font-semibold text-blue-300 mb-1">balanced profile</div>
                  <div className="text-white/50">F16 → Q8 under pressure</div>
                  <div className="text-white/35 mt-1">Priority: quality + stability</div>
                </div>
                <div className="rounded-xl border border-white/8 bg-white/3 p-3 text-xs">
                  <div className="font-semibold text-green-300 mb-1">quality profile</div>
                  <div className="text-white/50">F16 → Q8 under pressure</div>
                  <div className="text-white/35 mt-1">Priority: best output</div>
                </div>
              </div>
            </OptimizationCard>

            <OptimizationCard
              number="08"
              title="NoSwapGuard — exact-math pre-flight guarantee"
              tag="Safety · Every request"
              tagColor="orange"
              icon="🛡️"
            >
              <p>
                NoSwapGuard is autotune&apos;s exact-math hard guarantee — completely different in
                approach from the percentage-based Live pressure system (#02). Where Live pressure fires
                at 80% RAM as a proactive heuristic, NoSwapGuard only activates when the math says
                swap is about to happen. It queries Ollama for the actual model architecture — layers,
                KV heads, head dimension — and computes the precise bytes the KV allocation will need:
              </p>
              <Formula>{"kv_bytes = 2 × n_layers × kv_heads × head_dim × num_ctx × precision_bytes"}</Formula>
              <p>
                If <code className="text-white/70">kv_bytes + 1.5 GB safety margin &gt; available RAM</code>,
                it fires. Not because RAM reads as &ldquo;80% full&rdquo; — because the exact number
                of bytes won&apos;t fit.
              </p>
              <p>
                On Apple Silicon, when RAM fills up macOS starts compressing memory pages, then pages
                them to your NVMe drive. Either path is catastrophic for inference — generation speed
                drops from 30+ tok/s to under 5 tok/s, and the whole machine becomes sluggish.
                Ollama doesn&apos;t prevent this — it allocates what it&apos;s told and lets the OS
                handle the consequences. autotune runs in front and checks first.
              </p>
              <div className="rounded-xl border border-white/8 bg-black/30 p-4">
                <p className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-3">
                  Reduction levels — applied in order until it fits
                </p>
                <div className="space-y-1.5 text-xs font-mono">
                  <div className="flex items-center gap-3">
                    <span className="text-green-300 w-20 shrink-0">Level 0</span>
                    <span className="text-white/50">Fits comfortably — no change</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-yellow-300 w-20 shrink-0">Level 1</span>
                    <span className="text-white/50">Trim context 25%</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-orange-300 w-20 shrink-0">Level 2</span>
                    <span className="text-white/50">Halve context</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-orange-400 w-20 shrink-0">Level 3</span>
                    <span className="text-white/50">Halve context + switch to Q8 KV (saves ~50% more)</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-red-400 w-20 shrink-0">Level 4</span>
                    <span className="text-white/50">Quarter context + Q8</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-red-500 w-20 shrink-0">Level 5</span>
                    <span className="text-white/50">Minimum (512 tokens) + Q8 — emergency floor</span>
                  </div>
                </div>
              </div>
              <p>
                The model&apos;s architecture is queried once from Ollama&apos;s{" "}
                <code className="text-white/70">/api/show</code> and cached — every calculation is
                exact, not estimated. This is the fundamental distinction from Live pressure&apos;s
                percentage tiers: NoSwapGuard knows precisely how many bytes are needed, not
                approximately how stressed the system looks.
              </p>
            </OptimizationCard>

            <OptimizationCard
              number="10"
              title="Pre-flight model fit analysis"
              tag="Safety · Before loading"
              tagColor="blue"
              icon="🔍"
            >
              <p>
                Before a model is loaded into memory, autotune runs a complete RAM analysis: will this
                model fit without causing swap?
              </p>
              <p>
                The analysis calculates the total memory requirement:
              </p>
              <Formula>total = model_weights + kv_cache(context, precision) + runtime_overhead (400 MB)</Formula>
              <p>
                It classifies the result as one of four states: <strong className="text-green-300">SAFE</strong> (under 85% RAM),{" "}
                <strong className="text-yellow-300">MARGINAL</strong> (85–92%),{" "}
                <strong className="text-orange-300">SWAP RISK</strong> (92–100%), or{" "}
                <strong className="text-red-400">OOM</strong> (over 100%).
              </p>
              <div className="space-y-2">
                <Callout color="blue">
                  If the model is tight but workable, autotune automatically caps the context window
                  to a safe maximum and recommends Q8 KV precision. You get the best performance
                  the hardware can deliver without ever touching swap.
                </Callout>
                <Callout color="orange">
                  If the model is too heavy, autotune suggests a lighter quantization:
                  &ldquo;Model requires ~14 GB but only 11 GB available. Pull Q4_K_M instead (~9 GB).&rdquo;
                  No guessing — the recommendation is calculated from the exact model architecture.
                </Callout>
              </div>
            </OptimizationCard>

          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/* ── SPEED OPTIMIZATIONS ── */}
      {/* ══════════════════════════════════════════════════════════════════════ */}

      <section id="speed" className="px-6 py-20 bg-white/[0.02] border-y border-white/5">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Speed — high impact</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-3">
            Five ways autotune reduces latency.
          </h2>
          <p className="text-white/50 mb-12 max-w-2xl text-sm">
            TTFT — time to first token — is what you feel as the &ldquo;thinking pause&rdquo; before
            the model starts responding. autotune reduces it through five distinct techniques, all
            working simultaneously, ordered here from most to least impactful.
          </p>

          <div className="space-y-6">

            <OptimizationCard
              number="03"
              title="System prompt prefix caching"
              tag="High impact · Multi-turn"
              tagColor="blue"
              icon="📌"
            >
              <p>
                In any multi-turn conversation, the system prompt — &ldquo;You are a helpful assistant.
                You prefer concise answers&rdquo; — is identical on every single turn. By default,
                Ollama re-processes (re-evaluates through every layer of the model) this entire system
                prompt from scratch on every message.
              </p>
              <p>
                autotune counts the system prompt&apos;s tokens and tells Ollama:{" "}
                <em>keep these first N tokens in the KV cache permanently</em>. The Ollama parameter
                for this is <code className="text-white/70">num_keep</code>. Once set, those tokens
                are evaluated exactly once — at the start of the conversation — and never again.
              </p>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-red-500/15 bg-red-500/5 p-4 text-xs">
                  <div className="font-semibold text-red-300 mb-2">Without prefix caching</div>
                  <div className="text-white/50 space-y-1">
                    <div>Turn 1: process system prompt (100 tokens) + message</div>
                    <div>Turn 2: process system prompt (100 tokens) + turn 1 + message</div>
                    <div>Turn 3: process system prompt (100 tokens) + turn 1-2 + message</div>
                    <div className="text-red-300 mt-2">System prompt re-processed every turn</div>
                  </div>
                </div>
                <div className="rounded-xl border border-green-500/15 bg-green-500/5 p-4 text-xs">
                  <div className="font-semibold text-green-300 mb-2">With prefix caching</div>
                  <div className="text-white/50 space-y-1">
                    <div>Turn 1: process system prompt (100 tokens) + message</div>
                    <div>Turn 2: skip system prompt ← new tokens only</div>
                    <div>Turn 3: skip system prompt ← new tokens only</div>
                    <div className="text-green-300 mt-2">Savings compound with every turn</div>
                  </div>
                </div>
              </div>
              <p>
                In agentic workloads where a session has 10+ turns, this compounding effect means
                TTFT actually <em>decreases</em> as the session grows — the opposite of what raw
                Ollama shows.
              </p>
            </OptimizationCard>

            <OptimizationCard
              number="04"
              title="Model keep-alive"
              tag="High impact · Session start"
              tagColor="green"
              icon="♾️"
            >
              <p>
                By default, Ollama unloads a model from RAM after 5 minutes of idle. The next time
                you send a message — even seconds later — it reads the entire model file from disk,
                loads it into GPU/Metal memory, and warms up the runtime. On a 5 GB model, this
                costs 1–4 seconds before your first token appears.
              </p>
              <p>
                autotune sets <code className="text-white/70">keep_alive=&quot;-1&quot;</code> (keep
                forever) on every request. The model stays in RAM between conversations.
              </p>
              <Callout color="green">
                <strong className="text-green-300">On the RAM question:</strong> The model&apos;s
                weights were already taking up RAM from the moment it was loaded. Setting keep-alive
                to forever means Ollama doesn&apos;t release and re-acquire that same RAM between
                sessions. It doesn&apos;t cost more memory — it just keeps the memory committed,
                which eliminates the reload time. You can disable this via{" "}
                <code className="text-white/70">autotune config set keep_alive_enabled false</code>.
              </Callout>
            </OptimizationCard>

            <OptimizationCard
              number="05"
              title="Context bucket snapping"
              tag="High impact · Every request"
              tagColor="violet"
              icon="🎯"
            >
              <p>
                After computing the minimum context size needed, autotune rounds it up to the nearest
                &ldquo;bucket&rdquo; from a fixed list:
              </p>
              <Formula>Buckets: 512 · 768 · 1024 · 1536 · 2048 · 3072 · 4096 · 6144 · 8192 · 12288 · 16384 · 32768</Formula>
              <p>
                Here&apos;s why this matters enormously: Ollama caches the KV buffer for the most
                recently used context length. If <code className="text-white/70">num_ctx</code> changes
                between requests — say 1,286 then 1,157 then 1,308 — Ollama must{" "}
                <strong className="text-white/85">reallocate the Metal buffer on every single call</strong>,
                even if the model is already loaded. This &ldquo;KV thrashing&rdquo; adds 100–300 ms
                of overhead per request and completely negates the benefit of smaller context windows.
              </p>
              <p>
                By snapping to buckets, prompts of 50–200 tokens all map to bucket 1,536. Ollama
                allocates it once and reuses the buffer on every subsequent request — zero reallocation
                cost. All bucket sizes are multiples of 256, which aligns with Metal&apos;s memory
                alignment boundaries for F16 tensors.
              </p>
            </OptimizationCard>

            <OptimizationCard
              number="07"
              title="Flash attention"
              tag="Medium impact · Every request"
              tagColor="violet"
              icon="⚡"
            >
              <p>
                Standard attention computes the full attention matrix in memory. For a context window
                of N tokens, this requires O(N²) memory — it grows fast and causes large memory spikes
                during the initial prompt processing phase.
              </p>
              <p>
                Flash attention is a mathematically identical algorithm that computes attention in
                tiles (blocks) rather than materializing the full matrix at once. It needs only O(N)
                memory for the same computation — the peak activation memory spike during prefill
                (the initial prompt processing) is dramatically smaller.
              </p>
              <p>
                autotune passes <code className="text-white/70">flash_attn: true</code> on every
                request. Models and Ollama builds that support it use it; those that don&apos;t
                silently ignore the flag. <strong className="text-white/85">Zero quality impact</strong> —
                it&apos;s purely an implementation optimization, not an approximation.
              </p>
            </OptimizationCard>

            <OptimizationCard
              number="09"
              title="Larger prefill batch size"
              tag="Medium impact · Long prompts"
              tagColor="blue"
              icon="🚀"
            >
              <p>
                During &ldquo;prefill&rdquo; — when the model processes your entire prompt before
                generating anything — tokens are fed through the model in chunks called batches.
                Ollama&apos;s default is 512 tokens per chunk.
              </p>
              <p>
                autotune sets <code className="text-white/70">num_batch=1024</code>. For a 700-token
                prompt: the default takes 2 GPU passes (0→512, 512→700). With 1024, it takes 1 pass.
                Fewer passes means fewer Metal kernel dispatches, which directly cuts prefill time
                for any prompt longer than 512 tokens.
              </p>
              <p>
                For short prompts (under 512 tokens), llama.cpp automatically caps the actual batch
                at the prompt length — so there&apos;s no extra memory allocation for short messages.
                At critical RAM pressure, autotune drops this back to 256 to reduce the peak
                activation tensor footprint.
              </p>
            </OptimizationCard>

          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/* ── ADAPTIVE INTELLIGENCE ── */}
      {/* ══════════════════════════════════════════════════════════════════════ */}

      <section id="intelligence" className="px-6 py-20">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Adaptive intelligence</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-3">
            Systems that watch and respond.
          </h2>
          <p className="text-white/50 mb-12 max-w-2xl text-sm">
            Static settings only get you so far. These two systems watch what&apos;s actually
            happening on your machine and respond in real time.
          </p>

          <div className="space-y-6">

            <OptimizationCard
              number="11"
              title="Hardware tuner — OS-level scheduling"
              tag="Moderate impact · During inference"
              tagColor="orange"
              icon="🔧"
            >
              <p>
                Before each inference call, autotune makes real changes to how your operating system
                schedules the inference process. After the call completes, everything is restored to normal.
              </p>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-3">
                  <div className="rounded-xl border border-white/8 bg-white/3 p-4 text-xs">
                    <div className="font-semibold text-orange-300 mb-2">macOS QOS class</div>
                    <div className="text-white/55">
                      Sets the thread to <code className="text-white/70">USER_INTERACTIVE</code> — the
                      highest scheduling priority macOS offers (the same class used for scrolling
                      animations and direct UI responses). The inference process literally gets more
                      CPU time than background tasks during generation.
                    </div>
                  </div>
                  <div className="rounded-xl border border-white/8 bg-white/3 p-4 text-xs">
                    <div className="font-semibold text-orange-300 mb-2">Python GC disabled</div>
                    <div className="text-white/55">
                      Python&apos;s garbage collector runs &ldquo;stop the world&rdquo; pauses where
                      all Python code stops — potentially for tens of milliseconds. During streaming
                      generation, these create visible hitches in output. autotune disables GC during
                      inference (collecting first to clean up) and re-enables it after.
                    </div>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="rounded-xl border border-white/8 bg-white/3 p-4 text-xs">
                    <div className="font-semibold text-orange-300 mb-2">Process priority (nice)</div>
                    <div className="text-white/55">
                      Raises both the autotune process priority and — where permitted — the Ollama
                      server process priority on macOS and Linux. The OS scheduler gives higher-priority
                      processes more CPU time slices, directly improving inference throughput when
                      the system is under load from other apps.
                    </div>
                  </div>
                  <div className="rounded-xl border border-white/8 bg-white/3 p-4 text-xs">
                    <div className="font-semibold text-orange-300 mb-2">Linux CPU governor</div>
                    <div className="text-white/55">
                      On Linux, attempts to set the CPU frequency governor to{" "}
                      <code className="text-white/70">performance</code> mode, disabling frequency
                      scaling so the CPU runs at full clock speed during inference (requires root;
                      silently skipped otherwise).
                    </div>
                  </div>
                </div>
              </div>
            </OptimizationCard>

            <OptimizationCard
              number="12"
              title="Adaptive session advisor"
              tag="Moderate impact · Live monitoring"
              tagColor="red"
              icon="🧠"
            >
              <p>
                During a session, the adaptive advisor continuously watches RAM usage, swap activity,
                tokens per second, and time to first token. It compares live metrics to a baseline
                it builds from your first few requests, and acts if things degrade.
              </p>
              <div className="rounded-xl border border-white/8 bg-black/30 p-4 mb-4">
                <p className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-3">
                  Health score (0–100) — updated every 30 seconds
                </p>
                <div className="space-y-1.5 text-xs font-mono">
                  <div className="flex gap-3"><span className="text-green-300 w-16">90–100</span><span className="text-white/50">Running smoothly</span></div>
                  <div className="flex gap-3"><span className="text-yellow-300 w-16">75–89</span><span className="text-white/50">Moderate load — watching closely</span></div>
                  <div className="flex gap-3"><span className="text-orange-300 w-16">55–74</span><span className="text-white/50">Memory pressure building</span></div>
                  <div className="flex gap-3"><span className="text-red-400 w-16">35–54</span><span className="text-white/50">Stressed — action recommended</span></div>
                  <div className="flex gap-3"><span className="text-red-600 w-16">0–34</span><span className="text-white/50">Critical — immediate action needed</span></div>
                </div>
              </div>
              <p>
                When the score drops below a threshold, the advisor takes actions in order from{" "}
                <em>least to most disruptive</em>:
              </p>
              <div className="grid gap-1.5 text-xs">
                {[
                  ["1. Reduce concurrency", "Fewer parallel requests → less simultaneous KV pressure"],
                  ["2. Reduce context window", "Smaller window → smaller KV cache → RAM freed immediately"],
                  ["3. Lower KV precision", "F16 → Q8 → frees ~50% of KV memory at once"],
                  ["4. Enable prompt caching", "Forces prefix caching if not already active"],
                  ["5. Disable speculative decoding", "Frees draft-model memory if applicable"],
                  ["6. Lower quantization", "Suggests pulling a lighter model variant"],
                  ["7. Switch to smaller model", "Last resort — model is simply too large for current RAM"],
                ].map(([action, desc]) => (
                  <div key={action} className="flex gap-3 rounded-lg border border-white/5 bg-white/2 px-3 py-2">
                    <span className="text-violet-300 w-44 shrink-0">{action}</span>
                    <span className="text-white/45">{desc}</span>
                  </div>
                ))}
              </div>
              <p>
                There&apos;s a 20-second cooldown between actions to avoid thrashing, and the advisor
                waits 90 seconds of sustained stability before considering a scale-up. It also
                attributes performance changes — it knows whether a RAM spike was caused by loading
                a new model, KV growth, or a background application.
              </p>
            </OptimizationCard>

          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/* ── CONTEXT & CONVERSATION ── */}
      {/* ══════════════════════════════════════════════════════════════════════ */}

      <section id="context" className="px-6 py-20 bg-white/[0.02] border-y border-white/5">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Context &amp; conversation</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-3">
            Two systems for managing long conversations.
          </h2>
          <p className="text-white/50 mb-12 max-w-2xl text-sm">
            Every conversation turn adds tokens to the history. Without management, long sessions
            hit the context ceiling and either lose old messages or require a larger (more expensive)
            context window. autotune handles both automatically.
          </p>

          <div className="space-y-6">

            <OptimizationCard
              number="13"
              title="Context compressor"
              tag="Moderate impact · Long sessions"
              tagColor="violet"
              icon="🗜️"
            >
              <p>
                As conversation history grows and approaches the context limit, autotune selectively
                compresses older messages to make room — without deleting them entirely or losing their meaning.
              </p>
              <div className="rounded-xl border border-white/8 bg-black/30 p-4 mb-4">
                <p className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-3">
                  Context budget tiers
                </p>
                <div className="space-y-1.5 text-xs font-mono">
                  <div className="flex gap-3"><span className="text-green-300 w-20">&lt; 55%</span><span className="text-white/50">FULL — all turns verbatim, no compression</span></div>
                  <div className="flex gap-3"><span className="text-yellow-300 w-20">55–75%</span><span className="text-white/50">RECENT+FACTS — last 8 turns + fact summary for older</span></div>
                  <div className="flex gap-3"><span className="text-orange-300 w-20">75–90%</span><span className="text-white/50">COMPRESSED — last 6 turns (light compression) + compact summary</span></div>
                  <div className="flex gap-3"><span className="text-red-400 w-20">&gt; 90%</span><span className="text-white/50">EMERGENCY — last 4 turns (compressed) + one-line summary</span></div>
                </div>
              </div>
              <p>
                Compression is applied in order from lightest to most aggressive:
              </p>
              <div className="space-y-1.5 text-xs">
                {[
                  ["Strip noise", "Remove extra blank lines, trailing whitespace — lossless"],
                  ["Compress JSON blobs", "{\"key1\": ..., \"key2\": ...} → {/* 12 keys: key1, key2… */}"],
                  ["Shorten tool output", "Keep first 12 lines + last 6 lines, mark middle as omitted"],
                  ["Trim assistant messages", "Keep: first paragraph + up to 2 code blocks + last paragraph"],
                  ["Trim user messages", "Preserve first ~600 characters (the intent), trim repetition"],
                ].map(([step, desc]) => (
                  <div key={step} className="flex gap-3 rounded-lg border border-white/5 bg-white/2 px-3 py-2">
                    <span className="text-blue-300 w-36 shrink-0">{step}</span>
                    <span className="text-white/45">{desc}</span>
                  </div>
                ))}
              </div>
              <p>
                Code blocks are always preserved first — they carry the most information per token
                and losing them would make the context misleading. All truncation happens at sentence
                or paragraph boundaries, never mid-sentence.
              </p>
            </OptimizationCard>

            <OptimizationCard
              number="14"
              title="Conversation memory &amp; recall"
              tag="Selective impact · Across sessions"
              tagColor="blue"
              icon="💾"
            >
              <p>
                Every conversation you have is automatically saved to a local SQLite database on your
                machine — not sent anywhere. At the start of each new conversation, autotune searches
                your history for context that&apos;s relevant to what you&apos;re asking about now,
                and quietly injects it as a note in the system prompt.
              </p>
              <p>
                If you asked about FastAPI authentication three sessions ago, and now you&apos;re
                asking a related question, the model will have that prior context available without
                you having to re-explain it.
              </p>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-white/8 bg-white/3 p-4 text-xs">
                  <div className="font-semibold text-blue-300 mb-2">How search works</div>
                  <div className="text-white/55 space-y-1.5">
                    <div><strong className="text-white/70">Vector search (primary):</strong> Uses a local embedding model (nomic-embed-text, ~274 MB) to find semantically similar past exchanges — even if they use different words.</div>
                    <div><strong className="text-white/70">FTS5 keyword search (fallback):</strong> If the embedding model isn&apos;t available, falls back to full-text search across all stored conversations.</div>
                  </div>
                </div>
                <div className="rounded-xl border border-white/8 bg-white/3 p-4 text-xs">
                  <div className="font-semibold text-blue-300 mb-2">Injection threshold</div>
                  <div className="text-white/55">
                    Only injects if the best match has a cosine similarity above 0.38 — a
                    deliberately conservative threshold. The rule is: <em>it&apos;s better to
                    show no context than irrelevant noise</em>. Up to 3 relevant memories are
                    injected, capped at 1,200 characters total to avoid bloating the system prompt.
                  </div>
                </div>
              </div>
              <Callout color="green">
                <strong className="text-green-300">Privacy:</strong> All data stays local. The SQLite
                database lives at <code className="text-white/70">~/.autotune/recall.db</code> on your
                machine. Nothing is sent to any server. The embedding model runs entirely in Ollama on
                your hardware.
              </Callout>
            </OptimizationCard>

          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/* ── HONEST SECTION ── */}
      {/* ══════════════════════════════════════════════════════════════════════ */}

      <section id="honest" className="px-6 py-20">
        <div className="mx-auto max-w-5xl">
          <SectionLabel>Honesty</SectionLabel>
          <h2 className="text-3xl font-bold text-white sm:text-4xl mb-3">
            What autotune doesn&apos;t change.
          </h2>
          <p className="text-white/50 mb-10 max-w-2xl text-sm">
            We&apos;d rather be transparent about limitations than have you discover them yourself.
          </p>

          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {[
              {
                title: "Generation speed",
                detail: "Token throughput (tokens per second) is Metal GPU-bound on Apple Silicon and CUDA-bound on NVIDIA. autotune doesn't touch the generation loop. Benchmarks show ±2% variance — that's measurement noise, not a real difference.",
                verdict: "Unchanged",
                color: "white",
              },
              {
                title: "Model weights",
                detail: "autotune changes context window size, KV cache precision, and scheduling. It never modifies the model's actual weights. Output quality is identical — autotune changes how memory is managed, not what the model knows.",
                verdict: "Unchanged",
                color: "white",
              },
              {
                title: "First turn in agentic sessions",
                detail: "autotune pre-allocates a larger KV window for a full agentic session upfront. Turn 1 is ~80% slower as a result. From turn 2 onward, prefix-cache savings compound and total wall time comes out ~46% lower. Worth it for sessions with 3+ turns.",
                verdict: "Turn 1 is slower",
                color: "orange",
              },
              {
                title: "Swap on severely low RAM",
                detail: "NoSwapGuard prevents swap when RAM is adequate. If your machine is running critically low (e.g. multiple large models loaded simultaneously), the guard may not be able to reduce context enough to fit — it will tell you explicitly.",
                verdict: "Can't prevent everything",
                color: "orange",
              },
              {
                title: "No cloud or external dependency",
                detail: "autotune runs entirely locally. There's no API key, no account, no cloud service required. Anonymous telemetry is opt-in and off by default. Everything — inference, memory recall, embedding — runs on your hardware.",
                verdict: "Fully local",
                color: "green",
              },
              {
                title: "Output is identical",
                detail: "The same prompt, the same model, the same temperature — you will get equivalent responses with or without autotune. prompt_eval_count (how many tokens Ollama actually processed) is identical in both conditions. We just do it with less RAM.",
                verdict: "Zero quality tradeoff",
                color: "green",
              },
            ].map((item) => {
              const verdictColors: Record<string, string> = {
                white:  "text-white/60",
                orange: "text-orange-300",
                green:  "text-green-300",
              };
              return (
                <div key={item.title} className="rounded-2xl border border-white/8 bg-white/3 p-5">
                  <div className="text-sm font-semibold text-white mb-2">{item.title}</div>
                  <div className="text-xs text-white/50 leading-relaxed mb-3">{item.detail}</div>
                  <div className={`text-xs font-semibold ${verdictColors[item.color]}`}>
                    → {item.verdict}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="px-6 py-20 bg-white/[0.02] border-t border-white/5">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-2xl font-bold text-white mb-4">Run the proof on your machine.</h2>
          <p className="text-white/50 mb-8 max-w-xl mx-auto text-sm">
            Every number in our benchmarks is reproducible. One command runs a 30-second
            head-to-head using Ollama&apos;s own internal timers on your hardware.
          </p>
          <div className="mb-6 max-w-sm mx-auto rounded-xl border border-white/10 bg-black/40 px-5 py-3 font-mono text-sm text-green-300">
            autotune proof -m qwen3:8b
          </div>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/#quickstart"
              className="rounded-xl bg-violet-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-900/40 transition hover:bg-violet-500"
            >
              Get started →
            </Link>
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
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-white/5 px-6 py-10">
        <div className="mx-auto max-w-5xl flex flex-col items-center gap-4 sm:flex-row sm:justify-between text-xs text-white/30">
          <div className="flex flex-col items-center sm:items-start gap-1">
            <span>autotune v1.6.0 — MIT License</span>
            <a href="mailto:autotunellm@gmail.com" className="hover:text-white/60 transition-colors">autotunellm@gmail.com</a>
          </div>
          <div className="flex flex-wrap justify-center gap-6">
            <Link href="/" className="hover:text-white/60 transition-colors">Home</Link>
            <Link href="/install" className="hover:text-white/60 transition-colors">Install</Link>
            <Link href="/commands" className="hover:text-white/60 transition-colors">Commands</Link>
            <a href="https://github.com/tanavc1/local-llm-autotune" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">GitHub</a>
            <a href="https://pypi.org/project/llm-autotune/" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">PyPI</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
