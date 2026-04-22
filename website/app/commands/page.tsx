import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "autotune commands — full reference",
  description:
    "Complete reference for all autotune CLI commands. chat, proof, serve, memory, mlx, and more.",
};

// ─── helpers ─────────────────────────────────────────────────────────────────

function Code({ children, block = false }: { children: string; block?: boolean }) {
  if (block) {
    return (
      <div className="rounded-xl border border-white/10 bg-black/50 overflow-hidden mt-3">
        <div className="flex items-center gap-1.5 border-b border-white/8 px-4 py-2">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500/60" />
          <span className="h-2.5 w-2.5 rounded-full bg-yellow-500/60" />
          <span className="h-2.5 w-2.5 rounded-full bg-green-500/60" />
          <span className="ml-2 text-xs text-white/30">Terminal</span>
        </div>
        <pre className="overflow-x-auto p-4 text-sm font-mono text-green-300 leading-relaxed">
          <code>{children}</code>
        </pre>
      </div>
    );
  }
  return (
    <code className="rounded bg-white/8 px-1.5 py-0.5 text-sm font-mono text-green-300">
      {children}
    </code>
  );
}

function Badge({ children, color = "violet" }: { children: string; color?: "violet" | "blue" | "green" | "yellow" | "orange" }) {
  const colors = {
    violet: "bg-violet-500/15 text-violet-300 border-violet-500/25",
    blue:   "bg-blue-500/15 text-blue-300 border-blue-500/25",
    green:  "bg-green-500/15 text-green-300 border-green-500/25",
    yellow: "bg-yellow-500/15 text-yellow-300 border-yellow-500/25",
    orange: "bg-orange-500/15 text-orange-300 border-orange-500/25",
  };
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${colors[color]}`}>
      {children}
    </span>
  );
}

interface CommandCardProps {
  name: string;
  tagline: string;
  badge?: string;
  badgeColor?: "violet" | "blue" | "green" | "yellow" | "orange";
  description: string;
  usage: string;
  examples?: { label?: string; code: string }[];
  flags?: { flag: string; desc: string }[];
}

function CommandCard({
  name,
  tagline,
  badge,
  badgeColor = "violet",
  description,
  usage,
  examples,
  flags,
}: CommandCardProps) {
  return (
    <div id={name} className="rounded-2xl border border-white/8 bg-white/2 p-6 scroll-mt-24">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-3 flex-wrap">
          <code className="text-lg font-bold text-white font-mono">autotune {name}</code>
          {badge && <Badge color={badgeColor}>{badge}</Badge>}
        </div>
        <span className="text-sm text-white/40">{tagline}</span>
      </div>
      <p className="text-sm text-white/60 mb-4 leading-relaxed">{description}</p>
      <div className="mb-1">
        <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Usage</span>
        <Code block>{usage}</Code>
      </div>
      {examples && examples.length > 0 && (
        <div className="mt-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Examples</span>
          {examples.map((ex, i) => (
            <div key={i}>
              {ex.label && <div className="text-xs text-white/40 mt-3 mb-1">{ex.label}</div>}
              <Code block>{ex.code}</Code>
            </div>
          ))}
        </div>
      )}
      {flags && flags.length > 0 && (
        <div className="mt-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Flags</span>
          <div className="mt-2 flex flex-col gap-1.5">
            {flags.map((f) => (
              <div key={f.flag} className="flex items-start gap-3 text-sm">
                <code className="shrink-0 rounded bg-white/6 px-2 py-0.5 text-xs font-mono text-green-300/80">
                  {f.flag}
                </code>
                <span className="text-white/50">{f.desc}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── page ─────────────────────────────────────────────────────────────────────

export default function CommandsPage() {
  return (
    <div className="min-h-screen bg-[#09090f] text-[#e8e8f0]">
      {/* Nav */}
      <nav className="border-b border-white/5 bg-[#09090f]/90 backdrop-blur sticky top-0 z-50">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <a href="/" className="text-lg font-bold text-white">autotune</a>
          <div className="flex items-center gap-4 text-sm text-white/60">
            <a href="/install" className="hover:text-white transition-colors">Install</a>
            <a href="/commands" className="text-violet-300 font-medium">Commands</a>
            <a
              href="https://github.com/tanavc1/local-llm-autotune"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-5xl px-6 py-16">
        {/* Header */}
        <div className="mb-12">
          <p className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-3">
            Command reference
          </p>
          <h1 className="text-4xl font-bold text-white mb-4">All autotune commands</h1>
          <p className="text-white/55 text-lg max-w-2xl">
            Every command listed with examples and flags. New? Start with{" "}
            <Code>autotune recommend</Code> to get a hardware-matched model,
            then <Code>autotune proof</Code> to verify the improvement is real.
          </p>
        </div>

        {/* Quick nav */}
        <div className="mb-12 rounded-2xl border border-white/8 bg-white/2 p-5">
          <p className="text-xs font-semibold uppercase tracking-wider text-white/30 mb-4">Jump to</p>
          <div className="flex flex-wrap gap-2">
            {[
              "chat", "run", "hardware",
              "ls", "ps", "pull", "models", "unload",
              "serve", "recommend",
              "proof", "proof-suite", "bench", "user-bench", "agent-bench",
              "memory-search", "memory-list", "memory-stats", "memory-forget", "memory-setup",
              "mlx-list", "mlx-pull", "mlx-resolve",
              "telemetry", "storage",
              "doctor",
            ].map((cmd) => (
              <a
                key={cmd}
                href={`#${cmd}`}
                className="rounded-lg border border-white/10 bg-white/4 px-3 py-1.5 text-xs font-mono text-green-300 hover:border-white/20 hover:bg-white/8 transition-colors"
              >
                {cmd}
              </a>
            ))}
          </div>
        </div>

        {/* ── Section: Get started ─────────────────────────────────────────── */}
        <div className="mb-4">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-6">
            Get started
          </h2>
          <div className="flex flex-col gap-6">

            <CommandCard
              name="chat"
              tagline="Start an optimized conversation"
              badge="Start here"
              badgeColor="green"
              description="Open an interactive chat with a local model. autotune automatically sizes the context window, enables prefix caching, and monitors RAM in real-time — your computer stays fast even while the model is running."
              usage={`autotune chat --model qwen3:8b`}
              examples={[
                { label: "Basic chat", code: "autotune chat --model qwen3:8b" },
                {
                  label: "With a system prompt",
                  code: `autotune chat --model qwen3:8b --system "You are a concise coding assistant"`,
                },
                {
                  label: "Quality mode (slower but smarter responses)",
                  code: "autotune chat --model qwen3:8b --profile quality",
                },
                {
                  label: "Resume a previous conversation",
                  code: "autotune chat --model qwen3:8b --conv-id abc123",
                },
              ]}
              flags={[
                { flag: "--model, -m", desc: "Model to use (required). Run `autotune ls` for available models." },
                { flag: "--profile, -p", desc: "fast / balanced / quality. Default: balanced." },
                { flag: "--system, -s", desc: "Custom system prompt." },
                { flag: "--conv-id", desc: "Resume a saved conversation by ID." },
              ]}
            />

            <CommandCard
              name="run"
              tagline="Pre-flight check + chat"
              badge="Recommended"
              badgeColor="violet"
              description="Like `chat` but runs a memory analysis before loading the model. autotune warns you if the model might cause swap (which makes your computer feel slow), automatically picks the safest context window size, and chooses the right profile. Use this the first time you try a model."
              usage={`autotune run qwen3:8b`}
              examples={[
                { code: "autotune run qwen3:8b" },
                { code: "autotune run qwen2.5-coder:14b --profile balanced" },
                {
                  label: "With a custom system prompt",
                  code: `autotune run llama3.2 --system "You are a helpful assistant"`,
                },
                {
                  label: "Override swap warning and start anyway",
                  code: "autotune run qwen3:8b --force",
                },
              ]}
              flags={[
                { flag: "MODEL", desc: "Model name (positional argument, required)." },
                { flag: "--profile, -p", desc: "fast / balanced / quality / auto. Default: auto (autotune picks for you)." },
                { flag: "--system, -s", desc: "Custom system prompt." },
                { flag: "--force", desc: "Start even if swap risk is detected (not recommended)." },
                { flag: "--recall", desc: "Inject relevant context from past conversations." },
              ]}
            />

            <CommandCard
              name="hardware"
              tagline="See what your machine can do"
              description="Scans your CPU, RAM, and GPU. Shows which AI models fit in your available memory and which apps are consuming the most RAM right now. If closing one app would let you run a larger model, autotune will tell you."
              usage={`autotune hardware`}
              examples={[
                { label: "Full report with RAM tips", code: "autotune hardware" },
                { label: "Hardware only (skip RAM tips)", code: "autotune hardware --no-ram-tips" },
              ]}
              flags={[
                { flag: "--ram-tips / --no-ram-tips", desc: "Show top RAM consumers and model-unlock suggestions. Default: on." },
              ]}
            />
          </div>
        </div>

        {/* ── Section: Manage models ───────────────────────────────────────── */}
        <div className="mb-4 mt-14">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-6">
            Manage models
          </h2>
          <div className="flex flex-col gap-6">

            <CommandCard
              name="ls"
              tagline="List downloaded models with fit scores"
              description="Shows every model you've downloaded and whether it fits in your available RAM. Includes the safe context window size, recommended profile, and a warning if quantization is too aggressive for your hardware."
              usage={`autotune ls`}
              examples={[
                { code: "autotune ls" },
                { label: "Machine-readable JSON output", code: "autotune ls --json" },
              ]}
              flags={[
                { flag: "--json", desc: "Output as JSON instead of a table." },
              ]}
            />

            <CommandCard
              name="ps"
              tagline="See which models are loaded in memory right now"
              description="Shows every model currently loaded in RAM across Ollama, MLX, and LM Studio. Use this to check if a model is still occupying memory after you stopped chatting with it."
              usage={`autotune ps`}
            />

            <CommandCard
              name="pull"
              tagline="Download a model"
              description="Download any Ollama model directly from within autotune. Run without a model name to browse popular recommendations for your hardware."
              usage={`autotune pull qwen3:8b`}
              examples={[
                { label: "Download a specific model", code: "autotune pull qwen3:8b" },
                { label: "Browse popular models for your hardware", code: "autotune pull" },
                { label: "Download and immediately chat", code: "autotune pull qwen3:8b\nautotune chat --model qwen3:8b" },
              ]}
            />

            <CommandCard
              name="models"
              tagline="List all available models"
              description="Shows all models installed on this machine across Ollama, MLX, and LM Studio — with size on disk, architecture, quantization level, and quality tier based on public benchmarks (MMLU, HumanEval)."
              usage={`autotune models`}
              examples={[
                { label: "Local models", code: "autotune models" },
                { label: "View autotune's pre-configured model registry", code: "autotune models --registry" },
              ]}
              flags={[
                { flag: "--registry", desc: "Show autotune's internal model list instead of locally downloaded models." },
              ]}
            />

            <CommandCard
              name="unload"
              tagline="Free RAM immediately"
              description="Releases a model from memory without opening a chat session. Useful after a heavy session — frees RAM for other apps right away. If you omit the model name, you'll see an interactive picker."
              usage={`autotune unload qwen3:8b`}
              examples={[
                { label: "Unload a specific model", code: "autotune unload qwen3:8b" },
                { label: "Interactive picker (shows all loaded models)", code: "autotune unload" },
              ]}
            />
          </div>
        </div>

        {/* ── Section: Deploy & integrate ──────────────────────────────────── */}
        <div className="mb-4 mt-14">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-6">
            Deploy &amp; integrate
          </h2>
          <div className="flex flex-col gap-6">

            <CommandCard
              name="serve"
              tagline="Start an OpenAI-compatible API server"
              description="Starts a local API server that any OpenAI-compatible tool can connect to — Continue.dev, Open WebUI, LangChain, a Python script, anything. All autotune optimizations apply to every request automatically."
              usage={`autotune serve`}
              examples={[
                { label: "Default (localhost:8765)", code: "autotune serve" },
                { label: "Connect from Python", code: `from openai import OpenAI\nclient = OpenAI(base_url="http://localhost:8765/v1", api_key="autotune")` },
                { label: "Connect via curl", code: `curl http://localhost:8765/v1/models` },
                { label: "Apple Silicon — also enable MLX backend", code: "autotune serve --mlx" },
              ]}
              flags={[
                { flag: "--host", desc: "Bind address. Default: 127.0.0.1 (local only)." },
                { flag: "--port", desc: "Port. Default: 8765." },
                { flag: "--mlx", desc: "Enable MLX backend on Apple Silicon (~10–40% higher throughput, disables tool calling)." },
                { flag: "--reload", desc: "Auto-reload on code changes (dev mode)." },
              ]}
            />

            <CommandCard
              name="recommend"
              tagline="Get the best model for your hardware"
              description="Profiles your hardware and recommends the best model and settings for your machine. Shows alternatives for fastest, balanced, and best-quality modes — including exact `ollama pull` commands to get started instantly."
              usage={`autotune recommend`}
              examples={[
                { code: "autotune recommend" },
                { label: "Show only balanced recommendations", code: "autotune recommend --mode balanced" },
                { label: "Check a specific model", code: "autotune recommend --model qwen3:8b" },
                { label: "Show top 5 alternatives per mode", code: "autotune recommend --top 5" },
              ]}
              flags={[
                { flag: "--mode, -m", desc: "fastest / balanced / best_quality / all. Default: all." },
                { flag: "--model", desc: "Restrict recommendations to a single model." },
                { flag: "--top N", desc: "Number of alternatives to show per mode. Default: 3." },
                { flag: "--no-show-hardware", desc: "Skip the hardware profile section." },
              ]}
            />
          </div>
        </div>

        {/* ── Section: Benchmarking & Proof ────────────────────────────────── */}
        <div className="mb-4 mt-14">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-6">
            Benchmarking &amp; proof
          </h2>
          <p className="text-sm text-white/40 mb-6 -mt-2">
            Verify that autotune is actually helping on your specific machine. All timings come from Ollama&apos;s own Go nanosecond timers — nothing estimated.
          </p>
          <div className="flex flex-col gap-6">

            <CommandCard
              name="proof"
              tagline="Quick head-to-head benchmark (~30 seconds)"
              badge="Run this first"
              badgeColor="green"
              description="Runs a fast, honest benchmark comparing raw Ollama defaults against autotune. Measures TTFT (time to first token), KV cache RAM usage, RAM headroom, swap events, and generation speed. Includes a long-context test that shows TTFT improvements by comparing KV buffer allocation at 4096 tokens vs autotune's dynamically-sized buffer. Results saved to JSON."
              usage={`autotune proof --model qwen3:8b`}
              examples={[
                { label: "Auto-select the best installed model", code: "autotune proof" },
                { label: "Benchmark a specific model", code: "autotune proof --model qwen3:8b" },
                { label: "More runs for stabler numbers", code: "autotune proof --model qwen3:8b --runs 3" },
                { label: "Save results to a file", code: "autotune proof --model qwen3:8b --output results.json" },
                { label: "See which models are installed", code: "autotune proof --list-models" },
              ]}
              flags={[
                { flag: "--model, -m", desc: "Ollama model to benchmark. Auto-selects if omitted." },
                { flag: "--runs, -r", desc: "Runs per condition. 2 is fast (~30s); 3+ gives more stable numbers. Default: 2." },
                { flag: "--profile, -p", desc: "fast / balanced / quality. autotune profile to test against raw. Default: balanced." },
                { flag: "--output, -o", desc: "Save JSON results to this path. Defaults to proof_<model>.json." },
                { flag: "--list-models", desc: "List locally installed Ollama models and exit." },
              ]}
            />

            <CommandCard
              name="proof-suite"
              tagline="Multi-model statistical benchmark"
              badge="Deep analysis"
              badgeColor="blue"
              description="Runs a curated 5-prompt suite (factual, code, analysis, conversation, long output) through both raw Ollama and autotune across multiple models. Reports Ollama-internal timing, process-isolated RAM, and full statistical significance: Wilcoxon signed-rank test, Cohen's d effect size, and 95% confidence intervals."
              usage={`autotune proof-suite`}
              examples={[
                { label: "Default 3-model run", code: "autotune proof-suite" },
                { label: "Specific models", code: "autotune proof-suite -m llama3.2:3b -m qwen3:8b" },
                { label: "More runs per prompt for tighter stats", code: "autotune proof-suite -m qwen3:8b --runs 5" },
                { label: "Save full results", code: "autotune proof-suite --output results.json" },
              ]}
              flags={[
                { flag: "--models, -m", desc: "Ollama model IDs to benchmark. Repeat for multiple. Default: llama3.2:3b, qwen3:8b." },
                { flag: "--runs, -n", desc: "Inference runs per condition per prompt. Minimum 3 for statistics. Default: 3." },
                { flag: "--profile, -p", desc: "autotune profile to compare against raw Ollama. Default: balanced." },
                { flag: "--output, -o", desc: "Save full results to a JSON file." },
                { flag: "--list-models", desc: "List locally installed Ollama models and exit." },
              ]}
            />

            <CommandCard
              name="bench"
              tagline="Intensive multi-prompt benchmark"
              badge="Advanced"
              badgeColor="orange"
              description="Runs a full benchmark suite with multiple prompts across different task types (short, code, reasoning, analysis). Use this when you want detailed per-prompt breakdowns or want to compare two different autotune profiles against each other."
              usage={`autotune bench --model qwen3:8b`}
              examples={[
                { label: "Standard benchmark", code: "autotune bench --model qwen3:8b" },
                { label: "Duel mode: compare two profiles head-to-head", code: "autotune bench --model qwen3:8b --duel" },
                { label: "Raw Ollama only (no autotune)", code: "autotune bench --model qwen3:8b --raw" },
                { label: "Compare autotune vs raw", code: "autotune bench --model qwen3:8b --compare" },
                { label: "More runs for stable results", code: "autotune bench --model qwen3:8b --runs 5" },
              ]}
              flags={[
                { flag: "--model, -m", desc: "Ollama model to benchmark (required)." },
                { flag: "--runs, -r", desc: "Runs per prompt per mode. Default: 3." },
                { flag: "--profile, -p", desc: "autotune profile to use. Default: balanced." },
                { flag: "--duel", desc: "Compare two profiles against each other." },
                { flag: "--raw", desc: "Run raw Ollama only (no autotune)." },
                { flag: "--compare", desc: "Run both raw and autotune and show a side-by-side diff." },
                { flag: "--output, -o", desc: "Save results JSON to this path." },
              ]}
            />

            <CommandCard
              name="user-bench"
              tagline="Real-world user experience benchmark"
              description="Measures what users actually feel — not raw throughput. Runs autotune head-to-head against raw Ollama across realistic laptop workflows: background queries, sustained chat, agent loops, and code debugging. Reports in user-friendly language: swap events, RAM headroom, TTFT consistency, CPU spikes, and a 0–100 background impact score. Can run in the background (survives terminal close) with a desktop notification when done."
              usage={`autotune user-bench --model qwen3:8b`}
              examples={[
                { label: "Standard benchmark (~30 min)", code: "autotune user-bench --model qwen3:8b" },
                { label: "Quick mode: 2 scenarios (~10-15 min)", code: "autotune user-bench --model qwen3:8b --quick" },
                { label: "Run in background (keeps running after terminal close)", code: "autotune user-bench --model qwen3:8b --background" },
                { label: "Run on every locally installed model", code: "autotune user-bench --all-models --runs 2" },
              ]}
              flags={[
                { flag: "--model, -m", desc: "Ollama model to benchmark. Auto-selects first installed model if omitted." },
                { flag: "--profile, -p", desc: "autotune profile to use. Default: balanced." },
                { flag: "--runs, -r", desc: "Runs per scenario per condition. Default: 3." },
                { flag: "--quick, -q", desc: "Quick mode: 2 scenarios instead of 4 (~10-15 min)." },
                { flag: "--all-models", desc: "Run on every locally installed Ollama model." },
                { flag: "--background", desc: "Fork to background — survives terminal close, sends a desktop notification when done." },
                { flag: "--output-dir", desc: "Directory for result JSON files. Default: current directory." },
              ]}
            />

            <CommandCard
              name="agent-bench"
              tagline="Agentic multi-turn benchmark"
              description="Tests autotune on 5 realistic agentic tasks: code debugging, research synthesis, step planning, adversarial context, and extended sessions. The key story is TTFT growth curves — in raw Ollama, TTFT grows linearly with each conversation turn as the full 4096-token KV buffer fills. autotune's dynamic context sizing keeps TTFT flat by sizing the window to actual usage."
              usage={`autotune agent-bench`}
              examples={[
                { label: "Default run (all 5 tasks, 5 trials each)", code: "autotune agent-bench" },
                { label: "Specific models", code: "autotune agent-bench -m llama3.2:3b -m qwen3:8b" },
                { label: "Quick mode: 3 tasks, 2 trials (~20-30 min)", code: "autotune agent-bench --quick" },
                { label: "Specific tasks only", code: "autotune agent-bench --tasks code_debugger,extended_session" },
                { label: "Save results", code: "autotune agent-bench -m qwen3:8b --output agent_results.json" },
              ]}
              flags={[
                { flag: "--models, -m", desc: "Ollama model IDs to benchmark. Repeat for multiple. Default: llama3.2:3b, qwen3:8b." },
                { flag: "--trials, -n", desc: "Trials per condition per task. Min 3 recommended. Default: 5." },
                { flag: "--tasks, -t", desc: "Comma-separated task IDs. Options: code_debugger, research_synth, step_planner, adversarial_context, extended_session." },
                { flag: "--profile, -p", desc: "autotune profile to test. Default: balanced." },
                { flag: "--quick, -q", desc: "Quick mode: 3 tasks, 2 trials (~20-30 min)." },
                { flag: "--output, -o", desc: "Save full results JSON to this path." },
              ]}
            />
          </div>
        </div>

        {/* ── Section: Conversation memory ─────────────────────────────────── */}
        <div className="mb-4 mt-14">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-6">
            Conversation memory
          </h2>
          <p className="text-sm text-white/40 mb-6 -mt-2">
            autotune stores past conversations locally (SQLite + optional vector embeddings) so future chat sessions can surface relevant context automatically.
          </p>
          <div className="flex flex-col gap-6">

            <div id="memory-search" className="rounded-2xl border border-white/8 bg-white/2 p-6 scroll-mt-24">
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <code className="text-lg font-bold text-white font-mono">autotune memory search</code>
                <span className="text-sm text-white/40">Search past conversations</span>
              </div>
              <p className="text-sm text-white/60 mb-4 leading-relaxed">
                Search your conversation history by semantic meaning or keywords. Uses vector search (cosine similarity) when an embedding model is available, otherwise falls back to FTS5 full-text keyword search.
              </p>
              <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Usage</span>
              <Code block>{`autotune memory search "your query here"`}</Code>
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Examples</span>
                <Code block>{`autotune memory search "postgres migration"\nautotune memory search "FastAPI authentication" --top 10\nautotune memory search "React hooks" --min-score 0.4`}</Code>
              </div>
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Flags</span>
                <div className="mt-2 flex flex-col gap-1.5">
                  {[
                    { flag: "QUERY", desc: "Search query (required)." },
                    { flag: "--top, -n", desc: "Number of results to return. Default: 5." },
                    { flag: "--min-score", desc: "Minimum similarity score (0–1). Ignored for FTS5 fallback. Default: 0.20." },
                  ].map((f) => (
                    <div key={f.flag} className="flex items-start gap-3 text-sm">
                      <code className="shrink-0 rounded bg-white/6 px-2 py-0.5 text-xs font-mono text-green-300/80">{f.flag}</code>
                      <span className="text-white/50">{f.desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div id="memory-list" className="rounded-2xl border border-white/8 bg-white/2 p-6 scroll-mt-24">
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <code className="text-lg font-bold text-white font-mono">autotune memory list</code>
                <span className="text-sm text-white/40">Browse stored memories</span>
              </div>
              <p className="text-sm text-white/60 mb-4 leading-relaxed">
                List recently stored conversation memories with timestamps, model names, and a preview of each chunk. Use this to find a memory ID before deleting it.
              </p>
              <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Usage</span>
              <Code block>{`autotune memory list`}</Code>
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Examples</span>
                <Code block>{`autotune memory list\nautotune memory list --days 7\nautotune memory list --model qwen3:8b --limit 50`}</Code>
              </div>
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Flags</span>
                <div className="mt-2 flex flex-col gap-1.5">
                  {[
                    { flag: "--limit, -n", desc: "Number of memories to show. Default: 20." },
                    { flag: "--days", desc: "Only show memories from the last N days." },
                    { flag: "--model", desc: "Filter by model (e.g. qwen3:8b)." },
                  ].map((f) => (
                    <div key={f.flag} className="flex items-start gap-3 text-sm">
                      <code className="shrink-0 rounded bg-white/6 px-2 py-0.5 text-xs font-mono text-green-300/80">{f.flag}</code>
                      <span className="text-white/50">{f.desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div id="memory-stats" className="rounded-2xl border border-white/8 bg-white/2 p-6 scroll-mt-24">
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <code className="text-lg font-bold text-white font-mono">autotune memory stats</code>
                <span className="text-sm text-white/40">Memory store statistics</span>
              </div>
              <p className="text-sm text-white/60 mb-4 leading-relaxed">
                Show statistics about the local memory store: total chunks, how many have vector embeddings, database size, date range, breakdown by model, and whether semantic search is active.
              </p>
              <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Usage</span>
              <Code block>{`autotune memory stats`}</Code>
            </div>

            <div id="memory-forget" className="rounded-2xl border border-white/8 bg-white/2 p-6 scroll-mt-24">
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <code className="text-lg font-bold text-white font-mono">autotune memory forget</code>
                <span className="text-sm text-white/40">Delete memories</span>
              </div>
              <p className="text-sm text-white/60 mb-4 leading-relaxed">
                Delete one memory chunk, all memories for a specific conversation, or wipe the entire store. Use <Code>autotune memory list</Code> first to find the memory ID.
              </p>
              <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Usage</span>
              <Code block>{`autotune memory forget <memory-id>`}</Code>
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Examples</span>
                <Code block>{`autotune memory forget 42\nautotune memory forget --conv-id abc123\nautotune memory forget --all\nautotune memory forget --all --yes   # skip confirmation`}</Code>
              </div>
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Flags</span>
                <div className="mt-2 flex flex-col gap-1.5">
                  {[
                    { flag: "MEMORY_ID", desc: "ID of the specific memory chunk to delete (see autotune memory list)." },
                    { flag: "--all", desc: "Delete ALL memories (asks for confirmation unless --yes is passed)." },
                    { flag: "--conv-id", desc: "Delete all memories for a specific conversation ID." },
                    { flag: "--yes, -y", desc: "Skip confirmation prompt." },
                  ].map((f) => (
                    <div key={f.flag} className="flex items-start gap-3 text-sm">
                      <code className="shrink-0 rounded bg-white/6 px-2 py-0.5 text-xs font-mono text-green-300/80">{f.flag}</code>
                      <span className="text-white/50">{f.desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div id="memory-setup" className="rounded-2xl border border-white/8 bg-white/2 p-6 scroll-mt-24">
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <code className="text-lg font-bold text-white font-mono">autotune memory setup</code>
                <span className="text-sm text-white/40">Enable semantic search</span>
              </div>
              <p className="text-sm text-white/60 mb-4 leading-relaxed">
                Pull <code className="text-green-300/80 text-xs bg-white/6 px-1.5 py-0.5 rounded">nomic-embed-text</code> from Ollama (~274 MB) to enable semantic vector search across your conversation history. Without this, autotune uses FTS5 keyword search instead.
              </p>
              <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Usage</span>
              <Code block>{`autotune memory setup`}</Code>
            </div>
          </div>
        </div>

        {/* ── Section: Apple Silicon / MLX ─────────────────────────────────── */}
        <div className="mb-4 mt-14">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-6">
            Apple Silicon (MLX)
          </h2>
          <p className="text-sm text-white/40 mb-6 -mt-2">
            MLX runs LLMs entirely on-chip using Apple&apos;s unified memory and Metal GPU kernels — typically 10–40% faster than Ollama on the same model. Requires an Apple Silicon Mac (M1/M2/M3/M4).
          </p>
          <div className="flex flex-col gap-6">

            <div id="mlx-list" className="rounded-2xl border border-white/8 bg-white/2 p-6 scroll-mt-24">
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <code className="text-lg font-bold text-white font-mono">autotune mlx list</code>
                <span className="text-sm text-white/40">Show cached MLX models</span>
              </div>
              <p className="text-sm text-white/60 mb-4 leading-relaxed">
                List all MLX-format models already downloaded locally, with size on disk.
              </p>
              <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Usage</span>
              <Code block>{`autotune mlx list`}</Code>
            </div>

            <div id="mlx-pull" className="rounded-2xl border border-white/8 bg-white/2 p-6 scroll-mt-24">
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <code className="text-lg font-bold text-white font-mono">autotune mlx pull</code>
                <span className="text-sm text-white/40">Download an MLX model</span>
              </div>
              <p className="text-sm text-white/60 mb-4 leading-relaxed">
                Download an MLX-quantized model from the mlx-community on HuggingFace. You can use an Ollama model name (e.g. <Code>qwen3:8b</Code>) and autotune will resolve the correct MLX variant automatically.
              </p>
              <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Usage</span>
              <Code block>{`autotune mlx pull <model>`}</Code>
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Examples</span>
                <Code block>{`autotune mlx pull qwen3:8b\nautotune mlx pull llama3.2:3b\nautotune mlx pull qwen2.5-coder:14b --quant 8bit`}</Code>
              </div>
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Flags</span>
                <div className="mt-2 flex flex-col gap-1.5">
                  {[
                    { flag: "MODEL", desc: "Model name (required). Ollama name (e.g. qwen3:8b) or full HuggingFace ID." },
                    { flag: "--quant, -q", desc: "Quantization level: 4bit / 8bit / bf16. Default: 4bit." },
                  ].map((f) => (
                    <div key={f.flag} className="flex items-start gap-3 text-sm">
                      <code className="shrink-0 rounded bg-white/6 px-2 py-0.5 text-xs font-mono text-green-300/80">{f.flag}</code>
                      <span className="text-white/50">{f.desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div id="mlx-resolve" className="rounded-2xl border border-white/8 bg-white/2 p-6 scroll-mt-24">
              <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                <code className="text-lg font-bold text-white font-mono">autotune mlx resolve</code>
                <span className="text-sm text-white/40">Look up the MLX model ID</span>
              </div>
              <p className="text-sm text-white/60 mb-4 leading-relaxed">
                Show which MLX HuggingFace model ID would be used for a given Ollama model name. Useful to check before pulling.
              </p>
              <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Usage</span>
              <Code block>{`autotune mlx resolve <model>`}</Code>
              <div className="mt-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-white/30">Examples</span>
                <Code block>{`autotune mlx resolve qwen3:8b\nautotune mlx resolve llama3.2`}</Code>
              </div>
            </div>
          </div>
        </div>

        {/* ── Section: Settings ────────────────────────────────────────────── */}
        <div className="mb-4 mt-14">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-6">
            Settings
          </h2>
          <div className="flex flex-col gap-6">

            <CommandCard
              name="telemetry"
              tagline="View performance history and manage data collection"
              description="Shows a table of all recent inference runs with TTFT, throughput, RAM/swap pressure, CPU load, and completion status. Also manages opt-in/out for anonymous telemetry collection (hardware fingerprint + performance data sent to the autotune team to improve defaults)."
              usage={`autotune telemetry`}
              examples={[
                { label: "View recent runs", code: "autotune telemetry" },
                { label: "Filter by model", code: "autotune telemetry --model qwen3:8b" },
                { label: "View individual telemetry events", code: "autotune telemetry --events --model qwen3:8b" },
                { label: "Check consent status", code: "autotune telemetry --status" },
                { label: "Opt in to anonymous telemetry", code: "autotune telemetry --enable" },
                { label: "Opt out", code: "autotune telemetry --disable" },
              ]}
              flags={[
                { flag: "--model", desc: "Filter to a specific model ID." },
                { flag: "--limit", desc: "Number of recent runs to show. Default: 20." },
                { flag: "--events", desc: "Show individual telemetry events (RAM spikes, slow tokens, errors) instead of run history." },
                { flag: "--status", desc: "Show current telemetry consent status." },
                { flag: "--enable", desc: "Opt in to anonymous telemetry collection." },
                { flag: "--disable", desc: "Opt out — no further data will be sent." },
              ]}
            />

            <CommandCard
              name="storage"
              tagline="Manage local SQLite performance data"
              description="Enable or disable local SQLite storage of performance observations, telemetry events, and agent benchmark results. Model metadata is always stored regardless of this setting. Run without an argument to see the current status."
              usage={`autotune storage [on|off|status]`}
              examples={[
                { label: "Check current setting", code: "autotune storage status" },
                { label: "Enable local storage (default)", code: "autotune storage on" },
                { label: "Disable storage (e.g. shared / ephemeral machines)", code: "autotune storage off" },
              ]}
            />
          </div>
        </div>

        {/* ── Section: Diagnose ────────────────────────────────────────────── */}
        <div className="mb-4 mt-14">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-6">
            Diagnose
          </h2>
          <div className="flex flex-col gap-6">

            <CommandCard
              name="doctor"
              tagline="Check your installation"
              description="Runs a full health check: Python version, required packages, whether Ollama and other backends are reachable, RAM and swap headroom, and database health. Every check shows a clear pass/fail so you know exactly what needs fixing."
              usage={`autotune doctor`}
            />

            <CommandCard
              name="upgrade"
              tagline="Update to the latest version"
              description="Checks PyPI for a newer version of autotune, shows what changed, and upgrades with one keypress. Because autotune is updated frequently, run this any time something seems off or you want the latest improvements."
              usage={`autotune upgrade\nautotune upgrade --yes   # skip confirmation`}
            />
          </div>
        </div>

        {/* ── Ollama quick reference ────────────────────────────────────────── */}
        <div className="mt-16 rounded-2xl border border-white/8 bg-white/2 p-6">
          <h2 className="text-lg font-bold text-white mb-2">Ollama commands you&apos;ll use</h2>
          <p className="text-sm text-white/50 mb-5">
            autotune runs on top of Ollama. Here are the Ollama commands that pair with autotune.
          </p>
          <div className="flex flex-col gap-3 text-sm">
            {[
              { cmd: "ollama serve",           desc: "Start the Ollama background service (must be running before autotune chat)" },
              { cmd: "ollama pull qwen3:8b",   desc: "Download a model (use autotune pull instead to get hardware-aware recommendations)" },
              { cmd: "ollama list",             desc: "List downloaded models (autotune ls shows more detail)" },
              { cmd: "ollama ps",               desc: "See models in memory (autotune ps shows more detail)" },
              { cmd: "ollama rm qwen3:8b",      desc: "Delete a model from disk" },
              { cmd: "ollama --version",        desc: "Confirm Ollama is installed" },
            ].map((row) => (
              <div key={row.cmd} className="flex items-start gap-4">
                <code className="shrink-0 rounded bg-white/6 px-2 py-1 text-xs font-mono text-green-300/80">
                  {row.cmd}
                </code>
                <span className="text-white/50 text-xs pt-1">{row.desc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── CTA ──────────────────────────────────────────────────────────── */}
        <div className="mt-12 rounded-2xl border border-violet-500/20 bg-violet-500/5 p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-white mb-1">New to autotune?</h2>
            <p className="text-sm text-white/55">Follow the install guide to get running in 5 minutes.</p>
          </div>
          <a
            href="/install"
            className="shrink-0 rounded-xl bg-violet-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-violet-500 transition-colors"
          >
            Install guide →
          </a>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-white/5 px-6 py-8 mt-4">
        <div className="mx-auto max-w-5xl flex flex-col items-center gap-2 sm:flex-row sm:justify-between text-xs text-white/30">
          <div className="flex flex-col items-center sm:items-start gap-1">
            <span>autotune v1.0.0 — MIT License</span>
            <a href="mailto:autotunellm@gmail.com" className="hover:text-white/60 transition-colors">autotunellm@gmail.com</a>
          </div>
          <div className="flex gap-5">
            <a href="/" className="hover:text-white/60 transition-colors">Home</a>
            <a href="/install" className="hover:text-white/60 transition-colors">Install</a>
            <a href="/what-we-do" className="hover:text-white/60 transition-colors">All we do</a>
            <a href="https://github.com/tanavc1/local-llm-autotune" target="_blank" rel="noopener noreferrer" className="hover:text-white/60 transition-colors">GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
