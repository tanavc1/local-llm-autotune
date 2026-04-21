import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "autotune commands — full reference",
  description:
    "Complete reference for all autotune CLI commands. chat, serve, ls, hardware, pull, and more.",
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

function Badge({ children, color = "violet" }: { children: string; color?: "violet" | "blue" | "green" | "yellow" }) {
  const colors = {
    violet: "bg-violet-500/15 text-violet-300 border-violet-500/25",
    blue:   "bg-blue-500/15 text-blue-300 border-blue-500/25",
    green:  "bg-green-500/15 text-green-300 border-green-500/25",
    yellow: "bg-yellow-500/15 text-yellow-300 border-yellow-500/25",
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
  badgeColor?: "violet" | "blue" | "green" | "yellow";
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
  const sections = [
    {
      label: "Get started",
      commands: ["chat", "run", "hardware"],
    },
    {
      label: "Manage models",
      commands: ["ls", "ps", "pull", "models", "unload"],
    },
    {
      label: "Deploy & integrate",
      commands: ["serve", "recommend"],
    },
    {
      label: "Diagnose",
      commands: ["doctor"],
    },
  ];

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
            Every command listed with examples and flags. Start with{" "}
            <Code>autotune hardware</Code> to see what your machine can do.
          </p>
        </div>

        {/* Quick nav */}
        <div className="mb-12 rounded-2xl border border-white/8 bg-white/2 p-5">
          <p className="text-xs font-semibold uppercase tracking-wider text-white/30 mb-4">Jump to</p>
          <div className="flex flex-wrap gap-2">
            {[
              "chat", "run", "hardware", "ls", "ps",
              "pull", "models", "unload", "serve", "recommend", "doctor",
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

        {/* Section: Get started */}
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
                  label: "Fastest mode (best for quick Q&A)",
                  code: "autotune chat --model qwen3:8b --profile fast",
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

        {/* Section: Manage models */}
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
                { label: "Browse popular models", code: "autotune pull" },
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

        {/* Section: Deploy & integrate */}
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
              tagline="Get a configuration recommendation"
              description="Profiles your hardware and recommends the best model and settings for your machine. Shows alternatives for fastest, balanced, and best-quality modes. Run this if you're not sure which model to use."
              usage={`autotune recommend`}
              examples={[
                { code: "autotune recommend" },
                { label: "Show only balanced recommendations", code: "autotune recommend --mode balanced" },
                { label: "Check a specific model", code: "autotune recommend --model qwen3:8b" },
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

        {/* Section: Diagnose */}
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
          </div>
        </div>

        {/* Ollama quick reference */}
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

        {/* CTA */}
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
    </div>
  );
}
