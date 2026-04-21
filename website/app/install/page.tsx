import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Install autotune — Step-by-step guide",
  description:
    "Complete install guide for autotune. Works on Mac, Windows, and Linux. No experience needed.",
};

// ─── helpers ─────────────────────────────────────────────────────────────────

function Step({
  n,
  title,
  children,
}: {
  n: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex gap-5">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-violet-500/40 bg-violet-500/10 text-sm font-bold text-violet-300 mt-0.5">
        {n}
      </div>
      <div className="flex-1">
        <h3 className="text-base font-semibold text-white mb-2">{title}</h3>
        {children}
      </div>
    </div>
  );
}

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

function Callout({
  type,
  children,
}: {
  type: "tip" | "warning" | "info";
  children: React.ReactNode;
}) {
  const styles = {
    tip:     "border-green-500/25 bg-green-500/8 text-green-300",
    warning: "border-yellow-500/25 bg-yellow-500/8 text-yellow-300",
    info:    "border-blue-500/25 bg-blue-500/8 text-blue-300",
  };
  const icons = { tip: "💡", warning: "⚠️", info: "ℹ️" };
  return (
    <div className={`rounded-xl border p-4 ${styles[type]} mt-4`}>
      <p className="text-sm">
        <span className="mr-2">{icons[type]}</span>
        {children}
      </p>
    </div>
  );
}

// ─── page ─────────────────────────────────────────────────────────────────────

export default function InstallPage() {
  return (
    <div className="min-h-screen bg-[#09090f] text-[#e8e8f0]">
      {/* Nav */}
      <nav className="border-b border-white/5 bg-[#09090f]/90 backdrop-blur sticky top-0 z-50">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
          <a href="/" className="text-lg font-bold text-white">
            autotune
          </a>
          <div className="flex items-center gap-4 text-sm text-white/60">
            <a href="/install" className="text-violet-300 font-medium">Install</a>
            <a href="/commands" className="hover:text-white transition-colors">Commands</a>
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

      <div className="mx-auto max-w-3xl px-6 py-16">
        <div className="mb-12">
          <p className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-3">
            Complete install guide
          </p>
          <h1 className="text-4xl font-bold text-white mb-4">
            Get autotune running in 5 minutes
          </h1>
          <p className="text-white/55 text-lg">
            No experience with AI or command lines needed. Follow each step in order.
          </p>
        </div>

        {/* OS tabs note */}
        <div className="mb-10 rounded-2xl border border-white/8 bg-white/3 p-5">
          <p className="text-sm text-white/60">
            These instructions work on <strong className="text-white">Mac</strong>,{" "}
            <strong className="text-white">Windows</strong>, and{" "}
            <strong className="text-white">Linux</strong>. Windows users: use the
            Terminal app (search for &ldquo;Terminal&rdquo; in Start) or PowerShell.
          </p>
        </div>

        {/* Steps */}
        <div className="flex flex-col gap-10">

          {/* Step 1 */}
          <Step n="1" title="Open a terminal">
            <p className="text-sm text-white/60 mb-3">
              A terminal lets you type commands directly to your computer.
            </p>
            <div className="grid gap-3 sm:grid-cols-3 text-sm">
              {[
                { os: "Mac", how: "Press ⌘ + Space, type Terminal, press Enter" },
                { os: "Windows", how: "Press Win, type Terminal or PowerShell, press Enter" },
                { os: "Linux", how: "Press Ctrl + Alt + T" },
              ].map((o) => (
                <div key={o.os} className="rounded-xl border border-white/8 bg-white/3 p-3">
                  <div className="font-semibold text-white mb-1">{o.os}</div>
                  <div className="text-xs text-white/50">{o.how}</div>
                </div>
              ))}
            </div>
          </Step>

          {/* Step 2 */}
          <Step n="2" title="Install Ollama (the AI engine)">
            <p className="text-sm text-white/60 mb-1">
              Ollama runs AI models on your computer. autotune makes it faster.
            </p>
            <div className="flex flex-col gap-4 mt-4">
              <div>
                <div className="text-xs font-semibold text-white/50 uppercase tracking-wider mb-2">Mac (easiest)</div>
                <Code block>{`# Download the Mac installer from https://ollama.com/download
# Double-click the downloaded file and follow the instructions.
# Then verify it's working:
ollama --version`}</Code>
              </div>
              <div>
                <div className="text-xs font-semibold text-white/50 uppercase tracking-wider mb-2">Linux (one command)</div>
                <Code block>{`curl -fsSL https://ollama.com/install.sh | sh
ollama --version`}</Code>
              </div>
              <div>
                <div className="text-xs font-semibold text-white/50 uppercase tracking-wider mb-2">Windows</div>
                <Code block>{`# Download OllamaSetup.exe from https://ollama.com/download
# Run the installer. Then in PowerShell:
ollama --version`}</Code>
              </div>
            </div>
            <Callout type="tip">
              If <Code>{`ollama --version`}</Code> prints a version number like{" "}
              <Code>{`0.7.2`}</Code>, Ollama is installed correctly.
            </Callout>
          </Step>

          {/* Step 3 */}
          <Step n="3" title="Start Ollama">
            <p className="text-sm text-white/60 mb-1">
              Ollama needs to be running in the background before you can use it.
            </p>
            <div className="grid gap-3 sm:grid-cols-2 mt-4">
              <div>
                <div className="text-xs font-semibold text-white/50 uppercase tracking-wider mb-2">Mac / Linux</div>
                <Code block>{`ollama serve`}</Code>
                <p className="text-xs text-white/40 mt-2">
                  Leave this terminal open. Open a new one for the next steps.
                </p>
              </div>
              <div>
                <div className="text-xs font-semibold text-white/50 uppercase tracking-wider mb-2">Windows</div>
                <p className="text-xs text-white/50">
                  On Windows, Ollama starts automatically as a background service after installation.
                  You can check the system tray icon.
                </p>
              </div>
            </div>
          </Step>

          {/* Step 4 */}
          <Step n="4" title="Download an AI model">
            <p className="text-sm text-white/60 mb-1">
              Pick based on how much RAM your computer has. Not sure? Run{" "}
              <Code>{`autotune hardware`}</Code> after step 6.
            </p>
            <div className="mt-4 overflow-hidden rounded-xl border border-white/8">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/8 bg-white/4 text-left text-xs font-semibold uppercase tracking-wider text-white/50">
                    <th className="px-4 py-3">Your RAM</th>
                    <th className="px-4 py-3">Run this</th>
                    <th className="px-4 py-3 hidden sm:table-cell">Size</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { ram: "8 GB",  cmd: "ollama pull qwen3:4b",  size: "2.6 GB" },
                    { ram: "16 GB", cmd: "ollama pull qwen3:8b",  size: "5.2 GB", recommended: true },
                    { ram: "24 GB", cmd: "ollama pull qwen3:14b", size: "9.0 GB" },
                    { ram: "32 GB+", cmd: "ollama pull qwen3:30b-a3b", size: "17 GB" },
                  ].map((r) => (
                    <tr key={r.ram} className="border-b border-white/5">
                      <td className="px-4 py-3 text-xs font-medium text-white/60">{r.ram}</td>
                      <td className="px-4 py-3 font-mono text-xs text-green-300">
                        {r.cmd}
                        {r.recommended && (
                          <span className="ml-2 rounded-full bg-violet-500/20 px-2 py-0.5 text-xs text-violet-300">
                            recommended
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-xs text-white/40 hidden sm:table-cell">{r.size}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Code block>{`# For most people (16 GB RAM):
ollama pull qwen3:8b

# Watch the download progress in your terminal.
# This takes a few minutes depending on your internet speed.`}</Code>
            <Callout type="info">
              Models are downloaded once and stored on your computer. Nothing is sent to the cloud when you chat.
            </Callout>
          </Step>

          {/* Step 5 */}
          <Step n="5" title="Install Python (if you don't have it)">
            <p className="text-sm text-white/60 mb-1">
              autotune is a Python tool. You need Python 3.10 or newer.
            </p>
            <Code block>{`# Check if Python is already installed:
python3 --version
# Should print: Python 3.10.x or higher

# If not installed, download from https://python.org/downloads
# Mac users: you can also use Homebrew: brew install python@3.13`}</Code>
          </Step>

          {/* Step 6 */}
          <Step n="6" title="Install autotune">
            <Code block>{`pip install llm-autotune

# If you get a "command not found" error, try:
pip3 install llm-autotune

# Apple Silicon Mac (M1/M2/M3/M4)? Get faster inference too:
pip install "llm-autotune[mlx]"`}</Code>
            <Callout type="tip">
              After install, the <Code>{`autotune`}</Code> command will be available in your terminal.
            </Callout>
          </Step>

          {/* Step 7 */}
          <Step n="7" title="Check your hardware">
            <p className="text-sm text-white/60 mb-1">
              This shows you what autotune detected about your machine.
            </p>
            <Code block>{`autotune hardware`}</Code>
            <p className="text-sm text-white/50 mt-3">
              You&apos;ll see your CPU, RAM, and GPU. autotune uses this to pick the right settings automatically.
            </p>
          </Step>

          {/* Step 8 */}
          <Step n="8" title="Start chatting!">
            <p className="text-sm text-white/60 mb-1">
              That&apos;s it. autotune handles all the optimization automatically.
            </p>
            <Code block>{`# Replace qwen3:8b with whatever model you downloaded:
autotune chat --model qwen3:8b

# Type your question and press Enter.
# Type /quit to exit, or press Ctrl+C.`}</Code>
            <Callout type="tip">
              You should see the first word appear about 39% faster than running Ollama alone.
              The second message will be even faster — autotune caches your conversation context.
            </Callout>
          </Step>

        </div>

        {/* Troubleshooting */}
        <div className="mt-16">
          <h2 className="text-2xl font-bold text-white mb-6">Something went wrong?</h2>
          <div className="flex flex-col gap-4">
            {[
              {
                q: `"command not found: autotune"`,
                a: `Run pip install llm-autotune again. If that doesn't work, try pip3 install llm-autotune`,
              },
              {
                q: `"Ollama is not running"`,
                a: `Open a separate terminal and run: ollama serve — leave it open`,
              },
              {
                q: `"No models found"`,
                a: `You need to download a model first. Run: ollama pull qwen3:8b`,
              },
              {
                q: `First message is very slow (5–10 seconds)`,
                a: `Normal on first use. The model is being loaded from disk. Every message after this will be much faster.`,
              },
              {
                q: `"Not enough RAM" or the computer gets slow`,
                a: `Try a smaller model. For 8 GB RAM, use qwen3:4b instead of qwen3:8b`,
              },
            ].map((item) => (
              <div key={item.q} className="rounded-xl border border-white/8 bg-white/3 p-5">
                <div className="text-sm font-semibold text-white mb-2 font-mono">{item.q}</div>
                <div className="text-sm text-white/60">{item.a}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Next steps */}
        <div className="mt-16 rounded-2xl border border-violet-500/20 bg-violet-500/5 p-6">
          <h2 className="text-lg font-bold text-white mb-4">Next steps</h2>
          <div className="flex flex-col gap-3 text-sm text-white/60">
            <div>
              <code className="text-green-300">autotune ls</code>
              <span className="ml-3">See all your models and how well they fit your hardware</span>
            </div>
            <div>
              <code className="text-green-300">autotune ps</code>
              <span className="ml-3">Check which models are currently loaded in memory</span>
            </div>
            <div>
              <code className="text-green-300">autotune serve</code>
              <span className="ml-3">Start an API server (works with any app that uses OpenAI)</span>
            </div>
            <div>
              <a href="/commands" className="text-violet-300 hover:text-violet-200 transition-colors">
                → Full command reference
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
