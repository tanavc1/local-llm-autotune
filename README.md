# autotune — Local LLM Inference Optimizer

[![PyPI](https://img.shields.io/pypi/v/llm-autotune)](https://pypi.org/project/llm-autotune/)
[![Python](https://img.shields.io/pypi/pyversions/llm-autotune)](https://pypi.org/project/llm-autotune/)
[![CI](https://github.com/tanavc1/local-llm-autotune/actions/workflows/test.yml/badge.svg)](https://github.com/tanavc1/local-llm-autotune/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**39% faster time-to-first-word. 3× less KV cache. Drop-in for Ollama, LM Studio, and MLX.**

autotune is a middleware layer that makes your local LLMs noticeably faster and lighter — without changing your code or workflow. It computes the exact KV cache each request needs, pins your system prompt in memory, and manages context windows automatically.

```bash
pip install llm-autotune
autotune chat --model qwen3:8b   # that's it
```

Works with **Ollama**, **LM Studio**, and **MLX** (Apple Silicon native) out of the box.

---

## What autotune actually improves

Benchmarked on Apple M2 16 GB using Ollama's own nanosecond-precision internal timers — not Python wall-clock estimates. Results are means across 3 runs × 5 prompt types, with Wilcoxon signed-rank statistical testing and Cohen's d effect sizes.

| Metric | llama3.2:3b | gemma4:e2b | qwen3:8b | Average |
|--------|:-----------:|:----------:|:--------:|:-------:|
| **Time to first word (TTFT)** | −35% | −29% | **−53%** | **−39%** |
| **KV prefill time** | −66% | −64% | **−72%** | **−67%** |
| **KV cache RAM** | −66% | **−69%** | −66% | **−67%** |
| **Generation speed (tok/s)** | ±2% | ±0.2% | ±2.4% | **unchanged** |

> **Timing source:** `prompt_eval_duration`, `load_duration`, and `total_duration` from Ollama's Go runtime. Token counts (`prompt_eval_count`) are identical in both conditions — autotune right-sizes the buffer, not the content.

### What the numbers mean

**You wait 39% less for the first word.** On qwen3:8b that's 53% faster. On a long-context prompt, up to 89% faster. You feel this on every message.

**KV cache shrinks 3×.** Raw Ollama allocates a fixed 4,096-token KV buffer regardless of prompt length. autotune computes the exact size each request needs — for a typical chat message that frees 300–400 MB before inference even starts.

**Generation speed is unchanged.** Token generation on Apple Silicon is Metal GPU-bound. The ±2% variance in the data is measurement noise. autotune is transparent about this.

**122,778 KV buffer slots freed** across all benchmark runs — slots Ollama would have allocated, zeroed, and initialized for nothing.

### Verify it yourself

```bash
# Quick 45-second check on any model you have:
autotune proof --model qwen3:8b

# Full statistical benchmark with Wilcoxon p-values and Cohen's d:
autotune proof-suite --model qwen3:8b --runs 3
```

`autotune proof` runs two scenarios: a standard multi-turn session and a long-context code-review prompt where TTFT and KV allocation differences are most visible. Results are saved as JSON alongside your terminal output.

---

## Quickstart

### 1. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.com, then pull a model:
ollama pull qwen3:8b           # 5.2 GB — best general model for 16 GB machines
```

Not sure which model to use? Run `autotune recommend` after installing and it will pick the best model for your exact hardware.

### 2. Install autotune

```bash
pip install llm-autotune
```

**Requirements:** Python 3.10+, Ollama running locally.

```bash
# Apple Silicon acceleration (native Metal GPU kernels):
pip install "llm-autotune[mlx]"

# Development install:
git clone https://github.com/tanavc1/local-llm-autotune.git
cd local-llm-autotune && pip install -e ".[dev]"
```

### 3. Get a model recommendation for your hardware

```bash
autotune recommend
```

Profiles your CPU, RAM, and GPU, then scores every model in the registry against your hardware and recommends the best option with an exact `ollama pull` command to run.

### 4. Start chatting

```bash
autotune chat --model qwen3:8b                   # optimized chat, default profile
autotune chat --model qwen3:8b --profile fast    # minimum latency
autotune chat --model qwen3:8b --profile quality # largest context window
autotune chat --model qwen3:8b --no-swap         # guarantee no macOS swap
autotune chat --model qwen3:8b --system "You are a concise coding assistant."
```

### 5. Check what's running

```bash
autotune ps        # all models in memory — RAM, context, quant, age
autotune hardware  # CPU, RAM, GPU backend, and effective memory budget
autotune ls        # every locally installed model scored against your hardware
```

---

## Model recommendations by hardware

| RAM | Recommended model | Pull command | Why |
|-----|------------------|--------------|-----|
| 8 GB | `qwen3:4b` | `ollama pull qwen3:4b` | Best 4B available; hybrid thinking mode |
| 16 GB | `qwen3:8b` | `ollama pull qwen3:8b` | Near-frontier quality; best 8B as of 2026 |
| 16 GB (coding) | `qwen2.5-coder:7b` | `ollama pull qwen2.5-coder:7b` | Near GPT-4o on HumanEval at 7B |
| 24 GB | `qwen3:14b` | `ollama pull qwen3:14b` | Excellent reasoning; comfortable headroom |
| 24 GB (coding) | `qwen2.5-coder:14b` | `ollama pull qwen2.5-coder:14b` | Best open coding model at this size |
| 32 GB | `qwen3:30b-a3b` | `ollama pull qwen3:30b-a3b` | MoE: flagship quality at 7B inference cost |
| 64 GB+ | `qwen3:32b` | `ollama pull qwen3:32b` | Top dense open model |
| Reasoning | `deepseek-r1:14b` | `ollama pull deepseek-r1:14b` | Chain-of-thought; strong math and logic |

Run `autotune recommend` to get a personalised pick with scores for your exact hardware configuration.

---

## Features

| Feature | What happens |
|---------|-------------|
| **Dynamic KV sizing** | Computes the exact `num_ctx` each request needs — typically 4–8× less KV cache than Ollama's fixed 4,096-token default |
| **KV prefix caching** | Pins system-prompt tokens via `num_keep` so they're never re-evaluated each turn |
| **Model keep-alive** | Sets `keep_alive=-1` so the model stays loaded between conversations — eliminates reload latency |
| **Adaptive KV precision** | Automatically downgrades F16 → Q8 under memory pressure before any slowdown occurs |
| **Flash attention** | Enables `flash_attn=true` on every request — reduces peak KV activation memory |
| **Prefill batching** | Sets `num_batch=1024` (2× Ollama default) — fewer Metal kernel dispatches for long prompts |
| **Context management** | Trims conversation history at token budget thresholds, always at sentence/paragraph boundaries |
| **Inference queue** | FIFO queue (1 concurrent, 8 waiting) with HTTP 429 back-pressure — prevents memory thrashing |
| **OpenAI-compatible API** | Drop-in server at `localhost:8765/v1` — works with any OpenAI SDK |
| **MLX backend** | On M-series Macs, routes inference to MLX-LM for native Metal GPU kernels |
| **Persistent memory** | Every conversation saved to SQLite; semantically searches past sessions at startup |
| **No-swap guarantee** | `--no-swap` mode reduces context window to ensure zero macOS swap |

---

## Agentic workloads

Raw Ollama's fixed `num_ctx=4096` hurts most inside agent loops — where tool calls, observations, and reasoning steps accumulate. autotune sizes the session context once before the loop begins, holds it constant across all turns, and uses `num_keep` prefix caching so the system prompt is never re-evaluated after turn 1.

**Measured on `llama3.2:3b`, multi-turn tool-calling agent task:**

| Metric | Raw Ollama | autotune |
|--------|:----------:|:--------:|
| Model reloads per session | 0–1 | ~0 |
| Swap events | 1 of 3 trials | 0 |
| Tool call errors | 1 avg | 0 |
| Context tokens at session end | 3,043 | 1,946 (−36%) |
| TTFT trend per turn | grows | shrinks (prefix cache) |

For sessions with 3+ turns, prefix caching compounds — TTFT per turn falls as the conversation grows. Full methodology and raw data: [AGENT_BENCHMARK.md](AGENT_BENCHMARK.md)

---

## Chat commands

| Command | What it does |
|---------|-------------|
| `/help` | Show available commands |
| `/new` | Start a new conversation |
| `/history` | Show full conversation history |
| `/profile fast\|balanced\|quality` | Switch profile mid-conversation |
| `/model <id>` | Switch to a different model |
| `/system <text>` | Set or replace the system prompt |
| `/export` | Export conversation to Markdown |
| `/metrics` | Session stats: tok/s, TTFT, request count |
| `/recall` | Browse past conversations |
| `/recall search <query>` | Semantic search across all past sessions |
| `/pull <model>` | Pull a model from Ollama without leaving chat |
| `/quit` | Exit (also Ctrl-C) |

---

## Profiles

| Profile | Context | Temperature | KV precision | Best for |
|---------|--------:|:-----------:|:------------:|---------|
| `fast` ⚡ | 2,048 | 0.1 | Q8 | Quick lookups, autocomplete |
| `balanced` ⚖️ | 8,192 | 0.7 | F16 | General chat, coding |
| `quality` ✨ | 32,768 | 0.8 | F16 | Long documents, analysis |

---

## Apple Silicon (MLX)

```bash
pip install "llm-autotune[mlx]"
autotune mlx pull qwen3:8b        # download MLX-quantized model
autotune chat --model qwen3:8b    # automatically routes to MLX
autotune mlx list                 # show locally cached MLX models
```

MLX activates automatically on Apple Silicon — no configuration needed. Use Ollama-backed models when you need structured tool calls in agentic workflows.

---

## API server (OpenAI-compatible)

```bash
autotune serve
# → Listening at http://127.0.0.1:8765/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8765/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Per-request headers

```
X-Autotune-Profile: fast          # override profile for this request
X-Conversation-Id: a3f92c1b       # attach to a persistent conversation
```

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible, streaming or non-streaming |
| `GET /v1/models` | All available models across all backends |
| `GET /health` | Server status, queue depth, memory pressure |
| `GET /api/hardware` | Live hardware snapshot |
| `GET /api/profiles` | Profile definitions |
| `GET /api/running_models` | Models in memory with RAM, context, quant, age |
| `POST/GET/DELETE /api/conversations` | Persistent conversation CRUD |
| `GET /api/conversations/{id}/export` | Export as Markdown |

### Concurrency

```bash
AUTOTUNE_MAX_CONCURRENT=1    # parallel inference slots (default: 1)
AUTOTUNE_MAX_QUEUED=8        # max requests waiting (default: 8)
AUTOTUNE_WAIT_TIMEOUT=120    # seconds before a queued request gets 429 (default: 120)
```

---

## Embedding autotune in your application

```python
import autotune
from openai import OpenAI

autotune.start()                             # spawns server if not running; blocks until ready
client = OpenAI(**autotune.client_kwargs())  # {"base_url": "http://localhost:8765/v1", "api_key": "local"}

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

`start()` checks `/health` first and returns immediately if the server is already running.

### Options

```python
autotune.start(
    host="localhost",
    port=8765,
    timeout=30.0,       # raise TimeoutError if server isn't ready within this many seconds
    profile="balanced", # "fast" | "balanced" | "quality"
    use_mlx=False,      # True = MLX on Apple Silicon (faster, no tool calls)
    log_level="warning",
)
```

### Error handling

```python
try:
    response = client.chat.completions.create(...)
except Exception as e:
    error = e.response.json().get("detail", {})
    match error.get("type"):
        case "model_not_found":
            print(f"Run: ollama pull {error['model']}")
        case "memory_pressure":
            print("Not enough RAM. Try a smaller model or --profile fast.")
        case "backend_error":
            print(f"Backend error: {error['message']}\nSuggestion: {error['suggestion']}")
```

### Server RAM footprint

| Mode | Server RAM | Tool calling | Notes |
|------|-----------|:------------:|-------|
| `autotune.start()` (default) | ~94 MB | ✓ | Ollama-backed |
| `autotune.start(use_mlx=True)` | ~470 MB | ✗ | 10–40% faster on Apple Silicon |

---

## Agentic frameworks

autotune's OpenAI-compatible server is a drop-in local LLM backend for any framework that accepts a custom base URL.

```bash
autotune serve
```

### OpenClaw

```yaml
# openclaw/config.yaml
providers:
  - name: autotune-local
    api: openai-responses
    baseUrl: http://localhost:8765/v1
    apiKey: sk-local
    model: qwen3:8b
    supportsTools: true
```

### Hermes Agent

```yaml
# ~/.hermes/config.yaml
model:
  provider: custom
  base_url: http://localhost:8765/v1
  api_key: sk-local
  name: qwen3:8b
```

**Models confirmed for tool calling via Ollama:** `qwen3:8b`, `qwen3:14b`, `llama3.1:8b`, `qwen2.5-coder:14b`, `hermes3`

---

## How dynamic KV sizing works

Ollama allocates the full KV cache upfront before generating a single token. With `num_ctx=4096`, it zeros and initialises a 4,096-token buffer even if your prompt is 50 tokens. That initialization is what you wait for.

autotune computes the minimum `num_ctx` each request actually needs:

```
num_ctx = clamp(input_tokens + max_new_tokens + 256, 512, profile_max)
```

For a typical conversation message on `balanced` (max 8,192 tokens):
- Input: ~22 tokens → `num_ctx` = 22 + 1,024 + 256 = **1,302**
- Savings on qwen3:8b: 4,096 → 1,302 tokens = **~224 MB never allocated**

Context grows naturally as the conversation grows — the full history is included on every request and no tokens are ever dropped.

---

## Context management

autotune monitors `history_tokens / effective_budget` and selects a strategy automatically:

```
< 55%   FULL              — all turns verbatim
55–75%  RECENT+FACTS      — last 8 turns + structured facts block for older turns
75–90%  COMPRESSED        — last 6 turns (lightly compressed) + compact summary
> 90%   EMERGENCY         — last 4 turns (compressed) + one-line summary
```

Low-value chatter is dropped first. Code blocks, stack traces, and technical content are always preserved. All cutoffs happen at sentence or paragraph boundaries. The facts block for older turns is extracted deterministically — no extra LLM call required.

---

## Conversation memory

Every conversation is saved to a local SQLite database with full-text and vector similarity search. No flags required.

- **Automatic context injection** — at session start, autotune surfaces relevant facts from past conversations as a silent system note.
- **Session resume** — use `--conv-id <id>` to continue an exact past session with full context.
- **In-chat recall** — `/recall` to browse sessions; `/recall search <topic>` for semantic search.

| Path | Contents |
|------|----------|
| `~/.autotune/recall.db` | FTS5 + float32 vectors; turns, extracted facts |
| `~/Library/Application Support/autotune/autotune.db` | Hardware telemetry, run observations (macOS) |
| `~/.local/share/autotune/autotune.db` | Same (Linux) |

---

## Telemetry

```bash
autotune telemetry                    # last 20 inference runs
autotune telemetry --events           # notable events: swap spikes, OOMs
autotune telemetry --model qwen3:8b   # filter by model
```

**Anonymous cloud telemetry is opt-in and off by default:**

```bash
autotune telemetry --status    # check opt-in status
autotune telemetry --enable    # opt in
autotune telemetry --disable   # opt out
```

What is sent when opted in: CPU architecture, RAM size, GPU backend, tokens/sec, TTFT, context size, quantization label, session start/stop events. No hostnames, usernames, IP addresses, or conversation content. Data goes to a private Supabase instance and is never sold or shared.

---

## Troubleshooting

**"Ollama is not running."**
→ Start Ollama: `ollama serve` (in a separate terminal), or open the Ollama app.

**"No models found."**
→ Pull a model: `ollama pull qwen3:8b` or run `autotune recommend` for a hardware-matched suggestion.

**"Memory pressure — context 8192→6144 tokens"**
→ RAM is 88%+ full. Close other apps or switch to a smaller model.

**HTTP 429 — queue full**
→ Too many concurrent requests. Increase `AUTOTUNE_MAX_QUEUED` or wait for one to finish.

**First message is slow**
→ Expected — the model loads and the KV buffer initializes on the first request. Subsequent messages respond immediately.

---

## CLI command reference

### Get started

| Command | What it does |
|---------|-------------|
| `autotune run <model>` | Pre-flight RAM check + chat in one step. Best first command for any new model. |
| `autotune chat --model <id>` | Start an optimized chat session with a model already installed. |
| `autotune hardware` | Scan CPU/RAM/GPU, show which models fit, and suggest apps to close for more RAM. |
| `autotune recommend` | Profile your hardware and recommend the best model+settings. Prints exact `ollama pull` commands. |

### Manage models

| Command | What it does |
|---------|-------------|
| `autotune ls` | List downloaded models with fit scores, safe context window, and recommended profile. |
| `autotune ps` | Show every model currently loaded in RAM across Ollama, MLX, and LM Studio. |
| `autotune pull [model]` | Download an Ollama model. Omit the name to browse hardware-aware recommendations. |
| `autotune models` | List local models with size, architecture, and quality tier. `--registry` shows autotune's full catalog. |
| `autotune unload [model]` | Release a model from memory immediately. Interactive picker if no model specified. |

### Deploy & integrate

| Command | What it does |
|---------|-------------|
| `autotune serve` | Start an OpenAI-compatible API server on `localhost:8765`. All optimizations applied automatically. |

### Benchmarking & proof

| Command | Duration | What it does |
|---------|----------|-------------|
| `autotune proof -m <model>` | ~30 s | Quick head-to-head: raw Ollama vs autotune. Shows TTFT, KV RAM, swap events, RAM headroom. |
| `autotune proof-suite -m <model>` | ~10 min | 5-prompt statistical suite. Wilcoxon signed-rank + Cohen's d + 95% CI across multiple models. |
| `autotune bench -m <model>` | ~15 min | Intensive multi-prompt benchmark with `--duel`, `--raw`, and `--compare` modes. |
| `autotune user-bench -m <model>` | ~30 min | Real-world UX benchmark: swap events, TTFT consistency, CPU spikes, RAM headroom, 0–100 score. |
| `autotune agent-bench` | ~1–2 h | Agentic multi-turn benchmark across 5 tasks. Shows TTFT growth curves (the key story). |

```bash
# Typical proof workflow
autotune proof -m qwen3:8b                    # quick check (~30s)
autotune proof-suite -m qwen3:8b --runs 5     # statistical confirmation
autotune user-bench -m qwen3:8b --quick       # does it feel better?
```

**Key flags for `autotune proof`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model, -m` | auto | Ollama model ID. Auto-selects if omitted. |
| `--runs, -r` | `2` | Runs per condition. 3+ gives stabler numbers. |
| `--profile, -p` | `balanced` | autotune profile to test. |
| `--output, -o` | `proof_<model>.json` | Save JSON results. |
| `--list-models` | — | Print installed models and exit. |

### Conversation memory

| Command | What it does |
|---------|-------------|
| `autotune memory search "<query>"` | Search past conversations by meaning (vector) or keyword (FTS5 fallback). |
| `autotune memory list` | List recently stored memory chunks with timestamps and model names. |
| `autotune memory stats` | Show total chunks, vector coverage, DB size, date range, and per-model counts. |
| `autotune memory forget <id>` | Delete a specific memory chunk. `--all` wipes everything (with confirmation). |
| `autotune memory setup` | Pull `nomic-embed-text` (~274 MB) to enable semantic vector search. |

```bash
autotune memory setup                          # one-time: enable semantic search
autotune memory search "FastAPI auth"          # find relevant past sessions
autotune memory list --days 7                  # recent memories
autotune memory forget 42                      # remove a specific chunk
```

### Apple Silicon (MLX)

| Command | What it does |
|---------|-------------|
| `autotune mlx list` | List MLX models already cached locally. |
| `autotune mlx pull <model>` | Download MLX-quantized model from mlx-community on HuggingFace. Accepts Ollama names. |
| `autotune mlx resolve <model>` | Show which HuggingFace MLX model ID would be used for a given Ollama name. |

MLX is 10–40% faster than Ollama on the same model by running on Apple's unified memory and Metal GPU kernels.

```bash
autotune mlx pull qwen3:8b                     # download 4-bit MLX version
autotune mlx pull qwen2.5-coder:14b --quant 8bit
autotune serve --mlx                           # start API server using MLX backend
```

### Settings & diagnostics

| Command | What it does |
|---------|-------------|
| `autotune telemetry` | View recent inference runs (TTFT, tok/s, RAM, swap, CPU). |
| `autotune telemetry --enable` | Opt in to anonymous telemetry (hardware fingerprint + perf data). |
| `autotune telemetry --disable` | Opt out. No further data sent. |
| `autotune telemetry --status` | Show current consent status. |
| `autotune storage on\|off\|status` | Enable/disable local SQLite storage of performance observations. |
| `autotune doctor` | Full health check: Python, packages, Ollama connectivity, RAM/swap, DB health. |

---

## Architecture

```
autotune/
├── ttft/          ← TTFT optimisation (start here for latency work)
│   └── optimizer.py    TTFTOptimizer: dynamic num_ctx + keep_alive + num_keep
│
├── api/           ← Inference pipeline
│   ├── server.py       FastAPI server — OpenAI-compatible /v1 + FIFO queue
│   ├── chat.py         Terminal REPL with adaptive RAM + live stats
│   ├── kv_manager.py   KV options builder: flash_attn, num_batch, pressure tiers
│   ├── model_selector.py   Pre-flight fit analysis
│   └── backends/       Ollama, MLX, LM Studio, HuggingFace Inference API
│
├── context/       ← Context window management
│   ├── window.py       ContextWindow orchestrator
│   ├── budget.py       Tier thresholds (FULL → RECENT+FACTS → COMPRESSED → EMERGENCY)
│   ├── classifier.py   Message value scoring
│   ├── compressor.py   Tool output + long-content compression
│   └── extractor.py    Deterministic fact extraction
│
├── recall/        ← Conversation memory
│   ├── store.py        SQLite WAL: FTS5 full-text + float32 cosine vectors
│   └── manager.py      save / search / list conversations
│
├── db/            ← Persistence
│   └── store.py        SQLite: models, hardware, run_observations, telemetry_events
│
├── hardware/      ← Hardware detection
│   ├── profiler.py     CPU/GPU/RAM detection
│   └── ram_advisor.py  Real-time RAM pressure advice
│
├── memory/        ← Memory estimation + no-swap guarantee
│   ├── estimator.py    Model weights + KV + runtime overhead
│   └── noswap.py       NoSwapGuard: adjusts num_ctx to guarantee no swap
│
└── cli.py         ← Entry point (Click)
```

---

## License

MIT
