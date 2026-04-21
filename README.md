# autotune — Local LLM Inference Optimizer

[![PyPI](https://img.shields.io/pypi/v/llm-autotune)](https://pypi.org/project/llm-autotune/)
[![Python](https://img.shields.io/pypi/pyversions/llm-autotune)](https://pypi.org/project/llm-autotune/)
[![CI](https://github.com/tanavc1/local-llm-autotune/actions/workflows/test.yml/badge.svg)](https://github.com/tanavc1/local-llm-autotune/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**39% faster time-to-first-word. 3× less KV cache. Drop-in for Ollama, LM Studio, and MLX.**

autotune is a middleware layer that makes your local LLMs faster and lighter — without changing your code or workflow. It computes the exact KV cache each request needs, pins the system prompt in memory, and manages context windows automatically.

```bash
pip install llm-autotune
autotune chat --model qwen3:8b   # that's it
```

Works with **Ollama**, **LM Studio**, and **MLX** (Apple Silicon native) out of the box.

---

## What autotune actually improves

Benchmarked on Apple M2 16 GB using Ollama's own internal nanosecond timers (not wall-clock estimates). Results are means across 3 runs × 5 prompt types, with Wilcoxon signed-rank statistical testing and Cohen's d effect size.

| KPI | llama3.2:3b | gemma4:e2b | qwen3:8b | Average |
|-----|:-----------:|:----------:|:--------:|:-------:|
| **Time to first word (TTFT)** | −35% | −29% | **−53%** | **−39%** |
| **KV prefill time** | −66% | −64% | **−72%** | **−67%** |
| **KV cache size** | −66% | **−69%** | −66% | **−67%** |
| **Peak RAM (LLM process)** | −11% | −0% | −7% | −6% |
| **Generation speed (tok/s)** | −2% | +0.2% | +2.4% | +0.3% |

> **Timing source:** `prompt_eval_duration`, `total_duration`, and `load_duration` from Ollama's Go runtime — not Python clocks.

### What the numbers mean in plain English

**You wait 39% less for the first word.** On qwen3:8b — the most popular 8B model — that's 53% faster. On a complex long-context prompt, up to 89% faster. You feel this on every message.

**KV cache shrinks 3×.** Raw Ollama allocates a fixed 4096-token KV buffer regardless of prompt length. autotune computes the exact size each request needs. For a typical chat message that's 448–576 MB → 143–200 MB freed before inference even starts.

**Generation speed is unchanged.** Token generation on Apple Silicon is Metal GPU-bound. No software layer above Metal changes this. The ±2% range in our data is measurement noise — we don't claim otherwise.

**122,778 KV buffer slots freed** across all benchmark runs — slots Ollama never had to allocate or zero-initialize. No prompt tokens were dropped; `prompt_eval_count` is identical in both conditions.

### What autotune does NOT improve

| Metric | Why |
|--------|-----|
| **Generation throughput (tok/s)** | Metal GPU-bound. Measured +0.3% — that's noise. |
| **RAM (dramatically)** | Model weights dominate process RSS. KV savings are real but ~6–11% of total. |
| **Output quality** | autotune never truncates prompt tokens. `prompt_eval_count` is identical in both conditions — same content, smaller buffer. |

### Run the proof yourself

```bash
# Benchmark all three reference models:
python scripts/proof_suite.py

# Single model:
python scripts/proof_suite.py --models qwen3:8b --runs 5

# Any model you have:
python scripts/proof_suite.py --models YOUR_MODEL --runs 3
```

**Raw data:** Every run, every timing, full JSON — [`llama3.2:3b`](proof_results_v2.json) · [`gemma4:e2b`](proof_results_gemma4.json) · [`qwen3:8b`](proof_results_qwen3.json)

---

## Agentic workloads

Where raw Ollama's fixed `num_ctx=4096` hurts most is inside agent loops — where tool calls, observations, and reasoning steps accumulate across turns. When context exceeds the KV window, Ollama reloads the model. Latency spikes. Swap starts. The task stalls.

autotune computes a single `session_num_ctx` sized for the full task's context ceiling before the loop starts, then holds it constant across every turn. Combined with `num_keep` prefix caching — which pins the system prompt in KV so it's never re-evaluated after turn 1 — autotune keeps agent sessions stable as context accumulates.

**Measured on `llama3.2:3b`, multi-turn tool-calling agent task:**

| Metric | Raw Ollama | autotune |
|--------|:----------:|:--------:|
| Model reloads per session | 0–1 | ~0 |
| Swap events | 1 of 3 trials | 0 |
| TTFT trend per turn | −101 ms/turn | −435 ms/turn (prefix cache compounding) |
| Tool call errors | 1 avg | 0 |
| Context tokens at session end | 3,043 | 1,946 (−36%) |
| **Turn 1 TTFT** | **529 ms** | **953 ms (slower — expected)** |
| **Peak RAM** | **2.39 GB** | **2.85 GB (higher — expected)** |

**The trade-off is explicit and intentional:** autotune pre-allocates a larger KV window at session start. Turn 1 is slower and uses more RAM. From turn 2 onward, prefix caching pays that cost back — TTFT per turn falls as the session grows, while raw Ollama's grows. For 1–2 turn sessions, raw Ollama is faster.

> Full methodology, raw data, and where autotune doesn't help: [AGENT_BENCHMARK.md](AGENT_BENCHMARK.md)

---

## Quickstart

### 1. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen3:8b           # 5.2 GB — best general model for 16 GB machines
```

### 2. Install autotune

```bash
pip install llm-autotune
```

**Requirements:** Python 3.10+, Ollama running locally.

```bash
# Apple Silicon acceleration (native Metal GPU kernels):
pip install "llm-autotune[mlx]"

# Development:
git clone https://github.com/tanavc1/local-llm-autotune.git
cd local-llm-autotune && pip install -e ".[dev]"
```

### 3. Check your hardware

```bash
autotune hardware
```

Shows CPU, RAM, GPU backend, and the effective memory budget autotune uses when sizing the KV cache.

### 4. See what models fit

```bash
autotune ls
```

Scores every locally downloaded model against your hardware — shows whether it fits comfortably, has swap risk, or will OOM. Recommends a profile for each.

### 5. Start chatting

```bash
autotune run qwen3:8b                            # pre-flight check + optimized chat
autotune chat --model qwen3:8b                   # skip pre-flight (always optimized)
autotune chat --model qwen3:8b --profile fast    # fastest responses
autotune chat --model qwen3:8b --profile quality # largest context window
autotune chat --model qwen3:8b --no-swap         # guarantee no macOS swap
autotune chat --model qwen3:8b --system "You are a concise coding assistant."
```

### 6. Check what's running

```bash
autotune ps   # all models in memory — Ollama + MLX — with RAM, context, quant, age
```

---

## Model recommendations by hardware

| RAM | Recommended model | Size | Why |
|-----|------------------|------|-----|
| 8 GB | `qwen3:4b` | ~2.6 GB | Best 4B available; hybrid thinking mode |
| 16 GB | `qwen3:8b` | ~5.2 GB | Near-frontier quality; best 8B as of 2026 |
| 16 GB | `gemma4` | ~5.8 GB | Google's newest; multimodal, 128k context |
| 24 GB | `qwen3:14b` | ~9.0 GB | Excellent reasoning; comfortable headroom |
| 32 GB | `qwen3:30b-a3b` | ~17 GB | MoE: flagship quality at 7B inference cost |
| 64 GB+ | `qwen3:32b` | ~20 GB | Top dense open model |
| Coding | `qwen2.5-coder:14b` | ~9.0 GB | Best open coding model for 24 GB machines |
| Reasoning | `deepseek-r1:14b` | ~9.0 GB | Chain-of-thought; strong math and logic |

Run `autotune ls` to see how each installed model scores against your specific hardware.

---

## What it does

| Feature | What happens |
|---------|-------------|
| **Dynamic KV sizing** | Computes the exact `num_ctx` each request needs — typically 4–8× less KV cache than a fixed 4096-token buffer |
| **KV prefix caching** | Pins system-prompt tokens via `num_keep` so they're never re-evaluated each turn |
| **Adaptive KV precision** | Downgrades F16 → Q8 under memory pressure (80% → −10% ctx, 88% → −25% ctx + Q8, 93% → −50% ctx + Q8) |
| **Model keep-alive** | Sets `keep_alive=-1m` so the model stays loaded between turns — eliminates reload latency |
| **Flash attention** | Enables `flash_attn=true` on every request — reduces peak KV activation memory; zero quality impact |
| **Prefill batching** | Sets `num_batch=1024` (2× Ollama default) — fewer Metal kernel dispatches for long prompts |
| **Multi-tier context management** | Trims conversation history at token budget thresholds, never mid-sentence |
| **Inference queue** | FIFO (default: 1 concurrent, 8 waiting) with HTTP 429 back-pressure — prevents parallel inference from thrashing memory |
| **OpenAI-compatible API** | Drop-in server at `localhost:8765/v1` — works with any OpenAI SDK |
| **MLX backend** | On M-series Macs, routes inference to MLX-LM for native Metal GPU kernels |
| **Persistent memory** | Every conversation saved to SQLite; semantically searches past sessions at startup |
| **No-swap guarantee** | `--no-swap` mode reduces context window to ensure zero macOS swap |

---

## Chat commands

| Command | What it does |
|---------|-------------|
| `/help` | Show available commands |
| `/new` | Start a new conversation (keeps model and profile) |
| `/history` | Show full conversation history |
| `/profile fast\|balanced\|quality` | Switch profile mid-conversation |
| `/model <id>` | Switch to a different model |
| `/system <text>` | Set or replace the system prompt |
| `/export` | Export conversation to Markdown |
| `/metrics` | Session stats (tok/s, TTFT, request count) |
| `/backends` | Show which backends are running |
| `/models` | List all locally available models |
| `/recall` | Browse past conversations with dates and snippets |
| `/recall search <query>` | Semantic search across all past sessions |
| `/pull <model>` | Pull a model from Ollama without leaving chat |
| `/delete` | Delete the current conversation from history |
| `/quit` | Exit (also Ctrl-C) |

---

## Apple Silicon (MLX acceleration)

```bash
pip install "llm-autotune[mlx]"
autotune mlx pull qwen3:8b        # download MLX-quantized model
autotune chat --model qwen3:8b    # automatically routes to MLX
autotune mlx list                 # show locally cached MLX models
autotune mlx resolve llama3.2     # check which MLX repo would be used
```

MLX activates automatically on Apple Silicon — no configuration needed.

> **Tool calling note:** MLX models do not support OpenAI-format tool calls. If your workflow requires structured tool calls (e.g. agentic frameworks), use Ollama-only models or set `use_mlx=False` in `autotune.start()`.

---

## API server (OpenAI-compatible)

```bash
autotune serve
# Listening at http://127.0.0.1:8765/v1
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

### Concurrency tuning

The server serialises inference by default (1 concurrent, 8 queued). Tune with env vars:

```bash
AUTOTUNE_MAX_CONCURRENT=1    # parallel inference slots (default: 1)
AUTOTUNE_MAX_QUEUED=8        # max requests waiting (default: 8)
AUTOTUNE_WAIT_TIMEOUT=120    # seconds before queued request gets 429 (default: 120)
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

autotune manages the server lifecycle, model keep-alive, KV optimisation, and memory pressure automatically. `start()` is safe to call on every app launch — it checks `/health` first and returns immediately if already running.

### `autotune.start()` options

```python
autotune.start(
    host="localhost",   # bind interface
    port=8765,          # default 8765
    timeout=30.0,       # raise TimeoutError if server isn't ready within this many seconds
    profile="balanced", # "fast" | "balanced" | "quality"
    use_mlx=False,      # False = Ollama only (~94 MB RAM, full tool calling)
                        # True  = MLX on Apple Silicon (~470 MB, faster, no tool calls)
    log_level="warning",
)
```

### Check model readiness

```python
import httpx

status = httpx.get("http://localhost:8765/v1/models/qwen3:8b/status").json()
# status["status"]:  "ready" | "available" | "not_found"
# status["fit"]["class"]:  "safe" | "marginal" | "swap_risk" | "oom"
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

### Memory footprint

| Mode | Server RAM | Tool calling | Throughput |
|------|-----------|:------------:|-----------|
| `autotune.start()` (default) | ~94 MB | ✓ | Ollama |
| `autotune.start(use_mlx=True)` | ~470 MB | ✗ | MLX (10–40% faster on Apple Silicon) |

---

## Agentic frameworks

autotune's OpenAI-compatible server works as a drop-in local LLM provider for any framework that accepts a custom base URL.

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
    model: hermes3
    supportsTools: true
```

### Hermes Agent

```yaml
# ~/.hermes/config.yaml
model:
  provider: custom
  base_url: http://localhost:8765/v1
  api_key: sk-local
  name: hermes3
```

### Tool calling support

Models confirmed working for tool calling via Ollama: `hermes3`, `qwen3:8b`, `qwen3:14b`, `llama3.1:8b`, `qwen2.5-coder:14b`

Models that do **not** support tool calling: `llama3.2:3b`, `gemma4:e2b`

---

## Profiles

| Profile | Context | Temperature | KV precision | Use when |
|---------|--------:|:-----------:|:------------:|---------|
| `fast` ⚡ | 2,048 | 0.1 | Q8 | Quick lookups, autocomplete |
| `balanced` ⚖️ | 8,192 | 0.7 | F16 | General chat, coding |
| `quality` ✨ | 32,768 | 0.8 | F16 | Long-form writing, analysis |

---

## How dynamic KV sizing works

Ollama allocates the entire KV cache upfront before generating a single token. If `num_ctx=4096`, it zeros and initialises a 4096-token buffer even if your prompt is 50 tokens. That initialization is what you're waiting for.

autotune computes the minimum `num_ctx` each request actually needs:

```
num_ctx = clamp(input_tokens + max_new_tokens + 256, 512, profile_max)
```

For a short conversation on `balanced` (max 8,192):
- Input: ~22 tokens → `num_ctx` = 22 + 1,024 + 256 = **1,302**
- Savings on qwen3:8b: 4,096 → 1,302 tokens = **~224 MB of KV cache never allocated**

`num_ctx` grows naturally as the conversation grows since the full history is included on every request. No tokens are ever dropped.

---

## Context management tiers

autotune monitors `history_tokens / effective_budget` and selects a strategy automatically:

```
< 55%   FULL              — all turns verbatim
55–75%  RECENT+FACTS      — last 8 turns + structured facts block for older turns
75–90%  COMPRESSED        — last 6 turns (lightly compressed) + compact summary
> 90%   EMERGENCY         — last 4 turns (aggressively compressed) + one-line summary
```

Low-value chatter ("ok", "thanks") is dropped first. Code blocks, stack traces, and technical content are always preserved. All cutoffs happen at sentence or paragraph boundaries — never mid-sentence.

The facts block for older turns is extracted deterministically (no LLM call) and includes accomplishments, decisions, errors, and topics covered.

---

## Conversation memory and recall

Every conversation is automatically saved to a local SQLite database with both full-text search and vector similarity. No flags required.

- **Automatic context injection** — at session start, autotune searches past conversations for similar topics and injects relevant facts as a silent system message.
- **Session resume** — use `--conv-id <id>` to resume an exact past session with full context.
- **In-chat recall** — `/recall` to browse recent sessions; `/recall search <topic>` for semantic search.

### Storage paths

| Path | Contents |
|------|----------|
| `~/.autotune/recall.db` | FTS5 + float32 vectors; conversation turns, extracted facts |
| `~/Library/Application Support/autotune/autotune.db` | Hardware telemetry, run observations (macOS) |
| `~/.local/share/autotune/autotune.db` | Same (Linux) |

---

## Telemetry

### View past runs

```bash
autotune telemetry                    # last 20 inference runs
autotune telemetry --events           # notable events: swap spikes, OOMs
autotune telemetry --model qwen3:8b   # filter by model
```

### Anonymous cloud telemetry (opt-in, off by default)

```bash
autotune telemetry --status    # check opt-in status
autotune telemetry --enable    # opt in
autotune telemetry --disable   # opt out
```

**What is collected (only if opted in):**
- Hardware class: CPU architecture, RAM size, GPU backend — no hostnames, usernames, serial numbers, or IP addresses
- Model performance: tokens/sec, TTFT, context size, quantization label
- Session events: server start/stop, OOM events

Data goes to a private Supabase database. Never sold or shared. Collection logic: `autotune/telemetry/`.

### Local storage opt-out

```bash
autotune storage off     # disable local SQLite writes (run observations, telemetry)
autotune storage on      # re-enable
autotune storage status  # check current setting
```

---

## Known limitations

- **Generation speed** — Token generation is GPU-bound. autotune does not affect tok/s.
- **Turn 1 is slower for multi-turn sessions** — Pre-allocating a larger KV window costs time on the first turn; subsequent turns benefit from prefix caching.
- **Tool calling on MLX** — MLX models cannot relay OpenAI-format tool calls. Use Ollama-backed models for agentic workflows requiring tool calls.
- **Vision models** — autotune is text-only; image inputs are dropped.
- **Single-machine only** — autotune is designed for local, single-host inference, not distributed setups.

---

## Troubleshooting

**"Ollama is not running."**
→ Start Ollama: `ollama serve` (in a separate terminal)

**"No models found."**
→ Pull a model: `ollama pull qwen3:8b` or `autotune pull qwen3:8b`

**"Memory pressure — context 8192→6144 tokens"**
→ System RAM is 88%+ full. Close other apps or try a smaller model.

**HTTP 429 — queue full**
→ Too many concurrent requests. Increase `AUTOTUNE_MAX_QUEUED` or wait for one to finish.

**First message is slow**
→ Expected — KV buffer initialization on turn 1. Subsequent messages are fast (prefix cache).

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
│   └── noswap.py       NoSwapGuard: adjusts num_ctx/KV to guarantee no swap
│
└── cli.py         ← Entry point (Click)
```

---

## License

MIT
