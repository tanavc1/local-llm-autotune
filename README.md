# autotune — Local LLM Inference Optimizer

A middleware layer that makes locally-running LLMs faster, lighter, and smarter on your own hardware — with zero changes to your existing setup.

Works with **Ollama**, **LM Studio**, and **MLX** (Apple Silicon native) backends out of the box.

---

## What it does

autotune sits between your application and the local LLM backend. It automatically:

| Feature | What happens |
|---------|-------------|
| **Dynamic KV sizing** | Computes the exact `num_ctx` each request needs instead of allocating the profile max — typically 4–8× less KV cache memory |
| **KV prefix caching** | Pins system-prompt tokens in Ollama's KV cache via `num_keep` so they're never re-evaluated each turn |
| **Adaptive KV precision** | Downgrades KV cache from F16 → Q8 under memory pressure (80% → −10% ctx, 88% → −25% ctx + Q8, 93% → −50% ctx + Q8) |
| **Multi-tier context management** | Intelligently trims conversation history at token budget thresholds with no mid-sentence cuts |
| **Inference queue** | FIFO queue (default: 1 concurrent, 8 waiting) with HTTP 429 back-pressure — prevents parallel inference from thrashing memory |
| **Profile-based optimization** | `fast` / `balanced` / `quality` profiles tune temperature, context length, KV precision, and OS QoS class |
| **OpenAI-compatible API** | Drop-in replacement for `localhost:8765/v1` — works with any OpenAI SDK |
| **MLX backend (Apple Silicon)** | On M-series Macs, routes inference to MLX-LM — native Metal GPU kernels, unified memory, ~20% lower TTFT than Ollama on the same model |
| **Hardware telemetry** | Samples RAM/Swap/CPU every 250 ms, persists structured metrics to SQLite |

---

## Quickstart

### 1. Prerequisites

Install [Ollama](https://ollama.com) and pull at least one model:

```bash
ollama pull phi4-mini        # 2.5 GB — good starting point on any machine
ollama pull qwen2.5-coder:14b  # 9 GB — great coding model for 16+ GB RAM
```

### 2. Install autotune

```bash
git clone https://github.com/tanavc1/local-llm-autotune.git
cd local-llm-autotune
pip install -e .
```

**Requirements:** Python 3.10+, Ollama running locally.

### 3. Check your hardware

```bash
autotune hardware
```

Shows CPU, RAM, GPU backend, and the effective memory budget autotune uses when selecting models.

### 4. See what models fit

```bash
autotune ls
```

Scores every locally downloaded Ollama model against your hardware — shows whether it fits comfortably, has swap risk, or will OOM. Recommends a profile for each.

### 5. Start chatting

The fastest way to get started — autotune analyses memory fit, picks the right profile automatically, and opens a chat session:

```bash
autotune run phi4-mini:latest
```

Or start a chat with a specific profile:

```bash
autotune chat --model phi4-mini:latest                   # balanced (default)
autotune chat --model phi4-mini:latest --profile fast    # fastest responses
autotune chat --model phi4-mini:latest --profile quality # largest context
```

Set a system prompt:

```bash
autotune chat --model phi4-mini:latest --system "You are a concise coding assistant."
```

Resume a previous conversation (the ID is shown in the chat header):

```bash
autotune chat --model phi4-mini:latest --conv-id a3f92c1b
```

---

## Chat commands

Once inside a chat session, these slash commands are available:

| Command | What it does |
|---------|-------------|
| `/help` | Show available commands |
| `/new` | Start a new conversation (keeps model and profile) |
| `/history` | Show the full conversation history |
| `/profile fast\|balanced\|quality` | Switch profile mid-conversation |
| `/model <id>` | Switch to a different model |
| `/system <text>` | Set or replace the system prompt |
| `/export` | Export conversation to a Markdown file |
| `/metrics` | Show session performance stats (tok/s, TTFT, request count) |
| `/backends` | Show which backends are running (Ollama, LM Studio, HF API) |
| `/models` | List all locally available models |
| `/quit` | Exit (also Ctrl-C) |

---

## Apple Silicon (MLX acceleration)

On M-series Macs, install the MLX backend to use native Metal GPU kernels:

```bash
pip install -e ".[mlx]"           # install mlx-lm
autotune mlx pull phi4-mini       # download MLX-quantized model from mlx-community
autotune chat --model phi4-mini   # automatically routes to MLX on Apple Silicon
```

MLX is activated automatically when running on Apple Silicon — no configuration needed. autotune resolves the Ollama model name to the corresponding `mlx-community` HuggingFace repo.

```bash
autotune mlx list                 # show locally cached MLX models
autotune mlx resolve llama3.2     # check which MLX model ID would be used
```

---

## API server (OpenAI-compatible)

Run autotune as a server and point any existing OpenAI client at it:

```bash
autotune serve
# Listening at http://127.0.0.1:8765/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8765/v1", api_key="local")
response = client.chat.completions.create(
    model="phi4-mini:latest",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Autotune-specific headers

Pass these with any request to override behavior per-call:

```
X-Autotune-Profile: fast        # override profile (fast | balanced | quality)
X-Conversation-Id: a3f92c1b     # attach to a persistent conversation
```

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible, streaming or non-streaming |
| `GET /v1/models` | List all available models across all backends |
| `GET /health` | Server status, queue depth, memory pressure |
| `GET /api/hardware` | Live hardware snapshot |
| `GET /api/profiles` | Profile definitions |
| `POST /api/conversations` | Create a persistent conversation |
| `GET /api/conversations` | List conversations |
| `GET /api/conversations/{id}` | Get conversation + full message history |
| `DELETE /api/conversations/{id}` | Delete conversation |
| `GET /api/conversations/{id}/export` | Export as Markdown |

### Concurrency tuning

The server serialises inference by default (1 concurrent request, 8 queued). Requests beyond the queue limit receive HTTP 429 immediately. Tune with env vars:

```bash
AUTOTUNE_MAX_CONCURRENT=1    # parallel inference slots (default: 1)
AUTOTUNE_MAX_QUEUED=8        # max requests waiting for a slot (default: 8)
AUTOTUNE_WAIT_TIMEOUT=120    # seconds before a waiting request gets 429 (default: 120)
```

---

## Profiles

| Profile | Context | Temperature | KV precision | System QoS | Use when |
|---------|--------:|:-----------:|:------------:|:----------:|---------|
| `fast` ⚡ | 2,048 | 0.05 | Q8 | USER_INTERACTIVE | Quick lookups, autocomplete |
| `balanced` ⚖️ | 8,192 | 0.70 | F16 | USER_INITIATED | General chat, coding |
| `quality` ✨ | 32,768 | 0.80 | F16 | USER_INITIATED | Long-form writing, analysis |

`autotune run` with `--profile auto` (the default) analyses model size vs. available RAM and picks the profile automatically.

---

## Telemetry and benchmarks

### View past runs

```bash
autotune telemetry               # last 20 inference runs
autotune telemetry --events      # notable events: swap spikes, OOMs, slow tokens
```

### Run a benchmark

```bash
python scripts/benchmark.py --model phi4-mini:latest --runs 3
```

Runs the model against raw Ollama (no autotune) and autotune fast/balanced, measures TTFT and throughput across multiple prompts, and prints a before/after comparison.

### Where data is stored

All runs persist to SQLite automatically:
- **macOS:** `~/Library/Application Support/autotune/autotune.db`
- **Linux:** `~/.local/share/autotune/autotune.db`

---

## Benchmark results

Tested on **phi4-mini:latest** (2.5 GB, Q4_K_M) via Ollama on Apple M2 16 GB. 3 runs × 4 prompts. All results persisted to DB.

| Variant | tok/s | TTFT (ms) | RAM Δ (GB) | CPU avg % |
|---------|------:|----------:|-----------:|----------:|
| raw\_ollama | 31.1 ± 5.8 | 601 ± 1032 | +1.1 ± 0.6 | 19.1 |
| autotune/fast | 32.2 ± 4.6 | **224 ± 28** | +0.9 ± 0.7 | 14.0 |
| autotune/balanced | 31.9 ± 4.6 | **247 ± 76** | +1.0 ± 0.6 | 15.9 |

**TTFT −59%** (601 ms → 247 ms). The dominant driver is system-prompt prefix caching (`num_keep`): raw Ollama re-tokenises the full prefix on every turn; autotune pins those tokens in the KV cache so generation starts immediately. The `short_qa` cold-start improvement is most dramatic: 1,471 ms → 224 ms.

**CPU average −26%** (19.1% → 14.0% for fast). The fast profile sets OS QoS class, disables background GC pressure during inference, and right-sizes the context window.

### MLX backend (Apple Silicon)

| Backend | TTFT (ms) | tok/s |
|---------|----------:|------:|
| MLX (mlx-community/Phi-4-mini-instruct-4bit) | **334** | 34.4 |
| Ollama (phi4-mini:latest, Q4_K_M) | 416 | 40.7 |
| **MLX improvement** | **−20% TTFT** | — |

> **Honest note:** results are on a single small model. Larger models with longer system prompts show bigger TTFT gains from prefix caching (the benefit scales with system prompt length). On 7B+ models, MLX throughput consistently beats Ollama by 15–40% as Metal matrix-multiply throughput dominates.

---

## How dynamic KV sizing works

Ollama allocates the entire KV cache upfront before generating a single token. If `num_ctx=8192`, it allocates memory for 8,192 tokens even if your conversation is 50 tokens.

autotune computes the minimum `num_ctx` each request actually needs:

```
num_ctx = clamp(input_tokens + max_new_tokens + 256, 512, profile_max)
```

For a short conversation on the `balanced` profile (max 8,192):
- Input: ~22 tokens → `num_ctx` = 22 + 1,024 + 256 = **1,302**
- Savings on `qwen2.5-coder:14b`: 8,192 → 1,302 tokens = **~677 MB of KV cache freed**

`num_ctx` grows naturally as the conversation grows since the full history is included in every request.

---

## Context management tiers

autotune monitors `history_tokens / effective_budget` and selects a strategy automatically:

```
< 55%   FULL              — all turns verbatim, nothing dropped
55–75%  RECENT+FACTS      — last 8 turns + structured facts block for older turns
75–90%  COMPRESSED        — last 6 turns (lightly compressed) + compact summary
> 90%   EMERGENCY         — last 4 turns (aggressively compressed) + one-line summary
```

Low-value chatter ("ok", "thanks") is dropped first. Code blocks, stack traces, and technical content are always preserved. All cutoffs happen at sentence or paragraph boundaries — never mid-sentence.

The facts block injected for older turns is extracted deterministically (no LLM call) and includes accomplishments, active decisions, key facts, errors, and topics covered.

---

## Architecture

```
autotune/
├── api/
│   ├── server.py          # FastAPI server — OpenAI-compatible /v1 endpoints + FIFO queue
│   ├── kv_manager.py      # KV cache: num_keep, adaptive num_ctx, pressure thresholds
│   ├── ctx_utils.py       # Token estimation, dynamic num_ctx computation
│   ├── profiles.py        # fast / balanced / quality profiles
│   ├── conversation.py    # SQLite-backed persistent conversation state
│   ├── model_selector.py  # Pre-flight fit analysis: weights + KV + runtime overhead
│   ├── hardware_tuner.py  # OS-level tuning: nice, QoS class, GC, CPU governor
│   ├── chat.py            # Terminal REPL
│   └── backends/          # Ollama, LM Studio, MLX, HuggingFace Inference API
├── context/
│   ├── window.py          # ContextWindow orchestrator
│   ├── budget.py          # Tier thresholds and recent-window sizes
│   ├── classifier.py      # Message value scoring (0.0 chatter → 1.0 technical)
│   ├── compressor.py      # Tool output and long-content compression
│   └── extractor.py       # Deterministic fact extraction for summary blocks
├── bench/
│   └── runner.py          # Benchmark with 250 ms hardware sampling
├── db/
│   └── store.py           # SQLite: models, hardware, run_observations, telemetry_events
├── hardware/
│   └── profiler.py        # CPU/GPU/RAM detection
└── cli.py                 # Entry point (Click)
```

---

## License

MIT
