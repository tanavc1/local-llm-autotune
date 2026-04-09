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
| **Model keep-alive** | Sets `keep_alive=-1` so the model stays loaded in unified memory between turns — eliminates reload latency |
| **Multi-tier context management** | Intelligently trims conversation history at token budget thresholds with no mid-sentence cuts |
| **Inference queue** | FIFO queue (default: 1 concurrent, 8 waiting) with HTTP 429 back-pressure — prevents parallel inference from thrashing memory |
| **Profile-based optimization** | `fast` / `balanced` / `quality` profiles tune temperature, context length, KV precision, and OS QoS class |
| **OpenAI-compatible API** | Drop-in replacement for `localhost:8765/v1` — works with any OpenAI SDK |
| **MLX backend (Apple Silicon)** | On M-series Macs, routes inference to MLX-LM — native Metal GPU kernels, unified memory |
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
| `fast` ⚡ | 2,048 | 0.1 | Q8 | USER_INTERACTIVE | Quick lookups, autocomplete |
| `balanced` ⚖️ | 8,192 | 0.7 | F16 | USER_INITIATED | General chat, coding |
| `quality` ✨ | 32,768 | 0.8 | F16 | USER_INITIATED | Long-form writing, analysis |

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
# Full side-by-side comparison (raw Ollama vs autotune profiles)
python scripts/benchmark.py --model phi4-mini:latest --runs 3

# Or using the CLI:
autotune bench --model phi4-mini:latest --raw --tag baseline
autotune bench --model phi4-mini:latest --profile fast --tag fast_opt
autotune bench --compare baseline,fast_opt
```

### Where data is stored

All runs persist to SQLite automatically:
- **macOS:** `~/Library/Application Support/autotune/autotune.db`
- **Linux:** `~/.local/share/autotune/autotune.db`

---

## Benchmark results

> **What autotune actually improves: TTFT (time to first token).**
> Throughput is GPU-bound and autotune does not change it. The numbers below are honest.

### Methodology

**Hardware:** Apple M2, 16 GB unified memory (macOS Sequoia)  
**Model:** `phi4-mini:latest` — Q4_K_M quantization, 2.5 GB weights, served via Ollama  
**Test script:** `scripts/stress_test.py` — automated, no manual intervention  
**Scale:** 63 inference calls across 18 distinct prompts and 6 test phases

Two configurations are compared throughout:

| Configuration | What it does |
|--------------|-------------|
| `raw_ollama` | Zero autotune involvement. Pure Ollama defaults: `num_ctx` unspecified (Ollama picks its own default, typically 4096), `keep_alive=5m`, no OS tuning. Direct HTTP to `/v1/chat/completions`. |
| `autotune/balanced` | Three TTFT mechanisms applied (see below) + OS scheduling priority (`USER_INITIATED` QoS, GC disabled during inference). |

**Metrics** (psutil, sampled every 250 ms in a background thread):
- **TTFT** — wall time from HTTP request start to first streaming token byte
- Throughput (tok/s) — `len(response) / 4 / elapsed_sec`; same formula both sides
- CPU % — system-wide average across all cores during inference
- RAM Δ — `used` after minus `used` before; memory left behind per call

---

### Results

#### Overall (18 prompts, 63 calls)

| Metric | Raw Ollama | autotune/balanced | Δ |
|--------|----------:|------------------:|:--:|
| **TTFT** | 626 ms | **349 ms** | **−44%** |
| Throughput | 35.5 tok/s | 34.8 tok/s | −2% (noise) |
| CPU avg | 10.8% | 12.4% | +15% (wrapper overhead) |
| RAM Δ | +0.76 GB | +0.78 GB | neutral |

#### By scenario

| Scenario | Raw Ollama TTFT | autotune TTFT | Improvement |
|----------|---------------:|---------------:|:-----------:|
| General mix (10 prompts × 2 runs) | 421 ms | 404 ms | −4% |
| Sustained back-to-back (6 calls, no pause) | 282 ms | 265 ms | −6% |
| **Large-context input (>1 000 tokens)** | **2 015 ms** | **261 ms** | **−87%** |
| **Session continuity (`keep_alive` test)** | **1 227 ms** | **244 ms** | **−80%** |

The large-context and session tests show where autotune's value is clearest and most consistent. The general-mix improvement is real but modest.

---

### Where the TTFT reduction comes from

Three mechanisms work together, all owned by `autotune/ttft/optimizer.py`:

#### 1. Dynamic `num_ctx`

Ollama allocates the **entire** KV cache before generating a single token. With the default `num_ctx=4096` it allocates 4 096 token slots regardless of your actual input size — KV allocation is proportional to `num_ctx`, not to actual usage.

autotune computes the minimum that fits the request:

```
num_ctx = clamp(input_tokens + max_new_tokens + 256,  min=512,  max=profile_max)
```

Example with a 60-token question on the balanced profile:

| | num_ctx | KV allocation (qwen2.5-coder:14b F16) |
|--|--------:|--------------------------------------:|
| Raw Ollama | 4 096 | ~402 MB |
| autotune | 1 340 | ~131 MB |

Smaller KV allocation = less memory to initialise before the first token, which is the KV initialisation step that TTFT measures. The −87% on large-context prompts is this mechanism at work: raw Ollama's 4 096 context can barely fit a 1 000-token input, while autotune right-sizes to the actual content.

#### 2. `keep_alive = -1`

Ollama's default is `keep_alive=5m` — after five minutes idle the model is **fully unloaded** from unified memory. The next request pays a full model reload (1–4 s for phi4-mini; longer for larger models).

autotune always sends `keep_alive="-1"` (keep model resident indefinitely).

The cold-start test in the benchmark forces this condition explicitly: the raw-Ollama path unloads the model between calls, the autotune path does not.

```
Raw:      call 1 → reload (1 304 ms TTFT)   call 2 → reload (1 189 ms)   call 3 → reload (1 187 ms)
autotune: call 1 → load   ( 248 ms TTFT)   call 2 → warm  (  242 ms)   call 3 → warm  (  243 ms)
```

#### 3. `num_keep` (system-prompt prefix caching)

When a system prompt is present, autotune passes `num_keep = <system_prompt_tokens>` to Ollama. Ollama pins those tokens in the KV cache and never re-evaluates them on subsequent turns. Raw Ollama re-processes the full prompt from scratch on every call.

For a 120-token system prompt on a 30-turn conversation: autotune saves 120 tokens of attention computation on every single turn.

---

### What autotune does NOT improve

| Metric | Why it doesn't change |
|--------|----------------------|
| **Throughput (tok/s)** | Token generation on Apple Silicon runs on the Metal GPU. No software change above the Metal layer affects how fast the GPU generates tokens. autotune measured −2% (within noise). |
| **RAM usage** | At 16 GB with a 2.5 GB model there is no memory pressure, so the pressure guard never activates. RAM impact is neutral. |
| **CPU %** | The autotune wrapper adds Python overhead (KV option computation, hardware tuner, psutil calls) that slightly increases CPU%. Measured +15% CPU vs raw Ollama. |

---

### Prompt-by-prompt TTFT

| Prompt | Raw (ms) | autotune (ms) | Δ |
|--------|--------:|--------------:|:--:|
| simple factual | 257 | 310 | +21% |
| code (fibonacci) | 383 | 345 | −10% |
| reasoning chain | 378 | 368 | −3% |
| code with long system prompt | 514 | 373 | −27% |
| code review (large input) | 892 | 845 | −5% |
| explain transformer | 334 | 318 | −5% |
| multi-turn follow-up | 417 | 399 | −4% |
| math proof | 322 | 312 | −3% |
| system design | 407 | 405 | ~0% |
| creative technical | 352 | 361 | +3% |
| large context (pressure test) | 2 015 | 261 | **−87%** |
| cold-start / warm session | 1 227 | 244 | **−80%** |

The simple factual prompt shows autotune slightly slower on run 1 because `TTFTOptimizer` has a small warm-up cost on the first call (GC collect, psutil snapshot, option computation). By run 2 of the same session this vanishes. The large-context and cold-start improvements are consistent across all runs.

---

### Limitations

- **Single model.** All numbers are from `phi4-mini:latest` (2.5 GB, Q4_K_M) on Apple M2 16 GB. TTFT gains from `keep_alive=-1` will be proportionally larger on bigger models — a 9 GB model has a longer reload penalty than a 2.5 GB one.
- **`num_ctx` trade-off.** Smaller context means fewer tokens of conversation history fit. For short sessions this is pure win. For very long conversations autotune may need to trim history earlier (handled by `autotune.context.ContextWindow`).
- **Token estimation.** Throughput uses `len(response) / 4 / elapsed_sec`. The same formula is used for both configurations so the comparison is fair, but absolute tok/s may differ from Ollama's internal tokenizer count.

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
│
├── ttft/                  ← TTFT optimisation layer (start here for latency work)
│   ├── optimizer.py       #   TTFTOptimizer: dynamic num_ctx + keep_alive + num_keep
│   └── __init__.py        #   Public API: TTFTOptimizer, KEEP_ALIVE_FOREVER
│
├── api/                   Inference pipeline
│   ├── profiles.py        #   fast / balanced / quality profile definitions
│   ├── server.py          #   FastAPI server — OpenAI-compatible /v1 + FIFO queue
│   ├── chat.py            #   Terminal chat REPL
│   ├── conversation.py    #   SQLite-backed persistent conversation state
│   ├── ctx_utils.py       #   Token estimation + compute_num_ctx (used by ttft/)
│   ├── kv_manager.py      #   KV options builder (wraps ttft/ for legacy callers)
│   ├── hardware_tuner.py  #   OS-level tuning: nice, QoS class, GC, CPU governor
│   ├── model_selector.py  #   Pre-flight fit analysis: weights + KV + overhead
│   └── backends/          #   Ollama, LM Studio, MLX, HuggingFace Inference API
│
├── context/               Context window management for long conversations
│   ├── window.py          #   ContextWindow orchestrator
│   ├── budget.py          #   Tier thresholds (FULL → RECENT+FACTS → COMPRESSED → EMERGENCY)
│   ├── classifier.py      #   Message value scoring (0.0 chatter → 1.0 technical)
│   ├── compressor.py      #   Tool output and long-content compression
│   └── extractor.py       #   Deterministic fact extraction for summary blocks
│
├── bench/                 Benchmarking framework
│   └── runner.py          #   run_raw_ollama / run_bench_ollama_only / BenchResult
│
├── db/                    Persistence
│   └── store.py           #   SQLite: models, hardware, run_observations, telemetry_events
│
├── hardware/              Hardware detection
│   └── profiler.py        #   CPU/GPU/RAM detection (psutil + py-cpuinfo)
│
├── memory/                Memory estimation
│   └── estimator.py       #   Model weights + KV cache + runtime overhead
│
├── models/                Model registry
│   └── registry.py        #   9 OSS models with real MMLU/HumanEval/GSM8K scores
│
├── config/                Recommendation engine
│   └── generator.py       #   Multi-objective scoring: stability × speed × quality × context
│
└── cli.py                 Entry point (Click)
```

---

## License

MIT
