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

### Methodology

**Hardware:** Apple M2, 16 GB unified memory (macOS)  
**Model:** `phi4-mini:latest` — Q4_K_M quantization, ~2.5 GB weights, served via Ollama  
**Script:** `scripts/benchmark.py` — automated, no manual intervention  
**Design:** 3 runs × 4 prompts × 3 configurations = **36 total inference calls**

The three configurations tested:

| Configuration | Description |
|--------------|-------------|
| `raw_ollama` | Zero autotune involvement. Pure Ollama defaults: `num_ctx=4096`, `temperature=0.8`, `keep_alive=5m`, no OS QoS tuning, no GC control. Direct HTTP to Ollama's `/v1/chat/completions`. |
| `autotune/fast` | Dynamic `num_ctx`, `num_keep` prefix caching, `keep_alive=-1`, `repeat_penalty=1.15` inside Ollama options, Q8 KV cache, macOS `USER_INTERACTIVE` QoS, GC disabled during inference. |
| `autotune/balanced` | Same as fast but `num_ctx` capped at 8,192, `repeat_penalty=1.08`, F16 KV cache, `USER_INITIATED` QoS. |

**Prompts used** (diverse, representative workload):

| Prompt | Content |
|--------|---------|
| `short_qa` | "What is the capital of France?" (no system prompt) |
| `code_gen` | "Write a Python function that returns the sum of all even numbers." |
| `long_context` | Multi-turn: explanation of transformer attention, then a follow-up question about self-attention vs cross-attention |
| `system_prompt_repeat` | System prompt: expert Python developer. User: "Write a thread-safe LRU cache." |

**Metrics captured** (psutil, sampled every 250 ms in a background thread):
- TTFT: wall time from request start to first streaming token byte received
- Throughput (tok/s): output tokens estimated as characters/4, divided by elapsed wall time
- CPU%: average over all 250 ms samples taken during inference
- RAM Δ (GB): `psutil.virtual_memory().used` after − before (memory pressure left behind)
- Peak RAM (GB): maximum `psutil.virtual_memory().used` during inference

A 3-second cool-down separated runs to let memory settle. All results are written to the autotune SQLite database and exported to `benchmark_results.json`.

---

### Aggregate results

**4 prompts × 3 runs = 12 observations per configuration.**

| Configuration | TTFT (ms) | Throughput (tok/s) | CPU avg % | RAM Δ (GB) |
|--------------|----------:|------------------:|----------:|-----------:|
| `raw_ollama` | 601 ± 1032 | 31.1 ± 5.8 | 19.1 ± 7.0 | +1.12 ± 0.62 |
| `autotune/fast` | **224 ± 28** | 32.2 ± 4.6 | **14.0 ± 3.9** | +0.90 ± 0.69 |
| `autotune/balanced` | **247 ± 76** | 31.9 ± 4.6 | 15.9 ± 9.6 | +0.97 ± 0.61 |

**vs. raw Ollama:**

| Metric | autotune/fast | autotune/balanced |
|--------|:-------------:|:-----------------:|
| TTFT | **−62.8%** | −59.0% |
| Throughput | +3.5% | +2.4% |
| CPU avg | **−26.5%** | −16.4% |
| RAM Δ | −20.3% | −13.5% |

---

### Honest interpretation of the TTFT result

The −62.8% TTFT figure is accurate but warrants a plain-language explanation:

**The raw_ollama TTFT average of 601 ms includes one cold-start.** On the first `short_qa` run, raw Ollama had a 3,867 ms TTFT — the model had been unloaded from memory (Ollama's default `keep_alive=5m` expires). Runs 2 and 3 of that prompt were 260 ms and 286 ms. This is real-world behavior: Ollama's keep_alive default means the model unloads after 5 minutes of idle time, and the next request pays the full reload cost.

autotune sets `keep_alive=-1` (keep model loaded indefinitely), which eliminates reload latency entirely. On the autotune runs, all 12 TTFT measurements were between 161 ms and 266 ms with no outliers (σ = 28 ms for fast vs. σ = 1,032 ms for raw).

**If you exclude that one cold-start, the warm-inference TTFT improvement is still real:**

| | TTFT (warm only, n=11) |
|--|--|
| `raw_ollama` (excl. cold start) | 304 ± 85 ms |
| `autotune/fast` | 224 ± 28 ms (**−26.6%**) |
| `autotune/balanced` | 247 ± 76 ms (**−19.0%**) |

The remaining warm improvement comes from KV prefix caching (`num_keep`): autotune pins system-prompt tokens in Ollama's KV cache on the first request, so subsequent turns skip re-evaluating them. Raw Ollama re-processes the full prompt on every request.

---

### Per-prompt breakdown

| Prompt | Config | TTFT (ms) | tok/s | CPU % |
|--------|--------|----------:|------:|------:|
| `short_qa` | raw_ollama | 1471 | 28.4 | 18.8 |
| `short_qa` | autotune/fast | 216 | 34.3 | 10.2 |
| `short_qa` | autotune/balanced | 224 | 35.4 | 11.0 |
| `code_gen` | raw_ollama | 267 | 27.7 | 19.0 |
| `code_gen` | autotune/fast | 219 | 27.7 | 17.4 |
| `code_gen` | autotune/balanced | 218 | 28.3 | 9.3 |
| `long_context` | raw_ollama | 347 | 37.8 | 12.6 |
| `long_context` | autotune/fast | 219 | 38.4 | 13.5 |
| `long_context` | autotune/balanced | 255 | 36.5 | 15.4 |
| `system_prompt_repeat` | raw_ollama | 320 | 30.6 | 25.9 |
| `system_prompt_repeat` | autotune/fast | 240 | 28.5 | 15.0 |
| `system_prompt_repeat` | autotune/balanced | 288 | 27.5 | 27.9 |

---

### What drives each improvement

**TTFT (−19% to −63% depending on cold-start exposure)**
Two mechanisms combine:
1. `keep_alive=-1` prevents model unloading. Raw Ollama defaults to `keep_alive=5m`; after 5 minutes idle, the next request incurs a full model reload (3–4 seconds for phi4-mini, longer for larger models).
2. KV prefix caching via `num_keep`: the system prompt's tokens are pinned in the KV cache. Raw Ollama re-tokenises and re-attends over the system prompt on every single turn.

**CPU average (−16% to −27%)**
Three mechanisms:
1. **Dynamic `num_ctx`**: raw Ollama allocates a fixed 4,096-token KV cache regardless of input length. autotune sizes `num_ctx` to `input_tokens + max_new_tokens + 256`. For short queries, this is 500–800 tokens instead of 4,096 — less KV memory to allocate and fill.
2. **macOS QoS class**: `USER_INTERACTIVE` (fast) and `USER_INITIATED` (balanced) give the inference thread higher CPU scheduling priority. This concentrates CPU time during inference, which lowers the average % measured across all cores during the same wall-clock window.
3. **GC disabled during inference**: Python's garbage collector is suspended for the duration of the generation call. On the fast profile, this prevents GC pauses from showing up as spurious CPU samples.

**Throughput (+2–4%)**
Token generation on Apple Silicon is Metal GPU-bound. The marginal gain comes from smaller context windows requiring less memory bandwidth per attention pass. This is a real but modest effect — do not expect dramatic throughput gains from autotune alone. The GPU is the bottleneck.

**RAM Δ (−13% to −20%)**
Dynamic `num_ctx` is the primary driver: a smaller KV allocation means less dirty memory is left behind after each request. Raw Ollama's fixed 4,096 KV cache leaves a larger footprint.

---

### Limitations and scope

- **Single model, single machine.** Results are from phi4-mini:latest (Q4_K_M, 3.8B parameters) on Apple M2 16 GB. The TTFT gain from `keep_alive=-1` scales with reload cost — larger models (14B, 70B) will see larger absolute improvements from eliminating cold starts.
- **Token count estimation.** Throughput is approximated as `len(response) / 4 / elapsed_sec`. This is the same formula used for both raw and autotune measurements, so the comparison is fair, but absolute tok/s numbers may differ from Ollama's internal eval stats.
- **CPU % interpretation.** `psutil.cpu_percent()` reports system-wide CPU utilization averaged across all physical cores. On M2, Metal inference runs on the GPU die; the CPU samples reflect overhead (tokenisation, HTTP, Python runtime), not GPU compute.
- **n=12 per configuration.** With 3 runs × 4 prompts, the sample size is small. Standard deviations are reported; treat the absolute numbers as indicative, not definitive. The TTFT high variance in raw_ollama (σ = 1,032 ms) is driven almost entirely by the single cold-start measurement.

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
