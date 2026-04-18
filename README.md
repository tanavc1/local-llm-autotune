# autotune — Local LLM Inference Optimizer

[![PyPI](https://img.shields.io/pypi/v/llm-autotune)](https://pypi.org/project/llm-autotune/)
[![Python](https://img.shields.io/pypi/pyversions/llm-autotune)](https://pypi.org/project/llm-autotune/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/tanavc1/local-llm-autotune/blob/main/LICENSE)

**39% faster time-to-first-word. 67% less KV cache. Zero config changes.**

A drop-in middleware layer that makes your local LLMs faster and lighter on your own hardware — without touching your existing code or workflows.

```bash
pip install llm-autotune
autotune chat --model qwen3:8b   # that's it
```

Works with **Ollama**, **LM Studio**, and **MLX** (Apple Silicon native) out of the box.

---

## Benchmark Results

> Timings come from Ollama's own internal Go nanosecond timers — not Python estimates, not wall-clock guesses. Measured on Apple M2, 16 GB, macOS — hardware typical of anyone running local LLMs. These are observed means from 3 runs per condition; treat them as directional, not statistically proven.

### It works for agents too

Running LLMs inside an agent loop — where tool calls, observations, and reasoning steps accumulate across turns — is where raw Ollama's fixed `num_ctx=4096` setting causes the most pain. When context grows past the KV window, Ollama reloads the model. Latency spikes. Swap starts. The task stalls.

autotune solves this differently: before the agent loop starts, it computes a single `session_num_ctx` sized for the full task's context ceiling, then holds it constant for every turn. Ollama never sees a `num_ctx` change, so it never reloads. Combined with `num_keep` prefix caching — which pins the system prompt in KV so it's never re-evaluated after turn 1 — autotune keeps agent sessions stable as context accumulates.

**Measured on `llama3.2:3b`, `code_debugger` agentic task (multi-turn, tool-calling):**

| Metric | Raw Ollama | autotune | Notes |
|--------|:----------:|:--------:|-------|
| Model reloads per session | 0–1 | **~0** | Fixed from 7–10 in broken v1 |
| Swap events | 1 of 3 trials | **0** | KV memory management prevents pressure |
| TTFT growth per turn | −101 ms/turn | **−435 ms/turn** | TTFT falls faster — prefix cache working |
| Tool call errors | 1 avg | **0** | Lower temperature (0.3) reduces format errors |
| Context tokens at session end | 3,043 | **1,946 (−36%)** | Context trimming keeps sessions lean |
| Initial TTFT (turn 1) | 529 ms | 953 ms | **Autotune is slower on turn 1** — see below |
| Peak RAM | 2.39 GB | 2.85 GB | **Higher** — larger upfront KV allocation |

**The honest trade-off:** autotune pre-allocates a larger KV window at session start (sized to the task ceiling, not just the current message). Turn 1 is slower and uses more RAM. From turn 2 onward, prefix caching pays that cost back — TTFT per turn falls as the session grows, where raw Ollama's TTFT grows. For sessions of 3+ turns, the prefix cache savings accumulate. For 1–2 turn sessions, raw Ollama may be faster.

> Full methodology, raw data, and honest analysis of where autotune doesn't help: [AGENT_BENCHMARK.md](AGENT_BENCHMARK.md)

---

### What autotune actually improves

| KPI | llama3.2:3b | gemma4:e2b | qwen3:8b | Average |
|-----|:-----------:|:----------:|:--------:|:-------:|
| **Time to first word (TTFT)** | −35% | −29% | **−53%** | **−39%** |
| **KV prefill time** | −66% | −64% | **−72%** | **−67%** |
| **KV cache size** | −66% | **−69%** | −66% | **−67%** |
| **Peak RAM (LLM process)** | −11% | −0% | −7% | **−6%** |
| **Generation speed (tok/s)** | −2% | +0.2% | +2.4% | **+0.3%** |
| **End-to-end response time** | +0.5% | −0.9% | **−3.3%** | **−1.2%** |

> **KV buffer slots freed:** 40,215 (llama3.2:3b) · 42,348 (gemma4:e2b) · 40,215 (qwen3:8b) · **122,778 total** across all benchmark runs. This is the count of KV matrix slots Ollama never had to allocate or zero-initialise — computed as `(raw_num_ctx − tuned_num_ctx) × n_runs`. No prompt tokens were dropped; `prompt_eval_count` is identical in both conditions.

### What the numbers mean in plain English

**You wait 39% less for the first word.** On qwen3:8b — the most popular 8B model — that's 53% faster TTFT. On a complex prompt with a long context, it's up to 89% faster. You feel this on every single message.

**KV cache shrinks 3×.** Raw Ollama always allocates a 4096-token KV buffer regardless of how short your prompt is. autotune computes the exact size each request needs. For a typical chat message, that's 448–576 MB → 143–200 MB freed before inference even starts.

**Generation speed is unchanged.** Token generation on Apple Silicon is Metal GPU-bound. No software layer above Metal changes how fast the GPU generates tokens — and we don't pretend otherwise. The +0.3% average is measurement noise.

**RAM savings are real but modest.** Model weights dominate process RSS. The KV cache is a smaller fraction of total RSS on large models, so peak RAM drops 6–11% — meaningful on a 16 GB machine already at swap limits, but not dramatic.

**Zero swap events, zero model reloads** across all 3 models in both conditions. The `keep_alive=-1` setting holds the model in unified memory throughout. Your Mac's swap was at 5.5/6.0 GB before the benchmark — autotune didn't push it further.

### Benchmark methodology

**Hardware:** Apple M2 · 16 GB unified memory · macOS  
**Models:** `llama3.2:3b` (2.0 GB) · `gemma4:e2b` (7.2 GB) · `qwen3:8b` (5.2 GB)  
**Profile:** `autotune/balanced`  
**Design:** 3 runs per condition per prompt · 5 prompt types · controlled warmup per condition  
**Timing source:** Ollama's internal `prompt_eval_duration` / `total_duration` / `load_duration` fields — nanosecond Go runtime timers, not Python clocks  
**KV cache estimates:** `2 × n_layers × n_kv_heads × head_dim × num_ctx × dtype_bytes` from model architecture via Ollama `/api/show`

**Prompt types tested:**

| Type | Why |
|------|-----|
| Short factual Q&A | Baseline TTFT — smallest prompt, most sensitive to KV init time |
| Code generation | Throughput-dominated — long output, tests generation speed |
| Long-context analysis | Large prompt — maximum KV savings from dynamic num_ctx |
| Multi-turn conversation | Accumulated context — tests prefix caching via num_keep |
| Sustained long output | Long generation — tests KV quant under memory pressure |

**Full data:** Raw JSON with every run, load_ms, prefill_ms, swap_delta, reload_detected, and KV estimates for all 3 models: [`llama3.2:3b`](https://github.com/tanavc1/local-llm-autotune/blob/main/proof_results_v2.json) · [`gemma4:e2b`](https://github.com/tanavc1/local-llm-autotune/blob/main/proof_results_gemma4.json) · [`qwen3:8b`](https://github.com/tanavc1/local-llm-autotune/blob/main/proof_results_qwen3.json)

### Memory growth over turns

autotune also tracks how RAM grows across a multi-turn conversation. The model processes 4 sequential turns where each reply is added to the next prompt.

| Model | Raw RAM/turn | Autotune RAM/turn |
|-------|:-----------:|:-----------------:|
| llama3.2:3b | **+0.091 GB/turn** | **−0.069 GB/turn** |
| gemma4:e2b | +0.001 GB/turn | −0.011 GB/turn |
| qwen3:8b | −0.010 GB/turn | −0.011 GB/turn |

On llama3.2:3b, raw Ollama's RSS grows with each turn because the fixed 4096-token KV buffer is being more heavily used. autotune's dynamic context sizing adjusts per-turn — so RAM actually decreases slightly as the model settles.

### All 11 KPIs tracked

| KPI | How measured |
|-----|-------------|
| TTFT | `load_duration + prompt_eval_duration` — Ollama's internal timer |
| Prefill time | `prompt_eval_duration` — KV fill phase only |
| Total response time | `total_duration` — wall time start to last token |
| Peak RAM (LLM process) | Ollama runner process RSS, sampled at 100ms intervals |
| KV cache size | Estimated from model architecture (`n_layers × n_kv_heads × head_dim × num_ctx × bytes`) |
| Total context size | `num_ctx` + actual `prompt_eval_count` + `eval_count` per run |
| Memory growth over turns | RSS per turn in 4-turn sequential conversation |
| Swap pressure | `psutil.swap_memory()` delta before/after each run |
| Model reload count | `load_duration > 400ms` — distinguishes cold loads from Metal KV init (~100ms baseline) |
| Context size per request | `num_ctx` per run: raw always 4096, autotune dynamically 1174–1562 |
| KV buffer slots freed | `(raw_num_ctx − tuned_num_ctx) × n_runs` — counts KV matrix slots Ollama never allocated or zeroed. No prompt tokens are dropped; `prompt_eval_count` is identical in both conditions. |

### What autotune does NOT improve (honest)

| Metric | Why |
|--------|-----|
| **Generation throughput (tok/s)** | Metal GPU-bound. No software layer changes this. We measured +0.3% — that's noise. |
| **RAM (dramatically)** | Model weights dominate RSS. KV savings are real but ~6% total because weights are large. |
| **Quality** | autotune never truncates prompt tokens. `prompt_eval_count` is identical in both conditions — same actual content, smaller KV buffer. |

### Run the proof suite yourself

```bash
# Run on all three default models (sequentially — won't nuke your computer):
python scripts/proof_suite.py

# Run on a single model:
python scripts/proof_suite.py --models llama3.2:3b

# Run on any model you have installed:
python scripts/proof_suite.py --models YOUR_MODEL --runs 3

# Re-render the cross-model report from saved JSON:
python scripts/proof_report.py proof_results_*.json
```

Anyone can run this on any hardware, any model. If you have only one model installed, pass `--models` with that model's name. Results are saved as JSON for your own analysis.

---

## What it does

autotune sits between your application and the local LLM backend. It automatically:

| Feature | What happens |
|---------|-------------|
| **Dynamic KV sizing** | Computes the exact `num_ctx` each request needs instead of allocating the profile max — typically 4–8× less KV cache memory |
| **KV prefix caching** | Pins system-prompt tokens in Ollama's KV cache via `num_keep` so they're never re-evaluated each turn |
| **Adaptive KV precision** | Downgrades KV cache from F16 → Q8 under memory pressure (80% → −10% ctx, 88% → −25% ctx + Q8, 93% → −50% ctx + Q8) |
| **Model keep-alive** | Sets `keep_alive=-1m` so the model stays loaded in unified memory between turns — eliminates reload latency |
| **Flash attention** | Enables `flash_attn=true` on every request — reduces peak KV activation memory during attention computation; zero quality impact |
| **Prefill batching** | Sets `num_batch=1024` (2× Ollama default) — reduces Metal kernel dispatches for long prompts; under critical RAM pressure drops to 256 |
| **Multi-tier context management** | Intelligently trims conversation history at token budget thresholds with no mid-sentence cuts |
| **Inference queue** | FIFO queue (default: 1 concurrent, 8 waiting) with HTTP 429 back-pressure — prevents parallel inference from thrashing memory |
| **Profile-based optimization** | `fast` / `balanced` / `quality` profiles tune temperature, context length, KV precision, and OS QoS class |
| **OpenAI-compatible API** | Drop-in replacement for `localhost:8765/v1` — works with any OpenAI SDK |
| **MLX backend (Apple Silicon)** | On M-series Macs, routes inference to MLX-LM — native Metal GPU kernels, unified memory |
| **Persistent conversation memory** | Saves every conversation to SQLite; automatically injects relevant past context at session start; searchable by topic |
| **Hardware telemetry** | Samples RAM/Swap/CPU every 250 ms, persists structured metrics to SQLite |

---

## Quickstart

### 1. Prerequisites

Install [Ollama](https://ollama.com) and pull at least one model:

```bash
ollama pull qwen3:8b           # 5.2 GB — best general model for 16 GB laptops
ollama pull gemma4             # ~5.8 GB — Google's newest model, multimodal, 128k context
ollama pull qwen2.5-coder:14b  # 9 GB — top coding model for 24+ GB RAM
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

**`autotune run`** — pre-flight memory analysis + optimized chat. Checks whether the model fits in RAM, picks the right profile, then opens a chat session:

```bash
autotune run qwen3:8b
```

**`autotune chat`** — skip the pre-flight and go straight to optimized chat (adaptive-RAM monitoring, KV-manager, and context-optimizer are always active):

```bash
autotune chat --model qwen3:8b                   # balanced (default)
autotune chat --model qwen3:8b --profile fast    # fastest responses
autotune chat --model qwen3:8b --profile quality # largest context
autotune chat --model qwen3:8b --no-swap         # guarantee no macOS swap
```

Set a system prompt:

```bash
autotune chat --model qwen3:8b --system "You are a concise coding assistant."
```

Resume a previous conversation (the ID is shown in the chat header):

```bash
autotune chat --model qwen3:8b --conv-id a3f92c1b
```

### 6. Check what's running

```bash
autotune ps
```

Shows every model currently loaded in memory — across both Ollama and the MLX backend — with RAM usage, context size, quantization, and time loaded.

---

## Model recommendations by hardware

autotune works with any Ollama-compatible model. Here are our current picks for each hardware tier:

| RAM | Recommended model | Size | Why |
|-----|------------------|------|-----|
| 8 GB | `qwen3:4b` | ~2.6 GB | Best 4B model available; hybrid thinking mode |
| 16 GB | `qwen3:8b` | ~5.2 GB | Near-frontier quality; best 8B as of 2026 |
| 16 GB | `gemma4` | ~5.8 GB | Google's newest; multimodal, 128k context |
| 24 GB | `qwen3:14b` | ~9.0 GB | Excellent reasoning; fits well with headroom |
| 32 GB | `qwen3:30b-a3b` | ~17 GB | MoE: flagship quality at 7B inference cost |
| 64 GB+ | `qwen3:32b` | ~20 GB | Top dense open model |
| Coding | `qwen2.5-coder:14b` | ~9.0 GB | Best open coding model for 24 GB machines |
| Reasoning | `deepseek-r1:14b` | ~9.0 GB | Chain-of-thought; strong math and logic |

Run `autotune ls` to see how each downloaded model scores against your specific hardware.

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
| `/recall` | Browse past conversations with dates and snippets; pick one to resume |
| `/recall search <query>` | Search past conversations by topic — finds semantically related sessions |
| `/pull <model>` | Pull a model via Ollama without leaving chat |
| `/delete` | Delete the current conversation from history |
| `/quit` | Exit (also Ctrl-C) |

---

## Apple Silicon (MLX acceleration)

On M-series Macs, install the MLX backend to use native Metal GPU kernels:

```bash
pip install -e ".[mlx]"           # install mlx-lm
autotune mlx pull qwen3:8b        # download MLX-quantized model from mlx-community
autotune chat --model qwen3:8b    # automatically routes to MLX on Apple Silicon
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
    model="qwen3:8b",
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
| `GET /api/running_models` | All models currently in memory (Ollama + MLX), with RAM, ctx, quant, age |
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

## Embedding autotune in your application

autotune is designed to be embedded directly in Python applications. Your code points at the autotune server; autotune handles everything that makes local LLMs unreliable — OOM crashes, slow TTFT, model eviction between requests, memory pressure under load. You ship a reliable local AI app without becoming an LLM infrastructure expert.

### Minimal integration

```python
import autotune
from openai import OpenAI

autotune.start()                          # starts server if not running; blocks until ready
client = OpenAI(**autotune.client_kwargs())  # {"base_url": "http://localhost:8765/v1", "api_key": "local"}

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

That's the entire integration. autotune manages the server lifecycle, model keep-alive, KV optimisation, and memory pressure automatically.

### `autotune.start()`

```python
autotune.start(
    host="localhost",   # interface to bind — default local-only
    port=8765,          # port — default 8765
    timeout=30.0,       # seconds to wait for server ready — raises TimeoutError if exceeded
    profile="balanced", # default profile: "fast" | "balanced" | "quality"
    use_mlx=False,      # False = Ollama only (~94 MB RAM, full tool calling)
                        # True  = MLX on Apple Silicon (~470 MB, higher throughput, no tool calls)
    log_level="warning" # "warning" keeps server output out of your app's stdout
)
```

**`start()` is safe to call on every app launch.** It checks `/health` first — if the server is already running (started externally or by a previous call) it returns immediately. If it starts the server, it owns the process and `stop()` will terminate it.

### Checking model readiness before the first request

Use the `/v1/models/{model_id}/status` endpoint to know whether a model is warm before you show your UI:

```python
import httpx

status = httpx.get("http://localhost:8765/v1/models/qwen3:8b/status").json()

# status["status"] is one of:
#   "ready"      — model is loaded in memory; first token will be fast
#   "available"  — model is on disk but not loaded; expect a cold-start delay (~3-10s)
#   "not_found"  — model is not installed locally

if status["status"] == "not_found":
    print("Pull the model first: ollama pull qwen3:8b")
elif status["status"] == "available":
    show_loading_spinner()   # model will load on first request
elif status["status"] == "ready":
    show_chat_ui()           # model is warm, no delay

# Also includes a memory fit assessment:
# status["fit"]["class"]        → "safe" | "marginal" | "swap_risk" | "oom"
# status["fit"]["ram_util_pct"] → estimated RAM usage at inference time
# status["fit"]["warning"]      → plain-English warning if tight, else null
```

### Handling errors

All errors from autotune return structured JSON — no bare string parsing required:

```python
import httpx

try:
    response = client.chat.completions.create(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "Hello"}],
    )
except Exception as e:
    error = e.response.json().get("detail", {})

    match error.get("type"):
        case "model_not_found":
            # model isn't installed — prompt the user to pull it
            print(f"Run: ollama pull {error['model']}")

        case "memory_pressure":
            # model hit an OOM — suggest a smaller model or profile
            print("Not enough RAM. Try a smaller model or --profile fast.")

        case "backend_error":
            # Ollama isn't running or crashed
            print(f"Backend error: {error['message']}")
            print(f"Suggestion: {error['suggestion']}")
```

Every error body has `type`, `message`, `model`, `suggestion`, and `status_url` (points at the model status endpoint for a live fit assessment).

### Other API methods

```python
autotune.is_running()           # True/False — is the server accepting requests?
autotune.stop()                 # terminate the server autotune.start() launched
autotune.client_kwargs()        # returns {"base_url": "...", "api_key": "local"}
```

### Memory footprint

| Mode | Server RAM | Tool calling | Throughput |
|------|-----------|--------------|------------|
| `autotune.start()` (default) | **~94 MB** | ✓ Full support | Ollama |
| `autotune.start(use_mlx=True)` | ~470 MB | ✗ Broken | MLX (10–40% faster on Apple Silicon) |

The default is Ollama-only. On Linux/Windows, `use_mlx` has no effect — MLX is Apple Silicon only.

---

## Using autotune with agentic frameworks

autotune's OpenAI-compatible server works as a drop-in local LLM provider for any agentic framework that accepts a custom base URL. Start the server first, then point your framework at it — autotune handles KV optimisation, memory management, and model routing transparently in the background.

```bash
autotune serve
# Server running at http://localhost:8765/v1
```

### OpenClaw

[OpenClaw](https://openclaw.ai) is an open-source self-hosted agent framework with human-in-the-loop gates and sandboxed tool execution. It uses LiteLLM internally, so any OpenAI-compatible endpoint works.

**Step 1 — Pull a model that supports tool calling:**

```bash
ollama pull hermes3          # NousResearch Hermes 3 — strong tool-use model
ollama pull qwen3:8b         # Qwen 3 8B — good all-rounder with tool support
```

**Step 2 — Add autotune as a provider in OpenClaw's config:**

```yaml
# openclaw/config.yaml (or your OpenClaw provider settings)
providers:
  - name: autotune-local
    api: openai-responses
    baseUrl: http://localhost:8765/v1
    apiKey: sk-local          # any non-empty string — autotune doesn't validate keys
    model: hermes3            # use the Ollama model name exactly
    supportsTools: true       # set to true for hermes3, qwen3:8b; false for llama3.2:3b
```

**Step 3 — Start chatting.** OpenClaw will route all LLM calls through autotune. Dynamic KV sizing, memory pressure management, and model keep-alive apply automatically.

> **Note on `supportsTools`:** Only models with native tool/function-calling support should have `supportsTools: true`. On Ollama: `hermes3`, `qwen3:8b`, `qwen3:14b`, `llama3.1:8b`, `qwen2.5-coder:14b` all support it. `llama3.2:3b`, `gemma4:e2b` do not.

---

### Hermes Agent

[Hermes Agent](https://github.com/nousresearch/hermes-agent) (by Nous Research) is an autonomous agent framework with persistent memory, a built-in learning loop, and 40+ built-in tools. It uses the OpenAI Chat Completions standard exclusively.

**Step 1 — Pull a capable model:**

```bash
ollama pull hermes3          # recommended — designed specifically for the Hermes Agent ecosystem
ollama pull qwen3:8b         # good alternative
```

**Step 2 — Configure Hermes Agent to use autotune:**

```yaml
# ~/.hermes/config.yaml  (or wherever Hermes looks for config on your system)
model:
  provider: custom
  base_url: http://localhost:8765/v1
  api_key: sk-local
  name: hermes3              # Ollama model name
```

Or via the Hermes CLI:

```bash
hermes config set model.base_url http://localhost:8765/v1
hermes config set model.api_key sk-local
hermes config set model.name hermes3
hermes config set model.provider custom
```

**Step 3 — Run Hermes normally.** All inference goes through autotune. Because Hermes Agent accumulates memory and context across sessions, the dynamic KV sizing autotune applies on each request is especially useful — context windows stay right-sized rather than always allocating the profile max.

---

### Tool calling and Apple Silicon note

If you are on **Apple Silicon with MLX models installed** (via `autotune mlx pull`), autotune routes requests to the MLX backend by default because it is faster. The MLX backend does not currently relay OpenAI-style tool call responses back to the client. If your agent framework depends on structured tool calls:

- Use Ollama models that do **not** have an MLX equivalent cached, or
- Uninstall the MLX version of the model you want to use for agents (`autotune mlx list` to see what's cached), so autotune falls back to Ollama for that model.

**Models confirmed working for tool calling via Ollama:** `hermes3`, `qwen3:8b`, `qwen3:14b`, `llama3.1:8b`, `qwen2.5-coder:14b`

---

## Profiles

| Profile | Context | Temperature | KV precision | System QoS | Use when |
|---------|--------:|:-----------:|:------------:|:----------:|---------|
| `fast` ⚡ | 2,048 | 0.1 | Q8 | USER_INTERACTIVE | Quick lookups, autocomplete |
| `balanced` ⚖️ | 8,192 | 0.7 | F16 | USER_INITIATED | General chat, coding |
| `quality` ✨ | 32,768 | 0.8 | F16 | USER_INITIATED | Long-form writing, analysis |

`autotune run` with `--profile auto` (the default) analyses model size vs. available RAM and picks the profile automatically.

---

## How dynamic KV sizing works

Ollama allocates the entire KV cache upfront before generating a single token. If `num_ctx=4096`, it allocates memory for 4,096 tokens even if your prompt is 50 tokens — and zeros/initialises that entire buffer before prefill begins. That's what you're waiting for when you see slow TTFT.

autotune computes the minimum `num_ctx` each request actually needs:

```
num_ctx = clamp(input_tokens + max_new_tokens + 256, 512, profile_max)
```

For a short conversation on the `balanced` profile (max 8,192):
- Input: ~22 tokens → `num_ctx` = 22 + 1,024 + 256 = **1,302**
- Savings on `qwen3:8b`: 4,096 → 1,302 tokens = **~224 MB of KV cache never allocated**

`num_ctx` grows naturally as the conversation grows since the full history is included in every request. autotune never truncates or drops prompt tokens — the same content is always sent, just into a correctly-sized buffer.

---

## Telemetry and benchmarks

### View past runs

```bash
autotune telemetry               # last 20 inference runs
autotune telemetry --events      # notable events: swap spikes, OOMs, slow tokens
autotune telemetry --model qwen3:8b   # filter to a specific model
```

### Anonymous telemetry (opt-in)

autotune asks once whether you'd like to share anonymous performance data. You can change your answer at any time:

```bash
autotune telemetry --status    # check current opt-in status
autotune telemetry --enable    # opt in to anonymous telemetry
autotune telemetry --disable   # opt out — no further data is sent
```

**What is collected (only if opted in):**
- Hardware class: CPU architecture, RAM size, GPU backend — no hostnames, usernames, serial numbers, or IP addresses
- Model performance: tokens/sec, TTFT, context size, quantization label
- Session events: server start/stop, OOM events

Data goes to a private Supabase database. It is never sold or shared. The source for all collection logic is in `autotune/telemetry/`.

### Run the proof suite

```bash
# Default: all three benchmark models, 3 runs per condition:
python scripts/proof_suite.py

# Single model, more runs:
python scripts/proof_suite.py --models qwen3:8b --runs 5

# Save results for later analysis:
python scripts/proof_suite.py --output my_results.json

# Combine and render multiple result files:
python scripts/proof_report.py proof_results_*.json
```

Measures all 11 KPIs — TTFT, prefill time, total response time, peak RAM, KV cache size, context size per request, memory growth over turns, swap pressure, model reload count, KV buffer slots freed — across 5 prompt types (factual, code, long-context analysis, multi-turn, sustained generation).

### Where data is stored

All runs persist to SQLite automatically:
- **macOS:** `~/Library/Application Support/autotune/autotune.db`
- **Linux:** `~/.local/share/autotune/autotune.db`

---

## Conversation memory and recall

autotune records every conversation turn to a local SQLite database (`~/.autotune/recall.db`) using both full-text search and vector similarity. Memory is always-on — no flags required.

### What it does

- **Automatic context injection** — at the start of each chat session, autotune searches past conversations for topics similar to your current model and system prompt. Relevant facts are injected as a silent system message (only shown when `--verbose` is set). The 0.38 cosine similarity threshold filters out irrelevant memories.
- **Session linking** — conversations are stored with a unique ID shown in the chat header. Use `--conv-id <id>` to resume an exact past session.
- **In-chat recall** — use `/recall` to browse recent sessions with dates, model names, and first-turn snippets. Use `/recall search <topic>` for semantic search across all past conversations. Both commands offer a numbered prompt to resume the selected session with full context restored.
- **Model change detection** — the background watcher polls Ollama every 30 s; if a model unloads unexpectedly (crash, OOM), the chat interface notifies you immediately.

### Storage

| Path | Contents |
|------|----------|
| `~/.autotune/recall.db` | FTS5 + float32 vectors; conversation turns, extracted facts |
| `~/Library/Application Support/autotune/autotune.db` | Hardware telemetry, run observations (macOS) |
| `~/.local/share/autotune/autotune.db` | Same (Linux) |

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
│   │                      #   + flash_attn + num_batch + NoSwapGuard integration
│   └── __init__.py        #   Public API: TTFTOptimizer, KEEP_ALIVE_FOREVER
│
├── api/                   Inference pipeline
│   ├── profiles.py        #   fast / balanced / quality profile definitions
│   ├── server.py          #   FastAPI server — OpenAI-compatible /v1 + FIFO queue
│   ├── chat.py            #   Terminal chat REPL (adaptive-RAM + KV-manager + ctx-optimizer)
│   │                      #   /recall + /recall search, live tok/s TTFT stats, model watcher
│   ├── running_models.py  #   Cross-backend model visibility (Ollama + MLX state file)
│   ├── conversation.py    #   SQLite-backed persistent conversation state
│   ├── ctx_utils.py       #   Token estimation + compute_num_ctx (used by ttft/)
│   ├── kv_manager.py      #   KV options builder: flash_attn, num_batch, pressure tiers
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
├── recall/                ← Conversation memory system
│   ├── store.py           #   SQLite WAL-mode: FTS5 full-text + float32 cosine vectors
│   ├── manager.py         #   save_conversation / get_context_for / list_conversations
│   └── extractor.py       #   Chunk extraction + conversation value scoring
│
├── bench/                 Benchmarking framework
│   └── runner.py          #   run_raw_ollama / run_bench_ollama_only / BenchResult
│
├── db/                    Persistence
│   └── store.py           #   SQLite: models, hardware, run_observations, telemetry_events
│
├── hardware/              Hardware detection
│   ├── profiler.py        #   CPU/GPU/RAM detection (psutil + py-cpuinfo)
│   └── ram_advisor.py     #   Real-time RAM pressure advice and swap risk scoring
│
├── memory/                Memory estimation + no-swap guarantee
│   ├── estimator.py       #   Model weights + KV cache + runtime overhead
│   └── noswap.py          #   NoSwapGuard: adjusts num_ctx/KV to guarantee no swap
│
├── models/                Model registry
│   └── registry.py        #   OSS models with real MMLU/HumanEval/GSM8K scores
│
├── config/                Recommendation engine
│   └── generator.py       #   Multi-objective scoring: stability × speed × quality × context
│
└── cli.py                 Entry point (Click)
```

---

## License

MIT
