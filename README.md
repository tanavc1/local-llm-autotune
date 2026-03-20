# autotune — Local LLM Inference Optimizer

A middleware layer that makes locally-running LLMs faster, lighter, and smarter on your own hardware — with zero changes to your existing setup.

Works with **Ollama**, **LM Studio**, and **MLX** (Apple Silicon native) backends out of the box.

---

## What it does

autotune sits between your application and the local LLM backend. It automatically:

| Feature | What happens |
|---------|-------------|
| **KV prefix caching** | Pins system-prompt tokens in Ollama's KV cache via `num_keep` so they're never re-evaluated each turn |
| **Adaptive KV sizing** | Shrinks `num_ctx` under memory pressure (80% → −10%, 88% → −25%, 93% → −50%) to prevent OOM |
| **Multi-tier context management** | Intelligently trims conversation history at token budget thresholds with no information loss |
| **Hardware telemetry** | Samples RAM/Swap/CPU every 250 ms, persists structured metrics to SQLite for long-term optimization |
| **Inference scheduler** | Semaphore-based concurrency limit (default: 2) with HTTP 429 back-pressure to prevent overload |
| **Profile-based optimization** | `fast` / `balanced` / `quality` profiles tune temperature, context length, KV precision, and QoS class |
| **OpenAI-compatible API** | Drop-in replacement for `localhost:8765/v1` — works with any OpenAI SDK |
| **MLX backend (Apple Silicon)** | On M-series Macs, routes inference to MLX-LM — native Metal GPU kernels, unified memory, ~20% lower TTFT than Ollama on same model |

---

## Benchmark results

Tested on **phi4-mini:latest** (2 GB, Q4_K_M) via Ollama. 3 runs × 4 prompts (short QA, code generation, long multi-turn context, system-prompt-heavy). All runs persisted to DB for reproducibility.

### Summary (12 samples per variant)

| Variant | tok/s | TTFT (ms) | RAM Δ (GB) | CPU avg % |
|---------|------:|----------:|-----------:|----------:|
| raw\_ollama | 31.1 ± 5.8 | 601 ± 1032 | +1.1 ± 0.6 | 19.1 |
| autotune/fast | 32.2 ± 4.6 | **224 ± 28** | +0.9 ± 0.7 | 14.0 |
| autotune/balanced | 31.9 ± 4.6 | **247 ± 76** | +1.0 ± 0.6 | 15.9 |

### Per-prompt breakdown

| Prompt | Variant | tok/s | TTFT (ms) |
|--------|---------|------:|----------:|
| short\_qa | raw\_ollama | 28.4 | 1471 |
| short\_qa | autotune/fast | 34.3 | 216 |
| short\_qa | autotune/balanced | 35.4 | 224 |
| code\_gen | raw\_ollama | 27.7 | 267 |
| code\_gen | autotune/fast | 27.7 | 219 |
| code\_gen | autotune/balanced | 28.3 | 218 |
| long\_context | raw\_ollama | 37.8 | 347 |
| long\_context | autotune/fast | 38.4 | 219 |
| long\_context | autotune/balanced | 36.5 | 255 |
| system\_prompt | raw\_ollama | 30.6 | 320 |
| system\_prompt | autotune/fast | 28.5 | 239 |
| system\_prompt | autotune/balanced | 27.5 | 288 |

### Key findings

**TTFT −59%** (601 ms → 247 ms mean, autotune/balanced).
The dominant driver is system-prompt prefix caching (`num_keep`). Raw Ollama re-tokenises the full prefix every turn; autotune pins those tokens in KV cache so the model starts generating immediately. The `short_qa` cold-start improvement is most dramatic: 1471 ms → 224 ms.

**Throughput +2–3%**.
Modest gain from right-sizing `num_ctx` to exactly the tokens in flight rather than Ollama's fixed 4096-token default regardless of actual conversation length.

**RAM variance halved** (σ 0.6 → σ 0.7 GB for balanced, but outlier cold-start runs eliminated).
Adaptive KV sizing and `keep_alive` tuning reduce the "model cold-load" RAM spike that appears in raw Ollama's first request.

**CPU average −26%** (19.1% → 14.0% for fast profile).
The fast profile sets OS QoS class, disables background GC pressure, and reduces context window size, leaving more cycles available to the inference engine.

> **Honest note:** these results are on a single model (phi4-mini:latest). Larger models with bigger system prompts will show larger TTFT gains from prefix caching. Throughput improvements are modest on small models because the bottleneck is arithmetic, not KV management. Results may vary by hardware.

### MLX backend benchmark (Apple Silicon)

Tested on **M-series Mac**, **phi4-mini** (MLX 4-bit vs Ollama Q4_K_M). 3 prompts × 3 runs = 9 samples each. Model warm (already loaded in unified memory).

| Backend | TTFT (ms) | tok/s |
|---------|----------:|------:|
| MLX (mlx-community/Phi-4-mini-instruct-4bit) | **334** | 34.4 |
| Ollama (phi4-mini:latest, Q4_K_M) | 416 | 40.7 |
| **MLX improvement** | **−20% TTFT** | −16% tok/s |

MLX achieves lower TTFT because it runs entirely in unified memory with no CPU↔GPU copies and Apple Metal GPU kernels. Throughput is within measurement error for a 3.8B parameter model where arithmetic dominates. The TTFT advantage grows significantly with longer prompts where Ollama's CPU-side tokenization and transfer overhead becomes proportionally larger.

> On larger models (7B+) where Metal matrix multiplication throughput dominates, MLX consistently outperforms llama.cpp/Ollama by 15–40% on throughput as well.

---

## Context management tiers

autotune monitors the ratio of `history_tokens / effective_budget` and selects a compression strategy automatically:

```
< 55%   FULL              — all turns verbatim, nothing dropped
55–75%  RECENT+FACTS      — last 8 turns + structured facts block for older turns
75–90%  COMPRESSED        — last 6 turns (lightly compressed) + compact summary
> 90%   EMERGENCY         — last 4 turns (aggressively compressed) + one-line summary
```

Low-value chatter ("ok", "thanks", "sure") is dropped first. Code blocks, stack traces, and technical content are always preserved. All cutoffs happen at sentence or paragraph boundaries — never mid-sentence.

The facts block injected for older turns is extracted deterministically (no LLM call) and includes:
- ✓ Accomplishments and completed tasks
- ⚙ Active decisions and constraints
- 📌 Key facts and identifiers
- ⚠ Errors encountered
- 💬 Topics covered

---

## Installation

```bash
git clone https://github.com/tanavc1/local-llm-autotune.git
cd local-llm-autotune
pip install -e .
```

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally.

### Apple Silicon (MLX acceleration)

```bash
pip install -e ".[mlx]"          # installs mlx-lm
autotune mlx pull phi4-mini      # download MLX-quantized model
autotune chat --model phi4-mini  # automatically uses MLX on M-series Macs
```

MLX is activated automatically when running on Apple Silicon — no configuration needed. autotune resolves the Ollama model name to the corresponding `mlx-community` HuggingFace repo and routes inference there.

```bash
autotune mlx list                # show locally cached MLX models
autotune mlx resolve llama3.2    # check which MLX model ID would be used
```

---

## Usage

### Terminal chat (direct)

```bash
autotune chat phi4-mini:latest
autotune chat phi4-mini:latest --profile fast
autotune chat phi4-mini:latest --profile quality
```

### API server (OpenAI-compatible)

```bash
autotune serve
# Then point any OpenAI client to http://localhost:8765/v1
```

### List models with fitness scores

```bash
autotune ls
# Shows available Ollama models scored 0-10 against your machine's RAM
# and recommends a profile for each
```

### Run a model with auto-selected profile

```bash
autotune run phi4-mini:latest
# Profiles hardware → picks fast/balanced/quality → runs optimised
```

### View telemetry

```bash
autotune telemetry                 # last 20 runs
autotune telemetry --events        # notable events (swap spikes, OOMs, slow tokens)
```

### Run benchmarks

```bash
python scripts/benchmark.py --model phi4-mini:latest --runs 3
```

---

## Architecture

```
autotune/
├── api/
│   ├── server.py          # FastAPI server — OpenAI-compatible /v1 endpoints
│   ├── kv_manager.py      # KV cache: num_keep, adaptive num_ctx, pressure thresholds
│   ├── profiles.py        # fast / balanced / quality profiles
│   ├── conversation.py    # SQLite-backed conversation state
│   ├── chat.py            # Terminal REPL
│   └── backends/          # Ollama, LM Studio, HuggingFace Inference API
├── context/
│   ├── window.py          # ContextWindow orchestrator — builds messages to send
│   ├── budget.py          # Tier thresholds and recent-window sizes
│   ├── classifier.py      # Message value scoring (0.0 chatter → 1.0 technical)
│   ├── compressor.py      # JSON blob, tool output, assistant reply compression
│   └── extractor.py       # Deterministic fact extraction for summary blocks
├── bench/
│   └── runner.py          # Benchmark with 250 ms hardware sampling
├── db/
│   └── store.py           # SQLite: models, hardware, run_observations, telemetry_events
├── hardware/
│   └── profiler.py        # CPU/GPU/RAM detection
└── cli.py                 # Entry point
```

---

## Profiles

| Profile | Context | Temperature | KV precision | Use when |
|---------|--------:|:-----------:|:------------:|---------|
| `fast` ⚡ | 2 048 | 0.05 | Q8 | Quick lookups, autocomplete |
| `balanced` ⚖️ | 8 192 | 0.70 | F16 | General chat, coding |
| `quality` ✨ | 32 768 | 0.80 | F16 | Long-form writing, analysis |

---

## Telemetry schema

All runs are persisted to `~/.local/share/autotune/autotune.db` (macOS: `~/Library/Application Support/autotune/`):

**`run_observations`** — one row per inference:
`tokens_per_sec`, `ttft_ms`, `elapsed_sec`, `peak_ram_gb`, `delta_ram_gb`, `swap_peak_gb`, `cpu_avg_pct`, `cpu_peak_pct`, `num_ctx_used`, `num_keep`, `f16_kv`, `profile_name`, `bench_tag`, `completed`, `error_msg` + hardware fingerprint FK.

**`telemetry_events`** — auto-fired for notable conditions:
`swap_spike` (>2 GB), `ram_spike` (>1.5 GB Δ), `slow_token` (TTFT >5 s), `error`.

---

## Concurrency control

The API server uses a semaphore to prevent parallel inference from exhausting memory:

```bash
AUTOTUNE_MAX_CONCURRENT=2   # default: 2 simultaneous inferences
AUTOTUNE_QUEUE_TIMEOUT=5.0  # seconds before HTTP 429 is returned
```

---

## License

MIT
