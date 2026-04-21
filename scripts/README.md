# scripts/

Standalone benchmark and analysis scripts. All require Ollama running locally
(`ollama serve`) and at least one model installed.

---

## `user_bench.py` — User Experience Benchmark ⭐ Start here

Measures what users actually feel, not internal engine metrics.

```bash
# Quick smoke test (2 scenarios, auto-detects installed models):
python scripts/user_bench.py --quick

# Full benchmark on a specific model:
python scripts/user_bench.py --model qwen3:8b

# Run in background, get desktop notification when done:
python scripts/user_bench.py --model qwen3:8b --background

# Benchmark every installed model:
python scripts/user_bench.py --all-models --runs 3
```

**What it measures:**
- Swap events (goal: 0 — "computer never choked")
- RAM headroom for other apps during inference
- Time to first word (mean + worst-case)
- Response consistency (how predictable response times are)
- CPU spikes (proxy for fan noise / heat)
- Memory recovery time (how fast RAM returns after a call)
- Background impact score (0–100 composite)

---

## `proof_suite.py` — Technical 11-KPI Benchmark

The full statistical benchmark that produced the headline numbers in the README.

```bash
# All three reference models (takes ~20 min):
python scripts/proof_suite.py

# Single model, more runs:
python scripts/proof_suite.py --models qwen3:8b --runs 5

# Any model you have installed:
python scripts/proof_suite.py --models YOUR_MODEL --runs 3
```

**What it measures:** TTFT, prefill time, total response time, peak RAM, KV cache
size, context per request, memory growth over turns, swap pressure, model reload
count, KV buffer slots freed. Wilcoxon signed-rank + Cohen's d effect size.

---

## `agent_bench.py` — Agentic Task Benchmark

Multi-turn tool-calling agent benchmark. Tests whether context grows unboundedly
and whether the model reloads mid-session.

```bash
python scripts/agent_bench.py --model qwen3:8b --trials 3
```

---

## `stress_test.py` — Memory Pressure Stress Test

Six-phase stress test: warmup, baseline, autotune, sustained load, pressure, cold-start.

```bash
python scripts/stress_test.py --model qwen3:8b
```

---

## `benchmark.py` — Quick Comparison

Fast 4-prompt × 3-run benchmark for quick iteration.

```bash
python scripts/benchmark.py --model qwen3:8b
```

---

## `proof_report.py` — Cross-Model Report

Renders a cross-model summary from saved proof_suite JSON files.

```bash
python scripts/proof_report.py proof_results_*.json
```

---

## Output files

All scripts save results as JSON in the working directory:

| File | Script | Contents |
|------|--------|---------|
| `user_bench_<model>.json` | user_bench.py | User KPIs, head-to-head comparison |
| `proof_results_<model>.json` | proof_suite.py | 11 KPIs × all runs |
| `agent_bench_results.json` | agent_bench.py | Agent turn metrics |
| `stress_results.json` | stress_test.py | Stress phases + memory pressure |
| `benchmark_results.json` | benchmark.py | Quick comparison results |
