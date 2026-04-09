#!/usr/bin/env python3
"""
autotune Comprehensive Stress Test
===================================
Phases:
  1. Warmup          — discarded; just ensures model is loaded & GPU hot
  2. Baseline suite  — raw Ollama defaults, 10 diverse prompts × 2 runs
  3. Autotune suite  — autotune/balanced same prompts × 2 runs
  4. Sustained load  — 6 back-to-back calls, no pause (raw then autotune)
  5. Pressure test   — large-context prompts that push RAM limits
  6. Cold-start      — model unloaded between calls (keep_alive benefit)

Metrics per call:
  TTFT (ms), throughput (tok/s), elapsed (s), RAM before/peak/after/delta,
  swap peak, CPU avg/peak, num_ctx, KV precision, pressure level at call time

Models: phi4-mini:latest (primary), llama3.2:3b (secondary)
  - qwen2.5-coder:14b used in pressure phase if memory allows

Output: stress_results.json + terminal report
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune.bench.runner import (
    BenchResult,
    run_bench_ollama_only,
    run_raw_ollama,
    save_result,
)
from autotune.api.kv_manager import memory_pressure_snapshot

# ── Config ─────────────────────────────────────────────────────────────────

PRIMARY_MODEL   = "phi4-mini:latest"
SECONDARY_MODEL = "llama3.2:3b"
HEAVY_MODEL     = "qwen2.5-coder:14b"

OUTPUT_FILE = "stress_results.json"

LONG_SYSTEM = (
    "You are a highly knowledgeable expert assistant with deep expertise in "
    "computer science, mathematics, physics, and software engineering. "
    "Always structure your answers clearly. Cite reasoning step by step. "
    "When writing code, use best practices including type hints, docstrings, "
    "and error handling. Prefer correctness over brevity."
)

LONG_CODE_CONTEXT = """
You are a senior Python engineer reviewing the following code for bugs,
performance issues, and style. Respond with:
1. A summary of what the code does
2. All bugs found (with line references)
3. Performance suggestions
4. Refactored version

```python
import threading
import time
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self):
        self.cache = {}

    def __len__(self):
        return len(self.cache)


class ThreadSafeLRU(LRUCache):
    def __init__(self, capacity, ttl_seconds=None):
        super().__init__(capacity)
        self.lock = threading.Lock()
        self.ttl = ttl_seconds
        self.timestamps = {}

    def get(self, key):
        with self.lock:
            if self.ttl and key in self.timestamps:
                if time.time() - self.timestamps[key] > self.ttl:
                    del self.cache[key]
                    del self.timestamps[key]
                    return -1
            return super().get(key)

    def put(self, key, value):
        with self.lock:
            super().put(key, value)
            if self.ttl:
                self.timestamps[key] = time.time()

    def cleanup_expired(self):
        now = time.time()
        expired = [k for k, t in self.timestamps.items() if now - t > self.ttl]
        for k in expired:
            del self.cache[k]
            del self.timestamps[k]
```
"""  # noqa

# ── Test Prompts ────────────────────────────────────────────────────────────

PROMPTS = [
    {
        "name": "simple_factual",
        "category": "short",
        "messages": [{"role": "user", "content": "What is the speed of light in m/s?"}],
    },
    {
        "name": "code_fibonacci",
        "category": "code",
        "messages": [
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content":
             "Write an efficient Python function for Fibonacci numbers using "
             "memoization. Include type hints and a brief benchmark."},
        ],
    },
    {
        "name": "reasoning_chain",
        "category": "reasoning",
        "messages": [
            {"role": "user", "content":
             "A train leaves city A at 60 mph heading toward city B (300 miles away). "
             "Another train leaves city B at 90 mph heading toward city A at the same time. "
             "A fly starts at city A traveling at 150 mph, bouncing between the two trains "
             "until they collide. How far does the fly travel? Show all work."},
        ],
    },
    {
        "name": "long_system_code",
        "category": "code_with_ctx",
        "messages": [
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content":
             "Write a Python async HTTP client with connection pooling, "
             "exponential backoff retry, timeout handling, and proper context "
             "manager support. Include error types and comprehensive docstrings."},
        ],
    },
    {
        "name": "code_review",
        "category": "analysis",
        "messages": [
            {"role": "user", "content": LONG_CODE_CONTEXT},
        ],
    },
    {
        "name": "explain_transformer",
        "category": "long_explanation",
        "messages": [
            {"role": "system", "content": "You are an expert AI researcher. Be thorough."},
            {"role": "user", "content":
             "Explain in detail how the transformer architecture works: "
             "attention mechanisms, positional encoding, feed-forward layers, "
             "layer normalization, and how training via backpropagation works "
             "through the attention matrix."},
        ],
    },
    {
        "name": "multi_turn_followup",
        "category": "multi_turn",
        "messages": [
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "What is gradient descent?"},
            {"role": "assistant", "content":
             "Gradient descent is an optimization algorithm that iteratively "
             "adjusts parameters to minimize a loss function by moving in the "
             "direction of the negative gradient."},
            {"role": "user", "content":
             "Great. Now explain Adam optimizer, how it differs, "
             "and when you'd choose one over the other with concrete examples."},
        ],
    },
    {
        "name": "math_proof",
        "category": "math",
        "messages": [
            {"role": "user", "content":
             "Prove that the square root of 2 is irrational using proof by contradiction. "
             "Then explain why the same proof technique applies to the square root of any "
             "prime number. Provide a generalized version of the proof."},
        ],
    },
    {
        "name": "system_design",
        "category": "long_output",
        "messages": [
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content":
             "Design a distributed rate limiter that works across multiple servers. "
             "Include: architecture diagram (ASCII), Redis Lua script implementation, "
             "failure modes, consistency guarantees, and performance characteristics. "
             "The system must handle 100k requests/second across 50 nodes."},
        ],
    },
    {
        "name": "creative_technical",
        "category": "creative",
        "messages": [
            {"role": "user", "content":
             "Write a detailed technical blog post explaining how Git's content-addressable "
             "storage works under the hood. Include the SHA-1 hashing, object types (blob, "
             "tree, commit, tag), pack files, and how delta compression works. "
             "Target audience: experienced developers who've used Git but never looked "
             "at its internals."},
        ],
    },
]

# Sustained load prompts (shorter, back-to-back stress)
SUSTAINED_PROMPTS = [
    {"name": "s1", "messages": [{"role": "user", "content": "Explain recursion in 3 sentences."}]},
    {"name": "s2", "messages": [{"role": "user", "content": "What is a hash table? Give a Python example."}]},
    {"name": "s3", "messages": [{"role": "user", "content": "Write a quicksort in Python with type hints."}]},
    {"name": "s4", "messages": [{"role": "user", "content": "What is the CAP theorem? Name 3 databases for each combination."}]},
    {"name": "s5", "messages": [{"role": "user", "content": "Explain TCP three-way handshake with a diagram (ASCII art)."}]},
    {"name": "s6", "messages": [{"role": "user", "content": "What is Big O notation? Give examples of O(1), O(log n), O(n), O(n²)."}]},
]

# Large context pressure prompt (pushes num_ctx hard)
BIG_CONTEXT_PROMPT = {
    "name": "large_context",
    "messages": [
        {"role": "system", "content": LONG_SYSTEM * 3},   # ~300 tokens system prompt
        {"role": "user", "content": (
            "The following is a long technical document about distributed systems theory. "
            "Please summarize the key points, identify any contradictions, and suggest "
            "three follow-up research questions:\n\n"
            + "Distributed systems face fundamental challenges rooted in the CAP theorem, "
            "which states that a distributed data store cannot simultaneously provide more "
            "than two of the following three guarantees: Consistency (every read receives "
            "the most recent write or an error), Availability (every request receives a "
            "response, without guarantee it contains the most recent data), and Partition "
            "tolerance (the system continues operating despite network partitions). " * 8
            + "\n\nFurthermore, the PACELC theorem extends CAP by addressing latency "
            "tradeoffs even when the network is not partitioned. In the absence of "
            "partition (P), one must choose between latency (L) and consistency (C). "
            "This leads to four PACELC categories: PA/EL (e.g., DynamoDB), PA/EC "
            "(e.g., Cassandra with tunable consistency), PC/EL (rare), PC/EC (e.g., "
            "Google Spanner, VoltDB). " * 6
            + "\n\nConsensus algorithms such as Paxos and Raft are fundamental to "
            "building fault-tolerant distributed systems. Raft was designed to be more "
            "understandable than Paxos while providing equivalent guarantees. It separates "
            "the consensus problem into leader election, log replication, and safety. "
            "The leader receives client requests and replicates log entries to follower "
            "nodes. An entry is committed once a majority of nodes have written it. " * 5
        )},
    ],
}


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class SingleRun:
    phase: str
    prompt_name: str
    mode: str           # "raw" or "autotune"
    model: str
    run_idx: int
    ttft_ms: float
    tps: float
    elapsed_sec: float
    cpu_avg_pct: float
    cpu_peak_pct: float
    ram_before_gb: float
    ram_peak_gb: float
    ram_after_gb: float
    delta_ram_gb: float
    swap_peak_gb: float
    num_ctx_used: int
    f16_kv: Optional[bool]
    pressure_level: str
    error: Optional[str]
    completion_tokens: int


@dataclass
class PhaseResult:
    name: str
    runs: list[SingleRun] = field(default_factory=list)

    def raw_runs(self)    -> list[SingleRun]: return [r for r in self.runs if r.mode == "raw"]
    def tune_runs(self)   -> list[SingleRun]: return [r for r in self.runs if r.mode == "autotune"]

    def _stat(self, vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "stdev": None, "min": None, "max": None}
        return {
            "mean":  round(statistics.mean(vals), 2),
            "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
            "min":   round(min(vals), 2),
            "max":   round(max(vals), 2),
        }

    def summary(self, mode: str) -> dict:
        runs = self.raw_runs() if mode == "raw" else self.tune_runs()
        ok   = [r for r in runs if not r.error]
        if not ok:
            return {}
        return {
            "n_runs":       len(runs),
            "errors":       len(runs) - len(ok),
            "ttft_ms":      self._stat([r.ttft_ms for r in ok]),
            "tps":          self._stat([r.tps for r in ok]),
            "elapsed_sec":  self._stat([r.elapsed_sec for r in ok]),
            "cpu_avg_pct":  self._stat([r.cpu_avg_pct for r in ok]),
            "cpu_peak_pct": self._stat([r.cpu_peak_pct for r in ok]),
            "delta_ram_gb": self._stat([r.delta_ram_gb for r in ok]),
            "peak_ram_gb":  self._stat([r.ram_peak_gb for r in ok]),
            "swap_peak_gb": self._stat([r.swap_peak_gb for r in ok]),
        }


# ── Helpers ─────────────────────────────────────────────────────────────────

def _banner(text: str, width: int = 72) -> None:
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}")


def _sec(text: str) -> None:
    print(f"\n  ── {text} ──")


def _hw_snapshot() -> dict:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "ram_used_gb":   round(vm.used / 1024**3, 2),
        "ram_avail_gb":  round(vm.available / 1024**3, 2),
        "ram_pct":       vm.percent,
        "swap_used_gb":  round(sw.used / 1024**3, 2),
        "cpu_pct":       psutil.cpu_percent(interval=0.1),
    }


async def _run_one(
    phase: str,
    prompt: dict,
    mode: str,
    model: str,
    run_idx: int,
    profile: str = "balanced",
    timeout_sec: int = 240,
) -> SingleRun:
    """Execute one inference call, return structured SingleRun."""
    psnap = memory_pressure_snapshot()

    try:
        if mode == "raw":
            result: BenchResult = await asyncio.wait_for(
                run_raw_ollama(model, prompt["messages"], tag=f"stress_raw_{prompt['name']}"),
                timeout=timeout_sec,
            )
        else:
            result = await asyncio.wait_for(
                run_bench_ollama_only(
                    model,
                    prompt["messages"],
                    profile_name=profile,
                    tag=f"stress_tune_{prompt['name']}",
                    apply_hw_tuning=True,
                ),
                timeout=timeout_sec,
            )

        save_result(result)

        sr = SingleRun(
            phase=phase,
            prompt_name=prompt["name"],
            mode=mode,
            model=model,
            run_idx=run_idx,
            ttft_ms=result.ttft_ms,
            tps=result.tokens_per_sec,
            elapsed_sec=result.elapsed_sec,
            cpu_avg_pct=result.cpu_avg_pct,
            cpu_peak_pct=result.cpu_peak_pct,
            ram_before_gb=result.ram_before_gb,
            ram_peak_gb=result.ram_peak_gb,
            ram_after_gb=result.ram_after_gb,
            delta_ram_gb=result.delta_ram_gb,
            swap_peak_gb=result.swap_peak_gb,
            num_ctx_used=result.num_ctx_used or 4096,
            f16_kv=result.f16_kv_used,
            pressure_level=psnap["pressure_level"],
            error=result.error,
            completion_tokens=result.completion_tokens,
        )
        return sr

    except asyncio.TimeoutError:
        return SingleRun(
            phase=phase, prompt_name=prompt["name"], mode=mode, model=model,
            run_idx=run_idx, ttft_ms=0, tps=0, elapsed_sec=timeout_sec,
            cpu_avg_pct=0, cpu_peak_pct=0, ram_before_gb=0, ram_peak_gb=0,
            ram_after_gb=0, delta_ram_gb=0, swap_peak_gb=0,
            num_ctx_used=4096, f16_kv=None, pressure_level=psnap["pressure_level"],
            error=f"TIMEOUT after {timeout_sec}s", completion_tokens=0,
        )
    except Exception as exc:
        return SingleRun(
            phase=phase, prompt_name=prompt["name"], mode=mode, model=model,
            run_idx=run_idx, ttft_ms=0, tps=0, elapsed_sec=0,
            cpu_avg_pct=0, cpu_peak_pct=0, ram_before_gb=0, ram_peak_gb=0,
            ram_after_gb=0, delta_ram_gb=0, swap_peak_gb=0,
            num_ctx_used=4096, f16_kv=None, pressure_level=psnap["pressure_level"],
            error=str(exc), completion_tokens=0,
        )


def _fmt_result(sr: SingleRun) -> str:
    if sr.error:
        return f"❌ ERROR: {sr.error[:60]}"
    return (
        f"TTFT {sr.ttft_ms:>7.0f}ms  "
        f"TPS {sr.tps:>6.1f}  "
        f"CPU {sr.cpu_avg_pct:>4.1f}%  "
        f"RAM Δ{sr.delta_ram_gb:>+.2f}GB  "
        f"ctx {sr.num_ctx_used:>5d}  "
        f"swap {sr.swap_peak_gb:.2f}GB"
    )


async def warmup(model: str) -> None:
    """One throwaway call to get model loaded and GPU hot."""
    _sec(f"Warmup ({model}) — discarded")
    print("    Sending warmup prompt…", end=" ", flush=True)
    try:
        result = await asyncio.wait_for(
            run_raw_ollama(
                model,
                [{"role": "user", "content": "Say 'ready' and nothing else."}],
                tag="warmup",
            ),
            timeout=120,
        )
        status = "ok" if not result.error else f"error: {result.error[:40]}"
        print(f"done ({result.elapsed_sec:.1f}s) [{status}]")
    except asyncio.TimeoutError:
        print("timed out (model may not be available)")


# ── Phase runners ────────────────────────────────────────────────────────────

async def phase_main_suite(
    model: str,
    n_runs: int = 2,
) -> PhaseResult:
    """
    Phase 2+3: full 10-prompt suite, raw then autotune.
    Each prompt runs n_runs times for each mode.
    """
    phase = PhaseResult(name="main_suite")

    for mode_label, mode in [("RAW (Ollama defaults)", "raw"), ("AUTOTUNE/balanced", "autotune")]:
        _sec(f"Main suite — {mode_label} — {model}")
        for prompt in PROMPTS:
            for run_i in range(n_runs):
                hw = _hw_snapshot()
                print(
                    f"    [{mode_label:<22}] [{prompt['name']:<22}] run {run_i+1}/{n_runs}"
                    f"  RAM:{hw['ram_pct']:.0f}% avail:{hw['ram_avail_gb']:.1f}GB … ",
                    end="", flush=True,
                )
                sr = await _run_one("main_suite", prompt, mode, model, run_i)
                phase.runs.append(sr)
                print(_fmt_result(sr))
                if run_i < n_runs - 1:
                    await asyncio.sleep(2.0)  # brief settle between runs

            # Extra settle between prompts
            await asyncio.sleep(1.5)

        # Longer cooldown between modes
        _sec("Cooling down 10s between modes…")
        await asyncio.sleep(10.0)

    return phase


async def phase_sustained_load(
    model: str,
) -> PhaseResult:
    """
    Phase 4: 6 back-to-back calls with NO pause — simulates server burst load.
    Raw then autotune. Measures how each handles heat buildup.
    """
    phase = PhaseResult(name="sustained_load")

    for mode_label, mode in [("RAW", "raw"), ("AUTOTUNE", "autotune")]:
        _sec(f"Sustained load — {mode_label} — {model} (no pause between calls)")
        t_batch_start = time.perf_counter()
        for i, prompt in enumerate(SUSTAINED_PROMPTS):
            hw = _hw_snapshot()
            print(
                f"    [{mode_label}] [{prompt['name']}] #{i+1}/6"
                f"  RAM:{hw['ram_pct']:.0f}% … ",
                end="", flush=True,
            )
            sr = await _run_one("sustained_load", prompt, mode, model, i)
            phase.runs.append(sr)
            print(_fmt_result(sr))
            # NO sleep — that's the point of sustained load

        batch_elapsed = time.perf_counter() - t_batch_start
        print(f"\n    Batch total: {batch_elapsed:.1f}s")

        _sec("Cooling down 15s between modes…")
        await asyncio.sleep(15.0)

    return phase


async def phase_pressure_test(
    model: str,
) -> PhaseResult:
    """
    Phase 5: Large-context prompt 3× (raw) then 3× (autotune).
    Tests how each handles high RAM pressure — autotune should adapt num_ctx.
    """
    phase = PhaseResult(name="pressure_test")
    n_runs = 3

    for mode_label, mode in [("RAW", "raw"), ("AUTOTUNE", "autotune")]:
        _sec(f"Pressure test — large context — {mode_label} — {model}")
        for i in range(n_runs):
            hw = _hw_snapshot()
            psnap = memory_pressure_snapshot()
            print(
                f"    [{mode_label}] large_ctx run {i+1}/{n_runs}"
                f"  RAM:{hw['ram_pct']:.0f}% ({psnap['pressure_level']})"
                f"  swap:{hw['swap_used_gb']:.2f}GB … ",
                end="", flush=True,
            )
            sr = await _run_one(
                "pressure_test", BIG_CONTEXT_PROMPT, mode, model, i, timeout_sec=180
            )
            phase.runs.append(sr)
            print(_fmt_result(sr))
            await asyncio.sleep(3.0)

        await asyncio.sleep(10.0)

    return phase


async def phase_cold_start(
    model: str,
) -> PhaseResult:
    """
    Phase 6: Tests keep_alive benefit.
    - Raw: let Ollama unload the model (keep_alive default = 5min, but we
      explicitly unload via API), then reload. Measures cold TTFT.
    - Autotune: keep_alive=-1 means model stays loaded. Measures warm TTFT.
    """
    phase = PhaseResult(name="cold_start")

    COLD_PROMPT = {
        "name": "cold_warmth_test",
        "messages": [{"role": "user", "content": "What is 17 × 23? Show work."}],
    }

    import httpx

    async def _unload_model(m: str) -> None:
        """Force unload the model from Ollama's KV cache."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": m, "keep_alive": "0"},
                )
        except Exception:
            pass
        await asyncio.sleep(3.0)

    # Raw cold-start (force unload between each call)
    _sec(f"Cold start test — RAW (unloading model between calls) — {model}")
    for i in range(3):
        print(f"    [RAW] Unloading {model}…", end=" ", flush=True)
        await _unload_model(model)
        print("unloaded.", end="  ", flush=True)
        hw = _hw_snapshot()
        print(f"RAM:{hw['ram_pct']:.0f}%  ", end="", flush=True)
        sr = await _run_one("cold_start", COLD_PROMPT, "raw", model, i)
        phase.runs.append(sr)
        print(_fmt_result(sr))

    await asyncio.sleep(5.0)

    # Autotune warm (keep_alive=-1) — first call is still cold, subsequent warm
    _sec(f"Warm start test — AUTOTUNE keep_alive=-1 — {model}")
    for i in range(3):
        hw = _hw_snapshot()
        print(
            f"    [AUTOTUNE] call {i+1}/3"
            f"  RAM:{hw['ram_pct']:.0f}%  ",
            end="", flush=True,
        )
        sr = await _run_one("cold_start", COLD_PROMPT, "autotune", model, i)
        phase.runs.append(sr)
        print(_fmt_result(sr))
        await asyncio.sleep(1.0)  # minimal pause — model stays loaded

    return phase


async def phase_heavy_model() -> PhaseResult:
    """
    Phase 7: Heavy model (qwen2.5-coder:14b) — only if RAM allows.
    Tests autotune's adaptive num_ctx at scale vs raw defaults.
    """
    phase = PhaseResult(name="heavy_model")
    vm = psutil.virtual_memory()
    available_gb = vm.available / 1024**3

    if available_gb < 12.0:
        _sec(f"Heavy model phase SKIPPED — only {available_gb:.1f}GB available (need 12GB)")
        return phase

    _sec(f"Heavy model: {HEAVY_MODEL}  available RAM: {available_gb:.1f}GB")

    HEAVY_PROMPTS = [
        {
            "name": "hm_code",
            "messages": [
                {"role": "system", "content": LONG_SYSTEM},
                {"role": "user", "content":
                 "Write a production-ready Python async REST API using FastAPI with: "
                 "JWT auth, SQLAlchemy async ORM, Pydantic v2 models, rate limiting, "
                 "and comprehensive error handling. Include all necessary models and routes."},
            ],
        },
        {
            "name": "hm_reasoning",
            "messages": [{"role": "user", "content":
                "Describe in detail the differences between BFS and DFS graph traversal. "
                "Implement both in Python with iterative (not recursive) versions, "
                "compare their space/time complexity, and give 5 practical use cases each."}],
        },
    ]

    for mode_label, mode in [("RAW", "raw"), ("AUTOTUNE", "autotune")]:
        _sec(f"Heavy model — {mode_label}")
        for prompt in HEAVY_PROMPTS:
            hw = _hw_snapshot()
            print(
                f"    [{mode_label}] [{prompt['name']}]"
                f"  RAM:{hw['ram_pct']:.0f}% avail:{hw['ram_avail_gb']:.1f}GB … ",
                end="", flush=True,
            )
            sr = await _run_one("heavy_model", prompt, mode, HEAVY_MODEL, 0, timeout_sec=300)
            phase.runs.append(sr)
            print(_fmt_result(sr))
            await asyncio.sleep(5.0)

        await asyncio.sleep(15.0)

    return phase


# ── Report generator ─────────────────────────────────────────────────────────

def _pct_delta(baseline: float, tuned: float, lower_is_better: bool = False) -> str:
    if baseline == 0 or math.isnan(baseline) or math.isnan(tuned):
        return "—"
    delta = (tuned - baseline) / abs(baseline) * 100
    if lower_is_better:
        # Positive delta = got worse, negative = improved
        sign = "▼" if delta < 0 else "▲"
        return f"{sign}{abs(delta):.1f}%"
    else:
        sign = "▲" if delta > 0 else "▼"
        return f"{sign}{abs(delta):.1f}%"


def _mean_or_nan(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else float("nan")


def generate_report(phases: list[PhaseResult], model: str, wall_time: float) -> dict:
    """Build the full results dict and print a human report."""

    # ── Aggregate all good runs across phases ────────────────────────────────
    all_raw  = [r for p in phases for r in p.raw_runs()  if not r.error]
    all_tune = [r for p in phases for r in p.tune_runs() if not r.error]

    def agg(runs: list[SingleRun]) -> dict:
        if not runs:
            return {}
        return {
            "n":            len(runs),
            "ttft_ms":      _mean_or_nan([r.ttft_ms for r in runs]),
            "tps":          _mean_or_nan([r.tps for r in runs]),
            "elapsed_sec":  _mean_or_nan([r.elapsed_sec for r in runs]),
            "cpu_avg_pct":  _mean_or_nan([r.cpu_avg_pct for r in runs]),
            "cpu_peak_pct": _mean_or_nan([r.cpu_peak_pct for r in runs]),
            "delta_ram_gb": _mean_or_nan([r.delta_ram_gb for r in runs]),
            "peak_ram_gb":  _mean_or_nan([r.ram_peak_gb for r in runs]),
            "swap_peak_gb": _mean_or_nan([r.swap_peak_gb for r in runs]),
        }

    raw_agg  = agg(all_raw)
    tune_agg = agg(all_tune)

    # Per-prompt win/loss
    prompt_names = list(dict.fromkeys(r.prompt_name for r in all_raw + all_tune))
    wins = {"ttft": 0, "tps": 0, "cpu": 0, "ram": 0, "swap": 0}
    losses = {"ttft": 0, "tps": 0, "cpu": 0, "ram": 0, "swap": 0}
    ties = {"ttft": 0, "tps": 0, "cpu": 0, "ram": 0, "swap": 0}

    prompt_rows = []
    for pname in prompt_names:
        pr  = [r for r in all_raw  if r.prompt_name == pname]
        pt  = [r for r in all_tune if r.prompt_name == pname]
        if not pr or not pt:
            continue
        raw_ttft  = _mean_or_nan([r.ttft_ms for r in pr])
        tune_ttft = _mean_or_nan([r.ttft_ms for r in pt])
        raw_tps   = _mean_or_nan([r.tps for r in pr])
        tune_tps  = _mean_or_nan([r.tps for r in pt])
        raw_cpu   = _mean_or_nan([r.cpu_avg_pct for r in pr])
        tune_cpu  = _mean_or_nan([r.cpu_avg_pct for r in pt])
        raw_ram   = _mean_or_nan([r.delta_ram_gb for r in pr])
        tune_ram  = _mean_or_nan([r.delta_ram_gb for r in pt])
        raw_swap  = _mean_or_nan([r.swap_peak_gb for r in pr])
        tune_swap = _mean_or_nan([r.swap_peak_gb for r in pt])

        def _win(raw_v: float, tune_v: float, lower: bool, key: str) -> None:
            if math.isnan(raw_v) or math.isnan(tune_v):
                return
            THRESH = 0.03  # 3% threshold for tie
            delta = (tune_v - raw_v) / max(abs(raw_v), 0.001)
            if lower:
                if delta < -THRESH: wins[key] += 1
                elif delta > THRESH: losses[key] += 1
                else: ties[key] += 1
            else:
                if delta > THRESH: wins[key] += 1
                elif delta < -THRESH: losses[key] += 1
                else: ties[key] += 1

        _win(raw_ttft, tune_ttft, lower=True, key="ttft")
        _win(raw_tps, tune_tps, lower=False, key="tps")
        _win(raw_cpu, tune_cpu, lower=True, key="cpu")
        _win(raw_ram, tune_ram, lower=True, key="ram")
        _win(raw_swap, tune_swap, lower=True, key="swap")

        prompt_rows.append({
            "prompt":         pname,
            "raw_ttft_ms":    round(raw_ttft, 1),
            "tune_ttft_ms":   round(tune_ttft, 1),
            "ttft_delta":     _pct_delta(raw_ttft, tune_ttft, lower_is_better=True),
            "raw_tps":        round(raw_tps, 1),
            "tune_tps":       round(tune_tps, 1),
            "tps_delta":      _pct_delta(raw_tps, tune_tps, lower_is_better=False),
            "raw_cpu_pct":    round(raw_cpu, 1),
            "tune_cpu_pct":   round(tune_cpu, 1),
            "cpu_delta":      _pct_delta(raw_cpu, tune_cpu, lower_is_better=True),
            "raw_ram_delta":  round(raw_ram, 3),
            "tune_ram_delta": round(tune_ram, 3),
        })

    # ── Print report ─────────────────────────────────────────────────────────

    _banner(f"AUTOTUNE STRESS TEST RESULTS  —  {model}  —  {wall_time/60:.1f} min")

    # System info
    vm = psutil.virtual_memory()
    print(f"\n  Hardware: {platform.processor() or platform.machine()}")
    print(f"  OS:       {platform.system()} {platform.mac_ver()[0] or platform.release()}")
    print(f"  RAM:      {vm.total/1024**3:.1f} GB total  |  {vm.used/1024**3:.1f} GB used now")
    print(f"  Runs:     {len(all_raw)} raw  |  {len(all_tune)} autotune  |  {wall_time:.0f}s total")

    # Aggregate summary table
    print(f"\n  {'Metric':<28}  {'RAW (Ollama defaults)':>22}  {'AUTOTUNE/balanced':>20}  {'Δ':>10}")
    print("  " + "─" * 85)

    def _row(label: str, raw_v: float, tune_v: float, unit: str, lower: bool) -> None:
        if math.isnan(raw_v) or math.isnan(tune_v):
            return
        delta_str = _pct_delta(raw_v, tune_v, lower_is_better=lower)
        good = (
            (tune_v < raw_v * 0.97) if lower else (tune_v > raw_v * 1.03)
        )
        mark = "✓" if good else ("✗" if ((tune_v > raw_v * 1.03) if lower else (tune_v < raw_v * 0.97)) else "≈")
        print(f"  {label:<28}  {raw_v:>18.2f}{unit}  {tune_v:>16.2f}{unit}  {delta_str:>8}  {mark}")

    if raw_agg and tune_agg:
        _row("TTFT (ms)              ▼lower", raw_agg["ttft_ms"],      tune_agg["ttft_ms"],      "ms",  True)
        _row("Throughput (tok/s)     ▲higher", raw_agg["tps"],          tune_agg["tps"],          "t/s", False)
        _row("Elapsed (s)            ▼lower", raw_agg["elapsed_sec"],  tune_agg["elapsed_sec"],  "s",   True)
        _row("CPU avg %              ▼lower", raw_agg["cpu_avg_pct"],  tune_agg["cpu_avg_pct"],  "%",   True)
        _row("CPU peak %             ▼lower", raw_agg["cpu_peak_pct"], tune_agg["cpu_peak_pct"], "%",   True)
        _row("RAM delta (GB)         ▼lower", raw_agg["delta_ram_gb"], tune_agg["delta_ram_gb"], "GB",  True)
        _row("Peak RAM used (GB)     ▼lower", raw_agg["peak_ram_gb"],  tune_agg["peak_ram_gb"],  "GB",  True)
        _row("Peak swap (GB)         ▼lower", raw_agg["swap_peak_gb"], tune_agg["swap_peak_gb"], "GB",  True)

    # Per-prompt table
    print(f"\n  {'Prompt':<26}  {'TTFT raw':>9}  {'TTFT tune':>9}  Δ TTFT    {'TPS raw':>7}  {'TPS tune':>8}  Δ TPS    {'CPU Δ':>8}")
    print("  " + "─" * 100)
    for row in prompt_rows:
        print(
            f"  {row['prompt']:<26}"
            f"  {row['raw_ttft_ms']:>9.0f}ms"
            f"  {row['tune_ttft_ms']:>9.0f}ms"
            f"  {row['ttft_delta']:>8}"
            f"  {row['raw_tps']:>7.1f}"
            f"  {row['tune_tps']:>8.1f}"
            f"  {row['tps_delta']:>6}"
            f"  {row['cpu_delta']:>8}"
        )

    # Win/loss matrix
    total_w = sum(wins.values())
    total_l = sum(losses.values())
    total_t = sum(ties.values())
    total   = total_w + total_l + total_t
    print(f"\n  Win/Loss Matrix (autotune vs raw, per metric per prompt):")
    print(f"  {'Metric':<12}  {'Wins':>5}  {'Losses':>6}  {'Ties':>5}  Win rate")
    print("  " + "─" * 48)
    for key, label in [("ttft","TTFT"),("tps","TPS"),("cpu","CPU"),("ram","RAM"),("swap","Swap")]:
        t = wins[key] + losses[key] + ties[key]
        wr = wins[key]/t*100 if t else 0
        print(f"  {label:<12}  {wins[key]:>5}  {losses[key]:>6}  {ties[key]:>5}  {wr:>6.0f}%")
    print(f"  {'TOTAL':<12}  {total_w:>5}  {total_l:>6}  {total_t:>5}  {total_w/total*100:.0f}%" if total else "")

    # Phase summaries
    print(f"\n  Phase-level summaries:")
    for p in phases:
        if not p.runs:
            continue
        raw_ok  = [r for r in p.raw_runs()  if not r.error]
        tune_ok = [r for r in p.tune_runs() if not r.error]
        if not raw_ok and not tune_ok:
            continue
        raw_ttft  = _mean_or_nan([r.ttft_ms for r in raw_ok])
        tune_ttft = _mean_or_nan([r.ttft_ms for r in tune_ok])
        raw_tps   = _mean_or_nan([r.tps for r in raw_ok])
        tune_tps  = _mean_or_nan([r.tps for r in tune_ok])
        print(
            f"    {p.name:<18}"
            f"  raw: TTFT {raw_ttft:>8.0f}ms TPS {raw_tps:>5.1f}"
            f"  tune: TTFT {tune_ttft:>8.0f}ms TPS {tune_tps:>5.1f}"
            f"  ΔTTFT {_pct_delta(raw_ttft, tune_ttft, lower_is_better=True)}"
        )

    # Verdict
    _banner("VERDICT")
    if not raw_agg or not tune_agg:
        print("  Insufficient data for verdict.")
    else:
        ttft_imp  = (raw_agg["ttft_ms"] - tune_agg["ttft_ms"]) / raw_agg["ttft_ms"] * 100
        tps_imp   = (tune_agg["tps"] - raw_agg["tps"]) / max(raw_agg["tps"], 0.01) * 100
        cpu_imp   = (raw_agg["cpu_avg_pct"] - tune_agg["cpu_avg_pct"]) / max(raw_agg["cpu_avg_pct"], 0.01) * 100
        ram_imp   = raw_agg["delta_ram_gb"] - tune_agg["delta_ram_gb"]
        swap_imp  = raw_agg["swap_peak_gb"] - tune_agg["swap_peak_gb"]

        print(f"\n  Overall autotune/balanced vs raw Ollama defaults:")
        print(f"    TTFT:         {ttft_imp:>+.1f}%  ({'faster' if ttft_imp>0 else 'slower'} first token)")
        print(f"    Throughput:   {tps_imp:>+.1f}%  ({'higher' if tps_imp>0 else 'lower'} tok/s)")
        print(f"    CPU average:  {cpu_imp:>+.1f}%  ({'less' if cpu_imp>0 else 'more'} CPU utilization)")
        print(f"    RAM delta:    {ram_imp:>+.3f} GB  ({'less' if ram_imp>0 else 'more'} memory growth per call)")
        print(f"    Swap peak:    {swap_imp:>+.3f} GB  ({'less' if swap_imp>0 else 'more'} swap pressure)")

        # Honest assessment
        print(f"\n  Autotune key actions applied:")
        print(f"    • Dynamic num_ctx — right-sized to prompt, not Ollama's fixed 4096 default")
        print(f"    • keep_alive=-1   — model never unloads between calls")
        print(f"    • GC disabled     — Python GC pauses eliminated during inference")
        print(f"    • QOS class       — USER_INITIATED priority on macOS scheduler")
        print(f"    • Prefix caching  — system prompt tokens pinned in KV via num_keep")
        print(f"    • Pressure guard  — auto-reduces num_ctx + downgrades KV to Q8 under RAM pressure")

        wins_total = total_w
        total_possible = total
        if total_possible > 0:
            wr = wins_total / total_possible * 100
            print(f"\n  Win rate across all metrics: {wins_total}/{total_possible} ({wr:.0f}%)")
            if wr >= 60:
                print(f"  → Autotune demonstrates clear benefit for LLM inference on this hardware.")
            elif wr >= 40:
                print(f"  → Autotune shows mixed results — benefits on some metrics, neutral on others.")
            else:
                print(f"  → Results inconclusive — Ollama's defaults may already be well-tuned for this model/hardware.")

        print(f"\n  Honest caveats:")
        print(f"    • Token throughput is GPU-bound on Metal — autotune impact is mostly in TTFT + memory")
        print(f"    • Benefits scale more clearly with larger models (7B+) and long sessions")
        print(f"    • Cold-start TTFT is always high — keep_alive=-1 amortizes this over a session")
        print(f"    • num_ctx reduction may limit response length for very long conversations")

    print(f"\n{'═' * 72}\n")

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "wall_time_sec": round(wall_time, 1),
        "aggregate": {"raw": raw_agg, "autotune": tune_agg},
        "wins": wins, "losses": losses, "ties": ties,
        "prompt_rows": prompt_rows,
        "phases": [
            {
                "name": p.name,
                "raw_summary":  p.summary("raw"),
                "tune_summary": p.summary("autotune"),
                "runs": [asdict(r) for r in p.runs],
            }
            for p in phases
        ],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="autotune comprehensive stress test")
    parser.add_argument("--model",  default=PRIMARY_MODEL, help="Primary model to test")
    parser.add_argument("--runs",   type=int, default=2,   help="Runs per prompt per mode in main suite")
    parser.add_argument("--output", default=OUTPUT_FILE,   help="JSON output file")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip heavy 14B model phase")
    parser.add_argument("--quick",  action="store_true", help="Quick mode: 1 run per prompt, skip heavy")
    args = parser.parse_args()

    n_runs = 1 if args.quick else args.runs
    skip_heavy = args.skip_heavy or args.quick

    _banner(f"autotune Comprehensive Stress Test  |  model: {args.model}  |  runs/prompt: {n_runs}")
    print(f"\n  Prompts: {len(PROMPTS)} main  |  {len(SUSTAINED_PROMPTS)} sustained  |  1 pressure")
    print(f"  Phases: warmup → main suite → sustained load → pressure → cold-start"
          + ("" if skip_heavy else " → heavy model"))

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    print(f"\n  System snapshot before test:")
    print(f"    RAM: {vm.used/1024**3:.1f}/{vm.total/1024**3:.1f} GB ({vm.percent:.0f}% used)")
    print(f"    Swap: {sw.used/1024**3:.2f} GB used")
    print(f"    CPU cores: {psutil.cpu_count(logical=True)} logical / {psutil.cpu_count(logical=False)} physical")

    t_start = time.perf_counter()
    phases: list[PhaseResult] = []

    # ── Phase 1: Warmup ───────────────────────────────────────────────────────
    _banner("Phase 1: Warmup")
    await warmup(args.model)

    # ── Phase 2+3: Main suite ─────────────────────────────────────────────────
    _banner(f"Phase 2+3: Main Suite  ({len(PROMPTS)} prompts × {n_runs} runs × 2 modes)")
    p_main = await phase_main_suite(args.model, n_runs=n_runs)
    phases.append(p_main)

    # ── Phase 4: Sustained load ───────────────────────────────────────────────
    _banner("Phase 4: Sustained Load  (6 back-to-back calls, no pause)")
    p_sus = await phase_sustained_load(args.model)
    phases.append(p_sus)

    # ── Phase 5: Pressure test ────────────────────────────────────────────────
    _banner("Phase 5: Memory Pressure Test  (large context, 3 × 2 modes)")
    p_press = await phase_pressure_test(args.model)
    phases.append(p_press)

    # ── Phase 6: Cold-start ───────────────────────────────────────────────────
    _banner("Phase 6: Cold-Start vs Warm (keep_alive benefit)")
    p_cold = await phase_cold_start(args.model)
    phases.append(p_cold)

    # ── Phase 7: Heavy model (optional) ──────────────────────────────────────
    if not skip_heavy:
        _banner(f"Phase 7: Heavy Model ({HEAVY_MODEL})")
        p_heavy = await phase_heavy_model()
        phases.append(p_heavy)

    wall_time = time.perf_counter() - t_start

    # ── Report ────────────────────────────────────────────────────────────────
    results = generate_report(phases, args.model, wall_time)

    out_path = Path(__file__).parent.parent / args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
