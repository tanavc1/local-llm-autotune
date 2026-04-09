#!/usr/bin/env python3
"""
autotune benchmark — scientific comparison of autotune vs raw Ollama defaults.

Runs each prompt through:
  1. raw_ollama — zero autotune settings (Ollama defaults)
  2. autotune/fast
  3. autotune/balanced

Records all hardware metrics, prints a results table, and writes results to DB.

Usage:
    python scripts/benchmark.py [--model qwen3:8b] [--runs 3]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune.bench.runner import BenchResult, run_bench, run_raw_ollama, save_result


# ---------------------------------------------------------------------------
# Test prompts — diverse mix of complexity and length
# ---------------------------------------------------------------------------
PROMPTS = [
    {
        "name": "short_qa",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
    },
    {
        "name": "code_gen",
        "messages": [
            {"role": "user",
             "content": (
                 "Write a Python function that takes a list of integers and "
                 "returns the sum of all even numbers. Include a docstring."
             )},
        ],
    },
    {
        "name": "long_context",
        "messages": [
            {"role": "system",
             "content": (
                 "You are a helpful assistant. Be concise and accurate. "
                 "Always cite your reasoning step by step."
             )},
            {"role": "user",
             "content": "Explain how transformer attention mechanisms work."},
            {"role": "assistant",
             "content": (
                 "Transformer attention works by computing query, key, and value "
                 "projections from the input. The attention score is the dot product "
                 "of query and key, scaled by sqrt(d_k), then softmaxed to get "
                 "weights applied to values. This allows each token to attend to "
                 "all other tokens, capturing long-range dependencies efficiently."
             )},
            {"role": "user",
             "content": (
                 "Good. Now explain the difference between self-attention and "
                 "cross-attention, and when you would use each."
             )},
        ],
    },
    {
        "name": "system_prompt_repeat",
        "messages": [
            {"role": "system",
             "content": (
                 "You are an expert Python developer. Always produce clean, "
                 "idiomatic Python code. Prefer standard library over third-party "
                 "when possible. Comment complex logic. Use type hints."
             )},
            {"role": "user",
             "content": "Write a class that implements a thread-safe LRU cache."},
        ],
    },
]


# ---------------------------------------------------------------------------
# Result aggregator
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    label: str
    ttft_ms: list[float]
    tps: list[float]
    elapsed: list[float]
    ram_delta: list[float]
    peak_ram: list[float]
    cpu_avg: list[float]
    errors: int = 0

    def summary(self) -> dict:
        def _s(vals: list[float]) -> str:
            if not vals:
                return "—"
            return f"{statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f}" if len(vals) > 1 else f"{vals[0]:.1f}"

        return {
            "ttft_ms": _s(self.ttft_ms),
            "tps": _s(self.tps),
            "elapsed_s": _s(self.elapsed),
            "ram_delta_gb": _s(self.ram_delta),
            "peak_ram_gb": _s(self.peak_ram),
            "cpu_avg_pct": _s(self.cpu_avg),
            "errors": self.errors,
        }

    def mean_tps(self) -> float:
        return statistics.mean(self.tps) if self.tps else 0.0

    def mean_ttft(self) -> float:
        return statistics.mean(self.ttft_ms) if self.ttft_ms else 0.0

    def mean_ram_delta(self) -> float:
        return statistics.mean(self.ram_delta) if self.ram_delta else 0.0


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

async def _bench_one(
    label: str,
    model_id: str,
    prompt: dict,
    runs: int,
    profile: Optional[str],
) -> RunStats:
    stats = RunStats(
        label=label,
        ttft_ms=[], tps=[], elapsed=[], ram_delta=[], peak_ram=[], cpu_avg=[],
    )
    for run_idx in range(runs):
        print(f"  [{label}] run {run_idx+1}/{runs} ({prompt['name']})… ", end="", flush=True)
        try:
            if profile is None:
                result = await run_raw_ollama(model_id, prompt["messages"], tag=f"raw_{prompt['name']}")
            else:
                result = await run_bench(
                    model_id,
                    prompt["messages"],
                    profile_name=profile,
                    tag=f"autotune_{profile}_{prompt['name']}",
                )
            if result.error:
                print(f"ERROR: {result.error}")
                stats.errors += 1
                continue

            stats.ttft_ms.append(result.ttft_ms)
            stats.tps.append(result.tokens_per_sec)
            stats.elapsed.append(result.elapsed_sec)
            stats.ram_delta.append(result.delta_ram_gb)
            stats.peak_ram.append(result.ram_peak_gb)
            stats.cpu_avg.append(result.cpu_avg_pct)

            print(
                f"{result.tokens_per_sec:.1f} tok/s  "
                f"TTFT {result.ttft_ms:.0f}ms  "
                f"RAM Δ{result.delta_ram_gb:+.3f}GB"
            )
            # Persist to DB
            save_result(result)

            # Cool-down between runs to let RAM settle
            if run_idx < runs - 1:
                await asyncio.sleep(3.0)

        except Exception as e:
            print(f"EXCEPTION: {e}")
            stats.errors += 1

    return stats


async def run_all(model_id: str, n_runs: int) -> dict[str, dict[str, RunStats]]:
    """
    Returns: { prompt_name: { label: RunStats } }
    """
    results: dict[str, dict[str, RunStats]] = {}

    configs = [
        ("raw_ollama",      None),
        ("autotune/fast",   "fast"),
        ("autotune/balanced", "balanced"),
    ]

    for prompt in PROMPTS:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt['name']}")
        print('='*60)
        results[prompt["name"]] = {}

        for label, profile in configs:
            stats = await _bench_one(label, model_id, prompt, n_runs, profile)
            results[prompt["name"]][label] = stats

    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _pct_change(baseline: float, tuned: float) -> str:
    if baseline == 0:
        return "—"
    pct = (tuned - baseline) / baseline * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def print_report(all_results: dict, model_id: str) -> None:
    print("\n")
    print("=" * 80)
    print(f"  AUTOTUNE BENCHMARK RESULTS  —  model: {model_id}")
    print("=" * 80)

    # Aggregate across all prompts
    agg: dict[str, RunStats] = {}
    for prompt_results in all_results.values():
        for label, stats in prompt_results.items():
            if label not in agg:
                agg[label] = RunStats(
                    label=label, ttft_ms=[], tps=[], elapsed=[],
                    ram_delta=[], peak_ram=[], cpu_avg=[], errors=0,
                )
            agg[label].ttft_ms.extend(stats.ttft_ms)
            agg[label].tps.extend(stats.tps)
            agg[label].elapsed.extend(stats.elapsed)
            agg[label].ram_delta.extend(stats.ram_delta)
            agg[label].peak_ram.extend(stats.peak_ram)
            agg[label].cpu_avg.extend(stats.cpu_avg)
            agg[label].errors += stats.errors

    # Table header
    print(f"\n{'Variant':<22}  {'tok/s':>8}  {'TTFT(ms)':>10}  {'RAM Δ(GB)':>10}  {'CPU avg%':>9}  {'errors':>6}")
    print("-" * 75)

    baseline = agg.get("raw_ollama")
    for label, stats in agg.items():
        s = stats.summary()
        tps_note = ""
        if baseline and label != "raw_ollama" and baseline.mean_tps() > 0:
            tps_note = f"  ({_pct_change(baseline.mean_tps(), stats.mean_tps())})"

        print(
            f"{label:<22}  {s['tps']:>8}{tps_note:<12}"
            f"  {s['ttft_ms']:>10}"
            f"  {s['ram_delta_gb']:>10}"
            f"  {s['cpu_avg_pct']:>9}"
            f"  {s['errors']:>6}"
        )

    print("\n  Per-prompt breakdown:")
    print(f"  {'Prompt':<24}  {'Variant':<22}  {'tok/s':>8}  {'TTFT ms':>8}")
    print("  " + "-" * 70)
    for pname, presults in all_results.items():
        for label, stats in presults.items():
            if stats.tps:
                print(
                    f"  {pname:<24}  {label:<22}  "
                    f"{statistics.mean(stats.tps):>7.1f}  "
                    f"{statistics.mean(stats.ttft_ms):>8.0f}"
                )

    # Verdict
    if baseline and "autotune/balanced" in agg:
        bal = agg["autotune/balanced"]
        tps_delta = (bal.mean_tps() - baseline.mean_tps()) / max(baseline.mean_tps(), 0.01) * 100
        ttft_delta = (bal.mean_ttft() - baseline.mean_ttft()) / max(baseline.mean_ttft(), 0.01) * 100
        ram_delta = bal.mean_ram_delta() - baseline.mean_ram_delta()

        print("\n  SUMMARY vs raw Ollama:")
        print(f"    Throughput:   {tps_delta:+.1f}%  (autotune/balanced vs raw_ollama)")
        print(f"    TTFT:         {ttft_delta:+.1f}%")
        print(f"    RAM Δ:        {ram_delta:+.3f} GB  (negative = less memory pressure)")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(all_results: dict, model_id: str, path: str) -> None:
    out: dict = {"model": model_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": {}}
    for pname, presults in all_results.items():
        out["results"][pname] = {}
        for label, stats in presults.items():
            out["results"][pname][label] = {
                "ttft_ms_values": stats.ttft_ms,
                "tps_values": stats.tps,
                "elapsed_values": stats.elapsed,
                "ram_delta_values": stats.ram_delta,
                "peak_ram_values": stats.peak_ram,
                "cpu_avg_values": stats.cpu_avg,
                "errors": stats.errors,
                **stats.summary(),
            }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results exported to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="autotune benchmark")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model ID")
    parser.add_argument("--runs", type=int, default=3, help="Runs per config per prompt")
    parser.add_argument("--output", default="benchmark_results.json", help="JSON output path")
    args = parser.parse_args()

    print(f"autotune benchmark  |  model={args.model}  |  runs={args.runs}")
    print(f"Prompts: {[p['name'] for p in PROMPTS]}")
    print(f"Configs: raw_ollama, autotune/fast, autotune/balanced\n")

    all_results = asyncio.run(run_all(args.model, args.runs))
    print_report(all_results, args.model)
    export_json(all_results, args.model, args.output)


if __name__ == "__main__":
    main()
