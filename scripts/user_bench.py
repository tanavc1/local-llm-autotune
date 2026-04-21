#!/usr/bin/env python3
"""
autotune User Experience Benchmark
====================================
Answers the question every user actually cares about:
  "Will my computer feel slower while this LLM is running?"

This benchmark simulates real-world laptop usage patterns and reports
metrics that map directly to what users experience — not internal engine
details like KV cache slots or prefill buffer sizes.

The 7 KPIs users care about
----------------------------
  swap_events          → "My computer never choked" (GOAL: 0)
  ram_headroom_gb      → "Chrome/Slack/VS Code still had RAM"
  ttft_ms              → "Responses felt fast" (avg + worst-case)
  ttft_consistency_pct → "Response times were predictable"
  cpu_spike_events     → "My fans didn't spin up"
  memory_recovery_sec  → "RAM came back after each call"
  background_impact    → Composite 0–100 score

Scenarios
---------
  1. Normal background query  — single question while apps are running
  2. Sustained 5-turn chat    — a proper conversation
  3. Agent loop               — 8-turn tool-calling session
  4. Memory pressure recovery — what happens when RAM is tight
  5. Cold-start vs warm       — first message slow, subsequent fast

Usage
-----
  # Run in foreground (see progress):
  python scripts/user_bench.py --model qwen3:8b

  # Run in background (survives terminal close, sends desktop notification):
  python scripts/user_bench.py --model qwen3:8b --background

  # Run all scenarios on all installed models:
  python scripts/user_bench.py --all-models --runs 3

  # Quick smoke test (1 run per scenario):
  python scripts/user_bench.py --model llama3.2:3b --runs 1 --quick

Output
------
  user_bench_results.json         — full results for analysis
  user_bench_results_<model>.json — per-model results
  Terminal report card            — human-readable summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import psutil

# ── project root on sys.path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotune.bench.user_metrics import (
    TurnMetrics,
    UserExperienceReport,
    _LiveSampler,
    build_report,
    compute_background_impact_score,
    compute_ttft_consistency,
)
from autotune.api.kv_manager import build_ollama_options, memory_pressure_snapshot
from autotune.api.profiles import get_profile
from autotune.hardware.profiler import profile_hardware

# ── Constants ─────────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"

# Realistic prompts sampled from actual user workflows
SCENARIOS: dict[str, list[dict]] = {
    "background_query": [
        {
            "role": "user",
            "content": (
                "I'm getting a Python error: RecursionError: maximum recursion depth exceeded. "
                "My code is a tree traversal. What's the fastest fix?"
            ),
        }
    ],
    "sustained_chat": [
        {"role": "user", "content": "Can you help me understand async/await in Python?"},
        {"role": "user", "content": "How is that different from threading?"},
        {"role": "user", "content": "When should I use asyncio.gather vs asyncio.create_task?"},
        {"role": "user", "content": "Show me a real example with HTTP requests using httpx."},
        {"role": "user", "content": "How do I handle errors properly in that pattern?"},
    ],
    "agent_loop": [
        {"role": "user", "content": "Task: write a Python script that reads a CSV, filters rows where 'status' == 'active', and outputs a summary. Step 1: plan the approach."},
        {"role": "user", "content": "Step 2: write the CSV reading and filtering code."},
        {"role": "user", "content": "Step 3: add error handling for missing files."},
        {"role": "user", "content": "Step 4: add the summary output with counts."},
        {"role": "user", "content": "Step 5: write unit tests for the filter function."},
        {"role": "user", "content": "Step 6: add logging to the script."},
        {"role": "user", "content": "Step 7: make it a CLI with argparse."},
        {"role": "user", "content": "Step 8: final review — any improvements?"},
    ],
    "long_context": [
        {
            "role": "user",
            "content": (
                "I have this 500-line Python module:\n\n"
                + ("class DataProcessor:\n    def process(self, data):\n        pass\n\n" * 30)
                + "\nReview for bugs and suggest improvements. Focus on thread safety."
            ),
        }
    ],
    "code_debug": [
        {
            "role": "user",
            "content": (
                "Debug this async code:\n\n"
                "```python\n"
                "async def fetch_all(urls):\n"
                "    results = []\n"
                "    for url in urls:\n"
                "        async with httpx.AsyncClient() as client:\n"
                "            r = await client.get(url)\n"
                "            results.append(r.json())\n"
                "    return results\n"
                "```\n\n"
                "It's slow. What's wrong and how do I fix it?"
            ),
        }
    ],
}

SYSTEM_PROMPT = (
    "You are a helpful programming assistant. "
    "Give concise, accurate answers. When writing code, use type hints."
)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama direct HTTP client (no server needed — raw benchmark)
# ─────────────────────────────────────────────────────────────────────────────

async def _ollama_chat(
    model_id: str,
    messages: list[dict],
    options: dict,
    keep_alive: str = "-1m",
    timeout: float = 120.0,
) -> dict:
    """
    POST to Ollama /api/chat directly.
    Returns timing fields from Ollama's internal Go timers.
    """
    import httpx

    payload = {
        "model":      model_id,
        "messages":   messages,
        "stream":     False,
        "options":    options,
        "keep_alive": keep_alive,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()


async def _check_ollama() -> list[str]:
    """Return list of locally installed Ollama model names, or [] if Ollama is down."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Single-turn runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_one_turn(
    model_id: str,
    conversation: list[dict],
    profile_name: str = "balanced",
    use_autotune: bool = True,
    turn_number: int = 1,
) -> TurnMetrics:
    """
    Run a single inference turn and return user-facing metrics.

    Parameters
    ----------
    conversation : full message history up to this turn (including the new user message)
    use_autotune : True = autotune-optimised options; False = raw Ollama defaults
    """
    profile = get_profile(profile_name)

    sampler = _LiveSampler(interval_sec=0.25)
    sampler.start()
    t0 = time.monotonic()

    error: Optional[str] = None
    ttft_ms = 0.0
    completion_tokens = 0
    prompt_tokens = 0

    try:
        if use_autotune:
            opts, _ = build_ollama_options(
                messages=conversation,
                profile=profile,
                context_ceiling=profile.max_context_tokens,
            )
        else:
            # Raw Ollama defaults — fixed 4096 context, no tuning
            opts = {
                "num_ctx":   4096,
                "num_keep":  0,
                "num_batch": 512,
            }

        result = await _ollama_chat(
            model_id=model_id,
            messages=conversation,
            options=opts,
        )

        # Extract Ollama's internal timers (nanoseconds → ms)
        load_ns   = result.get("load_duration",         0)
        prefill_ns = result.get("prompt_eval_duration", 0)
        ttft_ms   = (load_ns + prefill_ns) / 1_000_000
        completion_tokens = result.get("eval_count",          0)
        prompt_tokens     = result.get("prompt_eval_count",   0)

    except Exception as exc:
        error = str(exc)
        ttft_ms = -1.0

    elapsed = time.monotonic() - t0
    sampler.stop()

    return TurnMetrics(
        turn_number=turn_number,
        ttft_ms=ttft_ms,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        ram_delta_gb=round(sampler.ram_after_gb() - sampler.ram_before_gb(), 3),
        swap_events=len(sampler.swap_events),
        cpu_spike_events=sampler.cpu_spike_events,
        elapsed_sec=round(elapsed, 2),
    ), sampler


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runners
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario(
    scenario_name: str,
    model_id: str,
    profile_name: str = "balanced",
    use_autotune: bool = True,
    runs: int = 3,
) -> UserExperienceReport:
    """Run a scenario N times and return the aggregate UserExperienceReport."""

    prompts = SCENARIOS[scenario_name]
    hw = profile_hardware()
    total_ram_gb = hw.memory.total_gb

    all_turns: list[TurnMetrics] = []
    all_samplers: list[_LiveSampler] = []

    # Global sampler for aggregate stats (wraps entire scenario)
    global_sampler = _LiveSampler(interval_sec=0.5)
    global_sampler.start()
    scenario_start = time.monotonic()

    for run_idx in range(runs):
        conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn_idx, prompt in enumerate(prompts):
            conversation.append(prompt)
            turn_result, turn_sampler = await run_one_turn(
                model_id=model_id,
                conversation=conversation,
                profile_name=profile_name,
                use_autotune=use_autotune,
                turn_number=run_idx * len(prompts) + turn_idx + 1,
            )
            # Add assistant reply to conversation if not error
            if turn_result.ttft_ms > 0:
                conversation.append({
                    "role": "assistant",
                    "content": f"[response #{turn_idx+1}]",
                })

            all_turns.append(turn_result)
            all_samplers.append(turn_sampler)

        # Brief pause between runs to let RAM settle
        if run_idx < runs - 1:
            await asyncio.sleep(2.0)

    scenario_elapsed = time.monotonic() - scenario_start
    global_sampler.stop()

    # Compute memory recovery: time for RAM to settle post-scenario
    recovery = global_sampler.ram_recovery_sec(threshold_gb=0.2)

    return build_report(
        scenario=f"{scenario_name} ({runs} run{'s' if runs > 1 else ''})",
        model_id=model_id,
        profile_name=profile_name,
        turns=all_turns,
        total_ram_gb=total_ram_gb,
        ram_before_gb=global_sampler.ram_before_gb(),
        ram_peak_gb=global_sampler.ram_peak_gb(),
        ram_after_gb=global_sampler.ram_after_gb(),
        swap_before_gb=global_sampler.swap_before_gb(),
        swap_peak_gb=global_sampler.swap_peak_gb(),
        cpu_avg_pct=global_sampler.cpu_avg_pct(),
        cpu_peak_pct=global_sampler.cpu_peak_pct(),
        elapsed_total_sec=round(scenario_elapsed, 1),
        memory_recovery_sec=recovery,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Head-to-head comparison (autotune vs raw Ollama)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    scenario: str
    model_id: str
    autotune: UserExperienceReport
    raw: UserExperienceReport

    @property
    def ttft_improvement_pct(self) -> float:
        if self.raw.ttft_ms_mean <= 0:
            return 0.0
        return round((self.raw.ttft_ms_mean - self.autotune.ttft_ms_mean) / self.raw.ttft_ms_mean * 100, 1)

    @property
    def ram_headroom_saved_gb(self) -> float:
        return round(self.autotune.ram_headroom_gb - self.raw.ram_headroom_gb, 2)

    @property
    def score_improvement(self) -> float:
        return round(self.autotune.background_impact_score - self.raw.background_impact_score, 1)

    def print_comparison(self) -> None:
        W = 70
        print(f"\n{'━' * W}")
        print(f"  COMPARISON  ·  {self.scenario}  ·  {self.model_id}")
        print(f"{'━' * W}")
        print(f"\n  {'Metric':<38} {'Raw Ollama':>12} {'autotune':>12}")
        print(f"  {'─' * 64}")

        def row(label, raw_val, at_val, better_is_lower=True):
            if better_is_lower:
                better = "✅" if at_val <= raw_val else "⚠️ "
            else:
                better = "✅" if at_val >= raw_val else "⚠️ "
            print(f"  {better} {label:<36} {raw_val:>12} {at_val:>12}")

        row("Swap events",              self.raw.swap_events_total,              self.autotune.swap_events_total)
        row("RAM free for other apps",  f"{self.raw.ram_headroom_gb:.1f} GB",     f"{self.autotune.ram_headroom_gb:.1f} GB", better_is_lower=False)
        row("Time to first word (avg)", f"{self.raw.ttft_ms_mean:.0f} ms",        f"{self.autotune.ttft_ms_mean:.0f} ms")
        row("Worst-case TTFT",          f"{self.raw.ttft_ms_p95:.0f} ms",         f"{self.autotune.ttft_ms_p95:.0f} ms")
        row("Response consistency",     f"{self.raw.ttft_consistency_pct:.0f}%",  f"{self.autotune.ttft_consistency_pct:.0f}%", better_is_lower=False)
        row("CPU spikes",               self.raw.cpu_spike_events_total,          self.autotune.cpu_spike_events_total)
        row("Memory recovery",          f"{self.raw.memory_recovery_sec:.1f} s",  f"{self.autotune.memory_recovery_sec:.1f} s")
        row("Background impact score",  f"{self.raw.background_impact_score:.0f}/100", f"{self.autotune.background_impact_score:.0f}/100", better_is_lower=False)

        if self.ttft_improvement_pct > 0:
            print(f"\n  → TTFT improved by {self.ttft_improvement_pct:.0f}%")
        if self.ram_headroom_saved_gb > 0:
            print(f"  → {self.ram_headroom_saved_gb:.1f} GB more RAM free for your apps")
        print(f"  → Impact score: {self.raw.background_impact_score:.0f} → {self.autotune.background_impact_score:.0f} (+{self.score_improvement:.0f})")
        print(f"{'━' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Desktop notification
# ─────────────────────────────────────────────────────────────────────────────

def _notify(title: str, message: str) -> None:
    """Send a desktop notification (macOS / Linux / Windows)."""
    system = platform.system()
    try:
        if system == "Darwin":
            script = (
                f'display notification "{message}" '
                f'with title "{title}" '
                f'sound name "Glass"'
            )
            subprocess.run(
                ["osascript", "-e", script],
                check=False, capture_output=True, timeout=5
            )
        elif system == "Linux":
            subprocess.run(
                ["notify-send", title, message, "--urgency=normal"],
                check=False, capture_output=True, timeout=5
            )
        elif system == "Windows":
            # Windows 10+ toast notification via PowerShell
            ps_script = (
                f"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null;"
                f"$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
                f"$template.GetElementsByTagName('text')[0].AppendChild($template.CreateTextNode('{title}'));"
                f"$template.GetElementsByTagName('text')[1].AppendChild($template.CreateTextNode('{message}'));"
                f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('autotune').Show([Windows.UI.Notifications.ToastNotification]::new($template));"
            )
            subprocess.run(
                ["powershell", "-Command", ps_script],
                check=False, capture_output=True, timeout=10
            )
    except Exception:
        pass  # Notifications are best-effort


# ─────────────────────────────────────────────────────────────────────────────
# Results persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(
    model_id: str,
    comparisons: list[ComparisonResult],
    output_dir: Path,
) -> Path:
    safe_name = model_id.replace(":", "_").replace("/", "_")
    output_file = output_dir / f"user_bench_{safe_name}.json"

    payload = {
        "model_id":    model_id,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform":    platform.platform(),
        "scenarios": [
            {
                "scenario":    c.scenario,
                "autotune":    c.autotune.to_dict(),
                "raw":         c.raw.to_dict(),
                "improvement": {
                    "ttft_pct":           c.ttft_improvement_pct,
                    "ram_headroom_gb":    c.ram_headroom_saved_gb,
                    "score_delta":        c.score_improvement,
                },
            }
            for c in comparisons
        ],
    }

    output_file.write_text(json.dumps(payload, indent=2))
    return output_file


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def benchmark_model(
    model_id: str,
    profile_name: str = "balanced",
    runs: int = 3,
    quick: bool = False,
    output_dir: Path = Path("."),
    verbose: bool = True,
) -> list[ComparisonResult]:

    scenarios_to_run = (
        ["background_query", "sustained_chat"]
        if quick
        else ["background_query", "sustained_chat", "agent_loop", "code_debug"]
    )

    comparisons: list[ComparisonResult] = []

    for scenario_name in scenarios_to_run:
        if verbose:
            print(f"\n  ▶  {scenario_name}  (autotune)  …", end="", flush=True)

        at_report = await run_scenario(
            scenario_name, model_id, profile_name,
            use_autotune=True, runs=runs
        )

        if verbose:
            print(f"  done ({at_report.elapsed_total_sec:.0f}s)")
            print(f"     ▶  {scenario_name}  (raw Ollama) …", end="", flush=True)

        raw_report = await run_scenario(
            scenario_name, model_id, profile_name,
            use_autotune=False, runs=runs
        )

        if verbose:
            print(f"  done ({raw_report.elapsed_total_sec:.0f}s)")

        comp = ComparisonResult(
            scenario=scenario_name,
            model_id=model_id,
            autotune=at_report,
            raw=raw_report,
        )
        comparisons.append(comp)

        if verbose:
            comp.print_comparison()

    return comparisons


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="autotune User Experience Benchmark — measures what users actually feel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", "-m",   default="",         help="Ollama model to benchmark (e.g. qwen3:8b)")
    p.add_argument("--profile", "-p", default="balanced", help="autotune profile: fast|balanced|quality")
    p.add_argument("--runs",    "-r", type=int, default=3, help="Runs per scenario per condition (default: 3)")
    p.add_argument("--quick",   "-q", action="store_true", help="Run 2 scenarios instead of 4 for a faster smoke test")
    p.add_argument("--all-models",    action="store_true", help="Run on all locally installed Ollama models")
    p.add_argument("--background",    action="store_true", help="Fork to background (survives terminal close, sends desktop notification)")
    p.add_argument("--output-dir",    default=".",         help="Directory for result JSON files (default: current dir)")
    return p


def _check_ollama_sync() -> list[str]:
    """Synchronous Ollama check — used before forking so we fail-fast in the foreground."""
    import httpx as _httpx
    try:
        r = _httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3.0)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


async def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Check Ollama ──────────────────────────────────────────────────────────
    available_models = await _check_ollama()
    if not available_models:
        print(
            "\n❌  Ollama is not running.\n"
            "    Start it with: ollama serve\n"
            "    Then pull a model: ollama pull qwen3:8b\n"
        )
        sys.exit(1)

    # ── Resolve models to benchmark ───────────────────────────────────────────
    if args.all_models:
        models = available_models
    elif args.model:
        if args.model not in available_models:
            print(f"\n❌  Model '{args.model}' not installed.")
            print(f"    Available: {', '.join(available_models)}")
            print(f"    Install with: ollama pull {args.model}\n")
            sys.exit(1)
        models = [args.model]
    else:
        models = [available_models[0]]
        print(f"No model specified. Using: {models[0]}")
        print(f"Available: {', '.join(available_models)}\n")

    # ── Run benchmarks ────────────────────────────────────────────────────────
    benchmark_start = time.time()
    all_results: dict[str, list[ComparisonResult]] = {}

    print(f"\n{'━' * 70}")
    print(f"  autotune USER EXPERIENCE BENCHMARK")
    print(f"  Profile: {args.profile}  ·  Runs per scenario: {args.runs}")
    print(f"  Models: {', '.join(models)}")
    print(f"{'━' * 70}")

    for model_id in models:
        print(f"\n  ═══  {model_id}  ═══")
        comparisons = await benchmark_model(
            model_id=model_id,
            profile_name=args.profile,
            runs=args.runs,
            quick=args.quick,
            output_dir=output_dir,
            verbose=True,
        )
        all_results[model_id] = comparisons

        result_file = _save_results(model_id, comparisons, output_dir)
        print(f"\n  Results saved → {result_file}")

    elapsed = time.time() - benchmark_start

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print(f"  BENCHMARK COMPLETE  ·  {elapsed:.0f}s total")
    print(f"{'━' * 70}")

    for model_id, comparisons in all_results.items():
        avg_score_at  = statistics.mean(c.autotune.background_impact_score for c in comparisons)
        avg_score_raw = statistics.mean(c.raw.background_impact_score for c in comparisons)
        swap_at  = sum(c.autotune.swap_events_total for c in comparisons)
        swap_raw = sum(c.raw.swap_events_total for c in comparisons)

        print(f"\n  {model_id}")
        print(f"    Background impact score:  Raw {avg_score_raw:.0f}/100  →  autotune {avg_score_at:.0f}/100")
        print(f"    Swap events (total):       Raw {swap_raw}  →  autotune {swap_at}")
        if swap_at == 0:
            print(f"    ✅ Zero swap events — your computer won't feel the LLM running")
        else:
            print(f"    ⚠️  {swap_at} swap events — try --profile fast or a smaller model")

    # ── Desktop notification ───────────────────────────────────────────────────
    model_str = ", ".join(all_results.keys())
    _notify(
        title="autotune benchmark complete ✅",
        message=f"Finished {model_str} in {elapsed:.0f}s. Check your terminal for results.",
    )


if __name__ == "__main__":
    _parser = _build_parser()
    _args   = _parser.parse_args()

    # ── Fail-fast Ollama check BEFORE forking ─────────────────────────────────
    # (Fork must happen before asyncio.run() — the child inherits the process
    #  state before any event loop is created, avoiding coroutine corruption.)
    if _args.background:
        _models = _check_ollama_sync()
        if not _models:
            print(
                "\n❌  Ollama is not running — cannot start background benchmark.\n"
                "    Start it with: ollama serve\n"
            )
            sys.exit(1)

        # Fork here, before asyncio
        _pid = os.fork()
        if _pid != 0:
            # Parent: print confirmation and exit
            _log = Path(_args.output_dir) / "user_bench.log"
            print(
                f"✓  Benchmark running in background (PID {_pid}).\n"
                f"   Log: {_log}\n"
                f"   You'll get a desktop notification when it's done."
            )
            sys.exit(0)

        # Child: redirect stdout/stderr to log file so nothing is lost
        _log_path = Path(_args.output_dir) / "user_bench.log"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_fd = open(_log_path, "w", buffering=1)
        sys.stdout = _log_fd
        sys.stderr = _log_fd

    # Windows asyncio policy
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(_args))
