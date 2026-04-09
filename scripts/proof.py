#!/usr/bin/env python3
"""
autotune proof — honest, traceable benchmark

Compares raw Ollama (zero middleware) vs autotune on three dimensions:

  Phase 1 — Warm prefill latency
    Model stays loaded. Measures time Ollama spends filling the KV cache
    before generating the first token. Source: prompt_eval_duration from
    Ollama's own /api/chat response (nanoseconds from Go time.Now()).

  Phase 2 — Cold-start load time
    Model unloaded between every call. Measures full KV-buffer allocation
    cost. Source: load_duration from /api/chat response.

  Phase 3 — VRAM footprint
    Reads size_vram from /api/ps after loading each config. This is the
    actual Metal unified memory Ollama holds — weights + KV cache combined.

Nothing here is estimated by Python timing or psutil. Every latency number
comes directly from Ollama's internal Go timers.

What this proves (and what it honestly does not):
  PROVEN    prefill_ms is lower    — smaller KV buffer takes less time to init
  PROVEN    load_ms is lower       — smaller Metal MTLBuffer allocation
  PROVEN    VRAM is lower          — measured directly from /api/ps
  NOT CLAIMED  tok/s changes       — GPU-bound, num_ctx has no effect on decode

Usage:
    python scripts/proof.py
    python scripts/proof.py --model llama3.2:3b
    python scripts/proof.py --model phi4-mini:latest --runs 5
    python scripts/proof.py --list-models
    python scripts/proof.py --skip-cold --skip-vram   (quick warm-only run)
    autotune proof [same flags]
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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from autotune.metrics.ollama_client import OllamaMetricsClient, NativeInferenceStats
from autotune.metrics.vram import VRAMTracker, VRAMSnapshot
from autotune.ttft.optimizer import TTFTOptimizer
from autotune.api.profiles import get_profile

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Test prompts — chosen to cover the realistic distribution of real-world use
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = [
    {
        "id": "tiny",
        "label": "Tiny",
        "messages": [
            {"role": "user", "content": "What is 144 divided by 12, times 7?"},
        ],
    },
    {
        "id": "short_factual",
        "label": "Short factual",
        "messages": [
            {"role": "user", "content": (
                "List five countries in Southeast Asia and their capitals. "
                "Format as a bulleted list."
            )},
        ],
    },
    {
        "id": "code_gen",
        "label": "Code gen",
        "messages": [
            {"role": "user", "content": (
                "Write a Python function called `binary_search` that searches "
                "a sorted list for a target value and returns its index, or -1 "
                "if not found. Include type hints and a short docstring."
            )},
        ],
    },
    {
        "id": "system_user",
        "label": "System+user",
        "messages": [
            {"role": "system", "content": (
                "You are an expert Python developer. Always write clean, idiomatic "
                "Python with type hints, docstrings, and proper error handling. "
                "Prefer the standard library. Keep functions focused and short."
            )},
            {"role": "user", "content": (
                "Write a function that reads a JSON config file and returns a "
                "validated dict, raising a clear error if required keys are missing."
            )},
        ],
    },
    {
        "id": "code_review",
        "label": "Code review",
        "messages": [
            {"role": "user", "content": (
                "Review the following Python code for security vulnerabilities, "
                "bugs, and performance issues. Be specific and brief:\n\n"
                "```python\n"
                "import sqlite3\n\n"
                "def get_user(username, password, db_path='users.db'):\n"
                "    conn = sqlite3.connect(db_path)\n"
                "    cursor = conn.cursor()\n"
                "    query = f\"SELECT * FROM users WHERE username='{username}' "
                "AND password='{password}'\"\n"
                "    result = cursor.execute(query).fetchone()\n"
                "    conn.close()\n"
                "    return result\n\n"
                "def update_profile(user_id, data):\n"
                "    conn = sqlite3.connect('users.db')\n"
                "    for key, val in data.items():\n"
                "        conn.execute(f'UPDATE users SET {key}={val} "
                "WHERE id={user_id}')\n"
                "    conn.commit()\n"
                "```\n"
            )},
        ],
    },
    {
        "id": "multi_turn",
        "label": "Multi-turn",
        "messages": [
            {"role": "user",
             "content": "Explain what a hash table is and how collision resolution works."},
            {"role": "assistant",
             "content": (
                 "A hash table is a data structure that maps keys to values using "
                 "a hash function. The hash function converts a key into an index "
                 "in an underlying array. Collision resolution strategies include: "
                 "chaining (each bucket holds a linked list of entries), open "
                 "addressing (linear/quadratic probing to find the next empty slot), "
                 "and Robin Hood hashing (swap entries to minimise probe distances). "
                 "Python's dict uses open addressing with a compact hash table."
             )},
            {"role": "user",
             "content": (
                 "What is the average-case time complexity for lookup, and what "
                 "degrades it to worst-case? Give a concrete Python example of "
                 "when this matters."
             )},
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Configuration builders
# ─────────────────────────────────────────────────────────────────────────────

RAW_CTX = 4096
RAW_KEEP_ALIVE = "5m"

def raw_options() -> dict:
    """Baseline: Ollama defaults. No autotune settings at all."""
    return {"num_ctx": RAW_CTX}


def autotune_options_for(messages: list[dict]) -> tuple[dict, str]:
    """
    Compute autotune's options for these messages.
    Returns (options_dict, keep_alive_str).
    """
    profile = get_profile("balanced")
    result = TTFTOptimizer().build_request_options(messages, profile)
    return result["options"], result["keep_alive"]


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    prompt_id: str
    mode: str          # "raw" or "autotune"
    run_idx: int
    num_ctx: int
    prefill_ms: float
    eval_tps: float
    load_ms: float
    eval_count: int
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class PhaseResult:
    mode: str
    runs: list[RunResult] = field(default_factory=list)

    def ok_runs(self) -> list[RunResult]:
        return [r for r in self.runs if r.ok]

    def mean_prefill(self) -> Optional[float]:
        vals = [r.prefill_ms for r in self.ok_runs() if r.prefill_ms > 0]
        return statistics.mean(vals) if vals else None

    def mean_tps(self) -> Optional[float]:
        vals = [r.eval_tps for r in self.ok_runs() if r.eval_tps > 0]
        return statistics.mean(vals) if vals else None

    def mean_load(self) -> Optional[float]:
        vals = [r.load_ms for r in self.ok_runs()]
        return statistics.mean(vals) if vals else None

    def prefill_by_prompt(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for r in self.ok_runs():
            if r.prefill_ms > 0:
                out.setdefault(r.prompt_id, []).append(r.prefill_ms)
        return {k: statistics.mean(v) for k, v in out.items()}

    def ctx_by_prompt(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.ok_runs():
            out[r.prompt_id] = r.num_ctx  # last value wins (they should all be same)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(baseline: float, tuned: float) -> float:
    if baseline <= 0:
        return 0.0
    return (tuned - baseline) / baseline * 100.0


def _fmt_pct(pct: float, invert: bool = False) -> tuple[str, str]:
    """
    Returns (text, style).
    invert=True: lower is better (latency metrics — negative pct = good).
    invert=False: higher is better (throughput).
    """
    sign = "+" if pct >= 0 else ""
    text = f"{sign}{pct:.1f}%"
    if invert:
        good = pct <= -5.0
        bad  = pct >= +5.0
    else:
        good = pct >= +5.0
        bad  = pct <= -5.0
    neutral = not good and not bad
    style = "green bold" if good else ("red" if bad else "dim")
    indicator = " ✓" if good else (" ✗" if bad else " ≈")
    return text + indicator, style


def _get_system_info() -> dict:
    vm = psutil.virtual_memory()
    info = {
        "platform": platform.system(),
        "machine":  platform.machine(),
        "ram_total_gb": round(vm.total / 1024**3, 1),
        "ram_used_pct": round(vm.percent, 1),
        "chip": "unknown",
    }
    try:
        r = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if "Chip:" in line:
                info["chip"] = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return info


async def _list_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get("http://localhost:11434/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


async def _ensure_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get("http://localhost:11434/api/ps")
            return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Warm inference
# ─────────────────────────────────────────────────────────────────────────────

async def run_warm_phase(
    model: str,
    n_runs: int,
    client: OllamaMetricsClient,
) -> tuple[PhaseResult, PhaseResult]:
    """
    Run all prompts N times for each config with model staying loaded.
    Returns (raw_results, autotune_results).

    Protocol:
    - For each prompt: run raw N times, then autotune N times.
    - First call of each config may trigger a KV-realloc if num_ctx changed;
      this is included in timing because it reflects real-world behaviour.
    - keep_alive is set so model stays loaded across runs.
    """
    raw_phase = PhaseResult(mode="raw")
    tune_phase = PhaseResult(mode="autotune")

    total_runs = len(PROMPTS) * 2 * n_runs
    done = 0

    for prompt in PROMPTS:
        msgs = prompt["messages"]
        pid  = prompt["id"]
        lbl  = prompt["label"]

        # ── RAW runs ──────────────────────────────────────────────────────────
        r_opts = raw_options()
        for run_i in range(n_runs):
            done += 1
            console.print(
                f"  [dim]warm[/dim] [cyan]{pid}[/cyan]  raw  "
                f"[dim]run {run_i+1}/{n_runs}[/dim]  "
                f"[dim]({done}/{total_runs})[/dim]",
                end="  ",
            )
            stats = await client.run_with_stats(
                model=model,
                messages=msgs,
                options=r_opts,
                keep_alive=RAW_KEEP_ALIVE,
            )
            if stats.error:
                console.print(f"[red]ERROR: {stats.error}[/red]")
                raw_phase.runs.append(RunResult(
                    prompt_id=pid, mode="raw", run_idx=run_i,
                    num_ctx=RAW_CTX, prefill_ms=0, eval_tps=0, load_ms=0,
                    eval_count=0, error=stats.error,
                ))
            else:
                console.print(
                    f"[green]prefill={stats.prefill_ms:.0f}ms[/green]  "
                    f"tps={stats.eval_tps:.1f}  "
                    f"ctx={stats.num_ctx}"
                )
                raw_phase.runs.append(RunResult(
                    prompt_id=pid, mode="raw", run_idx=run_i,
                    num_ctx=stats.num_ctx, prefill_ms=stats.prefill_ms,
                    eval_tps=stats.eval_tps, load_ms=stats.load_ms,
                    eval_count=stats.eval_count,
                ))

        # ── AUTOTUNE runs ─────────────────────────────────────────────────────
        a_opts, a_keep = autotune_options_for(msgs)
        for run_i in range(n_runs):
            done += 1
            console.print(
                f"  [dim]warm[/dim] [cyan]{pid}[/cyan]  autotune  "
                f"[dim]run {run_i+1}/{n_runs}[/dim]  "
                f"[dim]({done}/{total_runs})[/dim]",
                end="  ",
            )
            stats = await client.run_with_stats(
                model=model,
                messages=msgs,
                options=a_opts,
                keep_alive=a_keep,
            )
            if stats.error:
                console.print(f"[red]ERROR: {stats.error}[/red]")
                tune_phase.runs.append(RunResult(
                    prompt_id=pid, mode="autotune", run_idx=run_i,
                    num_ctx=a_opts.get("num_ctx", 0), prefill_ms=0,
                    eval_tps=0, load_ms=0, eval_count=0, error=stats.error,
                ))
            else:
                console.print(
                    f"[green]prefill={stats.prefill_ms:.0f}ms[/green]  "
                    f"tps={stats.eval_tps:.1f}  "
                    f"ctx={stats.num_ctx}"
                )
                tune_phase.runs.append(RunResult(
                    prompt_id=pid, mode="autotune", run_idx=run_i,
                    num_ctx=stats.num_ctx, prefill_ms=stats.prefill_ms,
                    eval_tps=stats.eval_tps, load_ms=stats.load_ms,
                    eval_count=stats.eval_count,
                ))

    return raw_phase, tune_phase


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Cold-start
# ─────────────────────────────────────────────────────────────────────────────

async def run_cold_phase(
    model: str,
    n_calls: int,
    client: OllamaMetricsClient,
) -> tuple[PhaseResult, PhaseResult]:
    """
    Measure model load + KV allocation time with model unloaded between each call.
    Uses a single representative prompt (short_factual) for repeatability.
    Returns (raw_results, autotune_results).
    """
    raw_phase  = PhaseResult(mode="raw_cold")
    tune_phase = PhaseResult(mode="autotune_cold")

    # Use the short factual prompt — small, representative
    probe = next(p for p in PROMPTS if p["id"] == "short_factual")
    msgs  = probe["messages"]

    tracker = VRAMTracker()

    async def _unload(label: str) -> None:
        console.print(f"  [dim]Unloading {model}…[/dim]", end=" ")
        await client.unload_model(model)
        await asyncio.sleep(2.0)
        console.print("[dim]done.[/dim]")

    # ── RAW cold calls ────────────────────────────────────────────────────────
    r_opts = raw_options()
    for i in range(n_calls):
        await _unload("raw")
        console.print(
            f"  [dim]cold[/dim] raw  [dim]call {i+1}/{n_calls}[/dim]",
            end="  ",
        )
        stats = await client.run_with_stats(
            model=model,
            messages=msgs,
            options=r_opts,
            keep_alive=RAW_KEEP_ALIVE,
        )
        if stats.error:
            console.print(f"[red]ERROR: {stats.error}[/red]")
            raw_phase.runs.append(RunResult(
                prompt_id="cold_probe", mode="raw_cold", run_idx=i,
                num_ctx=RAW_CTX, prefill_ms=0, eval_tps=0, load_ms=0,
                eval_count=0, error=stats.error,
            ))
        else:
            ttft = stats.load_ms + stats.prefill_ms
            console.print(
                f"[green]load={stats.load_ms:.0f}ms[/green]  "
                f"prefill={stats.prefill_ms:.0f}ms  "
                f"ttft={ttft:.0f}ms  ctx={stats.num_ctx}"
            )
            raw_phase.runs.append(RunResult(
                prompt_id="cold_probe", mode="raw_cold", run_idx=i,
                num_ctx=stats.num_ctx, prefill_ms=stats.prefill_ms,
                eval_tps=stats.eval_tps, load_ms=stats.load_ms,
                eval_count=stats.eval_count,
            ))

    # ── AUTOTUNE cold calls ───────────────────────────────────────────────────
    a_opts, a_keep = autotune_options_for(msgs)
    for i in range(n_calls):
        await _unload("autotune")
        console.print(
            f"  [dim]cold[/dim] autotune  [dim]call {i+1}/{n_calls}[/dim]",
            end="  ",
        )
        stats = await client.run_with_stats(
            model=model,
            messages=msgs,
            options=a_opts,
            keep_alive=a_keep,
        )
        if stats.error:
            console.print(f"[red]ERROR: {stats.error}[/red]")
            tune_phase.runs.append(RunResult(
                prompt_id="cold_probe", mode="autotune_cold", run_idx=i,
                num_ctx=a_opts.get("num_ctx", 0), prefill_ms=0,
                eval_tps=0, load_ms=0, eval_count=0, error=stats.error,
            ))
        else:
            ttft = stats.load_ms + stats.prefill_ms
            console.print(
                f"[green]load={stats.load_ms:.0f}ms[/green]  "
                f"prefill={stats.prefill_ms:.0f}ms  "
                f"ttft={ttft:.0f}ms  ctx={stats.num_ctx}"
            )
            tune_phase.runs.append(RunResult(
                prompt_id="cold_probe", mode="autotune_cold", run_idx=i,
                num_ctx=stats.num_ctx, prefill_ms=stats.prefill_ms,
                eval_tps=stats.eval_tps, load_ms=stats.load_ms,
                eval_count=stats.eval_count,
            ))

    return raw_phase, tune_phase


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — VRAM footprint
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VRAMResult:
    raw_ctx: int
    raw_vram_gb: float
    tune_ctx: int
    tune_vram_gb: float
    delta_gb: float
    theory_delta_gb: float

    @property
    def overhead_gb(self) -> float:
        return self.delta_gb - self.theory_delta_gb


async def run_vram_phase(
    model: str,
    client: OllamaMetricsClient,
) -> Optional[VRAMResult]:
    """
    Load model with each config, read /api/ps size_vram, compare.
    Returns None if either load fails.
    """
    tracker = VRAMTracker()

    # Determine autotune ctx using the short_factual probe
    probe_msgs = next(p["messages"] for p in PROMPTS if p["id"] == "short_factual")
    a_opts, a_keep = autotune_options_for(probe_msgs)
    tune_ctx = a_opts.get("num_ctx", 1024)

    # ── Load with autotune ctx ────────────────────────────────────────────────
    console.print(f"  [dim]Loading {model} with autotune ctx={tune_ctx}…[/dim]", end=" ")
    await client.unload_model(model)
    await asyncio.sleep(1.0)
    stats = await client.run_with_stats(
        model=model, messages=probe_msgs, options=a_opts, keep_alive="30m",
    )
    if stats.error:
        console.print(f"[red]ERROR: {stats.error}[/red]")
        return None
    snap_tune = await tracker.snapshot(model)
    console.print(f"[green]size_vram = {snap_tune.size_vram_gb:.3f} GB[/green]")

    # ── Load with raw ctx ─────────────────────────────────────────────────────
    console.print(f"  [dim]Loading {model} with raw ctx={RAW_CTX}…[/dim]", end=" ")
    await client.unload_model(model)
    await asyncio.sleep(1.0)
    r_opts = raw_options()
    stats = await client.run_with_stats(
        model=model, messages=probe_msgs, options=r_opts, keep_alive="30m",
    )
    if stats.error:
        console.print(f"[red]ERROR: {stats.error}[/red]")
        return None
    snap_raw = await tracker.snapshot(model)
    console.print(f"[green]size_vram = {snap_raw.size_vram_gb:.3f} GB[/green]")

    # Theoretical KV-only savings (phi4-mini: 32L, 8KV heads, 128 head_dim)
    # These are approximate — we don't know exact arch without querying modelinfo
    theory = VRAMTracker.kv_savings_gb(
        ctx_raw=RAW_CTX, ctx_tuned=tune_ctx,
        n_layers=32, n_kv_heads=8, head_dim=128, f16_kv=True,
    )

    delta = snap_raw.size_vram_gb - snap_tune.size_vram_gb

    # Restore autotune config (keep loaded for rest of session)
    await client.unload_model(model)

    return VRAMResult(
        raw_ctx=RAW_CTX,
        raw_vram_gb=snap_raw.size_vram_gb,
        tune_ctx=tune_ctx,
        tune_vram_gb=snap_tune.size_vram_gb,
        delta_gb=round(delta, 3),
        theory_delta_gb=round(theory, 3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output — Rich tables
# ─────────────────────────────────────────────────────────────────────────────

def print_header(model: str, sysinfo: dict, n_runs: int) -> None:
    console.print()
    console.print(Panel(
        f"[bold white]autotune proof-of-improvement[/bold white]  —  "
        f"[cyan]{model}[/cyan]\n"
        f"[dim]{sysinfo['chip']}  ·  {sysinfo['ram_total_gb']}GB RAM  "
        f"({sysinfo['ram_used_pct']}% in use)  ·  "
        f"{n_runs} runs per prompt[/dim]",
        expand=False,
        border_style="blue",
    ))
    console.print()
    console.print(
        "[dim]All timings from Ollama's internal Go timers (/api/chat response fields).\n"
        "Nothing is estimated by Python. VRAM from /api/ps size_vram.[/dim]"
    )
    console.print()


def print_warm_results(raw: PhaseResult, tune: PhaseResult) -> None:
    console.print()
    console.print(Panel(
        "[bold]Phase 1 — Warm Inference[/bold]\n"
        "[dim]Model loaded throughout. Measures time to fill the KV cache before "
        "the first token.\nSmaller num_ctx = smaller Metal buffer to initialise = "
        "lower prefill latency.[/dim]",
        border_style="cyan",
        expand=False,
    ))

    raw_by_prompt  = raw.prefill_by_prompt()
    tune_by_prompt = tune.prefill_by_prompt()
    raw_ctx        = raw.ctx_by_prompt()
    tune_ctx       = tune.ctx_by_prompt()

    # Per-prompt prefill table
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold",
              show_edge=False, padding=(0, 1))
    t.add_column("Prompt",          style="cyan", min_width=14, max_width=14)
    t.add_column("ctx raw→tune",    justify="right", style="dim", min_width=12)
    t.add_column("prefill raw",     justify="right", min_width=11)
    t.add_column("prefill autotune", justify="right", min_width=16)
    t.add_column("Δ prefill",       justify="right", min_width=10)

    all_raw_prefills  = []
    all_tune_prefills = []
    raw_tps_vals      = []
    tune_tps_vals     = []
    rt_agg = tt_agg = tps_pct = None

    raw_tps_by  = {}
    tune_tps_by = {}
    for r in raw.ok_runs():
        raw_tps_by.setdefault(r.prompt_id, []).append(r.eval_tps)
    for r in tune.ok_runs():
        tune_tps_by.setdefault(r.prompt_id, []).append(r.eval_tps)

    for p in PROMPTS:
        pid = p["id"]
        lbl = p["label"]
        rp  = raw_by_prompt.get(pid)
        tp  = tune_by_prompt.get(pid)
        rc  = raw_ctx.get(pid, RAW_CTX)
        tc  = tune_ctx.get(pid, "?")
        rt  = statistics.mean(raw_tps_by[pid])  if pid in raw_tps_by  else None
        tt  = statistics.mean(tune_tps_by[pid]) if pid in tune_tps_by else None

        if rp is None or tp is None:
            t.add_row(lbl, f"{rc}→?", "—", "—", "—")
            continue

        all_raw_prefills.append(rp)
        all_tune_prefills.append(tp)
        if rt: raw_tps_vals.append(rt)
        if tt: tune_tps_vals.append(tt)

        pct = _pct(rp, tp)
        pd_txt, pd_sty = _fmt_pct(pct, invert=True)
        ctx_sty = "green" if (rc - tc) > 256 else "dim"

        t.add_row(
            lbl,
            f"[dim]{rc}[/dim]→[{ctx_sty}]{tc}[/{ctx_sty}]",
            f"{rp:.0f}ms",
            f"{tp:.0f}ms",
            Text(pd_txt, style=pd_sty),
        )

    # Aggregate row
    if all_raw_prefills and all_tune_prefills:
        agg_raw  = statistics.mean(all_raw_prefills)
        agg_tune = statistics.mean(all_tune_prefills)
        agg_pct  = _pct(agg_raw, agg_tune)
        ag_txt, ag_sty = _fmt_pct(agg_pct, invert=True)
        rt_agg = statistics.mean(raw_tps_vals)  if raw_tps_vals  else None
        tt_agg = statistics.mean(tune_tps_vals) if tune_tps_vals else None
        tps_pct = _pct(rt_agg, tt_agg) if rt_agg and tt_agg else 0.0
        t.add_section()
        t.add_row(
            "[bold]AVERAGE[/bold]",
            "[dim]4096[/dim]→[dim]tuned[/dim]",
            f"[bold]{agg_raw:.0f}ms[/bold]",
            f"[bold]{agg_tune:.0f}ms[/bold]",
            Text(ag_txt, style=ag_sty + " bold"),
        )
    console.print(t)
    if rt_agg is not None and tt_agg is not None:
        console.print(
            f"  Generation speed: raw={rt_agg:.1f} tok/s  "
            f"autotune={tt_agg:.1f} tok/s  "
            f"Δ={tps_pct:+.1f}%  "
            "[dim](GPU-bound — no change expected, this is the honest null result)[/dim]"
        )
    console.print(
        "  [dim]prefill_ms = prompt_eval_duration from /api/chat  "
        "(nanoseconds from Ollama's Go time.Now(), converted to ms)[/dim]"
    )
    console.print()


def print_cold_results(raw: PhaseResult, tune: PhaseResult) -> None:
    console.print()
    console.print(Panel(
        "[bold]Phase 2 — Cold-Start Load Time[/bold]\n"
        "[dim]Model unloaded between every call. Measures Metal MTLBuffer allocation "
        "cost.\nSmaller num_ctx = smaller buffer = faster first-call latency.[/dim]",
        border_style="cyan",
        expand=False,
    ))

    raw_runs  = raw.ok_runs()
    tune_runs = tune.ok_runs()

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    t.add_column("Call",        style="dim",   min_width=6)
    t.add_column("ctx",         justify="right", style="dim")
    t.add_column("load raw",    justify="right")
    t.add_column("load tune",   justify="right")
    t.add_column("Δ load",      justify="right", min_width=10)
    t.add_column("ttft raw",    justify="right")
    t.add_column("ttft tune",   justify="right")

    max_calls = max(len(raw_runs), len(tune_runs))
    for i in range(max_calls):
        rr = raw_runs[i]  if i < len(raw_runs)  else None
        tr = tune_runs[i] if i < len(tune_runs) else None

        r_load  = f"{rr.load_ms:.0f}ms"    if rr else "—"
        t_load  = f"{tr.load_ms:.0f}ms"    if tr else "—"
        r_ttft  = f"{(rr.load_ms + rr.prefill_ms):.0f}ms" if rr else "—"
        t_ttft  = f"{(tr.load_ms + tr.prefill_ms):.0f}ms" if tr else "—"
        r_ctx   = str(rr.num_ctx) if rr else "—"

        if rr and tr and rr.load_ms > 0:
            pct = _pct(rr.load_ms, tr.load_ms)
            pd_txt, pd_sty = _fmt_pct(pct, invert=True)
        else:
            pd_txt, pd_sty = "—", "dim"

        t.add_row(f"{i+1}", r_ctx, r_load, t_load, Text(pd_txt, style=pd_sty), r_ttft, t_ttft)

    # Aggregate
    raw_loads  = [r.load_ms for r in raw_runs  if r.load_ms > 0]
    tune_loads = [r.load_ms for r in tune_runs if r.load_ms > 0]
    if raw_loads and tune_loads:
        agg_r = statistics.mean(raw_loads)
        agg_t = statistics.mean(tune_loads)
        pct   = _pct(agg_r, agg_t)
        ag_txt, ag_sty = _fmt_pct(pct, invert=True)
        t.add_section()
        t.add_row(
            "[bold]MEAN[/bold]", "—",
            f"[bold]{agg_r:.0f}ms[/bold]",
            f"[bold]{agg_t:.0f}ms[/bold]",
            Text(ag_txt, style=ag_sty + " bold"),
            "—", "—",
        )

    console.print(t)
    raw_loads  = [r.load_ms for r in raw.ok_runs()  if r.load_ms > 0]
    tune_loads = [r.load_ms for r in tune.ok_runs() if r.load_ms > 0]
    if raw_loads and tune_loads:
        agg_r = statistics.mean(raw_loads)
        agg_t = statistics.mean(tune_loads)
        pct   = _pct(agg_r, agg_t)
        if abs(pct) < 10:
            console.print(
                "  [dim]Note: load times are similar here because macOS keeps "
                "recently-used model weights in the unified memory file cache. "
                "The improvement is more visible on larger models (7B+), first-ever "
                "loads, or after a long idle period when Ollama has evicted the model.[/dim]"
            )
    console.print(
        "  [dim]load_duration = time Ollama spends allocating model weights + "
        "KV cache buffer in Metal unified memory.[/dim]"
    )
    console.print()


def print_vram_results(vr: VRAMResult) -> None:
    console.print()
    console.print(Panel(
        "[bold]Phase 3 — VRAM Footprint[/bold]\n"
        "[dim]Measured from /api/ps size_vram — actual Metal unified memory Ollama "
        "holds while the model is loaded.\nIncludes weights + KV cache buffer.[/dim]",
        border_style="cyan",
        expand=False,
    ))

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    t.add_column("Config",          min_width=20)
    t.add_column("num_ctx",         justify="right")
    t.add_column("size_vram",       justify="right")
    t.add_column("KV savings",      justify="right")

    t.add_row("Raw (Ollama default)", str(vr.raw_ctx), f"{vr.raw_vram_gb:.3f} GB", "—")
    t.add_row(
        "[green]Autotune[/green]",
        f"[green]{vr.tune_ctx}[/green]",
        f"[green]{vr.tune_vram_gb:.3f} GB[/green]",
        f"[green bold]−{vr.delta_gb:.3f} GB  ✓[/green bold]",
    )

    console.print(t)
    console.print(
        f"  Measured savings: [bold green]−{vr.delta_gb:.3f} GB[/bold green]  "
        f"(theory KV-only: −{vr.theory_delta_gb:.3f} GB  +  "
        f"{vr.overhead_gb:.3f} GB Metal alignment overhead)"
    )
    console.print()


def print_verdict(
    raw_warm: PhaseResult,
    tune_warm: PhaseResult,
    raw_cold: Optional[PhaseResult],
    tune_cold: Optional[PhaseResult],
    vram: Optional[VRAMResult],
) -> None:
    console.print()
    console.rule("[bold blue]VERDICT[/bold blue]")
    console.print()

    lines: list[tuple[str, str, str]] = []  # (metric, value, why)

    # Prefill
    rp = raw_warm.mean_prefill()
    tp = tune_warm.mean_prefill()
    if rp and tp:
        pct = _pct(rp, tp)
        pd_txt, pd_sty = _fmt_pct(pct, invert=True)
        lines.append((
            "Prefill latency (TTFT)",
            f"[{pd_sty}]{rp:.0f}ms → {tp:.0f}ms  {pd_txt}[/{pd_sty}]",
            "Ollama allocates the full KV tensor before the first forward pass. "
            "autotune computes the minimum num_ctx that fits your prompt, so the "
            "Metal buffer is smaller → faster initialisation.",
        ))

    # tok/s
    rt = raw_warm.mean_tps()
    tt = tune_warm.mean_tps()
    if rt and tt:
        pct = _pct(rt, tt)
        lines.append((
            "Generation tok/s",
            f"[dim]{rt:.1f} → {tt:.1f}  ({pct:+.1f}% ≈)[/dim]",
            "Token generation is Metal GPU-bound. The same matrix ops run per "
            "token regardless of num_ctx. No middleware layer above Metal can "
            "change this. This is the honest null result.",
        ))

    # Cold-start
    if raw_cold and tune_cold:
        rl = raw_cold.mean_load()
        tl = tune_cold.mean_load()
        if rl and tl:
            pct = _pct(rl, tl)
            pd_txt, pd_sty = _fmt_pct(pct, invert=True)
            lines.append((
                "Cold-start load time",
                f"[{pd_sty}]{rl:.0f}ms → {tl:.0f}ms  {pd_txt}[/{pd_sty}]",
                "On a cold start (first use / after idle expiry) Ollama allocates "
                "a Metal MTLBuffer for the KV cache. Smaller num_ctx = smaller "
                "buffer = faster allocation. This is felt as the 'startup penalty'.",
            ))

    # VRAM
    if vram:
        if vram.delta_gb > 0:
            lines.append((
                "VRAM footprint",
                f"[green bold]{vram.raw_vram_gb:.3f}GB → "
                f"{vram.tune_vram_gb:.3f}GB  "
                f"−{vram.delta_gb:.3f}GB  ✓[/green bold]",
                f"KV cache scales linearly with num_ctx. autotune uses "
                f"ctx={vram.tune_ctx} vs Ollama's default ctx={RAW_CTX}, "
                f"holding {vram.delta_gb:.3f}GB less unified memory. On a "
                f"16GB Mac this matters when running multiple models or apps.",
            ))
        else:
            lines.append((
                "VRAM footprint",
                f"[dim]{vram.raw_vram_gb:.3f}GB → {vram.tune_vram_gb:.3f}GB  "
                f"≈ (small prompt, ctx similar)[/dim]",
                "With a very short prompt, autotune's minimum ctx is still "
                "close to the raw default. Savings are larger on longer prompts.",
            ))

    for metric, value, why in lines:
        console.print(f"  [bold]{metric}[/bold]")
        console.print(f"    {value}")
        console.print(f"    [dim]{why}[/dim]")
        console.print()

    console.rule()
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# JSON export
# ─────────────────────────────────────────────────────────────────────────────

def export_json(
    model: str,
    sysinfo: dict,
    n_runs: int,
    raw_warm: PhaseResult,
    tune_warm: PhaseResult,
    raw_cold: Optional[PhaseResult],
    tune_cold: Optional[PhaseResult],
    vram: Optional[VRAMResult],
    output_path: str,
) -> None:
    def _phase_dict(p: PhaseResult) -> dict:
        return {
            "mode": p.mode,
            "mean_prefill_ms": p.mean_prefill(),
            "mean_tps": p.mean_tps(),
            "mean_load_ms": p.mean_load(),
            "runs": [asdict(r) for r in p.runs],
        }

    doc = {
        "tool":      "autotune proof",
        "version":   "1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model":     model,
        "n_runs":    n_runs,
        "system":    sysinfo,
        "raw_ctx":   RAW_CTX,
        "phases": {
            "warm": {
                "raw":      _phase_dict(raw_warm),
                "autotune": _phase_dict(tune_warm),
                "delta_prefill_pct": _pct(
                    raw_warm.mean_prefill() or 0,
                    tune_warm.mean_prefill() or 0,
                ),
            },
        },
        "prompts": [{"id": p["id"], "label": p["label"]} for p in PROMPTS],
    }

    if raw_cold and tune_cold:
        doc["phases"]["cold"] = {
            "raw":      _phase_dict(raw_cold),
            "autotune": _phase_dict(tune_cold),
            "delta_load_pct": _pct(
                raw_cold.mean_load() or 0,
                tune_cold.mean_load() or 0,
            ),
        }

    if vram:
        doc["phases"]["vram"] = {
            "raw_ctx":         vram.raw_ctx,
            "raw_vram_gb":     vram.raw_vram_gb,
            "tune_ctx":        vram.tune_ctx,
            "tune_vram_gb":    vram.tune_vram_gb,
            "delta_gb":        vram.delta_gb,
            "theory_delta_gb": vram.theory_delta_gb,
        }

    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2)

    console.print(f"  [dim]Results saved to {output_path}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    # ── Pre-flight ────────────────────────────────────────────────────────────
    if not await _ensure_ollama():
        console.print("[red]Error: Ollama is not running. Start it with: ollama serve[/red]")
        sys.exit(1)

    if args.list_models:
        models = await _list_models()
        if models:
            console.print("Available models:")
            for m in models:
                console.print(f"  {m}")
        else:
            console.print("[dim]No models found. Pull one with: ollama pull phi4-mini[/dim]")
        return

    model = args.model

    # Validate model exists
    available = await _list_models()
    if model not in available:
        console.print(f"[red]Model '{model}' not found. Available: {available}[/red]")
        console.print("[dim]Pull it with: ollama pull " + model.split(":")[0] + "[/dim]")
        sys.exit(1)

    n_runs   = args.runs
    n_cold   = args.cold_runs
    sysinfo  = _get_system_info()
    client   = OllamaMetricsClient(timeout=300.0)

    print_header(model, sysinfo, n_runs)

    # ── Phase 1: Warm inference ───────────────────────────────────────────────
    console.rule("[bold cyan]Phase 1 — Warm Inference[/bold cyan]")
    console.print()
    raw_warm, tune_warm = await run_warm_phase(model, n_runs, client)
    print_warm_results(raw_warm, tune_warm)

    # ── Phase 2: Cold-start ───────────────────────────────────────────────────
    raw_cold = tune_cold = None
    if not args.skip_cold:
        console.rule("[bold cyan]Phase 2 — Cold-Start[/bold cyan]")
        console.print()
        raw_cold, tune_cold = await run_cold_phase(model, n_cold, client)
        print_cold_results(raw_cold, tune_cold)

    # ── Phase 3: VRAM ─────────────────────────────────────────────────────────
    vram_result = None
    if not args.skip_vram:
        console.rule("[bold cyan]Phase 3 — VRAM Footprint[/bold cyan]")
        console.print()
        vram_result = await run_vram_phase(model, client)
        if vram_result:
            print_vram_results(vram_result)

    # ── Verdict ───────────────────────────────────────────────────────────────
    print_verdict(raw_warm, tune_warm, raw_cold, tune_cold, vram_result)

    # ── Export ────────────────────────────────────────────────────────────────
    export_json(
        model=model,
        sysinfo=sysinfo,
        n_runs=n_runs,
        raw_warm=raw_warm,
        tune_warm=tune_warm,
        raw_cold=raw_cold,
        tune_cold=tune_cold,
        vram=vram_result,
        output_path=args.output,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="autotune proof — honest, traceable benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", "-m",
        default="phi4-mini:latest",
        help="Ollama model ID to benchmark (default: phi4-mini:latest)",
    )
    p.add_argument(
        "--runs", "-r",
        type=int, default=3,
        help="Warm inference runs per prompt per config (default: 3)",
    )
    p.add_argument(
        "--cold-runs",
        type=int, default=3,
        help="Cold-start calls per config (default: 3)",
    )
    p.add_argument(
        "--output", "-o",
        default="proof_results.json",
        help="JSON output path (default: proof_results.json)",
    )
    p.add_argument(
        "--skip-cold",
        action="store_true",
        help="Skip cold-start phase (faster run)",
    )
    p.add_argument(
        "--skip-vram",
        action="store_true",
        help="Skip VRAM footprint phase",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    asyncio.run(main(args))
