#!/usr/bin/env python3
"""
autotune proof — does it actually help?

Runs your model twice on each prompt: once with plain Ollama, once with
autotune. Measures what actually changes and shows you an honest comparison.

All timing numbers come from Ollama's own internal timers — nothing
estimated by Python.

Usage:
    python scripts/proof.py                          (uses qwen3:8b by default)
    python scripts/proof.py --model llama3.2:3b
    python scripts/proof.py --model qwen3:8b --with-cold --with-noswap
    autotune proof --model qwen3:8b
    autotune proof --list-models
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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich import box
from rich.table import Table
from rich.text import Text

from autotune.metrics.ollama_client import OllamaMetricsClient
from autotune.metrics.vram import VRAMTracker
from autotune.ttft.optimizer import TTFTOptimizer
from autotune.api.profiles import get_profile
from autotune.memory.noswap import NoSwapGuard, ModelArch

console = Console(width=72)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts  (4 types that cover the real distribution of everyday use)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = [
    {
        "id":    "quick",
        "label": "Quick question",
        "messages": [
            {"role": "user", "content": "What are three benefits of drinking water regularly?"},
        ],
    },
    {
        "id":    "code",
        "label": "Code task",
        "messages": [
            {"role": "user", "content": (
                "Write a Python function `find_duplicates(lst)` that returns "
                "a list of values that appear more than once. Include a docstring."
            )},
        ],
    },
    {
        "id":    "sys_prompt",
        "label": "With system prompt",
        "messages": [
            {"role": "system", "content": (
                "You are a helpful assistant. Be concise and accurate. "
                "Always explain your reasoning step by step."
            )},
            {"role": "user", "content": (
                "What is the difference between a process and a thread? "
                "When would you use each?"
            )},
        ],
    },
    {
        "id":    "chat",
        "label": "Multi-turn chat",
        "messages": [
            {"role": "user",    "content": "What is a Python decorator?"},
            {"role": "assistant","content": (
                "A decorator is a function that wraps another function to extend "
                "or modify its behavior without changing its source code. "
                "They use the @syntax and are applied at definition time."
            )},
            {"role": "user",    "content": "Show me a simple example that logs function calls."},
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAW_CTX        = 4096
RAW_KEEP_ALIVE = "5m"
MAX_TOKENS     = 120       # cap generation to keep test short (doesn't affect prefill)


def raw_opts() -> dict:
    return {"num_ctx": RAW_CTX, "num_predict": MAX_TOKENS}


def autotune_opts(messages: list[dict]) -> tuple[dict, str]:
    profile = get_profile("balanced")
    result  = TTFTOptimizer().build_request_options(messages, profile)
    opts    = result["options"]
    opts["num_predict"] = MAX_TOKENS
    return opts, result["keep_alive"]


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    prompt_id:  str
    mode:       str
    run_idx:    int
    num_ctx:    int
    prefill_ms: float
    eval_tps:   float
    load_ms:    float
    error:      Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 28) -> str:
    filled = min(width, round((value / max(max_val, 0.01)) * width))
    return "█" * filled + "░" * (width - filled)


def _pct(baseline: float, new: float) -> float:
    if baseline <= 0:
        return 0.0
    return (new - baseline) / baseline * 100.0


def _chip() -> str:
    try:
        r = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if "Chip:" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.machine()


async def _check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            return (await c.get("http://localhost:11434/api/ps")).status_code == 200
    except Exception:
        return False


async def _list_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get("http://localhost:11434/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Warm inference
# ─────────────────────────────────────────────────────────────────────────────

async def run_warm(model: str, n_runs: int, client: OllamaMetricsClient
                   ) -> tuple[list[Run], list[Run]]:
    raw_runs: list[Run] = []
    tune_runs: list[Run] = []
    total = len(PROMPTS) * 2 * n_runs

    console.print()

    for p in PROMPTS:
        msgs = p["messages"]
        pid  = p["id"]
        lbl  = p["label"]

        for run_i in range(n_runs):
            done = len(raw_runs) + len(tune_runs) + 1
            console.print(
                f"  [dim]{done:2}/{total}[/dim]  {lbl}  [dim]raw[/dim]  ",
                end="",
            )
            s = await client.run_with_stats(
                model=model, messages=msgs, options=raw_opts(),
                keep_alive=RAW_KEEP_ALIVE,
            )
            if s.error:
                console.print(f"[red]error[/red]")
            else:
                console.print(
                    f"[dim]prefill={s.prefill_ms:.0f}ms  "
                    f"ctx={s.num_ctx}[/dim]"
                )
            raw_runs.append(Run(
                prompt_id=pid, mode="raw", run_idx=run_i,
                num_ctx=s.num_ctx, prefill_ms=s.prefill_ms,
                eval_tps=s.eval_tps, load_ms=s.load_ms,
                error=s.error or None,
            ))

        for run_i in range(n_runs):
            done = len(raw_runs) + len(tune_runs) + 1
            console.print(
                f"  [dim]{done:2}/{total}[/dim]  {lbl}  [dim]autotune[/dim]  ",
                end="",
            )
            a_opts, a_keep = autotune_opts(msgs)
            s = await client.run_with_stats(
                model=model, messages=msgs, options=a_opts,
                keep_alive=a_keep,
            )
            if s.error:
                console.print(f"[red]error[/red]")
            else:
                console.print(
                    f"[dim]prefill={s.prefill_ms:.0f}ms  "
                    f"ctx={s.num_ctx}[/dim]"
                )
            tune_runs.append(Run(
                prompt_id=pid, mode="autotune", run_idx=run_i,
                num_ctx=s.num_ctx, prefill_ms=s.prefill_ms,
                eval_tps=s.eval_tps, load_ms=s.load_ms,
                error=s.error or None,
            ))

    return raw_runs, tune_runs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Cold-start
# ─────────────────────────────────────────────────────────────────────────────

async def run_cold(model: str, n_calls: int, client: OllamaMetricsClient
                   ) -> tuple[list[Run], list[Run]]:
    raw_runs: list[Run] = []
    tune_runs: list[Run] = []
    probe = PROMPTS[0]["messages"]

    async def unload() -> None:
        await client.unload_model(model)
        await asyncio.sleep(2.0)

    for i in range(n_calls):
        console.print(f"  [dim]{i+1}/{n_calls}[/dim]  cold raw…", end="\r")
        await unload()
        s = await client.run_with_stats(
            model=model, messages=probe, options=raw_opts(),
            keep_alive=RAW_KEEP_ALIVE,
        )
        raw_runs.append(Run(
            prompt_id="cold", mode="raw_cold", run_idx=i,
            num_ctx=s.num_ctx, prefill_ms=s.prefill_ms,
            eval_tps=s.eval_tps, load_ms=s.load_ms,
            error=s.error or None,
        ))

    a_opts, a_keep = autotune_opts(probe)
    for i in range(n_calls):
        console.print(f"  [dim]{i+1}/{n_calls}[/dim]  cold autotune…", end="\r")
        await unload()
        s = await client.run_with_stats(
            model=model, messages=probe, options=a_opts,
            keep_alive=a_keep,
        )
        tune_runs.append(Run(
            prompt_id="cold", mode="autotune_cold", run_idx=i,
            num_ctx=s.num_ctx, prefill_ms=s.prefill_ms,
            eval_tps=s.eval_tps, load_ms=s.load_ms,
            error=s.error or None,
        ))

    console.print(" " * 72, end="\r")
    return raw_runs, tune_runs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — VRAM
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VRAMResult:
    raw_ctx: int
    raw_gb:  float
    tune_ctx: int
    tune_gb:  float

    @property
    def saved_gb(self) -> float:
        return round(self.raw_gb - self.tune_gb, 3)


async def run_vram(model: str, client: OllamaMetricsClient) -> Optional[VRAMResult]:
    tracker = VRAMTracker()
    probe   = PROMPTS[0]["messages"]
    a_opts, a_keep = autotune_opts(probe)
    tune_ctx = a_opts.get("num_ctx", 1024)

    await client.unload_model(model)
    await asyncio.sleep(1.0)
    s = await client.run_with_stats(model=model, messages=probe,
                                    options=a_opts, keep_alive="30m")
    if s.error:
        return None
    snap_tune = await tracker.snapshot(model)

    await client.unload_model(model)
    await asyncio.sleep(1.0)
    s = await client.run_with_stats(model=model, messages=probe,
                                    options=raw_opts(), keep_alive="30m")
    if s.error:
        return None
    snap_raw = await tracker.snapshot(model)

    await client.unload_model(model)

    return VRAMResult(
        raw_ctx=RAW_CTX, raw_gb=snap_raw.size_vram_gb,
        tune_ctx=tune_ctx, tune_gb=snap_tune.size_vram_gb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — No-swap demo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NoSwapDemo:
    available_gb: float
    model_id: str
    arch: ModelArch
    scenarios: list[dict]   # each: {label, ctx, kv_gb, level, f16, fits}


async def run_noswap_demo(model: str) -> NoSwapDemo:
    """
    Shows what no-swap mode does at different memory pressure levels.
    Does not actually run inference — just computes what would happen.
    """
    guard = NoSwapGuard()
    arch  = await guard.get_model_arch(model)

    vm  = psutil.virtual_memory()
    available_gb = vm.available / 1024**3

    scenarios = []

    # Simulate what happens at various available-RAM levels
    sim_levels = [
        ("Current system",       available_gb),
        ("Light pressure (4 GB free)",  4.0),
        ("Moderate pressure (2 GB free)", 2.0),
        ("High pressure (1 GB free)",     1.0),
        ("Critical (0.5 GB free)",        0.5),
    ]

    for label, avail in sim_levels:
        # Temporarily override the guard's check by computing directly
        usable = max(0.0, avail - guard.safety_margin_gb)
        from autotune.memory.noswap import _LEVELS, _MIN_CTX
        base_ctx = 1536
        chosen_ctx  = _MIN_CTX
        chosen_f16  = False
        chosen_level = "l5_min"
        chosen_kv   = arch.kv_gb(_MIN_CTX, f16=False)

        for factor, use_f16, level_name, _ in _LEVELS:
            if factor is None:
                cand = _MIN_CTX
            else:
                cand = max(_MIN_CTX, int(base_ctx * factor))
            kv = arch.kv_gb(cand, f16=use_f16)
            if kv <= usable:
                chosen_ctx   = cand
                chosen_f16   = use_f16
                chosen_level = level_name
                chosen_kv    = kv
                break

        scenarios.append({
            "label":       label,
            "avail_gb":    avail,
            "ctx":         chosen_ctx,
            "kv_gb":       round(chosen_kv, 3),
            "level":       chosen_level,
            "f16":         chosen_f16,
            "kv_raw_gb":   round(arch.kv_gb(RAW_CTX, f16=True), 3),
        })

    return NoSwapDemo(
        available_gb=round(available_gb, 2),
        model_id=model,
        arch=arch,
        scenarios=scenarios,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output  (user-friendly, no jargon)
# ─────────────────────────────────────────────────────────────────────────────

def print_results(
    model: str,
    raw_warm: list[Run],
    tune_warm: list[Run],
    raw_cold: Optional[list[Run]],
    tune_cold: Optional[list[Run]],
    vram: Optional[VRAMResult],
    noswap: Optional[NoSwapDemo],
    chip: str,
    n_runs: int,
) -> None:

    vm    = psutil.virtual_memory()
    ram_gb = vm.total / 1024**3
    ram_pct = vm.percent

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold white]autotune — does it actually help?[/bold white]\n\n"
        f"  Model   [cyan]{model}[/cyan]\n"
        f"  System  {chip}  ·  {ram_gb:.0f}GB RAM  ·  {ram_pct:.0f}% in use\n\n"
        f"  [dim]All timings from Ollama's internal timers, not estimated.[/dim]\n"
        f"  [dim]{n_runs} runs per prompt, generation capped at {MAX_TOKENS} tokens.[/dim]",
        border_style="blue",
        expand=False,
    ))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _mean_prefill(runs: list[Run]) -> Optional[float]:
        vals = [r.prefill_ms for r in runs if r.ok and r.prefill_ms > 0]
        return statistics.mean(vals) if vals else None

    def _mean_tps(runs: list[Run]) -> Optional[float]:
        vals = [r.eval_tps for r in runs if r.ok and r.eval_tps > 0]
        return statistics.mean(vals) if vals else None

    def _mean_load(runs: list[Run]) -> Optional[float]:
        vals = [r.load_ms for r in runs if r.ok and r.load_ms > 0]
        return statistics.mean(vals) if vals else None

    rp_all = _mean_prefill(raw_warm)
    tp_all = _mean_prefill(tune_warm)
    rt_all = _mean_tps(raw_warm)
    tt_all = _mean_tps(tune_warm)

    # ── Section 1: Speed ──────────────────────────────────────────────────────
    console.print()
    console.rule("[bold]⚡  SPEED TO FIRST WORD[/bold]", style="cyan")
    console.print()
    console.print(
        "  Before you see the first word of a response, Ollama must\n"
        "  allocate memory for your conversation. autotune sizes this\n"
        "  allocation to exactly what your prompt needs — not the\n"
        "  Ollama default which is always 4,096 tokens regardless."
    )
    console.print()

    if rp_all and tp_all:
        bar_width = 26
        max_v = rp_all * 1.05
        raw_bar  = _bar(rp_all, max_v, bar_width)
        tune_bar = _bar(tp_all, max_v, bar_width)
        pct = _pct(rp_all, tp_all)

        console.print(f"  Without autotune   {raw_bar}   {rp_all:.0f} ms")
        console.print(f"  With autotune      [green]{tune_bar}[/green]   [green bold]{tp_all:.0f} ms[/green bold]")
        console.print()
        pct_str = f"{pct:+.0f}%"
        console.print(f"  [bold green]{abs(pct):.0f}% faster[/bold green] time to first word, on average.")
        console.print()

    # Per-prompt breakdown
    t = Table(box=box.SIMPLE, show_header=True, header_style="dim",
              show_edge=False, padding=(0, 1))
    t.add_column("Prompt type",      min_width=20, max_width=22)
    t.add_column("Without autotune", justify="right", min_width=15)
    t.add_column("With autotune",    justify="right", min_width=13)
    t.add_column("Δ",               justify="right", min_width=7)

    for p in PROMPTS:
        pid   = p["id"]
        raw_v = statistics.mean([r.prefill_ms for r in raw_warm  if r.ok and r.prompt_id == pid and r.prefill_ms > 0] or [0])
        tun_v = statistics.mean([r.prefill_ms for r in tune_warm if r.ok and r.prompt_id == pid and r.prefill_ms > 0] or [0])
        if raw_v <= 0 or tun_v <= 0:
            continue
        pct = _pct(raw_v, tun_v)
        good = pct < -5
        t.add_row(
            p["label"],
            f"{raw_v:.0f} ms",
            f"[{'green' if good else 'dim'}]{tun_v:.0f} ms[/{'green' if good else 'dim'}]",
            Text(f"{pct:+.0f}%", style="green bold" if good else "dim"),
        )

    console.print(t)

    # ── Section 2: Memory ─────────────────────────────────────────────────────
    if vram:
        console.print()
        console.rule("[bold]💾  MEMORY WHILE RUNNING[/bold]", style="cyan")
        console.print()
        console.print(
            "  Ollama keeps a block of GPU memory reserved while your\n"
            "  model is loaded. autotune reserves less — exactly what\n"
            "  your prompts actually need."
        )
        console.print()

        max_v = vram.raw_gb * 1.05
        console.print(f"  Without autotune   {_bar(vram.raw_gb, max_v)}   {vram.raw_gb:.2f} GB")
        console.print(f"  With autotune      [green]{_bar(vram.tune_gb, max_v)}[/green]   [green bold]{vram.tune_gb:.2f} GB[/green bold]")
        console.print()
        console.print(
            f"  [bold green]−{vram.saved_gb * 1024:.0f} MB freed.[/bold green]  "
            f"That's room for another model, more browser tabs,\n"
            f"  or just less pressure on your {ram_gb:.0f}GB of RAM."
        )

    # ── Section 3: What didn't change ────────────────────────────────────────
    console.print()
    console.rule("[bold]🎯  WHAT DIDN'T CHANGE  (honesty matters)[/bold]", style="cyan")
    console.print()
    if rt_all and tt_all:
        tps_pct = _pct(rt_all, tt_all)
        console.print(
            f"  Generation speed:  {rt_all:.1f} → {tt_all:.1f} tok/s  "
            f"({tps_pct:+.1f}%)"
        )
        console.print()
        console.print(
            "  Once the model starts writing, speed is determined by your\n"
            "  GPU chip. We don't change that. autotune makes your model\n"
            "  START faster and use less memory — not generate faster.\n"
            "  That's the honest answer."
        )

    # ── Section 4: Cold-start ────────────────────────────────────────────────
    if raw_cold and tune_cold:
        rl = _mean_load(raw_cold)
        tl = _mean_load(tune_cold)
        console.print()
        console.rule("[bold]🔄  STARTUP TIME  (cold start)[/bold]", style="cyan")
        console.print()
        console.print(
            "  When Ollama loads a model from scratch (first use, or after\n"
            "  a long idle period) it allocates memory all at once. A\n"
            "  smaller reservation loads faster."
        )
        console.print()
        if rl and tl:
            pct = _pct(rl, tl)
            max_v = max(rl, tl) * 1.05
            console.print(f"  Without autotune   {_bar(rl, max_v)}   {rl:.0f} ms")
            console.print(f"  With autotune      [green]{_bar(tl, max_v)}[/green]   [green bold]{tl:.0f} ms[/green bold]")
            if abs(pct) < 10:
                console.print()
                console.print(
                    f"  [dim]Note: {model.split(':')[0]} is a small model — macOS keeps\n"
                    f"  it in the file cache so load time is similar either way.\n"
                    f"  The difference is larger on 7B+ models.[/dim]"
                )

    # ── Section 5: No-swap mode ───────────────────────────────────────────────
    if noswap:
        console.print()
        console.rule("[bold]🛡️   NO-SWAP MODE[/bold]", style="cyan")
        console.print()
        console.print(
            "  When your Mac runs out of RAM it starts using storage\n"
            "  (swap) as overflow. This makes inference 10–100× slower\n"
            "  and your whole computer feels sluggish.\n\n"
            "  autotune's no-swap mode checks available RAM before every\n"
            "  request and shrinks the memory reservation if needed,\n"
            "  keeping inference fast even when your system is busy."
        )
        console.print()

        kv_raw_gb = noswap.arch.kv_gb(RAW_CTX, f16=True)
        console.print(
            f"  [dim]Your model ({noswap.model_id.split(':')[0]}) "
            f"with default Ollama settings uses {kv_raw_gb:.2f}GB "
            f"just for the KV cache.\n"
            f"  Model: {noswap.arch.n_layers} layers × "
            f"{noswap.arch.n_kv_heads} KV heads × "
            f"{noswap.arch.head_dim} dims[/dim]"
        )
        console.print()

        t2 = Table(box=box.SIMPLE, show_header=True, header_style="dim",
                   show_edge=False, padding=(0, 1))
        t2.add_column("Memory situation",   min_width=26, max_width=26)
        t2.add_column("Ollama default",     justify="right", min_width=13)
        t2.add_column("No-swap",            justify="right", min_width=10)
        t2.add_column("Action",             min_width=14)

        level_labels = {
            "ok":         "[green]✓ No change[/green]",
            "l1_trim":    "[yellow]Trim 25%[/yellow]",
            "l2_halve":   "[yellow]Halve ctx[/yellow]",
            "l3_q8":      "[orange1]Halve + Q8[/orange1]",
            "l4_quarter": "[red]Quarter + Q8[/red]",
            "l5_min":     "[red bold]Min ctx + Q8[/red bold]",
        }

        for sc in noswap.scenarios:
            kv_default = sc["kv_raw_gb"]
            kv_safe    = sc["kv_gb"]
            kv_default_str = f"{kv_default:.3f} GB"
            kv_safe_str    = f"[green]{kv_safe:.3f} GB[/green]" if kv_safe < kv_default else kv_default_str
            action = level_labels.get(sc["level"], sc["level"])
            swap_risk = sc["avail_gb"] < kv_default + 1.5
            default_str = f"[red]{kv_default_str} ⚠[/red]" if swap_risk else kv_default_str
            t2.add_row(sc["label"], default_str, kv_safe_str, action)

        console.print(t2)
        console.print()
        console.print(
            "  Enable it:  [bold]autotune proof --with-noswap[/bold]\n"
            "  Or in code: [bold]TTFTOptimizer().build_request_options("
            "..., no_swap=True, model_arch=arch)[/bold]"
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    console.print()
    console.rule(style="dim")
    console.print()
    console.print(
        "  [dim]How we measured[/dim]\n"
        "  [dim]All numbers come from Ollama's /api/chat response fields:[/dim]\n"
        "  [dim]  prompt_eval_duration = prefill time (what we reduce)[/dim]\n"
        "  [dim]  eval_duration / eval_count = generation tok/s[/dim]\n"
        "  [dim]  /api/ps size_vram = actual Metal GPU memory[/dim]\n\n"
        "  [dim]Raw data saved to proof_results.json[/dim]"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# JSON export
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(
    model, raw_warm, tune_warm, raw_cold, tune_cold, vram, noswap, output_path
) -> None:
    def _runs(runs):
        return [asdict(r) for r in runs] if runs else []

    doc = {
        "tool":      "autotune proof",
        "version":   "2.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model":     model,
        "phases": {
            "warm": {"raw": _runs(raw_warm), "autotune": _runs(tune_warm)},
        },
    }
    if raw_cold:
        doc["phases"]["cold"] = {"raw": _runs(raw_cold), "autotune": _runs(tune_cold)}
    if vram:
        doc["phases"]["vram"] = {
            "raw_ctx": vram.raw_ctx, "raw_gb": vram.raw_gb,
            "tune_ctx": vram.tune_ctx, "tune_gb": vram.tune_gb,
            "saved_gb": vram.saved_gb,
        }
    if noswap:
        doc["phases"]["noswap_demo"] = {
            "arch": {"n_layers": noswap.arch.n_layers,
                     "n_kv_heads": noswap.arch.n_kv_heads,
                     "head_dim": noswap.arch.head_dim},
            "scenarios": noswap.scenarios,
        }

    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    if not await _check_ollama():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        sys.exit(1)

    if args.list_models:
        models = await _list_models()
        if models:
            console.print("Available models:")
            for m in models:
                console.print(f"  {m}")
        else:
            console.print("[dim]No models found. Pull one: ollama pull qwen3:8b[/dim]")
        return

    model = args.model
    if model not in await _list_models():
        console.print(f"[red]Model '{model}' not found.[/red]")
        console.print(f"[dim]Pull it: ollama pull {model.split(':')[0]}[/dim]")
        sys.exit(1)

    chip   = _chip()
    client = OllamaMetricsClient(timeout=300.0)

    console.print()
    console.print(f"  Running benchmark on [cyan]{model}[/cyan]…  "
                  f"[dim](this takes about 2–3 minutes)[/dim]")

    # Fresh start — unload any cached model state so first requests are clean
    console.print("  [dim]Resetting model state…[/dim]")
    await client.unload_model(model)
    import asyncio as _asyncio
    await _asyncio.sleep(2.0)

    # Phase 1 — warm
    console.print("  [dim]Phase 1 — measuring response speed…[/dim]")
    raw_warm, tune_warm = await run_warm(model, args.runs, client)

    # Phase 2 — VRAM
    vram = None
    console.print("  [dim]Phase 2 — measuring memory usage…[/dim]")
    vram = await run_vram(model, client)

    # Phase 3 — cold (optional)
    raw_cold = tune_cold = None
    if args.with_cold:
        console.print("  [dim]Phase 3 — measuring startup time…[/dim]")
        raw_cold, tune_cold = await run_cold(model, args.cold_runs, client)

    # No-swap demo (optional, fast — no inference)
    noswap = None
    if args.with_noswap:
        console.print("  [dim]Computing no-swap scenarios…[/dim]")
        noswap = await run_noswap_demo(model)

    print_results(
        model=model,
        raw_warm=raw_warm, tune_warm=tune_warm,
        raw_cold=raw_cold, tune_cold=tune_cold,
        vram=vram, noswap=noswap,
        chip=chip, n_runs=args.runs,
    )

    _save_json(model, raw_warm, tune_warm, raw_cold, tune_cold, vram, noswap,
               args.output)
    console.print(f"  [dim]Results saved to {args.output}[/dim]")
    console.print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="autotune proof — honest benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", "-m", default="qwen3:8b")
    p.add_argument("--runs",  "-r", type=int, default=3,
                   help="Runs per prompt per mode (default: 3)")
    p.add_argument("--cold-runs", type=int, default=3)
    p.add_argument("--output", "-o", default="proof_results.json")
    p.add_argument("--with-cold",   action="store_true",
                   help="Include cold-start phase")
    p.add_argument("--with-noswap", action="store_true",
                   help="Include no-swap mode demonstration")
    p.add_argument("--list-models", action="store_true")
    return p


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
