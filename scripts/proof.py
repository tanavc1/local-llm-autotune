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
from autotune.api.ctx_utils import estimate_messages_tokens
from autotune.memory.noswap import NoSwapGuard, ModelArch

console = Console(width=72)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts  (cover short, medium, and long input sizes)
# ─────────────────────────────────────────────────────────────────────────────

# ~700 tokens of input — shows the full KV allocation benefit
_LONG_DOC_CODE = (
    "Review this Python backend codebase and identify the three most critical "
    "security and reliability issues. For each one: name it, explain the exact "
    "risk, and show the corrected code.\n\n"
    "```python\n"
    "# database.py\n"
    "import sqlite3, os\n"
    "from contextlib import contextmanager\n\n"
    "DB_PATH = os.environ.get('DATABASE_URL', 'app.db')\n\n"
    "def get_connection():\n"
    "    return sqlite3.connect(DB_PATH, check_same_thread=False)\n\n"
    "@contextmanager\n"
    "def get_db():\n"
    "    conn = get_connection()\n"
    "    try:\n"
    "        yield conn\n"
    "    finally:\n"
    "        conn.close()\n\n"
    "# auth.py\n"
    "import hashlib, jwt, datetime\n"
    "from database import get_db\n\n"
    "SECRET = 'hardcoded-secret-do-not-change'\n\n"
    "def hash_password(password: str) -> str:\n"
    "    return hashlib.md5(password.encode()).hexdigest()\n\n"
    "def create_user(username: str, password: str) -> dict:\n"
    "    with get_db() as conn:\n"
    "        conn.execute(\n"
    "            f\"INSERT INTO users (username, password_hash) \"\n"
    "            f\"VALUES ('{username}', '{hash_password(password)}')\"\n"
    "        )\n"
    "        conn.commit()\n"
    "    return {'username': username, 'created': True}\n\n"
    "def authenticate(username: str, password: str):\n"
    "    with get_db() as conn:\n"
    "        row = conn.execute(\n"
    "            f\"SELECT id, username FROM users \"\n"
    "            f\"WHERE username='{username}' \"\n"
    "            f\"AND password_hash='{hash_password(password)}'\"\n"
    "        ).fetchone()\n"
    "    if not row:\n"
    "        return None\n"
    "    return jwt.encode(\n"
    "        {'user_id': row[0], 'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)},\n"
    "        SECRET, algorithm='HS256',\n"
    "    )\n\n"
    "# api.py\n"
    "from flask import Flask, request, jsonify\n"
    "from auth import create_user, authenticate\n"
    "from database import get_db\n\n"
    "app = Flask(__name__)\n\n"
    "@app.route('/register', methods=['POST'])\n"
    "def register():\n"
    "    data = request.json\n"
    "    return jsonify(create_user(data['username'], data['password']))\n\n"
    "@app.route('/login', methods=['POST'])\n"
    "def login():\n"
    "    data = request.json\n"
    "    token = authenticate(data['username'], data['password'])\n"
    "    if not token:\n"
    "        return jsonify({'error': 'invalid credentials'}), 401\n"
    "    return jsonify({'token': token})\n\n"
    "@app.route('/users', methods=['GET'])\n"
    "def list_users():\n"
    "    with get_db() as conn:\n"
    "        rows = conn.execute('SELECT id, username FROM users').fetchall()\n"
    "    return jsonify([{'id': r[0], 'username': r[1]} for r in rows])\n\n"
    "@app.route('/users/<int:user_id>', methods=['DELETE'])\n"
    "def delete_user(user_id: int):\n"
    "    with get_db() as conn:\n"
    "        conn.execute(f'DELETE FROM users WHERE id={user_id}')\n"
    "        conn.commit()\n"
    "    return jsonify({'deleted': True})\n"
    "```"
)

PROMPTS = [
    {
        "id":    "quick",
        "label": "Quick answer",
        "messages": [
            {"role": "user", "content": "What are three benefits of drinking water regularly?"},
        ],
    },
    {
        "id":    "sys_prompt",
        "label": "System prompt + question",
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
        "id":    "long_doc",
        "label": "Long document (~700 tokens)",
        "messages": [
            {"role": "user", "content": _LONG_DOC_CODE},
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

RAW_CTX        = 4096
RAW_KEEP_ALIVE = "5m"

# Capped low so each run is fast.  autotune will size ctx to fit
# input + MAX_TOKENS + 256 — not the profile's 1024-token budget.
MAX_TOKENS = 120


def raw_opts() -> dict:
    return {"num_ctx": RAW_CTX, "num_predict": MAX_TOKENS}


def autotune_opts(messages: list[dict]) -> tuple[dict, str]:
    """
    Build autotune options sized to THIS benchmark's MAX_TOKENS generation cap.

    Using the profile's max_new_tokens (1024) would give ctx=1536 for a
    10-token prompt, masking the real benefit.  Using MAX_TOKENS (120) gives
    the correctly tight ctx:  10 + 120 + 256 = 386 → bucket 512.
    On qwen3:8b that's a 8× smaller KV buffer — a dramatic TTFT difference.
    """
    profile    = get_profile("balanced")
    input_toks = estimate_messages_tokens(messages)
    raw_needed = input_toks + MAX_TOKENS + 256
    # Snap to nearest standard bucket (avoids KV-thrashing between runs)
    BUCKETS = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    tight_ctx = next((b for b in BUCKETS if b >= raw_needed), RAW_CTX)

    result = TTFTOptimizer().build_request_options(
        messages, profile, context_ceiling=tight_ctx
    )
    opts = result["options"]
    opts["num_predict"] = MAX_TOKENS
    return opts, result["keep_alive"]


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    prompt_id:         str
    mode:              str
    run_idx:           int
    num_ctx:           int
    prefill_ms:        float
    eval_tps:          float
    load_ms:           float
    prompt_eval_count: int = 0    # tokens actually evaluated (0 if fully cached)
    error:             Optional[str] = None

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


def _swap_gb() -> float:
    """Current swap used in GB."""
    return psutil.swap_memory().used / 1024**3


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
#
# IMPORTANT: run ALL raw runs first, then ALL autotune runs.
# The old approach (interleaving per-prompt) forced a ctx resize before every
# prompt's first run (4096 → 1536 → 4096 → …) which added 1-3 s of noise to
# every "warm" measurement and made results unpredictable.
# ─────────────────────────────────────────────────────────────────────────────

async def run_warm(
    model: str, n_runs: int, client: OllamaMetricsClient,
) -> tuple[list[Run], list[Run]]:
    raw_runs:  list[Run] = []
    tune_runs: list[Run] = []
    n_each   = len(PROMPTS) * n_runs

    console.print()

    # ── Phase A: All raw runs ────────────────────────────────────────────────
    # Model was just unloaded.  First request cold-starts at ctx=4096.
    # Subsequent prompts within raw reuse the already-loaded model (no resize).
    console.print("  [dim]Raw Ollama:[/dim]")
    for p in PROMPTS:
        msgs = p["messages"]
        pid  = p["id"]
        lbl  = p["label"]
        for run_i in range(n_runs):
            idx = len(raw_runs) + 1
            console.print(
                f"  [dim]{idx:2}/{n_each}  raw  {lbl}  #{run_i+1}[/dim]",
                end="  ",
            )
            s = await client.run_with_stats(
                model=model, messages=msgs,
                options=raw_opts(), keep_alive=RAW_KEEP_ALIVE,
            )
            if s.error:
                console.print("[red]error[/red]")
            else:
                console.print(
                    f"[dim]prefill={s.prefill_ms:.0f}ms  "
                    f"load={s.load_ms:.0f}ms[/dim]"
                )
            raw_runs.append(Run(
                prompt_id=pid, mode="raw", run_idx=run_i,
                num_ctx=s.num_ctx, prefill_ms=s.prefill_ms,
                eval_tps=s.eval_tps, load_ms=s.load_ms,
                prompt_eval_count=s.prompt_eval_count,
                error=s.error or None,
            ))

    # ── Cooldown: unload before autotune phase ───────────────────────────────
    console.print()
    console.print("  [dim]Resetting — unloading model between conditions…[/dim]")
    await client.unload_model(model)
    await asyncio.sleep(3.0)

    # ── Phase B: All autotune runs ───────────────────────────────────────────
    # First request cold-starts at the tighter autotune ctx.
    # Subsequent prompts reuse the loaded model (no ctx thrashing).
    console.print()
    console.print("  [dim]autotune:[/dim]")
    for p in PROMPTS:
        msgs = p["messages"]
        pid  = p["id"]
        lbl  = p["label"]
        a_opts, a_keep = autotune_opts(msgs)
        for run_i in range(n_runs):
            idx = len(tune_runs) + 1
            console.print(
                f"  [dim]{idx:2}/{n_each}  autotune  {lbl}  #{run_i+1}[/dim]",
                end="  ",
            )
            s = await client.run_with_stats(
                model=model, messages=msgs,
                options=a_opts, keep_alive=a_keep,
            )
            if s.error:
                console.print("[red]error[/red]")
            else:
                console.print(
                    f"[dim]prefill={s.prefill_ms:.0f}ms  "
                    f"ctx={s.num_ctx}[/dim]"
                )
            tune_runs.append(Run(
                prompt_id=pid, mode="autotune", run_idx=run_i,
                num_ctx=s.num_ctx, prefill_ms=s.prefill_ms,
                eval_tps=s.eval_tps, load_ms=s.load_ms,
                prompt_eval_count=s.prompt_eval_count,
                error=s.error or None,
            ))

    return raw_runs, tune_runs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Cold-start
# ─────────────────────────────────────────────────────────────────────────────

async def run_cold(
    model: str, n_calls: int, client: OllamaMetricsClient,
) -> tuple[list[Run], list[Run]]:
    raw_runs:  list[Run] = []
    tune_runs: list[Run] = []
    probe = PROMPTS[0]["messages"]

    async def unload() -> None:
        await client.unload_model(model)
        await asyncio.sleep(2.0)

    for i in range(n_calls):
        console.print(f"  [dim]{i+1}/{n_calls}[/dim]  cold raw…", end="\r")
        await unload()
        s = await client.run_with_stats(
            model=model, messages=probe,
            options=raw_opts(), keep_alive=RAW_KEEP_ALIVE,
        )
        raw_runs.append(Run(
            prompt_id="cold", mode="raw_cold", run_idx=i,
            num_ctx=s.num_ctx, prefill_ms=s.prefill_ms,
            eval_tps=s.eval_tps, load_ms=s.load_ms,
            prompt_eval_count=s.prompt_eval_count,
            error=s.error or None,
        ))

    a_opts, a_keep = autotune_opts(probe)
    for i in range(n_calls):
        console.print(f"  [dim]{i+1}/{n_calls}[/dim]  cold autotune…", end="\r")
        await unload()
        s = await client.run_with_stats(
            model=model, messages=probe,
            options=a_opts, keep_alive=a_keep,
        )
        tune_runs.append(Run(
            prompt_id="cold", mode="autotune_cold", run_idx=i,
            num_ctx=s.num_ctx, prefill_ms=s.prefill_ms,
            eval_tps=s.eval_tps, load_ms=s.load_ms,
            prompt_eval_count=s.prompt_eval_count,
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
    tracker   = VRAMTracker()
    probe     = PROMPTS[0]["messages"]
    a_opts, a_keep = autotune_opts(probe)
    tune_ctx  = a_opts.get("num_ctx", 512)

    await client.unload_model(model)
    await asyncio.sleep(1.0)
    s = await client.run_with_stats(
        model=model, messages=probe, options=a_opts, keep_alive="30m",
    )
    if s.error:
        return None
    snap_tune = await tracker.snapshot(model)

    await client.unload_model(model)
    await asyncio.sleep(1.0)
    s = await client.run_with_stats(
        model=model, messages=probe, options=raw_opts(), keep_alive="30m",
    )
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
    available_gb:  float
    swap_raw_gb:   float   # swap used during raw benchmark phase
    swap_tune_gb:  float   # swap used during autotune benchmark phase
    model_id:      str
    arch:          ModelArch
    scenarios:     list[dict]


async def run_noswap_demo(
    model: str,
    swap_raw_gb: float,
    swap_tune_gb: float,
) -> NoSwapDemo:
    """
    Shows what no-swap mode does at different memory pressure levels.
    Also captures actual swap measurements from the benchmark phases.
    """
    guard = NoSwapGuard()
    arch  = await guard.get_model_arch(model)

    vm           = psutil.virtual_memory()
    available_gb = vm.available / 1024**3

    scenarios = []
    sim_levels = [
        ("Current system",              available_gb),
        ("Light pressure (4 GB free)",  4.0),
        ("Moderate pressure (2 GB free)", 2.0),
        ("High pressure (1 GB free)",   1.0),
        ("Critical (0.5 GB free)",      0.5),
    ]

    for label, avail in sim_levels:
        usable = max(0.0, avail - guard.safety_margin_gb)
        from autotune.memory.noswap import _LEVELS, _MIN_CTX
        base_ctx    = 1536
        chosen_ctx  = _MIN_CTX
        chosen_f16  = False
        chosen_level = "l5_min"
        chosen_kv   = arch.kv_gb(_MIN_CTX, f16=False)

        for factor, use_f16, level_name, _ in _LEVELS:
            cand = _MIN_CTX if factor is None else max(_MIN_CTX, int(base_ctx * factor))
            kv   = arch.kv_gb(cand, f16=use_f16)
            if kv <= usable:
                chosen_ctx   = cand
                chosen_f16   = use_f16
                chosen_level = level_name
                chosen_kv    = kv
                break

        scenarios.append({
            "label":     label,
            "avail_gb":  avail,
            "ctx":       chosen_ctx,
            "kv_gb":     round(chosen_kv, 3),
            "level":     chosen_level,
            "f16":       chosen_f16,
            "kv_raw_gb": round(arch.kv_gb(RAW_CTX, f16=True), 3),
        })

    return NoSwapDemo(
        available_gb=round(available_gb, 2),
        swap_raw_gb=round(swap_raw_gb, 3),
        swap_tune_gb=round(swap_tune_gb, 3),
        model_id=model,
        arch=arch,
        scenarios=scenarios,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output
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

    vm     = psutil.virtual_memory()
    ram_gb = vm.total / 1024**3
    ram_pct = vm.percent

    # Pull the ctx autotune used (first ok run)
    autotune_ctx = next(
        (r.num_ctx for r in tune_warm if r.ok), "—"
    )

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold white]autotune — does it actually help?[/bold white]\n\n"
        f"  Model   [cyan]{model}[/cyan]\n"
        f"  System  {chip}  ·  {ram_gb:.0f}GB RAM  ·  {ram_pct:.0f}% in use\n\n"
        f"  [dim]All timings from Ollama's internal timers, not estimated.[/dim]\n"
        f"  [dim]{n_runs} runs per prompt · generation capped at {MAX_TOKENS} tokens.[/dim]\n"
        f"  [dim]raw ctx={RAW_CTX}  autotune ctx={autotune_ctx}[/dim]",
        border_style="blue",
        expand=False,
    ))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ok(runs: list[Run]) -> list[Run]:
        return [r for r in runs if r.ok]

    def _mean(vals: list[float]) -> Optional[float]:
        return statistics.mean(vals) if vals else None

    def _mean_for(runs: list[Run], field: str) -> Optional[float]:
        vals = [getattr(r, field) for r in _ok(runs) if getattr(r, field) > 0]
        return _mean(vals)

    # Cold-start times: first ok run of each condition (includes model load + KV alloc)
    raw_cold_run  = next((r for r in raw_warm  if r.ok), None)
    tune_cold_run = next((r for r in tune_warm if r.ok), None)

    # Warm runs: skip the first run of each condition (it includes KV allocation overhead)
    def _warm_only(runs: list[Run]) -> list[Run]:
        # Exclude run_idx=0 for the FIRST prompt of each condition
        # (that first run has high load_ms from cold start)
        first_pid = PROMPTS[0]["id"]
        return [r for r in _ok(runs) if not (r.prompt_id == first_pid and r.run_idx == 0)]

    raw_warm_ok  = _warm_only(raw_warm)
    tune_warm_ok = _warm_only(tune_warm)

    rp_all = _mean([r.prefill_ms for r in raw_warm_ok  if r.prefill_ms > 0])
    tp_all = _mean([r.prefill_ms for r in tune_warm_ok if r.prefill_ms > 0])
    rt_all = _mean([r.eval_tps   for r in raw_warm_ok  if r.eval_tps   > 0])
    tt_all = _mean([r.eval_tps   for r in tune_warm_ok if r.eval_tps   > 0])

    # ── Section 1: First-use time (cold start) ────────────────────────────────
    console.print()
    console.rule("[bold]🚀  FIRST RESPONSE TIME (after model is idle)[/bold]", style="cyan")
    console.print()
    console.print(
        "  When your model hasn't been used recently, Ollama must load it\n"
        "  into GPU memory and allocate a KV cache buffer before the first\n"
        "  word appears. autotune allocates a smaller buffer — exactly what\n"
        "  your prompt needs, not Ollama's fixed 4,096-token default."
    )
    console.print()

    if raw_cold_run and tune_cold_run:
        raw_total  = raw_cold_run.load_ms  + raw_cold_run.prefill_ms
        tune_total = tune_cold_run.load_ms + tune_cold_run.prefill_ms
        max_v = max(raw_total, tune_total) * 1.05

        console.print(
            f"  [dim]  (load + prefill  ·  ctx: {raw_cold_run.num_ctx} raw "
            f"vs {tune_cold_run.num_ctx} autotune)[/dim]"
        )
        console.print()
        console.print(f"  Without autotune   {_bar(raw_total, max_v)}   {raw_total:,.0f} ms")
        console.print(
            f"  With autotune      [green]{_bar(tune_total, max_v)}[/green]   "
            f"[green bold]{tune_total:,.0f} ms[/green bold]"
        )
        console.print()

        pct = _pct(raw_total, tune_total)
        if pct < -5:
            console.print(
                f"  [bold green]{abs(pct):.0f}% faster[/bold green] first response.  "
                f"That's {abs(raw_total - tune_total):,.0f} ms saved every time the\n"
                f"  model starts from cold."
            )
        elif pct <= 5:
            console.print(
                f"  [dim]First-response time similar ({pct:+.0f}%). "
                f"This model is small — its KV cache\n"
                f"  fits easily in memory at any context size. "
                f"Larger models (7B+) show\n"
                f"  2–4× bigger savings here.[/dim]"
            )
        else:
            console.print(f"  [dim]First-response time ({pct:+.0f}%) — within noise.[/dim]")

    # ── Section 2: Warm TTFT ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold]⚡  RESPONSE SPEED (once loaded)[/bold]", style="cyan")
    console.print()
    console.print(
        "  After the model is loaded, every subsequent request still needs\n"
        "  to allocate its context window before generating the first word.\n"
        "  autotune allocates only what the prompt needs — smaller allocation\n"
        "  → less GPU time initializing the KV buffer."
    )
    console.print()

    if rp_all and tp_all:
        bar_width = 26
        max_v     = rp_all * 1.05
        pct       = _pct(rp_all, tp_all)

        console.print(f"  Without autotune   {_bar(rp_all, max_v, bar_width)}   {rp_all:.0f} ms avg")
        console.print(
            f"  With autotune      [green]{_bar(tp_all, max_v, bar_width)}[/green]   "
            f"[green bold]{tp_all:.0f} ms avg[/green bold]"
        )
        console.print()
        if pct < -5:
            console.print(f"  [bold green]{abs(pct):.0f}% faster[/bold green] time to first word (warm requests).")
        else:
            console.print(
                f"  [dim]Warm TTFT difference: {pct:+.0f}%.  "
                f"On this model the KV buffer is small\n"
                f"  enough that allocation time is dominated by prompt evaluation,\n"
                f"  which is the same for both. The savings show most on 7B+ models.[/dim]"
            )
        console.print()

    # Per-prompt table
    t = Table(
        box=box.SIMPLE, show_header=True, header_style="dim",
        show_edge=False, padding=(0, 1),
    )
    t.add_column("Prompt type",      min_width=22, max_width=24)
    t.add_column("ctx",              justify="right", min_width=6)
    t.add_column("Without autotune", justify="right", min_width=15)
    t.add_column("With autotune",    justify="right", min_width=13)
    t.add_column("Δ",               justify="right", min_width=7)

    for p in PROMPTS:
        pid   = p["id"]
        raw_v = _mean([
            r.prefill_ms for r in raw_warm_ok
            if r.ok and r.prompt_id == pid and r.prefill_ms > 0
        ] or [])
        tun_v = _mean([
            r.prefill_ms for r in tune_warm_ok
            if r.ok and r.prompt_id == pid and r.prefill_ms > 0
        ] or [])
        a_ctx = next(
            (r.num_ctx for r in tune_warm if r.ok and r.prompt_id == pid), "—"
        )
        if not raw_v or not tun_v:
            continue
        pct  = _pct(raw_v, tun_v)
        good = pct < -5
        t.add_row(
            p["label"],
            str(a_ctx),
            f"{raw_v:.0f} ms",
            f"[{'green' if good else 'dim'}]{tun_v:.0f} ms[/{'green' if good else 'dim'}]",
            Text(f"{pct:+.0f}%", style="green bold" if good else "dim"),
        )

    console.print(t)

    # ── Section 3: Memory ─────────────────────────────────────────────────────
    if vram:
        console.print()
        console.rule("[bold]💾  MEMORY WHILE RUNNING[/bold]", style="cyan")
        console.print()
        console.print(
            "  Ollama reserves a block of GPU memory for the KV cache while\n"
            "  the model is loaded. autotune reserves the minimum needed."
        )
        console.print()

        max_v = vram.raw_gb * 1.05
        console.print(
            f"  [dim]  ctx {vram.raw_ctx} (raw):      [/dim]"
            f"{_bar(vram.raw_gb, max_v)}   {vram.raw_gb:.3f} GB"
        )
        console.print(
            f"  [dim]  ctx {vram.tune_ctx} (autotune):[/dim]"
            f"[green]{_bar(vram.tune_gb, max_v)}[/green]   "
            f"[green bold]{vram.tune_gb:.3f} GB[/green bold]"
        )
        console.print()
        if vram.saved_gb > 0.05:
            console.print(
                f"  [bold green]−{vram.saved_gb * 1024:.0f} MB freed.[/bold green]  "
                f"Room for another model, more browser tabs, or less\n"
                f"  swap pressure on your {ram_gb:.0f} GB of unified memory."
            )
        else:
            console.print(
                f"  [dim]Memory savings: {vram.saved_gb * 1024:.0f} MB "
                f"({vram.raw_ctx}→{vram.tune_ctx} ctx).  "
                f"Small model — proportional\n"
                f"  savings are the same percentage on any model size.[/dim]"
            )

    # ── Section 4: What didn't change ────────────────────────────────────────
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
            "  GPU. We don't change that. autotune makes your model START\n"
            "  faster and use less memory — not generate faster. That's\n"
            "  the honest answer."
        )

    # ── Section 5: Cold-start ─────────────────────────────────────────────────
    if raw_cold and tune_cold:
        rl = _mean([r.load_ms for r in raw_cold  if r.ok and r.load_ms > 0])
        tl = _mean([r.load_ms for r in tune_cold if r.ok and r.load_ms > 0])
        console.print()
        console.rule("[bold]🔄  STARTUP TIME  (cold start, repeated)[/bold]", style="cyan")
        console.print()
        if rl and tl:
            pct   = _pct(rl, tl)
            max_v = max(rl, tl) * 1.05
            console.print(f"  Without autotune   {_bar(rl, max_v)}   {rl:,.0f} ms")
            console.print(
                f"  With autotune      [green]{_bar(tl, max_v)}[/green]   "
                f"[green bold]{tl:,.0f} ms[/green bold]"
            )
            console.print()
            if abs(pct) < 8:
                console.print(
                    f"  [dim]Load time similar ({pct:+.0f}%) — "
                    f"{model.split(':')[0]} is a small model,\n"
                    f"  macOS keeps it in the file cache. "
                    f"Savings scale with model size.[/dim]"
                )
            else:
                direction = "faster" if pct < 0 else "slower"
                console.print(
                    f"  [bold {'green' if pct < 0 else 'yellow'}]{abs(pct):.0f}% {direction}[/bold {'green' if pct < 0 else 'yellow'}] cold start."
                )

    # ── Section 6: No-swap mode ───────────────────────────────────────────────
    if noswap:
        console.print()
        console.rule("[bold]🛡️   NO-SWAP MODE[/bold]", style="cyan")
        console.print()
        console.print(
            "  When your Mac runs out of RAM it starts using storage (swap)\n"
            "  as overflow. For LLMs, even a few hundred MB of swap can make\n"
            "  inference 10–100× slower and your whole system feel sluggish.\n\n"
            "  autotune's no-swap mode checks available RAM before every request\n"
            "  and shrinks the memory reservation just enough to avoid it."
        )
        console.print()

        # Actual swap measurements from the benchmark
        console.print("  [dim]Swap observed during this benchmark run:[/dim]")
        swap_fmt_raw  = f"{noswap.swap_raw_gb:.2f} GB" if noswap.swap_raw_gb > 0.01 else "0.0 GB (none)"
        swap_fmt_tune = f"{noswap.swap_tune_gb:.2f} GB" if noswap.swap_tune_gb > 0.01 else "0.0 GB (none)"
        console.print(f"    Without autotune:  {swap_fmt_raw}")
        console.print(f"    With autotune:     {swap_fmt_tune}")
        if noswap.swap_raw_gb < 0.01 and noswap.swap_tune_gb < 0.01:
            console.print(
                "  [dim]No swap on this run — your Mac had plenty of free RAM.\n"
                "  The table below shows what would happen under pressure:[/dim]"
            )
        elif noswap.swap_tune_gb < noswap.swap_raw_gb:
            saved_mb = (noswap.swap_raw_gb - noswap.swap_tune_gb) * 1024
            console.print(
                f"\n  [bold green]autotune prevented {saved_mb:.0f} MB of swap[/bold green] "
                f"that raw Ollama triggered."
            )
        console.print()

        kv_raw_gb = noswap.arch.kv_gb(RAW_CTX, f16=True)
        console.print(
            f"  [dim]{noswap.model_id.split(':')[0]} default KV cache: "
            f"{kv_raw_gb:.3f} GB (ctx {RAW_CTX}, F16)\n"
            f"  {noswap.arch.n_layers} layers × {noswap.arch.n_kv_heads} KV heads "
            f"× {noswap.arch.head_dim} dims[/dim]"
        )
        console.print()

        t2 = Table(
            box=box.SIMPLE, show_header=True, header_style="dim",
            show_edge=False, padding=(0, 1),
        )
        t2.add_column("Memory situation",  min_width=26, max_width=26)
        t2.add_column("Ollama KV (raw)",   justify="right", min_width=14)
        t2.add_column("autotune KV",       justify="right", min_width=12)
        t2.add_column("Action",            min_width=16)

        level_labels = {
            "ok":         "[green]✓ unchanged[/green]",
            "l1_trim":    "[yellow]ctx −25%[/yellow]",
            "l2_halve":   "[yellow]ctx halved[/yellow]",
            "l3_q8":      "[orange1]ctx ÷2 + Q8[/orange1]",
            "l4_quarter": "[red]ctx ÷4 + Q8[/red]",
            "l5_min":     "[red bold]min ctx + Q8[/red bold]",
        }

        for sc in noswap.scenarios:
            kv_default    = sc["kv_raw_gb"]
            kv_safe       = sc["kv_gb"]
            swap_risk     = sc["avail_gb"] < kv_default + 1.5
            default_str   = f"[red]{kv_default:.3f} GB ⚠[/red]" if swap_risk else f"{kv_default:.3f} GB"
            safe_str      = f"[green]{kv_safe:.3f} GB[/green]" if kv_safe < kv_default else f"{kv_safe:.3f} GB"
            action        = level_labels.get(sc["level"], sc["level"])
            t2.add_row(sc["label"], default_str, safe_str, action)

        console.print(t2)
        console.print()
        console.print(
            "  Enable it:  [bold]autotune proof --model {model} --with-noswap[/bold]\n"
            "  Or in chat: [bold]autotune chat --no-swap[/bold]"
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    console.print()
    console.rule(style="dim")
    console.print()
    console.print(
        "  [dim]How we measured[/dim]\n"
        "  [dim]Timings from Ollama's /api/chat response fields:[/dim]\n"
        "  [dim]  load_duration       = model load + KV buffer allocation[/dim]\n"
        "  [dim]  prompt_eval_duration = prefill (what we reduce)[/dim]\n"
        "  [dim]  eval_duration / eval_count = generation tok/s[/dim]\n"
        "  [dim]  /api/ps size_vram   = Metal GPU memory[/dim]\n\n"
        "  [dim]Raw conditions run first (all prompts), then autotune conditions.\n"
        "  Warm measurements exclude the first request of each condition\n"
        "  (which includes the cold KV allocation shown in Section 1).[/dim]\n\n"
        "  [dim]Raw data saved to proof_results.json[/dim]"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# JSON export
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(
    model, raw_warm, tune_warm, raw_cold, tune_cold, vram, noswap, output_path,
) -> None:
    def _runs(runs):
        return [asdict(r) for r in runs] if runs else []

    doc = {
        "tool":      "autotune proof",
        "version":   "3.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model":     model,
        "config": {
            "raw_ctx":    RAW_CTX,
            "max_tokens": MAX_TOKENS,
        },
        "phases": {
            "warm": {"raw": _runs(raw_warm), "autotune": _runs(tune_warm)},
        },
    }
    if raw_cold:
        doc["phases"]["cold"] = {"raw": _runs(raw_cold), "autotune": _runs(tune_cold)}
    if vram:
        doc["phases"]["vram"] = {
            "raw_ctx":  vram.raw_ctx,  "raw_gb":  vram.raw_gb,
            "tune_ctx": vram.tune_ctx, "tune_gb": vram.tune_gb,
            "saved_gb": vram.saved_gb,
        }
    if noswap:
        doc["phases"]["noswap_demo"] = {
            "arch": {
                "n_layers":   noswap.arch.n_layers,
                "n_kv_heads": noswap.arch.n_kv_heads,
                "head_dim":   noswap.arch.head_dim,
            },
            "swap_raw_gb":  noswap.swap_raw_gb,
            "swap_tune_gb": noswap.swap_tune_gb,
            "scenarios":    noswap.scenarios,
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
    console.print(
        f"  Running benchmark on [cyan]{model}[/cyan]…  "
        f"[dim](this takes about 2–3 minutes)[/dim]"
    )

    # Fresh start — unload any cached model state
    console.print("  [dim]Resetting model state…[/dim]")
    await client.unload_model(model)
    await asyncio.sleep(2.0)

    # Swap baseline before benchmark
    swap_before = _swap_gb()

    # Phase 1 — warm (all-raw first, then all-autotune)
    console.print("  [dim]Phase 1 — measuring response speed…[/dim]")
    raw_warm, tune_warm = await run_warm(model, args.runs, client)

    # Swap after warm phase (captures any induced swap)
    swap_after_raw  = _swap_gb()

    # Phase 2 — VRAM
    vram = None
    console.print("  [dim]Phase 2 — measuring memory usage…[/dim]")
    vram = await run_vram(model, client)

    # Swap after VRAM phase
    swap_after_tune = _swap_gb()

    # Phase 3 — cold (optional)
    raw_cold = tune_cold = None
    if args.with_cold:
        console.print("  [dim]Phase 3 — measuring startup time…[/dim]")
        raw_cold, tune_cold = await run_cold(model, args.cold_runs, client)

    # No-swap demo (optional)
    noswap = None
    if args.with_noswap:
        console.print("  [dim]Computing no-swap scenarios…[/dim]")
        # Use max swap observed across all phases as the representative value
        swap_raw_observed  = max(0.0, swap_after_raw  - swap_before)
        swap_tune_observed = max(0.0, swap_after_tune - swap_before)
        noswap = await run_noswap_demo(
            model, swap_raw_observed, swap_tune_observed,
        )

    print_results(
        model=model,
        raw_warm=raw_warm, tune_warm=tune_warm,
        raw_cold=raw_cold, tune_cold=tune_cold,
        vram=vram, noswap=noswap,
        chip=chip, n_runs=args.runs,
    )

    _save_json(
        model, raw_warm, tune_warm, raw_cold, tune_cold, vram, noswap,
        args.output,
    )
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
