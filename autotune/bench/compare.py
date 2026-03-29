"""
autotune/bench/compare.py — honest 1-vs-1 benchmark comparison engine.

Runs a curated prompt suite through two conditions:

  RAW OLLAMA   — zero autotune settings.  Direct HTTP to Ollama with its
                 factory defaults: num_ctx=4096, temp=0.8, no prefix cache,
                 no hardware tuning, no memory pressure management.

  AUTOTUNE     — full optimizer stack: dynamic context sizing, KV precision
                 management, system-prompt prefix caching (num_keep), RAM
                 pressure pre-emption, hardware QoS tuning, MLX routing on
                 Apple Silicon.

Every result is persisted to run_observations with a bench_tag so sessions
can be compared later with `autotune db`.

Output is a color-coded Rich report.  If autotune makes something WORSE it
shows that in red too — the numbers are what they are.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil
from rich import box as _box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from autotune.bench.runner import BenchResult, run_bench, run_raw_ollama, save_result


# ---------------------------------------------------------------------------
# Benchmark prompt suite
# ---------------------------------------------------------------------------

@dataclass
class BenchPrompt:
    name: str
    domain: str
    messages: list[dict]


BENCHMARK_PROMPTS: list[BenchPrompt] = [
    # 1. Short factual — TTFT-sensitive, minimal generation
    BenchPrompt(
        name="quick_answer",
        domain="Factual Q&A",
        messages=[
            {
                "role": "user",
                "content": (
                    "List the planets in our solar system from closest to furthest "
                    "from the Sun. For each, give one distinctive fact in one sentence."
                ),
            }
        ],
    ),

    # 2. Code generation — medium sustained generation, tests tok/s
    BenchPrompt(
        name="code_generation",
        domain="Code",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Python engineer. Write clean, idiomatic code "
                    "with type hints and docstrings."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Implement a MinHeap class in Python without using heapq. "
                    "Support push(val), pop() returning the minimum, peek(), "
                    "and __len__. Include at least 5 unit tests using unittest."
                ),
            },
        ],
    ),

    # 3. Reasoning — multi-step, tests context + medium output
    BenchPrompt(
        name="math_reasoning",
        domain="Reasoning",
        messages=[
            {
                "role": "user",
                "content": (
                    "Solve completely, showing every step:\n\n"
                    "Train A departs City X at 08:00 heading east at 75 mph. "
                    "Train B departs City Y — 420 miles east of X — at 09:30 heading west at 90 mph.\n"
                    "(a) At what clock time and distance from City X do they meet?\n"
                    "(b) If Train A is delayed 25 minutes, how does the answer change?\n"
                    "(c) What is the average speed of each train relative to the meeting point?"
                ),
            }
        ],
    ),

    # 4. Long sustained output — RAM pressure, KV management, max tok/s
    BenchPrompt(
        name="long_output",
        domain="Long generation",
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a comprehensive technical guide comparing SQL and NoSQL databases. "
                    "Cover ALL of the following in depth:\n"
                    "• Data models and schema philosophy\n"
                    "• ACID vs BASE consistency guarantees\n"
                    "• Horizontal vs vertical scaling trade-offs\n"
                    "• Replication and sharding strategies\n"
                    "• Indexing approaches and query patterns\n"
                    "• Concrete real-world databases for each category with use cases\n"
                    "• Decision framework: when to pick which\n\n"
                    "This will be used as an engineering reference — be thorough."
                ),
            }
        ],
    ),

    # 5. Multi-turn conversation — prefix cache benefit is largest here
    BenchPrompt(
        name="multi_turn",
        domain="Conversation",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior distributed-systems architect with 15 years of "
                    "production experience at scale. Give detailed, opinionated advice "
                    "backed by real examples. Always explain trade-offs explicitly."
                ),
            },
            {
                "role": "user",
                "content": "What authentication pattern would you recommend for a microservices architecture?",
            },
            {
                "role": "assistant",
                "content": (
                    "For microservices auth I'd use short-lived JWTs (15 min TTL) issued by a "
                    "central auth service, with opaque refresh tokens stored server-side. "
                    "Each service validates JWTs locally without hitting the auth service — "
                    "that's the key latency win. But JWT revocation is where most teams get "
                    "tripped up…"
                ),
            },
            {
                "role": "user",
                "content": "Exactly — how do you handle revocation without reintroducing state?",
            },
            {
                "role": "assistant",
                "content": (
                    "You can't fully avoid state for revocation — it's a fundamental CAP "
                    "trade-off. Pragmatic options: (1) short TTL + refresh token blacklist "
                    "hits the auth service only on refresh, (2) token fingerprinting via a "
                    "Bloom filter for fast probabilistic checks, (3) jti claim with a "
                    "distributed cache like Redis with a TTL matching the JWT expiry…"
                ),
            },
            {
                "role": "user",
                "content": (
                    "We're a team of 6 building B2B SaaS, ~50k users. "
                    "Give me a concrete recommendation: specific libraries, token lifetimes, "
                    "how you'd structure the auth service, and what you'd NOT do."
                ),
            },
        ],
    ),

    # 6. Code analysis — medium analytical output, tests context + quality
    BenchPrompt(
        name="code_review",
        domain="Analysis",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a meticulous security-focused code reviewer. "
                    "Categorize findings by severity (Critical / High / Medium / Low). "
                    "Be specific — cite line numbers and explain the exact risk."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Review this Flask login endpoint for ALL issues:\n\n"
                    "```python\n"
                    "import sqlite3\n"
                    "from flask import Flask, request\n\n"
                    "app = Flask(__name__)\n"
                    "db = sqlite3.connect('users.db')\n\n"
                    "@app.route('/login', methods=['POST'])\n"
                    "def login():\n"
                    "    u = request.form['username']\n"
                    "    p = request.form['password']\n"
                    "    q = f\"SELECT * FROM users WHERE username='{u}' AND password='{p}'\"\n"
                    "    row = db.cursor().execute(q).fetchone()\n"
                    "    if row:\n"
                    "        return {'token': row[0], 'admin': row[3]}\n"
                    "    return {'error': 'invalid credentials'}, 401\n"
                    "```\n\n"
                    "Cover: security vulnerabilities, correctness bugs, performance issues, "
                    "and production reliability concerns. Then provide a corrected version."
                ),
            },
        ],
    ),
]


# ---------------------------------------------------------------------------
# Per-condition aggregated stats
# ---------------------------------------------------------------------------

@dataclass
class ConditionStats:
    label: str
    mean_tps: float = 0.0
    mean_ttft_ms: float = 0.0
    mean_peak_ram_gb: float = 0.0
    mean_delta_ram_gb: float = 0.0
    mean_cpu_pct: float = 0.0
    mean_swap_peak_gb: float = 0.0
    mean_elapsed_sec: float = 0.0
    mean_completion_tokens: float = 0.0
    error_count: int = 0
    # Optimizer settings (meaningful only for tuned condition)
    avg_num_ctx: float = 0.0
    avg_num_keep: float = 0.0
    kv_f16: Optional[bool] = None   # True=F16, False=Q8, None=Ollama default
    raw_results: list[BenchResult] = field(default_factory=list, repr=False)


def _aggregate(results: list[BenchResult], label: str) -> ConditionStats:
    ok = [r for r in results if not r.error]
    errors = len(results) - len(ok)

    def _m(vals: list[float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    # KV precision: True=F16, False=Q8 — take mode of non-None values
    kv_vals = [r.f16_kv_used for r in ok if r.f16_kv_used is not None]
    kv_f16: Optional[bool] = None
    if kv_vals:
        kv_f16 = sum(1 for v in kv_vals if v) >= len(kv_vals) / 2

    return ConditionStats(
        label=label,
        mean_tps=round(_m([r.tokens_per_sec for r in ok]), 2),
        mean_ttft_ms=round(_m([r.ttft_ms for r in ok]), 1),
        mean_peak_ram_gb=round(_m([r.ram_peak_gb for r in ok]), 3),
        mean_delta_ram_gb=round(_m([r.delta_ram_gb for r in ok]), 3),
        mean_cpu_pct=round(_m([r.cpu_avg_pct for r in ok]), 1),
        mean_swap_peak_gb=round(_m([r.swap_peak_gb for r in ok]), 3),
        mean_elapsed_sec=round(_m([r.elapsed_sec for r in ok]), 2),
        mean_completion_tokens=round(_m([r.completion_tokens for r in ok]), 0),
        error_count=errors,
        avg_num_ctx=round(_m([r.num_ctx_used for r in ok if r.num_ctx_used]), 0),
        avg_num_keep=round(_m([r.num_keep_used for r in ok if r.num_keep_used]), 0),
        kv_f16=kv_f16,
        raw_results=results,
    )


# ---------------------------------------------------------------------------
# Per-prompt comparison
# ---------------------------------------------------------------------------

@dataclass
class PromptComparison:
    prompt: BenchPrompt
    raw: ConditionStats
    tuned: ConditionStats


@dataclass
class CompareReport:
    model_id: str
    profile_name: str
    n_runs: int
    hw_info: str
    comparisons: list[PromptComparison]
    timestamp: float


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

_WARMUP_MSG = [{"role": "user", "content": "Say 'ready' and nothing else."}]
_COOLDOWN_BETWEEN_RUNS_SEC = 4.0
_COOLDOWN_BETWEEN_CONDITIONS_SEC = 12.0


async def run_comparison(
    model_id: str,
    n_runs: int,
    profile_name: str,
    console: Console,
    save_db: bool = True,
) -> CompareReport:
    """
    Run the full 1-vs-1 benchmark.

    Order:
      1. Warmup (not counted) — loads model into Ollama memory
      2. All raw_ollama runs — pure Ollama, no autotune
      3. 12-second cool-down — let RAM settle between conditions
      4. All autotune runs — full optimizer stack
      5. Return CompareReport
    """
    n_prompts = len(BENCHMARK_PROMPTS)
    total_steps = 1 + (n_prompts * n_runs) + 1 + (n_prompts * n_runs)

    raw_results_map:   dict[str, list[BenchResult]] = {}
    tuned_results_map: dict[str, list[BenchResult]] = {}

    run_ts = int(time.time())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}", no_wrap=True),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        overall = progress.add_task("Benchmarking…", total=total_steps)

        # ── Warmup ──────────────────────────────────────────────────────────
        progress.update(overall, description="[dim]Warming up (loading model)…[/dim]")
        try:
            await run_raw_ollama(model_id, _WARMUP_MSG, tag="warmup")
        except Exception:
            pass   # warmup failure is non-fatal
        progress.advance(overall)

        # ── Round 1: Raw Ollama ──────────────────────────────────────────────
        for prompt in BENCHMARK_PROMPTS:
            raw_results_map[prompt.name] = []
            for i in range(n_runs):
                desc = (
                    f"[dim]Raw Ollama · {prompt.name}"
                    + (f"  run {i+1}/{n_runs}" if n_runs > 1 else "")
                    + "[/dim]"
                )
                progress.update(overall, description=desc)
                tag = f"raw_{prompt.name}_{run_ts}"
                try:
                    result = await run_raw_ollama(model_id, prompt.messages, tag=tag)
                    if save_db:
                        save_result(result)
                except Exception as exc:
                    result = _error_result(model_id, "raw_ollama_defaults", tag, str(exc))
                raw_results_map[prompt.name].append(result)
                progress.advance(overall)
                if i < n_runs - 1:
                    await asyncio.sleep(_COOLDOWN_BETWEEN_RUNS_SEC)

        # ── Cool-down between conditions ─────────────────────────────────────
        progress.update(overall, description="[dim]Cooling down between conditions…[/dim]")
        await asyncio.sleep(_COOLDOWN_BETWEEN_CONDITIONS_SEC)

        # ── Round 2: Autotune ────────────────────────────────────────────────
        for prompt in BENCHMARK_PROMPTS:
            tuned_results_map[prompt.name] = []
            for i in range(n_runs):
                desc = (
                    f"[cyan]Autotune/{profile_name} · {prompt.name}"
                    + (f"  run {i+1}/{n_runs}" if n_runs > 1 else "")
                    + "[/cyan]"
                )
                progress.update(overall, description=desc)
                tag = f"tuned_{profile_name}_{prompt.name}_{run_ts}"
                try:
                    result = await run_bench(
                        model_id,
                        prompt.messages,
                        profile_name=profile_name,
                        tag=tag,
                    )
                    if save_db:
                        save_result(result)
                except Exception as exc:
                    result = _error_result(model_id, profile_name, tag, str(exc))
                tuned_results_map[prompt.name].append(result)
                progress.advance(overall)
                if i < n_runs - 1:
                    await asyncio.sleep(_COOLDOWN_BETWEEN_RUNS_SEC)

        progress.update(overall, description="[green]Done.[/green]")

    # ── Build report ─────────────────────────────────────────────────────────
    comparisons = [
        PromptComparison(
            prompt=p,
            raw=_aggregate(raw_results_map[p.name], "Raw Ollama"),
            tuned=_aggregate(tuned_results_map[p.name], f"Autotune/{profile_name}"),
        )
        for p in BENCHMARK_PROMPTS
    ]

    hw_info = _hw_summary()

    return CompareReport(
        model_id=model_id,
        profile_name=profile_name,
        n_runs=n_runs,
        hw_info=hw_info,
        comparisons=comparisons,
        timestamp=time.time(),
    )


def _error_result(model_id: str, profile: str, tag: str, error: str) -> BenchResult:
    """Return a zero-filled BenchResult with the error message set."""
    from autotune.bench.runner import BenchResult
    return BenchResult(
        tag=tag, model_id=model_id, profile_name=profile,
        prompt_tokens=0, completion_tokens=0,
        ttft_ms=0.0, tokens_per_sec=0.0, elapsed_sec=0.0,
        ram_before_gb=0.0, ram_peak_gb=0.0, ram_after_gb=0.0,
        swap_before_gb=0.0, swap_peak_gb=0.0, swap_after_gb=0.0,
        cpu_avg_pct=0.0, cpu_peak_pct=0.0,
        error=error,
    )


def _hw_summary() -> str:
    try:
        from autotune.hardware.profiler import profile_hardware
        hw = profile_hardware()
        cpu_parts = hw.cpu.brand.split()
        cpu_short = " ".join(cpu_parts[:4]) if len(cpu_parts) >= 4 else hw.cpu.brand
        return f"{cpu_short}  /  {hw.memory.total_gb:.0f} GB RAM"
    except Exception:
        vm = psutil.virtual_memory()
        return f"Unknown CPU  /  {vm.total / 1024**3:.0f} GB RAM"


# ---------------------------------------------------------------------------
# Delta helpers
# ---------------------------------------------------------------------------

def _pct(raw: float, tuned: float) -> float:
    """(tuned - raw) / raw * 100 — callers interpret sign based on metric."""
    if raw == 0:
        return 0.0
    return (tuned - raw) / abs(raw) * 100.0


def _delta_cell(
    raw: float,
    tuned: float,
    higher_is_better: bool,
    fmt: str = ".1f",
    unit: str = "",
) -> Text:
    """
    Return a Rich Text object for a delta cell, colored green/red based on
    whether the change is beneficial or detrimental.
    """
    if raw == 0 and tuned == 0:
        return Text("—", style="dim")

    pct = _pct(raw, tuned)
    beneficial = (pct > 1.0 and higher_is_better) or (pct < -1.0 and not higher_is_better)
    detrimental = (pct < -1.0 and higher_is_better) or (pct > 1.0 and not higher_is_better)

    if abs(pct) < 1.0:
        style = "dim"
    elif beneficial:
        style = "green"
    elif detrimental:
        style = "red"
    else:
        style = "dim"

    sign = "+" if pct > 0 else ""
    text = f"{sign}{pct:.1f}%"
    return Text(text, style=style)


def _val(v: float, fmt: str = ".1f", unit: str = "") -> str:
    if v == 0.0:
        return "—"
    return f"{v:{fmt}}{unit}"


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(report: CompareReport, console: Console) -> None:  # noqa: C901
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(report.timestamp))
    n_prompts = len(report.comparisons)

    # ── Header ────────────────────────────────────────────────────────────────
    header = (
        f"[bold]{report.model_id}[/bold]\n"
        f"[dim]{report.hw_info}   ·   {ts}[/dim]\n"
        f"[dim]Profile: autotune/{report.profile_name}   ·   "
        f"{n_prompts} prompts × {report.n_runs} run{'s' if report.n_runs > 1 else ''} per condition[/dim]"
    )
    console.print(Panel(header, title="autotune benchmark", title_align="left", border_style="cyan"))

    # ── Per-prompt: Speed & Latency ────────────────────────────────────────────
    console.print(Rule("[bold]Speed & Latency[/bold]", style="dim"))
    t_speed = Table(box=_box.SIMPLE_HEAD, show_lines=False, expand=True)
    t_speed.add_column("Prompt",         style="bold",  no_wrap=True)
    t_speed.add_column("Domain",         style="dim",   no_wrap=True)
    t_speed.add_column("Raw  tok/s",     justify="right")
    t_speed.add_column("Tuned tok/s",    justify="right")
    t_speed.add_column("Δ tok/s",        justify="right")
    t_speed.add_column("Raw TTFT ms",    justify="right")
    t_speed.add_column("Tuned TTFT ms",  justify="right")
    t_speed.add_column("Δ TTFT",         justify="right")
    t_speed.add_column("Tokens out",     justify="right", style="dim")

    for cmp in report.comparisons:
        raw, tuned = cmp.raw, cmp.tuned
        tps_delta  = _delta_cell(raw.mean_tps, tuned.mean_tps, higher_is_better=True)
        ttft_delta = _delta_cell(raw.mean_ttft_ms, tuned.mean_ttft_ms, higher_is_better=False)

        # Show error indicator if runs failed
        raw_tps_str   = _val(raw.mean_tps,   ".1f") + (" [red]ERR[/red]" if raw.error_count else "")
        tuned_tps_str = _val(tuned.mean_tps, ".1f") + (" [red]ERR[/red]" if tuned.error_count else "")

        t_speed.add_row(
            cmp.prompt.name,
            cmp.prompt.domain,
            raw_tps_str,
            tuned_tps_str,
            tps_delta,
            _val(raw.mean_ttft_ms, ".0f", " ms"),
            _val(tuned.mean_ttft_ms, ".0f", " ms"),
            ttft_delta,
            f"{raw.mean_completion_tokens:.0f} / {tuned.mean_completion_tokens:.0f}",
        )
    console.print(t_speed)

    # ── Per-prompt: Memory & CPU ───────────────────────────────────────────────
    console.print(Rule("[bold]Memory & CPU[/bold]", style="dim"))
    t_mem = Table(box=_box.SIMPLE_HEAD, show_lines=False, expand=True)
    t_mem.add_column("Prompt",         style="bold", no_wrap=True)
    t_mem.add_column("Domain",         style="dim",  no_wrap=True)
    t_mem.add_column("Raw peak RAM",   justify="right")
    t_mem.add_column("Tuned peak RAM", justify="right")
    t_mem.add_column("Δ peak RAM",     justify="right")
    t_mem.add_column("Raw RAM Δ",      justify="right")
    t_mem.add_column("Tuned RAM Δ",    justify="right")
    t_mem.add_column("Δ RAM leak",     justify="right")
    t_mem.add_column("Raw CPU%",       justify="right")
    t_mem.add_column("Tuned CPU%",     justify="right")
    t_mem.add_column("Δ CPU",          justify="right")

    for cmp in report.comparisons:
        raw, tuned = cmp.raw, cmp.tuned
        ram_peak_delta = _delta_cell(raw.mean_peak_ram_gb,  tuned.mean_peak_ram_gb,  higher_is_better=False)
        ram_leak_delta = _delta_cell(raw.mean_delta_ram_gb, tuned.mean_delta_ram_gb, higher_is_better=False)
        cpu_delta      = _delta_cell(raw.mean_cpu_pct,      tuned.mean_cpu_pct,      higher_is_better=False)

        t_mem.add_row(
            cmp.prompt.name,
            cmp.prompt.domain,
            _val(raw.mean_peak_ram_gb,   ".2f", " GB"),
            _val(tuned.mean_peak_ram_gb, ".2f", " GB"),
            ram_peak_delta,
            f"{raw.mean_delta_ram_gb:+.3f} GB",
            f"{tuned.mean_delta_ram_gb:+.3f} GB",
            ram_leak_delta,
            _val(raw.mean_cpu_pct,   ".1f", "%"),
            _val(tuned.mean_cpu_pct, ".1f", "%"),
            cpu_delta,
        )
    console.print(t_mem)

    # ── Overall summary ────────────────────────────────────────────────────────
    console.print(Rule("[bold]Overall Summary  —  all prompts aggregated[/bold]", style="dim"))

    # Aggregate across all prompts
    def _agg(getter, condition: str) -> float:
        vals = [
            getter(cmp.raw if condition == "raw" else cmp.tuned)
            for cmp in report.comparisons
        ]
        return statistics.mean(vals) if vals else 0.0

    agg_raw_tps       = _agg(lambda s: s.mean_tps, "raw")
    agg_tuned_tps     = _agg(lambda s: s.mean_tps, "tuned")
    agg_raw_ttft      = _agg(lambda s: s.mean_ttft_ms, "raw")
    agg_tuned_ttft    = _agg(lambda s: s.mean_ttft_ms, "tuned")
    agg_raw_ram       = _agg(lambda s: s.mean_peak_ram_gb, "raw")
    agg_tuned_ram     = _agg(lambda s: s.mean_peak_ram_gb, "tuned")
    agg_raw_leak      = _agg(lambda s: s.mean_delta_ram_gb, "raw")
    agg_tuned_leak    = _agg(lambda s: s.mean_delta_ram_gb, "tuned")
    agg_raw_swap      = _agg(lambda s: s.mean_swap_peak_gb, "raw")
    agg_tuned_swap    = _agg(lambda s: s.mean_swap_peak_gb, "tuned")
    agg_raw_cpu       = _agg(lambda s: s.mean_cpu_pct, "raw")
    agg_tuned_cpu     = _agg(lambda s: s.mean_cpu_pct, "tuned")
    agg_raw_elapsed   = _agg(lambda s: s.mean_elapsed_sec, "raw")
    agg_tuned_elapsed = _agg(lambda s: s.mean_elapsed_sec, "tuned")

    t_sum = Table(box=_box.SIMPLE_HEAD, show_lines=False, expand=True)
    t_sum.add_column("Metric",              style="bold", no_wrap=True)
    t_sum.add_column("Raw Ollama",          justify="right")
    t_sum.add_column(f"Autotune/{report.profile_name}", justify="right")
    t_sum.add_column("Change",              justify="right")
    t_sum.add_column("Verdict",             justify="left")

    def _verdict(pct: float, higher_is_better: bool, threshold: float = 3.0) -> Text:
        beneficial = (pct > threshold and higher_is_better) or (pct < -threshold and not higher_is_better)
        detrimental = (pct < -threshold and higher_is_better) or (pct > threshold and not higher_is_better)
        if beneficial:
            return Text("✓ Better", style="green")
        elif detrimental:
            return Text("✗ Worse",  style="red")
        else:
            return Text("≈ Similar", style="dim")

    rows = [
        ("Throughput (tok/s)", agg_raw_tps,     agg_tuned_tps,     True,  ".1f", ""),
        ("TTFT (ms)",          agg_raw_ttft,    agg_tuned_ttft,    False, ".0f", " ms"),
        ("Peak RAM (GB)",      agg_raw_ram,     agg_tuned_ram,     False, ".2f", " GB"),
        ("RAM Δ per turn",     agg_raw_leak,    agg_tuned_leak,    False, "+.3f", " GB"),
        ("Swap peak (GB)",     agg_raw_swap,    agg_tuned_swap,    False, ".3f", " GB"),
        ("CPU avg (%)",        agg_raw_cpu,     agg_tuned_cpu,     False, ".1f", "%"),
        ("Total elapsed (s)",  agg_raw_elapsed, agg_tuned_elapsed, False, ".1f", " s"),
    ]

    wins = losses = 0
    for label, raw_val, tuned_val, hib, fmt, unit in rows:
        pct = _pct(raw_val, tuned_val)
        delta = _delta_cell(raw_val, tuned_val, higher_is_better=hib)
        v = _verdict(pct, hib)
        if "Better" in v.plain:
            wins += 1
        elif "Worse" in v.plain:
            losses += 1

        t_sum.add_row(
            label,
            f"{raw_val:{fmt}}{unit}" if raw_val != 0 else "—",
            f"{tuned_val:{fmt}}{unit}" if tuned_val != 0 else "—",
            delta,
            v,
        )
    console.print(t_sum)

    # ── Optimizer settings panel ───────────────────────────────────────────────
    tuned_stats_all = [cmp.tuned for cmp in report.comparisons]
    avg_ctx  = statistics.mean([s.avg_num_ctx for s in tuned_stats_all if s.avg_num_ctx]) \
               if any(s.avg_num_ctx for s in tuned_stats_all) else 0.0
    avg_keep = statistics.mean([s.avg_num_keep for s in tuned_stats_all if s.avg_num_keep]) \
               if any(s.avg_num_keep for s in tuned_stats_all) else 0.0
    kv_f16_vals = [s.kv_f16 for s in tuned_stats_all if s.kv_f16 is not None]
    kv_label = ("F16" if (sum(1 for v in kv_f16_vals if v) >= len(kv_f16_vals) / 2) else "Q8_0") \
               if kv_f16_vals else "F16 (default)"

    from autotune.api.profiles import get_profile as _gp
    try:
        prof = _gp(report.profile_name)
        max_ctx = prof.max_context_tokens
        max_tok = prof.max_new_tokens
        qos     = prof.qos_class
        gc_off  = prof.disable_gc_during_inference
    except Exception:
        max_ctx = max_tok = 0
        qos = "—"
        gc_off = False

    settings_lines = [
        f"  [bold]Context window:[/bold]  {avg_ctx:.0f} tokens avg  "
        f"[dim](Ollama default: 4096  ·  profile max: {max_ctx:,})[/dim]",
        f"  [bold]KV precision:[/bold]    {kv_label}  "
        f"[dim](Ollama default: F16)[/dim]",
        f"  [bold]Prefix cache:[/bold]    "
        + (f"{avg_keep:.0f} tokens pinned (system prompt)  [dim](Ollama: none)[/dim]"
           if avg_keep > 0 else "[dim]none (no system prompts in test)[/dim]"),
        f"  [bold]Max new tokens:[/bold]  {max_tok}  "
        f"[dim](Ollama default: unlimited / -1)[/dim]",
        f"  [bold]HW QoS class:[/bold]   {qos}  "
        f"[dim](Ollama default: DEFAULT)[/dim]",
        f"  [bold]GC during infer:[/bold] {'disabled' if gc_off else 'enabled'}  "
        f"[dim](Ollama: always enabled)[/dim]",
        f"  [bold]MLX routing:[/bold]     auto (Apple Silicon) — Ollama does not do this",
    ]
    console.print(Panel(
        "\n".join(settings_lines),
        title="What autotune configured (vs raw Ollama defaults)",
        title_align="left",
        border_style="dim",
    ))

    # ── Honest verdict ─────────────────────────────────────────────────────────
    total_metrics = len(rows)
    neutral = total_metrics - wins - losses

    if wins >= 5:
        verdict_style = "green"
        verdict_icon  = "✓"
        verdict_head  = f"autotune/{report.profile_name} is faster and leaner ({wins}/{total_metrics} metrics better)"
    elif wins > losses:
        verdict_style = "green"
        verdict_icon  = "✓"
        verdict_head  = f"autotune/{report.profile_name} wins overall  ({wins}W · {losses}L · {neutral}≈)"
    elif wins == losses:
        verdict_style = "yellow"
        verdict_icon  = "≈"
        verdict_head  = f"Mixed results — autotune and raw Ollama are comparable  ({wins}W · {losses}L · {neutral}≈)"
    else:
        verdict_style = "red"
        verdict_icon  = "!"
        verdict_head  = f"Raw Ollama outperformed autotune/{report.profile_name} on this workload  ({losses}L · {wins}W)"

    # Build per-metric bullet points
    bullets: list[str] = []
    tps_pct    = _pct(agg_raw_tps,     agg_tuned_tps)
    ttft_pct   = _pct(agg_raw_ttft,    agg_tuned_ttft)
    ram_pct    = _pct(agg_raw_ram,     agg_tuned_ram)
    cpu_pct    = _pct(agg_raw_cpu,     agg_tuned_cpu)
    swap_delta = agg_tuned_swap - agg_raw_swap

    if abs(tps_pct) >= 3:
        sym = "↑" if tps_pct > 0 else "↓"
        col = "green" if tps_pct > 0 else "red"
        bullets.append(f"  [{col}]{sym} Throughput {tps_pct:+.1f}% ({agg_raw_tps:.1f} → {agg_tuned_tps:.1f} tok/s)[/{col}]")

    if abs(ttft_pct) >= 3:
        sym = "↓" if ttft_pct < 0 else "↑"
        col = "green" if ttft_pct < 0 else "red"
        bullets.append(f"  [{col}]{sym} TTFT {ttft_pct:+.1f}% ({agg_raw_ttft:.0f} → {agg_tuned_ttft:.0f} ms)[/{col}]")

    if abs(ram_pct) >= 2:
        gb_delta = agg_tuned_ram - agg_raw_ram
        sym = "↓" if gb_delta < 0 else "↑"
        col = "green" if gb_delta < 0 else "red"
        bullets.append(f"  [{col}]{sym} Peak RAM {gb_delta:+.2f} GB ({agg_raw_ram:.2f} → {agg_tuned_ram:.2f} GB)[/{col}]")

    if abs(cpu_pct) >= 3:
        sym = "↓" if cpu_pct < 0 else "↑"
        col = "green" if cpu_pct < 0 else "red"
        bullets.append(f"  [{col}]{sym} CPU {cpu_pct:+.1f}% avg ({agg_raw_cpu:.1f} → {agg_tuned_cpu:.1f}%)[/{col}]")

    if abs(swap_delta) > 0.05:
        sym = "↓" if swap_delta < 0 else "↑"
        col = "green" if swap_delta < 0 else "red"
        bullets.append(f"  [{col}]{sym} Swap peak {swap_delta:+.2f} GB[/{col}]")

    if not bullets:
        bullets.append("  [dim]All metrics within ±3% — no significant difference detected.[/dim]")

    # Add context note about output length difference
    raw_ctx_avg  = statistics.mean([c.raw.avg_num_ctx for c in report.comparisons if c.raw.avg_num_ctx]) \
                   if any(c.raw.avg_num_ctx for c in report.comparisons) else 4096.0
    if abs(avg_ctx - raw_ctx_avg) > 200:
        bullets.append(
            f"\n  [dim]Note: context window differed — raw {raw_ctx_avg:.0f} tokens vs "
            f"autotune {avg_ctx:.0f} tokens.  Smaller context = less KV memory but "
            f"may affect very long conversations.[/dim]"
        )

    total_raw_errors  = sum(c.raw.error_count   for c in report.comparisons)
    total_tuned_errors = sum(c.tuned.error_count for c in report.comparisons)
    if total_raw_errors or total_tuned_errors:
        bullets.append(
            f"\n  [red]⚠  Errors: raw={total_raw_errors}  tuned={total_tuned_errors}  "
            f"— errored runs excluded from averages.[/red]"
        )

    verdict_body = f"[bold {verdict_style}]{verdict_icon}  {verdict_head}[/bold {verdict_style}]\n\n" + "\n".join(bullets)
    console.print(Panel(verdict_body, title="Verdict", title_align="left", border_style=verdict_style))
    console.print()


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(report: CompareReport, path: str, console: Console) -> None:
    def _stats_dict(s: ConditionStats) -> dict:
        return {
            "label":                s.label,
            "mean_tps":             s.mean_tps,
            "mean_ttft_ms":         s.mean_ttft_ms,
            "mean_peak_ram_gb":     s.mean_peak_ram_gb,
            "mean_delta_ram_gb":    s.mean_delta_ram_gb,
            "mean_cpu_pct":         s.mean_cpu_pct,
            "mean_swap_peak_gb":    s.mean_swap_peak_gb,
            "mean_elapsed_sec":     s.mean_elapsed_sec,
            "mean_completion_tokens": s.mean_completion_tokens,
            "error_count":          s.error_count,
            "avg_num_ctx":          s.avg_num_ctx,
            "avg_num_keep":         s.avg_num_keep,
            "kv_f16":               s.kv_f16,
        }

    out = {
        "model_id":    report.model_id,
        "profile":     report.profile_name,
        "n_runs":      report.n_runs,
        "hardware":    report.hw_info,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(report.timestamp)),
        "prompts": [
            {
                "name":   cmp.prompt.name,
                "domain": cmp.prompt.domain,
                "raw":    _stats_dict(cmp.raw),
                "tuned":  _stats_dict(cmp.tuned),
                "delta_tps_pct":  _pct(cmp.raw.mean_tps, cmp.tuned.mean_tps),
                "delta_ttft_pct": _pct(cmp.raw.mean_ttft_ms, cmp.tuned.mean_ttft_ms),
                "delta_ram_pct":  _pct(cmp.raw.mean_peak_ram_gb, cmp.tuned.mean_peak_ram_gb),
            }
            for cmp in report.comparisons
        ],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"[dim]Results saved to {path}[/dim]")
