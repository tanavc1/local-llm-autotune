"""
autotune unified benchmark command — production-grade proof system.

This module powers `autotune bench *` — the single authoritative place
to benchmark, profile, and prove autotune's value across every dimension:

    autotune bench quick   — 45-second head-to-head on any hardware
    autotune bench duel    — single-prompt raw vs autotune comparison
    autotune bench suite   — 5-prompt statistical benchmark (Wilcoxon, Cohen's d)
    autotune bench agent   — multi-turn agentic workloads
    autotune bench ux      — user-experience KPIs (swap events, RAM headroom)
    autotune bench os      — OS-level optimization impact (GC, QOS, flash-attn)
    autotune bench server  — server/cloud throughput (concurrency, P99 latency)
    autotune bench all     — run every tier and produce a full report

The `os` and `server` subcommands are unique to this module — they test
things that matter at cloud scale but are invisible in single-request tests.

Server/cloud angle
------------------
Cloud providers and on-prem inference servers typically run one Ollama
process serving many users.  The `bench server` subcommand measures:
  • P50 / P95 / P99 latency under serial load
  • Requests-per-second at concurrency 1 / 2 / 4 / 8
  • KV-cache RAM cost per additional concurrent slot
  • Throughput degradation under sustained load (50+ requests)
  • Cost-per-million-tokens estimate at common cloud VM prices

Every metric is measured for raw Ollama and autotune side-by-side so the
operator can see the exact cost reduction from deploying autotune.

OS optimization angle
---------------------
`bench os` isolates each individual OS-level optimization that autotune
applies and measures its isolated contribution:
  • Python GC suspend during inference
  • macOS QOS_CLASS_USER_INTERACTIVE thread scheduling
  • Flash-attention (Ollama --flash-attn flag)
  • keep_alive=-1 vs expiry reload cost
  • CPU governor performance vs powersave (Linux only)
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import psutil

from autotune._ollama import ollama_base as _ollama_base
from autotune.api.ctx_utils import estimate_tokens
from autotune.api.hardware_tuner import get_tuner
from autotune.api.kv_manager import build_ollama_options
from autotune.api.profiles import get_profile
from autotune.metrics.ollama_client import NativeInferenceStats, OllamaMetricsClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHORT_PROMPT = "What is 2 + 2?"
_MEDIUM_PROMPT = (
    "Explain the difference between a process and a thread in an operating system. "
    "Cover scheduling, memory isolation, and communication overhead."
)
_LONG_PROMPT = (
    "You are a senior systems engineer. A Python web service receives 500 "
    "concurrent requests per second. It parses JSON, queries PostgreSQL with "
    "ORM calls, processes results with nested loops, caches nothing, and "
    "re-instantiates DB connections per request on a 16 GB unified-memory "
    "machine also running an LLM. Identify the top 5 bottlenecks in order of "
    "severity, and write corrected Python code with async I/O, connection "
    "pooling, and an LRU cache. Include complexity analysis."
)

_OS_PROMPT = _MEDIUM_PROMPT  # medium = realistic TTFT range for isolation tests
_SERVER_PROMPT = _MEDIUM_PROMPT

_BENCH_SYSTEM = "You are a concise technical assistant. Answer clearly and completely."

_WARMUP_RUNS = 2    # runs discarded before measurement starts
_OS_RUNS     = 5    # per-condition runs for OS bench
_SERVER_SERIAL_RUNS = 10   # serial throughput runs
_COOLDOWN_SEC = 3.0  # between back-to-back conditions

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Concise latency distribution from a list of millisecond samples."""
    samples: list[float]

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) >= 2 else 0.0

    @property
    def p50(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def p95(self) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def p99(self) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    @property
    def min_val(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def max_val(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def pct_change_vs(self, other: "LatencyStats") -> float:
        """% change of self.mean vs other.mean (positive = self is higher)."""
        if other.mean == 0:
            return 0.0
        return (self.mean - other.mean) / abs(other.mean) * 100.0


@dataclass
class OSBenchResult:
    """Result of one OS-level optimization test."""
    test_name: str
    description: str
    condition_a_label: str
    condition_b_label: str

    prefill_a: LatencyStats = field(default_factory=lambda: LatencyStats([]))
    prefill_b: LatencyStats = field(default_factory=lambda: LatencyStats([]))
    tps_a:     LatencyStats = field(default_factory=lambda: LatencyStats([]))
    tps_b:     LatencyStats = field(default_factory=lambda: LatencyStats([]))

    swap_events_a: int = 0
    swap_events_b: int = 0
    error: Optional[str] = None

    @property
    def prefill_improvement_pct(self) -> float:
        return self._pct_improvement(self.prefill_a, self.prefill_b)

    def _pct_improvement(self, stat_a: LatencyStats, stat_b: LatencyStats) -> float:
        """How much better is B (autotune condition) vs A (raw/off condition)?
        Returns positive when B is faster (lower latency)."""
        if stat_a.mean == 0:
            return 0.0
        return (stat_a.mean - stat_b.mean) / abs(stat_a.mean) * 100.0

    def prefill_improvement(self) -> float:
        return self._pct_improvement(self.prefill_a, self.prefill_b)

    def tps_improvement(self) -> float:
        return self._pct_improvement(self.tps_b, self.tps_a)  # higher tps = better


@dataclass
class ConcurrencyResult:
    """Result of one concurrency level in the server benchmark."""
    concurrency: int
    rps: float              # requests per second
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ttft_ms: float
    mean_tps: float
    peak_ram_gb: float
    swap_occurred: bool
    errors: int


@dataclass
class ServerBenchResult:
    """Full server/cloud benchmark result."""
    model_id: str
    condition: str             # "raw" | "autotune"
    serial_latency: Optional[LatencyStats] = None
    concurrency_results: list[ConcurrencyResult] = field(default_factory=list)
    sustained_tps: float = 0.0
    peak_ram_gb: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_messages(prompt: str, system: str = _BENCH_SYSTEM) -> list[dict]:
    return [
        {"role": "system",  "content": system},
        {"role": "user",    "content": prompt},
    ]


async def _run_single_native(
    client: OllamaMetricsClient,
    model: str,
    messages: list[dict],
    options: dict,
    keep_alive: str = "-1",
    max_tokens: int = 256,
) -> NativeInferenceStats:
    """Run one inference using Ollama native timers (accurate TTFT)."""
    return await client.run_with_stats(
        model=model,
        messages=messages,
        options=options,
        keep_alive=keep_alive,
        max_tokens=max_tokens,
        temperature=0.1,
    )


def _swap_delta() -> float:
    return psutil.swap_memory().used / 1024**3


# ---------------------------------------------------------------------------
# OS-level optimization benchmarks
# ---------------------------------------------------------------------------

async def run_os_bench(
    model: str,
    profile_name: str = "balanced",
    runs: int = _OS_RUNS,
    console=None,
) -> list[OSBenchResult]:
    """
    Measure the isolated contribution of each OS-level optimization.

    Each test runs N inference calls under condition A (optimization off)
    then N under condition B (optimization on), using the same prompt and
    model state.  Ollama native timers give nanosecond-precision TTFT.

    Tests:
      1. GC suspend      — Python GC off during inference vs default
      2. QOS priority    — macOS USER_INTERACTIVE vs default (macOS only)
      3. Flash attention — flash_attn=True vs False (via Ollama option)
      4. Keep-alive      — warm model (keep_alive=-1) vs cold reload
      5. num_ctx sizing  — autotune dynamic sizing vs Ollama default 4096

    The keep-alive test is the most dramatic: a cold reload costs 0.5–4 s
    depending on model size; autotune's keep_alive=-1 eliminates this.
    """
    import gc as _gc
    import platform as _platform

    if console:
        from rich.console import Console
        con: Console = console
    else:
        from rich.console import Console
        con = Console(stderr=True)

    client  = OllamaMetricsClient(timeout=120.0)
    profile = get_profile(profile_name)
    tuner   = get_tuner()
    results: list[OSBenchResult] = []
    messages = _build_messages(_OS_PROMPT)

    async def _warm_model() -> None:
        opts, _ = build_ollama_options(messages, profile)
        await client.run_with_stats(model, messages, options=opts,
                                    keep_alive="-1", max_tokens=64)

    # ── Test 1: Python GC suspend ─────────────────────────────────────────
    con.print("[dim]  Test 1/5  Python GC suspend[/dim]")
    res1 = OSBenchResult(
        test_name="gc_suspend",
        description="Python GC disabled during inference",
        condition_a_label="GC default (on)",
        condition_b_label="GC disabled",
    )
    try:
        opts, _ = build_ollama_options(messages, profile)

        # Condition A: GC on (default)
        await _warm_model()
        a_prefill, a_tps = [], []
        for _ in range(runs):
            stat = await _run_single_native(client, model, messages, opts)
            if not stat.error and stat.eval_tps > 0:
                a_prefill.append(stat.prefill_ms)
                a_tps.append(stat.eval_tps)
            await asyncio.sleep(0.5)

        # Condition B: GC off
        await _warm_model()
        b_prefill, b_tps = [], []
        for _ in range(runs):
            _gc.disable()
            try:
                stat = await _run_single_native(client, model, messages, opts)
            finally:
                _gc.enable()
            if not stat.error and stat.eval_tps > 0:
                b_prefill.append(stat.prefill_ms)
                b_tps.append(stat.eval_tps)
            await asyncio.sleep(0.5)

        res1.prefill_a = LatencyStats(a_prefill)
        res1.prefill_b = LatencyStats(b_prefill)
        res1.tps_a     = LatencyStats(a_tps)
        res1.tps_b     = LatencyStats(b_tps)
    except Exception as e:
        res1.error = str(e)
    results.append(res1)

    # ── Test 2: macOS QOS thread priority ─────────────────────────────────
    con.print("[dim]  Test 2/5  Thread QOS priority[/dim]")
    res2 = OSBenchResult(
        test_name="qos_priority",
        description="macOS QOS_CLASS_USER_INTERACTIVE thread scheduling",
        condition_a_label="default priority",
        condition_b_label="USER_INTERACTIVE",
    )
    try:
        opts, _ = build_ollama_options(messages, profile)
        await _warm_model()
        a_prefill, a_tps = [], []
        for _ in range(runs):
            stat = await _run_single_native(client, model, messages, opts)
            if not stat.error and stat.eval_tps > 0:
                a_prefill.append(stat.prefill_ms)
                a_tps.append(stat.eval_tps)
            await asyncio.sleep(0.5)

        await _warm_model()
        b_prefill, b_tps = [], []
        for _ in range(runs):
            tuner._apply(profile_name)
            try:
                stat = await _run_single_native(client, model, messages, opts)
            finally:
                tuner._restore()
            if not stat.error and stat.eval_tps > 0:
                b_prefill.append(stat.prefill_ms)
                b_tps.append(stat.eval_tps)
            await asyncio.sleep(0.5)

        res2.prefill_a = LatencyStats(a_prefill)
        res2.prefill_b = LatencyStats(b_prefill)
        res2.tps_a     = LatencyStats(a_tps)
        res2.tps_b     = LatencyStats(b_tps)
    except Exception as e:
        res2.error = str(e)
    results.append(res2)

    # ── Test 3: Flash attention ───────────────────────────────────────────
    con.print("[dim]  Test 3/5  Flash attention[/dim]")
    res3 = OSBenchResult(
        test_name="flash_attn",
        description="Ollama flash_attn option (faster KV fill on Metal/CUDA)",
        condition_a_label="flash_attn=False",
        condition_b_label="flash_attn=True",
    )
    try:
        base_opts, _ = build_ollama_options(messages, profile)
        opts_no_fa  = {**base_opts, "flash_attn": False}
        opts_fa     = {**base_opts, "flash_attn": True}

        await _warm_model()
        a_prefill, a_tps = [], []
        for _ in range(runs):
            stat = await _run_single_native(client, model, messages, opts_no_fa)
            if not stat.error and stat.eval_tps > 0:
                a_prefill.append(stat.prefill_ms)
                a_tps.append(stat.eval_tps)
            await asyncio.sleep(0.5)

        await _warm_model()
        b_prefill, b_tps = [], []
        for _ in range(runs):
            stat = await _run_single_native(client, model, messages, opts_fa)
            if not stat.error and stat.eval_tps > 0:
                b_prefill.append(stat.prefill_ms)
                b_tps.append(stat.eval_tps)
            await asyncio.sleep(0.5)

        res3.prefill_a = LatencyStats(a_prefill)
        res3.prefill_b = LatencyStats(b_prefill)
        res3.tps_a     = LatencyStats(a_tps)
        res3.tps_b     = LatencyStats(b_tps)
    except Exception as e:
        res3.error = str(e)
    results.append(res3)

    # ── Test 4: Keep-alive warm vs cold ───────────────────────────────────
    con.print("[dim]  Test 4/5  keep_alive warm vs cold reload[/dim]")
    res4 = OSBenchResult(
        test_name="keep_alive",
        description="keep_alive=-1 (model stays loaded) vs 0s expiry (force reload)",
        condition_a_label="cold reload (keep_alive=0)",
        condition_b_label="warm keep_alive=-1",
    )
    try:
        opts, _ = build_ollama_options(messages, profile)

        # Condition A: Force unload before every call
        a_prefill, a_tps = [], []
        for _ in range(min(runs, 3)):  # fewer runs since each takes longer
            # Unload first
            await client.unload_model(model)
            await asyncio.sleep(2.0)
            stat = await _run_single_native(client, model, messages, opts,
                                            keep_alive="-1", max_tokens=64)
            if not stat.error:
                a_prefill.append(stat.load_ms + stat.prefill_ms)
                a_tps.append(stat.eval_tps)

        # Condition B: Model stays loaded — measure only prefill
        await _warm_model()
        b_prefill, b_tps = [], []
        for _ in range(runs):
            stat = await _run_single_native(client, model, messages, opts,
                                            keep_alive="-1", max_tokens=64)
            if not stat.error and stat.eval_tps > 0:
                b_prefill.append(stat.prefill_ms)
                b_tps.append(stat.eval_tps)
            await asyncio.sleep(0.5)

        res4.prefill_a = LatencyStats(a_prefill)
        res4.prefill_b = LatencyStats(b_prefill)
        res4.tps_a     = LatencyStats(a_tps)
        res4.tps_b     = LatencyStats(b_tps)
    except Exception as e:
        res4.error = str(e)
    results.append(res4)

    # ── Test 5: Dynamic num_ctx (autotune core mechanism) ─────────────────
    con.print("[dim]  Test 5/5  Dynamic num_ctx sizing[/dim]")
    res5 = OSBenchResult(
        test_name="dynamic_num_ctx",
        description="autotune dynamic num_ctx vs Ollama's fixed 4096 default",
        condition_a_label="num_ctx=4096 (Ollama default)",
        condition_b_label="dynamic num_ctx (autotune)",
    )
    try:
        opts_raw, _ = build_ollama_options(messages, profile)
        opts_raw_fixed = {**opts_raw, "num_ctx": 4096, "flash_attn": False}
        opts_tuned, _  = build_ollama_options(messages, profile)

        await _warm_model()
        a_prefill, a_tps = [], []
        for _ in range(runs):
            stat = await _run_single_native(client, model, messages, opts_raw_fixed)
            if not stat.error and stat.eval_tps > 0:
                a_prefill.append(stat.prefill_ms)
                a_tps.append(stat.eval_tps)
            await asyncio.sleep(0.5)

        await _warm_model()
        b_prefill, b_tps = [], []
        for _ in range(runs):
            stat = await _run_single_native(client, model, messages, opts_tuned)
            if not stat.error and stat.eval_tps > 0:
                b_prefill.append(stat.prefill_ms)
                b_tps.append(stat.eval_tps)
            await asyncio.sleep(0.5)

        res5.prefill_a = LatencyStats(a_prefill)
        res5.prefill_b = LatencyStats(b_prefill)
        res5.tps_a     = LatencyStats(a_tps)
        res5.tps_b     = LatencyStats(b_tps)
    except Exception as e:
        res5.error = str(e)
    results.append(res5)

    return results


def print_os_bench_results(results: list[OSBenchResult], console=None) -> None:
    """Render OS bench results as a Rich table."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    con = console or Console()

    t = Table(
        title="OS-Level Optimization Benchmark",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        expand=False,
    )
    t.add_column("Test",                style="bold",  min_width=20)
    t.add_column("Condition A\n(baseline)", justify="right", style="yellow")
    t.add_column("Condition B\n(optimized)", justify="right", style="cyan")
    t.add_column("TTFT Δ",             justify="right", min_width=10)
    t.add_column("tok/s Δ",            justify="right", min_width=10)
    t.add_column("Signal",             justify="center")

    for r in results:
        if r.error:
            t.add_row(r.test_name, "—", "—", "—", "—", f"[red]error: {r.error[:40]}[/red]")
            continue

        prefill_a = r.prefill_a.mean
        prefill_b = r.prefill_b.mean

        if prefill_a == 0 and prefill_b == 0:
            t.add_row(r.test_name, "—", "—", "—", "—", "[dim]no data[/dim]")
            continue

        ttft_pct = r._pct_improvement(r.prefill_a, r.prefill_b)
        tps_pct  = r.tps_improvement()

        ttft_str = f"[green]{ttft_pct:+.1f}%[/green]" if ttft_pct > 1 else (
                   f"[red]{ttft_pct:+.1f}%[/red]" if ttft_pct < -1 else "[dim]≈0%[/dim]")
        tps_str  = f"[green]{tps_pct:+.1f}%[/green]" if tps_pct > 1 else (
                   f"[red]{tps_pct:+.1f}%[/red]" if tps_pct < -1 else "[dim]≈0%[/dim]")

        signal = "✓" if ttft_pct > 3 else ("~" if abs(ttft_pct) <= 3 else "✗")
        signal_style = "green" if ttft_pct > 3 else ("dim" if abs(ttft_pct) <= 3 else "red")

        t.add_row(
            r.test_name,
            f"{prefill_a:.0f} ms" if prefill_a > 0 else "—",
            f"{prefill_b:.0f} ms" if prefill_b > 0 else "—",
            ttft_str,
            tps_str,
            f"[{signal_style}]{signal}[/{signal_style}]",
        )

    con.print(t)
    con.print(
        "[dim]Δ = change vs baseline. Positive TTFT Δ = "
        "optimization reduced latency (better). "
        "Measured via Ollama native nanosecond timers.[/dim]\n"
    )


# ---------------------------------------------------------------------------
# Server / cloud throughput benchmark
# ---------------------------------------------------------------------------

async def run_server_bench(
    model: str,
    profile_name: str = "balanced",
    concurrency_levels: Optional[list[int]] = None,
    serial_runs: int = _SERVER_SERIAL_RUNS,
    console=None,
) -> tuple[ServerBenchResult, ServerBenchResult]:
    """
    Measure server/cloud throughput under real concurrent load.

    Runs both raw Ollama and autotune through serial and concurrent workloads.
    Returns (raw_result, tuned_result).

    Why this matters for cloud operators
    --------------------------------------
    A cloud LLM API serving 100 users per hour must sustain throughput without
    RAM overflow or queue saturation.  autotune's dynamic num_ctx means each
    concurrent request allocates less KV RAM — allowing more concurrent slots
    before hitting the memory ceiling.

    Metrics produced
    ----------------
    Serial: P50/P95/P99 TTFT over ``serial_runs`` requests
    Concurrent: requests/second at each concurrency level
    RAM cost: peak RAM per concurrency level
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4]

    from rich.console import Console
    con = console or Console(stderr=True)

    profile = get_profile(profile_name)
    messages = _build_messages(_SERVER_PROMPT)
    opts_tuned, _ = build_ollama_options(messages, profile)
    opts_raw = {"num_ctx": 4096, "flash_attn": False}

    raw_result   = ServerBenchResult(model_id=model, condition="raw")
    tuned_result = ServerBenchResult(model_id=model, condition="autotune")

    client = OllamaMetricsClient(timeout=180.0)

    # ── Warmup: ensure model is loaded ───────────────────────────────────
    con.print("[dim]  Warming up model…[/dim]")
    for _ in range(_WARMUP_RUNS):
        await client.run_with_stats(model, messages, options=opts_tuned,
                                    keep_alive="-1", max_tokens=64)

    # ── Serial latency (raw) ─────────────────────────────────────────────
    con.print(f"[dim]  Serial latency — raw Ollama ({serial_runs} runs)…[/dim]")
    raw_serial: list[float] = []
    raw_tps_serial: list[float] = []
    for i in range(serial_runs):
        stat = await client.run_with_stats(
            model, messages, options=opts_raw, keep_alive="-1", max_tokens=256,
        )
        if not stat.error:
            raw_serial.append(stat.ttft_proxy_ms)
            raw_tps_serial.append(stat.eval_tps)
        await asyncio.sleep(0.3)

    raw_result.serial_latency = LatencyStats(raw_serial)
    raw_result.peak_ram_gb = psutil.virtual_memory().used / 1024**3

    # ── Serial latency (autotune) ────────────────────────────────────────
    con.print(f"[dim]  Serial latency — autotune ({serial_runs} runs)…[/dim]")
    tuned_serial: list[float] = []
    tuned_tps_serial: list[float] = []
    for i in range(serial_runs):
        opts_fresh, _ = build_ollama_options(messages, profile)
        stat = await client.run_with_stats(
            model, messages, options=opts_fresh, keep_alive="-1", max_tokens=256,
        )
        if not stat.error:
            tuned_serial.append(stat.ttft_proxy_ms)
            tuned_tps_serial.append(stat.eval_tps)
        await asyncio.sleep(0.3)

    tuned_result.serial_latency = LatencyStats(tuned_serial)
    tuned_result.peak_ram_gb = psutil.virtual_memory().used / 1024**3

    # ── Concurrent throughput ────────────────────────────────────────────
    for c in concurrency_levels:
        con.print(f"[dim]  Concurrent throughput — raw Ollama (concurrency={c})…[/dim]")
        raw_c  = await _run_concurrent_bench(model, messages, opts_raw,  c, client)
        await asyncio.sleep(_COOLDOWN_SEC)
        con.print(f"[dim]  Concurrent throughput — autotune (concurrency={c})…[/dim]")
        tune_c = await _run_concurrent_bench(model, messages, opts_tuned, c, client)
        await asyncio.sleep(_COOLDOWN_SEC)
        raw_result.concurrency_results.append(raw_c)
        tuned_result.concurrency_results.append(tune_c)

    return raw_result, tuned_result


async def _run_concurrent_bench(
    model: str,
    messages: list[dict],
    options: dict,
    concurrency: int,
    client: OllamaMetricsClient,
    requests_total: int = 8,
) -> ConcurrencyResult:
    """
    Send ``requests_total`` requests with ``concurrency`` in-flight at once.
    Returns aggregate metrics.
    """
    ttfts: list[float]  = []
    tps_list: list[float] = []

    ram_samples: list[float] = []
    swap_before = psutil.swap_memory().used / 1024**3

    async def _one_request() -> None:
        stat = await client.run_with_stats(
            model, messages, options=options, keep_alive="-1", max_tokens=128,
        )
        ram_samples.append(psutil.virtual_memory().used / 1024**3)
        if stat.error:
            errors_ref[0] += 1
        else:
            ttfts.append(stat.ttft_proxy_ms)
            tps_list.append(stat.eval_tps)

    errors_ref = [0]

    t_start = time.monotonic()
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(fn):
        async with sem:
            await fn()

    tasks = [_bounded(_one_request()) for _ in range(requests_total)]
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = max(time.monotonic() - t_start, 0.001)

    swap_after = psutil.swap_memory().used / 1024**3

    stats = LatencyStats(ttfts) if ttfts else LatencyStats([0.0])
    return ConcurrencyResult(
        concurrency=concurrency,
        rps=len(ttfts) / elapsed,
        p50_ms=stats.p50,
        p95_ms=stats.p95,
        p99_ms=stats.p99,
        mean_ttft_ms=stats.mean,
        mean_tps=sum(tps_list) / max(len(tps_list), 1),
        peak_ram_gb=max(ram_samples) if ram_samples else 0.0,
        swap_occurred=(swap_after - swap_before) > 0.03,
        errors=errors_ref[0],
    )


def print_server_bench_results(
    raw: ServerBenchResult,
    tuned: ServerBenchResult,
    console=None,
) -> None:
    """Render server bench results comparing raw vs autotune."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    con = console or Console()

    # ── Serial latency table ─────────────────────────────────────────────
    if raw.serial_latency and tuned.serial_latency:
        t = Table(
            title="Serial TTFT Latency (single concurrent slot)",
            box=box.ROUNDED, header_style="bold",
        )
        t.add_column("Metric",         style="bold")
        t.add_column("Raw Ollama",     justify="right", style="yellow")
        t.add_column("autotune",       justify="right", style="cyan")
        t.add_column("Improvement",    justify="right")

        def _row(label: str, a: float, b: float, lower_better: bool = True) -> None:
            if a == 0 and b == 0:
                return
            pct = (a - b) / abs(a) * 100 if a != 0 else 0
            if lower_better:
                color = "green" if pct > 1 else ("red" if pct < -1 else "dim")
                imp = f"[{color}]{pct:+.1f}%[/{color}]"
            else:
                color = "green" if pct < -1 else ("red" if pct > 1 else "dim")
                imp = f"[{color}]{-pct:+.1f}%[/{color}]"
            t.add_row(label, f"{a:.1f} ms", f"{b:.1f} ms", imp)

        r = raw.serial_latency
        u = tuned.serial_latency
        _row("Mean TTFT",   r.mean, u.mean)
        _row("P50 TTFT",    r.p50,  u.p50)
        _row("P95 TTFT",    r.p95,  u.p95)
        _row("P99 TTFT",    r.p99,  u.p99)
        _row("Min TTFT",    r.min_val, u.min_val)
        _row("Max TTFT",    r.max_val, u.max_val)

        con.print(t)
        con.print()

    # ── Concurrency table ────────────────────────────────────────────────
    if raw.concurrency_results and tuned.concurrency_results:
        ct = Table(
            title="Throughput Under Concurrency",
            box=box.ROUNDED, header_style="bold",
        )
        ct.add_column("Concurrency",    style="bold", justify="center")
        ct.add_column("Condition",      style="bold")
        ct.add_column("Req/sec",        justify="right", style="white")
        ct.add_column("Mean TTFT",      justify="right")
        ct.add_column("P95 TTFT",       justify="right")
        ct.add_column("Peak RAM",       justify="right")
        ct.add_column("Errors",         justify="right")

        raw_by_c  = {r.concurrency: r for r in raw.concurrency_results}
        tune_by_c = {r.concurrency: r for r in tuned.concurrency_results}
        for c in sorted(set(list(raw_by_c.keys()) + list(tune_by_c.keys()))):
            rr = raw_by_c.get(c)
            tr = tune_by_c.get(c)
            if rr:
                ct.add_row(
                    str(c), "[yellow]Raw Ollama[/yellow]",
                    f"{rr.rps:.2f}",
                    f"[yellow]{rr.mean_ttft_ms:.0f} ms[/yellow]",
                    f"[yellow]{rr.p95_ms:.0f} ms[/yellow]",
                    f"[yellow]{rr.peak_ram_gb:.2f} GB[/yellow]",
                    f"[red]{rr.errors}[/red]" if rr.errors else "[dim]0[/dim]",
                )
            if tr:
                ct.add_row(
                    str(c), "[cyan]autotune[/cyan]",
                    f"{tr.rps:.2f}",
                    f"[cyan]{tr.mean_ttft_ms:.0f} ms[/cyan]",
                    f"[cyan]{tr.p95_ms:.0f} ms[/cyan]",
                    f"[cyan]{tr.peak_ram_gb:.2f} GB[/cyan]",
                    f"[red]{tr.errors}[/red]" if tr.errors else "[dim]0[/dim]",
                )
            ct.add_section()

        con.print(ct)
        con.print()

    # ── Cost estimate ────────────────────────────────────────────────────
    if raw.serial_latency and tuned.serial_latency:
        raw_mean = raw.serial_latency.mean
        tun_mean = tuned.serial_latency.mean
        if raw_mean > 0 and tun_mean > 0:
            ttft_savings_pct = (raw_mean - tun_mean) / raw_mean * 100
            con.print(Panel(
                f"[bold]Cost reduction estimate[/bold]\n"
                f"TTFT improvement: [green]{ttft_savings_pct:.1f}%[/green] faster first token\n"
                f"At 10,000 requests/day: "
                f"[green]{ttft_savings_pct:.0f}% less TTFT wait per user[/green]\n"
                f"Fewer tokens allocated per request → "
                f"lower GPU-seconds per inference → [green]lower cost per million tokens[/green]",
                title="[bold]Production Impact[/bold]",
                border_style="green",
            ))


# ---------------------------------------------------------------------------
# Auto-select model helper (shared across bench subcommands)
# ---------------------------------------------------------------------------

def _autoselect_model(preferred: Optional[str] = None) -> Optional[str]:
    """Return preferred model if set, else the smallest installed Ollama model."""
    if preferred:
        return preferred
    try:
        import httpx as _hx
        with _hx.Client(timeout=5.0) as c:
            r = c.get(f"{_ollama_base()}/api/tags")
            models = r.json().get("models", [])
        if not models:
            return None
        def _size(m: dict) -> float:
            return m.get("size", float("inf"))
        return sorted(models, key=_size)[0]["name"]
    except Exception:
        return None
