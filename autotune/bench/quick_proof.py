"""
autotune quick proof — does autotune actually help on YOUR machine?

Runs in ≤30 seconds.  Measures the metrics users care about:
  • Time to first word (TTFT)    — load_ms + prefill_ms from Ollama's Go timer
  • Context window (num_ctx)     — autotune dynamic vs Ollama fixed 4096
  • KV cache size                — estimated from model architecture
  • RAM free for other apps      — system available memory floor
  • Swap events                  — any swap pressure during inference (goal: 0)
  • Generation speed (tok/s)     — honestly reported; autotune does NOT change this

Uses the same Ollama-native timers as proof_suite so numbers are comparable.

Design constraints
------------------
  - 1 probe prompt × N runs × 2 conditions = total ≤ 30s for warm models
  - Prompt chosen to maximise KV savings (multi-turn with system prompt)
  - max_tokens capped at 120 to keep generation time short
  - Honest: tok/s is reported and labelled "unchanged (expected)" when flat
"""
from __future__ import annotations

import asyncio
import json
import statistics
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx
import psutil

_OLLAMA_BASE    = "http://localhost:11434"
_RAW_CTX        = 4096
_KEEP_ALIVE     = "30m"
_MAX_TOKENS     = 120      # keep generation fast; TTFT is what we measure
_RELOAD_MS      = 400.0    # load_ms above this → cold reload happened
_COOLDOWN_SEC   = 2.0      # pause between raw and autotune conditions

# ─────────────────────────────────────────────────────────────────────────────
# Proof prompt — multi-turn with a substantive system prompt.
#
# Why this prompt?
#   - System prompt (~22 tokens) → num_keep pins them; prefix cache kicks in
#   - Two prior turns (~100 tokens) → accumulated context → dynamic num_ctx
#     will be ~350-450 tokens vs raw 4096, showing maximum TTFT improvement
#   - The final question is open-ended → model generates real text, not one word
# ─────────────────────────────────────────────────────────────────────────────
PROOF_MESSAGES: list[dict] = [
    {
        "role": "system",
        "content": (
            "You are a senior software engineer with 15 years of production experience. "
            "Give concise, precise answers with real examples."
        ),
    },
    {
        "role": "user",
        "content": "When should I use Redis instead of in-process caching?",
    },
    {
        "role": "assistant",
        "content": (
            "Use in-process caching (e.g. lru_cache) when data is per-instance "
            "and you only have one process. Use Redis for shared state across "
            "multiple processes or replicas, atomic operations, pub/sub, or TTL "
            "eviction. The cost: ~1 ms network RTT and a serialization step."
        ),
    },
    {
        "role": "user",
        "content": "Our API has 8 replicas behind a load balancer. Which should we use?",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal RAM sampler (system-level, not process-isolated)
# ─────────────────────────────────────────────────────────────────────────────

class _RamSampler:
    """Samples system-available RAM and swap every 100ms in a background thread."""

    def __init__(self) -> None:
        self._free: list[float] = []
        self._swap_start = 0.0
        self._swap_end   = 0.0
        self._running    = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._swap_start = psutil.swap_memory().used / 1024 ** 3
        vm = psutil.virtual_memory()
        self._free.append(vm.available / 1024 ** 3)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._swap_end = psutil.swap_memory().used / 1024 ** 3

    def _loop(self) -> None:
        while self._running:
            vm = psutil.virtual_memory()
            self._free.append(vm.available / 1024 ** 3)
            time.sleep(0.1)

    def free_floor_gb(self) -> float:
        return min(self._free) if self._free else 0.0

    def swap_delta_gb(self) -> float:
        return max(0.0, self._swap_end - self._swap_start)

    def swap_occurred(self) -> bool:
        return self.swap_delta_gb() > 0.031   # 32 MB threshold


# ─────────────────────────────────────────────────────────────────────────────
# Single-run result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RunResult:
    condition:          str    # "raw" | "autotune"
    num_ctx:            int
    prefill_ms:         float  # prompt_eval_duration from Ollama Go timer
    load_ms:            float  # load_duration (KV alloc phase)
    eval_tps:           float  # generation tok/s
    eval_count:         int    # tokens generated
    prompt_tokens:      int    # tokens in prompt (Ollama-reported)
    kv_cache_mb:        float  # estimated KV size for this num_ctx
    free_floor_gb:      float  # min system available RAM during run
    swap_occurred:      bool
    reload_detected:    bool   # load_ms > threshold → model was evicted
    error: Optional[str] = None

    @property
    def ttft_ms(self) -> float:
        return self.load_ms + self.prefill_ms

    @property
    def ok(self) -> bool:
        return self.error is None and self.eval_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# KV cache estimator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ModelArch:
    n_layers:   int = 0
    n_kv_heads: int = 0
    head_dim:   int = 0

    def kv_cache_mb(self, num_ctx: int, dtype_bytes: int = 2) -> float:
        if not (self.n_layers and self.n_kv_heads and self.head_dim):
            return 0.0
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * num_ctx * dtype_bytes / (1024 ** 2)


async def _fetch_arch(model_id: str) -> _ModelArch:
    arch = _ModelArch()
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.post(f"{_OLLAMA_BASE}/api/show", json={"name": model_id, "verbose": True})
        mi = r.json().get("model_info", {})

        def _find(suffix: str) -> int:
            for k, v in mi.items():
                if k.endswith(suffix) and isinstance(v, (int, float)):
                    return int(v)
            return 0

        arch.n_layers   = _find(".block_count")
        arch.n_kv_heads = _find(".attention.head_count_kv") or _find(".attention.kv_heads")
        arch.head_dim   = _find(".attention.head_dim") or _find(".attention.key_length")
        if not arch.head_dim:
            emb   = _find(".embedding_length")
            heads = _find(".attention.head_count")
            if emb and heads:
                arch.head_dim = emb // heads
    except Exception:
        pass
    return arch


# ─────────────────────────────────────────────────────────────────────────────
# Ollama HTTP call
# ─────────────────────────────────────────────────────────────────────────────

async def _call_ollama(
    model_id: str,
    messages: list[dict],
    options: dict,
    max_tokens: int = _MAX_TOKENS,
) -> dict:
    payload = {
        "model":      model_id,
        "messages":   messages,
        "stream":     False,
        "options":    {**options, "num_predict": max_tokens},
        "keep_alive": _KEEP_ALIVE,
    }
    async with httpx.AsyncClient(timeout=90.0) as c:
        r = await c.post(f"{_OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Single instrumented run
# ─────────────────────────────────────────────────────────────────────────────

async def _measured_run(
    model_id: str,
    condition: str,
    options: dict,
    arch: _ModelArch,
    kv_dtype_bytes: int,
) -> _RunResult:
    sampler = _RamSampler()
    sampler.start()
    error = None

    try:
        data = await _call_ollama(model_id, PROOF_MESSAGES, options)
        load_ms    = data.get("load_duration",         0) / 1_000_000
        prefill_ms = data.get("prompt_eval_duration",  0) / 1_000_000
        total_ms   = data.get("total_duration",        0) / 1_000_000
        eval_cnt   = data.get("eval_count",            0)
        eval_dur   = data.get("eval_duration",         0) / 1_000_000_000
        prompt_cnt = data.get("prompt_eval_count",     0)
        tps        = eval_cnt / eval_dur if eval_dur > 0 else 0.0
    except Exception as exc:
        error     = str(exc)
        load_ms   = prefill_ms = tps = 0.0
        eval_cnt  = prompt_cnt = 0
    finally:
        sampler.stop()

    num_ctx = options.get("num_ctx", _RAW_CTX)
    return _RunResult(
        condition       = condition,
        num_ctx         = num_ctx,
        prefill_ms      = prefill_ms,
        load_ms         = load_ms,
        eval_tps        = tps,
        eval_count      = eval_cnt,
        prompt_tokens   = prompt_cnt,
        kv_cache_mb     = arch.kv_cache_mb(num_ctx, kv_dtype_bytes),
        free_floor_gb   = sampler.free_floor_gb(),
        swap_occurred   = sampler.swap_occurred(),
        reload_detected = (load_ms > _RELOAD_MS),
        error           = error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuickProofResult:
    model_id:       str
    profile_name:   str
    n_runs:         int
    elapsed_sec:    float
    raw_runs:       list[_RunResult]
    tuned_runs:     list[_RunResult]

    # ── Aggregated stats ──────────────────────────────────────────────────────

    def _mean(self, runs: list[_RunResult], attr: str) -> float:
        vals = [getattr(r, attr) for r in runs if r.ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def raw_ttft_ms(self) -> float:
        return self._mean(self.raw_runs, "ttft_ms")

    @property
    def tuned_ttft_ms(self) -> float:
        return self._mean(self.tuned_runs, "ttft_ms")

    @property
    def ttft_improvement_pct(self) -> float:
        if self.raw_ttft_ms <= 0:
            return 0.0
        return (self.raw_ttft_ms - self.tuned_ttft_ms) / self.raw_ttft_ms * 100

    @property
    def raw_kv_mb(self) -> float:
        return self._mean(self.raw_runs, "kv_cache_mb")

    @property
    def tuned_kv_mb(self) -> float:
        return self._mean(self.tuned_runs, "kv_cache_mb")

    @property
    def kv_pct(self) -> float:
        if self.raw_kv_mb <= 0:
            return 0.0
        return (self.raw_kv_mb - self.tuned_kv_mb) / self.raw_kv_mb * 100

    @property
    def raw_num_ctx(self) -> int:
        ok = [r for r in self.raw_runs if r.ok]
        return ok[0].num_ctx if ok else _RAW_CTX

    @property
    def tuned_num_ctx(self) -> int:
        ok = [r for r in self.tuned_runs if r.ok]
        return ok[0].num_ctx if ok else 0

    @property
    def ctx_pct(self) -> float:
        if self.raw_num_ctx <= 0:
            return 0.0
        return (self.raw_num_ctx - self.tuned_num_ctx) / self.raw_num_ctx * 100

    @property
    def raw_free_gb(self) -> float:
        return self._mean(self.raw_runs, "free_floor_gb")

    @property
    def tuned_free_gb(self) -> float:
        return self._mean(self.tuned_runs, "free_floor_gb")

    @property
    def raw_swap_events(self) -> int:
        return sum(1 for r in self.raw_runs if r.swap_occurred)

    @property
    def tuned_swap_events(self) -> int:
        return sum(1 for r in self.tuned_runs if r.swap_occurred)

    @property
    def raw_tps(self) -> float:
        return self._mean(self.raw_runs, "eval_tps")

    @property
    def tuned_tps(self) -> float:
        return self._mean(self.tuned_runs, "eval_tps")

    def to_dict(self) -> dict:
        return {
            "model_id":       self.model_id,
            "profile":        self.profile_name,
            "n_runs":         self.n_runs,
            "elapsed_sec":    round(self.elapsed_sec, 1),
            "ttft_raw_ms":    round(self.raw_ttft_ms, 1),
            "ttft_tuned_ms":  round(self.tuned_ttft_ms, 1),
            "ttft_pct":       round(self.ttft_improvement_pct, 1),
            "kv_raw_mb":      round(self.raw_kv_mb, 1),
            "kv_tuned_mb":    round(self.tuned_kv_mb, 1),
            "kv_pct":         round(self.kv_pct, 1),
            "ctx_raw":        self.raw_num_ctx,
            "ctx_tuned":      self.tuned_num_ctx,
            "ctx_pct":        round(self.ctx_pct, 1),
            "free_raw_gb":    round(self.raw_free_gb, 2),
            "free_tuned_gb":  round(self.tuned_free_gb, 2),
            "swap_raw":       self.raw_swap_events,
            "swap_tuned":     self.tuned_swap_events,
            "tps_raw":        round(self.raw_tps, 1),
            "tps_tuned":      round(self.tuned_tps, 1),
            "raw_runs":       [asdict(r) for r in self.raw_runs],
            "tuned_runs":     [asdict(r) for r in self.tuned_runs],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_quick_proof(
    model_id:     str,
    profile_name: str = "balanced",
    n_runs:       int = 2,
    output_path:  Optional[Path] = None,
    on_step:      Optional[callable] = None,
) -> QuickProofResult:
    """
    Run the quick proof benchmark.

    Parameters
    ----------
    model_id     : Ollama model tag (e.g. "qwen3:8b")
    profile_name : autotune profile name ("fast" | "balanced" | "quality")
    n_runs       : runs per condition (default 2, total ≤ 30s for warm models)
    output_path  : if provided, save JSON results there
    on_step      : optional callback(str) for progress messages

    Returns
    -------
    QuickProofResult with all metrics populated
    """
    from autotune.api.kv_manager import build_ollama_options
    from autotune.api.profiles import get_profile

    def _step(msg: str) -> None:
        if on_step:
            on_step(msg)

    profile = get_profile(profile_name)

    t0 = time.monotonic()

    # Fetch model architecture for KV estimation (in parallel with warmup)
    _step("Fetching model architecture…")
    arch = await _fetch_arch(model_id)

    # ── Warmup: ensure model is loaded, discard cold-start cost ─────────────
    _step("Warming up model…")
    try:
        await _call_ollama(
            model_id,
            [{"role": "user", "content": "Say exactly: ready"}],
            {"num_ctx": _RAW_CTX, "num_predict": 4},
        )
        await asyncio.sleep(0.5)
    except Exception:
        pass

    # ── Raw baseline (Ollama factory defaults) ───────────────────────────────
    raw_opts    = {"num_ctx": _RAW_CTX}
    raw_runs: list[_RunResult] = []

    for i in range(n_runs):
        _step(f"Raw Ollama  run {i + 1}/{n_runs}…")
        r = await _measured_run(model_id, "raw", raw_opts, arch, kv_dtype_bytes=2)
        raw_runs.append(r)
        if i < n_runs - 1:
            await asyncio.sleep(0.5)

    # ── Brief cooldown before switching conditions ────────────────────────────
    _step("Switching to autotune…")
    await asyncio.sleep(_COOLDOWN_SEC)

    # ── autotune condition ────────────────────────────────────────────────────
    tuned_opts, _ = build_ollama_options(PROOF_MESSAGES, profile)
    kv_dtype = 1 if tuned_opts.get("kv_cache_type") else 2
    tuned_runs: list[_RunResult] = []

    for i in range(n_runs):
        _step(f"autotune    run {i + 1}/{n_runs}…")
        r = await _measured_run(model_id, "autotune", tuned_opts, arch, kv_dtype_bytes=kv_dtype)
        tuned_runs.append(r)
        if i < n_runs - 1:
            await asyncio.sleep(0.5)

    elapsed = time.monotonic() - t0

    result = QuickProofResult(
        model_id=model_id,
        profile_name=profile_name,
        n_runs=n_runs,
        elapsed_sec=elapsed,
        raw_runs=raw_runs,
        tuned_runs=tuned_runs,
    )

    if output_path:
        output_path.write_text(json.dumps(result.to_dict(), indent=2))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Rich console output
# ─────────────────────────────────────────────────────────────────────────────

def print_proof_result(result: QuickProofResult, console, output_path: Optional[Path] = None) -> None:
    """Print the formatted proof result table to a Rich console."""
    from rich.table import Table
    from rich import box as _box
    from rich.rule import Rule
    from rich.text import Text

    W = 72

    console.print()
    console.rule(f"[bold]autotune proof[/bold]  ·  [cyan]{result.model_id}[/cyan]  ·  {result.n_runs} runs  ·  {result.elapsed_sec:.0f}s")
    console.print()

    ok_raw   = any(r.ok for r in result.raw_runs)
    ok_tuned = any(r.ok for r in result.tuned_runs)

    if not ok_raw or not ok_tuned:
        console.print("[red]One or more runs failed. Check that Ollama is running and the model is installed.[/red]")
        return

    def _pct_label(pct: float, better_is_lower: bool = True) -> str:
        if abs(pct) < 1:
            return "unchanged ✓"
        sign = "−" if pct > 0 else "+"
        icon = "✅" if (pct > 0) == better_is_lower else "⚠️"
        return f"{sign}{abs(pct):.0f}% {icon}"

    def _abs_label(raw: float, tuned: float, unit: str, better_is_higher: bool = False) -> str:
        delta = tuned - raw
        if abs(delta) < 0.01:
            return "unchanged ✓"
        sign = "+" if delta > 0 else "−"
        icon = "✅" if (delta > 0) == better_is_higher else "⚠️"
        return f"{sign}{abs(delta):.1f} {unit} {icon}"

    t = Table(box=_box.SIMPLE_HEAD, show_header=True, show_edge=False, pad_edge=False)
    t.add_column("Metric",                     style="white",       width=32)
    t.add_column("Raw Ollama",                 style="dim white",   justify="right", width=13)
    t.add_column("autotune",                   style="green",       justify="right", width=13)
    t.add_column("Change",                                          justify="right", width=18)

    # TTFT
    t.add_row(
        "Time to first word (TTFT)",
        f"{result.raw_ttft_ms:.0f} ms",
        f"{result.tuned_ttft_ms:.0f} ms",
        _pct_label(result.ttft_improvement_pct),
    )

    # Context window
    t.add_row(
        "Context window (num_ctx)",
        f"{result.raw_num_ctx:,}",
        f"{result.tuned_num_ctx:,}",
        _pct_label(result.ctx_pct),
    )

    # KV cache
    kv_row_raw   = f"{result.raw_kv_mb:.0f} MB"   if result.raw_kv_mb   > 0 else "N/A"
    kv_row_tuned = f"{result.tuned_kv_mb:.0f} MB" if result.tuned_kv_mb > 0 else "N/A"
    kv_change    = _pct_label(result.kv_pct) if result.raw_kv_mb > 0 else "N/A"
    t.add_row("KV cache size (est.)", kv_row_raw, kv_row_tuned, kv_change)

    # RAM headroom
    t.add_row(
        "RAM free for other apps",
        f"{result.raw_free_gb:.1f} GB",
        f"{result.tuned_free_gb:.1f} GB",
        _abs_label(result.raw_free_gb, result.tuned_free_gb, "GB", better_is_higher=True),
    )

    # Swap
    swap_raw_str   = str(result.raw_swap_events)   + (" ✅" if result.raw_swap_events == 0   else " ⚠️")
    swap_tuned_str = str(result.tuned_swap_events) + (" ✅" if result.tuned_swap_events == 0 else " ⚠️")
    swap_change = "— ✅" if result.tuned_swap_events == 0 else f"+{result.tuned_swap_events} ⚠️"
    t.add_row("Swap events", swap_raw_str, swap_tuned_str, swap_change)

    # Generation speed — honest
    tps_change: str
    if abs(result.raw_tps) < 0.1:
        tps_change = "N/A"
    elif abs(result.tuned_tps - result.raw_tps) / max(result.raw_tps, 0.01) < 0.05:
        tps_change = "unchanged ✓"
    else:
        delta_pct = (result.tuned_tps - result.raw_tps) / result.raw_tps * 100
        icon = "✅" if delta_pct > 0 else "—"
        tps_change = f"{'+'  if delta_pct > 0 else '−'}{abs(delta_pct):.0f}% {icon}"

    t.add_row(
        "Generation speed (tok/s)",
        f"{result.raw_tps:.1f}" if result.raw_tps > 0 else "N/A",
        f"{result.tuned_tps:.1f}" if result.tuned_tps > 0 else "N/A",
        tps_change,
    )

    console.print(t)
    console.print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    lines: list[str] = []

    if result.ttft_improvement_pct >= 5:
        lines.append(
            f"[green]✅[/green]  autotune reduced time-to-first-word by "
            f"[bold]{result.ttft_improvement_pct:.0f}%[/bold] "
            f"([dim]{result.raw_ttft_ms:.0f} ms → {result.tuned_ttft_ms:.0f} ms[/dim])"
        )
    elif result.ttft_improvement_pct > 0:
        lines.append(
            f"[yellow]→[/yellow]  TTFT improved modestly ({result.ttft_improvement_pct:.0f}%) — "
            f"model may already be warm or RAM is plentiful."
        )
    else:
        lines.append(
            f"[yellow]⚠[/yellow]  TTFT did not improve. "
            f"Try a larger context prompt or ensure the model was warmed identically."
        )

    if result.raw_kv_mb > 0 and result.kv_pct >= 5:
        kv_freed = result.raw_kv_mb - result.tuned_kv_mb
        lines.append(
            f"[green]✅[/green]  {kv_freed:.0f} MB of KV cache freed "
            f"([dim]{result.raw_kv_mb:.0f} MB → {result.tuned_kv_mb:.0f} MB, "
            f"−{result.kv_pct:.0f}%[/dim])"
        )

    free_delta = result.tuned_free_gb - result.raw_free_gb
    if free_delta >= 0.05:
        lines.append(
            f"[green]✅[/green]  {free_delta:.1f} GB more RAM free for your other apps during inference"
        )

    if result.tuned_swap_events == 0:
        lines.append("[green]✅[/green]  Zero swap events — your computer won't feel the LLM running")
    else:
        lines.append(
            f"[yellow]⚠[/yellow]  {result.tuned_swap_events} swap event(s) — "
            f"consider a smaller model or --profile fast"
        )

    # Honest generation speed note
    if abs(result.raw_tps) > 0.1:
        lines.append(
            "[dim]ℹ  Generation speed is GPU/CPU-bound — autotune does not change it[/dim]"
        )

    console.rule("[dim]Verdict[/dim]")
    for line in lines:
        console.print(f"  {line}")
    console.print()

    # ── Footer ────────────────────────────────────────────────────────────────
    if output_path:
        console.print(f"  [dim]Results saved → {output_path}[/dim]")

    console.print(
        "  [dim]For statistical significance (Wilcoxon, Cohen's d): "
        "[bold]autotune proof-suite[/bold][/dim]"
    )
    console.print()
    console.rule()
