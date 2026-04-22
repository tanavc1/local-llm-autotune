#!/usr/bin/env python3
"""
autotune proof suite v2.0 — multi-model scientific benchmark.

Measures the real-world impact of autotune versus raw Ollama defaults.
Uses Ollama's *internal* timers (not Python clock estimates) for TTFT and
throughput, and tracks the Ollama runner process in isolation for RAM and swap.

KPIs measured (all per-condition, per-turn where applicable):
  • TTFT                    — load_ms + prefill_ms (ms before first token appears)
  • Prefill time            — prompt_eval_duration from Ollama native timer (ms)
  • Total response time     — total_duration from Ollama native timer (ms)
  • Peak RAM (LLM process)  — Ollama runner RSS at inference peak (GB)
  • KV cache size (est.)    — estimated from num_ctx × model architecture (MB)
  • Total context size      — actual num_ctx + prompt + output token counts
  • Memory growth over turns— RAM delta across N sequential turns (GB/turn)
  • Did swap occur          — psutil.swap_memory() delta per run
  • Model reload count      — load_ms > threshold → model was evicted & reloaded
  • Context size per request— autotune's dynamic num_ctx vs raw 4096
  • Tokens saved            — KV tokens freed by context shrinkage (tokens × runs)

Statistics: Wilcoxon signed-rank (n<10) or paired t-test (n≥10)
            + Cohen's d effect sizes + 95% CI on mean difference.

Default models: llama3.2:3b  gemma4:e2b  qwen3:8b  (run one at a time)
Override with --models or AUTOTUNE_PROOF_MODELS env var.

Usage
-----
    python scripts/proof_suite.py
    python scripts/proof_suite.py --models llama3.2:3b
    python scripts/proof_suite.py --runs 5 --output proof.json
    python scripts/proof_suite.py --list-models
    autotune proof-suite
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict, replace as dc_replace
from pathlib import Path
from typing import Optional

# ── Resolve project root so the script works both standalone and installed ──
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
import psutil
from rich import box as _box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from autotune.api.kv_manager import build_ollama_options
from autotune.api.profiles import get_profile
from autotune.hardware.profiler import profile_hardware
from autotune.metrics.ollama_client import OllamaMetricsClient, NativeInferenceStats

try:
    from scipy import stats as _scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_OLLAMA_BASE          = "http://localhost:11434"
_KEEP_ALIVE_LOADED    = "30m"    # both conditions share same keep_alive for fairness
_RAW_NUM_CTX          = 4096     # Ollama's default num_ctx (v0.6+)
_COOLDOWN_RUN_SEC     = 3.0      # pause between runs of the same condition
_COOLDOWN_COND_SEC    = 10.0     # pause between raw → autotune switch
_RAM_SAMPLE_HZ        = 0.10     # seconds between Ollama RAM samples
_RELOAD_THRESHOLD_MS  = 400.0    # load_ms above this → model was actually evicted/reloaded
# Note: Apple M2 Metal shows ~100 ms load_ms baseline on every call (KV cache init overhead).
# True model reloads (cold load from disk) are 500 ms+ for 3B models, 2000ms+ for 8B.
_GROWTH_TURNS         = 4        # number of turns for memory-growth test

DEFAULT_MODELS = ["llama3.2:3b", "gemma4:e2b", "qwen3:8b"]
PROFILE_NAME   = "balanced"


# ─────────────────────────────────────────────────────────────────────────────
# Run configuration — controls depth vs. speed trade-off
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """Controls which prompts, how many runs, and what cooldowns to use."""
    mode: str                  # "quick" | "complete"
    prompts: list              # list[BenchPrompt]; populated after PROMPTS is defined
    n_runs: int                # runs per condition per prompt
    run_growth: bool           # whether to run the multi-turn memory growth test
    cooldown_run_sec: float    # pause between runs of the same condition
    cooldown_cond_sec: float   # pause between raw → autotune switch
    multi_model: bool          # True = all models, False = auto-select first available


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark prompt suite — 5 prompts covering TTFT, throughput, KV-heavy,
#   prefix-cache, and sustained generation scenarios
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchPrompt:
    id: str
    label: str
    domain: str
    messages: list[dict]
    expected_win: str


PROMPTS: list[BenchPrompt] = [
    BenchPrompt(
        id="short_factual",
        label="Short factual Q&A",
        domain="Factual",
        messages=[
            {"role": "user", "content":
             "What are the three laws of thermodynamics? Give one real-world "
             "example for each law in two sentences."},
        ],
        expected_win="TTFT (smaller num_ctx = faster KV initialisation)",
    ),
    BenchPrompt(
        id="code_generation",
        label="Code generation",
        domain="Code",
        messages=[
            {"role": "system", "content":
             "You are a senior Python engineer. Write clean, idiomatic code "
             "with type hints."},
            {"role": "user", "content":
             "Implement a thread-safe LRU cache class in Python without using "
             "functools.lru_cache. Support get(key), put(key, val), and a "
             "configurable max_size. Include a brief docstring."},
        ],
        expected_win="Throughput (KV quant frees memory bandwidth for generation)",
    ),
    BenchPrompt(
        id="long_context",
        label="Long-context analysis",
        domain="Analysis",
        messages=[
            {"role": "system", "content":
             "You are a meticulous security-focused code reviewer. "
             "Categorize findings by severity: Critical / High / Medium / Low. "
             "Cite exact line numbers and explain the precise risk."},
            {"role": "user", "content":
             "Review this Flask authentication endpoint for ALL issues:\n\n"
             "```python\n"
             "import sqlite3\n"
             "from flask import Flask, request\n"
             "import hashlib\n\n"
             "app = Flask(__name__)\n"
             "db = sqlite3.connect('users.db')\n\n"
             "@app.route('/login', methods=['POST'])\n"
             "def login():\n"
             "    u = request.form['username']\n"
             "    p = request.form['password']\n"
             "    q = f\"SELECT * FROM users WHERE username='{u}' "
             "AND password='{hashlib.md5(p.encode()).hexdigest()}'\"\n"
             "    row = db.cursor().execute(q).fetchone()\n"
             "    if row:\n"
             "        return {'token': row[0], 'admin': row[3]}\n"
             "    return {'error': 'invalid'}, 401\n"
             "```\n\n"
             "@app.route('/register', methods=['POST'])\n"
             "def register():\n"
             "    u = request.form['username']\n"
             "    p = request.form['password']\n"
             "    db.execute(f\"INSERT INTO users VALUES "
             "('{u}', '{hashlib.md5(p.encode()).hexdigest()}')\")\n"
             "    db.commit()\n"
             "    return {'ok': True}\n\n"
             "Cover: SQL injection, auth bypass, crypto weakness, "
             "session management, race conditions, error leakage. "
             "Then provide a fully corrected version."},
        ],
        expected_win="TTFT + RAM (large prompt → max KV savings from dynamic num_ctx)",
    ),
    BenchPrompt(
        id="multi_turn",
        label="Multi-turn conversation",
        domain="Conversation",
        messages=[
            {"role": "system", "content":
             "You are a senior distributed-systems architect with 15 years of "
             "production experience. Give detailed, opinionated advice with "
             "real examples. Always explain trade-offs explicitly."},
            {"role": "user", "content":
             "What authentication pattern do you recommend for microservices?"},
            {"role": "assistant", "content":
             "Short-lived JWTs (15 min TTL) from a central auth service, with "
             "opaque refresh tokens stored server-side. Services validate JWTs "
             "locally — no auth service roundtrip per request. The main trap "
             "is revocation; most teams underestimate how hard that is."},
            {"role": "user", "content":
             "How do you handle revocation without reintroducing global state?"},
            {"role": "assistant", "content":
             "You can't fully avoid state for revocation — it's a CAP "
             "trade-off. Pragmatic options: (1) short TTL + refresh blacklist "
             "only on refresh, (2) jti claim + Redis cache at JWT TTL, "
             "(3) Bloom filter for fast probabilistic pre-check."},
            {"role": "user", "content":
             "We're 6 engineers building B2B SaaS (~50k users). Give me a "
             "concrete recommendation: specific libraries, token lifetimes, "
             "service structure, and what NOT to do."},
        ],
        expected_win="TTFT + prefix cache (system prompt pinned in KV via num_keep)",
    ),
    BenchPrompt(
        id="sustained_output",
        label="Sustained long output",
        domain="Long generation",
        messages=[
            {"role": "user", "content":
             "Write a comprehensive technical comparison of SQL vs NoSQL "
             "databases covering: data models, ACID vs BASE guarantees, "
             "horizontal scaling, replication strategies, indexing, and a "
             "concrete decision framework. This will be used as an engineering "
             "team reference — be thorough and cite real databases."},
        ],
        expected_win="Throughput + RAM stability (KV quant reduces pressure on long outputs)",
    ),
]

# Growth-test prompts: 4 sequential turns that accumulate context naturally
_GROWTH_TURNS_MSGS: list[dict] = [
    {"role": "user",
     "content": "Explain how transformer attention mechanisms work, including "
                "multi-head attention, scaled dot-product, and positional encoding."},
    {"role": "user",
     "content": "Now explain how the KV cache works in inference and why it matters for speed."},
    {"role": "user",
     "content": "Compare those ideas to how RNNs handle sequence memory — "
                "what are the practical trade-offs in production?"},
    {"role": "user",
     "content": "Given everything above, what architecture would you choose for a "
                "real-time chat assistant on edge hardware with 8 GB RAM? Be specific."},
]


# Run configuration presets — defined after PROMPTS so we can reference them
_QUICK_PROMPT_IDS = {"short_factual", "long_context", "multi_turn"}

# Quick (~5-8 min): 3 KV-heavy prompts, 2 runs, no growth, short cooldowns
QUICK_CONFIG = RunConfig(
    mode="quick",
    prompts=[p for p in PROMPTS if p.id in _QUICK_PROMPT_IDS],
    n_runs=2,
    run_growth=False,
    cooldown_run_sec=1.0,
    cooldown_cond_sec=5.0,
    multi_model=False,
)

# Complete (~20-30 min): all 5 prompts, 3 runs, growth test, full cooldowns
COMPLETE_CONFIG = RunConfig(
    mode="complete",
    prompts=list(PROMPTS),
    n_runs=3,
    run_growth=True,
    cooldown_run_sec=_COOLDOWN_RUN_SEC,
    cooldown_cond_sec=_COOLDOWN_COND_SEC,
    multi_model=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# KV cache size estimator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OllamaModelInfo:
    """Architecture info extracted from Ollama for KV cache estimation."""
    model_id:    str
    n_layers:    int     = 0
    n_kv_heads:  int     = 0
    head_dim:    int     = 0
    # derived
    kv_bytes_per_token: int = 0   # 2 × n_layers × n_kv_heads × head_dim × dtype_bytes

    def is_valid(self) -> bool:
        return self.n_layers > 0 and self.n_kv_heads > 0 and self.head_dim > 0

    def estimate_kv_cache_mb(self, num_ctx: int, dtype_bytes: int = 2) -> float:
        """Estimate KV cache memory for a given context length."""
        if not self.is_valid():
            return 0.0
        # K + V for each layer, each KV head, each context token
        bytes_ = 2 * self.n_layers * self.n_kv_heads * self.head_dim * num_ctx * dtype_bytes
        return bytes_ / (1024 ** 2)


async def _fetch_model_info(model_id: str) -> OllamaModelInfo:
    """
    Query Ollama's /api/show endpoint to get model architecture for KV estimation.

    Looks for llama.block_count, llama.attention.kv_heads, llama.attention.head_count_kv,
    and llama.attention.head_dim in the model_info dict.  Falls back to zero on any
    key miss (estimate will be shown as 'N/A' in the report).

    Async so it can be gathered in parallel with the raw warmup call.
    """
    info = OllamaModelInfo(model_id=model_id)
    try:
        async with httpx.AsyncClient() as _hclient:
            resp = await _hclient.post(
                f"{_OLLAMA_BASE}/api/show",
                json={"name": model_id, "verbose": True},
                timeout=10.0,
            )
        data = resp.json()
        mi = data.get("model_info", {})

        # Ollama uses model-family prefixes (llama, gemma, qwen2, etc.)
        # We search all keys generically.
        def _find(suffix: str) -> int:
            for k, v in mi.items():
                if k.endswith(suffix) and isinstance(v, (int, float)):
                    return int(v)
            return 0

        info.n_layers   = _find(".block_count")
        info.n_kv_heads = _find(".attention.head_count_kv") or _find(".attention.kv_heads")
        info.head_dim   = _find(".attention.head_dim") or _find(".attention.key_length")

        # If head_dim still missing, derive from embedding_length / n_heads
        if not info.head_dim:
            emb   = _find(".embedding_length")
            heads = _find(".attention.head_count")
            if emb and heads:
                info.head_dim = emb // heads

        if info.is_valid():
            info.kv_bytes_per_token = 2 * info.n_layers * info.n_kv_heads * info.head_dim * 2
    except Exception:
        pass
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Ollama process-isolated RAM + swap sampler
# ─────────────────────────────────────────────────────────────────────────────

class OllamaRamSampler:
    """
    Samples the Ollama runner process RSS every _RAM_SAMPLE_HZ seconds.

    Also records system-wide swap usage and free memory floor so we can
    detect swap pressure induced by inference.
    """

    def __init__(self) -> None:
        self._pid: Optional[int]    = _find_ollama_runner_pid()
        self._rss_samples:  list[float] = []   # GB  Ollama process RSS
        self._free_samples: list[float] = []   # GB  system available RAM
        self._swap_start:   float       = 0.0  # GB  swap used before run
        self._swap_end:     float       = 0.0  # GB  swap used after run
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._swap_start = psutil.swap_memory().used / 1024 ** 3
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
            try:
                if self._pid:
                    proc = psutil.Process(self._pid)
                    rss = proc.memory_info().rss / 1024 ** 3
                    self._rss_samples.append(rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._pid = _find_ollama_runner_pid()
            vm = psutil.virtual_memory()
            self._free_samples.append(vm.available / 1024 ** 3)
            time.sleep(_RAM_SAMPLE_HZ)

    # ── Accessors ──────────────────────────────────────────────────────────

    def peak_ollama_gb(self) -> float:
        return max(self._rss_samples) if self._rss_samples else 0.0

    def baseline_ollama_gb(self) -> float:
        return self._rss_samples[0] if self._rss_samples else 0.0

    def delta_ollama_gb(self) -> float:
        if len(self._rss_samples) < 2:
            return 0.0
        return self.peak_ollama_gb() - self.baseline_ollama_gb()

    def free_floor_gb(self) -> float:
        return min(self._free_samples) if self._free_samples else 0.0

    def swap_delta_gb(self) -> float:
        """Increase in swap usage during this run.  Positive = swap pressure."""
        return max(0.0, self._swap_end - self._swap_start)

    def swap_occurred(self) -> bool:
        """True if swap usage increased meaningfully (> 32 MB)."""
        return self.swap_delta_gb() > 0.031


def _find_ollama_runner_pid() -> Optional[int]:
    """Return the PID of the Ollama model-runner process (highest RSS among ollama procs)."""
    candidates: list[tuple[int, int]] = []
    for proc in psutil.process_iter(["name", "pid", "memory_info", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if "ollama" not in name:
                continue
            rss = proc.info["memory_info"].rss
            candidates.append((rss, proc.info["pid"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# ─────────────────────────────────────────────────────────────────────────────
# Single-run result — carries every measured KPI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    condition: str           # "raw" | "autotune"
    prompt_id: str
    num_ctx: int             # context window used

    # Ollama-native timings (nanosecond timer from Ollama's Go runtime)
    prefill_ms: float        # prompt_eval_duration (KV fill phase)
    load_ms: float           # load_duration (model load + KV alloc)
    eval_tps: float          # true generation tok/s
    total_ms: float          # total_duration

    # Token counts (from Ollama)
    eval_count: int          # tokens generated
    prompt_eval_count: int   # tokens in the prompt (actual processed)

    # Memory KPIs
    ollama_peak_ram_gb: float   # Ollama runner process peak RSS
    ollama_ram_delta_gb: float  # RSS increase during this run
    free_floor_gb: float        # lowest system free RAM (swap pressure indicator)
    swap_delta_gb: float        # swap usage increase during run (GB)
    swap_occurred: bool         # True if swap pressure detected

    # KV cache estimation
    kv_cache_mb: float          # estimated KV cache size for this num_ctx
    reload_detected: bool       # True if load_ms > threshold (model was reloaded)

    error: Optional[str] = None

    @property
    def ttft_ms(self) -> float:
        """TTFT = load + prefill (load ≈ 0 on warm model)."""
        return self.load_ms + self.prefill_ms

    @property
    def ok(self) -> bool:
        return self.error is None and self.eval_count > 0

    @property
    def total_tokens(self) -> int:
        """Total context tokens processed in this run."""
        return self.prompt_eval_count + self.eval_count


# ─────────────────────────────────────────────────────────────────────────────
# Per-turn memory growth data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GrowthPoint:
    turn: int
    num_ctx: int
    prompt_tokens: int
    total_tokens: int       # prompt + output so far
    ollama_ram_gb: float
    prefill_ms: float
    kv_cache_mb: float


# ─────────────────────────────────────────────────────────────────────────────
# Run helpers — raw baseline vs autotune
# ─────────────────────────────────────────────────────────────────────────────

async def _run_raw(
    model: str, prompt: BenchPrompt, model_info: OllamaModelInfo, max_tokens: int = 512
) -> RunResult:
    """Raw Ollama: no autotune options, factory defaults."""
    client   = OllamaMetricsClient(base_url=_OLLAMA_BASE)
    raw_opts = {"num_ctx": _RAW_NUM_CTX}
    kv_mb    = model_info.estimate_kv_cache_mb(_RAW_NUM_CTX, dtype_bytes=2)  # f16

    sampler = OllamaRamSampler()
    sampler.start()
    try:
        stats: NativeInferenceStats = await client.run_with_stats(
            model=model, messages=prompt.messages,
            options=raw_opts, keep_alive=_KEEP_ALIVE_LOADED,
            max_tokens=max_tokens, temperature=0.8,
        )
    finally:
        sampler.stop()

    return RunResult(
        condition="raw",       prompt_id=prompt.id,
        num_ctx=_RAW_NUM_CTX,  prefill_ms=stats.prefill_ms,
        load_ms=stats.load_ms, eval_tps=stats.eval_tps,
        total_ms=stats.total_ms,
        eval_count=stats.eval_count,
        prompt_eval_count=stats.prompt_eval_count,
        ollama_peak_ram_gb=sampler.peak_ollama_gb(),
        ollama_ram_delta_gb=sampler.delta_ollama_gb(),
        free_floor_gb=sampler.free_floor_gb(),
        swap_delta_gb=sampler.swap_delta_gb(),
        swap_occurred=sampler.swap_occurred(),
        kv_cache_mb=kv_mb,
        reload_detected=(stats.load_ms > _RELOAD_THRESHOLD_MS),
        error=stats.error,
    )


async def _run_autotune(
    model: str, prompt: BenchPrompt, model_info: OllamaModelInfo,
    profile_name: str = PROFILE_NAME
) -> RunResult:
    """Autotune: dynamic num_ctx, KV precision, flash attention, prefix cache."""
    profile   = get_profile(profile_name)
    client    = OllamaMetricsClient(base_url=_OLLAMA_BASE)
    opts, _   = build_ollama_options(prompt.messages, profile)
    max_tokens = min(profile.max_new_tokens, 512)
    tuned_ctx  = opts["num_ctx"]
    # Autotune often enables kv_cache_type=q8_0 → 1 byte per element
    kv_dtype_bytes = 1 if opts.get("kv_cache_type") else 2
    kv_mb = model_info.estimate_kv_cache_mb(tuned_ctx, dtype_bytes=kv_dtype_bytes)

    sampler = OllamaRamSampler()
    sampler.start()
    try:
        stats: NativeInferenceStats = await client.run_with_stats(
            model=model, messages=prompt.messages,
            options=opts, keep_alive=_KEEP_ALIVE_LOADED,
            max_tokens=max_tokens,
            temperature=profile.temperature, top_p=profile.top_p,
        )
    finally:
        sampler.stop()

    return RunResult(
        condition="autotune", prompt_id=prompt.id,
        num_ctx=tuned_ctx,    prefill_ms=stats.prefill_ms,
        load_ms=stats.load_ms, eval_tps=stats.eval_tps,
        total_ms=stats.total_ms,
        eval_count=stats.eval_count,
        prompt_eval_count=stats.prompt_eval_count,
        ollama_peak_ram_gb=sampler.peak_ollama_gb(),
        ollama_ram_delta_gb=sampler.delta_ollama_gb(),
        free_floor_gb=sampler.free_floor_gb(),
        swap_delta_gb=sampler.swap_delta_gb(),
        swap_occurred=sampler.swap_occurred(),
        kv_cache_mb=kv_mb,
        reload_detected=(stats.load_ms > _RELOAD_THRESHOLD_MS),
        error=stats.error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Memory growth over turns
# ─────────────────────────────────────────────────────────────────────────────

async def _run_growth_test(
    model: str,
    profile_name: str,
    model_info: OllamaModelInfo,
    console: Console,
) -> tuple[list[GrowthPoint], list[GrowthPoint]]:
    """
    Run _GROWTH_TURNS sequential turns and record RAM after each.

    Returns (raw_growth, tuned_growth) — each is a list of GrowthPoints,
    one per turn.  Accumulated context grows each turn: each assistant reply
    is added to the message list before the next user message.
    """
    profile = get_profile(profile_name)

    async def _measure_turn(
        messages: list[dict], opts: dict, keep_alive: str, max_tokens: int,
        temperature: float, turn: int, model_info: OllamaModelInfo,
        dtype_bytes: int,
    ) -> tuple[NativeInferenceStats, float, float]:
        """Returns (stats, peak_ollama_ram_gb, kv_mb)."""
        client  = OllamaMetricsClient(base_url=_OLLAMA_BASE)
        sampler = OllamaRamSampler()
        sampler.start()
        try:
            stats = await client.run_with_stats(
                model=model, messages=messages, options=opts,
                keep_alive=keep_alive, max_tokens=max_tokens,
                temperature=temperature,
            )
        finally:
            sampler.stop()
        kv_mb = model_info.estimate_kv_cache_mb(opts["num_ctx"], dtype_bytes)
        return stats, sampler.peak_ollama_gb(), kv_mb

    raw_growth:   list[GrowthPoint] = []
    tuned_growth: list[GrowthPoint] = []

    # ── Warmup raw growth condition (ensure model is at num_ctx=4096) ─────
    try:
        client_wm = OllamaMetricsClient(base_url=_OLLAMA_BASE)
        await client_wm.run_with_stats(
            model=model, messages=[{"role": "user", "content": "Say 'ready'."}],
            options={"num_ctx": _RAW_NUM_CTX}, keep_alive=_KEEP_ALIVE_LOADED,
            max_tokens=4,
        )
        await asyncio.sleep(1.0)
    except Exception:
        pass

    # ── Raw growth ─────────────────────────────────────────────────────────
    raw_msgs: list[dict] = []
    raw_cumulative_tokens = 0
    for i, user_msg in enumerate(_GROWTH_TURNS_MSGS):
        raw_msgs = raw_msgs + [user_msg]
        opts_raw = {"num_ctx": _RAW_NUM_CTX}
        stats, peak_gb, kv_mb = await _measure_turn(
            raw_msgs, opts_raw, _KEEP_ALIVE_LOADED,
            max_tokens=256, temperature=0.8, turn=i + 1,
            model_info=model_info, dtype_bytes=2,
        )
        raw_cumulative_tokens += stats.prompt_eval_count + stats.eval_count
        raw_growth.append(GrowthPoint(
            turn=i + 1, num_ctx=_RAW_NUM_CTX,
            prompt_tokens=stats.prompt_eval_count,
            total_tokens=raw_cumulative_tokens,
            ollama_ram_gb=peak_gb,
            prefill_ms=stats.prefill_ms,
            kv_cache_mb=kv_mb,
        ))
        if stats.response_text:
            raw_msgs = raw_msgs + [{"role": "assistant", "content": stats.response_text}]
        await asyncio.sleep(2.0)

    await asyncio.sleep(_COOLDOWN_COND_SEC)

    # ── Autotune growth ────────────────────────────────────────────────────
    tuned_msgs: list[dict] = []
    tuned_cumulative_tokens = 0
    for i, user_msg in enumerate(_GROWTH_TURNS_MSGS):
        tuned_msgs = tuned_msgs + [user_msg]
        opts_tuned, _ = build_ollama_options(tuned_msgs, profile)
        kv_dtype = 1 if opts_tuned.get("kv_cache_type") else 2
        max_tokens = min(profile.max_new_tokens, 256)
        stats, peak_gb, kv_mb = await _measure_turn(
            tuned_msgs, opts_tuned, _KEEP_ALIVE_LOADED,
            max_tokens=max_tokens, temperature=profile.temperature, turn=i + 1,
            model_info=model_info, dtype_bytes=kv_dtype,
        )
        tuned_cumulative_tokens += stats.prompt_eval_count + stats.eval_count
        tuned_growth.append(GrowthPoint(
            turn=i + 1, num_ctx=opts_tuned["num_ctx"],
            prompt_tokens=stats.prompt_eval_count,
            total_tokens=tuned_cumulative_tokens,
            ollama_ram_gb=peak_gb,
            prefill_ms=stats.prefill_ms,
            kv_cache_mb=kv_mb,
        ))
        if stats.response_text:
            tuned_msgs = tuned_msgs + [{"role": "assistant", "content": stats.response_text}]
        await asyncio.sleep(2.0)

    return raw_growth, tuned_growth


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StatResult:
    metric: str
    n: int
    raw_mean: float
    tuned_mean: float
    pct_change: float          # positive = autotune produced higher value
    cohens_d: float
    p_value: float
    ci95_lo: float             # 95% CI on (tuned - raw)
    ci95_hi: float
    wins: int                  # runs where autotune was better
    direction_consistent: bool # wins == n
    test_name: str
    higher_is_better: bool

    @property
    def sig_stars(self) -> str:
        if self.p_value < 0.001: return "***"
        if self.p_value < 0.01:  return "**"
        if self.p_value < 0.05:  return "*"
        if self.p_value < 0.10:  return "†"
        return ""

    @property
    def effect_label(self) -> str:
        d = abs(self.cohens_d)
        if d >= 1.20: return "very large"
        if d >= 0.80: return "large"
        if d >= 0.50: return "medium"
        if d >= 0.20: return "small"
        return "negligible"

    @property
    def improved(self) -> bool:
        if self.higher_is_better:
            return self.pct_change > 0
        return self.pct_change < 0


def _stat(
    metric: str,
    raw_vals: list[float],
    tuned_vals: list[float],
    higher_is_better: bool,
) -> StatResult:
    """
    Wilcoxon signed-rank (n<10) or paired t-test (n≥10) on paired observations.
    Adds Cohen's d and 95% CI on mean difference.
    """
    n = min(len(raw_vals), len(tuned_vals))
    raw_a   = raw_vals[:n]
    tuned_a = tuned_vals[:n]

    raw_mean   = statistics.mean(raw_a)   if raw_a   else float("nan")
    tuned_mean = statistics.mean(tuned_a) if tuned_a else float("nan")

    if raw_mean and raw_mean != 0:
        pct_change = (tuned_mean - raw_mean) / abs(raw_mean) * 100
    else:
        pct_change = 0.0

    diffs = [t - r for r, t in zip(raw_a, tuned_a)]
    if len(diffs) >= 2:
        d_mean = statistics.mean(diffs)
        d_std  = statistics.stdev(diffs)
        if d_std > 0:
            raw_d    = d_mean / d_std
            cohens_d = max(-10.0, min(10.0, raw_d))
        else:
            cohens_d = 0.0
    else:
        cohens_d = 0.0
        d_mean   = 0.0

    if len(diffs) >= 2:
        d_std  = statistics.stdev(diffs)
        se     = d_std / math.sqrt(len(diffs))
        t_crit = 2.0 if len(diffs) < 30 else 1.96
        ci95_lo = d_mean - t_crit * se
        ci95_hi = d_mean + t_crit * se
    else:
        ci95_lo = ci95_hi = d_mean if diffs else 0.0

    p_value   = 1.0
    test_name = "insufficient_data"
    if n >= 3 and _SCIPY:
        if n >= 10:
            _, p_value = _scipy_stats.ttest_rel(tuned_a, raw_a)
            test_name  = "paired_t"
        else:
            try:
                _, p_value = _scipy_stats.wilcoxon(tuned_a, raw_a)
                test_name  = "wilcoxon"
            except ValueError:
                p_value   = 1.0
                test_name = "wilcoxon(tied)"

    if higher_is_better:
        wins = sum(1 for r, t in zip(raw_a, tuned_a) if t > r)
    else:
        wins = sum(1 for r, t in zip(raw_a, tuned_a) if t < r)

    return StatResult(
        metric=metric, n=n,
        raw_mean=raw_mean, tuned_mean=tuned_mean,
        pct_change=pct_change, cohens_d=cohens_d,
        p_value=p_value,
        ci95_lo=ci95_lo, ci95_hi=ci95_hi,
        wins=wins, direction_consistent=(wins == n),
        test_name=test_name, higher_is_better=higher_is_better,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PromptStats:
    prompt: BenchPrompt
    raw_runs:    list[RunResult]
    tuned_runs:  list[RunResult]

    # Per-metric stats (each is a StatResult comparing raw vs autotune)
    ttft:       StatResult   # TTFT (load_ms + prefill_ms)
    prefill_ms: StatResult   # KV fill phase only
    eval_tps:   StatResult   # generation throughput
    total_ms:   StatResult   # end-to-end wall time
    ollama_ram: StatResult   # peak Ollama process RSS
    kv_cache:   StatResult   # estimated KV cache (MB)

    num_ctx_raw:   float
    num_ctx_tuned: float

    # Simple aggregates (not stat-tested — derived values)
    tokens_saved: int        # (raw_num_ctx - tuned_num_ctx) × n_runs
    swap_occurred_raw:   int  # runs where swap pressure occurred (raw)
    swap_occurred_tuned: int  # runs where swap pressure occurred (autotune)
    reload_count_raw:   int   # runs where model reload was detected (raw)
    reload_count_tuned: int   # runs where model reload was detected (autotune)


@dataclass
class ModelReport:
    model_id: str
    profile: str
    n_runs: int
    hw_str: str
    timestamp: float
    model_info: OllamaModelInfo
    prompt_stats: list[PromptStats]
    raw_growth:   list[GrowthPoint]
    tuned_growth: list[GrowthPoint]
    skipped: bool = False
    skip_reason: str = ""

    def overall_wins(self) -> int:
        wins = 0
        for ps in self.prompt_stats:
            for sr in [ps.ttft, ps.eval_tps, ps.ollama_ram, ps.kv_cache, ps.prefill_ms]:
                if sr.improved:
                    wins += 1
        return wins

    def total_metrics(self) -> int:
        return len(self.prompt_stats) * 5   # ttft, tps, ram, kv, prefill per prompt

    def total_tokens_saved(self) -> int:
        return sum(ps.tokens_saved for ps in self.prompt_stats)

    def total_swaps_raw(self) -> int:
        return sum(ps.swap_occurred_raw for ps in self.prompt_stats)

    def total_swaps_tuned(self) -> int:
        return sum(ps.swap_occurred_tuned for ps in self.prompt_stats)

    def total_reloads_raw(self) -> int:
        return sum(ps.reload_count_raw for ps in self.prompt_stats)

    def total_reloads_tuned(self) -> int:
        return sum(ps.reload_count_tuned for ps in self.prompt_stats)


# ─────────────────────────────────────────────────────────────────────────────
# Warmup helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _warmup_raw(model: str, console: Console) -> None:
    """Load model at num_ctx=4096 so raw runs don't pay a reload penalty."""
    msg = [{"role": "user", "content": "Say 'ready'."}]
    try:
        client = OllamaMetricsClient(base_url=_OLLAMA_BASE)
        await client.run_with_stats(
            model=model, messages=msg,
            options={"num_ctx": _RAW_NUM_CTX}, keep_alive=_KEEP_ALIVE_LOADED,
            max_tokens=4,
        )
    except Exception as exc:
        console.print(f"[yellow]Raw warmup warning: {exc}[/yellow]")


async def _warmup_autotune(
    model: str, prompt: BenchPrompt, profile_name: str, console: Console
) -> None:
    """
    Load model at this prompt's exact autotune num_ctx.

    Ollama resizes its KV cache whenever num_ctx changes.  One untimed request
    ensures all n_runs start with the model at the correct context length.
    """
    profile = get_profile(profile_name)
    opts, _ = build_ollama_options(prompt.messages, profile)
    try:
        client = OllamaMetricsClient(base_url=_OLLAMA_BASE)
        await client.run_with_stats(
            model=model, messages=prompt.messages,
            options=opts, keep_alive=_KEEP_ALIVE_LOADED, max_tokens=4,
        )
        await asyncio.sleep(0.5)
    except Exception as exc:
        console.print(f"[yellow]Autotune warmup warning ({prompt.id}): {exc}[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_model_benchmark(
    model_id: str,
    run_cfg: RunConfig,
    profile_name: str,
    console: Console,
) -> ModelReport:
    """
    Full benchmark for one model.

    Order:
      1. Parallel: fetch model architecture info + raw warmup (overlap I/O)
      2. All raw runs (run_cfg.prompts × n_runs, with cooldown between runs)
      3. Cooldown between conditions
      4. All autotune runs (per-prompt warmup + n_runs)
      5. Memory growth test (complete mode only)
      6. Compute statistics
    """
    hw     = profile_hardware()
    hw_str = (
        f"{hw.cpu.brand.upper()} {hw.cpu.architecture}  "
        f"{hw.memory.total_gb:.0f} GB RAM  "
        f"{hw.gpu.name if hw.gpu else 'CPU-only'}"
    )

    n_prompts    = len(run_cfg.prompts)
    growth_steps = (_GROWTH_TURNS * 2 + 1) if run_cfg.run_growth else 0
    total_steps  = (1 + n_prompts * run_cfg.n_runs + 1
                    + n_prompts * (1 + run_cfg.n_runs)
                    + growth_steps)

    raw_map:   dict[str, list[RunResult]] = {p.id: [] for p in run_cfg.prompts}
    tuned_map: dict[str, list[RunResult]] = {p.id: [] for p in run_cfg.prompts}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description:<52}"),
        BarColumn(bar_width=22),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console, transient=False,
    ) as progress:
        task = progress.add_task(f"  {model_id}", total=total_steps)

        # ── Parallel: model architecture info + raw warmup ────────────────
        # Overlapping these two I/O-bound calls saves ~2–4 s per model.
        progress.update(task,
                        description=f"  [dim]Warmup + fetching arch for {model_id}…[/dim]")
        model_info, _ = await asyncio.gather(
            _fetch_model_info(model_id),
            _warmup_raw(model_id, console),
        )
        if model_info.is_valid():
            console.print(
                f"[dim]  {model_id}: {model_info.n_layers} layers, "
                f"{model_info.n_kv_heads} KV heads, head_dim={model_info.head_dim} → "
                f"KV raw={model_info.estimate_kv_cache_mb(_RAW_NUM_CTX):.0f} MB[/dim]"
            )
        else:
            console.print(
                f"[dim yellow]  Could not fetch architecture info — "
                f"KV estimates will show N/A[/dim yellow]"
            )
        progress.advance(task)

        # ── Raw runs ───────────────────────────────────────────────────────
        for prompt in run_cfg.prompts:
            for run_i in range(run_cfg.n_runs):
                n_label = f" [{run_i+1}/{run_cfg.n_runs}]" if run_cfg.n_runs > 1 else ""
                progress.update(task,
                                description=f"  [dim]Raw   · {prompt.label}{n_label}[/dim]")
                result = await _run_raw(model_id, prompt, model_info)
                raw_map[prompt.id].append(result)
                progress.advance(task)
                if run_i < run_cfg.n_runs - 1:
                    await asyncio.sleep(run_cfg.cooldown_run_sec)
            await asyncio.sleep(run_cfg.cooldown_run_sec)

        # ── Condition switch cooldown ─────────────────────────────────────
        progress.update(task, description="  [dim]Cooling down between conditions…[/dim]")
        await asyncio.sleep(run_cfg.cooldown_cond_sec)

        # ── Autotune runs ──────────────────────────────────────────────────
        for prompt in run_cfg.prompts:
            progress.update(task,
                            description=f"  [cyan]Warmup (autotune): {prompt.label}…[/cyan]")
            await _warmup_autotune(model_id, prompt, profile_name, console)
            progress.advance(task)

            for run_i in range(run_cfg.n_runs):
                n_label = f" [{run_i+1}/{run_cfg.n_runs}]" if run_cfg.n_runs > 1 else ""
                progress.update(task,
                                description=f"  [cyan]Autotune · {prompt.label}{n_label}[/cyan]")
                result = await _run_autotune(model_id, prompt, model_info, profile_name)
                tuned_map[prompt.id].append(result)
                progress.advance(task)
                if run_i < run_cfg.n_runs - 1:
                    await asyncio.sleep(run_cfg.cooldown_run_sec)
            await asyncio.sleep(run_cfg.cooldown_run_sec)

        # ── Memory growth test ─────────────────────────────────────────────
        raw_growth:   list[GrowthPoint] = []
        tuned_growth: list[GrowthPoint] = []
        if run_cfg.run_growth:
            progress.update(task,
                            description="  [yellow]Memory growth test (multi-turn)…[/yellow]")
            try:
                raw_growth, tuned_growth = await _run_growth_test(
                    model_id, profile_name, model_info, console
                )
            except Exception as exc:
                console.print(f"[yellow]Growth test warning: {exc}[/yellow]")
            for _ in range(growth_steps):
                progress.advance(task)

    # ── Compute statistics per prompt ──────────────────────────────────────
    prompt_stats_list: list[PromptStats] = []
    for prompt in run_cfg.prompts:
        raw_ok   = [r for r in raw_map[prompt.id]   if r.ok]
        tuned_ok = [r for r in tuned_map[prompt.id] if r.ok]

        def _vals(runs: list[RunResult], attr: str) -> list[float]:
            return [getattr(r, attr) for r in runs]

        n_raw    = statistics.mean(_vals(raw_ok, "num_ctx"))   if raw_ok   else float(_RAW_NUM_CTX)
        n_tuned  = statistics.mean(_vals(tuned_ok, "num_ctx")) if tuned_ok else 0.0
        tok_saved = int(max(0, n_raw - n_tuned) * len(raw_ok))

        ps = PromptStats(
            prompt=prompt,
            raw_runs=raw_map[prompt.id],
            tuned_runs=tuned_map[prompt.id],
            ttft=_stat(
                "TTFT (ms)",
                [r.ttft_ms for r in raw_ok], [r.ttft_ms for r in tuned_ok],
                higher_is_better=False,
            ),
            prefill_ms=_stat(
                "Prefill (ms)",
                _vals(raw_ok, "prefill_ms"), _vals(tuned_ok, "prefill_ms"),
                higher_is_better=False,
            ),
            eval_tps=_stat(
                "Throughput t/s",
                _vals(raw_ok, "eval_tps"), _vals(tuned_ok, "eval_tps"),
                higher_is_better=True,
            ),
            total_ms=_stat(
                "Total time (ms)",
                _vals(raw_ok, "total_ms"), _vals(tuned_ok, "total_ms"),
                higher_is_better=False,
            ),
            ollama_ram=_stat(
                "Peak RAM (GB)",
                _vals(raw_ok, "ollama_peak_ram_gb"), _vals(tuned_ok, "ollama_peak_ram_gb"),
                higher_is_better=False,
            ),
            kv_cache=_stat(
                "KV cache (MB)",
                _vals(raw_ok, "kv_cache_mb"), _vals(tuned_ok, "kv_cache_mb"),
                higher_is_better=False,
            ),
            num_ctx_raw=n_raw,
            num_ctx_tuned=n_tuned,
            tokens_saved=tok_saved,
            swap_occurred_raw=sum(1 for r in raw_ok if r.swap_occurred),
            swap_occurred_tuned=sum(1 for r in tuned_ok if r.swap_occurred),
            reload_count_raw=sum(1 for r in raw_ok if r.reload_detected),
            reload_count_tuned=sum(1 for r in tuned_ok if r.reload_detected),
        )
        prompt_stats_list.append(ps)

    return ModelReport(
        model_id=model_id, profile=profile_name,
        n_runs=run_cfg.n_runs, hw_str=hw_str, timestamp=time.time(),
        model_info=model_info,
        prompt_stats=prompt_stats_list,
        raw_growth=raw_growth,
        tuned_growth=tuned_growth,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

_CON_W = 100

def _pct_cell(pct: float, higher_is_better: bool) -> Text:
    improved = (pct > 0) if higher_is_better else (pct < 0)
    color = "green" if improved else ("red" if abs(pct) > 1 else "dim")
    sign  = "+" if pct >= 0 else ""
    return Text(f"{sign}{pct:.1f}%", style=color)


def _sig_cell(sr: StatResult) -> Text:
    stars = sr.sig_stars
    if stars:
        return Text(f"{sr.p_value:.3f}{stars}",
                    style="bold green" if sr.improved else "bold red")
    return Text(f"{sr.p_value:.3f}", style="dim")


def _effect_cell(sr: StatResult) -> Text:
    label  = sr.effect_label
    styles = {
        "very large": "bold cyan", "large": "cyan",
        "medium": "yellow", "small": "dim", "negligible": "dim red",
    }
    return Text(f"d={sr.cohens_d:+.2f} ({label})", style=styles.get(label, ""))


def _bar(pct: float, width: int = 10) -> str:
    """Simple ASCII bar: 0–100% → filled blocks."""
    filled = int(abs(pct) / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model full report
# ─────────────────────────────────────────────────────────────────────────────

def print_model_report(report: ModelReport, console: Console) -> None:
    if report.skipped:
        console.print(Panel(
            f"[yellow]SKIPPED — {report.skip_reason}[/yellow]",
            title=f"[bold]{report.model_id}[/bold]", border_style="yellow",
        ))
        return

    console.print()
    console.rule(
        f"[bold cyan]{report.model_id}[/bold cyan]  ·  "
        f"autotune/{report.profile}  ·  n={report.n_runs}/condition",
        style="cyan",
    )
    console.print(f"[dim]Hardware: {report.hw_str}[/dim]")
    mi = report.model_info
    if mi.is_valid():
        console.print(
            f"[dim]Architecture: {mi.n_layers} layers · "
            f"{mi.n_kv_heads} KV heads · head_dim={mi.head_dim} · "
            f"KV@4096={mi.estimate_kv_cache_mb(4096):.0f} MB (f16)[/dim]"
        )
    console.print()

    # ── Per-prompt breakdown ───────────────────────────────────────────────
    for ps in report.prompt_stats:
        raw_ok   = [r for r in ps.raw_runs   if r.ok]
        tuned_ok = [r for r in ps.tuned_runs if r.ok]
        if not raw_ok or not tuned_ok:
            console.print(f"[red]  {ps.prompt.label}: all runs errored — skipping[/red]")
            continue

        t = Table(
            title=f"[bold]{ps.prompt.label}[/bold]  [dim]({ps.prompt.domain})[/dim]",
            box=_box.SIMPLE_HEAVY, show_header=True,
            header_style="bold", width=_CON_W,
        )
        t.add_column("KPI",              width=16)
        t.add_column("Raw Ollama",       width=13, justify="right")
        t.add_column("Autotune",         width=13, justify="right")
        t.add_column("Change",           width=10, justify="right")
        t.add_column("p-value",          width=11, justify="right")
        t.add_column("Effect size",      width=22, justify="left")
        t.add_column("All runs?",        width=10, justify="center")

        for sr in [ps.ttft, ps.prefill_ms, ps.eval_tps, ps.total_ms, ps.ollama_ram, ps.kv_cache]:
            unit = ""
            if "ms" in sr.metric:     unit = " ms"
            elif "t/s" in sr.metric:  unit = " t/s"
            elif "GB" in sr.metric:   unit = " GB"
            elif "MB" in sr.metric:   unit = " MB"

            if sr.raw_mean == 0 and sr.tuned_mean == 0:
                raw_fmt   = Text("N/A", style="dim")
                tuned_fmt = Text("N/A", style="dim")
                t.add_row(sr.metric, raw_fmt, tuned_fmt,
                          Text("—", style="dim"), Text("—", style="dim"),
                          Text("architecture info unavailable", style="dim"),
                          Text("—", style="dim"))
                continue

            raw_fmt   = f"{sr.raw_mean:.1f}{unit}"
            tuned_fmt = f"{sr.tuned_mean:.1f}{unit}"
            consist   = (
                Text(f"✓ {sr.wins}/{sr.n}", style="green")
                if sr.direction_consistent
                else Text(f"  {sr.wins}/{sr.n}", style="yellow")
            )
            t.add_row(
                sr.metric, raw_fmt, tuned_fmt,
                _pct_cell(sr.pct_change, sr.higher_is_better),
                _sig_cell(sr), _effect_cell(sr), consist,
            )

        # Context window row
        ctx_ratio = ps.num_ctx_raw / ps.num_ctx_tuned if ps.num_ctx_tuned else 1.0
        t.add_row(
            "num_ctx (tokens)",
            f"{ps.num_ctx_raw:.0f}",
            f"{ps.num_ctx_tuned:.0f}",
            Text(f"−{(1-1/ctx_ratio)*100:.0f}%", style="cyan"),
            Text("—", style="dim"),
            Text(f"{ctx_ratio:.1f}× smaller context window", style="dim cyan"),
            Text("—", style="dim"),
        )

        # Total context tokens row
        raw_ctx_tokens = statistics.mean(
            [r.prompt_eval_count + r.eval_count for r in raw_ok]
        ) if raw_ok else 0
        tuned_ctx_tokens = statistics.mean(
            [r.prompt_eval_count + r.eval_count for r in tuned_ok]
        ) if tuned_ok else 0
        t.add_row(
            "Total tokens used",
            f"{raw_ctx_tokens:.0f}",
            f"{tuned_ctx_tokens:.0f}",
            Text(f"{((tuned_ctx_tokens-raw_ctx_tokens)/raw_ctx_tokens*100) if raw_ctx_tokens else 0:+.0f}%",
                 style="dim"),
            Text("—", style="dim"),
            Text("prompt + generated tokens per call", style="dim"),
            Text("—", style="dim"),
        )
        console.print(t)

        # Swap + reload sidebar
        swap_r = ps.swap_occurred_raw
        swap_t = ps.swap_occurred_tuned
        rel_r  = ps.reload_count_raw
        rel_t  = ps.reload_count_tuned
        n_r    = len(raw_ok)

        swap_color = "red" if swap_r > swap_t else ("green" if swap_t < swap_r else "dim")
        console.print(
            f"  [dim]Swap events:[/dim] "
            f"Raw [{'red' if swap_r else 'green'}]{swap_r}/{n_r}[/{'red' if swap_r else 'green'}] "
            f"→ Autotune [{'red' if swap_t else 'green'}]{swap_t}/{n_r}[/{'red' if swap_t else 'green'}]   "
            f"[dim]Model reloads:[/dim] "
            f"Raw [{'yellow' if rel_r else 'dim'}]{rel_r}/{n_r}[/{'yellow' if rel_r else 'dim'}] "
            f"→ Autotune [{'yellow' if rel_t else 'dim'}]{rel_t}/{n_r}[/{'yellow' if rel_t else 'dim'}]"
        )
        console.print(
            f"  [dim]Tokens saved by context shrink: "
            f"[cyan]{ps.tokens_saved:,}[/cyan] tokens across {n_r} runs[/dim]"
        )
        console.print()

    # ── Memory growth over turns ───────────────────────────────────────────
    if report.raw_growth and report.tuned_growth:
        console.print()
        console.rule("[bold yellow]Memory Growth Over Turns[/bold yellow]", style="yellow")
        console.print(
            "[dim]Multi-turn conversation: each turn adds to accumulated context. "
            "RAM measured after each turn.[/dim]\n"
        )
        g = Table(box=_box.SIMPLE_HEAVY, header_style="bold", width=_CON_W)
        g.add_column("Turn", width=6, justify="center")
        g.add_column("Raw  num_ctx",    width=13, justify="right")
        g.add_column("Tuned num_ctx",   width=13, justify="right")
        g.add_column("Raw RAM (GB)",    width=13, justify="right")
        g.add_column("Tuned RAM (GB)",  width=14, justify="right")
        g.add_column("RAM saved",       width=12, justify="right")
        g.add_column("Raw KV (MB)",     width=12, justify="right")
        g.add_column("Tuned KV (MB)",   width=13, justify="right")

        for raw_pt, tun_pt in zip(report.raw_growth, report.tuned_growth):
            ram_saved   = raw_pt.ollama_ram_gb - tun_pt.ollama_ram_gb
            ram_saved_c = "green" if ram_saved > 0.01 else ("dim" if abs(ram_saved) < 0.01 else "red")
            g.add_row(
                str(raw_pt.turn),
                f"{raw_pt.num_ctx}",
                f"{tun_pt.num_ctx}",
                f"{raw_pt.ollama_ram_gb:.2f}",
                f"{tun_pt.ollama_ram_gb:.2f}",
                Text(f"{ram_saved:+.3f} GB", style=ram_saved_c),
                f"{raw_pt.kv_cache_mb:.0f}" if raw_pt.kv_cache_mb else "N/A",
                f"{tun_pt.kv_cache_mb:.0f}" if tun_pt.kv_cache_mb else "N/A",
            )
        console.print(g)

        # Growth rate
        if len(report.raw_growth) >= 2 and len(report.tuned_growth) >= 2:
            raw_growth_rate  = (report.raw_growth[-1].ollama_ram_gb
                                - report.raw_growth[0].ollama_ram_gb) / (_GROWTH_TURNS - 1)
            tuned_growth_rate = (report.tuned_growth[-1].ollama_ram_gb
                                 - report.tuned_growth[0].ollama_ram_gb) / (_GROWTH_TURNS - 1)
            color = "green" if tuned_growth_rate < raw_growth_rate else "yellow"
            console.print(
                f"  [dim]RAM growth rate per turn:[/dim]  "
                f"Raw {raw_growth_rate:+.3f} GB/turn  →  "
                f"Autotune [{color}]{tuned_growth_rate:+.3f} GB/turn[/{color}]"
            )
        console.print()

    # ── Model summary ──────────────────────────────────────────────────────
    wins     = report.overall_wins()
    total    = report.total_metrics()
    win_pct  = wins / total * 100 if total else 0

    all_ttft   = [ps.ttft.pct_change       for ps in report.prompt_stats if ps.ttft.n > 0]
    all_pre    = [ps.prefill_ms.pct_change  for ps in report.prompt_stats if ps.prefill_ms.n > 0]
    all_tps    = [ps.eval_tps.pct_change    for ps in report.prompt_stats if ps.eval_tps.n > 0]
    all_ram    = [ps.ollama_ram.pct_change  for ps in report.prompt_stats if ps.ollama_ram.n > 0]
    all_kv     = [ps.kv_cache.pct_change    for ps in report.prompt_stats if ps.kv_cache.n > 0 and ps.kv_cache.raw_mean > 0]
    all_ctx    = [ps.num_ctx_raw / ps.num_ctx_tuned for ps in report.prompt_stats if ps.num_ctx_tuned > 0]

    mean_ttft  = statistics.mean(all_ttft) if all_ttft else 0.0
    mean_pre   = statistics.mean(all_pre)  if all_pre  else 0.0
    mean_tps   = statistics.mean(all_tps)  if all_tps  else 0.0
    mean_ram   = statistics.mean(all_ram)  if all_ram  else 0.0
    mean_kv    = statistics.mean(all_kv)   if all_kv   else 0.0
    mean_ctx   = statistics.mean(all_ctx)  if all_ctx  else 1.0

    sig_count = sum(
        1 for ps in report.prompt_stats
        for sr in [ps.ttft, ps.eval_tps, ps.ollama_ram, ps.prefill_ms]
        if sr.p_value < 0.10 and sr.improved
    )

    style = "green" if win_pct >= 60 else ("yellow" if win_pct >= 40 else "red")
    icon  = "✓" if win_pct >= 60 else ("≈" if win_pct >= 40 else "✗")

    lines = [
        f"[bold {style}]{icon}  autotune/{report.profile} won {wins}/{total} metrics "
        f"({win_pct:.0f}%) across {len(report.prompt_stats)} prompts[/bold {style}]",
        "",
        "  [bold]KPI summary (autotune vs. raw Ollama):[/bold]",
        f"    ⏱  Time to first word (TTFT):   {abs(mean_ttft):.0f}% {'faster' if mean_ttft<0 else 'slower'}  "
        f"  [dim]← smaller KV buffer fills faster[/dim]",
        f"    ⏱  KV prefill time:              {abs(mean_pre):.0f}% {'faster' if mean_pre<0 else 'slower'}  "
        f"  [dim]← dynamic num_ctx reduces prefill work[/dim]",
        f"    ⚡ Generation speed:             {mean_tps:+.0f}% {'faster' if mean_tps>0 else 'slower'}  "
        f"  [dim]← KV quant frees memory bandwidth[/dim]",
        f"    🧠 Peak RAM:                     {abs(mean_ram):.0f}% {'less' if mean_ram<0 else 'more'}  "
        f"  [dim]← smaller KV cache, less process RSS[/dim]",
    ]
    if mean_kv != 0:
        lines.append(
            f"    💾 KV cache size:               {abs(mean_kv):.0f}% smaller  "
            f"  [dim]← {mean_ctx:.1f}× fewer tokens allocated[/dim]"
        )
    lines += [
        f"    💡 Context window:              auto-sized to {1/mean_ctx*100:.0f}% of raw default  "
        f"  [dim]({_RAW_NUM_CTX} → ~{_RAW_NUM_CTX/mean_ctx:.0f} tokens)[/dim]",
        f"    🎯 Tokens freed:                {report.total_tokens_saved():,} tokens across all runs",
        "",
        f"    🔄 Model reloads (raw vs auto):  "
        f"{report.total_reloads_raw()} vs {report.total_reloads_tuned()}",
        f"    💿 Swap events (raw vs auto):    "
        f"{report.total_swaps_raw()} vs {report.total_swaps_tuned()}",
        "",
        f"  [dim]{sig_count} metrics reached p<0.10. Effect sizes (Cohen's d) are independent "
        f"of sample size — use them alongside p-values.[/dim]",
    ]
    console.print(Panel("\n".join(lines), title="Model Summary", border_style=style))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model summary
# ─────────────────────────────────────────────────────────────────────────────

def print_cross_model_summary(reports: list[ModelReport], console: Console) -> None:
    valid = [r for r in reports if not r.skipped and r.prompt_stats]
    if not valid:
        return

    console.print()
    console.rule("[bold white]Cross-Model Summary[/bold white]", style="white")
    console.print()

    t = Table(box=_box.SIMPLE_HEAVY, header_style="bold", width=_CON_W)
    t.add_column("Model",         width=20)
    t.add_column("TTFT",          width=10, justify="right")
    t.add_column("Prefill",       width=10, justify="right")
    t.add_column("Speed",         width=10, justify="right")
    t.add_column("RAM",           width=10, justify="right")
    t.add_column("KV cache",      width=10, justify="right")
    t.add_column("ctx ratio",     width=10, justify="right")
    t.add_column("Tok saved",     width=12, justify="right")
    t.add_column("Win rate",      width=10, justify="right")

    for r in reports:
        if r.skipped:
            t.add_row(r.model_id, *["—"] * 8, "[yellow]SKIP[/yellow]")
            continue

        ps_list  = r.prompt_stats
        ttft_pct = statistics.mean([p.ttft.pct_change      for p in ps_list if p.ttft.n > 0])
        pre_pct  = statistics.mean([p.prefill_ms.pct_change for p in ps_list if p.prefill_ms.n > 0])
        tps_pct  = statistics.mean([p.eval_tps.pct_change   for p in ps_list if p.eval_tps.n > 0])
        ram_pct  = statistics.mean([p.ollama_ram.pct_change  for p in ps_list if p.ollama_ram.n > 0])
        kv_vals  = [p.kv_cache.pct_change for p in ps_list if p.kv_cache.n > 0 and p.kv_cache.raw_mean > 0]
        kv_pct   = statistics.mean(kv_vals) if kv_vals else float("nan")
        ctx_r    = statistics.mean([p.num_ctx_raw / p.num_ctx_tuned
                                    for p in ps_list if p.num_ctx_tuned > 0])
        wp       = r.overall_wins() / r.total_metrics() * 100

        t.add_row(
            r.model_id,
            _pct_cell(ttft_pct, higher_is_better=False),
            _pct_cell(pre_pct,  higher_is_better=False),
            _pct_cell(tps_pct,  higher_is_better=True),
            _pct_cell(ram_pct,  higher_is_better=False),
            _pct_cell(kv_pct,   higher_is_better=False) if not math.isnan(kv_pct) else Text("N/A", style="dim"),
            Text(f"{ctx_r:.1f}×", style="cyan"),
            Text(f"{r.total_tokens_saved():,}", style="cyan"),
            Text(f"{wp:.0f}%", style="green" if wp >= 60 else "yellow"),
        )
    console.print(t)

    # Grand summary
    mean_ttft = statistics.mean([
        statistics.mean([p.ttft.pct_change for p in r.prompt_stats if p.ttft.n > 0])
        for r in valid
    ])
    mean_tps = statistics.mean([
        statistics.mean([p.eval_tps.pct_change for p in r.prompt_stats if p.eval_tps.n > 0])
        for r in valid
    ])
    mean_ram = statistics.mean([
        statistics.mean([p.ollama_ram.pct_change for p in r.prompt_stats if p.ollama_ram.n > 0])
        for r in valid
    ])
    mean_ctx = statistics.mean([
        statistics.mean([p.num_ctx_raw / p.num_ctx_tuned
                         for p in r.prompt_stats if p.num_ctx_tuned > 0])
        for r in valid
    ])
    total_tok = sum(r.total_tokens_saved() for r in valid)
    total_swap_r = sum(r.total_swaps_raw()  for r in valid)
    total_swap_t = sum(r.total_swaps_tuned() for r in valid)

    n_prompt_types = len({p.id for r in valid for ps in r.prompt_stats for p in [ps.prompt]})
    console.print(Panel(
        f"Across [bold]{len(valid)} model(s)[/bold] and [bold]{n_prompt_types} prompt types[/bold]:\n\n"
        f"  [green]⏱  You wait {abs(mean_ttft):.0f}% less for the first word to appear.[/green]\n"
        f"  [green]⚡ Text generation is {mean_tps:+.0f}% faster.[/green]\n"
        f"  [green]🧠 Your Mac uses {abs(mean_ram):.0f}% less RAM during inference.[/green]\n"
        f"  [cyan]💾 autotune shrinks the context window to {1/mean_ctx*100:.0f}% of Ollama's "
        f"default — freeing KV cache without quality loss.[/cyan]\n"
        f"  [cyan]💡 {total_tok:,} KV tokens freed across all runs.[/cyan]\n"
        f"  [{'green' if total_swap_t < total_swap_r else 'yellow'}]"
        f"💿 Swap events: {total_swap_r} raw → {total_swap_t} autotune.[/{'green' if total_swap_t < total_swap_r else 'yellow'}]\n\n"
        f"  [dim]Statistical notes: Wilcoxon signed-rank (n<10), paired t-test (n≥10). "
        f"Cohen's d on paired differences. 95% CI via t-distribution. "
        f"KV cache estimates from model architecture (n_layers × n_kv_heads × head_dim).[/dim]",
        title="[bold]What this means for you[/bold]",
        border_style="green",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Quick-mode verdict panel
# ─────────────────────────────────────────────────────────────────────────────

def print_quick_verdict(report: ModelReport, console: Console) -> None:
    """Compact, focused verdict for --quick mode: 4 core KPIs + a clear scorecard."""
    if report.skipped or not report.prompt_stats:
        return

    ps_list  = report.prompt_stats
    ttft_pcts = [ps.ttft.pct_change       for ps in ps_list if ps.ttft.n > 0]
    pre_pcts  = [ps.prefill_ms.pct_change  for ps in ps_list if ps.prefill_ms.n > 0]
    ram_pcts  = [ps.ollama_ram.pct_change  for ps in ps_list if ps.ollama_ram.n > 0]
    kv_pcts   = [ps.kv_cache.pct_change    for ps in ps_list
                 if ps.kv_cache.n > 0 and ps.kv_cache.raw_mean > 0]
    ctx_ratios = [ps.num_ctx_raw / ps.num_ctx_tuned
                  for ps in ps_list if ps.num_ctx_tuned > 0]

    mean_ttft = statistics.mean(ttft_pcts)   if ttft_pcts  else 0.0
    mean_pre  = statistics.mean(pre_pcts)    if pre_pcts   else 0.0
    mean_ram  = statistics.mean(ram_pcts)    if ram_pcts   else 0.0
    mean_kv   = statistics.mean(kv_pcts)     if kv_pcts    else 0.0
    mean_ctx  = statistics.mean(ctx_ratios)  if ctx_ratios else 1.0

    wins    = report.overall_wins()
    total   = report.total_metrics()
    win_pct = wins / total * 100 if total else 0.0

    def _mk_row(icon: str, label: str, pct: float, lower_better: bool = True) -> str:
        improved   = (pct < 0) if lower_better else (pct > 0)
        color      = "green" if improved else "red"
        bar_filled = int(min(abs(pct), 100) / 100 * 14)
        bar        = "█" * bar_filled + "░" * (14 - bar_filled)
        sign       = "−" if pct < 0 else "+"
        return f"  {icon}  {label:<32}  [{color}]{sign}{abs(pct):.0f}%  {bar}[/{color}]"

    ctx_avg = _RAW_NUM_CTX / mean_ctx if mean_ctx else _RAW_NUM_CTX

    style = "green" if win_pct >= 60 else ("yellow" if win_pct >= 40 else "red")
    icon  = "✓" if win_pct >= 60 else ("≈" if win_pct >= 40 else "✗")

    lines: list[str] = [
        f"[bold {style}]{icon}  autotune won {wins}/{total} metrics "
        f"({win_pct:.0f}%) — {len(ps_list)} prompts · n={report.n_runs} runs/condition"
        f"[/bold {style}]",
        "",
        _mk_row("⏱", "Time to first token (TTFT)", mean_ttft),
        _mk_row("⚡", "KV prefill time",             mean_pre),
        _mk_row("🧠", "Peak RAM (Ollama process)",   mean_ram),
    ]
    if mean_kv != 0:
        lines.append(_mk_row("💾", "KV cache footprint", mean_kv))
    ctx_color = "cyan" if mean_ctx > 1.05 else "dim"
    lines += [
        f"  📐  {'Context window':<32}  [{ctx_color}]{_RAW_NUM_CTX} → "
        f"{ctx_avg:.0f} tokens  ({mean_ctx:.1f}× smaller)[/{ctx_color}]",
        f"  🎯  {'KV buffer slots freed':<32}  [cyan]{report.total_tokens_saved():,}"
        f" tokens across all runs[/cyan]",
        "",
        f"  [dim]Swap events:   raw {report.total_swaps_raw()} → "
        f"autotune {report.total_swaps_tuned()}[/dim]",
        f"  [dim]Model reloads: raw {report.total_reloads_raw()} → "
        f"autotune {report.total_reloads_tuned()}[/dim]",
        "",
        f"  [dim]Run [bold]--complete[/bold] for all 5 prompts, growth test, "
        f"and per-prompt stats tables.[/dim]",
    ]

    console.print()
    console.rule(
        f"[bold cyan]{report.model_id}[/bold cyan]  ·  "
        f"Quick Proof  ·  autotune/{report.profile}",
        style="cyan",
    )
    console.print(f"[dim]Hardware: {report.hw_str}[/dim]")
    mi = report.model_info
    if mi.is_valid():
        console.print(
            f"[dim]Architecture: {mi.n_layers} layers · "
            f"{mi.n_kv_heads} KV heads · head_dim={mi.head_dim} · "
            f"KV@4096={mi.estimate_kv_cache_mb(4096):.0f} MB (f16)[/dim]"
        )
    console.print()
    console.print(Panel(
        "\n".join(lines),
        title="[bold]autotune vs. raw Ollama — Quick Verdict[/bold]",
        border_style=style,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# JSON export
# ─────────────────────────────────────────────────────────────────────────────

def _sr_dict(sr: StatResult) -> dict:
    return {
        "metric":               sr.metric,
        "n":                    sr.n,
        "raw_mean":             round(sr.raw_mean,   3),
        "tuned_mean":           round(sr.tuned_mean, 3),
        "pct_change":           round(sr.pct_change, 2),
        "cohens_d":             round(sr.cohens_d,   3),
        "p_value":              round(sr.p_value,    4),
        "ci95_lo":              round(sr.ci95_lo,    3),
        "ci95_hi":              round(sr.ci95_hi,    3),
        "wins":                 sr.wins,
        "direction_consistent": sr.direction_consistent,
        "test_name":            sr.test_name,
        "significance":         sr.sig_stars or "ns",
        "effect_label":         sr.effect_label,
        "improved":             sr.improved,
    }


def export_json(reports: list[ModelReport], path: str, console: Console) -> None:
    # Use prompts from first valid report so the JSON reflects what actually ran
    first_valid  = next((r for r in reports if not r.skipped and r.prompt_stats), None)
    run_prompts  = [ps.prompt for ps in first_valid.prompt_stats] if first_valid else PROMPTS
    out: dict = {
        "tool":      "autotune proof suite",
        "version":   "2.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile":   PROFILE_NAME,
        "prompts":   [{"id": p.id, "label": p.label, "domain": p.domain} for p in run_prompts],
        "kpis_measured": [
            "TTFT (ms)", "Prefill time (ms)", "Total response time (ms)",
            "Peak RAM - LLM process (GB)", "KV cache size est. (MB)",
            "Total context size (tokens)", "Memory growth over turns (GB/turn)",
            "Swap pressure (bool + GB delta)", "Model reload count",
            "Context size per request (tokens)", "Tokens saved by context shrink",
        ],
        "models": [],
    }

    for r in reports:
        if r.skipped:
            out["models"].append({
                "model_id": r.model_id, "skipped": True, "reason": r.skip_reason
            })
            continue

        mi = r.model_info
        model_entry: dict = {
            "model_id":  r.model_id,
            "n_runs":    r.n_runs,
            "hardware":  r.hw_str,
            "win_rate":  round(r.overall_wins() / r.total_metrics() * 100, 1),
            "total_tokens_saved": r.total_tokens_saved(),
            "total_swaps_raw":    r.total_swaps_raw(),
            "total_swaps_tuned":  r.total_swaps_tuned(),
            "total_reloads_raw":  r.total_reloads_raw(),
            "total_reloads_tuned":r.total_reloads_tuned(),
            "model_architecture": {
                "n_layers":   mi.n_layers,
                "n_kv_heads": mi.n_kv_heads,
                "head_dim":   mi.head_dim,
                "kv_mb_at_4096_f16": round(mi.estimate_kv_cache_mb(4096, 2), 1),
            } if mi.is_valid() else None,
            "memory_growth": {
                "raw":   [
                    {"turn": g.turn, "num_ctx": g.num_ctx,
                     "prompt_tokens": g.prompt_tokens,
                     "total_tokens": g.total_tokens,
                     "ollama_ram_gb": round(g.ollama_ram_gb, 3),
                     "prefill_ms": round(g.prefill_ms, 1),
                     "kv_cache_mb": round(g.kv_cache_mb, 1)}
                    for g in r.raw_growth
                ],
                "tuned": [
                    {"turn": g.turn, "num_ctx": g.num_ctx,
                     "prompt_tokens": g.prompt_tokens,
                     "total_tokens": g.total_tokens,
                     "ollama_ram_gb": round(g.ollama_ram_gb, 3),
                     "prefill_ms": round(g.prefill_ms, 1),
                     "kv_cache_mb": round(g.kv_cache_mb, 1)}
                    for g in r.tuned_growth
                ],
            },
            "prompts": [],
        }

        for ps in r.prompt_stats:
            model_entry["prompts"].append({
                "prompt_id":     ps.prompt.id,
                "prompt_label":  ps.prompt.label,
                "num_ctx_raw":   ps.num_ctx_raw,
                "num_ctx_tuned": ps.num_ctx_tuned,
                "tokens_saved":  ps.tokens_saved,
                "swap_occurred_raw":   ps.swap_occurred_raw,
                "swap_occurred_tuned": ps.swap_occurred_tuned,
                "reload_count_raw":    ps.reload_count_raw,
                "reload_count_tuned":  ps.reload_count_tuned,
                # All KPI stat results
                "ttft":       _sr_dict(ps.ttft),
                "prefill_ms": _sr_dict(ps.prefill_ms),
                "eval_tps":   _sr_dict(ps.eval_tps),
                "total_ms":   _sr_dict(ps.total_ms),
                "ollama_ram": _sr_dict(ps.ollama_ram),
                "kv_cache":   _sr_dict(ps.kv_cache),
                "raw_runs": [
                    {
                        "prefill_ms":     round(r2.prefill_ms, 3),
                        "load_ms":        round(r2.load_ms, 3),
                        "ttft_ms":        round(r2.ttft_ms, 3),
                        "eval_tps":       round(r2.eval_tps, 3),
                        "total_ms":       round(r2.total_ms, 3),
                        "eval_count":     r2.eval_count,
                        "prompt_tokens":  r2.prompt_eval_count,
                        "ollama_peak_gb": round(r2.ollama_peak_ram_gb, 3),
                        "swap_delta_gb":  round(r2.swap_delta_gb, 4),
                        "swap_occurred":  r2.swap_occurred,
                        "kv_cache_mb":    round(r2.kv_cache_mb, 1),
                        "reload_detected":r2.reload_detected,
                        "num_ctx":        r2.num_ctx,
                        "ok":             r2.ok,
                    }
                    for r2 in ps.raw_runs
                ],
                "tuned_runs": [
                    {
                        "prefill_ms":     round(r2.prefill_ms, 3),
                        "load_ms":        round(r2.load_ms, 3),
                        "ttft_ms":        round(r2.ttft_ms, 3),
                        "eval_tps":       round(r2.eval_tps, 3),
                        "total_ms":       round(r2.total_ms, 3),
                        "eval_count":     r2.eval_count,
                        "prompt_tokens":  r2.prompt_eval_count,
                        "ollama_peak_gb": round(r2.ollama_peak_ram_gb, 3),
                        "swap_delta_gb":  round(r2.swap_delta_gb, 4),
                        "swap_occurred":  r2.swap_occurred,
                        "kv_cache_mb":    round(r2.kv_cache_mb, 1),
                        "reload_detected":r2.reload_detected,
                        "num_ctx":        r2.num_ctx,
                        "ok":             r2.ok,
                    }
                    for r2 in ps.tuned_runs
                ],
            })
        out["models"].append(model_entry)

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    console.print(f"[dim]Full results saved → {path}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Ollama availability
# ─────────────────────────────────────────────────────────────────────────────

def _probe_ollama() -> bool:
    try:
        r = httpx.get(f"{_OLLAMA_BASE}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _installed_models() -> list[str]:
    try:
        r = httpx.get(f"{_OLLAMA_BASE}/api/tags", timeout=2.0)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="proof_suite",
        description="autotune proof suite v2.0 — multi-model scientific benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Quick proof on best-available model (~5-8 min, DEFAULT):
  python scripts/proof_suite.py

  # Quick proof on a specific model:
  python scripts/proof_suite.py --models llama3.2:3b

  # Comprehensive run on all 3 default models (~20-30 min):
  python scripts/proof_suite.py --complete

  # Complete run with more statistical power, saved to file:
  python scripts/proof_suite.py --complete --runs 5 --output proof.json

  # Skip the memory growth test in complete mode:
  python scripts/proof_suite.py --complete --no-growth

  # List installed models:
  python scripts/proof_suite.py --list-models
        """,
    )
    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--quick", action="store_true", default=False,
        help=(
            "Quick proof: 3 KV-heavy prompts, 2 runs/condition, no growth test, "
            "short cooldowns, auto-selects one model. ~5-8 min. [DEFAULT]"
        ),
    )
    mode_grp.add_argument(
        "--complete", action="store_true", default=False,
        help=(
            "Full benchmark: all 5 prompts, 3 runs/condition, growth test, "
            "full cooldowns, all default models. ~20-30 min."
        ),
    )
    p.add_argument(
        "--models", nargs="+", metavar="MODEL", default=None,
        help=(
            "Ollama model IDs to benchmark. Quick mode defaults to the first "
            f"available from {DEFAULT_MODELS}; complete mode runs all three "
            "(or AUTOTUNE_PROOF_MODELS env var)."
        ),
    )
    p.add_argument(
        "--runs", "-n", type=int, default=None, metavar="N",
        help=(
            "Override runs per condition per prompt. "
            "Defaults: 2 (quick) / 3 (complete). Min 3 for statistics."
        ),
    )
    p.add_argument(
        "--profile", default=PROFILE_NAME,
        choices=["fast", "balanced", "quality"],
        help="autotune profile to compare against raw Ollama. (default: balanced)",
    )
    p.add_argument(
        "--output", "-o", metavar="PATH",
        help="Save full results to a JSON file.",
    )
    p.add_argument(
        "--no-growth", action="store_true",
        help="Skip the multi-turn memory growth test (complete mode only).",
    )
    p.add_argument(
        "--list-models", action="store_true",
        help="List locally installed Ollama models and exit.",
    )
    return p


def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()
    console = Console(width=_CON_W)

    if args.list_models:
        if not _probe_ollama():
            console.print("[red]Ollama is not running.[/red]")
            sys.exit(1)
        models = _installed_models()
        console.print("\n[bold]Installed Ollama models:[/bold]")
        for m in models:
            console.print(f"  {m}")
        console.print()
        return

    if not _probe_ollama():
        console.print(Panel(
            "[red]No models found or Ollama could not start.[/red]\n\n"
            "Pull a model with:  [bold]autotune pull llama3.2:3b[/bold]\n"
            "(autotune starts Ollama automatically)",
            title="Error", border_style="red",
        ))
        sys.exit(1)

    # Determine mode — default is quick unless --complete is specified
    is_complete = args.complete
    run_cfg: RunConfig = COMPLETE_CONFIG if is_complete else QUICK_CONFIG

    # Override runs if caller supplied --runs
    if args.runs is not None:
        run_cfg = dc_replace(run_cfg, n_runs=args.runs)

    # --no-growth only applies in complete mode
    if args.no_growth and is_complete:
        run_cfg = dc_replace(run_cfg, run_growth=False)

    # Resolve model list
    installed  = _installed_models()
    env_models = os.environ.get("AUTOTUNE_PROOF_MODELS", "")

    if args.models:
        models = args.models
    elif env_models:
        models = [m.strip() for m in env_models.split(",") if m.strip()]
    elif is_complete:
        models = DEFAULT_MODELS
    else:
        # Quick mode: auto-select the first available default model
        models = []
        for m in DEFAULT_MODELS:
            if m in installed:
                models = [m]
                break
        if not models:
            if installed:
                models = [installed[0]]
            else:
                console.print(Panel(
                    "[red]No Ollama models are installed.[/red]\n\n"
                    "Pull one with:  [bold]autotune pull llama3.2:3b[/bold]",
                    title="No models found", border_style="red",
                ))
                sys.exit(1)

    mode_label = "complete" if is_complete else "quick"
    console.print()
    console.print(Panel(
        f"[bold]autotune proof suite v2.0[/bold]  ·  [cyan]{mode_label} mode[/cyan]\n\n"
        f"Models:  {', '.join(models)}\n"
        f"Profile: autotune/{args.profile}\n"
        f"Runs:    {run_cfg.n_runs} per condition per prompt\n"
        f"Prompts: {', '.join(p.label for p in run_cfg.prompts)}\n"
        f"Growth:  {'yes' if run_cfg.run_growth else 'no'}\n"
        f"KPIs:    TTFT · Prefill · Throughput · RAM · KV cache · "
        f"Context · Swap · Reloads · Tokens freed\n\n"
        f"[dim]Each model runs sequentially to avoid memory contention.\n"
        f"Swap usage: {psutil.swap_memory().used/1024**3:.1f} GB / "
        f"{psutil.swap_memory().total/1024**3:.1f} GB before benchmark.[/dim]",
        title="[bold cyan]autotune proof suite[/bold cyan]",
        border_style="cyan",
    ))

    reports: list[ModelReport] = []

    for model_id in models:
        console.print()
        if model_id not in installed:
            console.print(
                Panel(
                    f"[yellow]{model_id}[/yellow] is not installed.\n\n"
                    f"Pull it with:  [bold]autotune pull {model_id}[/bold]\n\n"
                    f"Installed models:\n"
                    + "\n".join(f"  {m}" for m in installed),
                    title="Model not found", border_style="yellow",
                )
            )
            reports.append(ModelReport(
                model_id=model_id, profile=args.profile, n_runs=run_cfg.n_runs,
                hw_str="", timestamp=time.time(),
                model_info=OllamaModelInfo(model_id=model_id),
                prompt_stats=[], raw_growth=[], tuned_growth=[],
                skipped=True, skip_reason="not installed",
            ))
            continue

        console.rule(f"[bold]Benchmarking {model_id}[/bold]", style="cyan")
        try:
            report = asyncio.run(run_model_benchmark(
                model_id, run_cfg, args.profile, console,
            ))
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted — saving partial results.[/yellow]")
            break
        except Exception as exc:
            console.print(f"[red]Error benchmarking {model_id}: {exc}[/red]")
            reports.append(ModelReport(
                model_id=model_id, profile=args.profile, n_runs=run_cfg.n_runs,
                hw_str="", timestamp=time.time(),
                model_info=OllamaModelInfo(model_id=model_id),
                prompt_stats=[], raw_growth=[], tuned_growth=[],
                skipped=True, skip_reason=str(exc),
            ))
            continue

        reports.append(report)

        # Choose output format based on mode
        if is_complete:
            print_model_report(report, console)
        else:
            print_quick_verdict(report, console)

        # Pause between models so Ollama can free memory
        if model_id != models[-1]:
            console.print(f"\n[dim]Pausing 20s before next model…[/dim]")
            time.sleep(20)

    if len([r for r in reports if not r.skipped]) > 1:
        print_cross_model_summary(reports, console)

    if args.output:
        export_json(reports, args.output, console)
    elif reports:
        default_path = str(_ROOT / "proof_results.json")
        export_json(reports, default_path, console)


if __name__ == "__main__":
    main()
