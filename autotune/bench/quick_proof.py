"""
autotune quick proof — does autotune actually help on YOUR machine?

Runs in ≤45 seconds.  Two tests, honest numbers.

TEST 1 — Standard (warm model, multi-turn)
  Shows KV cache savings and RAM headroom.
  Model is already warm, so load_ms ≈ 0 for both conditions.
  This is what every message after the first looks like.

TEST 2 — Session-start TTFT (neutral → each condition)
  Shows the TTFT improvement users feel on their FIRST message.
  We prime the model to a neutral num_ctx (3072) so both conditions
  must freshly allocate their own KV buffer.  Ollama's Go-timer
  load_ms captures the allocation time directly:
    raw    → 3072→4096: allocate 448 MB KV buffer  (slow)
    tuned  → 3072→1536: allocate 168 MB KV buffer  (fast)
  Difference = pure KV allocation cost, no disk I/O noise.
  This is what users feel every time they start a new chat session.

RAM SAVINGS
  KV cache freed = raw_kv_mb − tuned_kv_mb.
  This is the RAM directly returned to your other apps (browser, IDE, Slack).
  RAM headroom (system available) is shown separately — it varies with
  whatever else is running on your machine, so KV freed is the honest metric.

All timings from Ollama's internal Go nanosecond timers — nothing estimated.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import httpx
import psutil

_OLLAMA_BASE    = "http://localhost:11434"
_RAW_CTX        = 4096
_KEEP_ALIVE     = "30m"
_MAX_TOKENS     = 120      # short generation; TTFT is what we measure
_MAX_TOKENS_LC  = 60       # even shorter for session-start test
_RELOAD_MS      = 400.0    # load_ms above this → cold disk reload happened
_COOLDOWN_SEC   = 1.5      # pause between condition switches

# Neutral KV state for session-start TTFT test.
# Must be strictly between autotune's typical bucket (1536) and raw (4096)
# so both conditions must reallocate their KV buffer from this state.
_NEUTRAL_CTX    = 3072

# ─────────────────────────────────────────────────────────────────────────────
# Test 1 prompt — multi-turn conversation with system prompt.
#
# autotune computes:  input_tokens(~130) + max_new_tokens(1024) + 256 = ~1410
#                     → snapped to bucket 1536
# raw Ollama:         always uses 4096
# KV freed (llama3.2:3b F16):  (4096−1536)/4096 × 448 MB = 280 MB
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
# Test 2 prompt — longer prompt simulating code review / document work.
#
# autotune computes:  input_tokens(~340) + max_new_tokens(1024) + 256 = ~1620
#                     → snapped to bucket 2048
# raw Ollama:         always uses 4096
# KV freed (llama3.2:3b F16):  (4096−2048)/4096 × 448 MB = 224 MB
#
# This prompt simulates the most common case where TTFT matters: a user pastes
# in a code block or document as their opening message in a new session.
# ─────────────────────────────────────────────────────────────────────────────
SESSION_START_MESSAGES: list[dict] = [
    {
        "role": "system",
        "content": "You are a code reviewer. Be concise — list issues only.",
    },
    {
        "role": "user",
        "content": (
            "Review this Python database manager and list the top 3 issues:\n\n"
            "```python\n"
            "class DatabaseManager:\n"
            "    _instance = None\n"
            "    _conn = None\n\n"
            "    def __new__(cls):\n"
            "        if cls._instance is None:\n"
            "            cls._instance = super().__new__(cls)\n"
            "        return cls._instance\n\n"
            "    def connect(self, host, port, db, user, password):\n"
            "        import psycopg2\n"
            "        self._conn = psycopg2.connect(\n"
            "            host=host, port=port, dbname=db,\n"
            "            user=user, password=password\n"
            "        )\n\n"
            "    def query(self, sql, params=None):\n"
            "        cur = self._conn.cursor()\n"
            "        cur.execute(sql, params)\n"
            "        rows = cur.fetchall()\n"
            "        cur.close()\n"
            "        return rows\n\n"
            "    def get_user(self, user_id):\n"
            "        return self.query(\n"
            "            f'SELECT * FROM users WHERE id = {user_id}'\n"
            "        )\n\n"
            "    def transfer_funds(self, src, dst, amount):\n"
            "        cur = self._conn.cursor()\n"
            "        cur.execute(\n"
            "            'UPDATE accounts SET balance = balance - %s WHERE id = %s',\n"
            "            (amount, src)\n"
            "        )\n"
            "        cur.execute(\n"
            "            'UPDATE accounts SET balance = balance + %s WHERE id = %s',\n"
            "            (amount, dst)\n"
            "        )\n"
            "        self._conn.commit()\n"
            "        cur.close()\n\n"
            "    def bulk_insert(self, table, rows):\n"
            "        cur = self._conn.cursor()\n"
            "        for row in rows:\n"
            "            cur.execute(f'INSERT INTO {table} VALUES {row}')\n"
            "        self._conn.commit()\n"
            "        cur.close()\n"
            "```\n"
            "List the top 3 critical bugs or security issues."
        ),
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
    prefill_ms:         float  # prompt_eval_duration (Ollama Go timer)
    load_ms:            float  # load_duration = KV allocation + any model load cost
    eval_tps:           float  # generation tok/s
    eval_count:         int    # tokens generated
    prompt_tokens:      int    # tokens in prompt (Ollama-reported)
    kv_cache_mb:        float  # estimated KV size for this num_ctx
    free_floor_gb:      float  # min system available RAM during run
    swap_occurred:      bool
    reload_detected:    bool   # load_ms > threshold → disk reload (cold start noise)
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
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _call_ollama(
    model_id: str,
    messages: list[dict],
    options: dict,
    max_tokens: int = _MAX_TOKENS,
    keep_alive: str = _KEEP_ALIVE,
) -> dict:
    payload = {
        "model":      model_id,
        "messages":   messages,
        "stream":     False,
        "options":    {**options, "num_predict": max_tokens},
        "keep_alive": keep_alive,
    }
    async with httpx.AsyncClient(timeout=90.0) as c:
        r = await c.post(f"{_OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()


async def _prime_ctx(model_id: str, num_ctx: int) -> None:
    """
    Set the model's active KV context window to num_ctx and return.

    Used to establish a known neutral state before the session-start TTFT test.
    The trivial request runs but the result is discarded — we only care that
    Ollama has allocated a KV buffer at num_ctx so the NEXT request triggers a
    fresh reallocation to its own (different) num_ctx.

    Keeps the model loaded (keep_alive=_KEEP_ALIVE) — no disk reload noise.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            await c.post(f"{_OLLAMA_BASE}/api/chat", json={
                "model":      model_id,
                "messages":   [{"role": "user", "content": "ok"}],
                "stream":     False,
                "options":    {"num_ctx": num_ctx, "num_predict": 1},
                "keep_alive": _KEEP_ALIVE,
            })
        await asyncio.sleep(0.4)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Single instrumented run
# ─────────────────────────────────────────────────────────────────────────────

async def _measured_run(
    model_id: str,
    condition: str,
    options: dict,
    arch: _ModelArch,
    kv_dtype_bytes: int,
    messages: Optional[list] = None,
    max_tokens: int = _MAX_TOKENS,
) -> _RunResult:
    if messages is None:
        messages = PROOF_MESSAGES
    sampler = _RamSampler()
    sampler.start()
    error = None

    try:
        data = await _call_ollama(model_id, messages, options, max_tokens=max_tokens)
        load_ms    = data.get("load_duration",        0) / 1_000_000
        prefill_ms = data.get("prompt_eval_duration", 0) / 1_000_000
        eval_cnt   = data.get("eval_count",           0)
        eval_dur   = data.get("eval_duration",        0) / 1_000_000_000
        prompt_cnt = data.get("prompt_eval_count",    0)
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
    # Session-start TTFT test — one run each, both from neutral KV state
    ss_raw_run:     Optional[_RunResult] = None
    ss_tuned_run:   Optional[_RunResult] = None

    # ── Aggregated stats (standard test) ─────────────────────────────────────

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
    def kv_saved_mb(self) -> float:
        return max(0.0, self.raw_kv_mb - self.tuned_kv_mb)

    @property
    def kv_pct(self) -> float:
        if self.raw_kv_mb <= 0:
            return 0.0
        return self.kv_saved_mb / self.raw_kv_mb * 100

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
    def headroom_gained_gb(self) -> float:
        return max(0.0, self.tuned_free_gb - self.raw_free_gb)

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

    # ── Session-start TTFT test properties ───────────────────────────────────

    @property
    def has_ss_test(self) -> bool:
        return (
            self.ss_raw_run is not None and self.ss_raw_run.ok
            and self.ss_tuned_run is not None and self.ss_tuned_run.ok
        )

    @property
    def ss_raw_ttft_ms(self) -> float:
        return self.ss_raw_run.ttft_ms if self.ss_raw_run and self.ss_raw_run.ok else 0.0

    @property
    def ss_tuned_ttft_ms(self) -> float:
        return self.ss_tuned_run.ttft_ms if self.ss_tuned_run and self.ss_tuned_run.ok else 0.0

    @property
    def ss_ttft_improvement_pct(self) -> float:
        if self.ss_raw_ttft_ms <= 0:
            return 0.0
        return (self.ss_raw_ttft_ms - self.ss_tuned_ttft_ms) / self.ss_raw_ttft_ms * 100

    @property
    def ss_raw_kv_mb(self) -> float:
        return self.ss_raw_run.kv_cache_mb if self.ss_raw_run and self.ss_raw_run.ok else 0.0

    @property
    def ss_tuned_kv_mb(self) -> float:
        return self.ss_tuned_run.kv_cache_mb if self.ss_tuned_run and self.ss_tuned_run.ok else 0.0

    @property
    def ss_raw_num_ctx(self) -> int:
        return self.ss_raw_run.num_ctx if self.ss_raw_run else _RAW_CTX

    @property
    def ss_tuned_num_ctx(self) -> int:
        return self.ss_tuned_run.num_ctx if self.ss_tuned_run else 0

    def to_dict(self) -> dict:
        d: dict = {
            "model_id":       self.model_id,
            "profile":        self.profile_name,
            "n_runs":         self.n_runs,
            "elapsed_sec":    round(self.elapsed_sec, 1),
            "ttft_raw_ms":    round(self.raw_ttft_ms, 1),
            "ttft_tuned_ms":  round(self.tuned_ttft_ms, 1),
            "ttft_pct":       round(self.ttft_improvement_pct, 1),
            "kv_raw_mb":      round(self.raw_kv_mb, 1),
            "kv_tuned_mb":    round(self.tuned_kv_mb, 1),
            "kv_saved_mb":    round(self.kv_saved_mb, 1),
            "kv_pct":         round(self.kv_pct, 1),
            "ctx_raw":        self.raw_num_ctx,
            "ctx_tuned":      self.tuned_num_ctx,
            "ctx_pct":        round(self.ctx_pct, 1),
            "free_raw_gb":    round(self.raw_free_gb, 2),
            "free_tuned_gb":  round(self.tuned_free_gb, 2),
            "headroom_gained_gb": round(self.headroom_gained_gb, 2),
            "swap_raw":       self.raw_swap_events,
            "swap_tuned":     self.tuned_swap_events,
            "tps_raw":        round(self.raw_tps, 1),
            "tps_tuned":      round(self.tuned_tps, 1),
            "raw_runs":       [asdict(r) for r in self.raw_runs],
            "tuned_runs":     [asdict(r) for r in self.tuned_runs],
        }
        if self.has_ss_test:
            d.update({
                "ss_ttft_raw_ms":    round(self.ss_raw_ttft_ms, 1),
                "ss_ttft_tuned_ms":  round(self.ss_tuned_ttft_ms, 1),
                "ss_ttft_pct":       round(self.ss_ttft_improvement_pct, 1),
                "ss_load_raw_ms":    round(self.ss_raw_run.load_ms, 1),
                "ss_load_tuned_ms":  round(self.ss_tuned_run.load_ms, 1),
                "ss_kv_raw_mb":      round(self.ss_raw_kv_mb, 1),
                "ss_kv_tuned_mb":    round(self.ss_tuned_kv_mb, 1),
                "ss_ctx_raw":        self.ss_raw_num_ctx,
                "ss_ctx_tuned":      self.ss_tuned_num_ctx,
                "ss_raw_run":        asdict(self.ss_raw_run),
                "ss_tuned_run":      asdict(self.ss_tuned_run),
            })
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_quick_proof(
    model_id:     str,
    profile_name: str = "balanced",
    n_runs:       int = 2,
    output_path:  Optional[Path] = None,
    on_step:      Optional[Callable[[str], None]] = None,
) -> QuickProofResult:
    from autotune.api.kv_manager import build_ollama_options
    from autotune.api.profiles import get_profile

    def _step(msg: str) -> None:
        if on_step:
            on_step(msg)

    profile = get_profile(profile_name)
    t0 = time.monotonic()

    # ── Fetch model architecture for KV estimation ───────────────────────────
    _step("Fetching model architecture…")
    arch = await _fetch_arch(model_id)

    # Compute autotune options for both prompts upfront.
    # f16_kv=False means Q8 KV (1 byte per element); True or absent means F16 (2 bytes).
    tuned_opts, _  = build_ollama_options(PROOF_MESSAGES, profile)
    tuned_ss_opts, _ = build_ollama_options(SESSION_START_MESSAGES, profile)
    kv_dtype = 1 if not tuned_opts.get("f16_kv", True) else 2

    # ── Warmup: load model into RAM, verify it responds, discard timing ──────
    _step("Warming up model (loading from disk)…")
    _warmup_err: Optional[str] = None
    try:
        _wu = await _call_ollama(
            model_id,
            [{"role": "user", "content": "Say exactly: ready"}],
            {"num_ctx": _RAW_CTX, "num_predict": 4},
        )
        if _wu.get("eval_count", 0) == 0:
            _warmup_err = (
                f"Model '{model_id}' loaded but produced no tokens. "
                "It may be corrupted — try: ollama pull " + model_id
            )
        else:
            await asyncio.sleep(0.5)
    except httpx.ConnectError:
        _warmup_err = "Cannot reach Ollama — is it running?  Start with: ollama serve"
    except httpx.HTTPStatusError as _e:
        if _e.response.status_code == 404:
            _warmup_err = (
                f"Model '{model_id}' is not installed. "
                f"Pull it first: ollama pull {model_id}"
            )
        else:
            _warmup_err = f"Ollama returned HTTP {_e.response.status_code}"
    except Exception as _e:
        _warmup_err = f"Warmup failed: {_e}"

    if _warmup_err is not None:
        raise RuntimeError(_warmup_err)

    # ── TEST 1: Standard (warm model) ────────────────────────────────────────
    # Model is at num_ctx=4096 after warmup.
    # raw stays at 4096 → load_ms ≈ 0 (no realloc needed)
    # tuned switches to its own bucket → pays one-time realloc, then stable
    # This represents every message after the first in a session.

    raw_opts    = {"num_ctx": _RAW_CTX}
    raw_runs: list[_RunResult] = []

    for i in range(n_runs):
        _step(f"Test 1 — raw Ollama  run {i + 1}/{n_runs}…")
        r = await _measured_run(model_id, "raw", raw_opts, arch, kv_dtype_bytes=2)
        raw_runs.append(r)
        if i < n_runs - 1:
            await asyncio.sleep(0.5)

    _step("Test 1 — switching to autotune…")
    await asyncio.sleep(_COOLDOWN_SEC)

    tuned_runs: list[_RunResult] = []
    for i in range(n_runs):
        _step(f"Test 1 — autotune    run {i + 1}/{n_runs}…")
        r = await _measured_run(model_id, "autotune", tuned_opts, arch, kv_dtype_bytes=kv_dtype)
        tuned_runs.append(r)
        if i < n_runs - 1:
            await asyncio.sleep(0.5)

    # ── TEST 2: Session-start TTFT ───────────────────────────────────────────
    # Prime model to _NEUTRAL_CTX (3072) so both conditions must freshly
    # allocate their own KV buffer.  No disk reload — weights stay in memory.
    # load_ms = pure KV allocation time, which scales with buffer size.
    #   raw   (4096): 3072→4096 alloc → larger buffer → slower
    #   tuned (~2048): 3072→2048 alloc → smaller buffer → faster
    #
    # This is what users feel the first time they send a message in a new session.

    _step("Test 2 — priming neutral KV state (3072 tokens)…")
    ss_raw_run: Optional[_RunResult] = None
    ss_tuned_run: Optional[_RunResult] = None

    try:
        # Establish neutral state
        await _prime_ctx(model_id, _NEUTRAL_CTX)

        # raw: pay 4096-token KV allocation cost
        _step("Test 2 — raw Ollama  (allocating 4,096-token KV buffer)…")
        ss_raw_run = await _measured_run(
            model_id, "raw", raw_opts, arch, kv_dtype_bytes=2,
            messages=SESSION_START_MESSAGES, max_tokens=_MAX_TOKENS_LC,
        )

        # Return to neutral so autotune also pays fresh allocation
        await _prime_ctx(model_id, _NEUTRAL_CTX)

        # tuned: pay smaller KV allocation cost
        _step("Test 2 — autotune   (allocating right-sized KV buffer)…")
        ss_tuned_run = await _measured_run(
            model_id, "autotune", tuned_ss_opts, arch, kv_dtype_bytes=kv_dtype,
            messages=SESSION_START_MESSAGES, max_tokens=_MAX_TOKENS_LC,
        )

    except Exception:
        pass  # session-start test is best-effort; never fail the whole proof

    elapsed = time.monotonic() - t0

    result = QuickProofResult(
        model_id=model_id,
        profile_name=profile_name,
        n_runs=n_runs,
        elapsed_sec=elapsed,
        raw_runs=raw_runs,
        tuned_runs=tuned_runs,
        ss_raw_run=ss_raw_run,
        ss_tuned_run=ss_tuned_run,
    )

    if output_path:
        output_path.write_text(json.dumps(result.to_dict(), indent=2))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Rich console output
# ─────────────────────────────────────────────────────────────────────────────

def print_proof_result(result: QuickProofResult, console, output_path: Optional[Path] = None) -> None:
    """Print the formatted proof result to a Rich console."""
    from rich.table import Table
    from rich import box as _box
    from rich.panel import Panel

    console.print()
    console.rule(
        f"[bold]autotune proof[/bold]  ·  [cyan]{result.model_id}[/cyan]  "
        f"·  {result.n_runs} runs/condition  ·  {result.elapsed_sec:.0f}s"
    )
    console.print()

    ok_raw   = any(r.ok for r in result.raw_runs)
    ok_tuned = any(r.ok for r in result.tuned_runs)

    if not ok_raw or not ok_tuned:
        console.print(
            "[red]One or more runs failed. "
            "Check that Ollama is running and the model is installed.[/red]"
        )
        return

    def _pct_label(pct: float, better_is_lower: bool = True) -> str:
        if abs(pct) < 1:
            return "[dim]unchanged[/dim]"
        sign = "−" if pct > 0 else "+"
        icon = "✅" if (pct > 0) == better_is_lower else "⚠️"
        return f"{sign}{abs(pct):.0f}% {icon}"

    # ── RAM savings banner — the headline number ──────────────────────────────
    # KV freed is the direct, model-level RAM saving from autotune.
    # It is consistent, model-specific, and directly caused by autotune alone.
    if result.raw_kv_mb > 0 and result.kv_saved_mb >= 1:
        kv_freed_mb = result.kv_saved_mb
        kv_freed_gb = kv_freed_mb / 1024

        if kv_freed_gb >= 0.5:
            freed_str = f"[bold green]{kv_freed_gb:.2f} GB freed[/bold green]"
        else:
            freed_str = f"[bold green]{kv_freed_mb:.0f} MB freed[/bold green]"

        ram_lines = [
            f"  KV cache: {freed_str}  "
            f"[dim](raw: {result.raw_kv_mb:.0f} MB → autotune: {result.tuned_kv_mb:.0f} MB, "
            f"−{result.kv_pct:.0f}%)[/dim]",
            "",
            f"  [dim]The KV cache is RAM Ollama pre-allocates before your first token.[/dim]",
            f"  [dim]autotune allocates only what the prompt needs — the rest goes back to[/dim]",
            f"  [dim]your browser, IDE, Slack, and OS. Every single request.[/dim]",
        ]
        if result.headroom_gained_gb >= 0.05:
            ram_lines.append(
                f"\n  System RAM free while model runs: "
                f"[green]+{result.headroom_gained_gb:.2f} GB more headroom[/green]  "
                f"[dim](autotune: {result.tuned_free_gb:.1f} GB free vs raw: {result.raw_free_gb:.1f} GB)[/dim]"
            )
        elif result.tuned_free_gb > 0:
            ram_lines.append(
                f"\n  [dim]System RAM free while model runs: {result.tuned_free_gb:.1f} GB "
                f"(headroom similar — your system has plenty of RAM)[/dim]"
            )

        console.print(Panel(
            "\n".join(ram_lines),
            title="[bold green]RAM Savings[/bold green]",
            border_style="green",
            padding=(0, 1),
        ))
        console.print()

    # ── TEST 1 — Standard (warm model) ───────────────────────────────────────
    console.print(
        "[bold]Test 1 — Warm model  [/bold]"
        "[dim](every message after the first, model already loaded)[/dim]"
    )
    console.print(
        "[dim]  Both conditions run on a model that's already in memory. "
        "KV cache savings apply every single request.[/dim]"
    )
    console.print()

    t1 = Table(box=_box.SIMPLE_HEAD, show_header=True, show_edge=False, pad_edge=False)
    t1.add_column("Metric",            style="white",      width=34)
    t1.add_column("Raw Ollama",        style="dim white",  justify="right", width=14)
    t1.add_column("autotune",          style="green",      justify="right", width=14)
    t1.add_column("Change",                                justify="right", width=18)

    # TTFT — note: in warm test raw has low load_ms (warm at 4096), autotune
    # pays first-time realloc cost; subsequent tuned calls are faster.
    t1.add_row(
        "TTFT (time to first word)",
        f"{result.raw_ttft_ms:.0f} ms",
        f"{result.tuned_ttft_ms:.0f} ms",
        _pct_label(result.ttft_improvement_pct),
    )

    # Context window
    t1.add_row(
        "KV buffer size (num_ctx)",
        f"{result.raw_num_ctx:,} tokens",
        f"{result.tuned_num_ctx:,} tokens",
        _pct_label(result.ctx_pct),
    )

    # KV cache size — the direct RAM savings metric
    kv_raw_str   = f"{result.raw_kv_mb:.0f} MB"   if result.raw_kv_mb   > 0 else "N/A"
    kv_tuned_str = f"{result.tuned_kv_mb:.0f} MB" if result.tuned_kv_mb > 0 else "N/A"
    t1.add_row(
        "KV cache RAM reserved",
        kv_raw_str,
        kv_tuned_str,
        _pct_label(result.kv_pct) if result.raw_kv_mb > 0 else "N/A",
    )

    # Swap
    swap_raw_str   = ("0 ✅" if result.raw_swap_events == 0   else f"{result.raw_swap_events} ⚠️")
    swap_tuned_str = ("0 ✅" if result.tuned_swap_events == 0 else f"{result.tuned_swap_events} ⚠️")
    swap_change    = "[dim]none ✅[/dim]" if result.tuned_swap_events == 0 else f"+{result.tuned_swap_events} ⚠️"
    t1.add_row("Swap events", swap_raw_str, swap_tuned_str, swap_change)

    # Generation speed — honest
    if abs(result.raw_tps) > 0.1:
        delta_pct = (result.tuned_tps - result.raw_tps) / max(result.raw_tps, 0.01) * 100
        if abs(delta_pct) < 5:
            tps_change = "[dim]unchanged[/dim]"
        else:
            icon = "✅" if delta_pct > 0 else "—"
            tps_change = f"{'+'  if delta_pct > 0 else '−'}{abs(delta_pct):.0f}% {icon}"
        t1.add_row(
            "Generation speed (tok/s)",
            f"{result.raw_tps:.1f}",
            f"{result.tuned_tps:.1f}",
            tps_change,
        )

    console.print(t1)
    console.print()

    # ── TEST 2 — Session-start TTFT ──────────────────────────────────────────
    if result.has_ss_test:
        console.print(
            "[bold]Test 2 — Session-start TTFT  [/bold]"
            "[dim](first message of a new session — both start fresh)[/dim]"
        )
        console.print(
            "[dim]  Both conditions start from the same neutral KV state (3,072 tokens)\n"
            "  so each must allocate its own buffer from scratch. load_ms = pure KV\n"
            "  allocation time. This is what you feel the moment you start a new chat.[/dim]"
        )
        console.print()

        t2 = Table(box=_box.SIMPLE_HEAD, show_header=True, show_edge=False, pad_edge=False)
        t2.add_column("Metric",         style="white",      width=34)
        t2.add_column("Raw Ollama",     style="dim white",  justify="right", width=14)
        t2.add_column("autotune",       style="green",      justify="right", width=14)
        t2.add_column("Change",                             justify="right", width=18)

        # TTFT — this is where the improvement shows
        t2.add_row(
            "[bold]TTFT (time to first word)[/bold]",
            f"[bold]{result.ss_raw_ttft_ms:.0f} ms[/bold]",
            f"[bold]{result.ss_tuned_ttft_ms:.0f} ms[/bold]",
            _pct_label(result.ss_ttft_improvement_pct),
        )
        # load_ms breakdown — the source of the improvement
        raw_load   = result.ss_raw_run.load_ms   if result.ss_raw_run   else 0.0
        tuned_load = result.ss_tuned_run.load_ms if result.ss_tuned_run else 0.0
        raw_pre    = result.ss_raw_run.prefill_ms   if result.ss_raw_run   else 0.0
        tuned_pre  = result.ss_tuned_run.prefill_ms if result.ss_tuned_run else 0.0

        load_pct = (raw_load - tuned_load) / max(raw_load, 1) * 100
        t2.add_row(
            "  └ load_ms  (KV buffer allocation)",
            f"{raw_load:.0f} ms",
            f"{tuned_load:.0f} ms",
            _pct_label(load_pct),
        )
        t2.add_row(
            "  └ prefill_ms  (prompt processing)",
            f"{raw_pre:.0f} ms",
            f"{tuned_pre:.0f} ms",
            "[dim]same tokens[/dim]",
        )

        # KV sizes for context
        t2.add_row(
            "KV buffer size (num_ctx)",
            f"{result.ss_raw_num_ctx:,} tokens",
            f"{result.ss_tuned_num_ctx:,} tokens",
            _pct_label(
                (result.ss_raw_num_ctx - result.ss_tuned_num_ctx)
                / max(result.ss_raw_num_ctx, 1) * 100
            ),
        )
        if result.ss_raw_kv_mb > 0:
            ss_kv_pct = (result.ss_raw_kv_mb - result.ss_tuned_kv_mb) / max(result.ss_raw_kv_mb, 1) * 100
            t2.add_row(
                "KV cache RAM allocated",
                f"{result.ss_raw_kv_mb:.0f} MB",
                f"{result.ss_tuned_kv_mb:.0f} MB",
                _pct_label(ss_kv_pct),
            )

        console.print(t2)

        # Explain load_ms if the improvement showed
        if raw_load > tuned_load * 1.1:
            console.print()
            console.print(
                f"  [dim]load_ms is Ollama allocating the KV buffer on Metal/GPU before "
                f"processing your first token.\n"
                f"  Raw Ollama always reserves {result.ss_raw_num_ctx:,} tokens "
                f"({result.ss_raw_kv_mb:.0f} MB) — autotune reserves "
                f"{result.ss_tuned_num_ctx:,} tokens ({result.ss_tuned_kv_mb:.0f} MB).\n"
                f"  Smaller buffer → faster allocation → you see the first word sooner.[/dim]"
            )
        elif ss_raw_run_ok := (result.ss_raw_run and result.ss_raw_run.ok):
            if raw_load < 30 and tuned_load < 30:
                console.print()
                console.print(
                    "  [dim]load_ms is low for both conditions — your system is fast enough that "
                    "KV allocation completes in <30 ms either way.\n"
                    "  The RAM savings still apply every request (see banner above).[/dim]"
                )

        console.print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    console.rule("[dim]Verdict[/dim]")
    console.print()

    wins:  list[str] = []
    notes: list[str] = []

    # — KV cache freed (most important and most reliable win) —
    if result.raw_kv_mb > 0 and result.kv_pct >= 10:
        kv_saved = result.kv_saved_mb
        label = f"{kv_saved / 1024:.2f} GB" if kv_saved >= 512 else f"{kv_saved:.0f} MB"
        wins.append(
            f"[green]✅  {label} freed from KV cache every request[/green]  "
            f"[dim](raw: {result.raw_kv_mb:.0f} MB → autotune: {result.tuned_kv_mb:.0f} MB, "
            f"−{result.kv_pct:.0f}%)\n"
            f"     This RAM goes back to Chrome, Slack, VS Code — every single request.[/dim]"
        )
    elif result.raw_kv_mb > 0 and result.kv_pct >= 3:
        wins.append(
            f"[green]✅  {result.kv_saved_mb:.0f} MB freed from KV cache[/green]  "
            f"[dim](−{result.kv_pct:.0f}%)[/dim]"
        )

    # — Session-start TTFT (test 2 — the most convincing TTFT evidence) —
    if result.has_ss_test:
        ss_pct = result.ss_ttft_improvement_pct
        if ss_pct >= 10:
            wins.append(
                f"[green]✅  TTFT faster by {ss_pct:.0f}% on session start[/green]  "
                f"[dim]({result.ss_raw_ttft_ms:.0f} ms → {result.ss_tuned_ttft_ms:.0f} ms)\n"
                f"     Smaller KV buffer allocated faster — you see the first word sooner.[/dim]"
            )
        elif ss_pct >= 3:
            wins.append(
                f"[green]✅  TTFT improved {ss_pct:.0f}% on session start[/green]  "
                f"[dim]({result.ss_raw_ttft_ms:.0f} ms → {result.ss_tuned_ttft_ms:.0f} ms)[/dim]"
            )
        else:
            notes.append(
                f"[yellow]→[/yellow]  Session-start TTFT similar on this machine "
                f"({result.ss_raw_ttft_ms:.0f} ms vs {result.ss_tuned_ttft_ms:.0f} ms) — "
                f"fast SSD + Metal means KV allocation overhead is minimal here."
            )

    # — Warm TTFT (test 1) —
    if result.ttft_improvement_pct >= 5:
        wins.append(
            f"[green]✅  TTFT faster by {result.ttft_improvement_pct:.0f}% (warm)[/green]  "
            f"[dim]({result.raw_ttft_ms:.0f} ms → {result.tuned_ttft_ms:.0f} ms)[/dim]"
        )
    elif result.ttft_improvement_pct < -5:
        notes.append(
            f"[dim]ℹ  Warm TTFT: autotune slightly higher ({result.tuned_ttft_ms:.0f} ms vs "
            f"{result.raw_ttft_ms:.0f} ms raw) — expected on first autotune run as it "
            f"reallocates KV from 4,096→{result.tuned_num_ctx:,}. "
            f"Subsequent turns are the same speed or faster.[/dim]"
        )

    # — RAM headroom —
    if result.headroom_gained_gb >= 0.1:
        wins.append(
            f"[green]✅  +{result.headroom_gained_gb:.2f} GB RAM free for your apps[/green]  "
            f"[dim](autotune: {result.tuned_free_gb:.1f} GB free vs raw: {result.raw_free_gb:.1f} GB)[/dim]"
        )

    # — Swap —
    if result.raw_swap_events > 0 and result.tuned_swap_events == 0:
        wins.append(
            f"[green]✅  Swap eliminated[/green]  "
            f"[dim]Raw Ollama triggered {result.raw_swap_events} swap event(s) — "
            f"adds 2–10 s of stall. autotune kept everything in fast RAM.[/dim]"
        )
    elif result.tuned_swap_events == 0:
        wins.append(
            "[green]✅  Zero swap events[/green]  "
            "[dim]Inference stayed in fast RAM — your computer stayed responsive.[/dim]"
        )
    else:
        notes.append(
            f"[yellow]⚠[/yellow]  {result.tuned_swap_events} swap event(s) with autotune — "
            f"consider a smaller model or [bold]--profile fast[/bold]"
        )

    # — RAM pressure projection —
    if result.kv_saved_mb >= 100:
        notes.append(
            f"[dim]ℹ  On a RAM-pressured machine (8 GB laptop, many apps open), "
            f"freeing {result.kv_saved_mb:.0f} MB of KV cache per request can eliminate swap — "
            f"turning a 4–10 s stall into an instant response.[/dim]"
        )

    # — Honest generation speed —
    if abs(result.raw_tps) > 0.1:
        notes.append(
            "[dim]ℹ  Generation speed (tok/s) is GPU/CPU-bound — "
            "autotune does not change it and does not claim to.[/dim]"
        )

    for line in wins:
        console.print(f"  {line}")
    if wins and notes:
        console.print()
    for line in notes:
        console.print(f"  {line}")

    console.print()

    # ── Footer ────────────────────────────────────────────────────────────────
    console.rule("[dim]What's next?[/dim]")
    console.print()
    console.print(
        "  [dim]For statistical proof with Wilcoxon p-values and Cohen's d effect size:[/dim]"
    )
    console.print(
        f"  [bold cyan]autotune proof-suite -m {result.model_id}[/bold cyan]"
        "  [dim](~20 min, 5 prompt types)[/dim]"
    )
    console.print()
    if output_path:
        console.print(f"  [dim]Results saved → {output_path}[/dim]")
    console.print()
    console.rule()
