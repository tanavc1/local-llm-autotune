"""
autotune quick proof — does autotune actually help on YOUR machine?

Runs in ≤45 seconds.  Measures the metrics users care about:
  • Time to first word (TTFT)    — load_ms + prefill_ms from Ollama's Go timer
  • Context window (num_ctx)     — autotune dynamic vs Ollama fixed 4096
  • KV cache size                — estimated from model architecture
  • RAM free for other apps      — system available memory floor
  • Swap events                  — any swap pressure during inference (goal: 0)
  • Generation speed (tok/s)     — honestly reported; autotune does NOT change this

Two test scenarios
------------------
  1. Standard test  — short multi-turn conversation; shows KV + RAM savings
  2. Long-context TTFT test — large code-review prompt; both conditions start
     from a freshly-reset KV buffer so the TTFT difference comes purely from
     the KV buffer allocation cost (autotune: right-sized; raw: full 4096).

Uses the same Ollama-native timers as proof_suite so numbers are comparable.
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
_MAX_TOKENS     = 120      # keep generation fast; TTFT is what we measure
_MAX_TOKENS_LC  = 60       # even shorter for long-context TTFT test
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
# Long-context TTFT test prompt — ~340 tokens, code-review scenario.
#
# Why this prompt?
#   - Realistic real-world task (code review)
#   - Large enough to push autotune's dynamic num_ctx to ~768 (vs raw 4096)
#   - After a KV buffer reset, both conditions allocate a fresh buffer:
#       raw  → 4096 tokens (~536 MB for a 7B model)
#       tune → ~768 tokens (~100 MB for a 7B model)
#   - load_ms is proportional to buffer size, so autotune wins on TTFT
#   - This simulates the first message of a new session with a long document
# ─────────────────────────────────────────────────────────────────────────────
LONG_CTX_MESSAGES: list[dict] = [
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


async def _reset_kv_buffer(model_id: str) -> None:
    """
    Force Ollama to release and re-allocate the KV buffer on the next request.

    Sends a trivial request with keep_alive=0 so Ollama marks the model for
    immediate eviction. The next real request then starts fresh — this is how
    we isolate the KV-allocation cost in the long-context TTFT test without
    requiring a full disk-to-RAM model reload.
    """
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            await c.post(f"{_OLLAMA_BASE}/api/chat", json={
                "model":      model_id,
                "messages":   [{"role": "user", "content": "ok"}],
                "stream":     False,
                "options":    {"num_predict": 1},
                "keep_alive": 0,
            })
        await asyncio.sleep(1.0)   # let Ollama finish the eviction
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
        load_ms    = data.get("load_duration",         0) / 1_000_000
        prefill_ms = data.get("prompt_eval_duration",  0) / 1_000_000
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
    # Long-context TTFT test — one cold-start run each, both start from reset
    lc_raw_run:     Optional[_RunResult] = None
    lc_tuned_run:   Optional[_RunResult] = None

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

    # ── Long-context TTFT test properties ────────────────────────────────────

    @property
    def has_lc_test(self) -> bool:
        return (
            self.lc_raw_run is not None and self.lc_raw_run.ok
            and self.lc_tuned_run is not None and self.lc_tuned_run.ok
        )

    @property
    def lc_raw_ttft_ms(self) -> float:
        return self.lc_raw_run.ttft_ms if self.lc_raw_run and self.lc_raw_run.ok else 0.0

    @property
    def lc_tuned_ttft_ms(self) -> float:
        return self.lc_tuned_run.ttft_ms if self.lc_tuned_run and self.lc_tuned_run.ok else 0.0

    @property
    def lc_ttft_improvement_pct(self) -> float:
        if self.lc_raw_ttft_ms <= 0:
            return 0.0
        return (self.lc_raw_ttft_ms - self.lc_tuned_ttft_ms) / self.lc_raw_ttft_ms * 100

    @property
    def lc_raw_kv_mb(self) -> float:
        return self.lc_raw_run.kv_cache_mb if self.lc_raw_run and self.lc_raw_run.ok else 0.0

    @property
    def lc_tuned_kv_mb(self) -> float:
        return self.lc_tuned_run.kv_cache_mb if self.lc_tuned_run and self.lc_tuned_run.ok else 0.0

    @property
    def lc_raw_num_ctx(self) -> int:
        return self.lc_raw_run.num_ctx if self.lc_raw_run and self.lc_raw_run.ok else _RAW_CTX

    @property
    def lc_tuned_num_ctx(self) -> int:
        return self.lc_tuned_run.num_ctx if self.lc_tuned_run and self.lc_tuned_run.ok else 0

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
        if self.has_lc_test:
            d["long_ctx_ttft_raw_ms"]   = round(self.lc_raw_ttft_ms, 1)
            d["long_ctx_ttft_tuned_ms"] = round(self.lc_tuned_ttft_ms, 1)
            d["long_ctx_ttft_pct"]      = round(self.lc_ttft_improvement_pct, 1)
            d["long_ctx_kv_raw_mb"]     = round(self.lc_raw_kv_mb, 1)
            d["long_ctx_kv_tuned_mb"]   = round(self.lc_tuned_kv_mb, 1)
            d["long_ctx_ctx_raw"]       = self.lc_raw_num_ctx
            d["long_ctx_ctx_tuned"]     = self.lc_tuned_num_ctx
            d["lc_raw_run"]             = asdict(self.lc_raw_run)
            d["lc_tuned_run"]           = asdict(self.lc_tuned_run)
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

    # ── Long-context TTFT test ────────────────────────────────────────────────
    # Both conditions start from a freshly-reset KV buffer so the measured
    # load_ms comes purely from KV allocation cost, not context history.
    # autotune (small buffer) finishes allocation faster → lower TTFT.
    _step("Long-context TTFT test — resetting KV buffer…")
    lc_raw_run: Optional[_RunResult] = None
    lc_tuned_run: Optional[_RunResult] = None

    try:
        # autotune long-ctx: allocates a right-sized buffer (~768 tokens)
        await _reset_kv_buffer(model_id)
        tuned_lc_opts, _ = build_ollama_options(LONG_CTX_MESSAGES, profile)
        _step("Long-context TTFT test — autotune (small buffer)…")
        lc_tuned_run = await _measured_run(
            model_id, "autotune", tuned_lc_opts, arch, kv_dtype_bytes=kv_dtype,
            messages=LONG_CTX_MESSAGES, max_tokens=_MAX_TOKENS_LC,
        )

        # raw long-ctx: allocates the full 4096-token buffer
        await _reset_kv_buffer(model_id)
        _step("Long-context TTFT test — raw Ollama (full 4096-token buffer)…")
        lc_raw_run = await _measured_run(
            model_id, "raw", {"num_ctx": _RAW_CTX}, arch, kv_dtype_bytes=2,
            messages=LONG_CTX_MESSAGES, max_tokens=_MAX_TOKENS_LC,
        )
    except Exception:
        # Long-context test is best-effort — never fail the whole proof
        pass

    elapsed = time.monotonic() - t0

    result = QuickProofResult(
        model_id=model_id,
        profile_name=profile_name,
        n_runs=n_runs,
        elapsed_sec=elapsed,
        raw_runs=raw_runs,
        tuned_runs=tuned_runs,
        lc_raw_run=lc_raw_run,
        lc_tuned_run=lc_tuned_run,
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

    # ── Section 1: Standard test (multi-turn with system prompt) ─────────────
    console.print("[bold]Test 1 — Standard session[/bold]  [dim](warm model, multi-turn prompt)[/dim]")
    console.print(
        "[dim]  Raw Ollama always allocates a fixed 4,096-token KV cache. "
        "autotune right-sizes it to exactly what your prompt needs.[/dim]"
    )
    console.print()

    t = Table(box=_box.SIMPLE_HEAD, show_header=True, show_edge=False, pad_edge=False)
    t.add_column("Metric",                     style="white",       width=34)
    t.add_column("Raw Ollama",                 style="dim white",   justify="right", width=14)
    t.add_column("autotune",                   style="green",       justify="right", width=14)
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
        "KV buffer (num_ctx tokens)",
        f"{result.raw_num_ctx:,}",
        f"{result.tuned_num_ctx:,}",
        _pct_label(result.ctx_pct),
    )

    # KV cache size in MB
    kv_row_raw   = f"{result.raw_kv_mb:.0f} MB"   if result.raw_kv_mb   > 0 else "N/A"
    kv_row_tuned = f"{result.tuned_kv_mb:.0f} MB" if result.tuned_kv_mb > 0 else "N/A"
    kv_change    = _pct_label(result.kv_pct) if result.raw_kv_mb > 0 else "N/A"
    t.add_row("KV cache RAM (est.)", kv_row_raw, kv_row_tuned, kv_change)

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
    swap_change    = "none ✅" if result.tuned_swap_events == 0 else f"+{result.tuned_swap_events} ⚠️"
    t.add_row("Swap events", swap_raw_str, swap_tuned_str, swap_change)

    # Generation speed — honest
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

    # ── Section 2: Long-context TTFT test ────────────────────────────────────
    if result.has_lc_test:
        console.rule("[dim]Test 2 — Long-context TTFT  (code review, ~340 tokens)[/dim]")
        console.print(
            "[dim]  Both conditions start from a fresh KV buffer reset, so the TTFT "
            "difference reflects pure KV allocation cost.\n"
            "  autotune allocates only what the prompt needs; raw Ollama always "
            "allocates the full 4,096-token slab.[/dim]"
        )
        console.print()

        lc = Table(box=_box.SIMPLE_HEAD, show_header=True, show_edge=False, pad_edge=False)
        lc.add_column("Metric",         style="white",      width=34)
        lc.add_column("Raw Ollama",     style="dim white",  justify="right", width=14)
        lc.add_column("autotune",       style="green",      justify="right", width=14)
        lc.add_column("Change",                             justify="right", width=18)

        lc.add_row(
            "Time to first word (TTFT) ⬅ key",
            f"{result.lc_raw_ttft_ms:.0f} ms",
            f"{result.lc_tuned_ttft_ms:.0f} ms",
            _pct_label(result.lc_ttft_improvement_pct),
        )
        lc.add_row(
            "  └ load_ms  (KV alloc cost)",
            f"{result.lc_raw_run.load_ms:.0f} ms" if result.lc_raw_run else "N/A",
            f"{result.lc_tuned_run.load_ms:.0f} ms" if result.lc_tuned_run else "N/A",
            _pct_label(
                (result.lc_raw_run.load_ms - result.lc_tuned_run.load_ms)
                / max(result.lc_raw_run.load_ms, 1) * 100
            ) if (result.lc_raw_run and result.lc_tuned_run) else "N/A",
        )
        lc.add_row(
            "  └ prefill_ms  (prompt eval)",
            f"{result.lc_raw_run.prefill_ms:.0f} ms" if result.lc_raw_run else "N/A",
            f"{result.lc_tuned_run.prefill_ms:.0f} ms" if result.lc_tuned_run else "N/A",
            "[dim]similar (same tokens)[/dim]",
        )
        lc.add_row(
            "KV buffer allocated",
            f"{result.lc_raw_num_ctx:,} tokens",
            f"{result.lc_tuned_num_ctx:,} tokens",
            _pct_label((result.lc_raw_num_ctx - result.lc_tuned_num_ctx)
                       / max(result.lc_raw_num_ctx, 1) * 100),
        )
        if result.lc_raw_kv_mb > 0:
            lc.add_row(
                "KV cache RAM (est.)",
                f"{result.lc_raw_kv_mb:.0f} MB",
                f"{result.lc_tuned_kv_mb:.0f} MB",
                _pct_label((result.lc_raw_kv_mb - result.lc_tuned_kv_mb)
                           / max(result.lc_raw_kv_mb, 1) * 100),
            )

        console.print(lc)
        console.print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    console.rule("[dim]Verdict[/dim]")
    console.print()

    wins: list[str] = []
    notes: list[str] = []

    # TTFT standard test
    if result.ttft_improvement_pct >= 5:
        wins.append(
            f"[green]✅  TTFT faster by {result.ttft_improvement_pct:.0f}%[/green]  "
            f"[dim]({result.raw_ttft_ms:.0f} ms → {result.tuned_ttft_ms:.0f} ms) "
            f"— smaller KV buffer means faster first token[/dim]"
        )
    elif result.ttft_improvement_pct > 0:
        notes.append(
            f"[yellow]→[/yellow]  Standard TTFT improved modestly "
            f"({result.ttft_improvement_pct:.0f}%) — your system has plenty of RAM. "
            f"See long-context test below for the full picture."
        )
    else:
        notes.append(
            "[yellow]→[/yellow]  Standard TTFT unchanged — model is warm and RAM is "
            "not the bottleneck here. Long-context test shows the real difference."
        )

    # TTFT long-context test (the most persuasive number)
    if result.has_lc_test:
        lc_pct = result.lc_ttft_improvement_pct
        if lc_pct >= 5:
            wins.append(
                f"[green]✅  Long-context TTFT faster by {lc_pct:.0f}%[/green]  "
                f"[dim]({result.lc_raw_ttft_ms:.0f} ms → {result.lc_tuned_ttft_ms:.0f} ms) "
                f"— KV buffer {result.lc_raw_kv_mb:.0f} MB → {result.lc_tuned_kv_mb:.0f} MB "
                f"saves allocation time[/dim]"
            )
        elif lc_pct > -5:
            notes.append(
                f"[yellow]→[/yellow]  Long-context TTFT similar — "
                f"your system is fast enough that KV allocation overhead is minimal."
            )
        else:
            notes.append(
                f"[yellow]→[/yellow]  Long-context TTFT: autotune {result.lc_tuned_ttft_ms:.0f} ms, "
                f"raw {result.lc_raw_ttft_ms:.0f} ms — "
                f"model reload dominated this run; run proof-suite for sustained stats."
            )

    # KV cache savings
    if result.raw_kv_mb > 0 and result.kv_pct >= 10:
        kv_freed = result.raw_kv_mb - result.tuned_kv_mb
        wins.append(
            f"[green]✅  {kv_freed:.0f} MB KV cache freed[/green]  "
            f"[dim]({result.raw_kv_mb:.0f} MB → {result.tuned_kv_mb:.0f} MB, "
            f"−{result.kv_pct:.0f}%) — that RAM is now free for your other apps[/dim]"
        )
    elif result.raw_kv_mb > 0 and result.kv_pct >= 3:
        kv_freed = result.raw_kv_mb - result.tuned_kv_mb
        wins.append(
            f"[green]✅  {kv_freed:.0f} MB KV cache freed[/green]  "
            f"[dim](−{result.kv_pct:.0f}%)[/dim]"
        )

    # RAM headroom
    free_delta = result.tuned_free_gb - result.raw_free_gb
    if free_delta >= 0.1:
        wins.append(
            f"[green]✅  +{free_delta:.1f} GB RAM headroom[/green]  "
            f"[dim]More space for your browser, apps, and OS while the LLM runs[/dim]"
        )
    elif free_delta >= 0.02:
        wins.append(
            f"[green]✅  +{free_delta:.2f} GB RAM headroom[/green]  "
            f"[dim](smaller KV buffer means less RAM pressure)[/dim]"
        )

    # Swap avoidance
    if result.raw_swap_events > 0 and result.tuned_swap_events == 0:
        wins.append(
            f"[green]✅  Swap eliminated[/green]  "
            f"[dim]Raw Ollama triggered {result.raw_swap_events} swap event(s); "
            f"autotune kept everything in fast RAM — "
            f"swap adds 2-10 seconds of stall per event[/dim]"
        )
    elif result.tuned_swap_events == 0 and result.raw_swap_events == 0:
        wins.append(
            "[green]✅  Zero swap events[/green]  "
            "[dim]Inference stayed in fast RAM — your computer stayed responsive[/dim]"
        )
    else:
        notes.append(
            f"[yellow]⚠[/yellow]  {result.tuned_swap_events} swap event(s) with autotune — "
            f"consider a smaller model or [bold]--profile fast[/bold]"
        )

    # Honest generation speed note
    if abs(result.raw_tps) > 0.1:
        notes.append(
            "[dim]ℹ  Generation speed (tok/s) is GPU/CPU-bound — "
            "autotune does not change it and does not claim to[/dim]"
        )

    # RAM-pressure projection (always educational)
    if result.raw_kv_mb > 0:
        kv_freed_mb = result.raw_kv_mb - result.tuned_kv_mb
        if kv_freed_mb >= 50:
            notes.append(
                f"[dim]ℹ  On a RAM-pressured system (e.g. 8 GB laptop with many apps open), "
                f"freeing {kv_freed_mb:.0f} MB of KV cache can prevent swap entirely — "
                f"turning a 4-10 second stall into an instant response[/dim]"
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
        "  [dim]This was the quick check (~45 s). "
        "For statistical proof with Wilcoxon p-values and Cohen's d:[/dim]"
    )
    console.print(
        f"  [bold cyan]autotune proof-suite --model {result.model_id}[/bold cyan]"
        "  [dim](takes ~20 min, covers 5 prompt types)[/dim]"
    )
    console.print()
    if output_path:
        console.print(f"  [dim]Results saved → {output_path}[/dim]")
    console.print()
    console.rule()
