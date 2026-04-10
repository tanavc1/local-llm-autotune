"""
Benchmark runner for autotune.

Runs an inference call while sampling system metrics every 250 ms in a
background thread.  Records everything to the run_observations DB table with
an experiment tag so runs can be compared.

Metrics captured:
  - TTFT (time-to-first-token, ms)
  - Throughput (tokens/s, approximated via char/4)
  - Total elapsed wall time (s)
  - RAM used before / peak / after (GB)
  - Swap used before / peak / after (GB)
  - CPU % average during inference
  - Token count (input estimate + output)
  - Delta RAM / Delta Swap (memory pressure leakage)
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil

from autotune.api.backends.chain import get_chain
from autotune.api.ctx_utils import estimate_tokens
from autotune.api.hardware_tuner import get_tuner
from autotune.api.kv_manager import build_ollama_options
from autotune.api.profiles import get_profile
from autotune.db.fingerprint import hardware_to_db_dict
from autotune.db.store import get_db
from autotune.hardware.profiler import profile_hardware
from autotune.ttft import TTFTOptimizer


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    tag: str
    model_id: str
    profile_name: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tokens_per_sec: float
    elapsed_sec: float

    # Memory snapshots
    ram_before_gb: float
    ram_peak_gb: float
    ram_after_gb: float
    swap_before_gb: float
    swap_peak_gb: float
    swap_after_gb: float

    # CPU
    cpu_avg_pct: float
    cpu_peak_pct: float

    # Derived
    delta_ram_gb: float = field(init=False)
    delta_swap_gb: float = field(init=False)

    # Content
    response_text: str = ""
    error: Optional[str] = None
    num_ctx_used: Optional[int] = None
    num_keep_used: Optional[int] = None   # tokens pinned in KV (prefix cache)
    f16_kv_used: Optional[bool] = None   # KV precision: True=F16, False=Q8

    def __post_init__(self) -> None:
        self.delta_ram_gb = round(self.ram_after_gb - self.ram_before_gb, 3)
        self.delta_swap_gb = round(self.swap_after_gb - self.swap_before_gb, 3)

    def to_db_dict(self, hardware_id: str) -> dict:
        d: dict = {
            "model_id":           self.model_id,
            "hardware_id":        hardware_id,
            "quant":              "unknown",
            "context_len":        self.num_ctx_used or 0,
            "n_gpu_layers":       -1,
            "profile_name":       self.profile_name,
            "bench_tag":          self.tag,
            # structured telemetry columns
            "tokens_per_sec":     round(self.tokens_per_sec, 1),
            "gen_tokens_per_sec": round(self.tokens_per_sec, 1),
            "ttft_ms":            round(self.ttft_ms, 1),
            "elapsed_sec":        round(self.elapsed_sec, 2),
            "peak_ram_gb":        round(self.ram_peak_gb, 3),
            "peak_vram_gb":       round(self.ram_peak_gb, 3),   # unified memory
            "ram_before_gb":      round(self.ram_before_gb, 3),
            "ram_after_gb":       round(self.ram_after_gb, 3),
            "delta_ram_gb":       round(self.delta_ram_gb, 3),
            "swap_before_gb":     round(self.swap_before_gb, 3),
            "swap_peak_gb":       round(self.swap_peak_gb, 3),
            "swap_after_gb":      round(self.swap_after_gb, 3),
            "delta_swap_gb":      round(self.delta_swap_gb, 3),
            "cpu_avg_pct":        round(self.cpu_avg_pct, 1),
            "cpu_peak_pct":       round(self.cpu_peak_pct, 1),
            "prompt_tokens":      self.prompt_tokens,
            "completion_tokens":  self.completion_tokens,
            "completed":          0 if self.error else 1,
            "oom":                0,
            "error_msg":          self.error,
        }
        # Optional KV metadata
        if self.num_keep_used is not None:
            d["num_keep"] = self.num_keep_used
        if self.f16_kv_used is not None:
            d["f16_kv"] = int(self.f16_kv_used)
        # Keep legacy notes field for backwards compatibility with compare_runs()
        d["notes"] = (
            f"bench_tag={self.tag} "
            f"profile={self.profile_name} "
            f"elapsed={self.elapsed_sec:.1f}s "
            f"delta_ram={self.delta_ram_gb:+.3f}GB "
            f"delta_swap={self.delta_swap_gb:+.3f}GB "
            f"cpu_avg={self.cpu_avg_pct:.1f}%"
        )
        return d


# ---------------------------------------------------------------------------
# Background sampler
# ---------------------------------------------------------------------------

class _SystemSampler:
    """Samples psutil every interval_sec in a daemon thread."""

    def __init__(self, interval_sec: float = 0.25) -> None:
        self.interval = interval_sec
        self._ram_samples: list[float] = []
        self._swap_samples: list[float] = []
        self._cpu_samples: list[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            self._ram_samples.append(vm.used / 1024**3)
            self._swap_samples.append(sw.used / 1024**3)
            self._cpu_samples.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)

    def peak_ram_gb(self) -> float:
        return max(self._ram_samples) if self._ram_samples else 0.0

    def peak_swap_gb(self) -> float:
        return max(self._swap_samples) if self._swap_samples else 0.0

    def avg_cpu_pct(self) -> float:
        return statistics.mean(self._cpu_samples) if self._cpu_samples else 0.0

    def peak_cpu_pct(self) -> float:
        return max(self._cpu_samples) if self._cpu_samples else 0.0


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

async def run_bench(
    model_id: str,
    messages: list[dict],
    profile_name: str = "balanced",
    tag: str = "bench",
    apply_hw_tuning: bool = True,
) -> BenchResult:
    """
    Run one benchmark inference and return a BenchResult.

    Does NOT write to DB — caller decides whether to persist.
    """
    profile = get_profile(profile_name)
    chain = get_chain()
    tuner = get_tuner()

    # Snapshot system state before anything
    vm_before = psutil.virtual_memory()
    sw_before = psutil.swap_memory()
    ram_before = vm_before.used / 1024**3
    swap_before = sw_before.used / 1024**3

    # Estimate input tokens
    prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

    # Start background sampler (samples before apply so we get accurate before)
    # Prime psutil CPU measurement — first call after process start returns 0.0
    psutil.cpu_percent(interval=None)

    # Compute dynamic KV window + prefix-cache + pressure-adjusted num_ctx
    ollama_opts, _ = build_ollama_options(messages, profile)

    sampler = _SystemSampler(interval_sec=0.25)
    sampler.start()

    if apply_hw_tuning:
        tuner._apply(profile_name)

    t_start = time.perf_counter()
    first_token_t: Optional[float] = None
    collected: list[str] = []
    backend_used = "?"
    error_msg: Optional[str] = None

    try:
        async for chunk in chain.stream(
            model_id,
            messages,
            max_new_tokens=profile.max_new_tokens,
            temperature=profile.temperature,
            top_p=profile.top_p,
            repetition_penalty=profile.repetition_penalty,
            timeout=profile.request_timeout_sec,
            num_ctx=ollama_opts["num_ctx"],
            ollama_options=ollama_opts,
        ):
            backend_used = chunk.backend
            if chunk.content:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                collected.append(chunk.content)
    except Exception as e:
        error_msg = str(e)
    finally:
        if apply_hw_tuning:
            tuner._restore()
        sampler.stop()

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    ttft_ms = (first_token_t - t_start) * 1000 if first_token_t else 0.0

    content = "".join(collected)
    comp_tokens = estimate_tokens(content)
    tps = comp_tokens / max(elapsed, 0.01)

    # Snapshot after
    vm_after = psutil.virtual_memory()
    sw_after = psutil.swap_memory()
    ram_after = vm_after.used / 1024**3
    swap_after = sw_after.used / 1024**3

    return BenchResult(
        tag=tag,
        model_id=model_id,
        profile_name=profile_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=comp_tokens,
        ttft_ms=round(ttft_ms, 1),
        tokens_per_sec=round(tps, 2),
        elapsed_sec=round(elapsed, 2),
        ram_before_gb=round(ram_before, 3),
        ram_peak_gb=round(sampler.peak_ram_gb(), 3),
        ram_after_gb=round(ram_after, 3),
        swap_before_gb=round(swap_before, 3),
        swap_peak_gb=round(sampler.peak_swap_gb(), 3),
        swap_after_gb=round(swap_after, 3),
        cpu_avg_pct=round(sampler.avg_cpu_pct(), 1),
        cpu_peak_pct=round(sampler.peak_cpu_pct(), 1),
        response_text=content,
        error=error_msg,
        num_ctx_used=ollama_opts["num_ctx"],
        num_keep_used=ollama_opts.get("num_keep"),
        f16_kv_used=ollama_opts.get("f16_kv", True),
    )


async def run_raw_ollama(
    model_id: str,
    messages: list[dict],
    tag: str = "raw_ollama",
) -> BenchResult:
    """
    Hit Ollama with ZERO autotune settings — exactly what a user would get
    if they called Ollama directly without any wrapper.

    Defaults Ollama uses when nothing is specified:
      num_ctx      = 4096  (its own default, NOT model's max context)
      temperature  = 0.8
      num_predict  = -1    (generate until stop token — we cap at 2048 for safety)
      keep_alive   = 5m    (model unloads after 5 min idle)
      No GC tuning, no QOS priority, no nice level
    """
    import httpx

    vm_before = psutil.virtual_memory()
    sw_before = psutil.swap_memory()
    ram_before = vm_before.used / 1024**3
    swap_before = sw_before.used / 1024**3
    prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

    sampler = _SystemSampler(interval_sec=0.25)
    sampler.start()

    t_start = time.perf_counter()
    first_token_t: Optional[float] = None
    collected: list[str] = []
    error_msg: Optional[str] = None

    try:
        async with httpx.AsyncClient(timeout=360.0) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": messages,
                    "stream": True,
                    # Explicitly no num_ctx, no temperature — pure Ollama defaults
                },
                headers={"Content-Type": "application/json"},
            ) as resp:
                import json as _json
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue   # SSE uses blank lines as separators — skip
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue
                    try:
                        chunk = _json.loads(line[6:])
                    except Exception:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        if first_token_t is None:
                            first_token_t = time.perf_counter()
                        collected.append(content)
    except Exception as e:
        error_msg = str(e)
    finally:
        sampler.stop()

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    ttft_ms = (first_token_t - t_start) * 1000 if first_token_t else 0.0
    content = "".join(collected)
    comp_tokens = estimate_tokens(content)
    tps = comp_tokens / max(elapsed, 0.01)

    vm_after = psutil.virtual_memory()
    sw_after = psutil.swap_memory()

    return BenchResult(
        tag=tag,
        model_id=model_id,
        profile_name="raw_ollama_defaults",
        prompt_tokens=prompt_tokens,
        completion_tokens=comp_tokens,
        ttft_ms=round(ttft_ms, 1),
        tokens_per_sec=round(tps, 2),
        elapsed_sec=round(elapsed, 2),
        ram_before_gb=round(ram_before, 3),
        ram_peak_gb=round(sampler.peak_ram_gb(), 3),
        ram_after_gb=round(vm_after.used / 1024**3, 3),
        swap_before_gb=round(swap_before, 3),
        swap_peak_gb=round(sampler.peak_swap_gb(), 3),
        swap_after_gb=round(sw_after.used / 1024**3, 3),
        cpu_avg_pct=round(sampler.avg_cpu_pct(), 1),
        cpu_peak_pct=round(sampler.peak_cpu_pct(), 1),
        response_text=content,
        error=error_msg,
        num_ctx_used=4096,   # Ollama's actual default (confirmed via /api/ps)
    )


async def run_bench_ollama_only(
    model_id: str,
    messages: list[dict],
    profile_name: str = "balanced",
    tag: str = "bench_ollama",
    apply_hw_tuning: bool = True,
) -> BenchResult:
    """
    Benchmark autotune's optimised settings against Ollama directly.

    Routes DIRECTLY to the Ollama HTTP API (no MLX / LM Studio fallback) and
    applies all three TTFT mechanisms via :class:`autotune.ttft.TTFTOptimizer`:

      1. Dynamic ``num_ctx``  — sized to this request, not the profile max
      2. ``keep_alive=-1``    — model never unloads between calls
      3. ``num_keep``         — system-prompt tokens pinned in KV cache

    OS-level tuning (QOS class, GC disable) is applied separately via
    :class:`autotune.api.hardware_tuner.HardwareTuner` when ``apply_hw_tuning``
    is True.

    Compare against :func:`run_raw_ollama` to isolate the autotune delta.
    """
    import httpx
    import json as _json

    profile = get_profile(profile_name)
    tuner   = get_tuner()
    _ttft   = TTFTOptimizer()

    vm_before = psutil.virtual_memory()
    sw_before = psutil.swap_memory()
    ram_before  = vm_before.used / 1024**3
    swap_before = sw_before.used / 1024**3
    prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

    psutil.cpu_percent(interval=None)

    # ── All three TTFT mechanisms in one call ────────────────────────────────
    ttft_result = _ttft.build_request_options(messages, profile)
    ollama_options: dict = {**ttft_result["options"]}
    keep_alive: str      = ttft_result["keep_alive"]
    # ────────────────────────────────────────────────────────────────────────

    # Layer in remaining inference parameters (not TTFT-related)
    if profile.repetition_penalty != 1.0:
        ollama_options["repeat_penalty"] = profile.repetition_penalty

    sampler = _SystemSampler(interval_sec=0.25)
    sampler.start()

    if apply_hw_tuning:
        tuner._apply(profile_name)

    t_start = time.perf_counter()
    first_token_t: Optional[float] = None
    collected: list[str] = []
    error_msg: Optional[str] = None

    try:
        async with httpx.AsyncClient(timeout=360.0) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/v1/chat/completions",
                json={
                    "model":       model_id,
                    "messages":    messages,
                    "stream":      True,
                    "max_tokens":  profile.max_new_tokens,
                    "temperature": profile.temperature,
                    "top_p":       profile.top_p,
                    "options":     ollama_options,
                    "keep_alive":  keep_alive,
                },
                headers={"Content-Type": "application/json"},
            ) as resp:
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if not line.startswith("data: "):
                        continue
                    try:
                        chunk = _json.loads(line[6:])
                    except Exception:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        if first_token_t is None:
                            first_token_t = time.perf_counter()
                        collected.append(content)
    except Exception as e:
        error_msg = str(e)
    finally:
        if apply_hw_tuning:
            tuner._restore()
        sampler.stop()

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    ttft_ms = (first_token_t - t_start) * 1000 if first_token_t else 0.0

    content = "".join(collected)
    comp_tokens = estimate_tokens(content)
    tps = comp_tokens / max(elapsed, 0.01)

    vm_after = psutil.virtual_memory()
    sw_after = psutil.swap_memory()

    return BenchResult(
        tag=tag,
        model_id=model_id,
        profile_name=profile_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=comp_tokens,
        ttft_ms=round(ttft_ms, 1),
        tokens_per_sec=round(tps, 2),
        elapsed_sec=round(elapsed, 2),
        ram_before_gb=round(ram_before, 3),
        ram_peak_gb=round(sampler.peak_ram_gb(), 3),
        ram_after_gb=round(vm_after.used / 1024**3, 3),
        swap_before_gb=round(swap_before, 3),
        swap_peak_gb=round(sampler.peak_swap_gb(), 3),
        swap_after_gb=round(sw_after.used / 1024**3, 3),
        cpu_avg_pct=round(sampler.avg_cpu_pct(), 1),
        cpu_peak_pct=round(sampler.peak_cpu_pct(), 1),
        response_text=content,
        error=error_msg,
        num_ctx_used=ollama_options["num_ctx"],
        num_keep_used=ollama_options.get("num_keep"),
        f16_kv_used=ollama_options.get("f16_kv", True),
    )


def save_result(result: BenchResult) -> int:
    """Persist a BenchResult to the run_observations DB table. Returns row ID."""
    import time
    hw = profile_hardware()
    hw_dict = hardware_to_db_dict(hw)
    db = get_db()
    db.upsert_hardware(hw_dict)

    # Auto-register the model if it's not already in the models table.
    # Ollama model IDs (e.g. "qwen3:8b") won't be there — insert a
    # minimal stub so the FK constraint is satisfied.
    if not db.get_model(result.model_id):
        name = result.model_id.split("/")[-1].split(":")[0]
        db.upsert_model({
            "id": result.model_id,
            "name": name,
            "organization": "ollama",
            "family": name,
            "fetched_at": time.time(),
        })

    row_id = db.log_run(result.to_db_dict(hw_dict["id"]))

    # Emit structured telemetry events for notable conditions so they can be
    # queried independently of the run record.
    hw_id = hw_dict["id"]
    if result.swap_peak_gb > 2.0:
        db.log_telemetry_event(
            "swap_spike",
            value_num=result.swap_peak_gb,
            value_text=f"peak_swap={result.swap_peak_gb:.2f}GB during {result.tag}",
            run_id=row_id, hardware_id=hw_id, model_id=result.model_id,
        )
    if result.delta_ram_gb > 1.5:
        db.log_telemetry_event(
            "ram_spike",
            value_num=result.delta_ram_gb,
            value_text=f"delta_ram={result.delta_ram_gb:+.2f}GB during {result.tag}",
            run_id=row_id, hardware_id=hw_id, model_id=result.model_id,
        )
    if result.error:
        db.log_telemetry_event(
            "error",
            value_text=result.error,
            run_id=row_id, hardware_id=hw_id, model_id=result.model_id,
        )
    if result.ttft_ms > 5000:
        db.log_telemetry_event(
            "slow_token",
            value_num=result.ttft_ms,
            value_text=f"ttft={result.ttft_ms:.0f}ms (profile={result.profile_name})",
            run_id=row_id, hardware_id=hw_id, model_id=result.model_id,
        )

    return row_id
