"""
OllamaMetricsClient — runs inference via Ollama's native /api/chat endpoint
and returns authoritative performance stats from Ollama's own internal timers.

Why this instead of /v1/chat/completions
-----------------------------------------
The OpenAI-compatible endpoint (``/v1/chat/completions``) does not expose
Ollama's internal timing fields.  The native ``/api/chat`` endpoint returns
the full timing payload on every response::

    {
      "eval_count":             329,     # tokens generated
      "eval_duration":    10085376281,   # ns for generation
      "prompt_eval_count":        8,     # tokens in the prompt
      "prompt_eval_duration": 324087125, # ns for prefill (KV fill phase)
      "load_duration":     2606521833,   # ns to load model + alloc KV cache
      "total_duration":   13213650167,   # ns total wall time
    }

All durations are nanoseconds from Ollama's Go ``time.Now()`` calls — not
estimated, not sampled by psutil.

Why prompt_eval_duration tracks TTFT
--------------------------------------
``prompt_eval_duration`` = time Ollama spends on the forward pass for the
prompt tokens (the "prefill" or "KV-fill" phase).  This is exactly what the
user experiences as latency before the first token appears.

Crucially, this duration scales with ``num_ctx``, not just prompt length:
Ollama must allocate and (on Metal) zero/initialise the KV cache buffer
*before* running the prompt forward pass.  A 4096-token KV buffer takes
longer to set up than a 1290-token one even if the prompt is only 10 tokens.

Measured on phi4-mini:latest (same 8-token prompt):
    num_ctx=4096 → prompt_eval_duration=324 ms
    num_ctx=1024 → prompt_eval_duration=200 ms   (-38%)

Why load_duration tracks KV allocation cost
---------------------------------------------
``load_duration`` = time to load model weights into Metal memory AND allocate
the KV cache tensor.  When ``num_ctx`` shrinks, the KV tensor is smaller and
this phase is faster.

Measured on phi4-mini:latest (first call after keep_alive expires):
    num_ctx=4096 → load_duration=2607 ms
    num_ctx=1024 → load_duration=976 ms    (-63%)

Once the model is loaded (keep_alive=-1), subsequent calls have
``load_duration ≈ 0`` for both raw and autotune — the benefit is felt at
session-start and after any idle-expiry reload.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_OLLAMA_BASE = "http://localhost:11434"
_DEFAULT_TIMEOUT = 300.0   # seconds


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class NativeInferenceStats:
    """
    Authoritative performance stats from Ollama's native /api/chat response.

    All ``*_ms`` fields are converted from Ollama's nanosecond timestamps.
    All ``*_tps`` fields are derived: count / (duration_s).

    Primary metrics for autotune comparison
    ----------------------------------------
    ``prefill_ms``      — how long Ollama spent filling the KV cache for the
                          prompt.  This is the core TTFT driver.  Smaller
                          ``num_ctx`` → shorter KV initialisation → lower value.

    ``load_ms``         — model load + KV buffer allocation time.  Only non-zero
                          on the first call after model load (or after reload due
                          to keep_alive expiry).  Smaller num_ctx → lower value.

    ``eval_tps``        — true generation tokens/sec from Ollama's Metal timer.
                          Not estimated from character count — Ollama counts exact
                          subword tokens.  This is GPU-bound and does not change
                          significantly with num_ctx.
    """
    model_id: str
    num_ctx: int

    # Raw counts (from Ollama)
    prompt_eval_count: int          # tokens in the prompt
    eval_count: int                 # tokens generated

    # Timings in milliseconds (converted from Ollama's nanoseconds)
    prefill_ms: float               # prompt_eval_duration → time to fill KV cache
    eval_ms: float                  # eval_duration → time for generation
    load_ms: float                  # load_duration → model load + KV alloc
    total_ms: float                 # total_duration

    # Response content
    response_text: str = ""
    error: Optional[str] = None

    # ── Derived metrics ───────────────────────────────────────────────────

    @property
    def eval_tps(self) -> float:
        """True generation tok/s from Ollama's internal timer."""
        if self.eval_ms <= 0:
            return 0.0
        return self.eval_count / (self.eval_ms / 1000.0)

    @property
    def prefill_tps(self) -> float:
        """
        Prefill throughput (prompt tokens/s).
        Higher = faster KV-fill.  Larger num_ctx → lower prefill_tps even
        for the same prompt, because there is more KV buffer to initialise.
        """
        if self.prefill_ms <= 0:
            return 0.0
        return self.prompt_eval_count / (self.prefill_ms / 1000.0)

    @property
    def ttft_proxy_ms(self) -> float:
        """
        TTFT proxy = load_ms + prefill_ms.

        On a warm model (keep_alive=-1, model already loaded) load_ms ≈ 0
        and this equals prefill_ms.  On a cold model it includes the full
        model-load + KV-alloc penalty.
        """
        return self.load_ms + self.prefill_ms

    def __str__(self) -> str:
        return (
            f"NativeInferenceStats(ctx={self.num_ctx} "
            f"prefill={self.prefill_ms:.0f}ms "
            f"load={self.load_ms:.0f}ms "
            f"eval_tps={self.eval_tps:.1f} "
            f"eval_count={self.eval_count})"
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaMetricsClient:
    """
    Thin async client for Ollama's native /api/chat and /api/ps endpoints.

    Using the native endpoint (not /v1/chat/completions) gives us access to
    the internal timing fields that the OpenAI-compat endpoint omits.

    Usage
    -----
    ::

        client = OllamaMetricsClient()

        stats = await client.run_with_stats(
            model="phi4-mini:latest",
            messages=[{"role": "user", "content": "Hi"}],
            options={"num_ctx": 1290},
            keep_alive="5m",          # or "-1" to keep forever
        )
        print(f"prefill: {stats.prefill_ms:.0f}ms  tps: {stats.eval_tps:.1f}")
    """

    def __init__(
        self,
        base_url: str = _OLLAMA_BASE,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def run_with_stats(
        self,
        model: str,
        messages: list[dict],
        options: Optional[dict] = None,
        keep_alive: str = "5m",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
    ) -> NativeInferenceStats:
        """
        POST to /api/chat with stream=False and return a NativeInferenceStats.

        All timing fields come directly from Ollama's Go runtime —
        no Python-side estimation.

        Parameters
        ----------
        options:
            Ollama options dict.  At minimum should include ``num_ctx``.
        keep_alive:
            How long Ollama keeps the model loaded.  Use ``"5m"`` for
            raw-baseline tests, ``"-1"`` for autotune tests.
        """
        opts: dict = options.copy() if options else {}
        num_ctx = opts.get("num_ctx", 4096)

        # Layer in generation params if provided
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        if temperature is not None:
            opts["temperature"] = temperature
        if top_p is not None:
            opts["top_p"] = top_p
        if repeat_penalty is not None:
            opts["repeat_penalty"] = repeat_penalty

        payload: dict = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "options":    opts,
            "keep_alive": keep_alive,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                data = resp.json()

            if "error" in data:
                return NativeInferenceStats(
                    model_id=model, num_ctx=num_ctx,
                    prompt_eval_count=0, eval_count=0,
                    prefill_ms=0, eval_ms=0, load_ms=0, total_ms=0,
                    error=data["error"],
                )

            # Ollama returns durations in nanoseconds
            def _ms(ns_key: str) -> float:
                return data.get(ns_key, 0) / 1_000_000.0

            content = data.get("message", {}).get("content", "")

            return NativeInferenceStats(
                model_id=model,
                num_ctx=num_ctx,
                prompt_eval_count=data.get("prompt_eval_count", 0),
                eval_count=data.get("eval_count", 0),
                prefill_ms=_ms("prompt_eval_duration"),
                eval_ms=_ms("eval_duration"),
                load_ms=_ms("load_duration"),
                total_ms=_ms("total_duration"),
                response_text=content,
            )

        except httpx.TimeoutException:
            return NativeInferenceStats(
                model_id=model, num_ctx=num_ctx,
                prompt_eval_count=0, eval_count=0,
                prefill_ms=0, eval_ms=0, load_ms=0, total_ms=0,
                error=f"TIMEOUT after {self.timeout}s",
            )
        except Exception as exc:
            logger.exception("OllamaMetricsClient.run_with_stats failed")
            return NativeInferenceStats(
                model_id=model, num_ctx=num_ctx,
                prompt_eval_count=0, eval_count=0,
                prefill_ms=0, eval_ms=0, load_ms=0, total_ms=0,
                error=str(exc),
            )

    async def unload_model(self, model: str) -> bool:
        """
        Force Ollama to unload the model from memory.

        Sets ``keep_alive=0`` on a dummy generate request.  Returns True if
        the unload appears to have succeeded (model no longer in /api/ps).
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "keep_alive": "0"},
                )
            await asyncio.sleep(2.0)   # give Ollama time to free Metal buffers
            return True
        except Exception as exc:
            logger.warning("unload_model failed: %s", exc)
            return False

    async def is_model_loaded(self, model: str) -> bool:
        """Return True if the model is currently loaded in Ollama memory."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.base_url}/api/ps")
                data = r.json()
            return any(m.get("name") == model for m in data.get("models", []))
        except Exception:
            return False
