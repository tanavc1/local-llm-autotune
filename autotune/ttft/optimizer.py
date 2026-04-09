"""
TTFTOptimizer — the single authoritative source for TTFT-affecting Ollama options.

DO NOT duplicate TTFT logic elsewhere in the codebase.  If you are building
Ollama request options and care about latency, import this class.

Design contract
---------------
- ``build_request_options`` returns a ready-to-use dict.
  Pass ``result["options"]`` to Ollama's ``options`` field.
  Pass ``result["keep_alive"]`` to Ollama's top-level ``keep_alive`` field.
- Memory-pressure adjustments (num_ctx reduction, KV downgrade) are applied
  here at call time, so the options always reflect live system state.
- Hardware tuning (QOS class, GC disable) is applied separately by
  :class:`autotune.api.hardware_tuner.HardwareTuner` — that layer is about
  CPU scheduling, not KV cache decisions.

Why keep these three together
------------------------------
Dynamic num_ctx, keep_alive, and num_keep interact:
  - num_ctx drives how much KV memory Ollama allocates → affects TTFT
  - keep_alive prevents the KV cache from being discarded → makes warm-path fast
  - num_keep tells Ollama which prefix to retain in its KV → skips re-evaluation

Separating them would make it easy to "accidentally" override one without the
other and silently degrade TTFT.  Keeping them in one place means any TTFT
regression is localised here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import psutil

from autotune.api.ctx_utils import compute_num_ctx, estimate_tokens

if TYPE_CHECKING:
    from autotune.api.profiles import Profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEEP_ALIVE_FOREVER = "-1"   # tell Ollama: never unload the model

# Memory-pressure thresholds that trigger num_ctx/KV reduction.
# These are intentionally low — we'd rather accept a slightly smaller context
# than thrash swap during inference.
_PRESSURE_MODERATE = 80.0   # ≥ 80% used → trim num_ctx by 10%
_PRESSURE_HIGH     = 88.0   # ≥ 88% used → trim by 25%, downgrade KV F16→Q8
_PRESSURE_CRITICAL = 93.0   # ≥ 93% used → halve num_ctx, force Q8 KV


# ---------------------------------------------------------------------------
# TTFTOptimizer
# ---------------------------------------------------------------------------

class TTFTOptimizer:
    """
    Builds the complete set of Ollama options that minimise TTFT.

    This class is stateless — create one per request or reuse the same
    instance; there is no mutable state between calls.

    Parameters passed to ``build_request_options`` are sufficient to produce
    a complete options dict; no global state is mutated.

    Example
    -------
    ::

        from autotune.ttft import TTFTOptimizer
        from autotune.api.profiles import get_profile

        optimizer = TTFTOptimizer()
        profile = get_profile("balanced")
        result = optimizer.build_request_options(messages, profile)

        # In your Ollama HTTP call:
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": profile.max_new_tokens,
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "options": result["options"],          # ← all three mechanisms
            "keep_alive": result["keep_alive"],     # ← mechanism 2
        }
    """

    def build_request_options(
        self,
        messages: list[dict],
        profile: "Profile",
        context_ceiling: Optional[int] = None,
        kv_precision_override: Optional[str] = None,
    ) -> dict:
        """
        Return a dict with two keys::

            {
                "options":    { ... },    # pass as Ollama's ``options`` field
                "keep_alive": "-1",       # pass as Ollama's top-level ``keep_alive``
                "_debug":     { ... },    # diagnostic info (not sent to Ollama)
            }

        Mechanism 1 — Dynamic num_ctx
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Computes the minimum num_ctx that fits this request, applies any
        ModelSelector ceiling (``context_ceiling``), then applies live
        memory-pressure reductions.

        Mechanism 2 — keep_alive
        ~~~~~~~~~~~~~~~~~~~~~~~~
        Always returns ``keep_alive = "-1"`` (model stays in memory forever).
        This is non-negotiable for TTFT: a reloading model adds 1–4 s to the
        first call in any session.

        Mechanism 3 — num_keep (prefix caching)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        If the first message(s) are system-role, pins those tokens in Ollama's
        KV via ``num_keep``.  Ollama will never re-evaluate the system prompt
        across turns in the same conversation — pure latency savings.

        Parameters
        ----------
        messages:
            OpenAI-style message list for this request.
        profile:
            The active inference profile (controls max_context_tokens, KV
            precision preference, whether to cache system prompts, etc.)
        context_ceiling:
            Hard upper bound on num_ctx from ModelSelector pre-flight analysis.
            Prevents the context from exceeding what's safe for this model/HW.
        kv_precision_override:
            ``"Q8_0"`` or ``"F16"`` from ModelSelector.  Overrides profile
            default KV precision before live pressure checks.
        """
        # ── Mechanism 1: Dynamic num_ctx ─────────────────────────────────────
        num_ctx = compute_num_ctx(messages, profile)

        # Apply ModelSelector ceiling before pressure (pressure can only reduce further)
        if context_ceiling is not None and context_ceiling > 0:
            if num_ctx > context_ceiling:
                logger.debug("num_ctx capped by model selector: %d → %d", num_ctx, context_ceiling)
                num_ctx = context_ceiling

        # ── KV cache precision (f16_kv flag) ─────────────────────────────────
        # Profile default
        f16_kv: bool = profile.kv_cache_precision != "q8"

        # ModelSelector override
        if kv_precision_override == "Q8_0":
            f16_kv = False
            logger.debug("KV precision: → Q8 (model selector override)")
        elif kv_precision_override == "F16":
            f16_kv = True

        # ── Live memory-pressure reductions ──────────────────────────────────
        vm = psutil.virtual_memory()
        ram_pct = vm.percent
        pressure_level = "normal"
        original_ctx = num_ctx

        if ram_pct >= _PRESSURE_CRITICAL:
            pressure_level = "critical"
            num_ctx = max(512, num_ctx // 2)
            f16_kv = False
            logger.warning(
                "TTFT: critical memory pressure %.1f%% — "
                "num_ctx %d→%d, KV downgraded to Q8",
                ram_pct, original_ctx, num_ctx,
            )
        elif ram_pct >= _PRESSURE_HIGH:
            pressure_level = "high"
            num_ctx = max(512, int(num_ctx * 0.75))
            f16_kv = False
            logger.info(
                "TTFT: high memory pressure %.1f%% — "
                "num_ctx %d→%d, KV downgraded to Q8",
                ram_pct, original_ctx, num_ctx,
            )
        elif ram_pct >= _PRESSURE_MODERATE:
            pressure_level = "moderate"
            num_ctx = max(512, int(num_ctx * 0.90))
            logger.info(
                "TTFT: moderate memory pressure %.1f%% — "
                "num_ctx %d→%d",
                ram_pct, original_ctx, num_ctx,
            )

        # ── Mechanism 3: num_keep (prefix caching) ───────────────────────────
        num_keep = 0
        if profile.system_prompt_cache:
            for m in messages:
                if m.get("role") == "system":
                    num_keep += estimate_tokens(m.get("content", ""))
                else:
                    break   # only leading system messages are cached

        # ── Assemble options dict ─────────────────────────────────────────────
        options: dict = {
            "num_ctx": num_ctx,
            "f16_kv": f16_kv,
        }
        if num_keep > 0:
            options["num_keep"] = num_keep

        debug = {
            "pressure_level":  pressure_level,
            "ram_pct":         round(ram_pct, 1),
            "num_ctx_raw":     original_ctx,
            "num_ctx_final":   num_ctx,
            "num_keep":        num_keep,
            "f16_kv":          f16_kv,
            "keep_alive":      KEEP_ALIVE_FOREVER,
        }
        logger.debug("TTFTOptimizer: %s", debug)

        return {
            "options":    options,
            "keep_alive": KEEP_ALIVE_FOREVER,   # Mechanism 2
            "_debug":     debug,
        }
