"""
KV Cache Manager — central authority for KV budget decisions.

Responsibilities
----------------
1. Prefix-cache the system prompt via Ollama's `num_keep` parameter.
   Ollama keeps the first `num_keep` tokens of every request in KV across
   turns.  When the system prompt is the same across turns it is never
   re-evaluated — pure latency and memory win at no quality cost.

2. Dynamically size `num_ctx` to the minimum needed for this request
   (delegates to ctx_utils.compute_num_ctx).

3. Set KV precision (`f16_kv`) based on the profile and current memory.

4. Detect memory pressure before issuing options and pre-emptively reduce
   `num_ctx` to avoid OOM / swap thrash.

5. Expose a lightweight telemetry check that callers can log.

All decisions are stateless with respect to conversation history — callers
pass messages in, get an options dict out.  No global state is mutated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import psutil

from autotune.api.ctx_utils import (
    compute_num_ctx,
    estimate_tokens,
    ollama_options_for_profile,
)

if TYPE_CHECKING:
    from autotune.api.profiles import Profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory pressure thresholds
# ---------------------------------------------------------------------------

_PRESSURE_HIGH_PCT     = 88.0   # reduce num_ctx by 25 %
_PRESSURE_MODERATE_PCT = 80.0   # reduce num_ctx by 10 %
_PRESSURE_CRITICAL_PCT = 93.0   # hard limit — refuse or shrink to minimum


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_num_keep(messages: list[dict], profile: "Profile") -> int:
    """
    Return `num_keep`: how many leading tokens Ollama should pin in KV.

    When `profile.system_prompt_cache=True` and the first message(s) are
    system-role, those tokens are kept across turns so Ollama never
    re-evaluates the system prompt.  This directly reduces TTFT on every
    turn after the first.

    Returns 0 when the profile does not request caching or there is no
    leading system message.
    """
    if not profile.system_prompt_cache:
        return 0
    kept = 0
    for m in messages:
        if m.get("role") == "system":
            kept += estimate_tokens(m.get("content", ""))
        else:
            break   # only leading system messages count
    return kept


def build_ollama_options(
    messages: list[dict],
    profile: "Profile",
    context_ceiling: Optional[int] = None,
    kv_precision_override: Optional[str] = None,
) -> dict:
    """
    Return the complete Ollama `options` dict for this request.

    Fields set
    ----------
    num_ctx   — dynamic minimum that fits input + max_new_tokens + buffer,
                further capped by context_ceiling and live RAM pressure
    f16_kv    — KV cache precision (F16 or Q8), determined by:
                  1. Profile default (fast=Q8, balanced/quality=F16)
                  2. kv_precision_override from ModelSelector pre-flight
                  3. Live RAM pressure: HIGH/CRITICAL force Q8 regardless
                Only Q8 and F16 are supported by Ollama's f16_kv flag.
    num_keep  — system-prompt prefix tokens to pin in KV (if profile wants it)

    Parameters
    ----------
    context_ceiling : hard upper bound on num_ctx, set by ModelSelector from
                      pre-flight fit analysis.  Prevents the context from
                      exceeding what the model selector determined is safe
                      for this model on this hardware.  When None, only live
                      RAM pressure governs the ceiling.
    kv_precision_override : "Q8_0" or "F16" from ModelSelector.recommended_kv.
                      Overrides the profile default before pressure checks.
                      Useful when ModelSelector assessed MARGINAL/SWAP_RISK fit
                      and recommends Q8 to halve KV memory consumption.
    """
    # Start with the base options (num_ctx + f16_kv from profile)
    opts = ollama_options_for_profile(messages, profile)

    # Apply ModelSelector KV precision override (pre-flight assessment).
    # This takes priority over profile default but can still be further
    # overridden by live RAM pressure below.
    if kv_precision_override == "Q8_0":
        if opts.get("f16_kv", True):
            logger.debug("KV precision: F16 → Q8 (model selector override)")
            opts["f16_kv"] = False
    elif kv_precision_override == "F16":
        opts["f16_kv"] = True

    # Apply ModelSelector ceiling BEFORE pressure reduction so pressure
    # reduction can only further reduce, never expand beyond the safe limit.
    if context_ceiling is not None and context_ceiling > 0:
        if opts["num_ctx"] > context_ceiling:
            logger.debug(
                "num_ctx capped by model selector: %d → %d",
                opts["num_ctx"], context_ceiling,
            )
            opts["num_ctx"] = context_ceiling

    # Add prefix-cache pinning
    num_keep = compute_num_keep(messages, profile)
    if num_keep > 0:
        opts["num_keep"] = num_keep
        logger.debug("num_keep=%d (system prompt prefix cached)", num_keep)

    # Apply live memory-pressure reductions.
    # num_ctx is reduced first (cheap, immediate).  KV precision is downgraded
    # from F16 to Q8 under HIGH/CRITICAL pressure — this halves the KV cache
    # memory footprint and is more effective than a num_ctx reduction alone
    # because it applies across all future tokens, not just the current window.
    vm = psutil.virtual_memory()
    ram_pct = vm.percent
    original_ctx = opts["num_ctx"]

    if ram_pct >= _PRESSURE_CRITICAL_PCT:
        opts["num_ctx"] = max(512, int(original_ctx * 0.50))
        logger.warning(
            "Critical memory pressure %.1f%% — halving num_ctx: %d → %d",
            ram_pct, original_ctx, opts["num_ctx"],
        )
        if opts.get("f16_kv", True):
            opts["f16_kv"] = False
            logger.warning(
                "Critical memory pressure %.1f%% — KV precision downgraded F16 → Q8",
                ram_pct,
            )
    elif ram_pct >= _PRESSURE_HIGH_PCT:
        opts["num_ctx"] = max(512, int(original_ctx * 0.75))
        logger.debug(
            "High memory pressure %.1f%% — reducing num_ctx: %d → %d",
            ram_pct, original_ctx, opts["num_ctx"],
        )
        if opts.get("f16_kv", True):
            opts["f16_kv"] = False
            logger.warning(
                "High memory pressure %.1f%% — KV precision downgraded F16 → Q8",
                ram_pct,
            )
    elif ram_pct >= _PRESSURE_MODERATE_PCT:
        opts["num_ctx"] = max(512, int(original_ctx * 0.90))
        logger.debug(
            "Moderate memory pressure %.1f%% — reducing num_ctx: %d → %d",
            ram_pct, original_ctx, opts["num_ctx"],
        )
        # Moderate pressure: keep KV precision but log a heads-up
        if opts.get("f16_kv", True):
            logger.debug(
                "Moderate memory pressure %.1f%% — KV still F16; "
                "will downgrade at %.1f%%",
                ram_pct, _PRESSURE_HIGH_PCT,
            )

    return opts


def memory_pressure_snapshot() -> dict:
    """
    Return a point-in-time snapshot of memory pressure for telemetry.

    Returns
    -------
    dict with keys: ram_pct, swap_pct, available_gb, swap_used_gb,
    pressure_level ("normal" | "moderate" | "high" | "critical")
    """
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    ram_pct = vm.percent

    if ram_pct >= _PRESSURE_CRITICAL_PCT:
        level = "critical"
    elif ram_pct >= _PRESSURE_HIGH_PCT:
        level = "high"
    elif ram_pct >= _PRESSURE_MODERATE_PCT:
        level = "moderate"
    else:
        level = "normal"

    return {
        "ram_pct":       round(ram_pct, 1),
        "swap_pct":      round(sw.percent, 1),
        "available_gb":  round(vm.available / 1024**3, 3),
        "swap_used_gb":  round(sw.used / 1024**3, 3),
        "pressure_level": level,
    }


def kv_memory_estimate_mb(
    num_ctx: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    f16_kv: bool = True,
) -> float:
    """
    Estimate KV cache memory for a given context size and model architecture.

    Formula: 2 * layers * kv_heads * head_dim * num_ctx * bytes_per_elem
    (×2 because we store both K and V)
    """
    bytes_per_elem = 2.0 if f16_kv else 1.0   # f16=2B, q8=1B per element
    bytes_total = 2 * n_layers * n_kv_heads * head_dim * num_ctx * bytes_per_elem
    return bytes_total / (1024 ** 2)
