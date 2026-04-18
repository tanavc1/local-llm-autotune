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
    no_swap_arch=None,   # Optional[autotune.memory.noswap.ModelArch]
    prompt_caching_override: Optional[bool] = None,
) -> tuple[dict, list[str]]:
    """
    Return ``(options_dict, autotune_notices)`` for this Ollama request.

    ``options_dict`` fields
    -----------------------
    num_ctx   — dynamic minimum that fits input + max_new_tokens + buffer,
                further capped by context_ceiling and live RAM pressure
    f16_kv    — KV cache precision (F16 or Q8), determined by:
                  1. Profile default (fast=Q8, balanced/quality=F16)
                  2. kv_precision_override from ModelSelector pre-flight
                  3. Live RAM pressure: HIGH/CRITICAL force Q8 regardless
                Only Q8 and F16 are supported by Ollama's f16_kv flag.
    num_keep  — system-prompt prefix tokens to pin in KV (if profile or
                prompt_caching_override requests it)

    ``autotune_notices``
    --------------------
    Human-readable strings describing live adjustments made this request
    (e.g. "RAM 88% — context 8,192→6,144 tokens, KV F16→Q8").  Empty when
    everything is nominal.  Callers decide whether/how to display them.

    Parameters
    ----------
    context_ceiling : hard upper bound on num_ctx, set by the advisor or
                      pre-flight fit analysis.
    kv_precision_override : "Q8_0" or "F16" from the advisor.  Overrides
                      the profile default before pressure checks run.
    no_swap_arch : ModelArch from autotune.memory.noswap.  When provided,
                      NoSwapGuard runs after all other reductions.
    prompt_caching_override : when True, force prefix-cache pinning even if
                      the active profile does not enable system_prompt_cache.
                      Set by the advisor's improve_cache_reuse action.
    """
    # Start with the base options (num_ctx + f16_kv from profile)
    opts = ollama_options_for_profile(messages, profile)
    notices: list[str] = []

    # ── Flash attention + prefill batch size ────────────────────────────────
    # flash_attn reduces peak activation memory during the attention computation.
    # It is mathematically equivalent to standard attention — zero quality impact.
    # Models/builds that don't support it silently ignore the flag.
    opts["flash_attn"] = True

    # num_batch controls how many prompt tokens Ollama processes in a single
    # GPU pass during prefill.  Default (512) processes a 700-token prompt in
    # two passes; 1024 processes it in one → directly halves the number of
    # Metal kernel dispatches for long prompts.
    # llama.cpp caps the actual batch at min(num_batch, remaining_tokens), so
    # short prompts (<512 tokens) allocate no extra activation memory.
    opts["num_batch"] = 1024

    # Apply advisor KV precision override (pre-flight / live assessment).
    # Takes priority over profile default, but can still be further overridden
    # by live RAM pressure below.
    if kv_precision_override == "Q8_0":
        if opts.get("f16_kv", True):
            logger.debug("KV precision: F16 → Q8 (advisor override)")
            opts["f16_kv"] = False
    elif kv_precision_override == "F16":
        opts["f16_kv"] = True

    # Apply context ceiling BEFORE pressure reduction so pressure can only
    # further reduce, never expand beyond the safe limit.
    if context_ceiling is not None and context_ceiling > 0:
        if opts["num_ctx"] > context_ceiling:
            logger.debug(
                "num_ctx capped by advisor: %d → %d",
                opts["num_ctx"], context_ceiling,
            )
            opts["num_ctx"] = context_ceiling

    # Prefix-cache pinning: honour profile setting OR advisor override.
    use_caching = profile.system_prompt_cache or bool(prompt_caching_override)
    if use_caching:
        num_keep = compute_num_keep(messages, profile)
        # compute_num_keep respects profile.system_prompt_cache; if we only have
        # the override, compute the token count directly.
        if num_keep == 0 and prompt_caching_override:
            for m in messages:
                if m.get("role") == "system":
                    num_keep += estimate_tokens(m.get("content", ""))
                else:
                    break
        if num_keep > 0:
            # Never let num_keep consume the entire context window — always
            # leave at least 512 tokens for the actual conversation turn.
            num_keep = min(num_keep, max(0, opts["num_ctx"] - 512))
            if num_keep > 0:
                opts["num_keep"] = num_keep
                logger.debug("num_keep=%d (system prompt prefix cached)", num_keep)

    # ── Live memory-pressure reductions ─────────────────────────────────────
    # num_ctx is reduced first (cheap, immediate effect on KV allocation).
    # KV precision is downgraded from F16→Q8 at HIGH/CRITICAL pressure —
    # this halves the KV footprint and is more effective than context reduction
    # alone because it applies to all future tokens, not just the current window.
    vm = psutil.virtual_memory()
    ram_pct = vm.percent
    original_ctx = opts["num_ctx"]
    was_f16 = opts.get("f16_kv", True)

    if ram_pct >= _PRESSURE_CRITICAL_PCT:
        opts["num_ctx"] = max(512, int(original_ctx * 0.50))
        # At critical pressure the model is reloaded for the new num_ctx, so
        # reducing num_batch here is free — it piggybacks on the forced reload.
        opts["num_batch"] = 256
        logger.warning(
            "Critical RAM %.1f%% — halving num_ctx %d→%d, batch 1024→256",
            ram_pct, original_ctx, opts["num_ctx"],
        )
        kv_note = ""
        if was_f16:
            opts["f16_kv"] = False
            kv_note = ", KV F16→Q8"
            logger.warning("Critical RAM %.1f%% — KV precision F16→Q8", ram_pct)
        notices.append(
            f"RAM {ram_pct:.0f}% (critical) — "
            f"context {original_ctx:,}→{opts['num_ctx']:,} tokens{kv_note}, batch→256"
        )

    elif ram_pct >= _PRESSURE_HIGH_PCT:
        opts["num_ctx"] = max(512, int(original_ctx * 0.75))
        logger.debug(
            "High RAM %.1f%% — reducing num_ctx %d→%d",
            ram_pct, original_ctx, opts["num_ctx"],
        )
        kv_note = ""
        if was_f16:
            opts["f16_kv"] = False
            kv_note = ", KV F16→Q8"
            logger.warning("High RAM %.1f%% — KV precision F16→Q8", ram_pct)
        notices.append(
            f"RAM {ram_pct:.0f}% — "
            f"context {original_ctx:,}→{opts['num_ctx']:,} tokens{kv_note}"
        )

    elif ram_pct >= _PRESSURE_MODERATE_PCT:
        new_ctx = max(512, int(original_ctx * 0.90))
        if new_ctx < original_ctx:
            opts["num_ctx"] = new_ctx
            logger.debug(
                "Moderate RAM %.1f%% — reducing num_ctx %d→%d",
                ram_pct, original_ctx, new_ctx,
            )
            notices.append(
                f"RAM {ram_pct:.0f}% — context {original_ctx:,}→{new_ctx:,} tokens"
            )

    # ── No-swap guarantee (applied last, after all other reductions) ────────
    if no_swap_arch is not None:
        from autotune.memory.noswap import NoSwapGuard
        from autotune.ttft.optimizer import _snap_to_bucket
        guard = NoSwapGuard()
        decision = guard.apply(
            num_ctx=opts["num_ctx"],
            f16_kv=opts.get("f16_kv", True),
            arch=no_swap_arch,
            snap_fn=_snap_to_bucket,
        )
        if decision.ctx_changed:
            logger.info(
                "NoSwapGuard: ctx %d→%d, KV %s  (level=%s, avail=%.2fGB)",
                opts["num_ctx"], decision.num_ctx,
                "F16" if decision.f16_kv else "Q8",
                decision.level,
                decision.available_gb,
            )
        opts["num_ctx"] = decision.num_ctx
        opts["f16_kv"] = decision.f16_kv

    return opts, notices


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
