"""
NoSwapGuard — prevents inference from pushing the system into swap.

The problem
-----------
On Apple Silicon (unified memory architecture), when RAM fills up macOS
starts compressing memory pages, then pages to NVMe. Either path causes
inference to crater — generation speed drops from 30+ tok/s to <5 tok/s,
and the whole computer becomes unresponsive.

The culprit is Ollama's KV cache. For a 14B model at the default
num_ctx=4096, Ollama allocates ~800MB of Metal memory *before* the first
token is generated. If available RAM < 800MB, the allocation spills.

Ollama doesn't prevent this — it's a server that allocates what it's told
and lets the OS handle the consequences. autotune runs in front of Ollama
and can read RAM state before every request.

Algorithm
---------
Before building request options, NoSwapGuard:

1. Measures available RAM (psutil.virtual_memory().available — the true
   free + reclaimable number, not "used").

2. Subtracts SAFETY_MARGIN_GB (default 1.5 GB) to stay well clear of
   pressure boundaries. macOS starts compressing at ~85% utilisation and
   the safety margin keeps us below that.

3. Estimates KV bytes needed for this request:
       kv_bytes = 2 × n_layers × n_kv_heads × head_dim × num_ctx × bytes_per_elem

4. If kv_bytes > usable (available - safety_margin), applies reductions
   in order until it fits:

   Level 0  no change           — fits with room to spare
   Level 1  ctx × 0.75          — mild pressure, trim 25%
   Level 2  ctx × 0.50          — moderate, halve the context
   Level 3  ctx × 0.50 + Q8 KV  — Q8 halves KV memory (2B→1B per element)
   Level 4  ctx × 0.25 + Q8 KV  — severe pressure
   Level 5  ctx = 512  + Q8 KV  — minimum viable context (emergency)

5. Each reduced ctx is re-snapped to the nearest bucket (prevents KV
   thrashing even under pressure).

Model architecture
------------------
KV size scales with n_layers, n_kv_heads, and head_dim — all model-specific.
NoSwapGuard queries /api/show once per model and caches the result. Falls
back to conservative defaults (32L / 8KV / 128 head_dim) for unknown models.

Usage
-----
::

    guard = NoSwapGuard()
    arch  = await guard.get_model_arch("qwen3:8b")
    decision = guard.apply(num_ctx=1536, f16_kv=True, arch=arch)

    if decision.level != "ok":
        print(f"Reduced ctx {decision.reduced_from} → {decision.num_ctx} "
              f"({decision.level}) — {decision.reason}")

    # Use decision.num_ctx and decision.f16_kv in your Ollama options
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx
import psutil

logger = logging.getLogger(__name__)

_OLLAMA_BASE   = "http://localhost:11434"
_SAFETY_MARGIN = 1.5    # GB always kept free (macOS starts compressing ~15% headroom)
_MIN_CTX       = 512    # never go below this


# ---------------------------------------------------------------------------
# Architecture cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelArch:
    n_layers:   int
    n_kv_heads: int
    head_dim:   int
    arch_name:  str = "unknown"

    def kv_gb(self, num_ctx: int, f16: bool = True) -> float:
        """
        Theoretical KV cache size in GB.
        formula: 2 (K+V) × layers × kv_heads × head_dim × ctx × bytes_per_elem
        """
        bpe = 2 if f16 else 1   # F16 = 2 bytes, Q8 = 1 byte
        return (2 * self.n_layers * self.n_kv_heads * self.head_dim * num_ctx * bpe) / 1024**3

# Conservative fallback for unknown models
_FALLBACK_ARCH = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128, arch_name="fallback")

# In-process cache so we only query /api/show once per model
_arch_cache: dict[str, ModelArch] = {}


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------

@dataclass
class NoSwapDecision:
    """
    The (num_ctx, f16_kv) pair that safely fits in available RAM.
    """
    num_ctx:       int
    f16_kv:        bool
    level:         str      # "ok" | "l1_trim" | "l2_halve" | "l3_q8" | "l4_quarter_q8" | "l5_emergency"
    reduced_from:  int      # original num_ctx before reduction
    reason:        str      # human-readable explanation
    available_gb:  float    # what psutil reported
    kv_gb_before:  float    # KV size with original ctx
    kv_gb_after:   float    # KV size with reduced ctx
    safety_margin: float    # _SAFETY_MARGIN used

    @property
    def ctx_changed(self) -> bool:
        return self.num_ctx != self.reduced_from

    @property
    def kv_saved_gb(self) -> float:
        return round(self.kv_gb_before - self.kv_gb_after, 3)


# ---------------------------------------------------------------------------
# Reduction levels
# ---------------------------------------------------------------------------

# (factor, f16_kv, level_name, description)
_LEVELS = [
    (1.00, True,  "ok",         "fits comfortably"),
    (0.75, True,  "l1_trim",    "light pressure: ctx trimmed 25%"),
    (0.50, True,  "l2_halve",   "moderate pressure: ctx halved"),
    (0.50, False, "l3_q8",      "high pressure: ctx halved + Q8 KV (saves ~50% KV memory)"),
    (0.25, False, "l4_quarter", "severe pressure: ctx quartered + Q8 KV"),
    (None, False, "l5_min",     "critical: minimum ctx (512) + Q8 KV"),
]


# ---------------------------------------------------------------------------
# NoSwapGuard
# ---------------------------------------------------------------------------

class NoSwapGuard:
    """
    Pre-flight RAM check that prevents any inference from triggering swap.

    Stateless — call apply() before each request. The only shared state
    is the in-process model architecture cache (_arch_cache), which is
    populated lazily via get_model_arch().
    """

    def __init__(self, safety_margin_gb: float = _SAFETY_MARGIN) -> None:
        self.safety_margin_gb = safety_margin_gb

    # ── Architecture detection ────────────────────────────────────────────────

    @staticmethod
    async def get_model_arch(model_id: str) -> ModelArch:
        """
        Query /api/show for model architecture parameters.
        Results are cached for the lifetime of the process.

        Returns _FALLBACK_ARCH on any error.
        """
        if model_id in _arch_cache:
            return _arch_cache[model_id]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    f"{_OLLAMA_BASE}/api/show",
                    json={"name": model_id},
                )
                info = r.json().get("model_info", {})

            arch_name  = info.get("general.architecture", "llama")
            n_layers   = int(info.get(f"{arch_name}.block_count", 32))
            n_kv_heads = int(info.get(f"{arch_name}.attention.head_count_kv", 8))
            n_heads    = int(info.get(f"{arch_name}.attention.head_count", 32))
            embed_dim  = int(info.get(f"{arch_name}.embedding_length", 4096))
            head_dim   = embed_dim // max(n_heads, 1)

            arch = ModelArch(
                n_layers=n_layers,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                arch_name=arch_name,
            )
            _arch_cache[model_id] = arch
            logger.debug("NoSwapGuard: arch for %s: %s", model_id, arch)
            return arch

        except Exception as exc:
            logger.warning("NoSwapGuard: /api/show failed for %s: %s — using fallback", model_id, exc)
            _arch_cache[model_id] = _FALLBACK_ARCH
            return _FALLBACK_ARCH

    # ── Core application ─────────────────────────────────────────────────────

    def apply(
        self,
        num_ctx: int,
        f16_kv: bool,
        arch: ModelArch,
        snap_fn=None,          # callable(int) -> int, e.g. _snap_to_bucket
    ) -> NoSwapDecision:
        """
        Return the (num_ctx, f16_kv) pair that fits in available RAM.

        Parameters
        ----------
        num_ctx:   proposed context length (already bucket-snapped)
        f16_kv:    proposed KV precision from TTFTOptimizer
        arch:      model architecture (from get_model_arch)
        snap_fn:   optional function to snap ctx to a KV-reuse-friendly bucket

        Returns a NoSwapDecision. If level == "ok", no change was needed.
        """
        vm = psutil.virtual_memory()
        available_gb = vm.available / 1024**3
        usable_gb    = max(0.0, available_gb - self.safety_margin_gb)

        kv_gb_before = arch.kv_gb(num_ctx, f16=f16_kv)

        def _snap(ctx: int) -> int:
            if snap_fn:
                return snap_fn(max(_MIN_CTX, ctx))
            return max(_MIN_CTX, ctx)

        for factor, use_f16, level_name, description in _LEVELS:
            if factor is None:
                candidate_ctx = _MIN_CTX
            else:
                candidate_ctx = _snap(int(num_ctx * factor))

            kv_candidate = arch.kv_gb(candidate_ctx, f16=use_f16)

            if kv_candidate <= usable_gb:
                return NoSwapDecision(
                    num_ctx=candidate_ctx,
                    f16_kv=use_f16,
                    level=level_name,
                    reduced_from=num_ctx,
                    reason=description,
                    available_gb=round(available_gb, 2),
                    kv_gb_before=round(kv_gb_before, 3),
                    kv_gb_after=round(kv_candidate, 3),
                    safety_margin=self.safety_margin_gb,
                )

        # Should never reach here (512 + Q8 is tiny), but be safe
        kv_min = arch.kv_gb(_MIN_CTX, f16=False)
        return NoSwapDecision(
            num_ctx=_MIN_CTX,
            f16_kv=False,
            level="l5_min",
            reduced_from=num_ctx,
            reason="critical memory pressure — minimum viable ctx",
            available_gb=round(available_gb, 2),
            kv_gb_before=round(kv_gb_before, 3),
            kv_gb_after=round(kv_min, 3),
            safety_margin=self.safety_margin_gb,
        )

    # ── Convenience ──────────────────────────────────────────────────────────

    @staticmethod
    def ram_state() -> tuple[float, float, float]:
        """Returns (total_gb, used_gb, available_gb) from psutil."""
        vm = psutil.virtual_memory()
        return (
            round(vm.total    / 1024**3, 2),
            round(vm.used     / 1024**3, 2),
            round(vm.available / 1024**3, 2),
        )

    @staticmethod
    def would_swap(required_gb: float, safety_margin_gb: float = _SAFETY_MARGIN) -> bool:
        """
        Quick check: would allocating `required_gb` push us into swap?
        Returns True if allocation would exceed available minus safety margin.
        """
        vm = psutil.virtual_memory()
        available_gb = vm.available / 1024**3
        return required_gb > (available_gb - safety_margin_gb)
