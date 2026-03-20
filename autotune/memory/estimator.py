"""Runtime memory estimator for a given model + config combination."""

from __future__ import annotations

from dataclasses import dataclass

from autotune.models.registry import ModelProfile, QUANTIZATIONS

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Fixed overhead from the runtime itself (llama.cpp / Python process, etc.)
RUNTIME_OVERHEAD_GB: float = 0.35

# Fraction of the budget we keep as swap/OOM safety headroom
SAFETY_MARGIN_FRACTION: float = 0.10


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MemoryEstimate:
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    safety_margin_gb: float
    total_required_gb: float   # weights + kv + overhead + safety
    peak_gb: float             # weights + kv + overhead (no safety pad)
    fits: bool
    headroom_gb: float         # budget - peak (negative means OOM)

    @property
    def efficiency(self) -> float:
        """Fraction of budget actually used (0–1).  Lower = more headroom."""
        if self.total_required_gb <= 0:
            return 0.0
        return min(1.0, self.peak_gb / (self.peak_gb + max(0, self.headroom_gb)))


# ---------------------------------------------------------------------------
# Core estimation logic
# ---------------------------------------------------------------------------

def estimate_memory(
    model: ModelProfile,
    quant: str,
    context_len: int,
    n_gpu_layers: int,
    available_gb: float,
) -> MemoryEstimate:
    """
    Estimate total memory required and whether it fits within `available_gb`.

    Parameters
    ----------
    model       : target ModelProfile
    quant       : quantization key from QUANTIZATIONS (e.g. "Q4_K_M")
    context_len : number of tokens in the context window to budget for
    n_gpu_layers: layers offloaded to GPU (0 = CPU-only)
    available_gb: effective memory the runtime can use (VRAM or system RAM)
    """
    if quant not in QUANTIZATIONS:
        raise ValueError(f"Unknown quantization: {quant!r}")

    # 1. Model weights
    weights_gb = model.weight_gb(quant)

    # 2. KV cache
    #    When layers are split across CPU + GPU the KV cache lives on the GPU
    #    for GPU layers and in RAM for CPU layers. We conservatively assume the
    #    full KV cache must fit in the target memory pool (VRAM or RAM).
    kv_gb = model.kv_cache_gb(context_len)

    # 3. Fixed runtime overhead
    overhead_gb = RUNTIME_OVERHEAD_GB

    # 4. Safety margin (applied on top of everything else)
    safety_gb = (weights_gb + kv_gb + overhead_gb) * SAFETY_MARGIN_FRACTION

    total_required = weights_gb + kv_gb + overhead_gb + safety_gb
    peak = weights_gb + kv_gb + overhead_gb

    headroom = available_gb - peak
    fits = headroom >= 0

    return MemoryEstimate(
        weights_gb=weights_gb,
        kv_cache_gb=kv_gb,
        overhead_gb=overhead_gb,
        safety_margin_gb=safety_gb,
        total_required_gb=total_required,
        peak_gb=peak,
        fits=fits,
        headroom_gb=headroom,
    )
