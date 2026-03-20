"""
Config generator and multi-objective scorer.

Pipeline:
  1. Enumerate candidate configs (model × quant × context × gpu_layers)
  2. Estimate memory for each candidate
  3. Discard configs that don't fit
  4. Score survivors on four axes: fit, stability, speed, quality
  5. Return top-N per mode (fastest / balanced / best_quality)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from autotune.hardware.profiler import HardwareProfile
from autotune.memory.estimator import MemoryEstimate, estimate_memory
from autotune.models.registry import (
    MODEL_REGISTRY,
    QUANTIZATIONS,
    ModelProfile,
    QuantizationSpec,
    list_models,
)

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

CONTEXT_LENGTHS: list[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# Fraction of model layers to place on GPU (0 = CPU-only, 1.0 = all-GPU)
GPU_LAYER_FRACTIONS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CandidateConfig:
    model: ModelProfile
    quant: str
    context_len: int
    n_gpu_layers: int           # absolute layer count on GPU
    gpu_layer_fraction: float   # 0–1


@dataclass
class ScoredConfig:
    candidate: CandidateConfig
    memory: MemoryEstimate

    # Raw component scores (0–1 each)
    fit_score: float            # 1 if fits, 0 otherwise
    stability_score: float      # headroom relative to budget
    speed_score: float          # estimated relative throughput
    quality_score: float        # model size + quant fidelity

    # Final weighted composite
    composite: float

    # Human-readable explanation of why this was chosen
    rationale: str = ""


@dataclass
class Recommendation:
    mode: str                   # "fastest" | "balanced" | "best_quality"
    primary: ScoredConfig
    alternatives: list[ScoredConfig]


# ---------------------------------------------------------------------------
# Mode weight profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeWeights:
    stability: float
    speed: float
    quality: float
    context: float              # reward for larger context window


MODE_WEIGHTS: dict[str, ModeWeights] = {
    "fastest": ModeWeights(stability=0.20, speed=0.55, quality=0.10, context=0.15),
    "balanced": ModeWeights(stability=0.30, speed=0.30, quality=0.25, context=0.15),
    "best_quality": ModeWeights(stability=0.20, speed=0.10, quality=0.55, context=0.15),
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _stability_score(mem: MemoryEstimate, available_gb: float) -> float:
    """Fraction of available memory left after peak usage (0–1)."""
    if not mem.fits:
        return 0.0
    return min(1.0, mem.headroom_gb / available_gb)


def _speed_score(
    candidate: CandidateConfig,
    hw: HardwareProfile,
) -> float:
    """
    Estimated relative throughput normalised to 0–1.

    Factors:
      - Quantization speed multiplier (higher bits = slower)
      - GPU layer fraction (more GPU layers = faster, assuming GPU available)
      - CPU core count for CPU-only inference
    """
    q_spec: QuantizationSpec = QUANTIZATIONS[candidate.quant]
    quant_factor = q_spec.speed_multiplier / QUANTIZATIONS["F16"].speed_multiplier

    if hw.has_gpu and candidate.gpu_layer_fraction > 0:
        # GPU acceleration: throughput scales roughly with fraction of layers on GPU
        gpu_boost = 1.0 + candidate.gpu_layer_fraction * 3.0  # up to 4× over CPU
    else:
        # CPU-only: normalise by core count; 16+ physical cores ≈ best
        gpu_boost = min(1.0, hw.cpu.physical_cores / 16.0)

    raw = quant_factor * gpu_boost
    # Normalise: maximum possible raw ≈ 4 * 4 = 16 (F16 quant_factor=1, gpu_boost=4)
    return min(1.0, raw / 16.0)


def _quality_score(candidate: CandidateConfig) -> float:
    """
    Model quality as a blend of parameter count normalised score and quant fidelity.

    Parameter score: log-linear from 1B (0.0) to 70B (1.0).
    Quant fidelity: from registry (0–1).
    """
    params_b = candidate.model.parameters_b
    # log scale: log(1)=0 → 0.0, log(70)≈4.25 → 1.0
    param_score = min(1.0, math.log(max(1.0, params_b)) / math.log(70))

    quant_fidelity = QUANTIZATIONS[candidate.quant].quality_score

    return 0.55 * param_score + 0.45 * quant_fidelity


def _context_score(candidate: CandidateConfig) -> float:
    """Reward larger context windows (log-scale, 512 → 0, 32768 → 1)."""
    return min(1.0, math.log2(candidate.context_len / 512) / math.log2(64))


def _composite(
    stability: float,
    speed: float,
    quality: float,
    context: float,
    weights: ModeWeights,
) -> float:
    return (
        weights.stability * stability
        + weights.speed * speed
        + weights.quality * quality
        + weights.context * context
    )


def _build_rationale(sc: ScoredConfig, mode: str) -> str:
    c = sc.candidate
    mem = sc.memory
    quant_desc = QUANTIZATIONS[c.quant].description
    gpu_note = (
        f"all {c.n_gpu_layers} layers on GPU"
        if c.gpu_layer_fraction == 1.0
        else f"{c.n_gpu_layers} of {c.model.n_layers} layers on GPU"
        if c.n_gpu_layers > 0
        else "CPU-only inference"
    )
    return (
        f"{c.model.name} @ {c.quant} ({quant_desc}), "
        f"context {c.context_len} tokens, {gpu_note}. "
        f"Uses {mem.peak_gb:.2f} GB of {mem.peak_gb + mem.headroom_gb:.2f} GB "
        f"available ({mem.headroom_gb:.2f} GB headroom). "
        f"Mode: {mode}."
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def _enumerate_candidates(hw: HardwareProfile) -> list[CandidateConfig]:
    candidates: list[CandidateConfig] = []
    for model in list_models():
        for quant in model.quantization_options:
            for ctx in CONTEXT_LENGTHS:
                # Skip context windows beyond model's trained max
                if ctx > model.context_window:
                    continue
                if hw.has_gpu:
                    fractions = GPU_LAYER_FRACTIONS
                else:
                    fractions = [0.0]  # CPU-only machine
                for frac in fractions:
                    n_gpu = round(model.n_layers * frac)
                    candidates.append(
                        CandidateConfig(
                            model=model,
                            quant=quant,
                            context_len=ctx,
                            n_gpu_layers=n_gpu,
                            gpu_layer_fraction=frac,
                        )
                    )
    return candidates


def _score_candidate(
    candidate: CandidateConfig,
    hw: HardwareProfile,
    weights: ModeWeights,
) -> Optional[ScoredConfig]:
    mem = estimate_memory(
        model=candidate.model,
        quant=candidate.quant,
        context_len=candidate.context_len,
        n_gpu_layers=candidate.n_gpu_layers,
        available_gb=hw.effective_memory_gb,
    )

    if not mem.fits:
        return None

    stability = _stability_score(mem, hw.effective_memory_gb)
    speed = _speed_score(candidate, hw)
    quality = _quality_score(candidate)
    context = _context_score(candidate)
    comp = _composite(stability, speed, quality, context, weights)

    return ScoredConfig(
        candidate=candidate,
        memory=mem,
        fit_score=1.0,
        stability_score=stability,
        speed_score=speed,
        quality_score=quality,
        composite=comp,
    )


def _deduplicate(scored: list[ScoredConfig], keep: int = 10) -> list[ScoredConfig]:
    """
    Keep top-K results while ensuring model variety:
    at most 2 configs per (model_id, quant) pair so we see different options.
    """
    seen: dict[str, int] = {}
    out: list[ScoredConfig] = []
    for sc in scored:
        key = f"{sc.candidate.model.id}:{sc.candidate.quant}"
        if seen.get(key, 0) >= 2:
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(sc)
        if len(out) >= keep:
            break
    return out


def generate_recommendations(
    hw: HardwareProfile,
    modes: Optional[list[str]] = None,
    top_n: int = 3,
) -> dict[str, Recommendation]:
    """
    Generate recommendations for each requested mode.

    Returns a dict of mode → Recommendation (primary + alternatives).
    """
    if modes is None:
        modes = list(MODE_WEIGHTS.keys())

    candidates = _enumerate_candidates(hw)
    recommendations: dict[str, Recommendation] = {}

    for mode in modes:
        weights = MODE_WEIGHTS[mode]
        scored: list[ScoredConfig] = []

        for c in candidates:
            sc = _score_candidate(c, hw, weights)
            if sc is not None:
                scored.append(sc)

        if not scored:
            continue

        scored.sort(key=lambda s: s.composite, reverse=True)
        top = _deduplicate(scored, keep=top_n + 5)[:top_n]

        for sc in top:
            sc.rationale = _build_rationale(sc, mode)

        primary = top[0]
        alternatives = top[1:]
        recommendations[mode] = Recommendation(
            mode=mode,
            primary=primary,
            alternatives=alternatives,
        )

    return recommendations
