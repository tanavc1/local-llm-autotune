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

# Per-mode context ceiling: prevents tiny models from gaming context bonuses
# in fast/balanced modes where large context is rarely needed.
MODE_MAX_CONTEXT: dict[str, int] = {
    "fastest":      4096,   # short responses, low latency
    "balanced":     16384,  # everyday use; 8k covers most tasks
    "best_quality": 32768,  # let quality models use their full context
}

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
    # context weight kept small (0.05–0.10): large context helps but shouldn't
    # override model quality, especially in best_quality mode where users
    # explicitly want the smartest model, not the one with the most tokens.
    "fastest":      ModeWeights(stability=0.20, speed=0.55, quality=0.18, context=0.07),
    "balanced":     ModeWeights(stability=0.25, speed=0.28, quality=0.37, context=0.10),
    "best_quality": ModeWeights(stability=0.15, speed=0.05, quality=0.75, context=0.05),
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _stability_score(mem: MemoryEstimate, available_gb: float) -> float:
    """
    Score based on memory headroom using a tiered (non-linear) scale.

    The original linear formula (headroom / available) created a perverse
    incentive: a tiny model with 5 GB headroom scored 0.62 stability while
    a large model with 1 GB headroom scored 0.12, causing the recommendation
    engine to systematically prefer tiny models.

    Tiered thresholds better reflect real risk:
      ≥ 25% headroom  → fully safe (1.0)   – OS can page freely
      10–25% headroom → mostly safe (0.6–1.0)
      5–10% headroom  → marginal   (0.3–0.6)
       0–5% headroom  → risky      (0.0–0.3)
    """
    if not mem.fits:
        return 0.0
    frac = mem.headroom_gb / max(available_gb, 0.1)
    if frac >= 0.25:
        return 1.0
    elif frac >= 0.10:
        return 0.60 + (frac - 0.10) / 0.15 * 0.40  # 0.60 → 1.0
    elif frac >= 0.05:
        return 0.30 + (frac - 0.05) / 0.05 * 0.30  # 0.30 → 0.60
    else:
        return frac / 0.05 * 0.30                   # 0.00 → 0.30


def _speed_score(
    candidate: CandidateConfig,
    hw: HardwareProfile,
) -> float:
    """
    Estimated relative throughput normalised to 0–1.

    Factors:
      - Quantization speed multiplier (higher compression → fewer memory ops)
      - GPU layer fraction (more layers on GPU = faster Metal execution)
      - CPU core count for CPU-only inference
      - Model parameter count: larger models are slower regardless of quant.
        A 14B Q2_K model is NOT faster than a 7B Q4_K_M — it still has 2×
        the parameters to compute. Not accounting for this caused the engine
        to incorrectly prefer large Q2_K models in fastest mode.

    Size normalization: 7B is the community reference for "mainstream fast".
    A 14B model at any quant is roughly 2× slower than a 7B at the same quant.
    We use a mild log-inverse scale so the penalty grows gradually with size.
    """
    q_spec: QuantizationSpec = QUANTIZATIONS[candidate.quant]
    quant_factor = q_spec.speed_multiplier / QUANTIZATIONS["F16"].speed_multiplier

    if hw.has_gpu and candidate.gpu_layer_fraction > 0:
        gpu_boost = 1.0 + candidate.gpu_layer_fraction * 3.0  # up to 4× over CPU
    else:
        gpu_boost = min(1.0, hw.cpu.physical_cores / 16.0)

    # Size penalty: throughput ∝ 1 / params_b (more params = more FLOPs per token)
    # Reference = 7B. Cap at 2× bonus for tiny models, floor at ~0.1× for 70B.
    # Uses square-root to soften the penalty (avoids making large models useless).
    params_b = candidate.model.parameters_b
    size_factor = min(2.0, math.sqrt(7.0 / max(params_b, 0.5)))

    raw = quant_factor * gpu_boost * size_factor
    # Maximum: quant_factor=3.2 (Q2_K), gpu_boost=4.0, size_factor=2.0 → 25.6
    return min(1.0, raw / 25.6)


def _quality_score(candidate: CandidateConfig) -> float:
    """
    Model quality as a blend of real benchmark data and quantization fidelity.

    When bench_mmlu is available (all registry models), use a chance-corrected
    MMLU score as the primary quality signal — this reflects what the model
    actually knows, not just how many parameters it has.

    Chance-corrected MMLU: (mmlu - 0.25) / 0.75
      0.25 = random baseline (4-choice), 0.75 = range to perfect score
      → llama-3.2-1b (32.8%) → 0.104   (very weak general knowledge)
      → llama-3.1-8b (66.7%) → 0.556   (solid mainstream model)
      → qwen2.5-14b  (79.8%) → 0.731   (strong model)
      → qwen2.5-72b  (86.5%) → 0.820   (near-frontier)

    Quantization penalty:
      Aggressive quants (Q2_K, Q3_K_M) on small models cause disproportionate
      quality loss — this is what causes "always wrong" and repetition loops.
      We apply an extra penalty to quants below Q4_K_S on models < 14B.
    """
    # Primary quality signal: real benchmarks when available
    bench = candidate.model.bench_mmlu
    if bench is not None:
        base_score = max(0.0, min(1.0, (bench - 0.25) / 0.75))
    else:
        # Fallback: log-linear parameter estimate
        params_b = candidate.model.parameters_b
        base_score = min(1.0, math.log(max(1.0, params_b)) / math.log(70))

    # Quantization fidelity (0–1 from registry)
    quant_fidelity = QUANTIZATIONS[candidate.quant].quality_score

    # Extra penalty for aggressive quants on models where they destroy quality.
    # Q2_K (fidelity=0.50) on models under ~34B causes unintelligible output —
    # this is the root cause of the "always wrong + repetition" user report.
    # Q3_K_M is similarly risky on models under 14B.
    # Only very large models (≥ 34B) can tolerate Q2_K with acceptable output.
    params_b = candidate.model.parameters_b
    if quant_fidelity < 0.75 and params_b < 34.0:
        # Penalty scales with aggression: Q2_K (0.50) → -0.20 penalty
        # Q4_K_S (0.78) → no penalty (above threshold)
        extra_penalty = max(0.0, (0.75 - quant_fidelity) / 0.75) * 0.20
        quant_fidelity = max(0.05, quant_fidelity - extra_penalty)

    return 0.60 * base_score + 0.40 * quant_fidelity


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

    # Include real benchmark data when available
    bench_note = ""
    if c.model.bench_mmlu is not None:
        bench_note = f" MMLU {c.model.bench_mmlu:.1%}"
        if c.model.bench_humaneval is not None:
            bench_note += f", HumanEval {c.model.bench_humaneval:.1%}"

    # Warn about aggressive quants that sacrifice quality
    quant_warn = ""
    q_fidelity = QUANTIZATIONS[c.quant].quality_score
    if q_fidelity < 0.75:
        quant_warn = f" [NOTE: {c.quant} reduces quality ~{(1-q_fidelity)*100:.0f}% vs F16 — acceptable only when memory is the constraint]"

    return (
        f"{c.model.name} @ {c.quant} ({quant_desc})."
        f"{bench_note}. "
        f"Context {c.context_len} tokens, {gpu_note}. "
        f"Uses {mem.peak_gb:.2f} GB of {mem.peak_gb + mem.headroom_gb:.2f} GB "
        f"available ({mem.headroom_gb:.2f} GB headroom). "
        f"Mode: {mode}.{quant_warn}"
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
        ctx_cap = MODE_MAX_CONTEXT.get(mode, 32768)
        scored: list[ScoredConfig] = []

        for c in candidates:
            # Skip context lengths that don't make sense for this mode.
            # This prevents tiny 1B models from winning best_quality by using
            # a 32768 context they don't need, inflating context + stability scores.
            if c.context_len > ctx_cap:
                continue
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
