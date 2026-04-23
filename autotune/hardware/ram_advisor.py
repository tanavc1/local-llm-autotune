"""RAM pressure advisor: maps closeable processes → newly unlockable LLM models."""

from __future__ import annotations

from dataclasses import dataclass, field

from autotune.hardware.profiler import ProcessInfo
from autotune.memory.estimator import RUNTIME_OVERHEAD_GB, SAFETY_MARGIN_FRACTION
from autotune.models.registry import MODEL_REGISTRY, ModelProfile

# Quantizations to probe, in preference order (best quality first)
_PROBE_QUANTS = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K"]
# Context length to assume for sizing (conservative 4096 tokens)
_PROBE_CTX = 4096


@dataclass
class UnlockSuggestion:
    """A single model that becomes runnable after a specific closure."""
    model: ModelProfile
    quant: str
    required_gb: float


@dataclass
class UnlockGroup:
    """A set of apps to close, plus every model that becomes available as a result.

    All models in the group require the same (or a subset of the same) closure
    action, so the process list is shown once and the models are listed beneath.
    """
    processes: list[ProcessInfo]       # close these apps
    freed_gb: float                    # total RAM freed
    available_after_gb: float          # available_now + freed_gb
    models: list[UnlockSuggestion]     # sorted by required_gb ascending


def _min_required_gb(model: ModelProfile, quant: str) -> float:
    """Minimum RAM needed to load a model (4 k ctx, including safety pad)."""
    weights = model.weight_gb(quant)
    kv = model.kv_cache_gb(_PROBE_CTX)
    overhead = RUNTIME_OVERHEAD_GB
    raw = weights + kv + overhead
    return raw * (1 + SAFETY_MARGIN_FRACTION)


def _proc_key(procs: list[ProcessInfo]) -> frozenset[int]:
    """Stable identity key for a list of processes (by PID)."""
    return frozenset(p.pid for p in procs)


def compute_unlock_suggestions(
    available_gb: float,
    hogs: list[ProcessInfo],
    *,
    max_models: int = 6,
    min_gain_gb: float = 0.5,
) -> list[UnlockGroup]:
    """
    Find models that are just out of reach and suggest which apps to close.

    Returns UnlockGroups — each group has a unique closure action (set of
    processes to close) and all models that would become runnable after it.
    This avoids repeating the same 'close X' advice for every model.
    """
    closeable = [p for p in hogs if p.is_closeable]
    if not closeable:
        return []

    # Build candidates: models that don't currently fit but could after some closure
    candidates: list[tuple[float, ModelProfile, str]] = []
    for model in MODEL_REGISTRY.values():
        for quant in _PROBE_QUANTS:
            if quant not in model.quantization_options:
                continue
            needed = _min_required_gb(model, quant)
            if needed <= available_gb:
                break  # already fits at this quant — skip model entirely
            candidates.append((needed, model, quant))
            break  # only cheapest quant per model

    # Sort by RAM need ascending: easiest wins first
    candidates.sort(key=lambda x: x[0])

    # For each candidate, greedily pick the fewest highest-RAM apps to close
    # Group by the resulting closure set (same PID set → same panel)
    groups: dict[frozenset[int], UnlockGroup] = {}
    seen_models: set[str] = set()
    total_models = 0

    for needed, model, quant in candidates:
        if model.id in seen_models or total_models >= max_models:
            continue
        shortfall = needed - available_gb
        if shortfall < min_gain_gb:
            continue

        chosen: list[ProcessInfo] = []
        freed = 0.0
        for proc in closeable:
            if freed >= shortfall:
                break
            chosen.append(proc)
            freed += proc.rss_gb

        if freed < shortfall or len(chosen) > 4:
            continue  # not achievable with ≤ 4 closures

        key = _proc_key(chosen)
        if key not in groups:
            groups[key] = UnlockGroup(
                processes=chosen,
                freed_gb=freed,
                available_after_gb=available_gb + freed,
                models=[],
            )
        groups[key].models.append(UnlockSuggestion(
            model=model,
            quant=quant,
            required_gb=needed,
        ))
        seen_models.add(model.id)
        total_models += 1

    # Return groups sorted by freed_gb ascending (smallest ask first)
    return sorted(groups.values(), key=lambda g: g.freed_gb)
