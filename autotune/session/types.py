"""Shared data types for the live session controller."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SessionState(Enum):
    OPTIMAL           = "optimal"
    WARNING           = "warning"
    ACTION_NEEDED     = "action_needed"
    DEGRADING         = "degrading"
    STABLE_RECOVERING = "stable_recovering"
    CRITICAL          = "critical"


class ThermalState(Enum):
    NOMINAL   = "nominal"
    WARM      = "warm"
    WARNING   = "warning"
    THROTTLING = "throttling"
    CRITICAL  = "critical"


@dataclass
class LLMProcess:
    pid: int
    name: str
    ram_gb: float
    cpu_percent: float
    runtime: str            # "ollama", "llama.cpp", "mlx", "lm_studio", "unknown"
    model_hint: str         # guessed model name from cmdline
    cmdline_snippet: str


@dataclass
class OllamaModel:
    name: str
    size_gb: float
    context_len: int
    vram_gb: float


@dataclass
class LiveMetrics:
    timestamp: float

    # RAM
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float

    # Swap
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float

    # GPU / Unified memory (None when unavailable)
    vram_total_gb: Optional[float]
    vram_used_gb: Optional[float]
    vram_percent: Optional[float]

    # CPU
    cpu_percent: float                      # overall
    cpu_per_core: list[float]

    # Thermals
    cpu_temp_c: Optional[float]
    gpu_temp_c: Optional[float]
    thermal_state: ThermalState
    cpu_speed_limit_pct: int                # 100 = no throttle

    # Detected LLM activity
    llm_processes: list[LLMProcess]
    ollama_models: list[OllamaModel]

    # Performance (set externally or via Ollama API)
    tokens_per_sec: Optional[float]         # prompt eval tok/s
    gen_tokens_per_sec: Optional[float]     # generation tok/s
    ttft_ms: Optional[float]
    queue_depth: int

    # Derived rates (set by monitor after accumulation)
    swap_growth_mb_per_min: float = 0.0
    ram_growth_mb_per_min: float = 0.0
    vram_growth_mb_per_min: float = 0.0


@dataclass
class SessionConfig:
    """Current inference configuration (may be advisory if we can't control the runtime)."""
    model_id: str
    model_name: str
    quant: str
    context_len: int
    n_gpu_layers: int
    n_total_layers: int
    backend: str                # "metal", "cuda", "rocm", "cpu"
    kv_cache_precision: str     # "f16", "q8", "q4"
    speculative_decoding: bool
    concurrency: int
    prompt_caching: bool

    # Derived limits
    weight_gb: float
    kv_cache_gb: float          # at current context
    total_budget_gb: float      # effective memory available

    def effective_memory_gb(self) -> float:
        return self.weight_gb + self.kv_cache_gb


@dataclass
class AdvisorDecision:
    action: str                 # human-readable action key
    reason: str
    severity: SessionState
    suggested_changes: dict     # {"context_len": 2048, ...}
    revertible: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionEvent:
    timestamp: float
    level: str                  # "INFO", "WARN", "ACTION", "CRITICAL", "OK"
    message: str

    def age_str(self) -> str:
        import time as _time
        delta = _time.time() - self.timestamp
        if delta < 60:
            return f"{int(delta)}s ago"
        return f"{int(delta/60)}m ago"
