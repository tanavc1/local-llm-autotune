"""
Optimization profiles: fast / balanced / quality.

Each profile controls inference parameters, context limits, hardware priority,
and backend preference.  Genuine differences — not just labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    name: str
    label: str

    # Generation parameters
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float

    # Context budget (tokens)
    max_context_tokens: int         # hard cap including history + system prompt
    system_prompt_cache: bool       # keep system prompt in KV (don't resend)

    # Hardware / runtime
    qos_class: str                  # "USER_INTERACTIVE" | "USER_INITIATED" | "DEFAULT"
    preferred_quants: list[str]     # preferred quantizations in order
    kv_cache_precision: str         # "f16" | "q8" | "q4"
    disable_gc_during_inference: bool
    speculative_decoding: bool

    # Backend preference order (first available wins)
    backend_preference: list[str]   # "ollama" | "lmstudio" | "hf_api"

    # Ollama-specific
    ollama_keep_alive: str          # "-1" = forever, "5m" = 5 minutes

    # Rate / concurrency
    request_timeout_sec: float
    description: str


PROFILES: dict[str, Profile] = {
    "fast": Profile(
        name="fast",
        label="⚡ Fast",
        max_new_tokens=384,
        temperature=0.05,
        top_p=0.9,
        repetition_penalty=1.0,
        max_context_tokens=2048,
        system_prompt_cache=True,
        qos_class="USER_INTERACTIVE",
        preferred_quants=["Q4_K_S", "Q4_K_M", "Q3_K_M", "Q2_K"],
        kv_cache_precision="q8",
        disable_gc_during_inference=True,
        speculative_decoding=False,
        backend_preference=["mlx", "ollama", "lmstudio", "hf_api"],
        ollama_keep_alive="-1",
        request_timeout_sec=30.0,
        description="Lowest latency. Short context, greedy decoding, high priority scheduling.",
    ),

    "balanced": Profile(
        name="balanced",
        label="⚖  Balanced",
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.05,
        max_context_tokens=8192,
        system_prompt_cache=True,
        qos_class="USER_INITIATED",
        preferred_quants=["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K"],
        kv_cache_precision="f16",
        disable_gc_during_inference=True,
        speculative_decoding=False,
        backend_preference=["mlx", "ollama", "lmstudio", "hf_api"],
        ollama_keep_alive="-1",
        request_timeout_sec=120.0,
        description="Good balance of speed and quality for everyday use.",
    ),

    "quality": Profile(
        name="quality",
        label="✨ Quality",
        max_new_tokens=4096,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.05,
        max_context_tokens=32768,
        system_prompt_cache=True,
        qos_class="USER_INITIATED",
        preferred_quants=["Q5_K_M", "Q6_K", "Q4_K_M", "Q8_0"],
        kv_cache_precision="f16",
        disable_gc_during_inference=False,
        speculative_decoding=False,
        backend_preference=["mlx", "ollama", "lmstudio", "hf_api"],
        ollama_keep_alive="-1",
        request_timeout_sec=300.0,
        description="Maximum output quality. Longer context, better sampling, no compromises.",
    ),
}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}. Choose: {list(PROFILES)}")
    return PROFILES[name]
