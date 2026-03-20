"""
Context window utilities.

Single source of truth for:
  - Token estimation (char/4 heuristic, consistent everywhere)
  - Dynamic num_ctx computation

The most impactful optimisation for both RAM pressure and TTFT is sizing
the KV cache to what the request actually needs rather than blindly using
the profile's maximum.  A 500-token prompt + 1024-token output only needs
~1800 tokens of KV cache, not the 8192 the balanced profile allows.

KV cache memory scales linearly with num_ctx:
  KV bytes = 2 * n_layers * n_kv_heads * head_dim * num_ctx * bytes_per_kv_elem

For qwen2.5-coder:14b (48L, 8KV, 128 head_dim) at f16:
  num_ctx=8192  → 805 MB
  num_ctx=1728  → 170 MB   (5× less for a typical 500-tok prompt)
  num_ctx=2048  → 201 MB   (fast profile with dynamic sizing)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autotune.api.profiles import Profile


def estimate_tokens(text: str) -> int:
    """Estimate token count using the char/4 heuristic. Returns at least 1."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Sum token estimates across a list of OpenAI-style message dicts."""
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def compute_num_ctx(messages: list[dict], profile: "Profile") -> int:
    """
    Return the minimum num_ctx that can fit this request.

    Algorithm:
        needed = input_tokens + profile.max_new_tokens + 256 (safety buffer)
        num_ctx = clamp(needed, 512, profile.max_context_tokens)

    Why this matters:
        Ollama allocates the full KV cache at request start.  Asking for more
        context than the conversation needs wastes unified memory and adds
        latency to the first-token KV-cache initialisation step.

    For multi-turn conversations the messages list already contains history,
    so the estimate naturally grows as the conversation grows — no special
    handling needed.
    """
    input_tokens = estimate_messages_tokens(messages)
    needed = input_tokens + profile.max_new_tokens + 256
    # Clamp: 512 minimum (Ollama needs room for generation), profile max as ceiling
    return max(512, min(needed, profile.max_context_tokens))


def ollama_options_for_profile(
    messages: list[dict],
    profile: "Profile",
) -> dict:
    """
    Build the base Ollama `options` dict for a given profile and message list.

    Includes:
      num_ctx   — dynamically sized to this request (reduces RAM + TTFT)
      f16_kv    — KV cache precision:
                    fast profile (kv_cache_precision="q8") → f16_kv=False  (Q8 KV, half the memory)
                    balanced/quality                        → f16_kv=True   (F16 KV, better quality)

    For full options including num_keep (prefix caching) and adaptive
    memory-pressure reduction, use autotune.api.kv_manager.build_ollama_options().
    """
    options: dict = {
        "num_ctx": compute_num_ctx(messages, profile),
    }
    # Map profile KV precision to Ollama's f16_kv flag.
    # f16_kv=False → Ollama uses Q8_0 for KV cache (2× less memory than F16).
    if profile.kv_cache_precision == "q8":
        options["f16_kv"] = False
    else:
        options["f16_kv"] = True
    return options
