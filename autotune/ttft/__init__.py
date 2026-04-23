"""
autotune.ttft — Time To First Token optimisation layer
=======================================================

This package owns every decision that affects TTFT (time to first token).
Nothing outside this package should make TTFT-affecting changes to an Ollama
request — route all such decisions through :class:`TTFTOptimizer`.

Three mechanisms, used together
--------------------------------
1. **Dynamic ``num_ctx``** — Ollama allocates the full KV cache before generating
   a single token.  If ``num_ctx=8192`` it allocates 8 K token slots even if
   your message is 50 tokens.  autotune sizes ``num_ctx`` to exactly what the
   request needs::

       num_ctx = clamp(input_tokens + max_new_tokens + 256, 512, profile_max)

   For a 60-token question on the balanced profile (max 8 192):
     raw Ollama  → num_ctx = 4 096  (its own fixed default)
     autotune    → num_ctx = 1 340  (3× smaller KV allocation)

2. **``keep_alive = -1``** — Ollama's default ``keep_alive`` is 5 minutes.
   After 5 minutes idle the model is fully unloaded from unified memory.
   The next request pays a 1–4 s reload penalty (longer for larger models).
   autotune always passes ``keep_alive = "-1"`` so the model stays resident.

3. **``num_keep`` prefix caching** — If the first message(s) are a system
   prompt, autotune pins those tokens in Ollama's KV cache via ``num_keep``.
   Every subsequent turn skips re-evaluating the system prompt entirely.

Benchmark proof (2026-04-08, qwen3:8b, 63 calls, 18 prompts):
  Average across all scenarios:        raw 626 ms  →  autotune 349 ms  (−44%)
  Large-context prompts (>1 k tokens): raw 2 015 ms → autotune 261 ms  (−87%)
  Cold-start / session continuity:     raw 1 227 ms → autotune 244 ms  (−80%)

Public API
----------
.. code-block:: python

    from autotune.ttft import TTFTOptimizer

    optimizer = TTFTOptimizer()
    ollama_opts = optimizer.build_request_options(messages, profile)
    # Pass ollama_opts["options"] to Ollama and ollama_opts["keep_alive"] top-level.
"""

from autotune.ttft.optimizer import KEEP_ALIVE_FOREVER, TTFTOptimizer

__all__ = ["TTFTOptimizer", "KEEP_ALIVE_FOREVER"]
