"""
autotune.api — Inference pipeline: profiles, server, chat, backends, KV management.

Package map
-----------
profiles.py         Profile dataclass (fast / balanced / quality) — controls every
                    inference parameter except TTFT-specific options.
                    TTFT options live in autotune.ttft, not here.

server.py           FastAPI server — OpenAI-compatible /v1 endpoints + FIFO queue.

chat.py             Terminal chat REPL (autotune chat / autotune run commands).

conversation.py     SQLite-backed persistent conversation state.

kv_manager.py       KV cache: ``build_ollama_options`` wires together TTFT options
                    (from autotune.ttft) with legacy callers.  New code should
                    call TTFTOptimizer directly.

ctx_utils.py        Token estimation + ``compute_num_ctx``.  Authoritative token
                    arithmetic — consumed by autotune.ttft.optimizer.

hardware_tuner.py   OS-level tuning: nice level, macOS QOS class, GC disable,
                    Linux CPU governor.  Orthogonal to TTFT — acts on the process
                    scheduler, not the KV cache.

model_selector.py   Pre-flight fit analysis: checks if model + KV cache will fit in
                    available RAM, returns a context ceiling and KV precision hint.

backends/           Concrete inference backends:
    chain.py            BackendChain — tries MLX → Ollama → LM Studio in order.
    ollama_pull.py      Pull models from Ollama registry.
    mlx_backend.py      Apple Silicon MLX-LM integration.
    openai_compat.py    OpenAI SDK compatibility shim.
"""
