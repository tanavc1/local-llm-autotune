"""
autotune.context — intelligent context window management.

Public API
----------
    from autotune.context import ContextWindow, BuiltContext

    cw = ContextWindow(max_ctx_tokens=8192)
    built = cw.build(
        history=messages,            # list[dict] role/content, oldest first
        system_prompt="...",
        new_user_message="...",
        reserved_for_output=1024,
    )
    # built.messages  → optimised list ready to send to Ollama
    # built.tier      → which compression strategy was used
    # built.budget_pct → fraction of token budget consumed
"""

from .window import BuiltContext, ContextWindow

__all__ = ["ContextWindow", "BuiltContext"]
