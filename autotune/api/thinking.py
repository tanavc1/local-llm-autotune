"""
Thinking-tag utilities for reasoning models (qwen3, deepseek-r1, qwq, …).

These models wrap their chain-of-thought in <think>…</think> before writing
their actual answer.  This module provides:

  - _strip_thinking(text)           — for already-collected text (non-streaming)
  - ThinkingStreamFilter            — stateful per-chunk filter for streaming
  - is_thinking_model(model_id)     — True when the model uses think tags
  - THINKING_OVERHEAD               — extra tokens to allocate for the think block

Import pattern (avoids circular deps):
    from autotune.api.thinking import strip_thinking, ThinkingStreamFilter, is_thinking_model
"""

from __future__ import annotations

import re
from typing import AsyncGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THINK_OPEN  = "<think>"
_THINK_CLOSE = "</think>"

# Matches a complete block OR an incomplete one (model cut off mid-think).
_RE_THINK = re.compile(r"<think>.*?</think>|<think>.*$", re.DOTALL)

# Models that emit a <think>…</think> block before their answer.
_THINKING_MODELS = ("qwen3", "deepseek-r1", "qwq", "deepthink", "marco-o1")

# Extra tokens granted for the think block so it doesn't eat the caller's budget.
THINKING_OVERHEAD = 1024


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_thinking_model(model_id: str) -> bool:
    """Return True if this model emits <think> blocks."""
    lower = model_id.lower()
    return any(name in lower for name in _THINKING_MODELS)


def strip_thinking(text: str) -> str:
    """Remove all <think>…</think> blocks from already-collected text."""
    return _RE_THINK.sub("", text).lstrip()


class ThinkingStreamFilter:
    """
    Stateful per-chunk filter that strips <think>…</think> from streaming text.

    Designed for token-by-token streaming where the open/close tags may be
    split across chunk boundaries.

    Usage::

        filt = ThinkingStreamFilter()
        for chunk_text in stream:
            visible = filt.feed(chunk_text)
            if visible:
                print(visible, end="", flush=True)
        # full clean text collected:
        full_answer = filt.collected_text()

    ``feed()`` returns the visible portion of each chunk (empty string means
    the chunk was entirely inside a think block).  ``collected_text()`` returns
    the concatenation of all visible portions — use this instead of joining
    the raw stream when you need the final text for storage.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._visible: list[str] = []

    def feed(self, text: str) -> str:
        """Filter one chunk of text; returns the visible portion."""
        buf = text
        out_parts: list[str] = []

        while buf:
            if self._in_think:
                pos = buf.find(_THINK_CLOSE)
                if pos == -1:
                    buf = ""          # still inside think — discard
                else:
                    buf = buf[pos + len(_THINK_CLOSE):].lstrip("\n")
                    self._in_think = False
            else:
                pos = buf.find(_THINK_OPEN)
                if pos == -1:
                    out_parts.append(buf)
                    buf = ""
                else:
                    if pos > 0:
                        out_parts.append(buf[:pos])
                    buf = buf[pos + len(_THINK_OPEN):]
                    self._in_think = True

        visible = "".join(out_parts)
        if visible:
            self._visible.append(visible)
        return visible

    def collected_text(self) -> str:
        """Return the full visible text seen so far (think blocks excluded)."""
        return "".join(self._visible)


async def filter_thinking_sse(
    source: AsyncGenerator[bytes, None],
) -> AsyncGenerator[bytes, None]:
    """
    Async generator that strips <think>…</think> from SSE byte chunks.

    Each chunk is a ``data: {…}\\n\\n`` JSON payload from the OpenAI streaming
    format.  The filter mutates the ``delta.content`` field and re-serialises.
    Finish-reason and [DONE] chunks are always passed through unchanged.
    """
    import json

    in_think = False

    async for raw_chunk in source:
        if raw_chunk == b"data: [DONE]\n\n":
            yield raw_chunk
            continue

        decoded = raw_chunk.decode("utf-8", errors="replace")
        if not decoded.startswith("data: "):
            yield raw_chunk
            continue

        try:
            payload = json.loads(decoded[6:])
        except json.JSONDecodeError:
            yield raw_chunk
            continue

        choices = payload.get("choices", [])
        if not choices:
            yield raw_chunk
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content")
        finish = choices[0].get("finish_reason")

        if not content:
            yield raw_chunk
            continue

        # Run the chunk through the state machine
        buf = content
        out_parts: list[str] = []

        while buf:
            if in_think:
                pos = buf.find(_THINK_CLOSE)
                if pos == -1:
                    buf = ""
                else:
                    buf = buf[pos + len(_THINK_CLOSE):].lstrip("\n")
                    in_think = False
            else:
                pos = buf.find(_THINK_OPEN)
                if pos == -1:
                    out_parts.append(buf)
                    buf = ""
                else:
                    if pos > 0:
                        out_parts.append(buf[:pos])
                    buf = buf[pos + len(_THINK_OPEN):]
                    in_think = True

        filtered = "".join(out_parts)

        if not filtered and not finish:
            continue  # entire chunk was thinking — drop it

        if filtered:
            delta["content"] = filtered
        else:
            delta.pop("content", None)

        choices[0]["delta"] = delta
        payload["choices"] = choices
        yield (f"data: {json.dumps(payload)}\n\n").encode()
