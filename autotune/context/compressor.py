"""
Message compressor — reduces token count of individual messages without
destroying the information that makes them useful.

Strategies (applied in order, lightest first)
----------------------------------------------
1. JSON / structured data blobs → compact key summary
2. Long tool / command output   → head + "N lines omitted" + tail
3. Long assistant messages      → first paragraph + code blocks + conclusion
4. Long user messages           → preserve intent, truncate repetition
5. Repeated whitespace / noise  → strip
"""

from __future__ import annotations

import json
import re
from typing import Optional


def _cut_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncate `text` to at most `max_chars`, cutting at the nearest sentence
    boundary before that limit.  Tries (in order):
      1. Last sentence-ending punctuation (. ! ?) followed by whitespace
      2. Last paragraph break (\n\n or \n)
      3. Last whitespace (word boundary)
      4. Hard cut at max_chars as last resort
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Try sentence boundary: '. ', '! ', '? ', '.\n', etc.
    for pat in (r'[.!?]["\']?\s', r'\n\n', r'\n', r'\s'):
        for m in reversed(list(re.finditer(pat, window))):
            end = m.end()
            if end > max_chars * 0.5:   # don't cut too early
                return window[:end].rstrip()
    return window


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# JSON blob: object or array with at least 120 chars of content
_JSON_BLOB_RE = re.compile(
    r'(\{[^{}]{120,}\}|\[[^\[\]]{120,}\])',
    re.DOTALL,
)

# Fenced code blocks
_CODE_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

# Lines that look like log / CLI output (timestamps, PIDs, hex addresses)
_LOG_LINE_RE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|"
    r"\[INFO\]|\[DEBUG\]|\[WARN\]|\[ERROR\]|"
    r"0x[0-9a-fA-F]+|pid=\d+)",
    re.IGNORECASE,
)

# Repeated blank lines
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Individual compression helpers
# ---------------------------------------------------------------------------

def _compress_json_blob(blob: str) -> str:
    """Replace a large JSON blob with a compact human-readable summary."""
    try:
        obj = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        # Not valid JSON — truncate at word boundary
        preview = _cut_at_sentence(blob, 80)
        return f"{preview}… [{len(blob) - len(preview)} chars omitted]"

    if isinstance(obj, dict):
        keys = list(obj.keys())
        sample = ", ".join(str(k) for k in keys[:6])
        suffix = "…" if len(keys) > 6 else ""
        return f'{{/* {len(keys)} keys: {sample}{suffix} */}}'
    elif isinstance(obj, list):
        item_type = type(obj[0]).__name__ if obj else "?"
        return f'[/* {len(obj)} × {item_type} */]'
    else:
        return str(obj)[:100]


def compress_json_in_content(content: str) -> str:
    """Replace all large JSON blobs embedded in a message."""
    return _JSON_BLOB_RE.sub(lambda m: _compress_json_blob(m.group(0)), content)


def compress_tool_output(
    content: str,
    head_lines: int = 10,
    tail_lines: int = 5,
    max_total_lines: int = 20,
) -> str:
    """
    Compress long tool / shell output.

    Keeps the first `head_lines` and last `tail_lines`, replacing the middle
    with a single "[N lines omitted]" annotation.
    """
    lines = content.splitlines()
    if len(lines) <= max_total_lines:
        return content

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    omitted = len(lines) - head_lines - tail_lines

    return (
        "\n".join(head)
        + f"\n… [{omitted} lines omitted] …\n"
        + "\n".join(tail)
    )


def compress_assistant_message(
    content: str,
    max_chars: int = 2000,
    aggressive: bool = False,
) -> str:
    """
    Shorten a long assistant reply.

    Strategy:
      1. Extract all fenced code blocks (always preserved — they are high-signal).
      2. Split prose into paragraphs.
      3. Keep: first paragraph (context/summary) + up to 2 code blocks + last paragraph.
      4. Mark how many original chars were omitted.

    When aggressive=True: max_chars is halved and only 1 code block is kept.
    """
    if len(content) <= max_chars:
        return content

    if aggressive:
        max_chars = max_chars // 2

    # Extract code blocks, preserving their language tags
    code_blocks: list[str] = []
    def _capture(m: re.Match) -> str:
        lang = m.group(1)
        body = m.group(2)
        code_blocks.append(f"```{lang}\n{body}```")
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    prose = _CODE_FENCE_RE.sub(_capture, content)

    # Split prose into paragraphs
    paragraphs = [p.strip() for p in prose.split("\n\n") if p.strip()]

    kept_parts: list[str] = []
    chars_used = 0

    # Always keep first paragraph
    if paragraphs:
        kept_parts.append(paragraphs[0])
        chars_used += len(paragraphs[0])

    # Inject code blocks (most signal per token)
    max_code = 1 if aggressive else 2
    for i, cb in enumerate(code_blocks[:max_code]):
        if chars_used + len(cb) < max_chars:
            kept_parts.append(cb)
            chars_used += len(cb)

    # Keep last paragraph if it's different from first (conclusion)
    if len(paragraphs) > 1:
        last = paragraphs[-1]
        if last != paragraphs[0] and chars_used + len(last) < max_chars:
            kept_parts.append(last)
            chars_used += len(last)

    result = "\n\n".join(kept_parts)

    # Re-insert any remaining code blocks referenced by placeholder
    for i, cb in enumerate(code_blocks):
        result = result.replace(f"__CODE_BLOCK_{i}__", cb)

    # If the first paragraph was itself longer than budget, cut it at a sentence
    # boundary rather than mid-word — ensures no sentence fragments in output.
    if len(result) > max_chars:
        result = _cut_at_sentence(result, max_chars)

    original_len = len(content)
    if original_len > len(result) + 50:
        result += f"\n\n[…{original_len - len(result)} chars from original omitted]"

    return result


def _strip_noise(content: str) -> str:
    """Strip repeated blank lines and trailing whitespace."""
    content = _MULTI_BLANK_RE.sub("\n\n", content)
    return content.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compress_message(
    role: str,
    content: str,
    aggressive: bool = False,
) -> str:
    """
    Compress a single message to reduce its token count.

    Parameters
    ----------
    role       : "user" | "assistant" | "tool" | "system"
    content    : raw message content
    aggressive : True in EMERGENCY tier — more aggressive reduction
    """
    if not content:
        return content

    # Step 1: noise removal (always)
    content = _strip_noise(content)

    # Step 2: compress embedded JSON blobs (always)
    content = compress_json_in_content(content)

    # Step 3: role-specific compression
    if role == "tool":
        content = compress_tool_output(
            content,
            head_lines=8 if aggressive else 12,
            tail_lines=4 if aggressive else 6,
            max_total_lines=15 if aggressive else 22,
        )
        return content

    if role == "assistant":
        max_chars = 800 if aggressive else 2000
        content = compress_assistant_message(content, max_chars=max_chars, aggressive=aggressive)
        return content

    if role == "user" and aggressive:
        # Preserve the intent (first ~600 chars) but trim repetition.
        # Cut at a sentence boundary so we never split mid-thought.
        max_chars = 600
        if len(content) > max_chars:
            cut = _cut_at_sentence(content, max_chars)
            content = cut + f"\n[…{len(content) - len(cut)} chars omitted]"
        return content

    return content
