"""
Message value classifier — scores each conversation turn for informational worth.

Used by the context window manager to decide which messages to drop first when
the token budget is under pressure.

Score range: 0.0 (pure chatter) → 1.0 (dense technical content)
"""

from __future__ import annotations

import re
from enum import Enum


class MessageValue(Enum):
    HIGH   = "high"    # code, specific facts, detailed technical content
    MEDIUM = "medium"  # explanatory content, context-setting
    LOW    = "low"     # chatter, greetings, trivial acknowledgments
    TOOL   = "tool"    # tool/command output — always compressible


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Pure chatter — matches the whole (stripped) content
_CHATTER_RE = re.compile(
    r"^(ok|okay|sure|thanks|thank you|ty|thx|great|got it|sounds good|"
    r"perfect|cool|nice|alright|yep|yes|no|nope|understood|i see|"
    r"i understand|makes sense|will do|noted|done|lgtm|"
    r"hi|hello|hey|sup|bye|goodbye|see you|later|cya|"
    r"sounds good to me|that works|let's do it|go ahead|"
    r"good point|fair enough|right|exactly|agreed)\W*$",
    re.IGNORECASE,
)

# High-value signal patterns
_CODE_FENCE_RE   = re.compile(r"```")
_INLINE_CODE_RE  = re.compile(r"`[^`]+`")
_STACK_TRACE_RE  = re.compile(r"(Traceback|Error:|Exception:|at line \d+)", re.I)
_NUMBERS_UNITS_RE = re.compile(
    r"\b\d+\.?\d*\s*(GB|MB|KB|ms|sec|s\b|tok|tokens|%|k\b|M\b)\b",
    re.IGNORECASE,
)
_TECHNICAL_RE = re.compile(
    r"\b(error|exception|function|class|method|import|def |return|async|await|"
    r"SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|JOIN|"
    r"token|model|inference|RAM|VRAM|CPU|GPU|memory|latency|throughput|"
    r"config|parameter|setting|result|output|response|request|"
    r"install|deploy|migrate|refactor|optimize|debug|fix|"
    r"API|REST|HTTP|JSON|YAML|SQL|bash|shell|docker|kubernetes)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"\?")
_BULLET_LIST_RE = re.compile(r"^[\s]*[-*•]\s", re.MULTILINE)
_NUMBERED_LIST_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)
_URL_RE = re.compile(r"https?://\S+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_message(role: str, content: str) -> tuple[MessageValue, float]:
    """
    Score a message for informational value.

    Returns
    -------
    (MessageValue enum, float 0.0–1.0)
        Higher score = more valuable = preserved longer under pressure.
    """
    if not content:
        return MessageValue.LOW, 0.0

    stripped = content.strip()

    # Tool outputs: always medium value (compressible but not droppable)
    if role == "tool":
        return MessageValue.TOOL, 0.5

    # Very short chatter check
    if len(stripped) < 60 and _CHATTER_RE.match(stripped):
        return MessageValue.LOW, 0.05

    score = 0.30   # baseline

    # ── Code content ────────────────────────────────────────────────────
    fence_count  = len(_CODE_FENCE_RE.findall(content))
    inline_count = len(_INLINE_CODE_RE.findall(content))
    if fence_count >= 2:
        score += 0.35     # full code block
    elif fence_count == 1:
        score += 0.15
    score += min(0.10, inline_count * 0.02)

    # ── Stack traces / error messages (always high value) ───────────────
    if _STACK_TRACE_RE.search(content):
        score += 0.30

    # ── Technical vocabulary ─────────────────────────────────────────────
    tech_hits = len(_TECHNICAL_RE.findall(content))
    score += min(0.25, tech_hits * 0.025)

    # ── Specific numbers / measurements ──────────────────────────────────
    num_hits = len(_NUMBERS_UNITS_RE.findall(content))
    score += min(0.15, num_hits * 0.04)

    # ── Lists (structured information) ───────────────────────────────────
    bullet_hits   = len(_BULLET_LIST_RE.findall(content))
    numbered_hits = len(_NUMBERED_LIST_RE.findall(content))
    score += min(0.10, (bullet_hits + numbered_hits) * 0.03)

    # ── URLs (references) ────────────────────────────────────────────────
    if _URL_RE.search(content):
        score += 0.05

    # ── Questions carry intent ───────────────────────────────────────────
    if role == "user" and _QUESTION_RE.search(content):
        score += 0.08

    # ── Length bonus — longer messages carry more information ─────────────
    n = len(content)
    if n > 1000:
        score += 0.15
    elif n > 400:
        score += 0.08
    elif n > 100:
        score += 0.03
    elif n < 20:
        score -= 0.15   # very short with no other signal → likely chatter

    score = max(0.0, min(1.0, score))

    if score >= 0.65:
        return MessageValue.HIGH, score
    elif score >= 0.30:
        return MessageValue.MEDIUM, score
    else:
        return MessageValue.LOW, score


def is_droppable(role: str, content: str, threshold: float) -> bool:
    """Return True if this message should be dropped at the given threshold."""
    _, score = score_message(role, content)
    return score < threshold
