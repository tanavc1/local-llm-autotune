"""
ContextWindow — the central orchestrator for intelligent context management.

Decision flow
-------------
                       ┌─────────────────────────────────────────────┐
  history + new_msg ──►│         compute effective budget             │
                       │  budget = max_ctx − sys − new_msg − reserve  │
                       └───────────────┬─────────────────────────────┘
                                       │
                           ┌───────────▼────────────┐
                           │   history_tokens/budget │
                           └───────────┬────────────┘
                  ┌────────────────────┼──────────────────────┐
               <55%                55-75%                  75-90%          >90%
            FULL tier        RECENT+FACTS tier         COMPRESSED         EMERGENCY
          all verbatim      recent 8 verbatim          recent 6           recent 4
                           + facts block for older    compressed          max compress
                           (drop chatter from old)    + summary block     + compact summary

Within every non-FULL tier:
  1. Low-value chatter is dropped from older turns first.
  2. Tool outputs are always compressed.
  3. Older turns that survive are replaced by a structured facts/summary block.
  4. The recent window is kept verbatim (RECENT+FACTS) or lightly compressed.

The result is a BuiltContext with .messages ready to send to Ollama and
metadata so callers can log and monitor context health.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from autotune.api.ctx_utils import estimate_tokens

from .budget import (
    DROP_THRESHOLD,
    RECENT_WINDOW,
    BudgetTier,
    classify_budget,
)
from .classifier import score_message
from .compressor import compress_message
from .extractor import ConversationFacts, build_summary_block, extract_facts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BuiltContext:
    """
    The output of ContextWindow.build() — what to send to the model
    plus enough metadata to understand what was done.
    """
    messages:         list[dict]   # ready to pass to Ollama / backend
    tokens_sent:      int          # estimated total tokens in messages
    budget_tokens:    int          # effective token budget (adjusted)
    budget_pct:       float        # tokens_sent / budget_tokens
    tier:             BudgetTier   # compression strategy applied
    turns_total:      int          # total turns in original history
    turns_kept:       int          # turns included verbatim or compressed
    turns_dropped:    int          # turns discarded (low-value chatter)
    turns_summarized: int          # turns replaced by a summary block
    summary_injected: bool         # whether a summary block was added
    facts:            Optional[ConversationFacts] = field(default=None)


# ---------------------------------------------------------------------------
# ContextWindow
# ---------------------------------------------------------------------------

class ContextWindow:
    """
    Intelligently trims and compresses conversation history to fit within
    a token budget while preserving maximum informational value.

    Usage
    -----
    cw = ContextWindow(max_ctx_tokens=profile.max_context_tokens)
    built = cw.build(
        history=db_messages,          # list[dict] role/content, oldest first
        system_prompt="...",          # always first, never trimmed
        new_user_message="...",       # always last, never trimmed
        reserved_for_output=1024,     # tokens to reserve for the model reply
    )
    # Pass built.messages to the backend
    """

    def __init__(self, max_ctx_tokens: int) -> None:
        if max_ctx_tokens < 256:
            raise ValueError(f"max_ctx_tokens must be ≥ 256, got {max_ctx_tokens}")
        self.max_ctx = max_ctx_tokens

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def build(
        self,
        history: list[dict],
        system_prompt: Optional[str] = None,
        new_user_message: Optional[str] = None,
        reserved_for_output: int = 512,
    ) -> BuiltContext:
        """
        Build the optimised messages list.

        Parameters
        ----------
        history            : full conversation history (oldest first).
                             Should NOT include system messages — pass those
                             via system_prompt.  Tool/user/assistant roles OK.
        system_prompt      : system message; always prepended, never trimmed.
        new_user_message   : the incoming user turn; always appended, never trimmed.
                             Pass None if it is already the last entry in history.
        reserved_for_output: tokens to leave free for the model's reply.
                             Use profile.max_new_tokens.
        """
        # ── Compute effective budget ─────────────────────────────────────
        sys_tokens = estimate_tokens(system_prompt or "")
        new_tokens = estimate_tokens(new_user_message or "")
        # 64-token formatting overhead (role labels, JSON scaffolding)
        overhead   = sys_tokens + new_tokens + reserved_for_output + 64
        budget     = max(256, self.max_ctx - overhead)

        # Filter system messages from history — handled separately
        turns = [m for m in history if m.get("role") != "system"]
        turns_total = len(turns)

        # ── Fast path: everything fits in FULL tier ───────────────────────
        history_tokens = sum(estimate_tokens(m.get("content", "")) for m in turns)

        state = classify_budget(history_tokens, budget)

        logger.debug(
            "ContextWindow: %d turns, ~%d tokens, budget=%d (%.0f%%) → tier=%s",
            turns_total, history_tokens, budget, state.pct * 100, state.tier.value,
        )

        if state.tier == BudgetTier.FULL:
            return self._tier_full(
                turns, system_prompt, new_user_message,
                overhead, budget, history_tokens,
            )
        if state.tier == BudgetTier.RECENT_PLUS_FACTS:
            return self._tier_recent_plus_facts(
                turns, system_prompt, new_user_message, overhead, budget,
            )
        if state.tier == BudgetTier.COMPRESSED:
            return self._tier_compressed(
                turns, system_prompt, new_user_message, overhead, budget,
            )
        return self._tier_emergency(
            turns, system_prompt, new_user_message, overhead, budget,
        )

    # ------------------------------------------------------------------ #
    # Tier implementations                                                 #
    # ------------------------------------------------------------------ #

    def _tier_full(
        self,
        turns: list[dict],
        sys_prompt: Optional[str],
        new_msg: Optional[str],
        overhead: int,
        budget: int,
        history_tokens: int,
    ) -> BuiltContext:
        """FULL — send all history verbatim."""
        msgs = self._assemble(sys_prompt, turns, new_msg)
        used = overhead + history_tokens
        return BuiltContext(
            messages=msgs,
            tokens_sent=used,
            budget_tokens=budget,
            # budget_pct = fraction of the TOTAL context window consumed
            budget_pct=round(used / max(self.max_ctx, 1), 3),
            tier=BudgetTier.FULL,
            turns_total=len(turns),
            turns_kept=len(turns),
            turns_dropped=0,
            turns_summarized=0,
            summary_injected=False,
        )

    def _tier_recent_plus_facts(
        self,
        turns: list[dict],
        sys_prompt: Optional[str],
        new_msg: Optional[str],
        overhead: int,
        budget: int,
    ) -> BuiltContext:
        """
        RECENT+FACTS — keep the recent window verbatim; replace older turns with
        a structured facts/summary block (after dropping low-value chatter).
        """
        window       = RECENT_WINDOW[BudgetTier.RECENT_PLUS_FACTS]
        drop_thresh  = DROP_THRESHOLD[BudgetTier.RECENT_PLUS_FACTS]

        recent = turns[-window:]
        older  = turns[:-window] if len(turns) > window else []

        # Drop low-value chatter from older turns
        older_kept    = [
            m for m in older
            if not _below_threshold(m, drop_thresh)
        ]
        dropped_count = len(older) - len(older_kept)

        # Build facts block from the surviving older turns
        facts         = extract_facts(older_kept) if older_kept else None
        summary_block = _make_summary_block(older_kept, facts, compact=False)

        body = summary_block + recent
        body_tokens = sum(estimate_tokens(m.get("content", "")) for m in body)
        used = overhead + body_tokens

        return BuiltContext(
            messages=self._assemble(sys_prompt, body, new_msg),
            tokens_sent=used,
            budget_tokens=budget,
            budget_pct=round(used / max(self.max_ctx, 1), 3),
            tier=BudgetTier.RECENT_PLUS_FACTS,
            turns_total=len(turns),
            turns_kept=len(recent),
            turns_dropped=dropped_count,
            turns_summarized=len(older_kept),
            summary_injected=bool(summary_block),
            facts=facts,
        )

    def _tier_compressed(
        self,
        turns: list[dict],
        sys_prompt: Optional[str],
        new_msg: Optional[str],
        overhead: int,
        budget: int,
    ) -> BuiltContext:
        """
        COMPRESSED — compress the recent window (tool outputs, long content);
        older turns become a compact summary block.
        """
        window      = RECENT_WINDOW[BudgetTier.COMPRESSED]
        drop_thresh = DROP_THRESHOLD[BudgetTier.COMPRESSED]

        recent = turns[-window:]
        older  = turns[:-window] if len(turns) > window else []

        # Compress recent window: drop chatter, compress tool outputs + long content
        compressed_recent: list[dict] = []
        dropped_recent = 0
        for m in recent:
            if _below_threshold(m, drop_thresh):
                dropped_recent += 1
                continue
            new_content = compress_message(
                m["role"], m.get("content", ""), aggressive=False
            )
            compressed_recent.append({"role": m["role"], "content": new_content})

        # Compact summary of all older content
        facts         = extract_facts(older) if older else None
        summary_block = _make_summary_block(older, facts, compact=False)

        body        = summary_block + compressed_recent
        body_tokens = sum(estimate_tokens(m.get("content", "")) for m in body)
        used        = overhead + body_tokens

        return BuiltContext(
            messages=self._assemble(sys_prompt, body, new_msg),
            tokens_sent=used,
            budget_tokens=budget,
            budget_pct=round(used / max(self.max_ctx, 1), 3),
            tier=BudgetTier.COMPRESSED,
            turns_total=len(turns),
            turns_kept=len(compressed_recent),
            turns_dropped=dropped_recent,
            turns_summarized=len(older),
            summary_injected=bool(summary_block),
            facts=facts,
        )

    def _tier_emergency(
        self,
        turns: list[dict],
        sys_prompt: Optional[str],
        new_msg: Optional[str],
        overhead: int,
        budget: int,
    ) -> BuiltContext:
        """
        EMERGENCY — 4-turn window, aggressive compression everywhere,
        ultra-compact one-line summary of all older content.
        """
        window      = RECENT_WINDOW[BudgetTier.EMERGENCY]
        drop_thresh = DROP_THRESHOLD[BudgetTier.EMERGENCY]

        recent = turns[-window:]
        older  = turns[:-window] if len(turns) > window else []

        # Aggressively compress the recent window
        compressed_recent: list[dict] = []
        dropped_recent = 0
        for m in recent:
            if _below_threshold(m, drop_thresh):
                dropped_recent += 1
                continue
            new_content = compress_message(
                m["role"], m.get("content", ""), aggressive=True
            )
            compressed_recent.append({"role": m["role"], "content": new_content})

        # Emergency summary: one-liners only from the older (non-recent) turns
        facts         = extract_facts(older) if older else None
        summary_block = _make_summary_block(older, facts, compact=True)

        body        = summary_block + compressed_recent
        body_tokens = sum(estimate_tokens(m.get("content", "")) for m in body)
        used        = overhead + body_tokens

        logger.warning(
            "ContextWindow EMERGENCY: %d turns dropped, %d summarised, %.0f%% of context used",
            dropped_recent,
            len(older),
            used / max(self.max_ctx, 1) * 100,
        )

        return BuiltContext(
            messages=self._assemble(sys_prompt, body, new_msg),
            tokens_sent=used,
            budget_tokens=budget,
            budget_pct=round(used / max(self.max_ctx, 1), 3),
            tier=BudgetTier.EMERGENCY,
            turns_total=len(turns),
            turns_kept=len(compressed_recent),
            turns_dropped=dropped_recent,
            turns_summarized=len(older),
            summary_injected=bool(summary_block),
            facts=facts,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assemble(
        sys_prompt: Optional[str],
        body: list[dict],
        new_msg: Optional[str],
    ) -> list[dict]:
        """Assemble the final messages list: system → body → new message."""
        result: list[dict] = []
        if sys_prompt:
            result.append({"role": "system", "content": sys_prompt})
        result.extend(body)
        if new_msg:
            result.append({"role": "user", "content": new_msg})
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _below_threshold(msg: dict, threshold: float) -> bool:
    """Return True if this message should be dropped at the given threshold."""
    if threshold == 0.0:
        return False
    _, score = score_message(msg.get("role", ""), msg.get("content", ""))
    return score < threshold


def _make_summary_block(
    messages: list[dict],
    facts: Optional[ConversationFacts],
    compact: bool,
) -> list[dict]:
    """
    Build a summary block as a system message, or return [] if nothing to summarise.
    """
    if not messages:
        return []
    if facts is None or facts.is_empty:
        # Even with no extractable facts, still note that turns were omitted
        turn_count = len([m for m in messages if m.get("role") in ("user", "assistant")])
        if turn_count == 0:
            return []
        text = (
            f"[{turn_count} earlier conversation turns have been summarised. "
            "No specific facts, decisions, or accomplishments were extracted.]"
        )
        return [{"role": "system", "content": text}]

    text = build_summary_block(messages, facts, compact=compact)
    return [{"role": "system", "content": text}]
