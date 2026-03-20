"""
Token budget management — determines which compression tier applies.

Tiers
-----
FULL              < 55 % of budget   — send everything verbatim
RECENT_PLUS_FACTS  55–75 %           — recent window + extracted facts block
COMPRESSED         75–90 %           — recent window compressed + summary
EMERGENCY          > 90 %            — 4-turn window, maximum compression

The thresholds are deliberately conservative: it is better to start compressing
slightly early than to saturate the context window and have Ollama silently
truncate the oldest content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BudgetTier(Enum):
    FULL              = "full"
    RECENT_PLUS_FACTS = "facts"
    COMPRESSED        = "compressed"
    EMERGENCY         = "emergency"


# Fraction of the adjusted budget that triggers each tier.
# "adjusted budget" = max_ctx_tokens − system prompt − new message − output reserve
TIER_THRESHOLDS: dict[BudgetTier, float] = {
    BudgetTier.FULL:              0.55,
    BudgetTier.RECENT_PLUS_FACTS: 0.75,
    BudgetTier.COMPRESSED:        0.90,
    BudgetTier.EMERGENCY:         1.00,
}

# How many recent turns to keep verbatim in each tier
RECENT_WINDOW: dict[BudgetTier, int] = {
    BudgetTier.FULL:              9_999,   # all
    BudgetTier.RECENT_PLUS_FACTS: 8,
    BudgetTier.COMPRESSED:        6,
    BudgetTier.EMERGENCY:         4,
}

# Minimum message-value score to keep a message (drop below this threshold)
DROP_THRESHOLD: dict[BudgetTier, float] = {
    BudgetTier.FULL:              0.00,   # keep everything
    BudgetTier.RECENT_PLUS_FACTS: 0.15,
    BudgetTier.COMPRESSED:        0.25,
    BudgetTier.EMERGENCY:         0.40,
}


@dataclass(frozen=True)
class BudgetState:
    total_tokens:    int        # effective budget (after subtracting fixed overhead)
    history_tokens:  int        # tokens consumed by history messages
    pct:             float      # history_tokens / total_tokens
    tier:            BudgetTier
    headroom:        int        # total_tokens - history_tokens

    @property
    def over_budget(self) -> bool:
        return self.pct >= 1.0


def classify_budget(history_tokens: int, effective_budget: int) -> BudgetState:
    """
    Determine the compression tier for the current conversation state.

    Parameters
    ----------
    history_tokens   : total token estimate of all history messages (not including
                       system prompt or new user turn — those are already subtracted
                       from effective_budget)
    effective_budget : max_ctx_tokens minus the fixed overhead (system prompt
                       tokens + new message tokens + output reservation)
    """
    pct = history_tokens / max(effective_budget, 1)

    if pct < TIER_THRESHOLDS[BudgetTier.FULL]:
        tier = BudgetTier.FULL
    elif pct < TIER_THRESHOLDS[BudgetTier.RECENT_PLUS_FACTS]:
        tier = BudgetTier.RECENT_PLUS_FACTS
    elif pct < TIER_THRESHOLDS[BudgetTier.COMPRESSED]:
        tier = BudgetTier.COMPRESSED
    else:
        tier = BudgetTier.EMERGENCY

    return BudgetState(
        total_tokens=effective_budget,
        history_tokens=history_tokens,
        pct=round(pct, 4),
        tier=tier,
        headroom=max(0, effective_budget - history_tokens),
    )
