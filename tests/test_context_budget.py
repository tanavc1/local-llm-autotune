"""Tests for autotune.context.budget — tier classification and BudgetState."""

import pytest

from autotune.context.budget import (
    BudgetTier,
    BudgetState,
    classify_budget,
    TIER_THRESHOLDS,
    RECENT_WINDOW,
    DROP_THRESHOLD,
)


class TestClassifyBudget:
    def test_full_tier_at_low_usage(self):
        # 100 tokens used out of 1000 budget = 10% → FULL
        state = classify_budget(100, 1000)
        assert state.tier == BudgetTier.FULL

    def test_full_tier_at_54_pct(self):
        state = classify_budget(540, 1000)
        assert state.tier == BudgetTier.FULL

    def test_recent_plus_facts_at_60_pct(self):
        state = classify_budget(600, 1000)
        assert state.tier == BudgetTier.RECENT_PLUS_FACTS

    def test_compressed_at_80_pct(self):
        state = classify_budget(800, 1000)
        assert state.tier == BudgetTier.COMPRESSED

    def test_emergency_at_95_pct(self):
        state = classify_budget(950, 1000)
        assert state.tier == BudgetTier.EMERGENCY

    def test_emergency_at_100_pct(self):
        state = classify_budget(1000, 1000)
        assert state.tier == BudgetTier.EMERGENCY

    def test_emergency_above_100_pct(self):
        state = classify_budget(1500, 1000)
        assert state.tier == BudgetTier.EMERGENCY

    def test_zero_history(self):
        state = classify_budget(0, 1000)
        assert state.tier == BudgetTier.FULL
        assert state.pct == 0.0

    def test_headroom_computed(self):
        state = classify_budget(300, 1000)
        assert state.headroom == 700

    def test_headroom_zero_when_over_budget(self):
        state = classify_budget(1500, 1000)
        assert state.headroom == 0

    def test_over_budget_property(self):
        assert classify_budget(1100, 1000).over_budget is True
        assert classify_budget(500, 1000).over_budget is False

    def test_pct_rounded(self):
        state = classify_budget(333, 1000)
        # Should be rounded to 4 decimal places
        assert state.pct == round(333 / 1000, 4)

    def test_zero_budget_does_not_divide_by_zero(self):
        # effective_budget=0 should not raise; max(0,1)=1 prevents ZeroDivisionError
        state = classify_budget(100, 0)
        assert state.tier == BudgetTier.EMERGENCY


class TestTierConstants:
    def test_full_threshold_is_55_pct(self):
        assert TIER_THRESHOLDS[BudgetTier.FULL] == 0.55

    def test_emergency_threshold_is_1(self):
        assert TIER_THRESHOLDS[BudgetTier.EMERGENCY] == 1.00

    def test_recent_window_decreases_with_pressure(self):
        assert RECENT_WINDOW[BudgetTier.FULL] > RECENT_WINDOW[BudgetTier.RECENT_PLUS_FACTS]
        assert RECENT_WINDOW[BudgetTier.RECENT_PLUS_FACTS] > RECENT_WINDOW[BudgetTier.COMPRESSED]
        assert RECENT_WINDOW[BudgetTier.COMPRESSED] > RECENT_WINDOW[BudgetTier.EMERGENCY]

    def test_drop_threshold_increases_with_pressure(self):
        assert DROP_THRESHOLD[BudgetTier.FULL] == 0.0
        assert DROP_THRESHOLD[BudgetTier.EMERGENCY] > DROP_THRESHOLD[BudgetTier.FULL]

    def test_tier_enum_values(self):
        assert BudgetTier.FULL.value == "full"
        assert BudgetTier.EMERGENCY.value == "emergency"


class TestContextWindow:
    """Integration tests for ContextWindow using real tier logic."""

    def test_full_tier_short_conversation(self):
        from autotune.context.window import ContextWindow
        cw = ContextWindow(max_ctx_tokens=8192)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = cw.build(history, new_user_message="How are you?", reserved_for_output=512)
        assert result.tier == BudgetTier.FULL
        assert result.turns_dropped == 0
        assert result.summary_injected is False

    def test_messages_include_system_prompt(self):
        from autotune.context.window import ContextWindow
        cw = ContextWindow(max_ctx_tokens=8192)
        result = cw.build(
            [],
            system_prompt="You are a helpful assistant.",
            new_user_message="Hello",
            reserved_for_output=512,
        )
        assert result.messages[0]["role"] == "system"
        assert "helpful" in result.messages[0]["content"]

    def test_new_user_message_is_last(self):
        from autotune.context.window import ContextWindow
        cw = ContextWindow(max_ctx_tokens=8192)
        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First reply"},
        ]
        result = cw.build(history, new_user_message="Second message", reserved_for_output=512)
        assert result.messages[-1]["role"] == "user"
        assert result.messages[-1]["content"] == "Second message"

    def test_min_ctx_tokens_validation(self):
        from autotune.context.window import ContextWindow
        with pytest.raises(ValueError, match="256"):
            ContextWindow(max_ctx_tokens=100)

    def test_emergency_tier_on_very_long_history(self):
        """Pack enough tokens to trigger EMERGENCY tier."""
        from autotune.context.window import ContextWindow
        cw = ContextWindow(max_ctx_tokens=1024)
        # Each message ~200 chars = 50 tokens; 20 messages = 1000 tokens
        # Budget ≈ 1024 - 512(output) - 64(overhead) = 448
        # 1000 / 448 = 223% → EMERGENCY
        history = []
        for i in range(20):
            history.append({"role": "user", "content": "x" * 200})
            history.append({"role": "assistant", "content": "y" * 200})
        result = cw.build(history, reserved_for_output=512)
        assert result.tier == BudgetTier.EMERGENCY

    def test_built_context_has_all_fields(self):
        from autotune.context.window import ContextWindow, BuiltContext
        cw = ContextWindow(max_ctx_tokens=8192)
        result = cw.build([], new_user_message="Hi", reserved_for_output=512)
        assert isinstance(result, BuiltContext)
        assert result.tokens_sent >= 0
        assert result.budget_tokens >= 0
        assert 0.0 <= result.budget_pct <= 2.0  # can be > 1.0 if over budget
