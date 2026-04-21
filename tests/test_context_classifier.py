"""
Tests for autotune.context.classifier — message value scoring.

Covers:
- Pure chatter detection (LOW value)
- Code block signal (HIGH value)
- Stack trace signal (HIGH value)
- Technical vocabulary scoring
- Numeric measurements
- Bullet/numbered lists
- URL presence
- User question bonus
- Length bonus/penalty
- Tool role always TOOL value
- Edge cases: empty content, very short, multi-signal
- is_droppable threshold semantics
"""

from __future__ import annotations

import pytest

from autotune.context.classifier import MessageValue, is_droppable, score_message


# ---------------------------------------------------------------------------
# Pure chatter → LOW
# ---------------------------------------------------------------------------

class TestChatterDetection:
    @pytest.mark.parametrize("text", [
        "ok",
        "okay",
        "sure",
        "thanks",
        "thank you",
        "ty",
        "thx",
        "great",
        "got it",
        "sounds good",
        "perfect",
        "cool",
        "nice",
        "yes",
        "no",
        "nope",
        "hi",
        "hello",
        "bye",
        "goodbye",
    ])
    def test_chatter_is_low(self, text: str):
        val, score = score_message("user", text)
        assert val == MessageValue.LOW
        assert score < 0.20, f"'{text}' should be LOW but got score {score}"

    def test_chatter_with_punctuation_is_low(self):
        val, score = score_message("user", "okay!")
        assert val == MessageValue.LOW

    def test_chatter_with_trailing_space_is_low(self):
        val, score = score_message("user", "thanks   ")
        assert val == MessageValue.LOW

    def test_chatter_case_insensitive(self):
        for text in ("OK", "SURE", "THANKS", "GREAT"):
            val, score = score_message("user", text)
            assert val == MessageValue.LOW, f"'{text}' should be LOW"

    def test_empty_content_is_low(self):
        val, score = score_message("user", "")
        assert val == MessageValue.LOW
        assert score == 0.0

    def test_whitespace_only_is_low(self):
        val, score = score_message("user", "   ")
        assert val == MessageValue.LOW


# ---------------------------------------------------------------------------
# Code content → HIGH
# ---------------------------------------------------------------------------

class TestCodeSignal:
    def test_full_code_block_is_high(self):
        content = (
            "Here is the solution:\n"
            "```python\n"
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)\n"
            "```"
        )
        val, score = score_message("assistant", content)
        assert val == MessageValue.HIGH
        assert score >= 0.65

    def test_inline_code_boosts_score(self):
        content = "Use `os.path.join()` instead of string concatenation."
        _, score = score_message("user", content)
        # Should be higher than baseline because of inline code
        assert score > 0.30

    def test_multiple_code_blocks_very_high(self):
        content = (
            "Option A:\n```python\nprint('hello')\n```\n\n"
            "Option B:\n```bash\necho 'hello'\n```"
        )
        val, score = score_message("assistant", content)
        assert val == MessageValue.HIGH

    def test_assistant_without_code_is_medium(self):
        content = "Python is a general-purpose programming language with clear syntax."
        val, score = score_message("assistant", content)
        assert val in (MessageValue.MEDIUM, MessageValue.HIGH)


# ---------------------------------------------------------------------------
# Stack traces / errors → HIGH
# ---------------------------------------------------------------------------

class TestErrorSignal:
    def test_traceback_is_high_or_medium(self):
        content = (
            "Traceback (most recent call last):\n"
            "  File 'app.py', line 42\n"
            "    raise ValueError('bad input')\n"
            "ValueError: bad input\n"
        )
        val, score = score_message("user", content)
        assert val in (MessageValue.HIGH, MessageValue.MEDIUM)
        assert score >= 0.55

    def test_error_keyword_boosts_score(self):
        content = "Error: connection refused at port 5432"
        _, score = score_message("user", content)
        assert score > 0.45

    def test_exception_word_boosts_score(self):
        content = "An Exception was raised during model loading."
        _, score = score_message("assistant", content)
        assert score > 0.30


# ---------------------------------------------------------------------------
# Technical vocabulary
# ---------------------------------------------------------------------------

class TestTechnicalVocabulary:
    def test_sql_keywords_boost(self):
        content = "SELECT * FROM users WHERE id = 42 JOIN orders ON orders.user_id = users.id"
        _, score = score_message("user", content)
        assert score > 0.30

    def test_api_terms_boost(self):
        content = "The REST API returns a JSON response with HTTP 200 status."
        _, score = score_message("user", content)
        assert score > 0.35

    def test_memory_terms_boost(self):
        content = "The model uses 8 GB of RAM and 512 MB of VRAM during inference."
        _, score = score_message("user", content)
        assert score > 0.45

    def test_inference_terms_boost(self):
        content = "Model inference latency: 120ms TTFT, 35 tok/s throughput."
        _, score = score_message("user", content)
        assert score > 0.45


# ---------------------------------------------------------------------------
# Numbers with units
# ---------------------------------------------------------------------------

class TestNumericSignal:
    @pytest.mark.parametrize("text", [
        "The model runs at 35 tok/s consistently.",
        "The peak RAM usage was 12 GB during the run.",
        "Response time was 250ms on average.",
        "We allocate 512 MB for the KV cache.",
        "CPU usage reached 85% during token generation.",
    ])
    def test_numeric_measurements_boost_score(self, text: str):
        _, score = score_message("user", text)
        assert score > 0.30, f"'{text}' should boost score but got {score}"


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

class TestListSignal:
    def test_bullet_list_boosts_score(self):
        content = (
            "Options:\n"
            "- Use SQLite for local storage\n"
            "- Use PostgreSQL for production\n"
            "- Use Redis for caching\n"
        )
        _, score = score_message("assistant", content)
        assert score > 0.35

    def test_numbered_list_boosts_score(self):
        content = (
            "Steps:\n"
            "1. Install dependencies\n"
            "2. Configure environment\n"
            "3. Run migrations\n"
        )
        _, score = score_message("assistant", content)
        assert score > 0.35


# ---------------------------------------------------------------------------
# URL presence
# ---------------------------------------------------------------------------

class TestURLSignal:
    def test_url_boosts_score(self):
        content = "See the docs at https://docs.python.org/3/library/os.path.html"
        _, score = score_message("user", content)
        assert score > 0.30


# ---------------------------------------------------------------------------
# Question bonus (user role only)
# ---------------------------------------------------------------------------

class TestQuestionBonus:
    def test_user_question_gets_bonus(self):
        _, user_score   = score_message("user", "How does the KV cache work?")
        _, assist_score = score_message("assistant", "How does the KV cache work?")
        assert user_score > assist_score

    def test_assistant_question_no_bonus(self):
        content = "Do you want me to explain this further?"
        _, score = score_message("assistant", content)
        # No user-question bonus but not penalised
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Length bonus / penalty
# ---------------------------------------------------------------------------

class TestLengthEffect:
    def test_very_short_non_chatter_penalised(self):
        content = "Done."
        _, score = score_message("assistant", content)
        assert score < 0.30

    def test_medium_length_gets_bonus(self):
        content = "a" * 150  # 150 chars > 100 threshold
        _, score = score_message("user", content)
        assert score > 0.30

    def test_long_message_gets_large_bonus(self):
        content = "a" * 1100  # >1000 chars
        _, score = score_message("assistant", content)
        assert score > 0.40


# ---------------------------------------------------------------------------
# Tool role
# ---------------------------------------------------------------------------

class TestToolRole:
    def test_tool_always_returns_tool_value(self):
        # Note: empty content is a special case — returns LOW before role check
        for content in ("result", "a" * 500, '{"key": "value"}', "line1\nline2\n"):
            val, score = score_message("tool", content)
            assert val == MessageValue.TOOL
            assert score == 0.5

    def test_tool_score_is_exactly_half(self):
        val, score = score_message("tool", "command output here")
        assert val == MessageValue.TOOL
        assert score == 0.5

    def test_tool_empty_content_is_low(self):
        # Empty content short-circuits before role check in current implementation
        val, score = score_message("tool", "")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------

class TestScoreClamping:
    def test_score_never_exceeds_1(self):
        content = (
            "```python\ncode\n```\n"
            "Traceback:\nError: bad stuff\n"
            "SELECT * FROM users JOIN orders WHERE id=42\n"
            "Memory: 512 MB RAM 35 tok/s\n"
            "https://example.com\n"
        ) * 5  # pile on many signals
        _, score = score_message("user", content)
        assert 0.0 <= score <= 1.0

    def test_score_never_below_0(self):
        val, score = score_message("user", "")
        assert score == 0.0


# ---------------------------------------------------------------------------
# is_droppable
# ---------------------------------------------------------------------------

class TestIsDroppable:
    def test_chatter_droppable_at_high_threshold(self):
        assert is_droppable("user", "ok", threshold=0.20)

    def test_chatter_droppable_at_low_threshold(self):
        assert is_droppable("user", "sure", threshold=0.10)

    def test_code_not_droppable_at_normal_threshold(self):
        code = "```python\ndef foo(): pass\n```"
        assert not is_droppable("assistant", code, threshold=0.40)

    def test_error_trace_not_droppable_at_high_threshold(self):
        trace = "Traceback (most recent call last):\n  ValueError: bad input"
        assert not is_droppable("user", trace, threshold=0.60)

    def test_threshold_zero_nothing_droppable(self):
        assert not is_droppable("user", "", threshold=0.0)

    def test_threshold_one_everything_droppable(self):
        content = (
            "```python\ndef foo(): pass\n```\n"
            "Traceback error\n"
            "SELECT * FROM users"
        )
        # Score is clamped to 1.0, threshold=1.0 means < 1.0 required to drop
        val, score = score_message("user", content)
        # If score < 1.0, it's droppable at threshold=1.0
        assert is_droppable("user", content, threshold=1.0) == (score < 1.0)

    @pytest.mark.parametrize("threshold", [0.0, 0.15, 0.30, 0.50, 0.70, 1.0])
    def test_droppable_consistent_with_score(self, threshold: float):
        """is_droppable must be equivalent to score < threshold."""
        content = "The function returns a list of integers."
        _, score = score_message("user", content)
        assert is_droppable("user", content, threshold=threshold) == (score < threshold)
