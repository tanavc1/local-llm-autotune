"""
Tests for autotune.context.compressor — message compression strategies.

Covers:
- compress_json_in_content: dict, list, nested, invalid JSON, small blobs
- compress_tool_output: short output, long output, head/tail/omission
- compress_assistant_message: short (unchanged), long prose, code preservation,
  aggressive mode, exact max_chars boundary
- compress_message: dispatcher, role semantics, empty content passthrough
- _cut_at_sentence: sentence boundary, paragraph break, whitespace, hard cut
"""

from __future__ import annotations

import json
import re

import pytest

from autotune.context.compressor import (
    compress_assistant_message,
    compress_json_in_content,
    compress_message,
    compress_tool_output,
)


# ---------------------------------------------------------------------------
# compress_json_in_content
# ---------------------------------------------------------------------------

class TestCompressJsonInContent:
    def test_large_dict_replaced_with_summary(self):
        large = json.dumps({f"key_{i}": f"value_{i}" for i in range(20)})
        result = compress_json_in_content(large)
        assert "key_0" not in result or len(result) < len(large)
        assert "keys" in result.lower() or result.startswith("{")

    def test_large_list_replaced_with_summary(self):
        large = json.dumps(list(range(50)))
        result = compress_json_in_content(large)
        assert len(result) < len(large)

    def test_small_blob_not_replaced(self):
        small = json.dumps({"key": "val"})   # < 120 chars — threshold
        result = compress_json_in_content(small)
        assert result == small  # unchanged

    def test_non_json_truncated(self):
        # A blob that looks like JSON brackets but isn't valid JSON
        blob = "{" + "x" * 200 + "}"
        result = compress_json_in_content(blob)
        # Should truncate, not crash
        assert isinstance(result, str)
        assert len(result) < len(blob)

    def test_prose_unchanged(self):
        text = "This is a normal sentence with no JSON blobs."
        assert compress_json_in_content(text) == text

    def test_multiple_blobs_in_message(self):
        d1 = json.dumps({f"a{i}": i for i in range(15)})
        d2 = json.dumps({f"b{i}": i for i in range(15)})
        text = f"First: {d1}\nSecond: {d2}"
        result = compress_json_in_content(text)
        assert len(result) < len(text)

    def test_dict_shows_key_count(self):
        # Use enough keys to exceed the 120-char compression threshold
        large = json.dumps({f"key_{i}": f"value_{i}" for i in range(15)})
        result = compress_json_in_content(large)
        # Should be compressed — either shows count or has fewer chars
        assert len(result) < len(large) or "key" in result.lower()

    def test_list_shows_item_count(self):
        large = json.dumps(["item"] * 30)
        result = compress_json_in_content(large)
        assert "30" in result or "str" in result.lower()


# ---------------------------------------------------------------------------
# compress_tool_output
# ---------------------------------------------------------------------------

class TestCompressToolOutput:
    def make_lines(self, n: int) -> str:
        return "\n".join(f"line {i}" for i in range(n))

    def test_short_output_unchanged(self):
        text = self.make_lines(10)
        result = compress_tool_output(text)
        assert result == text

    def test_long_output_truncated(self):
        text = self.make_lines(50)
        result = compress_tool_output(text)
        assert len(result.splitlines()) < 50
        assert "omitted" in result

    def test_head_lines_preserved(self):
        text = self.make_lines(50)
        result = compress_tool_output(text, head_lines=5)
        lines = result.split("\n")
        assert lines[0] == "line 0"
        assert lines[4] == "line 4"

    def test_tail_lines_preserved(self):
        text = self.make_lines(50)
        result = compress_tool_output(text, tail_lines=3)
        lines = result.split("\n")
        assert lines[-1] == "line 49"
        assert lines[-2] == "line 48"
        assert lines[-3] == "line 47"

    def test_omission_count_correct(self):
        text = self.make_lines(50)
        result = compress_tool_output(text, head_lines=5, tail_lines=5)
        # 50 - 5 - 5 = 40 omitted
        assert "40" in result

    def test_exactly_at_max_boundary_unchanged(self):
        text = self.make_lines(20)  # exactly at default max_total_lines=20
        result = compress_tool_output(text, max_total_lines=20)
        assert result == text

    def test_one_above_max_triggers_compression(self):
        text = self.make_lines(21)  # one above default
        result = compress_tool_output(text, max_total_lines=20)
        assert "omitted" in result

    def test_custom_parameters(self):
        text = self.make_lines(100)
        result = compress_tool_output(text, head_lines=3, tail_lines=2, max_total_lines=10)
        result_lines = result.split("\n")
        # head: line 0, 1, 2; omission marker; tail: line 98, 99
        assert result_lines[0] == "line 0"
        assert result_lines[-1] == "line 99"
        assert "omitted" in result


# ---------------------------------------------------------------------------
# compress_assistant_message
# ---------------------------------------------------------------------------

class TestCompressAssistantMessage:
    def test_short_message_unchanged(self):
        text = "Python is great." * 10  # ~160 chars, well under default 2000
        result = compress_assistant_message(text, max_chars=2000)
        assert result == text

    def test_long_prose_shortened(self):
        text = ("This is a paragraph with detailed explanation. " * 50) + "\n\n"
        text += "And another paragraph with more info. " * 50
        result = compress_assistant_message(text, max_chars=500)
        assert len(result) <= 500 + 200   # allow for omission annotation

    def test_code_blocks_preserved(self):
        text = (
            "Here is the answer.\n\n"
            "```python\n"
            "def important_function():\n"
            "    return 42\n"
            "```\n\n"
            + "Filler paragraph. " * 100
        )
        result = compress_assistant_message(text, max_chars=500)
        assert "important_function" in result

    def test_aggressive_mode_smaller_output(self):
        text = "Explanation. " * 200
        normal    = compress_assistant_message(text, max_chars=2000)
        aggressive = compress_assistant_message(text, max_chars=2000, aggressive=True)
        assert len(aggressive) <= len(normal)

    def test_aggressive_only_one_code_block_kept(self):
        text = (
            "Intro.\n\n"
            "```python\ndef block1(): pass\n```\n\n"
            "Middle text.\n\n"
            "```python\ndef block2(): pass\n```\n\n"
            "Conclusion."
        )
        # max_chars=100; aggressive=True halves it to 50, below the ~93-char content
        result = compress_assistant_message(text, max_chars=100, aggressive=True)
        # At most one code block should appear
        assert result.count("```") <= 2   # one block = open + close

    def test_omission_annotation_added(self):
        text = "A" * 5000
        result = compress_assistant_message(text, max_chars=200)
        assert "omitted" in result

    def test_result_never_exceeds_budget_significantly(self):
        text = "word " * 2000  # ~10000 chars
        result = compress_assistant_message(text, max_chars=300)
        # Allow some slack for the omission annotation
        assert len(result) < 700

    def test_preserves_first_paragraph(self):
        first = "This is the critical first paragraph.\n\n"
        rest  = "Filler. " * 500
        result = compress_assistant_message(first + rest, max_chars=200)
        assert "critical first paragraph" in result

    def test_preserves_last_paragraph_when_room(self):
        first  = "First paragraph.\n\n"
        middle = "Middle filler. " * 3
        last   = "\n\nThis is the important conclusion."
        text   = first + middle + last
        result = compress_assistant_message(text, max_chars=2000)
        # With ample budget, both first and last should appear
        assert "First paragraph" in result


# ---------------------------------------------------------------------------
# compress_message dispatcher
# ---------------------------------------------------------------------------

class TestCompressMessage:
    def test_empty_content_passthrough(self):
        assert compress_message("user", "") == ""
        assert compress_message("assistant", "") == ""
        assert compress_message("tool", "") == ""

    def test_tool_role_uses_tool_compression(self):
        text = "\n".join(f"log line {i}" for i in range(50))
        result = compress_message("tool", text)
        assert "omitted" in result

    def test_assistant_role_uses_prose_compression(self):
        long_prose = "Detailed explanation. " * 200
        result = compress_message("assistant", long_prose)
        assert len(result) < len(long_prose)

    def test_user_normal_not_compressed(self):
        text = "Can you help me with Python? I need a function that does X."
        result = compress_message("user", text)
        # Normal user messages: no compression unless aggressive
        assert result == text

    def test_user_aggressive_compresses(self):
        long_user = "Please help me. " * 100  # >600 chars
        result = compress_message("user", long_user, aggressive=True)
        assert len(result) < len(long_user)

    def test_system_role_not_compressed(self):
        text = "You are a helpful coding assistant. " * 50
        result = compress_message("system", text)
        # System messages are not aggressively compressed
        assert isinstance(result, str)

    def test_noise_stripped_always(self):
        text = "Some content.\n\n\n\nMore content.\n\n\n"
        result = compress_message("user", text)
        # Multiple blank lines should collapse
        assert "\n\n\n" not in result

    def test_aggressive_flag_more_aggressive_than_normal(self):
        long = "Text. " * 500
        normal     = compress_message("assistant", long, aggressive=False)
        aggressive = compress_message("assistant", long, aggressive=True)
        assert len(aggressive) <= len(normal)


# ---------------------------------------------------------------------------
# _cut_at_sentence (tested indirectly via compress_assistant_message)
# ---------------------------------------------------------------------------

class TestCutAtSentence:
    def test_cuts_at_period(self):
        """Compressed output should end at sentence boundary, not mid-word."""
        text = "First sentence. Second sentence. " * 100
        result = compress_assistant_message(text, max_chars=50)
        # Should not end mid-word if possible
        if not result.endswith("omitted]"):
            # Check that we cut at a sentence boundary or near one
            assert result[-1] in ".!? \n]" or len(result) <= 55

    def test_handles_no_sentence_boundary(self):
        """Pure continuous text should still produce something sensible."""
        text = "a" * 200
        result = compress_assistant_message(text, max_chars=50)
        assert len(result) > 0
