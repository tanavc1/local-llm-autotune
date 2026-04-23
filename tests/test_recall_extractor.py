"""Tests for autotune.recall.extractor — chunk extraction and value scoring."""

import pytest

from autotune.recall.extractor import estimate_conversation_value, extract_chunks

# ---------------------------------------------------------------------------
# extract_chunks
# ---------------------------------------------------------------------------

class TestExtractChunks:
    def test_empty_messages_returns_empty(self):
        assert extract_chunks([]) == []

    def test_system_messages_excluded(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language." * 5},
        ]
        chunks = extract_chunks(msgs)
        assert len(chunks) == 1
        assert "system" not in chunks[0]["text"].lower()
        assert "Python" in chunks[0]["text"]

    def test_user_assistant_pair_creates_chunk(self):
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4. This is basic arithmetic." * 3},
        ]
        chunks = extract_chunks(msgs, min_chars=20)
        assert len(chunks) == 1
        assert "User:" in chunks[0]["text"]
        assert "Assistant:" in chunks[0]["text"]

    def test_turn_indices_correct(self):
        msgs = [
            {"role": "user", "content": "First question that is long enough for the test."},
            {"role": "assistant", "content": "First answer. " * 10},
            {"role": "user", "content": "Second question that is also long enough now."},
            {"role": "assistant", "content": "Second answer. " * 10},
        ]
        chunks = extract_chunks(msgs, min_chars=20)
        assert len(chunks) == 2
        assert chunks[0]["turn_start"] == 0
        assert chunks[0]["turn_end"] == 1
        assert chunks[1]["turn_start"] == 2
        assert chunks[1]["turn_end"] == 3

    def test_orphaned_user_message(self):
        msgs = [
            {"role": "user", "content": "This user message has no assistant reply."},
        ]
        chunks = extract_chunks(msgs, min_chars=20)
        # Orphaned message creates a chunk if it meets min_chars
        assert len(chunks) == 1
        assert chunks[0]["turn_start"] == chunks[0]["turn_end"]

    def test_min_chars_filter(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        chunks = extract_chunks(msgs, min_chars=60)
        assert len(chunks) == 0

    def test_multiple_pairs(self):
        msgs = []
        for i in range(5):
            msgs.append({"role": "user", "content": f"Question {i} " * 5})
            msgs.append({"role": "assistant", "content": f"Answer {i} " * 10})
        chunks = extract_chunks(msgs, min_chars=20)
        assert len(chunks) == 5

    def test_user_content_truncated_at_limit(self):
        # _USER_LIMIT = 1000
        long_user = "x" * 2000
        msgs = [
            {"role": "user", "content": long_user},
            {"role": "assistant", "content": "Answer. " * 20},
        ]
        chunks = extract_chunks(msgs)
        # The chunk text should not contain the full 2000-char user message
        user_part = chunks[0]["text"].split("\nAssistant:")[0].replace("User: ", "")
        assert len(user_part) <= 1000

    def test_assistant_content_truncated_at_limit(self):
        # _ASST_LIMIT = 2000
        long_asst = "y" * 3000
        msgs = [
            {"role": "user", "content": "Question that is long enough for the limit test."},
            {"role": "assistant", "content": long_asst},
        ]
        chunks = extract_chunks(msgs)
        asst_part = chunks[0]["text"].split("Assistant: ")[1]
        assert len(asst_part) <= 2000

    def test_assistant_without_preceding_user_skipped(self):
        msgs = [
            {"role": "assistant", "content": "I say something without a user prompt."},
            {"role": "user", "content": "Now the user asks something meaningful."},
            {"role": "assistant", "content": "And I answer properly with a long response." * 3},
        ]
        chunks = extract_chunks(msgs, min_chars=20)
        # Only the user+assistant pair at indices 1,2 (in turns list) should create a chunk
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# estimate_conversation_value
# ---------------------------------------------------------------------------

class TestEstimateConversationValue:
    def test_meaningful_exchange(self):
        msgs = [
            {"role": "user", "content": "Can you explain what recursion is in programming?"},
            {"role": "assistant", "content": "Recursion is when a function calls itself. " * 5},
        ]
        assert estimate_conversation_value(msgs) is True

    def test_too_short_user(self):
        msgs = [
            {"role": "user", "content": "hi"},  # < 20 chars
            {"role": "assistant", "content": "Hello! How can I help you today? " * 5},
        ]
        assert estimate_conversation_value(msgs) is False

    def test_too_short_assistant(self):
        msgs = [
            {"role": "user", "content": "Tell me about machine learning please."},
            {"role": "assistant", "content": "It's complex."},  # < 50 chars
        ]
        assert estimate_conversation_value(msgs) is False

    def test_empty_conversation(self):
        assert estimate_conversation_value([]) is False

    def test_only_system_messages(self):
        msgs = [{"role": "system", "content": "You are helpful."}]
        assert estimate_conversation_value(msgs) is False

    def test_system_message_ignored(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Can you explain what recursion is in programming?"},
            {"role": "assistant", "content": "Recursion is a function calling itself. " * 5},
        ]
        assert estimate_conversation_value(msgs) is True
