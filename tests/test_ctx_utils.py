"""Tests for autotune.api.ctx_utils — token estimation and num_ctx computation."""

from unittest.mock import MagicMock

import pytest

from autotune.api.ctx_utils import (
    compute_num_ctx,
    estimate_messages_tokens,
    estimate_tokens,
    ollama_options_for_profile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(max_new_tokens=512, max_context_tokens=8192, kv_cache_precision="f16"):
    p = MagicMock()
    p.max_new_tokens = max_new_tokens
    p.max_context_tokens = max_context_tokens
    p.kv_cache_precision = kv_cache_precision
    return p


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string_returns_0(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        # "hello" = 5 chars → 5//4 = 1, but max(1, 1) = 1
        assert estimate_tokens("hello") == 1

    def test_four_chars_is_one_token(self):
        assert estimate_tokens("abcd") == 1

    def test_eight_chars_is_two_tokens(self):
        assert estimate_tokens("abcdefgh") == 2

    def test_hundred_chars(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_non_ascii(self):
        # Multi-byte chars still counted by char length
        text = "こんにちは"  # 5 chars
        assert estimate_tokens(text) == 1


# ---------------------------------------------------------------------------
# estimate_messages_tokens
# ---------------------------------------------------------------------------

class TestEstimateMessagesTokens:
    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "a" * 40}]
        assert estimate_messages_tokens(msgs) == 10

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "a" * 40},  # 10 tokens
            {"role": "user", "content": "b" * 80},    # 20 tokens
            {"role": "assistant", "content": "c" * 40}, # 10 tokens
        ]
        assert estimate_messages_tokens(msgs) == 40

    def test_missing_content_key(self):
        msgs = [{"role": "user"}]
        # estimate_tokens("") == 0
        assert estimate_messages_tokens(msgs) == 0


# ---------------------------------------------------------------------------
# compute_num_ctx
# ---------------------------------------------------------------------------

class TestComputeNumCtx:
    def test_small_input_hits_minimum(self):
        # 4-char input = 1 token; 1 + 512 + 256 = 769 < 8192 but > 512
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = [{"role": "user", "content": "hi"}]
        result = compute_num_ctx(msgs, profile)
        assert result >= 512
        assert result <= 8192

    def test_small_input_minimum_floor(self):
        # Tiny input should not go below 512
        profile = _make_profile(max_new_tokens=0, max_context_tokens=8192)
        msgs = [{"role": "user", "content": "hi"}]
        result = compute_num_ctx(msgs, profile)
        assert result == 512

    def test_large_input_capped_at_profile_max(self):
        # 40000-char message = 10000 tokens
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = [{"role": "user", "content": "a" * 40000}]
        result = compute_num_ctx(msgs, profile)
        assert result == 8192

    def test_grows_with_message_history(self):
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        short_msgs = [{"role": "user", "content": "a" * 40}]
        long_msgs = [
            {"role": "user", "content": "a" * 40},
            {"role": "assistant", "content": "b" * 800},
            {"role": "user", "content": "c" * 400},
        ]
        assert compute_num_ctx(long_msgs, profile) > compute_num_ctx(short_msgs, profile)

    def test_exact_formula(self):
        profile = _make_profile(max_new_tokens=256, max_context_tokens=4096)
        # 400-char content = 100 tokens; 100 + 256 + 256 = 612
        msgs = [{"role": "user", "content": "a" * 400}]
        result = compute_num_ctx(msgs, profile)
        assert result == 612


# ---------------------------------------------------------------------------
# ollama_options_for_profile
# ---------------------------------------------------------------------------

class TestOllamaOptionsForProfile:
    def test_f16_kv_when_precision_not_q8(self):
        profile = _make_profile(kv_cache_precision="f16")
        opts = ollama_options_for_profile([], profile)
        assert opts["f16_kv"] is True

    def test_q8_kv_when_precision_is_q8(self):
        profile = _make_profile(kv_cache_precision="q8")
        opts = ollama_options_for_profile([], profile)
        assert opts["f16_kv"] is False

    def test_num_ctx_present(self):
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = [{"role": "user", "content": "hello"}]
        opts = ollama_options_for_profile(msgs, profile)
        assert "num_ctx" in opts
        assert opts["num_ctx"] >= 512
