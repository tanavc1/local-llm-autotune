"""Tests for autotune.api.server utility functions.

Covers: _strip_thinking, _filter_thinking_stream, _is_thinking_model,
        _is_chat_model, Message.normalize_content, ChatRequest validation,
        and CompletionRequest validation.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autotune.api.server import (
    ChatRequest,
    CompletionRequest,
    Message,
    _is_chat_model,
)
from autotune.api.thinking import (
    _THINK_CLOSE,
    _THINK_OPEN,
    ThinkingStreamFilter,
)
from autotune.api.thinking import (
    filter_thinking_sse as _filter_thinking_stream,
)
from autotune.api.thinking import (
    is_thinking_model as _is_thinking_model,
)
from autotune.api.thinking import (
    strip_thinking as _strip_thinking,
)

# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------

class TestStripThinking:
    def test_no_think_block(self):
        assert _strip_thinking("hello world") == "hello world"

    def test_complete_block_removed(self):
        text = "<think>internal reasoning</think>Final answer"
        assert _strip_thinking(text) == "Final answer"

    def test_multiple_blocks_removed(self):
        text = "<think>step 1</think>Answer<think>step 2</think> is 42"
        assert _strip_thinking(text) == "Answer is 42"

    def test_incomplete_block_removed(self):
        # Model was cut off mid-think — no closing tag
        text = "<think>reasoning that was never closed"
        assert _strip_thinking(text) == ""

    def test_multiline_block_removed(self):
        text = "<think>\nline one\nline two\n</think>\n\nReal answer"
        assert _strip_thinking(text) == "Real answer"

    def test_empty_string(self):
        assert _strip_thinking("") == ""

    def test_leading_whitespace_stripped_after_removal(self):
        text = "<think>x</think>   answer"
        assert _strip_thinking(text) == "answer"

    def test_no_mutation_without_tags(self):
        text = "plain text with no tags"
        result = _strip_thinking(text)
        assert result == text


# ---------------------------------------------------------------------------
# _filter_thinking_stream (streaming SSE path)
# ---------------------------------------------------------------------------

def _sse(content: str | None = None, finish: str | None = None) -> bytes:
    """Build a minimal SSE chunk as the server would emit it."""
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    choices = [{"index": 0, "delta": delta, "finish_reason": finish}]
    payload = {"id": "x", "object": "chat.completion.chunk",
                "created": 0, "model": "m", "choices": choices}
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _collect(chunks: list[bytes]) -> list[bytes]:
    """Run _filter_thinking_stream on a fixed list of chunks."""
    async def _source():
        for c in chunks:
            yield c
    return [c async for c in _filter_thinking_stream(_source())]


class TestFilterThinkingStream:
    def test_passthrough_no_think(self):
        chunks = [_sse("Hello"), _sse(" world"), b"data: [DONE]\n\n"]
        out = asyncio.run(_collect(chunks))
        texts = [json.loads(c.decode()[6:])["choices"][0]["delta"].get("content", "")
                 for c in out if c != b"data: [DONE]\n\n"]
        assert "".join(texts) == "Hello world"

    def test_complete_think_block_stripped(self):
        # Think block entirely in one chunk
        chunks = [_sse("<think>internal</think>Answer"), b"data: [DONE]\n\n"]
        out = asyncio.run(_collect(chunks))
        texts = [json.loads(c.decode()[6:])["choices"][0]["delta"].get("content", "")
                 for c in out if c != b"data: [DONE]\n\n"]
        assert "".join(texts).strip() == "Answer"

    def test_think_block_spanning_chunks(self):
        # Think opens in chunk 1, closes in chunk 2
        chunks = [
            _sse("<think>part one"),
            _sse(" part two</think>Real answer"),
            b"data: [DONE]\n\n",
        ]
        out = asyncio.run(_collect(chunks))
        texts = [json.loads(c.decode()[6:])["choices"][0]["delta"].get("content", "")
                 for c in out if c != b"data: [DONE]\n\n"]
        assert "".join(texts).strip() == "Real answer"

    def test_done_sentinel_passes_through(self):
        chunks = [b"data: [DONE]\n\n"]
        out = asyncio.run(_collect(chunks))
        assert out == [b"data: [DONE]\n\n"]

    def test_non_data_chunk_passes_through(self):
        raw = b"keep-alive: \n\n"
        out = asyncio.run(_collect([raw]))
        assert out == [raw]

    def test_finish_reason_chunk_passes_through(self):
        # Finish chunk has no content — should not be dropped
        finish_chunk = _sse(content=None, finish="stop")
        out = asyncio.run(_collect([finish_chunk, b"data: [DONE]\n\n"]))
        assert len(out) == 2  # finish chunk + DONE

    def test_entirely_think_chunk_dropped(self):
        # A chunk whose full content is inside a think block — should be dropped
        chunks = [_sse("<think>internal only"), _sse(" more thinking"), b"data: [DONE]\n\n"]
        out = asyncio.run(_collect(chunks))
        # Only the DONE sentinel should remain
        data_chunks = [c for c in out if c != b"data: [DONE]\n\n"]
        assert data_chunks == []


# ---------------------------------------------------------------------------
# ThinkingStreamFilter (text-level, used by CLI chat and /v1/completions)
# ---------------------------------------------------------------------------

class TestThinkingStreamFilter:
    def test_passthrough_no_think(self):
        filt = ThinkingStreamFilter()
        assert filt.feed("hello world") == "hello world"
        assert filt.collected_text() == "hello world"

    def test_complete_block_stripped(self):
        filt = ThinkingStreamFilter()
        visible = filt.feed("<think>internal</think>Answer")
        assert "think" not in visible
        assert "Answer" in visible
        assert filt.collected_text() == visible

    def test_cross_chunk_block(self):
        filt = ThinkingStreamFilter()
        v1 = filt.feed("<think>step one")   # enters think state
        v2 = filt.feed(" step two</think>Real answer")  # exits think state
        assert v1 == ""
        assert v2 == "Real answer"
        assert filt.collected_text() == "Real answer"

    def test_collected_text_excludes_thinking(self):
        filt = ThinkingStreamFilter()
        filt.feed("<think>secret reasoning</think>")
        filt.feed("The answer is 42")
        assert "secret" not in filt.collected_text()
        assert filt.collected_text() == "The answer is 42"

    def test_empty_string(self):
        filt = ThinkingStreamFilter()
        assert filt.feed("") == ""
        assert filt.collected_text() == ""

    def test_multiple_think_blocks(self):
        filt = ThinkingStreamFilter()
        filt.feed("<think>step 1</think>")
        v = filt.feed("part A<think>step 2</think>part B")
        assert v == "part Apart B"
        assert filt.collected_text() == "part Apart B"


# ---------------------------------------------------------------------------
# _is_thinking_model
# ---------------------------------------------------------------------------

class TestIsThinkingModel:
    @pytest.mark.parametrize("model_id", [
        "qwen3:8b", "qwen3-14b", "qwen3:latest",
        "deepseek-r1:7b", "deepseek-r1-distill-14b",
        "qwq-32b", "marco-o1", "deepthink-7b",
    ])
    def test_thinking_models_detected(self, model_id: str):
        assert _is_thinking_model(model_id)

    @pytest.mark.parametrize("model_id", [
        "llama3.2:3b", "qwen2.5-7b", "phi4", "gemma2:9b",
        "mistral:7b", "deepseek-coder-v2", "mixtral:8x7b",
    ])
    def test_non_thinking_models_not_detected(self, model_id: str):
        assert not _is_thinking_model(model_id)

    def test_case_insensitive(self):
        assert _is_thinking_model("Qwen3:8B")
        assert _is_thinking_model("DeepSeek-R1:7B")


# ---------------------------------------------------------------------------
# _is_chat_model
# ---------------------------------------------------------------------------

class TestIsChatModel:
    @pytest.mark.parametrize("model_id,source", [
        ("llama3.2:3b", "ollama"),
        ("qwen3:8b", "ollama"),
        ("mlx-community/Qwen3-8B", "mlx"),
        ("lmstudio-community/gemma", "lmstudio"),
    ])
    def test_chat_models_pass(self, model_id: str, source: str):
        assert _is_chat_model(model_id, source)

    @pytest.mark.parametrize("model_id,source", [
        ("nomic-embed-text", "ollama"),
        ("all-minilm-l6-v2", "ollama"),
        ("cross-encoder/ms-marco", "lmstudio"),
        ("clip-vit-base-patch32", "mlx"),
        ("whisper-base", "ollama"),
        ("llama3.2:3b", "hf_cache"),    # non-servable source
        ("llama3.2:3b", "gguf"),
    ])
    def test_non_chat_models_rejected(self, model_id: str, source: str):
        assert not _is_chat_model(model_id, source)


# ---------------------------------------------------------------------------
# Message.normalize_content
# ---------------------------------------------------------------------------

class TestMessageNormalizeContent:
    def test_string_passthrough(self):
        m = Message(role="user", content="hello")
        assert m.content == "hello"

    def test_none_becomes_empty(self):
        m = Message(role="user", content=None)
        assert m.content == ""

    def test_multimodal_list_text_extracted(self):
        content = [{"type": "text", "text": "What is this?"}, {"type": "image_url", "url": "..."}]
        m = Message(role="user", content=content)
        assert m.content == "What is this?"

    def test_multimodal_multiple_text_parts(self):
        content = [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}]
        m = Message(role="user", content=content)
        assert m.content == "Part 1 Part 2"

    def test_invalid_role_raises(self):
        with pytest.raises(Exception):
            Message(role="invalid_role", content="hi")

    def test_tool_role_accepted(self):
        m = Message(role="tool", content="result", tool_call_id="call_1")
        assert m.role == "tool"


# ---------------------------------------------------------------------------
# ChatRequest validation
# ---------------------------------------------------------------------------

class TestChatRequestValidation:
    def _base(self, **kwargs):
        return {
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": "hi"}],
            **kwargs,
        }

    def test_valid_minimal(self):
        req = ChatRequest(**self._base())
        assert req.model == "llama3.2:3b"
        assert req.stream is False  # OpenAI spec default

    def test_invalid_profile_raises(self):
        with pytest.raises(Exception):
            ChatRequest(**self._base(profile="turbo"))

    def test_temperature_out_of_range(self):
        with pytest.raises(Exception):
            ChatRequest(**self._base(temperature=3.0))

    def test_temperature_zero_is_valid(self):
        req = ChatRequest(**self._base(temperature=0.0))
        assert req.temperature == 0.0

    def test_max_tokens_negative_raises(self):
        with pytest.raises(Exception):
            ChatRequest(**self._base(max_tokens=0))

    def test_openai_extra_fields_accepted(self):
        # These are sent by every OpenAI-compatible client — must not 422
        req = ChatRequest(**self._base(
            stop=["\n"],
            n=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            user="test-user",
            seed=42,
            response_format={"type": "text"},
            tools=[],
            tool_choice="none",
        ))
        assert req.seed == 42


# ---------------------------------------------------------------------------
# CompletionRequest validation
# ---------------------------------------------------------------------------

class TestCompletionRequestValidation:
    def test_string_prompt(self):
        req = CompletionRequest(model="qwen3:8b", prompt="def hello(")
        assert req.prompt == "def hello("

    def test_list_prompt_accepted(self):
        req = CompletionRequest(model="qwen3:8b", prompt=["line1", "line2"])
        assert req.prompt == ["line1", "line2"]

    def test_suffix_fim(self):
        req = CompletionRequest(model="qwen3:8b", prompt="def foo(", suffix="):\n    pass")
        assert req.suffix == "):\n    pass"

    def test_default_max_tokens(self):
        req = CompletionRequest(model="qwen3:8b", prompt="x")
        assert req.max_tokens == 256

    def test_stream_defaults_false(self):
        req = CompletionRequest(model="qwen3:8b", prompt="x")
        assert req.stream is False


# ---------------------------------------------------------------------------
# Regression: _THINK_OPEN / _THINK_CLOSE importable and used by filter
# ---------------------------------------------------------------------------

class TestThinkConstants:
    """Verify _THINK_OPEN/_THINK_CLOSE are exported and used correctly.

    These constants are referenced by the inline think-tag state machine in
    /v1/completions streaming — importing them here acts as a canary that
    catches the NameError that would otherwise crash thinking-model streams.
    """

    def test_think_open_is_open_tag(self):
        assert _THINK_OPEN == "<think>"

    def test_think_close_is_close_tag(self):
        assert _THINK_CLOSE == "</think>"

    def test_filter_uses_constants_correctly(self):
        filt = ThinkingStreamFilter()
        visible = filt.feed(f"{_THINK_OPEN}internal reasoning{_THINK_CLOSE}answer")
        assert visible == "answer"
        assert filt.collected_text() == "answer"

    def test_filter_across_chunk_boundary(self):
        """Think open/close can arrive in separate chunks."""
        filt = ThinkingStreamFilter()
        assert filt.feed(_THINK_OPEN + "hidden") == ""
        assert filt.feed("still hidden") == ""
        assert filt.feed(_THINK_CLOSE + "visible") == "visible"

    def test_inline_state_machine_matches_filter(self):
        """The inline state machine in /v1/completions streaming must behave
        identically to ThinkingStreamFilter for the same input."""
        text = f"{_THINK_OPEN}think{_THINK_CLOSE}answer"
        # Simulate inline state machine from server.py _completions_stream
        buf = text
        parts: list[str] = []
        in_think = False
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
                    parts.append(buf)
                    buf = ""
                else:
                    if pos > 0:
                        parts.append(buf[:pos])
                    buf = buf[pos + len(_THINK_OPEN):]
                    in_think = True
        assert "".join(parts) == "answer"
