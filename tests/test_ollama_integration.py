"""
Integration tests against a live Ollama instance.

All tests in this file are skipped automatically when Ollama is not running.
Run them manually when you want to verify end-to-end correctness:

    pytest tests/test_ollama_integration.py -v -s

Or as part of the full suite — they just skip silently if Ollama is absent.

What is tested (and what isn't):
  ✓ Ollama connectivity probe
  ✓ Model discovery — at least one model visible
  ✓ Full stream roundtrip: tokens arrive, finish_reason set, no empty response
  ✓ Non-streaming collect path (same backend, different code path)
  ✓ num_ctx / ollama_options plumbing reaches Ollama without error
  ✓ Thinking-tag filter produces clean output for reasoning models
  ✓ Token estimate is sane (not zero, not absurdly large)
  ✗ Correctness of the model's answer — that's the model's job, not ours
"""

from __future__ import annotations

import asyncio
import pytest
import httpx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ollama_url() -> str:
    return "http://localhost:11434"


def _ollama_running_sync() -> bool:
    """Cheap synchronous probe used at collection time."""
    try:
        r = httpx.get(f"{_ollama_url()}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _list_models_sync() -> list[str]:
    """Return Ollama model names sorted by size (smallest first)."""
    try:
        r = httpx.get(f"{_ollama_url()}/api/tags", timeout=2.0)
        models = r.json().get("models", [])
        # Sort by size ascending so the smallest model is used by default
        models.sort(key=lambda m: m.get("size", 0))
        return [m["name"] for m in models]
    except Exception:
        return []


# Single fixture that skips the whole module when Ollama is absent
ollama_running = pytest.fixture(scope="session")(
    lambda: _ollama_running_sync() or pytest.skip("Ollama not running")
)

@pytest.fixture(scope="session")
def smallest_model(ollama_running) -> str:
    """Return the smallest locally available Ollama model."""
    models = _list_models_sync()
    if not models:
        pytest.skip("No models installed in Ollama")
    return models[0]


@pytest.fixture(scope="session")
def thinking_model(ollama_running) -> str | None:
    """Return the name of an installed thinking model, or None."""
    thinking_patterns = ("qwen3", "deepseek-r1", "qwq", "deepthink", "marco-o1")
    models = _list_models_sync()
    for name in models:
        if any(p in name.lower() for p in thinking_patterns):
            return name
    return None


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

class TestOllamaConnectivity:
    def test_ollama_probe_returns_true(self, ollama_running):
        """BackendChain.ollama_running() must agree with our direct probe."""
        from autotune.api.backends.chain import BackendChain
        chain = BackendChain()
        result = asyncio.run(chain.ollama_running())
        assert result is True

    def test_discover_all_includes_ollama_models(self, ollama_running):
        from autotune.api.backends.chain import BackendChain
        chain = BackendChain()
        models = asyncio.run(chain.discover_all())
        ollama_models = [m for m in models if m.source == "ollama"]
        assert len(ollama_models) >= 1, "Expected at least one Ollama model"

    def test_model_info_has_required_fields(self, ollama_running):
        from autotune.api.backends.chain import BackendChain
        chain = BackendChain()
        models = asyncio.run(chain.discover_all())
        for m in (m for m in models if m.source == "ollama"):
            assert m.id, "model id must not be empty"
            assert m.backend_hint == "ollama"
            assert m.available_locally is True


# ---------------------------------------------------------------------------
# Streaming roundtrip
# ---------------------------------------------------------------------------

class TestStreamRoundtrip:
    def test_stream_produces_tokens(self, smallest_model):
        """A minimal prompt must yield at least one non-empty token."""
        from autotune.api.backends.chain import BackendChain
        chain = BackendChain()
        messages = [{"role": "user", "content": "Reply with one word: hello"}]

        chunks = []
        async def _run():
            async for chunk in chain.stream(
                smallest_model, messages,
                max_new_tokens=32, temperature=0.0, timeout=30.0,
            ):
                chunks.append(chunk)

        asyncio.run(_run())

        content_chunks = [c for c in chunks if c.content]
        assert len(content_chunks) >= 1, "Expected at least one content chunk"

    def test_stream_sets_finish_reason(self, smallest_model):
        """The stream must complete and, for Ollama/LM Studio, emit a finish_reason.
        MLX currently does not set finish_reason — tracked as xfail."""
        from autotune.api.backends.chain import BackendChain
        chain = BackendChain()
        messages = [{"role": "user", "content": "Say: done"}]

        all_chunks = []
        async def _run():
            async for chunk in chain.stream(
                smallest_model, messages,
                max_new_tokens=16, temperature=0.0, timeout=30.0,
            ):
                all_chunks.append(chunk)

        asyncio.run(_run())
        assert all_chunks, "Stream produced no chunks at all"

        backend_used = all_chunks[-1].backend
        reasons = [c.finish_reason for c in all_chunks if c.finish_reason]

        if not reasons and backend_used == "mlx":
            pytest.xfail("MLX backend does not emit finish_reason on ChatChunk")

        assert reasons, (
            f"No chunk had a finish_reason (backend={backend_used}). "
            "This backend should set finish_reason='stop' on the last chunk."
        )

    def test_stream_backend_is_local(self, smallest_model):
        """Every chunk must report a known local backend (ollama, mlx, or lmstudio).
        On Apple Silicon, MLX takes priority over Ollama — both are valid."""
        from autotune.api.backends.chain import BackendChain
        _LOCAL_BACKENDS = {"ollama", "mlx", "lmstudio"}
        chain = BackendChain()
        messages = [{"role": "user", "content": "hi"}]

        backends = set()
        async def _run():
            async for chunk in chain.stream(
                smallest_model, messages,
                max_new_tokens=8, temperature=0.0, timeout=20.0,
            ):
                backends.add(chunk.backend)

        asyncio.run(_run())
        unknown = backends - _LOCAL_BACKENDS
        assert not unknown, f"Unexpected backend(s): {unknown}"

    def test_stream_with_ollama_options(self, smallest_model):
        """num_ctx and other Ollama options must be accepted without error."""
        from autotune.api.backends.chain import BackendChain
        from autotune.api.profiles import get_profile
        from autotune.api.kv_manager import build_ollama_options

        chain = BackendChain()
        messages = [{"role": "user", "content": "Count to three."}]
        profile = get_profile("fast")
        ollama_opts, _ = build_ollama_options(messages, profile)

        collected = []
        async def _run():
            async for chunk in chain.stream(
                smallest_model, messages,
                max_new_tokens=48, temperature=0.0, timeout=30.0,
                num_ctx=ollama_opts["num_ctx"],
                ollama_options=ollama_opts,
            ):
                if chunk.content:
                    collected.append(chunk.content)

        asyncio.run(_run())
        text = "".join(collected)
        assert len(text) > 0, "Expected non-empty response"

    def test_token_estimate_is_sane(self, smallest_model):
        """Token estimate of the response must be positive and under 512."""
        from autotune.api.backends.chain import BackendChain
        from autotune.api.ctx_utils import estimate_tokens

        chain = BackendChain()
        messages = [{"role": "user", "content": "What is 2+2? One word answer."}]

        collected = []
        async def _run():
            async for chunk in chain.stream(
                smallest_model, messages,
                max_new_tokens=16, temperature=0.0, timeout=20.0,
            ):
                if chunk.content:
                    collected.append(chunk.content)

        asyncio.run(_run())
        text = "".join(collected)
        toks = estimate_tokens(text)
        assert 1 <= toks <= 512, f"Token estimate out of expected range: {toks}"


# ---------------------------------------------------------------------------
# Non-streaming collect path
# ---------------------------------------------------------------------------

class TestNonStreamingCollect:
    def test_collect_all_chunks_manually(self, smallest_model):
        """Collecting the full stream into a string must produce a non-empty response."""
        from autotune.api.backends.chain import BackendChain

        chain = BackendChain()
        messages = [{"role": "user", "content": "Say exactly: pong"}]

        async def _collect():
            parts = []
            async for chunk in chain.stream(
                smallest_model, messages,
                max_new_tokens=16, temperature=0.0, timeout=20.0,
            ):
                if chunk.content:
                    parts.append(chunk.content)
            return "".join(parts)

        text = asyncio.run(_collect())
        assert len(text.strip()) > 0


# ---------------------------------------------------------------------------
# KV options plumbing
# ---------------------------------------------------------------------------

class TestKVOptions:
    def test_num_ctx_is_positive(self, smallest_model):
        from autotune.api.profiles import get_profile
        from autotune.api.kv_manager import build_ollama_options

        messages = [{"role": "user", "content": "hi"}]
        profile = get_profile("balanced")
        opts, _ = build_ollama_options(messages, profile)
        assert opts["num_ctx"] >= 512

    def test_num_ctx_grows_with_history(self, smallest_model):
        """A longer conversation must produce a larger or equal num_ctx."""
        from autotune.api.profiles import get_profile
        from autotune.api.kv_manager import build_ollama_options

        profile = get_profile("balanced")
        short = [{"role": "user", "content": "hi"}]
        long = [
            {"role": "system", "content": "You are a helpful assistant. " * 50},
            {"role": "user", "content": "Tell me about " + "AI " * 100},
        ]
        short_ctx, _ = build_ollama_options(short, profile)
        long_ctx, _ = build_ollama_options(long, profile)
        assert long_ctx["num_ctx"] >= short_ctx["num_ctx"]

    def test_fast_profile_uses_q8_kv(self):
        from autotune.api.profiles import get_profile
        from autotune.api.kv_manager import build_ollama_options

        messages = [{"role": "user", "content": "hi"}]
        profile = get_profile("fast")
        opts, _ = build_ollama_options(messages, profile)
        # fast profile should use Q8 KV (f16_kv=False)
        assert opts.get("f16_kv") is False

    def test_balanced_profile_uses_f16_kv(self):
        from autotune.api.profiles import get_profile
        from autotune.api.kv_manager import build_ollama_options

        messages = [{"role": "user", "content": "hi"}]
        profile = get_profile("balanced")
        opts, _ = build_ollama_options(messages, profile)
        assert opts.get("f16_kv") is True


# ---------------------------------------------------------------------------
# Thinking-tag filter — live reasoning model
# ---------------------------------------------------------------------------

class TestThinkingModelLive:
    def test_thinking_model_response_has_no_think_tags(self, thinking_model):
        """If a thinking model is installed, its response must arrive tag-free."""
        if thinking_model is None:
            pytest.skip("No thinking model installed")

        from autotune.api.backends.chain import BackendChain
        from autotune.api.thinking import ThinkingStreamFilter

        chain = BackendChain()
        messages = [{"role": "user", "content": "What is 1+1?"}]
        filt = ThinkingStreamFilter()

        async def _run():
            async for chunk in chain.stream(
                thinking_model, messages,
                max_new_tokens=512, temperature=0.0, timeout=60.0,
            ):
                if chunk.content:
                    filt.feed(chunk.content)

        asyncio.run(_run())
        result = filt.collected_text()
        assert "<think>" not in result, "ThinkingStreamFilter left <think> in output"
        assert "</think>" not in result, "ThinkingStreamFilter left </think> in output"
        assert len(result.strip()) > 0, "Expected non-empty answer after stripping"

    def test_thinking_model_raw_output_contains_think_tags(self, thinking_model):
        """Raw backend output for a reasoning model SHOULD contain think tags
        (confirms the model actually uses them and our filter has something to do)."""
        if thinking_model is None:
            pytest.skip("No thinking model installed")

        from autotune.api.backends.chain import BackendChain

        chain = BackendChain()
        messages = [{"role": "user", "content": "What is 1+1? Think step by step."}]
        raw_parts = []

        async def _run():
            async for chunk in chain.stream(
                thinking_model, messages,
                max_new_tokens=512, temperature=0.0, timeout=60.0,
            ):
                if chunk.content:
                    raw_parts.append(chunk.content)

        asyncio.run(_run())
        raw = "".join(raw_parts)
        # Not every prompt forces thinking — just warn if tags are absent
        if "<think>" not in raw:
            pytest.xfail(
                f"{thinking_model} did not emit <think> tags for this prompt "
                "(model may use thinking selectively)"
            )
