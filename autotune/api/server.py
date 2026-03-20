"""
autotune API server — OpenAI-compatible + autotune-specific endpoints.

Any client that speaks the OpenAI API (Python SDK, curl, etc.) can use this
server just by setting base_url="http://localhost:8765/v1".

Optimization profiles exposed via the `X-Autotune-Profile` header or
`profile` field in the request body.

KV-cache strategy:
  - System prompt is always sent first in every request (never stripped).
  - Ollama naturally caches the prompt prefix across turns in the same session.
  - For HF API, context is stateless — full history sent each time.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

import psutil
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from autotune.hardware.profiler import profile_hardware
from .backends.chain import BackendChain, ModelNotAvailableError, get_chain
from .backends.openai_compat import AuthError, BackendError
from .conversation import ConversationManager, get_conv_manager
from .ctx_utils import estimate_tokens
from .kv_manager import build_ollama_options
from .hardware_tuner import get_tuner
from .profiles import PROFILES, get_profile

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

_VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {v!r}")
        return v


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stream: bool = True
    # autotune extensions
    profile: str = "balanced"
    conversation_id: Optional[str] = None
    system: Optional[str] = None             # shorthand system prompt

    @field_validator("profile")
    @classmethod
    def profile_must_be_valid(cls, v: str) -> str:
        if v not in PROFILES:
            raise ValueError(f"profile must be one of {list(PROFILES.keys())}, got {v!r}")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("max_tokens must be >= 1")
        return v


class ConversationCreateRequest(BaseModel):
    model_id: str
    profile: str = "balanced"
    system_prompt: Optional[str] = None
    title: Optional[str] = None


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_hw = None
_chain: Optional[BackendChain] = None
_conv: Optional[ConversationManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hw, _chain, _conv
    _hw = profile_hardware()
    _chain = get_chain()
    _conv = get_conv_manager()
    # Pre-probe backends
    await _chain.ollama_running()
    await _chain.lmstudio_running()
    yield


app = FastAPI(
    title="autotune LLM API",
    description="OpenAI-compatible local LLM API with hardware optimization",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Inference task scheduler — bounded FIFO queue
#
# Local LLMs are single-threaded on the hardware (GPU/ANE/unified memory).
# Allowing concurrent requests causes:
#   • memory contention  (KV caches compete for the same physical RAM)
#   • context fragmentation (Ollama allocates separate KV arenas per request)
#   • mutual slowdown  (both requests take 2× longer — zero net throughput gain)
#
# Design: true first-in-first-out queue with a hard depth limit.
#   1. Up to MAX_CONCURRENT requests run at the same time (default: 1).
#   2. Up to MAX_QUEUED additional requests wait in strict arrival order.
#   3. If the queue is full → 429 immediately (caller retries; no further wait).
#   4. If a waiting request times out → its slot is passed to the next waiter.
#
# Env overrides:
#   AUTOTUNE_MAX_CONCURRENT   (default: 1)   — parallel inference slots
#   AUTOTUNE_MAX_QUEUED       (default: 8)   — max waiting requests
#   AUTOTUNE_WAIT_TIMEOUT     (default: 120) — seconds a request waits for a slot
# ---------------------------------------------------------------------------

import asyncio as _asyncio
import os as _os
from collections import deque as _deque

_MAX_CONCURRENT  = int(_os.environ.get("AUTOTUNE_MAX_CONCURRENT", "1"))
_MAX_QUEUED      = int(_os.environ.get("AUTOTUNE_MAX_QUEUED",     "8"))
_WAIT_TIMEOUT    = float(_os.environ.get("AUTOTUNE_WAIT_TIMEOUT", "120.0"))


class _QueueFullError(Exception):
    def __init__(self, depth: int) -> None:
        self.depth = depth


class _InferenceQueue:
    """
    Bounded FIFO queue for inference requests.

    Callers must call ``await release()`` in a ``finally`` block after every
    successful ``await acquire()``.
    """

    def __init__(self, max_concurrent: int, max_queued: int) -> None:
        self._slots  = max_concurrent
        self._max_q  = max_queued
        self._active = 0
        self._waiters: _deque[_asyncio.Future] = _deque()
        self._lock   = _asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        """Number of requests currently waiting for a slot (excludes active)."""
        return sum(1 for f in self._waiters if not f.done())

    async def acquire(self, timeout: float = 120.0) -> None:
        """
        Wait for a slot in strict FIFO order.

        Raises
        ------
        _QueueFullError   — immediately, if the waiting queue is already at capacity.
        asyncio.TimeoutError — if ``timeout`` seconds elapse before a slot is granted.
        """
        loop = _asyncio.get_running_loop()
        fut: _asyncio.Future | None = None

        async with self._lock:
            if self._active < self._slots:
                self._active += 1
                return                          # fast path — slot free immediately

            live = sum(1 for f in self._waiters if not f.done())
            if live >= self._max_q:
                raise _QueueFullError(live)     # queue full — reject right away

            fut = loop.create_future()
            self._waiters.append(fut)

        # Wait outside the lock.  asyncio.shield keeps the inner future alive
        # even when wait_for cancels the wrapper on timeout.
        try:
            await _asyncio.wait_for(_asyncio.shield(fut), timeout=timeout)
        except _asyncio.TimeoutError:
            # Two scenarios:
            #  (a) fut is still in the deque  → remove it; no slot was granted.
            #  (b) release() already popped fut and set_result'd it → we hold a
            #      slot we cannot use; pass it to the next waiter or free it.
            async with self._lock:
                try:
                    self._waiters.remove(fut)
                    # Scenario (a): cleanly removed, no slot to give back.
                except ValueError:
                    # Scenario (b): slot was handed to us; we must pass it on.
                    self._pass_slot_locked()
            raise

    def _pass_slot_locked(self) -> None:
        """Pass the current slot to the next non-done waiter, or decrement active.
        Must be called while self._lock is held."""
        while self._waiters:
            nxt = self._waiters[0]
            self._waiters.popleft()
            if not nxt.done():
                nxt.set_result(None)
                return          # active count stays the same — handed off
        self._active -= 1       # nobody left waiting

    async def release(self) -> None:
        """Release the current slot.  Must be called after every successful acquire()."""
        async with self._lock:
            self._pass_slot_locked()


_inference_queue: _InferenceQueue | None = None


def _get_queue() -> _InferenceQueue:
    """Return the module-level queue, creating it lazily."""
    global _inference_queue
    if _inference_queue is None:
        _inference_queue = _InferenceQueue(_MAX_CONCURRENT, _MAX_QUEUED)
    return _inference_queue

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_chunk_json(content: str, model: str, chunk_id: str, finish_reason=None) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _make_completion_json(
    content: str,
    model: str,
    chunk_id: str,
    profile: str,
    backend: str,
    ttft_ms: float,
    tps: float,
    prompt_tokens: int,
    conv_id: Optional[str],
) -> dict:
    return {
        "id": chunk_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": estimate_tokens(content),
            "total_tokens": prompt_tokens + estimate_tokens(content),
        },
        "autotune": {
            "profile": profile,
            "backend": backend,
            "ttft_ms": round(ttft_ms, 1),
            "tokens_per_sec": round(tps, 1),
            "conversation_id": conv_id,
        },
    }


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    chain = get_chain()
    all_models = await chain.discover_all()

    data = []
    for m in all_models:
        data.append({
            "id": m.id,
            "object": "model",
            "created": 0,
            "owned_by": m.id.split("/")[0] if "/" in m.id else "local",
            "autotune": {
                "source": m.source,
                "available_locally": m.available_locally,
                "size_gb": m.size_gb,
                "backend": m.backend_hint,
            },
        })

    # Always add HF API as an option if token is set
    if os.environ.get("HF_TOKEN"):
        data.append({
            "id": "hf_api/*",
            "object": "model",
            "created": 0,
            "owned_by": "huggingface",
            "autotune": {
                "source": "hf_api",
                "available_locally": False,
                "note": "Any HuggingFace model ID works via HF Inference API",
            },
        })

    return {"object": "list", "data": data}


@app.get("/v1/models/local")
async def list_local_models():
    chain = get_chain()
    all_models = await chain.discover_all()
    local = [m for m in all_models if m.available_locally]
    return {"object": "list", "data": [
        {"id": m.id, "source": m.source, "size_gb": m.size_gb, "backend": m.backend_hint}
        for m in local
    ]}


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest,
    x_autotune_profile: Optional[str] = Header(None),
    x_conversation_id: Optional[str] = Header(None),
):
    # ── Bounded FIFO inference queue ─────────────────────────────────────
    queue = _get_queue()
    try:
        await queue.acquire(timeout=_WAIT_TIMEOUT)
    except _QueueFullError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "queue_full",
                "message": (
                    f"Request queue is full ({exc.depth}/{_MAX_QUEUED} waiting). "
                    "Retry when a slot is available."
                ),
                "queue_depth": exc.depth,
                "max_queued": _MAX_QUEUED,
            },
        )
    except _asyncio.TimeoutError:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "queue_timeout",
                "message": (
                    f"Request waited {_WAIT_TIMEOUT:.0f}s for an inference slot "
                    "but none became available. Retry later."
                ),
                "wait_timeout_sec": _WAIT_TIMEOUT,
            },
        )

    try:
        response = await _chat_completions_inner(req, x_autotune_profile, x_conversation_id)
    except Exception:
        # Release slot immediately on any error (generator never runs)
        await queue.release()
        raise

    if req.stream:
        # For streaming: the response body is iterated lazily by the ASGI server
        # AFTER this function returns.  The queue slot must stay held until the
        # last byte has been sent — wrap the body iterator to release on completion
        # or client disconnect.
        orig_iter = response.body_iterator
        async def _stream_and_release():
            try:
                async for chunk in orig_iter:
                    yield chunk
            finally:
                await queue.release()
        response.body_iterator = _stream_and_release()
        return response
    else:
        # Non-streaming: response is fully collected, safe to release now
        await queue.release()
        return response


async def _chat_completions_inner(
    req: ChatRequest,
    x_autotune_profile: Optional[str],
    x_conversation_id: Optional[str],
):
    """Core chat completions logic, called once the task semaphore is held."""
    profile_name = x_autotune_profile or req.profile or "balanced"
    conv_id = x_conversation_id or req.conversation_id

    try:
        profile = get_profile(profile_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chain = get_chain()
    conv_mgr = get_conv_manager()
    tuner = get_tuner()

    # ── Conversation management ──────────────────────────────────────────
    raw_messages = [m.model_dump() for m in req.messages]

    if conv_id:
        conv = conv_mgr.get(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail=f"Conversation {conv_id!r} not found")
        # Use system prompt from request (if provided) or stored one
        system = req.system or conv.get("system_prompt")
        if req.system:
            conv_mgr.update_system_prompt(conv_id, req.system)
        # Build context from history
        user_text = next(
            (m["content"] for m in reversed(raw_messages) if m["role"] == "user"), ""
        )
        messages, _ = conv_mgr.build_context(
            conv_id,
            profile.max_context_tokens,
            new_user_message=user_text,
            reserved_for_output=profile.max_new_tokens,
        )
        conv_mgr.add_message(conv_id, "user", user_text)
    else:
        # Stateless mode: use messages as-is
        messages = raw_messages
        if req.system:
            # Prepend system prompt if not already there
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": req.system})

    # ── Generation params (profile defaults, request overrides) ─────────
    max_tokens = req.max_tokens or profile.max_new_tokens
    temperature = req.temperature if req.temperature is not None else profile.temperature
    top_p = req.top_p if req.top_p is not None else profile.top_p
    rep_penalty = req.repetition_penalty if req.repetition_penalty is not None else profile.repetition_penalty
    timeout = profile.request_timeout_sec

    # ── Dynamic KV-cache sizing ──────────────────────────────────────────
    # Compute only the context window this request actually needs instead of
    # always allocating profile.max_context_tokens.  Reduces both unified
    # memory pressure and KV-cache init latency (direct TTFT improvement).
    ollama_opts = build_ollama_options(messages, profile)

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    first_token_time: Optional[float] = None
    full_content: list[str] = []
    backend_name = "unknown"

    async def generate_stream() -> AsyncGenerator[bytes, None]:
        nonlocal first_token_time, backend_name

        # Apply hardware optimizations inside the generator so they are
        # released in the finally block when streaming completes.
        tuner._apply(profile_name)
        try:
            async for chunk in chain.stream(
                req.model,
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                timeout=timeout,
                num_ctx=ollama_opts["num_ctx"],
                ollama_options=ollama_opts,
            ):
                backend_name = chunk.backend
                if first_token_time is None and chunk.content:
                    first_token_time = time.time()

                if chunk.content:
                    full_content.append(chunk.content)
                    yield _make_chunk_json(chunk.content, req.model, chunk_id).encode()

                if chunk.finish_reason:
                    yield _make_chunk_json("", req.model, chunk_id, chunk.finish_reason).encode()
                    break

        except ModelNotAvailableError as e:
            err = json.dumps({"error": {"message": str(e), "type": "model_not_found"}})
            yield f"data: {err}\n\n".encode()
        except AuthError as e:
            err = json.dumps({"error": {"message": str(e), "type": "auth_error"}})
            yield f"data: {err}\n\n".encode()
        except BackendError as e:
            err = json.dumps({"error": {"message": str(e), "type": "backend_error"}})
            yield f"data: {err}\n\n".encode()
        finally:
            tuner._restore()
            # Log metrics to DB + conversation
            elapsed = time.time() - start_time
            content = "".join(full_content)
            comp_tokens = estimate_tokens(content)
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
            tps = comp_tokens / max(elapsed, 0.01)

            if conv_id and content:
                conv_mgr.add_message(
                    conv_id, "assistant", content,
                    ttft_ms=ttft_ms, tokens_per_sec=tps, backend=backend_name,
                )

            try:
                from autotune.db.store import get_db
                from autotune.db.fingerprint import hardware_to_db_dict
                hw = _hw
                if hw:
                    db = get_db()
                    hw_dict = hardware_to_db_dict(hw)
                    db.upsert_hardware(hw_dict)
                    db.log_run({
                        "model_id": req.model,
                        "hardware_id": hw_dict["id"],
                        "quant": "unknown",
                        "context_len": ollama_opts["num_ctx"],
                        "n_gpu_layers": -1,
                        "tokens_per_sec": round(tps, 1),
                        "gen_tokens_per_sec": round(tps, 1),
                        "ttft_ms": round(ttft_ms, 1),
                        "notes": (
                            f"profile={profile_name} backend={backend_name} "
                            f"f16_kv={ollama_opts.get('f16_kv', True)}"
                        ),
                    })
            except Exception as _db_exc:
                logger.debug("metrics DB log failed: %s", _db_exc)

        yield b"data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    # ── Non-streaming: collect full response ─────────────────────────────
    collected: list[str] = []
    backend_used = "unknown"
    t0 = time.time()
    ttft = 0.0
    tuner._apply(profile_name)
    try:
        async for chunk in chain.stream(
            req.model, messages,
            max_new_tokens=max_tokens, temperature=temperature,
            top_p=top_p, repetition_penalty=rep_penalty, timeout=timeout,
            num_ctx=ollama_opts["num_ctx"],
            ollama_options=ollama_opts,
        ):
            if not collected and chunk.content:
                ttft = (time.time() - t0) * 1000
            if chunk.content:
                collected.append(chunk.content)
            backend_used = chunk.backend
    except (ModelNotAvailableError, AuthError, BackendError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        tuner._restore()

    content_out = "".join(collected)
    elapsed2 = time.time() - t0
    comp_tokens = estimate_tokens(content_out)
    tps2 = comp_tokens / max(elapsed2, 0.01)
    prompt_tokens = estimate_tokens("".join(m.get("content", "") for m in messages))

    if conv_id and content_out:
        conv_mgr.add_message(conv_id, "assistant", content_out,
                              ttft_ms=ttft, tokens_per_sec=tps2, backend=backend_used)

    return _make_completion_json(
        content_out, req.model, chunk_id, profile_name,
        backend_used, ttft, tps2, prompt_tokens, conv_id,
    )


# ---------------------------------------------------------------------------
# /api/* — autotune-specific endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    from autotune.api.kv_manager import memory_pressure_snapshot
    chain = get_chain()
    ollama = await chain.ollama_running()
    lms = await chain.lmstudio_running()
    hf_token = bool(os.environ.get("HF_TOKEN"))
    q = _get_queue()
    mem = memory_pressure_snapshot()
    return {
        "status": "ok",
        "version": "0.1.0",
        "backends": {
            "ollama": ollama,
            "lmstudio": lms,
            "hf_api": hf_token,
        },
        "queue": {
            "active":        q.active,
            "queued":        q.queued,
            "max_concurrent": _MAX_CONCURRENT,
            "max_queued":     _MAX_QUEUED,
        },
        "memory": {
            "ram_pct":        mem["ram_pct"],
            "available_gb":   mem["available_gb"],
            "swap_used_gb":   mem["swap_used_gb"],
            "pressure_level": mem["pressure_level"],
        },
        "profiles": list(PROFILES.keys()),
    }


@app.get("/api/hardware")
async def hardware_status():
    if _hw is None:
        raise HTTPException(status_code=503, detail="Hardware not profiled yet")
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "os": _hw.os_version,
        "cpu": _hw.cpu.brand,
        "cpu_cores": _hw.cpu.physical_cores,
        "ram_total_gb": round(_hw.memory.total_gb, 1),
        "ram_available_gb": round(vm.available / 1024**3, 2),
        "ram_percent": round(vm.percent, 1),
        "swap_used_gb": round(sw.used / 1024**3, 2),
        "gpu": _hw.gpu.name if _hw.gpu else None,
        "gpu_backend": _hw.gpu.backend if _hw.gpu else None,
        "is_unified_memory": _hw.gpu.is_unified_memory if _hw.gpu else False,
        "effective_budget_gb": round(_hw.effective_memory_gb, 2),
    }


@app.get("/api/profiles")
async def list_profiles():
    return {name: {
        "label": p.label,
        "description": p.description,
        "max_new_tokens": p.max_new_tokens,
        "temperature": p.temperature,
        "max_context_tokens": p.max_context_tokens,
        "kv_cache_precision": p.kv_cache_precision,
    } for name, p in PROFILES.items()}


# ── Conversations ────────────────────────────────────────────────────────

@app.post("/api/conversations")
async def create_conversation(req: ConversationCreateRequest):
    mgr = get_conv_manager()
    conv_id = mgr.create(
        model_id=req.model_id,
        profile=req.profile,
        system_prompt=req.system_prompt,
        title=req.title,
    )
    return {"conversation_id": conv_id, "model_id": req.model_id, "profile": req.profile}


@app.get("/api/conversations")
async def list_conversations(limit: int = 20):
    mgr = get_conv_manager()
    convs = mgr.list_all(limit=limit)
    return {"conversations": convs}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    mgr = get_conv_manager()
    conv = mgr.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id!r} not found")
    messages = mgr.get_messages(conv_id)
    return {**conv, "messages": messages}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    mgr = get_conv_manager()
    if not mgr.delete(conv_id):
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": conv_id}


@app.get("/api/conversations/{conv_id}/export")
async def export_conversation(conv_id: str):
    mgr = get_conv_manager()
    md = mgr.export_markdown(conv_id)
    return {"markdown": md, "conversation_id": conv_id}
