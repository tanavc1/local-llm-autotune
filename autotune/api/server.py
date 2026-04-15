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
from typing import Any, AsyncGenerator, Optional, Union

logger = logging.getLogger(__name__)

import psutil
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNFE

try:
    _VERSION = _pkg_version("llm-autotune")
except _PkgNFE:
    _VERSION = "0.1.0"

from autotune.hardware.profiler import profile_hardware
from .backends.chain import BackendChain, ModelNotAvailableError, get_chain
from .backends.openai_compat import AuthError, BackendError
from .conversation import ConversationManager, get_conv_manager
from .ctx_utils import estimate_tokens
from .kv_manager import build_ollama_options
from .hardware_tuner import get_tuner
from .profiles import PROFILES, get_profile
from .thinking import (
    THINKING_OVERHEAD as _THINKING_OVERHEAD,
    ThinkingStreamFilter,
    filter_thinking_sse as _filter_thinking_stream,
    is_thinking_model as _is_thinking_model,
    strip_thinking as _strip_thinking,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

_VALID_ROLES = frozenset({"system", "user", "assistant", "tool", "function"})


class Message(BaseModel):
    role: str
    content: Union[str, list, None] = None
    # tool/function call fields — accepted and ignored for local routing
    name: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {v!r}")
        return v

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v: Any) -> str:
        """Accept str, list of content-parts, or None.  Always return str."""
        if v is None:
            return ""
        if isinstance(v, list):
            # OpenAI multi-modal content: [{"type": "text", "text": "..."}, ...]
            parts = []
            for part in v:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                # image_url and other types are silently dropped — local models
                # don't support vision inputs via this proxy path
            return " ".join(parts)
        return str(v)


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stream: bool = False   # OpenAI spec default is false; clients set true explicitly
    # ── Standard OpenAI fields accepted and passed through / ignored ─────────
    # These are sent by virtually every OpenAI-compatible client.  Not
    # accepting them causes 422 errors that are completely opaque to users.
    stop: Optional[Union[str, list[str]]] = None
    n: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    # ── autotune extensions ──────────────────────────────────────────────────
    profile: str = "balanced"
    conversation_id: Optional[str] = None
    system: Optional[str] = None

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


class CompletionRequest(BaseModel):
    """OpenAI /v1/completions (legacy + FIM autocomplete used by Continue.dev)."""
    model: str
    prompt: Union[str, list[str]]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    suffix: Optional[str] = None      # FIM suffix
    n: Optional[int] = None
    echo: Optional[bool] = None
    user: Optional[str] = None
    seed: Optional[int] = None


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
    version=_VERSION,
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

# Sources that represent directly-servable chat models.
# HF cache and GGUF entries need manual setup before they can be routed to,
# so we hide them from /v1/models to avoid polluting client model pickers.
_SERVABLE_SOURCES = frozenset({"ollama", "mlx", "lmstudio"})

# Sub-strings in model IDs that identify non-chat (embedding / encoder) models.
# These should never appear in a chat completion endpoint's model list.
_EMBEDDING_PATTERNS = (
    "embed", "cross-encoder", "sentence-transformer",
    "reranker", "clip", "whisper", "tts", "ocr", "layout",
    "minilm",   # all-minilm-l6-v2 and similar sentence-transformer models
)


def _is_chat_model(model_id: str, source: str) -> bool:
    """Return True if this model entry is a usable chat/completion model."""
    if source not in _SERVABLE_SOURCES:
        return False
    lower = model_id.lower()
    return not any(pat in lower for pat in _EMBEDDING_PATTERNS)


@app.get("/v1/models")
async def list_models():
    chain = get_chain()
    all_models = await chain.discover_all()

    data = []
    for m in all_models:
        if not _is_chat_model(m.id, m.source):
            continue
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


@app.get("/v1/models/{model_id:path}/status")
async def model_status(model_id: str):
    """
    Return the readiness state of a specific model.

    Status values
    -------------
    "ready"      — model is loaded in memory right now; first token will be fast
    "available"  — model is on disk but not loaded; expect a cold-start delay
    "not_found"  — model is not available locally

    Response also includes a memory fit assessment so your application can warn
    users before a request fails.

    Example
    -------
        GET /v1/models/qwen3:8b/status
        {
            "model": "qwen3:8b",
            "status": "ready",
            "backend": "ollama",
            "fit": {
                "class": "safe",
                "ram_util_pct": 62.3,
                "available_gb": 8.3,
                "warning": null
            }
        }
    """
    from autotune.api.running_models import get_running_models
    from autotune.api.model_selector import ModelSelector
    import psutil as _psutil

    # ── 1. Is the model currently loaded in memory? ───────────────────────
    running = get_running_models()
    loaded = next(
        (m for m in running if model_id.lower() in m.name.lower() or m.name.lower() in model_id.lower()),
        None,
    )
    if loaded:
        status = "ready"
        backend = loaded.backend
    else:
        status = "not_found"
        backend = None

    # ── 2. Is it available on disk (Ollama tags)? ─────────────────────────
    size_gb: Optional[float] = None
    params_b: Optional[float] = None
    quant: str = "Q4_K_M"

    if status == "not_found":
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get("http://localhost:11434/api/tags")
                tags = r.json().get("models", [])
            for m in tags:
                name = m.get("name", "")
                if model_id.lower() in name.lower() or name.lower() in model_id.lower():
                    status = "available"
                    backend = "ollama"
                    size_gb = m.get("size", 0) / 1024**3
                    details = m.get("details", {})
                    quant = details.get("quantization_level", "Q4_K_M")
                    # Derive rough param count from name (e.g. "8b" → 8.0)
                    import re
                    m2 = re.search(r"(\d+(?:\.\d+)?)\s*b", name.lower())
                    if m2:
                        params_b = float(m2.group(1))
                    break
        except Exception:
            pass

    # ── 3. Check MLX cache ────────────────────────────────────────────────
    if status == "not_found":
        try:
            from autotune.api.backends.mlx_backend import list_cached_mlx_models, resolve_mlx_model_id
            mlx_id = resolve_mlx_model_id(model_id)
            if mlx_id and any(m["id"] == mlx_id for m in list_cached_mlx_models()):
                status = "available"
                backend = "mlx"
        except Exception:
            pass

    # ── 4. Memory fit assessment ──────────────────────────────────────────
    fit: dict = {}
    if _hw is not None and size_gb is not None:
        try:
            sel = ModelSelector(_hw.effective_memory_gb, _hw.memory.total_gb)
            report = sel.assess(model_id, size_gb, params_b, quant)
            fit = {
                "class": report.fit_class.value,
                "ram_util_pct": report.ram_util_pct,
                "available_gb": round(_hw.effective_memory_gb, 1),
                "total_ram_gb": round(_hw.memory.total_gb, 1),
                "warning": report.warning,
            }
            if report.suggested_quant:
                fit["suggested_quant"] = report.suggested_quant
                fit["suggested_quant_gb"] = report.suggested_quant_gb
        except Exception as _fit_exc:
            logger.debug("fit assessment failed for %s: %s", model_id, _fit_exc)
    elif _hw is not None and status != "not_found":
        # Model is loaded but we couldn't read size — report RAM only
        vm = _psutil.virtual_memory()
        fit = {
            "class": "unknown",
            "available_gb": round(vm.available / 1024**3, 1),
            "total_ram_gb": round(_hw.memory.total_gb, 1),
            "warning": None,
        }

    response: dict = {
        "model": model_id,
        "status": status,
        "backend": backend,
    }
    if fit:
        response["fit"] = fit
    if loaded:
        response["loaded_since"] = loaded.loaded_since
        response["ram_gb"] = round(loaded.ram_gb, 2)

    return response


# ---------------------------------------------------------------------------
# /v1/completions  — legacy completion API + FIM (used by Continue.dev autocomplete)
# ---------------------------------------------------------------------------

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    """
    OpenAI-compatible text completion endpoint.

    Continue.dev uses this for tab-autocomplete (FIM).  We convert the
    prompt (and optional suffix) into a chat message so it can be routed
    through the same Ollama/MLX backend chain that chat uses.

    FIM (fill-in-the-middle) is handled by embedding the suffix hint in the
    system prompt — this is the most portable approach across local models.
    """
    queue = _get_queue()
    try:
        await queue.acquire(timeout=_WAIT_TIMEOUT)
    except _QueueFullError as exc:
        raise HTTPException(status_code=429, detail={
            "error": "queue_full", "queue_depth": exc.depth,
        })
    except _asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail={"error": "queue_timeout"})

    prompt_text = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
    requested_max = req.max_tokens or 256
    max_tokens = (requested_max + _THINKING_OVERHEAD) if _is_thinking_model(req.model) else requested_max
    temperature = req.temperature if req.temperature is not None else 0.2
    top_p = req.top_p if req.top_p is not None else 0.95

    # Build messages: if a suffix was provided (FIM), hint the model via system prompt
    if req.suffix:
        system_content = (
            "Complete the code. Output ONLY the code that fills in the middle — "
            "no explanation, no markdown fences.\n"
            f"The code continues with:\n{req.suffix}"
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt_text},
        ]
    else:
        messages = [{"role": "user", "content": prompt_text}]

    chain = get_chain()
    profile = get_profile("fast")   # autocomplete always uses the fast profile
    ollama_opts, _ = build_ollama_options(messages, profile)
    if _is_thinking_model(req.model):
        ollama_opts["num_ctx"] = min(
            ollama_opts["num_ctx"] + _THINKING_OVERHEAD,
            profile.max_context_tokens,
        )
    tuner = get_tuner()

    chunk_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    thinking_model = _is_thinking_model(req.model)

    if req.stream:
        # True streaming — hold the queue slot until the last byte is sent,
        # mirroring how /v1/chat/completions handles its stream lifecycle.
        async def _completions_stream() -> AsyncGenerator[bytes, None]:
            in_think = False
            tuner._apply("fast")
            try:
                async for chunk in chain.stream(
                    req.model, messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=30.0,
                    num_ctx=ollama_opts["num_ctx"],
                    ollama_options=ollama_opts,
                ):
                    if not chunk.content:
                        continue

                    # Strip thinking tags inline for reasoning models
                    text_out = chunk.content
                    if thinking_model:
                        buf = chunk.content
                        parts: list[str] = []
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
                        text_out = "".join(parts)

                    if text_out:
                        payload = {
                            "id": chunk_id,
                            "object": "text_completion",
                            "created": created_ts,
                            "model": req.model,
                            "choices": [{"text": text_out, "index": 0, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(payload)}\n\n".encode()

                    if chunk.finish_reason:
                        done = {
                            "id": chunk_id,
                            "object": "text_completion",
                            "created": created_ts,
                            "model": req.model,
                            "choices": [{"text": "", "index": 0, "finish_reason": chunk.finish_reason}],
                        }
                        yield f"data: {json.dumps(done)}\n\n".encode()
                        break

            except (ModelNotAvailableError, AuthError, BackendError) as exc:
                err = json.dumps({"error": {"message": str(exc), "type": "backend_error"}})
                yield f"data: {err}\n\n".encode()
            finally:
                tuner._restore()
                await queue.release()

            yield b"data: [DONE]\n\n"

        return StreamingResponse(_completions_stream(), media_type="text/event-stream")

    # ── Non-streaming (most common for FIM autocomplete) ──────────────────
    collected: list[str] = []
    try:
        tuner._apply("fast")
        try:
            async for chunk in chain.stream(
                req.model, messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=30.0,
                num_ctx=ollama_opts["num_ctx"],
                ollama_options=ollama_opts,
            ):
                if chunk.content:
                    collected.append(chunk.content)
        except (ModelNotAvailableError, AuthError, BackendError) as e:
            raise HTTPException(
                status_code=503,
                detail=_make_error_body(_error_type(e), str(e), req.model),
            )
        finally:
            tuner._restore()
    finally:
        await queue.release()

    text = _strip_thinking("".join(collected))
    prompt_tokens = estimate_tokens(prompt_text)
    comp_tokens = estimate_tokens(text)
    return {
        "id": chunk_id,
        "object": "text_completion",
        "created": created_ts,
        "model": req.model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens": prompt_tokens + comp_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Structured error helpers
# ---------------------------------------------------------------------------

def _error_type(exc: Exception) -> str:
    if isinstance(exc, ModelNotAvailableError):
        return "model_not_found"
    if isinstance(exc, AuthError):
        return "auth_error"
    return "backend_error"


def _make_error_body(error_type: str, message: str, model_id: str) -> dict:
    """
    Build a structured error body that applications can handle programmatically.

    Fields
    ------
    type        : machine-readable error category
    message     : human-readable explanation
    model       : the model that was requested
    suggestion  : plain-English recovery step
    docs_url    : link to the relevant docs section

    Compared to a bare 503 string, this lets application code do:

        if error["type"] == "model_not_found":
            # pull the model or show the user a friendly install prompt
        elif error["type"] == "memory_pressure":
            # degrade gracefully — use a smaller context or model
    """
    suggestion = ""
    if error_type == "model_not_found":
        base = model_id.split(":")[0].split("/")[-1].lower()
        suggestion = (
            f"Pull the model first: ollama pull {model_id}  "
            f"or check available models at GET /v1/models"
        )
    elif error_type == "backend_error":
        low = message.lower()
        if any(k in low for k in ("out of memory", "oom", "memory", "killed")):
            error_type = "memory_pressure"
            suggestion = (
                f"The model exceeded available RAM. "
                f"Try a smaller model or check GET /v1/models/{model_id}/status "
                f"for a fit assessment before sending requests."
            )
        else:
            suggestion = (
                "Check that Ollama is running: ollama serve  "
                "or inspect server logs for details."
            )
    elif error_type == "auth_error":
        suggestion = "Set HF_TOKEN if using HuggingFace models, or use a local model."

    body: dict = {
        "type": error_type,
        "message": message,
        "model": model_id,
    }
    if suggestion:
        body["suggestion"] = suggestion
    # Add a fit check hint so developers know where to look
    body["status_url"] = f"/v1/models/{model_id}/status"
    return body


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
    requested_max_tokens = req.max_tokens or profile.max_new_tokens
    # Reasoning models (qwen3, deepseek-r1, etc.) spend tokens on a <think>
    # block before writing their answer.  We add a thinking overhead so the
    # model always has room to finish its thought AND produce the requested
    # number of answer tokens.  The overhead is invisible to the caller —
    # thinking tags are stripped before the response is returned.
    if _is_thinking_model(req.model):
        max_tokens = requested_max_tokens + _THINKING_OVERHEAD
    else:
        max_tokens = requested_max_tokens
    temperature = req.temperature if req.temperature is not None else profile.temperature
    top_p = req.top_p if req.top_p is not None else profile.top_p
    rep_penalty = req.repetition_penalty if req.repetition_penalty is not None else profile.repetition_penalty
    timeout = profile.request_timeout_sec

    # ── Dynamic KV-cache sizing ──────────────────────────────────────────
    # Compute only the context window this request actually needs instead of
    # always allocating profile.max_context_tokens.  Reduces both unified
    # memory pressure and KV-cache init latency (direct TTFT improvement).
    ollama_opts, _ = build_ollama_options(messages, profile)
    # For reasoning models: num_ctx must also include the thinking overhead or
    # Ollama will truncate the model mid-thought (num_ctx is the hard limit,
    # not max_new_tokens).  Cap at profile.max_context_tokens so the safety
    # ceiling is still respected.
    if _is_thinking_model(req.model):
        ollama_opts["num_ctx"] = min(
            ollama_opts["num_ctx"] + _THINKING_OVERHEAD,
            profile.max_context_tokens,
        )

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    first_token_time: Optional[float] = None
    backend_name = "unknown"
    # ThinkingStreamFilter tracks visible content for both client display AND
    # conversation storage — so the DB never sees raw <think> blocks.
    _think_filt = ThinkingStreamFilter()

    async def _raw_stream() -> AsyncGenerator[bytes, None]:
        """Inner generator: produces raw SSE bytes from the backend."""
        nonlocal first_token_time, backend_name

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
                    _think_filt.feed(chunk.content)   # track visible text
                    yield _make_chunk_json(chunk.content, req.model, chunk_id).encode()

                if chunk.finish_reason:
                    yield _make_chunk_json("", req.model, chunk_id, chunk.finish_reason).encode()
                    break

        except ModelNotAvailableError as e:
            err = json.dumps({"error": _make_error_body("model_not_found", str(e), req.model)})
            yield f"data: {err}\n\n".encode()
        except AuthError as e:
            err = json.dumps({"error": _make_error_body("auth_error", str(e), req.model)})
            yield f"data: {err}\n\n".encode()
        except BackendError as e:
            err = json.dumps({"error": _make_error_body("backend_error", str(e), req.model)})
            yield f"data: {err}\n\n".encode()
        finally:
            tuner._restore()
            # Use the filter's collected_text() — think blocks already excluded.
            elapsed = time.time() - start_time
            content = _think_filt.collected_text()
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
        return StreamingResponse(
            _filter_thinking_stream(_raw_stream()),
            media_type="text/event-stream",
        )

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
        raise HTTPException(
            status_code=503,
            detail=_make_error_body(_error_type(e), str(e), req.model),
        )
    finally:
        tuner._restore()

    content_out = _strip_thinking("".join(collected))
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
        "version": _VERSION,
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


@app.get("/api/running_models")
async def running_models_endpoint():
    """Return all LLMs currently resident in memory across all local backends."""
    from autotune.api.running_models import get_running_models
    models = get_running_models()
    return {
        "models": [
            {
                "name": m.name,
                "backend": m.backend,
                "ram_gb": round(m.ram_gb, 3),
                "context_len": m.context_len,
                "loaded_since": m.loaded_since,
                "expires_at": m.expires_at,
                "quant": m.quant,
                "family": m.family,
                "age": m.age_str,
                "expires_in": m.expires_str,
            }
            for m in models
        ],
        "count": len(models),
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
