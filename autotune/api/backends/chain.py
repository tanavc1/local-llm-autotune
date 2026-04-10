"""
Backend chain: discovers available inference backends and routes requests
to the best one for a given model.

Priority (first available wins):
  1. MLX             – Apple Silicon only; fastest local option (unified memory,
                       Metal GPU, ~10-40% higher tok/s than Ollama on same model)
  2. Ollama          – best local experience, handles KV caching natively
  3. LM Studio       – alternative local runtime
  4. HuggingFace API – always available with HF_TOKEN, rate-limited without
  5. (error)         – helpful message if none work

Model discovery:
  - MLX: HF cache (~/.cache/huggingface/hub) for mlx-community models
  - Ollama /api/tags
  - LM Studio /v1/models
  - HuggingFace cache ~/.cache/huggingface/hub
  - Local GGUF files in common locations
  - HF Hub API (remote)
"""

from __future__ import annotations

import os
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx

from .base import Backend, ChatChunk
from .mlx_backend import (
    IS_APPLE_SILICON,
    mlx_available,
    resolve_mlx_model_id,
    get_mlx_backend,
    list_cached_mlx_models,
)
from .openai_compat import ModelNotAvailableError, OpenAICompatBackend

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

_GGUF_SEARCH_PATHS = [
    Path.home() / "Downloads",
    Path.home() / ".local" / "share" / "autotune" / "models",
    Path.home() / "models",
    Path("/opt/models"),
]


@dataclass
class ModelInfo:
    id: str                          # HF-style org/name
    name: str                        # short display name
    source: str                      # "ollama" | "lmstudio" | "hf_cache" | "gguf" | "hf_api"
    available_locally: bool
    path: Optional[str] = None       # local path if applicable
    size_gb: Optional[float] = None
    backend_hint: str = ""           # backend that should serve it


async def _probe_ollama() -> list[ModelInfo]:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            data = r.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size = m.get("size", 0) / 1024**3
            models.append(ModelInfo(
                id=name, name=name.split(":")[0], source="ollama",
                available_locally=True, size_gb=round(size, 2), backend_hint="ollama",
            ))
        return models
    except Exception:
        return []


async def _probe_lmstudio() -> list[ModelInfo]:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("http://localhost:1234/v1/models")
            data = r.json()
        return [
            ModelInfo(id=m["id"], name=m["id"], source="lmstudio",
                      available_locally=True, backend_hint="lmstudio")
            for m in data.get("data", [])
        ]
    except Exception:
        return []


def _scan_hf_cache() -> list[ModelInfo]:
    if not _HF_CACHE.exists():
        return []
    models = []
    for item in _HF_CACHE.iterdir():
        if not item.is_dir() or not item.name.startswith("models--"):
            continue
        parts = item.name[len("models--"):].split("--")
        if len(parts) >= 2:
            model_id = "/".join(parts)
            snapshots = item / "snapshots"
            if snapshots.exists() and any(snapshots.iterdir()):
                models.append(ModelInfo(
                    id=model_id, name=parts[-1], source="hf_cache",
                    available_locally=True, path=str(item), backend_hint="hf_local",
                ))
    return models


def _scan_gguf() -> list[ModelInfo]:
    models = []
    for base in _GGUF_SEARCH_PATHS:
        if not base.exists():
            continue
        for gguf in base.glob("**/*.gguf"):
            size = gguf.stat().st_size / 1024**3
            models.append(ModelInfo(
                id=gguf.stem, name=gguf.stem, source="gguf",
                available_locally=True, path=str(gguf), size_gb=round(size, 2),
                backend_hint="ollama",   # suggest loading via `ollama create`
            ))
    return models


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


def _make_ollama_backend(model_id: str) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        base_url="http://localhost:11434",
        api_key="ollama",
        backend_name="ollama",
    )


def _make_lmstudio_backend() -> OpenAICompatBackend:
    return OpenAICompatBackend(
        base_url="http://localhost:1234",
        api_key="lm-studio",
        backend_name="lmstudio",
    )


def _make_hf_backend(model_id: str) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        base_url=f"https://api-inference.huggingface.co/models/{model_id}",
        api_key=_hf_token(),
        backend_name="hf_api",
    )


# ---------------------------------------------------------------------------
# BackendChain
# ---------------------------------------------------------------------------

_PROBE_TTL_SEC = 30   # Re-probe backends after this many seconds


class BackendChain:
    """
    Selects the best available backend for a model and streams the response.

    Availability probes are cached for _PROBE_TTL_SEC seconds so that a
    backend that starts up after the server starts will be picked up on the
    next request after the TTL expires.
    """

    def __init__(self) -> None:
        self._ollama_models: Optional[list[ModelInfo]] = None
        self._lmstudio_models: Optional[list[ModelInfo]] = None
        self._ollama_ok: Optional[bool] = None
        self._lmstudio_ok: Optional[bool] = None
        self._ollama_probed_at: float = 0.0
        self._lmstudio_probed_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Discovery                                                            #
    # ------------------------------------------------------------------ #

    async def discover_all(self) -> list[ModelInfo]:
        """Return all models visible from all sources."""
        ollama = await _probe_ollama()
        lms = await _probe_lmstudio()
        hf_cache = _scan_hf_cache()
        gguf = _scan_gguf()

        # MLX models (Apple Silicon only)
        mlx_models: list[ModelInfo] = []
        if mlx_available():
            for m in list_cached_mlx_models():
                mlx_models.append(ModelInfo(
                    id=m["id"], name=m["name"], source="mlx",
                    available_locally=True, size_gb=m["size_gb"], backend_hint="mlx",
                ))

        seen: set[str] = set()
        all_models: list[ModelInfo] = []
        for m in mlx_models + ollama + lms + hf_cache + gguf:
            if m.id not in seen:
                seen.add(m.id)
                all_models.append(m)

        self._ollama_models = ollama
        self._lmstudio_models = lms
        # Don't assume Ollama availability from model list alone (could be empty
        # even when Ollama is running).  Let ollama_running() do a real probe.
        # Reset the TTL so the next resolve() call re-probes properly.
        self._ollama_ok = None
        self._ollama_probed_at = 0.0

        return sorted(all_models, key=lambda m: (m.source, m.id))

    async def ollama_running(self) -> bool:
        now = _time.monotonic()
        if self._ollama_ok is not None and (now - self._ollama_probed_at) < _PROBE_TTL_SEC:
            return self._ollama_ok
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                r = await client.get("http://localhost:11434/api/tags")
                self._ollama_ok = r.status_code == 200
                self._ollama_models = (await _probe_ollama()) if self._ollama_ok else []
                self._ollama_probed_at = now
                return self._ollama_ok
        except Exception:
            self._ollama_ok = False
            self._ollama_probed_at = now
            return False

    async def lmstudio_running(self) -> bool:
        now = _time.monotonic()
        if self._lmstudio_ok is not None and (now - self._lmstudio_probed_at) < _PROBE_TTL_SEC:
            return self._lmstudio_ok
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                r = await client.get("http://localhost:1234/v1/models")
                self._lmstudio_ok = r.status_code == 200
                self._lmstudio_probed_at = now
                return self._lmstudio_ok
        except Exception:
            self._lmstudio_ok = False
            self._lmstudio_probed_at = now
            return False

    def _ollama_has_model(self, model_id: str) -> bool:
        if not self._ollama_models:
            return False
        short = model_id.split("/")[-1].lower()
        for m in self._ollama_models:
            if m.id.lower() == model_id.lower():
                return True
            if short in m.id.lower():
                return True
        return False

    def _lmstudio_has_model(self, model_id: str) -> bool:
        if not self._lmstudio_models:
            return False
        for m in self._lmstudio_models:
            if model_id.lower() in m.id.lower():
                return True
        return False

    # ------------------------------------------------------------------ #
    # Resolution                                                           #
    # ------------------------------------------------------------------ #

    async def resolve(self, model_id: str) -> tuple[Backend, str]:
        """
        Return (backend, canonical_model_id) for the given model.

        On Apple Silicon, MLX is tried first — it outperforms Ollama by 10–40%
        on the same model using native Metal GPU kernels and unified memory.
        Falls back to Ollama/LM Studio/HF when no MLX equivalent exists.

        Raises ModelNotAvailableError if nothing can serve the model.
        """
        # 1. MLX — Apple Silicon only, highest throughput
        # Only route to MLX if the model is *locally cached* — we never want to
        # hit HuggingFace just because a mapping entry exists.
        if IS_APPLE_SILICON and mlx_available():
            mlx_id = resolve_mlx_model_id(model_id)
            if mlx_id is not None:
                cached_ids = {m["id"] for m in list_cached_mlx_models()}
                if mlx_id in cached_ids:
                    return get_mlx_backend(), mlx_id
                # Mapping exists but model not downloaded — fall through to Ollama

        # 2. Ollama
        if await self.ollama_running():
            if self._ollama_has_model(model_id):
                return _make_ollama_backend(model_id), model_id
            # Ollama is running but doesn't have the model — still try it
            # (user may have it under a different name)

        # 3. LM Studio
        if await self.lmstudio_running() and self._lmstudio_has_model(model_id):
            return _make_lmstudio_backend(), model_id

        # 4. HuggingFace Inference API
        token = _hf_token()
        if not token:
            mlx_hint = ""
            if IS_APPLE_SILICON:
                base = model_id.split(":")[0].split("/")[-1].lower()
                mlx_hint = (
                    f"\nFor Apple Silicon, pull a pre-quantized MLX model:\n"
                    f"  autotune mlx pull {base}\n"
                )
            instructions = (
                f"\n\nModel '{model_id}' not found locally.\n"
                "To use HuggingFace models:\n"
                "  export HF_TOKEN=your_token_here\n"
                "Get a free token at https://huggingface.co/settings/tokens\n\n"
                "To load a model locally:\n"
                f"  ollama pull {model_id.split('/')[-1].lower()}\n"
                f"{mlx_hint}"
                "Or run `autotune fetch-many` to see available models."
            )
            raise ModelNotAvailableError(instructions)

        return _make_hf_backend(model_id), model_id

    # ------------------------------------------------------------------ #
    # Streaming                                                            #
    # ------------------------------------------------------------------ #

    async def stream(
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        timeout: float = 120.0,
        num_ctx: Optional[int] = None,
        ollama_options: Optional[dict] = None,
    ) -> AsyncGenerator[ChatChunk, None]:
        backend, canonical_id = await self.resolve(model_id)

        if backend.name == "mlx":
            # MLX backend: pass generation params directly; it manages memory itself
            async for chunk in backend.stream(
                canonical_id,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                timeout=timeout,
            ):
                yield chunk
            return

        # Ollama / LM Studio / HF — OpenAI-compatible path
        extra_body = {}
        if backend.name == "ollama":
            # Build Ollama options — num_ctx is the single most impactful setting:
            # it determines KV-cache size allocated in VRAM/unified memory.
            # Under-setting wastes nothing; over-setting causes OOM or swapping.
            options: dict = dict(ollama_options or {})
            if num_ctx is not None:
                options["num_ctx"] = num_ctx
            # repeat_penalty must live inside the Ollama options dict — Ollama
            # does NOT honor the top-level OpenAI `repetition_penalty` parameter.
            # Without this, any repetition_penalty setting (including the fast
            # profile's value) is silently ignored, causing infinite repeat loops.
            if repetition_penalty != 1.0:
                options["repeat_penalty"] = repetition_penalty
            if options:
                extra_body["options"] = options
            extra_body["keep_alive"] = "-1m"   # keep model in VRAM/unified-memory

        async for chunk in backend.stream(
            canonical_id,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            timeout=timeout,
            extra_body=extra_body,
        ):
            yield chunk


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_chain: Optional[BackendChain] = None


def get_chain() -> BackendChain:
    global _chain
    if _chain is None:
        _chain = BackendChain()
    return _chain


async def resolve_backend(model_id: str) -> tuple[Backend, str]:
    return await get_chain().resolve(model_id)


async def unload_ollama_model(model_id: str) -> bool:
    """Tell Ollama to immediately evict `model_id` from memory.

    Sends a minimal POST /api/generate with keep_alive=0.  This is the
    official Ollama mechanism for on-demand unloading.  Returns True if
    Ollama acknowledged the request (status 200).
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": model_id, "keep_alive": 0},
            )
            return r.status_code == 200
    except Exception:
        return False
