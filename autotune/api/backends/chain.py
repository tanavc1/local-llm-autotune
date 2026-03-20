"""
Backend chain: discovers available inference backends and routes requests
to the best one for a given model.

Priority (first available wins):
  1. Ollama          – best local experience, handles KV caching natively
  2. LM Studio       – alternative local runtime
  3. HuggingFace API – always available with HF_TOKEN, rate-limited without
  4. (error)         – helpful message if none work

Model discovery:
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

        seen: set[str] = set()
        all_models: list[ModelInfo] = []
        for m in ollama + lms + hf_cache + gguf:
            if m.id not in seen:
                seen.add(m.id)
                all_models.append(m)

        self._ollama_models = ollama
        self._lmstudio_models = lms
        # Ollama is considered running if discover_all was called (probes succeeded).
        # We set True unconditionally here because _probe_ollama returns [] on
        # connection failure — but discover_all is not called as an availability check.
        self._ollama_ok = True
        self._ollama_probed_at = _time.monotonic()

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

    async def resolve(self, model_id: str) -> tuple[OpenAICompatBackend, str]:
        """
        Return (backend, canonical_model_id) for the given model.

        Raises ModelNotAvailableError if nothing can serve the model.
        """
        # 1. Ollama
        if await self.ollama_running():
            if self._ollama_has_model(model_id):
                return _make_ollama_backend(model_id), model_id
            # Ollama is running but doesn't have the model — still try it
            # (user may have it under a different name)

        # 2. LM Studio
        if await self.lmstudio_running() and self._lmstudio_has_model(model_id):
            return _make_lmstudio_backend(), model_id

        # 3. HuggingFace Inference API
        token = _hf_token()
        if not token:
            instructions = (
                f"\n\nModel '{model_id}' not found locally.\n"
                "To use HuggingFace models:\n"
                "  export HF_TOKEN=your_token_here\n"
                "Get a free token at https://huggingface.co/settings/tokens\n\n"
                "To load a model locally:\n"
                f"  ollama pull {model_id.split('/')[-1].lower()}\n"
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

        extra_body = {}
        if backend.name == "ollama":
            # Build Ollama options — num_ctx is the single most impactful setting:
            # it determines KV-cache size allocated in VRAM/unified memory.
            # Under-setting wastes nothing; over-setting causes OOM or swapping.
            options: dict = dict(ollama_options or {})
            if num_ctx is not None:
                options["num_ctx"] = num_ctx
            if options:
                extra_body["options"] = options
            extra_body["keep_alive"] = "-1"   # keep model in VRAM/unified-memory

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


async def resolve_backend(model_id: str) -> tuple[OpenAICompatBackend, str]:
    return await get_chain().resolve(model_id)
