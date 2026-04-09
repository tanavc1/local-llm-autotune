"""
Local model discovery — everything downloaded/available on this machine.

Sources queried:
  1. Ollama  (/api/tags + /api/show for each)
  2. MLX cache (~/.cache/huggingface/hub, mlx-community models)
  3. LM Studio (/v1/models)

Each entry includes size, family, parameter count, quantization, and a
quality tier from the public benchmark lookup table.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from autotune.models.quality import QualityInfo, get_quality


_OLLAMA_BASE = "http://localhost:11434"
_LMSTUDIO_BASE = "http://localhost:1234"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LocalModel:
    id: str                          # e.g. "qwen3:8b"
    name: str                        # base name without tag
    source: str                      # "ollama" | "mlx" | "lmstudio"
    size_gb: Optional[float]
    family: Optional[str]            # "llama" | "qwen2" | "phi" | …
    parameter_size: Optional[str]    # "3.2B" | "14B" | …
    quantization: Optional[str]      # "Q4_K_M" | "F16" | …
    context_length: Optional[int]    # max context tokens
    modified: Optional[str]          # ISO date string
    quality: Optional[QualityInfo]
    mlx_available: bool = False      # True if an MLX version is also cached


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _ollama_request(path: str, body: Optional[dict] = None, timeout: float = 3.0) -> Optional[dict]:
    """Sync HTTP helper — runs in thread pool when called from async code."""
    try:
        if body is not None:
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                f"{_OLLAMA_BASE}{path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        else:
            req = urllib.request.Request(f"{_OLLAMA_BASE}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _fetch_ollama_models() -> list[LocalModel]:
    """Return all Ollama models with details from /api/show."""
    tags_resp = _ollama_request("/api/tags")
    if not tags_resp:
        return []

    models: list[LocalModel] = []
    for entry in tags_resp.get("models", []):
        model_id = entry.get("name", "")
        size_bytes = entry.get("size", 0)
        size_gb = round(size_bytes / 1024**3, 2) if size_bytes else None
        modified = entry.get("modified_at", "")[:10]  # "2025-03-12T…" → "2025-03-12"

        # Pull architecture details via /api/show
        show = _ollama_request("/api/show", {"name": model_id}, timeout=5.0)
        details = (show or {}).get("details", {})
        model_info = (show or {}).get("model_info", {})

        family = details.get("family") or details.get("families", [None])[0]
        param_size = details.get("parameter_size")
        quantization = details.get("quantization_level")

        # Context length: Ollama reports it in model_info for GGUF models
        context_length: Optional[int] = None
        for key, val in model_info.items():
            if "context_length" in key.lower() or "ctx_length" in key.lower():
                try:
                    context_length = int(val)
                except (TypeError, ValueError):
                    pass
                break

        base_name = model_id.split(":")[0]
        quality = get_quality(model_id)

        models.append(LocalModel(
            id=model_id,
            name=base_name,
            source="ollama",
            size_gb=size_gb,
            family=family,
            parameter_size=param_size,
            quantization=quantization,
            context_length=context_length,
            modified=modified,
            quality=quality,
        ))
    return models


# ---------------------------------------------------------------------------
# MLX (Apple Silicon)
# ---------------------------------------------------------------------------

def _fetch_mlx_models() -> list[LocalModel]:
    """Return locally cached MLX models."""
    try:
        from autotune.api.backends.mlx_backend import list_cached_mlx_models, mlx_available
        if not mlx_available():
            return []
        cached = list_cached_mlx_models()
        models: list[LocalModel] = []
        for m in cached:
            model_id = m.get("id", "")
            base = model_id.split("/")[-1]
            quality = get_quality(model_id)
            models.append(LocalModel(
                id=model_id,
                name=base,
                source="mlx",
                size_gb=m.get("size_gb"),
                family=None,
                parameter_size=None,
                quantization=None,
                context_length=None,
                modified=None,
                quality=quality,
            ))
        return models
    except Exception:
        return []


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------

def _fetch_lmstudio_models() -> list[LocalModel]:
    try:
        req = urllib.request.Request(f"{_LMSTUDIO_BASE}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read())
        return [
            LocalModel(
                id=m["id"], name=m["id"], source="lmstudio",
                size_gb=None, family=None, parameter_size=None,
                quantization=None, context_length=None, modified=None,
                quality=get_quality(m["id"]),
            )
            for m in data.get("data", [])
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_local_models() -> list[LocalModel]:
    """
    Synchronously fetch all locally available models from all sources.

    Safe to call from sync or async context (does not use asyncio).
    """
    ollama = _fetch_ollama_models()
    mlx    = _fetch_mlx_models()
    lms    = _fetch_lmstudio_models()

    # Mark Ollama models that also have a cached MLX version
    mlx_ids_lower = {m.id.lower() for m in mlx}
    for om in ollama:
        base = om.name.lower()
        if any(base in mlx_id for mlx_id in mlx_ids_lower):
            om.mlx_available = True

    return ollama + mlx + lms


def is_ollama_running() -> bool:
    return _ollama_request("/api/tags") is not None
