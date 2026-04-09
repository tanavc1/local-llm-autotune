"""
VRAMTracker — measures actual unified memory used by Ollama via /api/ps.

Why /api/ps instead of psutil
-------------------------------
On Apple Silicon, model weights and KV cache buffers are allocated as Metal
``MTLBuffer`` objects in the unified memory pool.  These show up in the macOS
VM subsystem as "GPU app memory" — they are counted in
``psutil.virtual_memory().used``, but are mixed in with all other system
allocations (browser, OS, daemons).  The per-call delta we tried to measure
was swamped by Python response buffers.

Ollama's ``/api/ps`` endpoint reports the ``size_vram`` field for each loaded
model.  This is the exact number of bytes Ollama is holding in unified memory
for that model at this instant — weights + KV cache combined.

KV cache scales with num_ctx
------------------------------
For a transformer model loaded with context length C, Ollama allocates a KV
cache of size::

    kv_bytes = 2 × n_layers × n_kv_heads × head_dim × C × bytes_per_elem

    qwen3:8b (32 layers, 8 KV heads, 128 head_dim, F16 = 2 bytes):
        C=4096 → 536 MB KV cache
        C=1290 → 169 MB KV cache
        Savings → 367 MB

The actual measured savings are slightly larger (~400–800 MB depending on
quantization workspace and Metal buffer alignment) because Ollama allocates
additional working memory for attention computation that also scales with C.

Measured (qwen3:8b, cold load, system RAM delta):
    ctx=4096 load → +2.108 GB system RAM
    ctx=512  load → +1.302 GB system RAM
    Savings       →  0.806 GB

Public API
----------
::

    from autotune.metrics.vram import VRAMTracker, VRAMSnapshot

    tracker = VRAMTracker()
    snap = await tracker.snapshot("qwen3:8b:latest")
    print(f"VRAM in use: {snap.size_vram_gb:.2f} GB")
    print(f"Context length loaded: {snap.context_length}")

    # Theoretical KV-only portion
    kv_gb = VRAMTracker.estimate_kv_gb(
        num_ctx=snap.context_length,
        n_layers=32, n_kv_heads=8, head_dim=128, f16_kv=True,
    )
    print(f"KV cache portion: {kv_gb:.2f} GB")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_OLLAMA_BASE = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class VRAMSnapshot:
    """
    Point-in-time snapshot of Ollama's memory usage for one model.

    Fields come directly from Ollama's ``/api/ps`` response.
    """
    model_id: str
    size_vram_gb: float     # total unified memory held: weights + KV cache
    context_length: int     # num_ctx the model was loaded with
    is_loaded: bool         # False if model is not currently in memory

    @property
    def size_vram_mb(self) -> float:
        return self.size_vram_gb * 1024.0

    def __str__(self) -> str:
        return (
            f"VRAMSnapshot(model={self.model_id!r} "
            f"vram={self.size_vram_gb:.2f}GB "
            f"ctx={self.context_length} "
            f"loaded={self.is_loaded})"
        )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class VRAMTracker:
    """
    Queries Ollama's /api/ps to track actual VRAM usage over time.

    Comparison method
    -----------------
    To measure how much memory autotune saves vs raw Ollama defaults:

    1. Unload model (``unload_model``)
    2. Load with autotune options (small num_ctx) → snapshot A
    3. Unload model
    4. Load with raw options (Ollama default num_ctx) → snapshot B
    5. delta = B.size_vram_gb - A.size_vram_gb  ← autotune memory savings

    The delta represents the KV cache portion that autotune avoids allocating.
    """

    def __init__(self, base_url: str = _OLLAMA_BASE) -> None:
        self.base_url = base_url.rstrip("/")

    async def snapshot(self, model_id: str) -> VRAMSnapshot:
        """
        Query /api/ps and return current VRAM usage for the given model.

        Returns a snapshot with ``is_loaded=False`` if the model is not
        currently loaded.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.base_url}/api/ps")
                data = r.json()

            for m in data.get("models", []):
                if m.get("name") == model_id:
                    size_bytes = m.get("size_vram", m.get("size", 0))
                    ctx = m.get("context_length", 0)
                    return VRAMSnapshot(
                        model_id=model_id,
                        size_vram_gb=round(size_bytes / 1024**3, 3),
                        context_length=ctx,
                        is_loaded=True,
                    )

            return VRAMSnapshot(
                model_id=model_id,
                size_vram_gb=0.0,
                context_length=0,
                is_loaded=False,
            )

        except Exception as exc:
            logger.warning("VRAMTracker.snapshot failed: %s", exc)
            return VRAMSnapshot(
                model_id=model_id,
                size_vram_gb=0.0,
                context_length=0,
                is_loaded=False,
            )

    async def wait_for_load(
        self,
        model_id: str,
        timeout_sec: float = 60.0,
        poll_interval: float = 0.5,
    ) -> Optional[VRAMSnapshot]:
        """
        Poll /api/ps until the model appears loaded, then return a snapshot.

        Returns None if timeout_sec elapses without the model loading.
        """
        elapsed = 0.0
        while elapsed < timeout_sec:
            snap = await self.snapshot(model_id)
            if snap.is_loaded:
                return snap
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        return None

    @staticmethod
    def estimate_kv_gb(
        num_ctx: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        f16_kv: bool = True,
    ) -> float:
        """
        Theoretical KV cache size in GB for given model architecture and context.

        Formula: 2 (K+V) × n_layers × n_kv_heads × head_dim × num_ctx × bytes_per_elem

        Note: actual Ollama allocation is typically larger due to Metal buffer
        alignment and attention workspace memory.  Use this as a lower bound.
        """
        bytes_per_elem = 2 if f16_kv else 1   # F16 = 2 bytes, Q8 = 1 byte
        total_bytes = (
            2 * n_layers * n_kv_heads * head_dim * num_ctx * bytes_per_elem
        )
        return round(total_bytes / 1024**3, 3)

    @staticmethod
    def kv_savings_gb(
        ctx_raw: int,
        ctx_tuned: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        f16_kv: bool = True,
    ) -> float:
        """
        Theoretical KV memory saved by reducing context from ctx_raw to ctx_tuned.
        """
        raw  = VRAMTracker.estimate_kv_gb(ctx_raw,   n_layers, n_kv_heads, head_dim, f16_kv)
        tune = VRAMTracker.estimate_kv_gb(ctx_tuned, n_layers, n_kv_heads, head_dim, f16_kv)
        return round(raw - tune, 3)
