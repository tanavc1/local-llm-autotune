"""
autotune recall – Ollama embedding backend for semantic memory.

Architecture
------------
MemoryEmbedder wraps Ollama's `/api/embeddings` endpoint and turns raw text
into compact float32 byte-strings that can be stored directly in the recall
SQLite store (see store.py) and compared with cosine similarity.

Model discovery
---------------
On first use the embedder hits `/api/tags` and selects the best available
embedding model from PREFERRED_MODELS (earlier entries win).  The probe result
is cached for the lifetime of the object; call reset() to force a re-probe,
e.g. after pulling a new model with `ollama pull`.

Threading / concurrency
-----------------------
All network I/O is async (httpx.AsyncClient).  embed_batch processes texts
sequentially because Ollama does not expose a native batching API; parallelism
would just queue inside the Ollama server anyway.

Error handling
--------------
Every network call is wrapped in a broad try/except.  Any failure (connection
refused, timeout, unexpected JSON shape, …) returns None rather than raising,
so callers can treat embeddings as optional and fall back to FTS search.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
import numpy as np


# ---------------------------------------------------------------------------
# MemoryEmbedder
# ---------------------------------------------------------------------------

class MemoryEmbedder:
    """Async embedding backend that delegates to a locally running Ollama server."""

    PREFERRED_MODELS: list[str] = [
        "nomic-embed-text",
        "mxbai-embed-large",
        "all-minilm",
        "snowflake-arctic-embed",
    ]

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url: str = base_url.rstrip("/")
        self._model: Optional[str] = None
        self._probed: bool = False
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> Optional[str]:
        """Return the discovered embedding model name, or None if not yet probed."""
        return self._model

    @property
    def dim(self) -> Optional[int]:
        """Return the embedding dimension, set after the first successful embed."""
        return self._dim

    # ------------------------------------------------------------------ #
    # Model discovery                                                      #
    # ------------------------------------------------------------------ #

    async def _discover_model(self) -> Optional[str]:
        """
        Query Ollama for available models and return the best embedding model.

        Checks each entry in PREFERRED_MODELS against the list of model names
        returned by /api/tags; a preferred name matches if it appears as a
        substring of any available model name (so "nomic-embed-text" matches
        "nomic-embed-text:latest", etc.).

        Sets _probed=True on completion. Returns None when Ollama is
        unreachable or no suitable model is found.  Does not re-probe if
        _probed is already True.
        """
        if self._probed:
            return self._model

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()

            available: list[str] = [m["name"] for m in data.get("models", [])]

            for preferred in self.PREFERRED_MODELS:
                for name in available:
                    if preferred in name:
                        self._model = name
                        self._probed = True
                        return self._model

        except Exception:
            pass

        self._probed = True
        return None

    # ------------------------------------------------------------------ #
    # Core embedding                                                       #
    # ------------------------------------------------------------------ #

    async def embed(self, text: str) -> Optional[bytes]:
        """
        Embed *text* using the discovered Ollama model.

        Steps:
          1. Discover model (cached after first call).
          2. Truncate text to 8000 characters.
          3. POST to /api/embeddings with a 30-second timeout.
          4. Convert the returned list[float] to a numpy float32 array and
             return its raw bytes so it can be stored directly as a BLOB.

        Returns None if no model is available or any error occurs.
        """
        model = await self._discover_model()
        if model is None:
            return None

        text = text[:8000]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                )
                resp.raise_for_status()
                data = resp.json()

            floats: list[float] = data["embedding"]
            vec = np.array(floats, dtype=np.float32)

            if self._dim is None:
                self._dim = len(vec)

            return vec.tobytes()

        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Batch embedding                                                      #
    # ------------------------------------------------------------------ #

    async def embed_batch(self, texts: list[str]) -> list[Optional[bytes]]:
        """
        Embed each text in *texts* sequentially and return a list of results.

        Ollama has no native batch endpoint, so texts are embedded one at a
        time.  Each element of the returned list mirrors the corresponding
        input: bytes on success, None on failure.
        """
        results: list[Optional[bytes]] = []
        for text in texts:
            result = await self.embed(text)
            results.append(result)
        return results

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    async def is_available(self) -> bool:
        """Return True if an embedding model is available on the Ollama server."""
        model = await self._discover_model()
        return model is not None

    def reset(self) -> None:
        """
        Clear the cached probe result so the next call re-discovers models.

        Useful after pulling a new model with `ollama pull` without restarting
        the autotune process.
        """
        self._probed = False
        self._model = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_embedder: Optional[MemoryEmbedder] = None


def get_embedder() -> MemoryEmbedder:
    """
    Return the process-wide MemoryEmbedder singleton, creating it if needed.

    The singleton is initialised with the default Ollama URL
    (http://localhost:11434).  To use a custom URL, construct a MemoryEmbedder
    directly rather than going through this factory.
    """
    global _embedder
    if _embedder is None:
        _embedder = MemoryEmbedder()
    return _embedder
