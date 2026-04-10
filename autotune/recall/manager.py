"""
autotune recall – high-level memory manager.

This is the single entry-point that chat sessions use.  It owns the full
lifecycle: extract → embed → store → retrieve → inject.

Usage inside a chat session
---------------------------
    from autotune.recall.manager import get_recall_manager

    mgr = get_recall_manager()

    # At conversation start — inject relevant past context:
    context_block = await mgr.get_context_for(messages)
    if context_block:
        system_prompt = (system_prompt or "") + "\\n\\n" + context_block

    # At conversation end — save to memory:
    saved = await mgr.save_conversation(conv_id, messages, model_id)

Design decisions
----------------
- The manager is a thin coordinator; storage is in RecallStore, embedding
  in MemoryEmbedder, and chunking in extractor.extract_chunks.
- Embedding is done before saving so results are immediately searchable,
  but a text-only fallback is used if Ollama isn't running.
- The context injection threshold (0.38) is intentionally conservative:
  it's better to show no context than irrelevant noise.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .embedder import MemoryEmbedder, get_embedder
from .extractor import estimate_conversation_value, extract_chunks
from .store import MemoryChunk, RecallStore, _db_path

logger = logging.getLogger(__name__)

# Cosine similarity threshold for injecting a memory.
# 0.38 on nomic-embed-text corresponds to "clearly on the same topic".
_INJECT_THRESHOLD = 0.38

# Hard cap on injected context characters to avoid bloating the prompt.
_MAX_INJECT_CHARS = 1_200

# Number of memories to retrieve for context injection.
_CONTEXT_TOP_K = 3


# ---------------------------------------------------------------------------
# Injection formatter
# ---------------------------------------------------------------------------

def _format_injection(memories: list[MemoryChunk]) -> str:
    """
    Format recalled memories as a clean block for system-prompt injection.

    Output looks like:

        ┌─ From your conversation history ──────────────────────────────
        │ [3 days ago · qwen3:8b]
        │ User: How do I do X in FastAPI?
        │ Assistant: You can use Y…
        │
        │ [1 week ago · qwen3:8b]
        │ User: …
        └───────────────────────────────────────────────────────────────
    """
    if not memories:
        return ""

    lines: list[str] = [
        "─── Relevant context from your past conversations ───",
    ]

    total_chars = 0
    for mem in memories:
        model_label = (mem.model_id or "unknown").split(":")[0]
        header = f"[{mem.age_str} · {model_label}]"
        body = mem.chunk_text

        # Truncate so we don't blow up the context window
        remaining = _MAX_INJECT_CHARS - total_chars
        if remaining <= 0:
            break
        if len(body) > remaining:
            body = body[:remaining].rsplit(" ", 1)[0] + "…"

        lines.append(f"\n{header}")
        lines.append(body)
        total_chars += len(body)

    lines.append("\n─── End of recalled context ───")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RecallManager
# ---------------------------------------------------------------------------

class RecallManager:
    """
    Coordinates memory storage and retrieval for chat sessions.

    Thread safety: the underlying SQLite store uses WAL mode so reads and
    writes can happen concurrently from different coroutines.  The embedder
    is async, so call all public methods with ``await``.
    """

    def __init__(
        self,
        store: Optional[RecallStore] = None,
        embedder: Optional[MemoryEmbedder] = None,
    ) -> None:
        self._store   = store   or RecallStore(_db_path())
        self._embedder = embedder or get_embedder()

    # ------------------------------------------------------------------ #
    # Save a conversation                                                  #
    # ------------------------------------------------------------------ #

    async def save_conversation(
        self,
        conv_id: str,
        messages: list[dict],
        model_id: str,
        created_at: Optional[float] = None,
    ) -> int:
        """
        Persist a completed conversation to memory.

        Skips conversations that:
        - Have already been saved (idempotent)
        - Have no meaningful exchanges (all < 20 chars)

        Returns the number of memory chunks saved (0 = nothing saved).
        """
        if not conv_id or not messages:
            return 0

        if self._store.is_saved(conv_id):
            logger.debug("recall: conv %s already saved, skipping", conv_id)
            return 0

        if not estimate_conversation_value(messages):
            logger.debug("recall: conv %s has no meaningful exchanges, skipping", conv_id)
            return 0

        raw_chunks = extract_chunks(messages)
        if not raw_chunks:
            return 0

        ts = created_at or time.time()

        # Embed each chunk.  If the embedder is unavailable, chunks are saved
        # text-only and will still be searchable via FTS5.
        enriched: list[dict] = []
        for chunk in raw_chunks:
            chunk["created_at"] = ts
            try:
                emb = await self._embedder.embed(chunk["text"])
                if emb is not None:
                    chunk["embedding"]       = emb
                    chunk["embedding_model"] = self._embedder.model_name
            except Exception as exc:
                logger.debug("recall: embed failed for chunk: %s", exc)
            enriched.append(chunk)

        saved = self._store.save(conv_id, model_id, enriched)
        if saved > 0:
            self._store.mark_saved(conv_id)
            logger.debug("recall: saved %d chunks for conv %s", saved, conv_id)

        return saved

    # ------------------------------------------------------------------ #
    # Retrieve context for a new conversation                             #
    # ------------------------------------------------------------------ #

    async def get_context_for(
        self,
        messages: list[dict],
        max_results: int = _CONTEXT_TOP_K,
    ) -> Optional[str]:
        """
        Return a formatted block of relevant memories, or None if nothing is
        relevant enough to inject.

        Searches by vector similarity first, falls back to FTS5 keyword
        search.  Only injects if the top result exceeds _INJECT_THRESHOLD.
        """
        if not messages:
            return None

        # Build a concise query from system prompt + first user message
        query_parts: list[str] = []
        for m in messages[:4]:
            role    = m.get("role", "")
            content = m.get("content", "")[:600]
            if role in ("user", "system") and content.strip():
                query_parts.append(content)
        query = " ".join(query_parts)[:1200].strip()

        if not query:
            return None

        memories = await self._search(query, top_k=max_results)
        if not memories:
            return None

        # Reject injection if best score is below threshold
        best = memories[0]
        if best.score > 0 and best.score < _INJECT_THRESHOLD:
            logger.debug(
                "recall: best score %.3f below threshold %.3f, not injecting",
                best.score, _INJECT_THRESHOLD,
            )
            return None

        return _format_injection(memories)

    # ------------------------------------------------------------------ #
    # Search (public — used by CLI)                                        #
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.20,
    ) -> list[MemoryChunk]:
        """
        Search memories.  Returns results ranked by relevance.

        Tries vector search first (requires an embedding model to be pulled).
        Falls back to FTS5 keyword search automatically.
        """
        return await self._search(query, top_k=top_k, min_score=min_score)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    async def _search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = _INJECT_THRESHOLD,
    ) -> list[MemoryChunk]:
        """Internal search used by both get_context_for and search."""
        # ── Vector search ──────────────────────────────────────────────
        try:
            emb = await self._embedder.embed(query)
            if emb is not None and self._embedder.model_name:
                results = self._store.search_vector(
                    emb,
                    embedding_model=self._embedder.model_name,
                    top_k=top_k,
                    min_score=min_score,
                )
                if results:
                    return results
        except Exception as exc:
            logger.debug("recall: vector search failed: %s", exc)

        # ── FTS5 keyword fallback ───────────────────────────────────────
        try:
            # Sanitize for FTS5: remove special chars that would cause parse errors
            safe_query = " ".join(
                w for w in query.split()
                if w.isalnum() or "-" in w
            )[:200]
            if safe_query:
                return self._store.search_fts(safe_query, top_k=top_k)
        except Exception as exc:
            logger.debug("recall: FTS search failed: %s", exc)

        return []

    # ------------------------------------------------------------------ #
    # Convenience pass-throughs for CLI commands                          #
    # ------------------------------------------------------------------ #

    def get_recent(
        self,
        limit: int = 20,
        model_id: Optional[str] = None,
        days: Optional[int] = None,
    ) -> list[MemoryChunk]:
        return self._store.get_recent(limit=limit, model_id=model_id, days=days)

    def delete(self, memory_id: int) -> bool:
        return self._store.delete(memory_id)

    def delete_conversation(self, conv_id: str) -> int:
        return self._store.delete_conversation(conv_id)

    def delete_all(self) -> int:
        return self._store.delete_all()

    def stats(self) -> dict:
        return self._store.stats()

    async def embedder_status(self) -> dict:
        """Return dict with embedding availability info for display."""
        available = await self._embedder.is_available()
        return {
            "available":   available,
            "model":       self._embedder.model_name,
            "preferred":   self._embedder.PREFERRED_MODELS,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[RecallManager] = None


def get_recall_manager() -> RecallManager:
    """Return the module-level RecallManager singleton (creates on first call)."""
    global _manager
    if _manager is None:
        _manager = RecallManager()
    return _manager
