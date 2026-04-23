"""
autotune recall – SQLite storage backend for persistent conversation memory.

Architecture
------------
Each conversation is broken into *chunks* (user+assistant exchange pairs) and
stored in the `memories` table.  Two search strategies are supported:

  1. Vector search  – embeddings are stored as raw float32 BLOBs.  At query
     time all vectors for a given embedding model are loaded into numpy and
     cosine similarity is computed in-process.  This keeps the implementation
     simple and dependency-free (no pgvector, no sqlite-vec extension needed).

  2. FTS5 keyword search – SQLite's built-in full-text search with the porter
     stemmer provides fast keyword fallback when no embedding is available or
     when the vector score is too low.

The database lives at ~/.autotune/recall.db on all platforms.  WAL mode is
enabled so reads never block writes, which matters when the chat loop is
streaming tokens while a background thread indexes the prior exchange.

Tables
------
  memories              – one row per chunk; embedding stored as BLOB
  memories_fts          – FTS5 virtual table (porter stemmer), kept in sync
                          via INSERT/DELETE/UPDATE triggers
  saved_conversations   – tracks which conv_ids have been fully memorized
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# DB location
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    """Return the path to recall.db, creating the parent directory if needed."""
    base = Path.home() / ".autotune"
    base.mkdir(parents=True, exist_ok=True)
    return base / "recall.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id         TEXT    NOT NULL,
    created_at      REAL    NOT NULL,
    model_id        TEXT,
    chunk_text      TEXT    NOT NULL,
    embedding       BLOB,
    embedding_model TEXT,
    turn_start      INTEGER NOT NULL DEFAULT 0,
    turn_end        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_mem_conv       ON memories(conv_id);
CREATE INDEX IF NOT EXISTS idx_mem_model      ON memories(model_id);
CREATE INDEX IF NOT EXISTS idx_mem_created    ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_mem_emb_model  ON memories(embedding_model);

-- FTS5 virtual table with porter stemmer for keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    chunk_text,
    content='memories',
    content_rowid='id',
    tokenize='porter ascii'
);

-- Keep FTS in sync with the main table
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, chunk_text)
        VALUES ('delete', old.id, old.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF chunk_text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, chunk_text)
        VALUES ('delete', old.id, old.chunk_text);
    INSERT INTO memories_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;

-- Track which conversations have been fully memorized
CREATE TABLE IF NOT EXISTS saved_conversations (
    conv_id     TEXT PRIMARY KEY,
    saved_at    REAL NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MemoryChunk:
    """A single recalled chunk of conversation."""

    id: int
    conv_id: str
    created_at: float
    model_id: Optional[str]
    chunk_text: str
    turn_start: int
    turn_end: int
    score: float = 0.0

    @property
    def age_str(self) -> str:
        """Return a human-readable age string, e.g. '3 days ago', '5 min ago'."""
        delta = time.time() - self.created_at
        if delta < 60:
            return "just now"
        if delta < 3600:
            mins = int(delta / 60)
            return f"{mins} min ago"
        if delta < 86400:
            hours = int(delta / 3600)
            return f"{hours} hr ago" if hours > 1 else "1 hr ago"
        if delta < 7 * 86400:
            days = int(delta / 86400)
            return f"{days} days ago" if days > 1 else "1 day ago"
        if delta < 30 * 86400:
            weeks = int(delta / (7 * 86400))
            return f"{weeks} weeks ago" if weeks > 1 else "1 week ago"
        months = int(delta / (30 * 86400))
        return f"{months} months ago" if months > 1 else "1 month ago"


# ---------------------------------------------------------------------------
# RecallStore
# ---------------------------------------------------------------------------

class RecallStore:
    """SQLite-backed store for conversation memory chunks."""

    def __init__(self, db_path: Path) -> None:
        self.path = db_path
        self._init_db()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist yet."""
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields an open connection and commits on success."""
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL is set per-connection; harmless if already set
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row, score: float = 0.0) -> MemoryChunk:
        return MemoryChunk(
            id=row["id"],
            conv_id=row["conv_id"],
            created_at=row["created_at"],
            model_id=row["model_id"],
            chunk_text=row["chunk_text"],
            turn_start=row["turn_start"],
            turn_end=row["turn_end"],
            score=score,
        )

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def save(
        self,
        conv_id: str,
        model_id: Optional[str],
        chunks: list[dict],
    ) -> int:
        """
        Persist a list of conversation chunks.

        Each dict in *chunks* must contain:
            text        (str)  – the chunk text (user + assistant exchange)
            created_at  (float) – unix timestamp of the exchange
            turn_start  (int)  – first turn index covered by this chunk
            turn_end    (int)  – last turn index covered by this chunk

        Optional keys:
            embedding        (bytes) – raw float32 vector as bytes
            embedding_model  (str)   – identifier for the embedding model used

        Returns the number of rows inserted.
        """
        inserted = 0
        with self._conn() as conn:
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO memories
                        (conv_id, created_at, model_id, chunk_text,
                         embedding, embedding_model, turn_start, turn_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conv_id,
                        chunk.get("created_at", time.time()),
                        model_id,
                        chunk["text"],
                        chunk.get("embedding"),
                        chunk.get("embedding_model"),
                        chunk.get("turn_start", 0),
                        chunk.get("turn_end", 0),
                    ),
                )
                inserted += 1
        return inserted

    def mark_saved(self, conv_id: str) -> None:
        """Record that *conv_id* has been fully memorized."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO saved_conversations (conv_id, saved_at)
                VALUES (?, ?)
                """,
                (conv_id, time.time()),
            )

    def is_saved(self, conv_id: str) -> bool:
        """Return True if *conv_id* has been marked as saved."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM saved_conversations WHERE conv_id = ?",
                (conv_id,),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search_vector(
        self,
        query_embedding: bytes,
        embedding_model: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[MemoryChunk]:
        """
        Semantic search using cosine similarity.

        Loads all embeddings for *embedding_model* from the DB, computes cosine
        similarity against *query_embedding* in numpy, and returns up to *top_k*
        results with score >= *min_score*, sorted by score descending.
        """
        query_vec = np.frombuffer(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0.0:
            return []

        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, conv_id, created_at, model_id, chunk_text,
                       embedding, turn_start, turn_end
                FROM memories
                WHERE embedding IS NOT NULL
                  AND embedding_model = ?
                """,
                (embedding_model,),
            ).fetchall()

        if not rows:
            return []

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            raw = row["embedding"]
            if not raw:
                continue
            vec = np.frombuffer(raw, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0.0:
                continue
            similarity = float(np.dot(query_vec, vec) / (query_norm * norm))
            if similarity >= min_score:
                scored.append((similarity, row))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [self._row_to_chunk(row, score=s) for s, row in scored[:top_k]]

    def search_fts(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        """
        Keyword search via FTS5.

        Uses FTS5's built-in BM25 ranking (lower rank = better match in SQLite's
        implementation, so we ORDER BY rank ASC).
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT m.id, m.conv_id, m.created_at, m.model_id,
                       m.chunk_text, m.turn_start, m.turn_end,
                       memories_fts.rank AS rank
                FROM memories_fts
                JOIN memories AS m ON memories_fts.rowid = m.id
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, top_k),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def get_recent(
        self,
        limit: int = 20,
        model_id: Optional[str] = None,
        days: Optional[int] = None,
    ) -> list[MemoryChunk]:
        """Return recently stored chunks, newest first."""
        q = "SELECT id, conv_id, created_at, model_id, chunk_text, turn_start, turn_end FROM memories WHERE 1=1"
        args: list = []

        if model_id is not None:
            q += " AND model_id = ?"
            args.append(model_id)

        if days is not None:
            cutoff = time.time() - days * 86400
            q += " AND created_at >= ?"
            args.append(cutoff)

        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)

        with self._conn() as conn:
            rows = conn.execute(q, args).fetchall()

        return [self._row_to_chunk(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Delete operations                                                    #
    # ------------------------------------------------------------------ #

    def delete(self, memory_id: int) -> bool:
        """Delete a single memory chunk by ID. Returns True if a row was removed."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cur.rowcount > 0

    def delete_conversation(self, conv_id: str) -> int:
        """Delete all chunks for *conv_id*. Returns the number of rows removed."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memories WHERE conv_id = ?", (conv_id,))
            conn.execute(
                "DELETE FROM saved_conversations WHERE conv_id = ?", (conv_id,)
            )
        return cur.rowcount

    def delete_all(self) -> int:
        """Delete every chunk and saved-conversation record. Returns total rows removed."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM saved_conversations")
        return cur.rowcount

    # ------------------------------------------------------------------ #
    # List conversations                                                   #
    # ------------------------------------------------------------------ #

    def list_conversations(self, limit: int = 20) -> list[dict]:
        """
        Return one row per distinct conv_id, ordered by most-recently-active.

        Each dict contains:
            conv_id      (str)
            model_id     (str)   — model used for most chunks
            first_at     (float) — unix timestamp of earliest chunk
            last_at      (float) — unix timestamp of latest chunk
            chunk_count  (int)   — number of stored exchange chunks
            sample_text  (str)   — text of the very first chunk (user's opening question)
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    g.conv_id,
                    COALESCE(MAX(g.model_id), 'unknown') AS model_id,
                    MIN(g.created_at)   AS first_at,
                    MAX(g.created_at)   AS last_at,
                    COUNT(*)            AS chunk_count,
                    (
                        SELECT m.chunk_text FROM memories m
                        WHERE  m.conv_id = g.conv_id
                        ORDER  BY m.id ASC LIMIT 1
                    )                   AS sample_text
                FROM memories g
                GROUP BY g.conv_id
                ORDER BY MAX(g.created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """Return summary statistics about the recall store."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            with_emb = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            ).fetchone()[0]

            oldest_row = conn.execute(
                "SELECT MIN(created_at) FROM memories"
            ).fetchone()[0]
            newest_row = conn.execute(
                "SELECT MAX(created_at) FROM memories"
            ).fetchone()[0]

            by_model_rows = conn.execute(
                """
                SELECT COALESCE(model_id, '<unknown>') AS model, COUNT(*) AS cnt
                FROM memories
                GROUP BY model_id
                ORDER BY cnt DESC
                """
            ).fetchall()

            emb_model_rows = conn.execute(
                """
                SELECT DISTINCT embedding_model
                FROM memories
                WHERE embedding_model IS NOT NULL
                """
            ).fetchall()

        size_mb = (
            round(self.path.stat().st_size / 1024 ** 2, 3) if self.path.exists() else 0.0
        )

        return {
            "total_chunks": total,
            "with_embeddings": with_emb,
            "size_mb": size_mb,
            "oldest_at": oldest_row,
            "newest_at": newest_row,
            "by_model": {row["model"]: row["cnt"] for row in by_model_rows},
            "embedding_models": [row["embedding_model"] for row in emb_model_rows],
        }
