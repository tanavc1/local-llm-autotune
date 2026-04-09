"""
Persistent conversation manager backed by SQLite.

Features
--------
- Full history stored per conversation
- Context window trimming (keeps system prompt + recent messages)
- System prompt change detection (for KV cache invalidation hints)
- Token estimation (char/4 heuristic — accurate enough for trimming)
- Export to Markdown
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from autotune.api.ctx_utils import estimate_tokens as _estimate_tokens
from autotune.db.store import _db_path

CONVERSATIONS_DB = _db_path().parent / "conversations.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS conversations (
    id                  TEXT PRIMARY KEY,
    model_id            TEXT NOT NULL,
    profile             TEXT NOT NULL DEFAULT 'balanced',
    system_prompt       TEXT,
    system_prompt_hash  TEXT,
    title               TEXT,
    created_at          REAL NOT NULL,
    last_active         REAL NOT NULL,
    total_tokens        INTEGER DEFAULT 0,
    message_count       INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id         TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,      -- user | assistant | system
    content         TEXT NOT NULL,
    tokens          INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    ttft_ms         REAL,
    tokens_per_sec  REAL,
    backend         TEXT
);

CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conv_id, created_at);
"""



def _hash_system_prompt(prompt: Optional[str]) -> Optional[str]:
    if not prompt:
        return None
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class ConversationManager:
    """Thread-safe conversation CRUD + context trimming."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        import sqlite3
        self._path = db_path or CONVERSATIONS_DB
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        # RLock (reentrant) so nested calls within the same thread don't deadlock
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Conversation lifecycle                                               #
    # ------------------------------------------------------------------ #

    def create(
        self,
        model_id: str,
        profile: str = "balanced",
        system_prompt: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        """Create a new conversation and return its ID."""
        conv_id = str(uuid.uuid4())[:8]
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO conversations
                   (id, model_id, profile, system_prompt, system_prompt_hash, title, created_at, last_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (conv_id, model_id, profile, system_prompt,
                 _hash_system_prompt(system_prompt), title, now, now),
            )
            self._conn.commit()
        return conv_id

    def get(self, conv_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
        return dict(row) if row else None

    def touch(self, conv_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET last_active = ? WHERE id = ?",
                (time.time(), conv_id),
            )
            self._conn.commit()

    def list_all(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT c.*, COUNT(m.id) as msg_count
                   FROM conversations c
                   LEFT JOIN messages m ON c.id = m.conv_id
                   GROUP BY c.id
                   ORDER BY c.last_active DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, conv_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conv_id,)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def update_title(self, conv_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id)
            )
            self._conn.commit()

    def update_system_prompt(self, conv_id: str, system_prompt: str) -> bool:
        """Returns True if the system prompt actually changed (KV cache invalidation hint)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT system_prompt_hash FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            if not row:
                return False
            new_hash = _hash_system_prompt(system_prompt)
            changed = row["system_prompt_hash"] != new_hash
            self._conn.execute(
                "UPDATE conversations SET system_prompt = ?, system_prompt_hash = ? WHERE id = ?",
                (system_prompt, new_hash, conv_id),
            )
            self._conn.commit()
        return changed

    # ------------------------------------------------------------------ #
    # Messages                                                             #
    # ------------------------------------------------------------------ #

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        ttft_ms: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        backend: Optional[str] = None,
    ) -> int:
        tokens = _estimate_tokens(content)
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO messages
                   (conv_id, role, content, tokens, created_at, ttft_ms, tokens_per_sec, backend)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (conv_id, role, content, tokens, now, ttft_ms, tokens_per_sec, backend),
            )
            self._conn.execute(
                """UPDATE conversations
                   SET total_tokens = total_tokens + ?,
                       message_count = message_count + 1,
                       last_active = ?
                   WHERE id = ?""",
                (tokens, now, conv_id),
            )
            self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def delete_last_message(self, conv_id: str, role: str = "user") -> bool:
        """
        Delete the most recent message with the given role from a conversation.

        Used to roll back an orphaned user turn when inference fails before
        producing any output — keeps the conversation history clean so that
        a failed turn is never silently replayed on the next request.

        Returns True if a message was deleted, False if none matched.
        """
        with self._lock:
            row = self._conn.execute(
                """SELECT id, tokens FROM messages
                   WHERE conv_id = ? AND role = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (conv_id, role),
            ).fetchone()
            if not row:
                return False
            self._conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
            self._conn.execute(
                """UPDATE conversations
                   SET total_tokens  = MAX(0, total_tokens  - ?),
                       message_count = MAX(0, message_count - 1)
                   WHERE id = ?""",
                (row["tokens"], conv_id),
            )
            self._conn.commit()
        return True

    def get_messages(self, conv_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE conv_id = ? ORDER BY created_at ASC",
                (conv_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Context building (with trimming)                                     #
    # ------------------------------------------------------------------ #

    def build_context(
        self,
        conv_id: str,
        max_tokens: int,
        new_user_message: Optional[str] = None,
        reserved_for_output: int = 512,
    ) -> tuple[list[dict[str, str]], bool]:
        """
        Build the messages list to send to the backend using the intelligent
        ContextWindow manager.

        Returns
        -------
        (messages, context_complete)
            context_complete is True when no history was dropped or summarised
            (useful as a KV-cache invalidation hint for callers).

        Compression tiers (based on token budget utilisation):
          FULL              < 55 %  — all history verbatim
          RECENT+FACTS     55–75 %  — recent 8 turns + facts block for older
          COMPRESSED       75–90 %  — recent 6 turns compressed + summary
          EMERGENCY         > 90 %  — 4 turns + ultra-compact summary

        Low-value chatter is dropped before summarisation in every non-FULL tier.
        Tool outputs are always compressed.
        """
        from autotune.context.window import ContextWindow

        conv = self.get(conv_id)
        if not conv:
            raise KeyError(f"Conversation {conv_id!r} not found")

        system_prompt = conv.get("system_prompt")

        # Pull history from DB (system messages excluded — handled by ContextWindow)
        raw_history = self.get_messages(conv_id)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in raw_history
            if m["role"] != "system"
        ]

        cw    = ContextWindow(max_ctx_tokens=max_tokens)
        built = cw.build(
            history=history,
            system_prompt=system_prompt,
            new_user_message=new_user_message,
            reserved_for_output=reserved_for_output,
        )

        if built.tier.value != "full":
            logger.debug(
                "conv=%s context tier=%s kept=%d dropped=%d summarised=%d (%.0f%% budget)",
                conv_id, built.tier.value,
                built.turns_kept, built.turns_dropped, built.turns_summarized,
                built.budget_pct * 100,
            )

        context_complete = (built.turns_dropped == 0 and not built.summary_injected)
        return built.messages, context_complete

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def export_markdown(self, conv_id: str) -> str:
        conv = self.get(conv_id)
        if not conv:
            return "Conversation not found."

        lines = [
            f"# Conversation {conv_id}",
            f"**Model:** {conv['model_id']}  |  **Profile:** {conv['profile']}",
            f"**Created:** {time.strftime('%Y-%m-%d %H:%M', time.localtime(conv['created_at']))}",
            "",
        ]
        if conv.get("system_prompt"):
            lines += [f"> **System:** {conv['system_prompt']}", ""]

        for msg in self.get_messages(conv_id):
            role = msg["role"].capitalize()
            lines.append(f"### {role}")
            lines.append(msg["content"])
            if msg.get("tokens_per_sec"):
                lines.append(f"*{msg['tokens_per_sec']:.1f} tok/s*")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_mgr: Optional[ConversationManager] = None


def get_conv_manager() -> ConversationManager:
    global _mgr
    if _mgr is None:
        _mgr = ConversationManager()
    return _mgr
