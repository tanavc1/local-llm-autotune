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

-- gateway_log captures every request/response pair that passes through the
-- /v1/chat/completions endpoint, whether or not a conversation_id was provided.
-- This is the authoritative source for the dashboard Conversations view.
CREATE TABLE IF NOT EXISTS gateway_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id           TEXT,               -- set when conversation_id header used
    model_id          TEXT NOT NULL,
    api_key_id        TEXT,               -- NULL for unauthenticated requests
    user_content      TEXT NOT NULL,
    assistant_content TEXT,               -- NULL on error / cancelled stream
    system_prompt     TEXT,
    ttft_ms           REAL,
    tokens_per_sec    REAL,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    backend           TEXT,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_glog_created ON gateway_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_glog_key     ON gateway_log(api_key_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_glog_model   ON gateway_log(model_id, created_at DESC);
"""



_TIER_LABELS: dict[str, str] = {
    "full":              "FULL",
    "recent_plus_facts": "RECENT+FACTS",
    "compressed":        "COMPRESSED",
    "emergency":         "EMERGENCY",
}


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
    # Gateway log — records every request/response through the proxy      #
    # ------------------------------------------------------------------ #

    def log_gateway_request(
        self,
        model_id: str,
        user_content: str,
        assistant_content: Optional[str] = None,
        system_prompt: Optional[str] = None,
        conv_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        ttft_ms: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        backend: Optional[str] = None,
    ) -> int:
        """Append one request/response pair to gateway_log. Thread-safe."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO gateway_log
                   (conv_id, model_id, api_key_id, user_content, assistant_content,
                    system_prompt, ttft_ms, tokens_per_sec, prompt_tokens,
                    completion_tokens, backend, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (conv_id, model_id, api_key_id, user_content, assistant_content,
                 system_prompt, ttft_ms, tokens_per_sec, prompt_tokens,
                 completion_tokens, backend, now),
            )
            self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    _SESSION_GAP_SEC = 300  # 5-minute inactivity = new session

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """
        Return session summaries combining two sources:

        1. gateway_log rows grouped by 5-minute inactivity + same api_key_id
           (captures every request through the proxy, new style)
        2. Named conversations from the conversations table that have NO
           gateway_log entries (older / CLI-created conversations)

        Sessions are returned newest-first.  Each dict has:
          session_id  – int (first gateway_log row id) or "conv:<conv_id>"
          source      – "gateway" | "conversation"
          start_ts    – unix timestamp of first turn
          end_ts      – unix timestamp of last turn
          turn_count  – number of user/assistant pairs
          models      – deduplicated model list
          api_key_id  – key used (None = unauthenticated)
          preview     – first 120 chars of first user message / title
          conv_id     – only present for source="conversation"
        """
        sessions: list[dict] = []

        # ── 1. Sessions from gateway_log ────────────────────────────────────
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, model_id, api_key_id, user_content, created_at
                   FROM gateway_log ORDER BY created_at DESC LIMIT 2000""",
            ).fetchall()

        if rows:
            entries = [dict(r) for r in rows]
            entries.reverse()  # oldest first for grouping

            current: list[dict] = [entries[0]]
            for entry in entries[1:]:
                prev = current[-1]
                same_key = entry["api_key_id"] == prev["api_key_id"]
                gap_ok = (entry["created_at"] - prev["created_at"]) < self._SESSION_GAP_SEC
                if same_key and gap_ok:
                    current.append(entry)
                else:
                    sessions.append(self._summarise_glog_session(current))
                    current = [entry]
            sessions.append(self._summarise_glog_session(current))

        # ── 2. Named conversations with no gateway_log coverage ─────────────
        with self._lock:
            conv_rows = self._conn.execute(
                """SELECT c.id, c.model_id, c.title, c.created_at, c.last_active,
                          c.message_count,
                          (SELECT content FROM messages
                           WHERE conv_id = c.id AND role = 'user'
                           ORDER BY created_at ASC LIMIT 1) AS first_msg,
                          (SELECT COUNT(*) FROM gateway_log WHERE conv_id = c.id) AS glog_count
                   FROM conversations c
                   ORDER BY c.last_active DESC LIMIT 200""",
            ).fetchall()

        for row in conv_rows:
            r = dict(row)
            if r["glog_count"] > 0:
                continue  # already represented in gateway sessions
            # user+assistant pairs → turn_count = floor(message_count / 2)
            turn_count = max(1, (r["message_count"] or 2) // 2)
            preview = r["first_msg"] or r["title"] or f"Conversation {r['id']}"
            sessions.append({
                "session_id": f"conv:{r['id']}",
                "source":     "conversation",
                "start_ts":   r["created_at"],
                "end_ts":     r["last_active"],
                "turn_count": turn_count,
                "models":     [r["model_id"]],
                "api_key_id": None,
                "preview":    preview[:120],
                "conv_id":    r["id"],
            })

        sessions.sort(key=lambda s: s["end_ts"], reverse=True)
        return sessions[:limit]

    @staticmethod
    def _summarise_glog_session(entries: list[dict]) -> dict:
        models: list[str] = []
        for e in entries:
            if e["model_id"] not in models:
                models.append(e["model_id"])
        first_user = entries[0]["user_content"] or ""
        return {
            "session_id": entries[0]["id"],
            "source":     "gateway",
            "start_ts":   entries[0]["created_at"],
            "end_ts":     entries[-1]["created_at"],
            "turn_count": len(entries),
            "models":     models,
            "api_key_id": entries[0]["api_key_id"],
            "preview":    first_user[:120],
        }

    def get_session_turns(self, session_id: int) -> list[dict]:
        """
        Return all gateway_log rows in the session whose first entry has
        the given *session_id*.
        """
        with self._lock:
            anchor = self._conn.execute(
                "SELECT api_key_id, created_at FROM gateway_log WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not anchor:
                return []

            api_key_id = anchor["api_key_id"]
            start_ts   = anchor["created_at"]

            if api_key_id is None:
                rows = self._conn.execute(
                    """SELECT * FROM gateway_log
                       WHERE created_at >= ? AND api_key_id IS NULL
                       ORDER BY created_at ASC LIMIT 500""",
                    (start_ts,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM gateway_log
                       WHERE created_at >= ? AND api_key_id = ?
                       ORDER BY created_at ASC LIMIT 500""",
                    (start_ts, api_key_id),
                ).fetchall()

        entries: list[dict] = []
        for row in rows:
            r = dict(row)
            if not entries:
                entries.append(r)
                continue
            if (r["created_at"] - entries[-1]["created_at"]) < self._SESSION_GAP_SEC:
                entries.append(r)
            else:
                break

        return entries

    def get_conversation_turns(self, conv_id: str) -> list[dict]:
        """Return messages for a named conversation as turn dicts matching gateway_log shape."""
        messages = self.get_messages(conv_id)
        turns: list[dict] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "user":
                user_content = msg["content"]
                asst_content = None
                ttft_ms      = None
                tps          = None
                backend      = None
                comp_tokens  = None
                created_at   = msg["created_at"]
                # Look ahead for the assistant reply
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    a = messages[i + 1]
                    asst_content = a["content"]
                    ttft_ms      = a.get("ttft_ms")
                    tps          = a.get("tokens_per_sec")
                    backend      = a.get("backend")
                    comp_tokens  = a.get("tokens")
                    i += 2
                else:
                    i += 1
                turns.append({
                    "model_id":          None,
                    "user_content":      user_content,
                    "assistant_content": asst_content,
                    "ttft_ms":           ttft_ms,
                    "tokens_per_sec":    tps,
                    "backend":           backend,
                    "completion_tokens": comp_tokens,
                    "created_at":        created_at,
                })
            else:
                i += 1
        return turns

    # ------------------------------------------------------------------ #
    # Context building (with trimming)                                     #
    # ------------------------------------------------------------------ #

    def build_context(
        self,
        conv_id: str,
        max_tokens: int,
        new_user_message: Optional[str] = None,
        reserved_for_output: int = 512,
    ) -> tuple[list[dict[str, str]], bool, str]:
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
        tier_label = _TIER_LABELS.get(built.tier.value, built.tier.value.upper())
        return built.messages, context_complete, tier_label

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
