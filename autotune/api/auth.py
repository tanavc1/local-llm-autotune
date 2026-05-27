"""
API key authentication for the autotune server.

Keys follow the format:  sk-at-<random_urlsafe_48>  (~384 bits of entropy).
Only the SHA-256 hash is persisted; the plaintext is shown once on creation.

Authentication flow
-------------------
1.  Client sends:  Authorization: Bearer sk-at-<token>
2.  Middleware hashes the token and does a fast dict lookup (in-memory cache).
3.  On cache miss, falls back to SQLite (populated once, cached thereafter).
4.  Invalid / revoked keys receive a machine-readable 401 / 403 body.

Admin flow
----------
Admin endpoints at /admin/* require:
    Authorization: Bearer <AUTOTUNE_ADMIN_KEY>
where AUTOTUNE_ADMIN_KEY is set by the server operator in the environment.
If the env var is not set, all /admin/* endpoints return 503.

Enforcement
-----------
Set AUTOTUNE_REQUIRE_API_KEY=1 to enforce API keys on /v1/* requests.
Without this flag, the server remains open (backward-compatible default).
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
import uuid
from datetime import date
from typing import Any, Optional

from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_KEY_PREFIX = "sk-at-"
_KEY_DISPLAY_LEN = 14  # "sk-at-" (6) + 8 random chars before the "..."


# ---------------------------------------------------------------------------
# Key generation + hashing
# ---------------------------------------------------------------------------

def generate_api_key() -> tuple[str, str]:
    """Return (full_plaintext_key, sha256_hex_hash).

    The plaintext is never stored — return it to the caller once and discard.
    """
    raw = secrets.token_urlsafe(36)   # 288 bits of URL-safe entropy
    full_key = f"{_KEY_PREFIX}{raw}"
    key_hash = _hash_key(full_key)
    return full_key, key_hash


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def key_display_prefix(full_key: str) -> str:
    return full_key[:_KEY_DISPLAY_LEN] + "..."


# ---------------------------------------------------------------------------
# In-memory key cache
# ---------------------------------------------------------------------------
# Each entry: key_hash -> key_record_dict (from SQLite).
# Populated on first verification, invalidated on revocation.
# Per-process cache is sufficient: a single server process manages all keys
# and revocation happens in-process via the admin API.

_key_cache: dict[str, dict[str, Any]] = {}


def _cache_put(key_hash: str, record: dict[str, Any]) -> None:
    _key_cache[key_hash] = record


def _cache_invalidate(key_hash: str) -> None:
    _key_cache.pop(key_hash, None)


def invalidate_key_cache_by_id(key_id: str) -> None:
    """Remove all cache entries for the given key ID (called on revocation)."""
    evict = [h for h, r in _key_cache.items() if r.get("id") == key_id]
    for h in evict:
        _key_cache.pop(h, None)


# ---------------------------------------------------------------------------
# Core verification (sync — SQLite access is synchronous in our store)
# ---------------------------------------------------------------------------

def verify_api_key_sync(raw_key: str) -> Optional[dict[str, Any]]:
    """
    Return the key record dict if the key exists in the database, or None if
    the key is completely unknown.

    The caller must check ``record["is_active"]`` to distinguish
    *valid* (is_active=True) from *revoked* (is_active=False, returns 403)
    versus *not found* (returns None → 401).

    Checks the in-memory cache first, falls back to SQLite.  Results are
    cached so subsequent requests for the same key are O(1).
    """
    if not raw_key.startswith(_KEY_PREFIX):
        return None

    key_hash = _hash_key(raw_key)

    # Fast path
    if key_hash in _key_cache:
        r = _key_cache[key_hash]
        # Sentinel: id=None means the key was looked up before and not found
        if r.get("id") is None:
            return None
        # Check expiry even on cached records (cache doesn't expire automatically)
        expires_at = r.get("expires_at")
        if expires_at and time.time() > expires_at:
            _cache_invalidate(key_hash)
            return {**r, "is_active": False, "_reason": "expired"}
        return r

    # Slow path: SQLite
    try:
        from autotune.db.store import get_db
        record = get_db().get_api_key_by_hash(key_hash)
    except Exception as exc:
        logger.error("api_key lookup failed: %s", exc)
        return None

    if record is None:
        # Cache a "not found" sentinel so we don't hammer SQLite for bad keys
        _cache_put(key_hash, {"is_active": False, "id": None})
        return None

    _cache_put(key_hash, record)

    # Check expiry before returning
    expires_at = record.get("expires_at")
    if expires_at and time.time() > expires_at:
        return {**record, "is_active": False, "_reason": "expired"}

    return record


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def _admin_key() -> str:
    return os.environ.get("AUTOTUNE_ADMIN_KEY", "").strip()


async def require_admin(authorization: Optional[str] = Header(None)) -> None:
    """Dependency: validate the AUTOTUNE_ADMIN_KEY bearer token."""
    admin = _admin_key()
    if not admin:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "admin_not_configured",
                "message": (
                    "Admin API is disabled. "
                    "Set the AUTOTUNE_ADMIN_KEY environment variable "
                    "to a strong secret to enable it."
                ),
            },
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            detail={
                "error": "missing_authorization",
                "message": "Authorization: Bearer <AUTOTUNE_ADMIN_KEY> required",
            },
        )
    token = authorization[len("Bearer "):]
    if not secrets.compare_digest(token.encode("utf-8"), admin.encode("utf-8")):
        raise HTTPException(
            status_code=403,
            detail={"error": "forbidden", "message": "Invalid admin key"},
        )


# ---------------------------------------------------------------------------
# Middleware helper — called from the ASGI middleware in server.py
# ---------------------------------------------------------------------------

def api_key_enforcement_enabled() -> bool:
    return os.environ.get("AUTOTUNE_REQUIRE_API_KEY", "").strip() in ("1", "true", "yes")


def build_auth_error_response(status: int, error: str, message: str) -> JSONResponse:
    headers: dict = {}
    if status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=status,
        content={"error": {"type": error, "message": message}},
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Usage logging — fire-and-forget, never raises
# ---------------------------------------------------------------------------

async def log_api_key_usage(
    *,
    key_id: str,
    key_name: str,
    model_id: str,
    backend: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    ttft_ms: float,
    status: str = "success",
    error_type: Optional[str] = None,
) -> None:
    """
    Persist one API-key usage record to SQLite and (optionally) Supabase.

    Designed for fire-and-forget from finally blocks — all exceptions are
    swallowed internally.
    """
    today = date.today().isoformat()
    row: dict[str, Any] = {
        "key_id":             key_id,
        "day":                today,
        "model_id":           model_id,
        "backend":            backend,
        "prompt_tokens":      max(0, prompt_tokens),
        "completion_tokens":  max(0, completion_tokens),
        "latency_ms":         round(latency_ms, 1),
        "ttft_ms":            round(ttft_ms, 1),
        "status":             status,
        "created_at":         time.time(),
    }
    if error_type:
        row["error_type"] = error_type

    # ── Local SQLite (always attempted) ──────────────────────────────────
    try:
        from autotune.db.store import get_db
        db = get_db()
        db.log_api_key_usage(row)
        db.touch_api_key(key_id)
    except Exception as exc:
        logger.debug("api_key_usage sqlite write failed: %s", exc)

    # ── Supabase mirror (fire-and-forget, best-effort) ─────────────────
    try:
        from autotune.telemetry.client import get_client
        tc = get_client()
        if tc:
            tc.record_api_key_usage({
                **row,
                "key_name": key_name,
            })
    except Exception:
        pass
