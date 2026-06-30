"""
Admin API router — key management and usage analytics.

All endpoints require:
    Authorization: Bearer <AUTOTUNE_ADMIN_KEY>

Endpoints
---------
POST   /admin/keys              Create a new API key
GET    /admin/keys              List all keys (metadata only, no plaintext)
GET    /admin/keys/{key_id}     Get a single key + its usage summary
DELETE /admin/keys/{key_id}     Revoke a key

GET    /admin/usage             Per-key, per-day consumption
GET    /admin/usage/summary     Aggregate totals across all keys
"""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from .auth import (
    generate_api_key,
    invalidate_key_cache_by_id,
    key_display_prefix,
    require_admin,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateKeyRequest(BaseModel):
    name: str
    metadata: Optional[dict] = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty")
        if len(v) > 128:
            raise ValueError("name must be ≤ 128 characters")
        return v


class RevokeKeyRequest(BaseModel):
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _key_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    return {
        "id":           row["id"],
        "name":         row["name"],
        "key_prefix":   row["key_prefix"],
        "is_active":    bool(row.get("is_active", True)),
        "created_at":   _ts_to_iso(row.get("created_at")),
        "last_used_at": _ts_to_iso(row.get("last_used_at")),
        "revoked_at":   _ts_to_iso(row.get("revoked_at")),
        "revoked_reason": row.get("revoked_reason"),
        "metadata":     meta or {},
    }


def _parse_date_param(value: Optional[str], default: date) -> date:
    if not value:
        return default
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_date", "message": f"Expected YYYY-MM-DD, got {value!r}"},
        )


# ---------------------------------------------------------------------------
# Key CRUD
# ---------------------------------------------------------------------------

@router.post("/keys", status_code=201)
async def create_key(
    req: CreateKeyRequest,
    _: None = Depends(require_admin),
) -> dict:
    """
    Create a new API key.

    The full plaintext key is returned **once** and never stored.
    Store it securely — it cannot be recovered.
    """
    from autotune.db.store import get_db

    full_key, key_hash = generate_api_key()
    key_id = str(uuid.uuid4())
    now = time.time()

    db_row = {
        "id":         key_id,
        "name":       req.name,
        "key_prefix": key_display_prefix(full_key),
        "key_hash":   key_hash,
        "is_active":  1,
        "created_at": now,
        "metadata":   json.dumps(req.metadata or {}),
    }

    try:
        get_db().create_api_key(db_row)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "db_error", "message": str(exc)})

    return {
        "id":         key_id,
        "name":       req.name,
        "key":        full_key,
        "key_prefix": key_display_prefix(full_key),
        "created_at": _ts_to_iso(now),
        "metadata":   req.metadata or {},
        "warning":    (
            "Store this key securely — it will NOT be shown again. "
            "All future requests must include: "
            "Authorization: Bearer <key>"
        ),
    }


@router.get("/keys")
async def list_keys(
    _: None = Depends(require_admin),
    include_revoked: bool = Query(False, description="Include revoked keys"),
) -> dict:
    """List all API keys (metadata only — no plaintext keys)."""
    from autotune.db.store import get_db

    rows = get_db().list_api_keys(include_revoked=include_revoked)
    return {
        "keys":  [_key_to_dict(r) for r in rows],
        "total": len(rows),
    }


@router.get("/keys/{key_id}")
async def get_key(
    key_id: str,
    _: None = Depends(require_admin),
) -> dict:
    """Get metadata and a 30-day usage summary for one API key."""
    from autotune.db.store import get_db
    db = get_db()

    row = db.get_api_key_by_id(key_id)
    if not row:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Key not found"})

    start = (date.today() - timedelta(days=30)).isoformat()
    end   = date.today().isoformat()
    usage = db.get_api_usage(key_id=key_id, start_day=start, end_day=end)
    totals = _aggregate_totals(usage)

    return {
        **_key_to_dict(row),
        "usage_30d": {
            "start":    start,
            "end":      end,
            "rows":     usage,
            "totals":   totals,
        },
    }


@router.delete("/keys/{key_id}")
async def revoke_key(
    key_id: str,
    body: Optional[RevokeKeyRequest] = None,
    _: None = Depends(require_admin),
) -> dict:
    """Revoke an API key.  Any in-flight requests complete; new requests are rejected."""
    from autotune.db.store import get_db
    db = get_db()

    row = db.get_api_key_by_id(key_id)
    if not row:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Key not found"})

    if not row.get("is_active"):
        raise HTTPException(
            status_code=409,
            detail={"error": "already_revoked", "message": "Key is already revoked"},
        )

    reason = (body.reason if body else None) or None
    ok = db.revoke_api_key(key_id, reason=reason)
    if not ok:
        raise HTTPException(status_code=500, detail={"error": "db_error", "message": "Revocation failed"})

    # Flush the key from the in-memory auth cache
    invalidate_key_cache_by_id(key_id)

    return {
        "revoked":  key_id,
        "name":     row["name"],
        "reason":   reason,
        "revoked_at": _ts_to_iso(time.time()),
    }


# ---------------------------------------------------------------------------
# Usage analytics
# ---------------------------------------------------------------------------

def _rows_to_csv(rows: list[dict], filename: str) -> StreamingResponse:
    if not rows:
        buf = io.StringIO()
        buf.write("")
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _aggregate_totals(rows: list[dict]) -> dict:
    total_req = sum(r.get("request_count", 0) for r in rows)
    total_prompt = sum(r.get("prompt_tokens", 0) for r in rows)
    total_comp = sum(r.get("completion_tokens", 0) for r in rows)
    total_err = sum(r.get("error_count", 0) for r in rows)
    latencies = [r["avg_latency_ms"] for r in rows if r.get("avg_latency_ms") is not None]
    ttfts     = [r["avg_ttft_ms"]     for r in rows if r.get("avg_ttft_ms")     is not None]
    return {
        "request_count":      total_req,
        "prompt_tokens":      total_prompt,
        "completion_tokens":  total_comp,
        "total_tokens":       total_prompt + total_comp,
        "error_count":        total_err,
        "avg_latency_ms":     round(sum(latencies) / len(latencies), 1) if latencies else None,
        "avg_ttft_ms":        round(sum(ttfts) / len(ttfts), 1)         if ttfts     else None,
    }


@router.get("/usage")
async def get_usage(
    _: None = Depends(require_admin),
    start:    Optional[str] = Query(None, description="Start date YYYY-MM-DD (default: 30 days ago)"),
    end:      Optional[str] = Query(None, description="End date YYYY-MM-DD (default: today)"),
    key_id:   Optional[str] = Query(None, description="Filter to a single key ID"),
    model_id: Optional[str] = Query(None, description="Filter to a single model ID"),
    format:   Optional[str] = Query(None, description="Response format: json (default) or csv"),
) -> Any:
    """
    Return per-key per-day consumption within the requested date window.

    Default window is the trailing 30 days.  Each row in ``usage`` represents
    one (key, day, model) combination with aggregate token counts and latency.
    The response also includes cross-key totals and a per-key breakdown.

    ---
    Example
    -------
    GET /admin/usage?start=2026-05-01&end=2026-05-19

    {
      "period": {"start": "2026-05-01", "end": "2026-05-19"},
      "usage": [
        {
          "day": "2026-05-19",
          "key_id": "...",
          "key_name": "prod-client",
          "key_prefix": "sk-at-xxxx...",
          "model_id": "qwen3:8b",
          "backend": "ollama",
          "request_count": 42,
          "prompt_tokens": 8400,
          "completion_tokens": 12600,
          "total_tokens": 21000,
          "avg_latency_ms": 1234.5,
          "avg_ttft_ms": 234.1,
          "error_count": 0
        }
      ],
      "totals": { ... },
      "by_key": [ ... ]
    }
    """
    from autotune.db.store import get_db
    db = get_db()

    today  = date.today()
    start_d = _parse_date_param(start, today - timedelta(days=30))
    end_d   = _parse_date_param(end,   today)

    if start_d > end_d:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_range", "message": "start must be ≤ end"},
        )

    # Fetch detailed rows (key × day × model)
    rows = db.get_api_usage(
        key_id=key_id,
        model_id=model_id,
        start_day=start_d.isoformat(),
        end_day=end_d.isoformat(),
    )

    # Per-key rollup
    key_totals: dict[str, dict] = {}
    for r in rows:
        kid = r["key_id"]
        if kid not in key_totals:
            key_totals[kid] = {
                "key_id":     kid,
                "key_name":   r.get("key_name"),
                "key_prefix": r.get("key_prefix"),
                "request_count":     0,
                "prompt_tokens":     0,
                "completion_tokens": 0,
                "total_tokens":      0,
                "error_count":       0,
            }
        kt = key_totals[kid]
        kt["request_count"]     += r.get("request_count",     0)
        kt["prompt_tokens"]     += r.get("prompt_tokens",     0)
        kt["completion_tokens"] += r.get("completion_tokens", 0)
        kt["total_tokens"]      += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
        kt["error_count"]       += r.get("error_count",       0)

    if format == "csv":
        csv_rows = [
            {
                "day":               r.get("day"),
                "key_name":          r.get("key_name"),
                "key_prefix":        r.get("key_prefix"),
                "model_id":          r.get("model_id"),
                "backend":           r.get("backend"),
                "request_count":     r.get("request_count", 0),
                "prompt_tokens":     r.get("prompt_tokens", 0),
                "completion_tokens": r.get("completion_tokens", 0),
                "total_tokens":      (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0),
                "avg_latency_ms":    r.get("avg_latency_ms"),
                "avg_ttft_ms":       r.get("avg_ttft_ms"),
                "error_count":       r.get("error_count", 0),
            }
            for r in rows
        ]
        filename = f"autotune-usage-{start_d.isoformat()}-to-{end_d.isoformat()}.csv"
        return _rows_to_csv(csv_rows, filename)

    return {
        "period": {
            "start": start_d.isoformat(),
            "end":   end_d.isoformat(),
            "days":  (end_d - start_d).days + 1,
        },
        "usage":    rows,
        "totals":   _aggregate_totals(rows),
        "by_key":   sorted(key_totals.values(), key=lambda x: -x["total_tokens"]),
    }


@router.get("/usage/summary")
async def get_usage_summary(
    _: None = Depends(require_admin),
    days:   int            = Query(30, ge=1, le=365, description="Trailing N days"),
    format: Optional[str]  = Query(None, description="Response format: json (default) or csv"),
) -> Any:
    """
    High-level usage summary for the trailing N days.

    Returns one row per API key with total tokens, requests, and error rate.
    Suitable for a billing overview or admin dashboard card.
    """
    from autotune.db.store import get_db
    db = get_db()

    today   = date.today()
    start_d = today - timedelta(days=days - 1)

    rows = db.get_api_usage(
        start_day=start_d.isoformat(),
        end_day=today.isoformat(),
    )

    # Roll up per key
    key_map: dict[str, dict] = {}
    for r in rows:
        kid = r["key_id"]
        if kid not in key_map:
            key_map[kid] = {
                "key_id":         kid,
                "key_name":       r.get("key_name"),
                "key_prefix":     r.get("key_prefix"),
                "request_count":  0,
                "prompt_tokens":  0,
                "completion_tokens": 0,
                "total_tokens":   0,
                "error_count":    0,
                "active_days":    set(),
            }
        km = key_map[kid]
        km["request_count"]     += r.get("request_count",     0)
        km["prompt_tokens"]     += r.get("prompt_tokens",     0)
        km["completion_tokens"] += r.get("completion_tokens", 0)
        km["total_tokens"]      += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
        km["error_count"]       += r.get("error_count",       0)
        km["active_days"].add(r.get("day"))

    summary = []
    for km in key_map.values():
        req = km["request_count"]
        summary.append({
            **{k: v for k, v in km.items() if k != "active_days"},
            "active_days":   len(km["active_days"]),
            "error_rate_pct": round(km["error_count"] / req * 100, 1) if req else 0.0,
        })

    summary.sort(key=lambda x: -x["total_tokens"])

    if format == "csv":
        filename = f"autotune-usage-summary-{start_d.isoformat()}-to-{today.isoformat()}.csv"
        return _rows_to_csv(summary, filename)

    all_rows_totals = _aggregate_totals(rows)
    return {
        "period": {
            "start": start_d.isoformat(),
            "end":   today.isoformat(),
            "days":  days,
        },
        "totals":  all_rows_totals,
        "by_key":  summary,
    }
