"""Dashboard FastAPI router — /dashboard UI and /api/dashboard/* JSON endpoints.

Authentication
--------------
When AUTOTUNE_ADMIN_KEY is set, the dashboard and all /api/dashboard/* endpoints
require a signed session cookie obtained by logging in at /dashboard/login.

When AUTOTUNE_ADMIN_KEY is NOT set, all dashboard endpoints are open (backward-
compatible default for single-user installs).

Cookie: autotune_session (httpOnly, samesite=strict, 24-hour expiry)
Signing: itsdangerous.URLSafeTimedSerializer keyed on AUTOTUNE_ADMIN_KEY
"""
from __future__ import annotations

import json as _json
import os
import secrets
import time as _time
import uuid as _uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, field_validator

import collections as _collections
import hashlib as _hashlib
import threading as _threading

from autotune.api.security_log import audit as _audit, real_ip as _real_ip
from autotune.db.store import _SAFE_COL

router = APIRouter(tags=["dashboard"])

# ---------------------------------------------------------------------------
# Dashboard API rate limiter — sliding window, per client IP
#
# Three tiers:
#   write  — key create / revoke              30 ops  / hour
#   refresh— catalog refresh                  10 ops  / min
#   read   — all other dashboard reads       300 reqs / min
#
# The session gate already requires authentication; rate limiting adds DoS
# protection and limits blast radius from a compromised session.
# ---------------------------------------------------------------------------

class _SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_sec: int) -> None:
        self._max = max_requests
        self._window = window_sec
        self._buckets: dict[str, _collections.deque] = {}
        self._lock = _threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = _time.time()
        cutoff = now - self._window
        with self._lock:
            bucket = self._buckets.setdefault(key, _collections.deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._max:
                return False
            bucket.append(now)
            return True


_write_limiter   = _SlidingWindowLimiter(30,  3600)  # 30/hr  — key create/revoke
_refresh_limiter = _SlidingWindowLimiter(10,    60)  # 10/min — catalog refresh
_read_limiter    = _SlidingWindowLimiter(300,   60)  # 300/min — reads


def _check_rate_limit(request: Request, limiter: _SlidingWindowLimiter) -> None:
    """Raise 429 if the per-IP rate limit is exceeded."""
    if not limiter.is_allowed(_real_ip(request)):
        _audit("dashboard_rate_limited", ip=_real_ip(request),
               path=request.url.path)
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limited",
                    "message": "Too many requests. Slow down and try again."},
        )


# ---------------------------------------------------------------------------
# Login rate-limiter — in-memory, per source IP
#
# After _LOCKOUT_THRESHOLD consecutive failures the IP is locked out for an
# exponentially growing delay (2^(n - threshold) seconds, capped at 5 min).
# A successful login resets the counter for that IP.
# ---------------------------------------------------------------------------

_LOCKOUT_THRESHOLD = 5   # failures before lockout kicks in
_LOCKOUT_MAX_SEC   = 300  # hard cap: 5 minutes

# ip -> {"failures": int, "locked_until": float}
_login_attempts: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Session revocation — SHA-256 hashes of revoked tokens
#
# The in-memory set is the fast path; SQLite is the backing store so revocations
# survive server restarts.  Hashes are used (never raw cookie values) so a DB
# breach cannot replay stolen tokens.
# ---------------------------------------------------------------------------

_revoked_sessions: set[str] = set()   # SHA-256 hashes of revoked tokens
_revoked_loaded = False                # True once we've pulled non-expired rows from DB


def _token_hash(token: str) -> str:
    return _hashlib.sha256(token.encode()).hexdigest()


def _load_revoked_from_db() -> None:
    """Populate _revoked_sessions from SQLite on first use; prune expired rows."""
    global _revoked_sessions, _revoked_loaded
    if _revoked_loaded:
        return
    try:
        from autotune.db.store import get_db
        _revoked_sessions = get_db().load_revoked_session_hashes()
    except Exception:
        pass  # DB not available yet; in-memory set still works for this process
    _revoked_loaded = True


def _login_is_locked(ip: str) -> tuple[bool, int]:
    """Return (locked, seconds_remaining). Only clears entries whose lock has expired."""
    rec = _login_attempts.get(ip)
    if not rec:
        return False, 0
    locked_until = rec.get("locked_until", 0.0)
    if locked_until <= 0:
        # Failures accumulating but no lock set yet
        return False, 0
    remaining = locked_until - _time.time()
    if remaining > 0:
        return True, int(remaining) + 1
    # Lock window has passed — clear entry so the IP gets a clean slate
    _login_attempts.pop(ip, None)
    return False, 0


def _login_record_failure(ip: str) -> None:
    rec = _login_attempts.setdefault(ip, {"failures": 0, "locked_until": 0.0})
    rec["failures"] += 1
    excess = rec["failures"] - _LOCKOUT_THRESHOLD
    if excess >= 0:
        delay = min(2 ** (excess + 1), _LOCKOUT_MAX_SEC)
        rec["locked_until"] = _time.time() + delay


def _login_reset(ip: str) -> None:
    _login_attempts.pop(ip, None)

_STATIC_DIR = Path(__file__).parent / "static"
_COOKIE_NAME = "autotune_session"
_SESSION_MAX_AGE = 86400  # 24 hours

# Security headers added to every HTML response served by the dashboard.
# connect-src 'self' prevents AJAX exfiltration even if XSS occurs.
# frame-ancestors 'none' prevents clickjacking.
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    ),
    "X-Frame-Options":      "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy":      "strict-origin",
    "Permissions-Policy":   (
        "camera=(), microphone=(), geolocation=(), "
        "payment=(), usb=(), interest-cohort=()"
    ),
    "Cache-Control":        "no-store",
    "Pragma":               "no-cache",
}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _admin_key() -> str:
    return os.environ.get("AUTOTUNE_ADMIN_KEY", "").strip()


def _signer():
    """Return a URLSafeTimedSerializer, or None if no admin key is configured."""
    key = _admin_key()
    if not key:
        return None
    from itsdangerous import URLSafeTimedSerializer
    return URLSafeTimedSerializer(key, salt="autotune-dashboard-v1")


def _verify_session(request: Request) -> bool:
    """Return True if the request carries a valid session cookie.

    Also returns True unconditionally when no admin key is configured —
    dashboard is open in that mode (backward-compat for single-user installs).
    """
    signer = _signer()
    if signer is None:
        return True  # no admin key → open access
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        return False
    _load_revoked_from_db()
    if _token_hash(token) in _revoked_sessions:
        return False
    try:
        from itsdangerous import BadSignature, SignatureExpired
        signer.loads(token, max_age=_SESSION_MAX_AGE)
        return True
    except Exception:
        return False


def _require_session_api(request: Request) -> None:
    """FastAPI dependency for /api/dashboard/* — raises 401 when not authenticated."""
    if not _verify_session(request):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "not_authenticated",
                "message": "Session expired or missing. Visit /dashboard/login to sign in.",
            },
        )


# ---------------------------------------------------------------------------
# Login page HTML (inline — matches index.html dark theme)
# ---------------------------------------------------------------------------

def _login_html(error: Optional[str] = None) -> str:
    error_block = (
        f'<div class="msg error">{error}</div>' if error else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>autotune — sign in</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:      #0d1117; --surface:  #161b22; --surface2: #21262d;
      --border:  #30363d; --text:    #e6edf3;  --muted:    #8b949e;
      --accent:  #58a6ff; --red:     #f85149;  --green:    #3fb950;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      background: var(--bg); color: var(--text);
      min-height: 100vh; display: flex;
      align-items: center; justify-content: center; font-size: 14px;
    }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 40px 36px;
      width: 100%; max-width: 400px;
    }}
    .logo {{
      font-size: 20px; font-weight: 700; color: var(--accent);
      letter-spacing: -0.5px; margin-bottom: 6px;
      display: flex; align-items: center; gap: 10px;
    }}
    .subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: 32px; }}
    label {{
      display: block; font-size: 11px; font-weight: 600;
      color: var(--muted); text-transform: uppercase;
      letter-spacing: 0.6px; margin-bottom: 6px;
    }}
    input[type=password] {{
      width: 100%; padding: 10px 12px;
      background: var(--bg); border: 1px solid var(--border);
      border-radius: 6px; color: var(--text);
      font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
      outline: none; transition: border-color 0.15s;
    }}
    input[type=password]:focus {{ border-color: var(--accent); }}
    button {{
      margin-top: 16px; width: 100%; padding: 11px;
      background: var(--accent); color: #0d1117;
      font-weight: 700; font-size: 14px;
      border: none; border-radius: 6px;
      cursor: pointer; transition: opacity 0.15s;
    }}
    button:hover {{ opacity: 0.85; }}
    .msg {{
      margin-top: 14px; padding: 10px 14px;
      border-radius: 6px; font-size: 13px;
    }}
    .msg.error {{
      background: rgba(248,81,73,0.10); border: 1px solid rgba(248,81,73,0.3);
      color: var(--red);
    }}
    .hint {{
      margin-top: 24px; padding-top: 18px; border-top: 1px solid var(--border);
      color: var(--muted); font-size: 12px; line-height: 1.65;
    }}
    code {{
      background: var(--surface2); padding: 1px 5px; border-radius: 3px;
      font-family: 'SF Mono', 'Fira Code', monospace; font-size: 11px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.2"
           stroke-linecap="round" stroke-linejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
      </svg>
      autotune
    </div>
    <div class="subtitle">Private Gateway &mdash; admin sign in</div>

    <form method="post" action="/dashboard/login">
      <label for="pw">Admin key</label>
      <input type="password" id="pw" name="password"
             placeholder="Your AUTOTUNE_ADMIN_KEY"
             autofocus autocomplete="current-password">
      <button type="submit">Sign in &rarr;</button>
    </form>

    {error_block}

    <div class="hint">
      Enter the value of <code>AUTOTUNE_ADMIN_KEY</code> configured on this
      server.  First time? Run <code>./setup.sh</code> to generate a key and
      write it to <code>.env</code>.
    </div>
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public endpoints — no session required
# ---------------------------------------------------------------------------

def _html_response(content: str, status_code: int = 200) -> HTMLResponse:
    """Return an HTMLResponse with dashboard security headers applied."""
    resp = HTMLResponse(content, status_code=status_code)
    for k, v in _SECURITY_HEADERS.items():
        resp.headers[k] = v
    return resp


@router.get("/dashboard/login", include_in_schema=False)
async def login_page(request: Request):
    """Serve the login page; redirect to /dashboard if already signed in."""
    if _verify_session(request):
        return RedirectResponse("/dashboard", status_code=302)
    return _html_response(_login_html())


@router.post("/dashboard/login", include_in_schema=False)
async def login_submit(request: Request):
    """Validate the admin key; set a signed session cookie on success."""
    import logging as _logging
    _log = _logging.getLogger(__name__)

    ip = _real_ip(request)

    # ── Rate-limit check ─────────────────────────────────────────────────
    locked, secs = _login_is_locked(ip)
    if locked:
        _log.warning("dashboard login blocked (rate-limit) ip=%s remaining=%ds", ip, secs)
        _audit("login_rate_limited", ip=ip, retry_after=secs)
        return _html_response(
            _login_html(
                error=f"Too many failed attempts. Try again in {secs} second{'s' if secs != 1 else ''}."
            ),
            status_code=429,
        )

    form = await request.form()
    password = str(form.get("password", ""))
    admin_key = _admin_key()

    if not admin_key:
        # No admin key configured — pass straight through (open mode).
        return RedirectResponse("/dashboard", status_code=303)

    if not secrets.compare_digest(password.encode("utf-8"), admin_key.encode("utf-8")):
        _login_record_failure(ip)
        failures = _login_attempts.get(ip, {}).get("failures", 1)
        _log.warning("dashboard login failed ip=%s attempt=%d", ip, failures)
        _audit("login_failure", ip=ip, attempt=failures)
        remaining_attempts = max(_LOCKOUT_THRESHOLD - failures, 0)
        hint = (
            f" ({remaining_attempts} attempt{'s' if remaining_attempts != 1 else ''} left before lockout)"
            if 0 < remaining_attempts < _LOCKOUT_THRESHOLD
            else ""
        )
        return _html_response(
            _login_html(error=f"Incorrect admin key. Please try again.{hint}"),
            status_code=401,
        )

    # ── Success ───────────────────────────────────────────────────────────
    _login_reset(ip)
    _log.info("dashboard login success ip=%s", ip)
    _audit("login_success", ip=ip)

    signer = _signer()
    token = signer.dumps({"u": "admin"})

    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie(
        _COOKIE_NAME,
        value=token,
        max_age=_SESSION_MAX_AGE,
        httponly=True,
        samesite="strict",
        path="/",
    )
    return response


@router.get("/dashboard/logout", include_in_schema=False)
async def logout(request: Request):
    """Clear the session cookie and redirect to the login page."""
    token = request.cookies.get(_COOKIE_NAME)
    if token:
        h = _token_hash(token)
        _revoked_sessions.add(h)
        try:
            from autotune.db.store import get_db
            get_db().add_revoked_session(h, expires_at=_time.time() + _SESSION_MAX_AGE)
        except Exception:
            pass  # in-memory revocation still active for this process
        _audit("session_logout", ip=_real_ip(request))
    resp = RedirectResponse("/dashboard/login", status_code=303)
    resp.delete_cookie(_COOKIE_NAME, path="/")
    return resp


@router.get("/api/dashboard/auth-status", include_in_schema=False)
async def auth_status():
    """Return whether auth is required on this server (no session needed)."""
    return {"auth_required": bool(_admin_key())}


# ---------------------------------------------------------------------------
# Dashboard UI — redirects to login when unauthenticated
# ---------------------------------------------------------------------------

@router.get("/dashboard", include_in_schema=False)
async def dashboard_ui(request: Request):
    if not _verify_session(request):
        return RedirectResponse("/dashboard/login", status_code=302)
    index = _STATIC_DIR / "index.html"
    if index.exists():
        from fastapi.responses import Response as _Response
        content = index.read_bytes()
        return _Response(
            content=content,
            media_type="text/html",
            headers=_SECURITY_HEADERS,
        )
    return _html_response("<h1>Dashboard static files not found</h1>", status_code=500)


# ---------------------------------------------------------------------------
# Protected API endpoints — require a valid session cookie
# ---------------------------------------------------------------------------

@router.get("/api/dashboard/overview", dependencies=[Depends(_require_session_api)])
async def dashboard_overview(request: Request):
    _check_rate_limit(request, _read_limiter)
    from .metrics import get_overview
    return get_overview()


@router.get("/api/dashboard/requests", dependencies=[Depends(_require_session_api)])
async def dashboard_requests():
    from .metrics import get_requests_timeseries
    return {"data": get_requests_timeseries()}


@router.get("/api/dashboard/ttft_trend", dependencies=[Depends(_require_session_api)])
async def dashboard_ttft_trend():
    from .metrics import get_ttft_trend
    return {"data": get_ttft_trend()}


@router.get("/api/dashboard/models", dependencies=[Depends(_require_session_api)])
async def dashboard_models():
    from .metrics import get_models_stats
    return {"models": get_models_stats()}


@router.get("/api/dashboard/comparison", dependencies=[Depends(_require_session_api)])
async def dashboard_comparison():
    from .metrics import get_comparison
    return get_comparison()


@router.get("/api/dashboard/keys", dependencies=[Depends(_require_session_api)])
async def dashboard_keys():
    from .metrics import get_api_keys_summary
    return {"keys": get_api_keys_summary()}


@router.get("/api/dashboard/usage/export", dependencies=[Depends(_require_session_api)])
async def dashboard_usage_export(
    days:     int           = Query(30, ge=1, le=365, description="Trailing N days"),
    key_id:   str           = Query("",  description="Filter to one key ID (optional)"),
    model_id: str           = Query("",  description="Filter to one model ID (optional)"),
):
    """Download per-key usage as a CSV file — session-cookie authenticated."""
    import csv
    import io
    from datetime import date, timedelta
    from autotune.db.store import get_db

    today   = date.today()
    start_d = today - timedelta(days=days - 1)
    rows = get_db().get_api_usage(
        key_id=key_id or None,
        model_id=model_id or None,
        start_day=start_d.isoformat(),
        end_day=today.isoformat(),
    )

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

    buf = io.StringIO()
    if csv_rows:
        writer = csv.DictWriter(buf, fieldnames=list(csv_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    buf.seek(0)

    filename = f"autotune-usage-{start_d.isoformat()}-to-{today.isoformat()}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/api/dashboard/slow", dependencies=[Depends(_require_session_api)])
async def dashboard_slow():
    from .metrics import get_slow_requests
    return {"requests": get_slow_requests()}


@router.get("/api/dashboard/suggestions", dependencies=[Depends(_require_session_api)])
async def dashboard_suggestions():
    from .metrics import get_suggestions
    return {"suggestions": get_suggestions()}


@router.get("/api/dashboard/onboarding", dependencies=[Depends(_require_session_api)])
async def dashboard_onboarding():
    from .metrics import get_onboarding_state
    return get_onboarding_state()


@router.get("/api/dashboard/security", dependencies=[Depends(_require_session_api)])
async def dashboard_security(request: Request, event_filter: str = "", severity: str = ""):
    _check_rate_limit(request, _read_limiter)
    from .metrics import (
        get_gateway_security,
        get_security_events_recent,
        get_security_stats_24h,
    )
    checks   = get_gateway_security()
    stats    = get_security_stats_24h()
    events   = get_security_events_recent(
        limit=200,
        event_filter=event_filter or None,
        severity_filter=severity or None,
    )

    # Running model state for the gateway status banner
    import os as _os
    running_models: list[dict] = []
    try:
        from autotune.api.running_models import get_running_models
        running_models = [{"name": m.name, "backend": m.backend} for m in get_running_models()]
    except Exception:
        pass
    api_key_on = _os.environ.get("AUTOTUNE_REQUIRE_API_KEY", "").strip() in ("1", "true", "yes")

    # Compute a security score from posture-category checks only.
    # "ok" and "info" both earn points — "info" means advisory/FYI, not a gap.
    # Only "warn" and "error" reduce the score.
    posture_checks = [c for c in checks if c.get("category", "security") == "security"]
    ok_count  = sum(1 for c in posture_checks if c["status"] in ("ok", "info"))
    total     = len(posture_checks)
    pct       = (ok_count / total * 100) if total else 100
    grade     = "A+" if pct == 100 else "A" if pct >= 90 else "B" if pct >= 75 else "C" if pct >= 60 else "D" if pct >= 40 else "F"

    return {
        "checks": checks,            # full list (overview card uses this)
        "posture": posture_checks,   # security-only subset (Security tab uses this)
        "score": {"earned": ok_count, "total": total, "pct": round(pct, 1), "grade": grade},
        "stats_24h": stats,
        "events": events,
        "gateway": {
            "model_count": len(running_models),
            "running_models": running_models,
            "api_key_enforcement": api_key_on,
        },
    }


@router.get("/api/dashboard/activity", dependencies=[Depends(_require_session_api)])
async def dashboard_activity():
    from .metrics import get_recent_activity
    return get_recent_activity()


@router.get("/api/dashboard/installed-models", dependencies=[Depends(_require_session_api)])
async def dashboard_installed_models():
    from .metrics import get_installed_models
    return {"models": get_installed_models()}


@router.get("/api/dashboard/sessions", dependencies=[Depends(_require_session_api)])
async def dashboard_sessions(limit: int = 50):
    """Return conversation sessions grouped from gateway_log."""
    from autotune.api.conversation import get_conv_manager
    limit = max(1, min(limit, 200))
    mgr = get_conv_manager()
    return {"sessions": mgr.list_sessions(limit=limit)}


@router.get("/api/dashboard/sessions/{session_id:path}", dependencies=[Depends(_require_session_api)])
async def dashboard_session_detail(session_id: str):
    """
    Return all turns for a session.
    session_id is either an integer (gateway_log session) or "conv:<conv_id>"
    (named conversation from the conversations table).
    """
    from autotune.api.conversation import get_conv_manager
    mgr = get_conv_manager()

    if session_id.startswith("conv:"):
        conv_id = session_id[5:]
        import re as _re
        if not _re.match(r'^[a-zA-Z0-9_\-]{1,64}$', conv_id):
            raise HTTPException(status_code=422, detail="Invalid conversation ID")
        turns = mgr.get_conversation_turns(conv_id)
        # Attach model_id from the conversation record
        conv = mgr.get(conv_id)
        model_id = conv["model_id"] if conv else None
        for t in turns:
            if t["model_id"] is None:
                t["model_id"] = model_id
    else:
        try:
            sid_int = int(session_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid session_id")
        turns = mgr.get_session_turns(sid_int)

    if not turns:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Session not found"})
    return {"session_id": session_id, "turns": turns}


@router.get("/api/dashboard/conversations", dependencies=[Depends(_require_session_api)])
async def dashboard_conversations(limit: int = 50):
    """Return explicitly-tracked conversations (those using X-Conversation-Id)."""
    from autotune.api.conversation import get_conv_manager
    limit = max(1, min(limit, 200))
    mgr = get_conv_manager()
    return {"conversations": mgr.list_all(limit=limit)}


@router.get("/api/dashboard/conversations/{conv_id}", dependencies=[Depends(_require_session_api)])
async def dashboard_conversation_detail(conv_id: str):
    """Return metadata + all messages for a specific conversation."""
    import re
    if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', conv_id):
        raise HTTPException(status_code=422, detail="Invalid conversation ID")
    from autotune.api.conversation import get_conv_manager
    mgr = get_conv_manager()
    conv = mgr.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "Conversation not found"})
    messages = mgr.get_messages(conv_id)
    return {"conversation": conv, "messages": messages}


def _validate_ollama_model_name(v: str) -> str:
    """Validate and normalise an Ollama model name.

    Valid shapes
    ------------
    name                        qwen3
    name:tag                    qwen3:8b
    namespace/name:tag          library/qwen3:8b
    registry/namespace/name:tag registry.ollama.ai/library/qwen3:8b

    Rejects
    -------
    Anything containing '://' — prevents SSRF where Ollama follows URL-like
    model references to internal services (e.g. http://internal:11434/...).
    '@' — credential embedding (user:pass@host).
    Leading/double slashes and control characters.
    """
    import re
    v = v.strip()
    if not v:
        raise ValueError("model must not be empty")
    if len(v) > 200:
        raise ValueError("model name too long")
    # Explicit URL-scheme check — belt-and-suspenders before the shape regex
    if re.search(r'[a-zA-Z][a-zA-Z0-9+\-.]*://', v):
        raise ValueError("Model name must not be a URL")
    if '@' in v or '//' in v:
        raise ValueError("Model name contains invalid characters")
    # Shape: [registry/][namespace/]name[:tag]
    # Each component: starts with alphanum, may contain alphanum, dot, hyphen, underscore
    _SEG = r'[a-zA-Z0-9][a-zA-Z0-9._\-]*'
    if not re.match(rf'^{_SEG}(?:/{_SEG})*(?::{_SEG})?$', v):
        raise ValueError(
            "Invalid model name — expected name[:tag] or namespace/name[:tag]"
        )
    return v


class _PullBody(BaseModel):
    model: str

    @field_validator("model")
    @classmethod
    def _model_valid(cls, v: str) -> str:
        return _validate_ollama_model_name(v)


@router.get("/api/dashboard/pull-check", dependencies=[Depends(_require_session_api)])
async def dashboard_pull_check(model: str, source: str = "ollama") -> dict:
    """Return a feasibility verdict for pulling *model* without downloading it."""
    from autotune.api.model_guard import check_feasibility
    try:
        model = _validate_ollama_model_name(model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    result = check_feasibility(model, source=source)
    return result.to_dict()


@router.post("/api/dashboard/pull", dependencies=[Depends(_require_session_api)])
async def dashboard_pull_model(req: _PullBody) -> dict:
    """Trigger an Ollama model pull in the background (blocked if infeasible)."""
    import asyncio
    import os as _os
    import httpx as _httpx
    from autotune.api.model_guard import check_feasibility

    guard = check_feasibility(req.model, source="ollama")
    if guard.verdict == "blocked":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "infeasible",
                "message": guard.reason,
                "feasibility": guard.to_dict(),
            },
        )

    ollama_url = _os.environ.get("AUTOTUNE_OLLAMA_URL", "http://localhost:11434").rstrip("/")

    async def _pull() -> None:
        try:
            async with _httpx.AsyncClient(timeout=600) as c:
                await c.post(f"{ollama_url}/api/pull", json={"name": req.model, "stream": False})
        except Exception:
            pass

    asyncio.create_task(_pull())
    response: dict = {"status": "pulling", "model": req.model}
    if guard.verdict == "warn":
        response["warning"] = guard.reason
    return response


class _PullHFBody(BaseModel):
    repo_id: str

    @field_validator("repo_id")
    @classmethod
    def _repo_valid(cls, v: str) -> str:
        import re
        v = v.strip()
        if not v:
            raise ValueError("repo_id must not be empty")
        if len(v) > 200:
            raise ValueError("repo_id too long")
        if re.search(r'[a-zA-Z][a-zA-Z0-9+\-.]*://', v) or '@' in v or '//' in v:
            raise ValueError("repo_id must not be a URL or contain invalid characters")
        # HuggingFace repo IDs are strictly owner/repo — exactly one slash
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._\-]*/[a-zA-Z0-9][a-zA-Z0-9._\-]*$', v):
            raise ValueError("Invalid HuggingFace repo_id format (expected owner/repo)")
        return v


@router.post("/api/dashboard/pull-hf", dependencies=[Depends(_require_session_api)])
async def dashboard_pull_hf(req: _PullHFBody) -> dict:
    """Download an MLX model from HuggingFace in the background."""
    import asyncio
    from autotune.api.model_guard import check_feasibility

    guard = check_feasibility(req.repo_id, source="mlx")
    if guard.verdict == "blocked":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "infeasible",
                "message": guard.reason,
                "feasibility": guard.to_dict(),
            },
        )

    async def _download() -> None:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=req.repo_id, ignore_patterns=["*.md", "*.txt"])
        except Exception:
            pass

    asyncio.create_task(_download())
    response: dict = {"status": "downloading", "repo_id": req.repo_id}
    if guard.verdict == "warn":
        response["warning"] = guard.reason
    return response


# ---------------------------------------------------------------------------
# Key management proxy — session-protected so the admin key never touches
# the browser.  These mirror /admin/keys but gate on the session cookie.
# ---------------------------------------------------------------------------

class _CreateKeyBody(BaseModel):
    name: str
    metadata: Optional[dict] = None
    expires_days: Optional[int] = None  # None = never expires; 1–3650 = TTL in days

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty")
        if len(v) > 128:
            raise ValueError("name must be ≤ 128 characters")
        return v

    @field_validator("expires_days")
    @classmethod
    def _expires_valid(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 3650):
            raise ValueError("expires_days must be between 1 and 3650")
        return v


@router.get("/api/dashboard/tokens", dependencies=[Depends(_require_session_api)])
async def dashboard_tokens():
    from .metrics import get_token_timeseries
    return {"data": get_token_timeseries()}


@router.get("/api/dashboard/usage-summary", dependencies=[Depends(_require_session_api)])
async def dashboard_usage_summary():
    from .metrics import get_usage_summary
    return get_usage_summary()


@router.get("/api/dashboard/catalog", dependencies=[Depends(_require_session_api)])
async def dashboard_catalog():
    """Return the full model catalog with metadata."""
    from autotune.models.catalog import load_catalog, format_for_api
    from autotune.models.catalog_updater import refresh_if_stale
    catalog = load_catalog()
    refresh_if_stale(catalog, background=True)
    return format_for_api(catalog)


@router.post("/api/dashboard/catalog/refresh", dependencies=[Depends(_require_session_api)])
async def dashboard_catalog_refresh(request: Request):
    """Force a synchronous catalog refresh and return the updated catalog."""
    _check_rate_limit(request, _refresh_limiter)
    from autotune.models.catalog_updater import force_refresh
    updated, new_count, upd_count = force_refresh()
    from autotune.models.catalog import format_for_api
    return {**format_for_api(updated), "new_count": new_count, "updated_count": upd_count}


@router.post(
    "/api/dashboard/admin/keys",
    dependencies=[Depends(_require_session_api)],
    status_code=201,
    include_in_schema=False,
)
async def dashboard_create_key(req: _CreateKeyBody, request: Request) -> dict:
    """Create a new API key via the dashboard (returns plaintext once)."""
    _check_rate_limit(request, _write_limiter)
    from autotune.api.auth import generate_api_key, key_display_prefix
    from autotune.db.store import get_db

    full_key, key_hash = generate_api_key()
    key_id = str(_uuid.uuid4())
    now = _time.time()

    expires_at = (now + req.expires_days * 86400) if req.expires_days else None

    db_row: dict = {
        "id":         key_id,
        "name":       req.name,
        "key_prefix": key_display_prefix(full_key),
        "key_hash":   key_hash,
        "is_active":  1,
        "created_at": now,
        "metadata":   _json.dumps(req.metadata or {}),
    }
    if expires_at is not None:
        db_row["expires_at"] = expires_at

    try:
        get_db().create_api_key(db_row)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "db_error", "message": str(exc)},
        )

    _audit("key_created", key_id=key_id, key_name=req.name,
           expires_days=req.expires_days)
    resp_body: dict = {
        "id":         key_id,
        "name":       req.name,
        "key":        full_key,
        "key_prefix": key_display_prefix(full_key),
        "created_at": now,
        "expires_at": expires_at,
        "warning":    "Store this key securely — it will NOT be shown again.",
    }
    return resp_body


@router.delete(
    "/api/dashboard/admin/keys/{key_id}",
    dependencies=[Depends(_require_session_api)],
    include_in_schema=False,
)
async def dashboard_revoke_key(key_id: str, request: Request) -> dict:
    """Revoke an API key via the dashboard."""
    _check_rate_limit(request, _write_limiter)
    from autotune.api.auth import invalidate_key_cache_by_id
    from autotune.db.store import get_db

    db = get_db()
    row = db.get_api_key_by_id(key_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": "Key not found"},
        )
    if not row.get("is_active"):
        raise HTTPException(
            status_code=409,
            detail={"error": "already_revoked", "message": "Key is already revoked"},
        )

    ok = db.revoke_api_key(key_id, reason="revoked via dashboard")
    if not ok:
        raise HTTPException(
            status_code=500,
            detail={"error": "db_error", "message": "Revocation failed"},
        )

    invalidate_key_cache_by_id(key_id)
    _audit("key_revoked", key_id=key_id, key_name=row["name"])
    return {"revoked": key_id, "name": row["name"]}


@router.get("/api/dashboard/usage/{key_id}", dependencies=[Depends(_require_session_api)])
async def dashboard_key_usage_trend(key_id: str, request: Request, days: int = 30):
    """Return daily request and token counts for a specific API key."""
    _check_rate_limit(request, _read_limiter)
    import re
    if not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", key_id):
        raise HTTPException(status_code=422, detail="Invalid key_id")
    days = max(7, min(days, 90))
    from .metrics import get_key_usage_trend
    return {"key_id": key_id, "days": days, "data": get_key_usage_trend(key_id, days)}


@router.get("/api/dashboard/optimizations", dependencies=[Depends(_require_session_api)])
async def dashboard_optimizations(request: Request, limit: int = 200, model: Optional[str] = None):
    """Return recent optimization events and 24-hour summary stats."""
    _check_rate_limit(request, _read_limiter)
    import re as _re
    limit = max(10, min(limit, 500))
    if model and not _re.match(r"^[\w.:/ \-]{1,120}$", model):
        raise HTTPException(status_code=422, detail="Invalid model filter")
    from .metrics import get_optimization_events, get_optimization_summary
    return {
        "events": get_optimization_events(limit=limit, model_id=model or None),
        "summary": get_optimization_summary(),
    }


@router.get("/api/dashboard/perf-trends", dependencies=[Depends(_require_session_api)])
async def dashboard_perf_trends(request: Request):
    """Return RAM, KV-cache, and TPS time-series for the Performance tab expanded charts."""
    _check_rate_limit(request, _read_limiter)
    from .metrics import get_perf_trends
    return get_perf_trends()


# ---------------------------------------------------------------------------
# Settings endpoints — GET returns all settings; POST updates allowed keys
# ---------------------------------------------------------------------------

# Settings that the API is allowed to write (guards against arbitrary key injection)
_WRITABLE_SETTINGS = {
    "default_qos_profile",
    "ollama_url",
    "catalog_refresh_interval_h",
    "dashboard_session_timeout_h",
    "show_thinking_tokens",
    "retention_days",
}

# Validators for writable settings
def _validate_setting(key: str, value: str) -> str:
    """Validate and normalise a setting value. Raises ValueError on bad input."""
    value = value.strip()
    if key == "default_qos_profile":
        if value not in ("fast", "balanced", "quality"):
            raise ValueError("Must be fast, balanced, or quality")
    elif key in ("catalog_refresh_interval_h", "dashboard_session_timeout_h"):
        n = int(value)
        if not (1 <= n <= 720):
            raise ValueError("Must be between 1 and 720")
        value = str(n)
    elif key == "retention_days":
        n = int(value)
        if not (0 <= n <= 3650):
            raise ValueError("Must be between 0 and 3650")
        value = str(n)
    elif key == "show_thinking_tokens":
        value = "1" if value in ("1", "true", "yes") else "0"
    elif key == "ollama_url":
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("Must be empty or a valid http/https URL")
        if len(value) > 512:
            raise ValueError("URL too long")
    return value


@router.get("/api/dashboard/settings", dependencies=[Depends(_require_session_api)])
async def dashboard_settings_get(request: Request):
    """Return current settings from DB + file-based prefs + read-only env vars."""
    _check_rate_limit(request, _read_limiter)
    import os as _os
    from autotune.db.store import get_db
    from autotune.db.storage_prefs import is_storage_enabled
    from autotune.telemetry.consent import is_opted_in as _tel_opted_in

    db = get_db()
    db_settings = db.get_all_settings()
    db_stats    = db.stats()

    # Read-only env-var derived status fields
    admin_key  = _os.environ.get("AUTOTUNE_ADMIN_KEY", "").strip()
    api_key_on = _os.environ.get("AUTOTUNE_REQUIRE_API_KEY", "").strip() in ("1", "true", "yes")
    max_body   = int(_os.environ.get("AUTOTUNE_MAX_BODY_BYTES", str(10 * 1024 * 1024)))
    cors_extra = _os.environ.get("AUTOTUNE_CORS_ORIGINS", "").strip()
    supa_url   = _os.environ.get("AUTOTUNE_SUPABASE_URL", "").strip()

    try:
        local_storage_on = is_storage_enabled()
    except Exception:
        local_storage_on = True

    try:
        remote_tel_on = _tel_opted_in()
    except Exception:
        remote_tel_on = False

    return {
        "db_settings":       db_settings,
        "db_stats":          db_stats,
        "file_prefs": {
            "local_storage":    local_storage_on,
            "remote_telemetry": remote_tel_on,
        },
        "env_readonly": {
            "admin_key_set":      bool(admin_key),
            "admin_key_len":      len(admin_key),
            "api_key_enforcement": api_key_on,
            "max_body_mb":        round(max_body / (1024 * 1024), 1),
            "cors_extra_origins": cors_extra,
            "supabase_url_set":   bool(supa_url),
            "supabase_url_hint":  supa_url[:40] + "…" if len(supa_url) > 40 else supa_url,
        },
    }


class _SettingUpdate(BaseModel):
    key:   str
    value: str

    @field_validator("key")
    @classmethod
    def _key_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("key must not be empty")
        if len(v) > 128:
            raise ValueError("key too long")
        if not _SAFE_COL.match(v):  # reuse the SQL-safe column regex
            raise ValueError("Invalid setting key")
        return v


class _SettingsBatch(BaseModel):
    updates: list[_SettingUpdate]


@router.post("/api/dashboard/settings", dependencies=[Depends(_require_session_api)])
async def dashboard_settings_post(req: _SettingsBatch, request: Request):
    """Update one or more writable settings."""
    _check_rate_limit(request, _write_limiter)
    from autotune.db.store import get_db
    from autotune.db.storage_prefs import set_storage_enabled
    from autotune.telemetry.consent import set_consent as _set_tel_opted_in

    errors: dict[str, str] = {}
    applied: dict[str, str] = {}

    for upd in req.updates:
        key   = upd.key
        value = upd.value.strip()

        # Handle file-based prefs separately
        if key == "local_storage":
            try:
                set_storage_enabled(value in ("1", "true", "yes"))
                applied[key] = value
            except Exception as e:
                errors[key] = str(e)
            continue

        if key == "remote_telemetry":
            try:
                _set_tel_opted_in(value in ("1", "true", "yes"))
                applied[key] = "1" if value in ("1", "true", "yes") else "0"
            except Exception as e:
                errors[key] = str(e)
            continue

        # DB settings
        if key not in _WRITABLE_SETTINGS:
            errors[key] = f"Setting '{key}' is read-only or unknown"
            continue

        try:
            value = _validate_setting(key, value)
            get_db().set_setting(key, value)
            applied[key] = value
        except (ValueError, TypeError) as e:
            errors[key] = str(e)

    if errors:
        return {"applied": applied, "errors": errors, "partial": True}
    return {"applied": applied, "errors": {}, "partial": False}


@router.post("/api/dashboard/settings/cleanup", dependencies=[Depends(_require_session_api)])
async def dashboard_settings_cleanup(request: Request):
    """Prune old run_observations rows per the retention_days setting."""
    _check_rate_limit(request, _write_limiter)
    from autotune.db.store import get_db
    db = get_db()
    try:
        days = int(db.get_setting("retention_days") or "90")
    except (TypeError, ValueError):
        days = 90
    deleted = db.cleanup_old_data(retention_days=days)
    db.optimize()
    return {"deleted": deleted, "retention_days": days}


@router.get("/api/dashboard/model-recommendations", dependencies=[Depends(_require_session_api)])
async def dashboard_model_recommendations(request: Request):
    """Return model recommendations scored against local hardware."""
    _check_rate_limit(request, _read_limiter)
    import dataclasses as _dc
    try:
        from autotune.hardware.profiler import profile_hardware
        from autotune.config.generator import generate_recommendations

        hw = profile_hardware()

        # Apple Silicon: psutil available_gb is misleadingly low due to macOS
        # memory compression. Use 75% of total_gb as the planning budget so the
        # engine recommends models that genuinely fit, not just what is
        # "free" right now.
        if hw.gpu and hw.gpu.is_unified_memory:
            budget_gb = hw.memory.total_gb * 0.75
            hw = _dc.replace(hw, memory=_dc.replace(hw.memory, available_gb=budget_gb))

        recs = generate_recommendations(
            hw, modes=["fastest", "balanced", "best_quality"], top_n=5
        )

        def _mem_total(mem) -> float:
            return round(
                mem.weights_gb + mem.kv_cache_gb
                + (getattr(mem, "overhead_gb", 0) or 0), 2
            )

        modes_out: dict = {}
        for mode_name, rec in recs.items():
            sc = rec.primary
            m  = sc.candidate.model
            mem = sc.memory
            modes_out[mode_name] = {
                "model_id":      m.id,
                "model_name":    m.name,
                "family":        m.family,
                "quant":         sc.candidate.quant,
                "context_len":   sc.candidate.context_len,
                "parameters_b":  m.parameters_b,
                "bench_mmlu":    m.bench_mmlu,
                "bench_humaneval": m.bench_humaneval,
                "bench_gsm8k":   m.bench_gsm8k,
                "ollama_tag":    m.ollama_tag,
                "weights_gb":    round(mem.weights_gb, 2),
                "kv_cache_gb":   round(mem.kv_cache_gb, 2),
                "total_gb":      _mem_total(mem),
                "headroom_gb":   round(mem.headroom_gb, 2),
                "fits":          mem.fits,
                "score":         round(sc.composite, 3),
                "rationale":     sc.rationale,
                "alternatives": [
                    {
                        "model_id":   alt.candidate.model.id,
                        "model_name": alt.candidate.model.name,
                        "quant":      alt.candidate.quant,
                        "total_gb":   _mem_total(alt.memory),
                        "bench_mmlu": alt.candidate.model.bench_mmlu,
                        "score":      round(alt.composite, 3),
                    }
                    for alt in rec.alternatives[:3]
                ],
            }

        return {
            "hardware": {
                "total_gb":   round(hw.memory.total_gb, 1),
                "budget_gb":  round(hw.memory.available_gb, 1),
                "is_unified": bool(hw.gpu and hw.gpu.is_unified_memory),
                "chip":       hw.gpu.name if hw.gpu else None,
            },
            "modes": modes_out,
        }
    except ImportError as exc:
        return {"error": f"Recommendation engine unavailable: {exc}", "modes": {}}
    except Exception:
        import logging as _logging
        _logging.getLogger(__name__).exception("model-recommendations failed")
        return {"error": "Internal error computing recommendations.", "modes": {}}
