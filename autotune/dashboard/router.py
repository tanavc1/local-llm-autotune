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

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, field_validator

router = APIRouter(tags=["dashboard"])

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
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
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
    form = await request.form()
    password = str(form.get("password", ""))
    admin_key = _admin_key()

    if not admin_key:
        # No admin key configured — pass straight through (open mode).
        return RedirectResponse("/dashboard", status_code=303)

    if not secrets.compare_digest(password.encode("utf-8"), admin_key.encode("utf-8")):
        return _html_response(_login_html(error="Incorrect admin key. Please try again."), status_code=401)

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
async def logout():
    """Clear the session cookie and redirect to the login page."""
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
async def dashboard_overview():
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


@router.get("/api/dashboard/slow", dependencies=[Depends(_require_session_api)])
async def dashboard_slow():
    from .metrics import get_slow_requests
    return {"requests": get_slow_requests()}


@router.get("/api/dashboard/suggestions", dependencies=[Depends(_require_session_api)])
async def dashboard_suggestions():
    from .metrics import get_suggestions
    return {"suggestions": get_suggestions()}


@router.get("/api/dashboard/security", dependencies=[Depends(_require_session_api)])
async def dashboard_security():
    from .metrics import get_gateway_security
    return {"checks": get_gateway_security()}


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


class _PullBody(BaseModel):
    model: str

    @field_validator("model")
    @classmethod
    def _model_valid(cls, v: str) -> str:
        import re
        v = v.strip()
        if not v:
            raise ValueError("model must not be empty")
        if len(v) > 200:
            raise ValueError("model name too long")
        if not re.match(r'^[a-zA-Z0-9][\w.:/\-]*$', v):
            raise ValueError("Invalid model name format")
        return v


@router.get("/api/dashboard/pull-check", dependencies=[Depends(_require_session_api)])
async def dashboard_pull_check(model: str, source: str = "ollama") -> dict:
    """Return a feasibility verdict for pulling *model* without downloading it."""
    from autotune.api.model_guard import check_feasibility
    if not model or len(model) > 200:
        raise HTTPException(status_code=422, detail="Invalid model name")
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
        if not re.match(r'^[a-zA-Z0-9][\w.\-]*/[\w.\-]+$', v):
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

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be empty")
        if len(v) > 128:
            raise ValueError("name must be ≤ 128 characters")
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
async def dashboard_catalog_refresh():
    """Force a synchronous catalog refresh and return the updated catalog."""
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
async def dashboard_create_key(req: _CreateKeyBody) -> dict:
    """Create a new API key via the dashboard (returns plaintext once)."""
    from autotune.api.auth import generate_api_key, key_display_prefix
    from autotune.db.store import get_db

    full_key, key_hash = generate_api_key()
    key_id = str(_uuid.uuid4())
    now = _time.time()

    db_row: dict = {
        "id":         key_id,
        "name":       req.name,
        "key_prefix": key_display_prefix(full_key),
        "key_hash":   key_hash,
        "is_active":  1,
        "created_at": now,
        "metadata":   _json.dumps(req.metadata or {}),
    }

    try:
        get_db().create_api_key(db_row)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "db_error", "message": str(exc)},
        )

    return {
        "id":         key_id,
        "name":       req.name,
        "key":        full_key,
        "key_prefix": key_display_prefix(full_key),
        "created_at": now,
        "warning":    "Store this key securely — it will NOT be shown again.",
    }


@router.delete(
    "/api/dashboard/admin/keys/{key_id}",
    dependencies=[Depends(_require_session_api)],
    include_in_schema=False,
)
async def dashboard_revoke_key(key_id: str) -> dict:
    """Revoke an API key via the dashboard."""
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
    return {"revoked": key_id, "name": row["name"]}


@router.get("/api/dashboard/usage/{key_id}", dependencies=[Depends(_require_session_api)])
async def dashboard_key_usage_trend(key_id: str, days: int = 30):
    """Return daily request and token counts for a specific API key."""
    import re
    if not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", key_id):
        raise HTTPException(status_code=422, detail="Invalid key_id")
    days = max(7, min(days, 90))
    from .metrics import get_key_usage_trend
    return {"key_id": key_id, "days": days, "data": get_key_usage_trend(key_id, days)}
