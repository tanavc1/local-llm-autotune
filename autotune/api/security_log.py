"""Structured security audit log for autotune.

All security-relevant events (auth failures, login attempts, key operations,
session lifecycle) are emitted to the ``autotune.security`` logger as
newline-delimited JSON records.

Operators can route this logger to a dedicated file, syslog, or a log
aggregation pipeline without touching the main application logger:

    import logging, json
    handler = logging.FileHandler("/var/log/autotune/security.log")
    logging.getLogger("autotune.security").addHandler(handler)

Events use WARNING for failures / suspicious activity, INFO for normal
security operations (key creation, successful login, logout).
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

_log = logging.getLogger("autotune.security")

# ---------------------------------------------------------------------------
# Real-IP extraction — shared by server.py and router.py
#
# When autotune sits behind nginx (Docker team profile) the direct TCP peer
# is the nginx container, not the real client.  nginx sets X-Real-IP to the
# actual outer IP.  We only trust that header when the direct peer is a
# known private/loopback address to prevent IP spoofing from the internet.
# ---------------------------------------------------------------------------

_TRUSTED_PREFIXES = (
    "127.", "::1",              # loopback
    "10.",                      # RFC-1918 class A
    "192.168.",                 # RFC-1918 class C
    "172.16.", "172.17.", "172.18.", "172.19.",   # RFC-1918 class B
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "fd", "fc",                 # IPv6 ULA (Docker)
)


def real_ip(request: "Request") -> str:
    """Return the true client IP, looking through trusted reverse-proxy headers."""
    direct = (request.client.host if request.client else "") or ""
    if any(direct.startswith(p) for p in _TRUSTED_PREFIXES):
        hdr = request.headers.get("X-Real-IP", "").strip()
        if hdr:
            return hdr
        fwd = request.headers.get("X-Forwarded-For", "").split(",")
        if fwd and fwd[0].strip():
            return fwd[0].strip()
    return direct or "unknown"

# Events that indicate a potential attack or policy violation
_WARN_EVENTS = frozenset({
    "login_failure",
    "login_rate_limited",
    "dashboard_rate_limited",
    "api_key_invalid",
    "api_key_revoked_used",
    "api_key_expired",
    "body_too_large",
    "cors_rejected",
})


def audit(event: str, **fields) -> None:
    """Emit one structured security audit record.

    Parameters
    ----------
    event   : machine-readable event name  (e.g. ``"login_failure"``)
    **fields: arbitrary context            (ip, path, key_id, …)

    The record is always a single JSON object on one line, making it
    trivial to grep, forward to a SIEM, or feed into ``jq``.
    It is also persisted to SQLite so the dashboard Security tab can
    display and query it without parsing log files.
    """
    ts = round(time.time(), 3)
    record: dict = {"event": event, "ts": ts, **fields}
    level = logging.WARNING if event in _WARN_EVENTS else logging.INFO
    _log.log(level, json.dumps(record, default=str))

    # Persist to SQLite — never let a DB error surface to the caller
    try:
        from autotune.db.store import get_db
        severity = "warning" if event in _WARN_EVENTS else "info"
        ip = str(fields["ip"]) if "ip" in fields else None
        path = str(fields["path"]) if "path" in fields else None
        details = {k: v for k, v in fields.items() if k not in ("ip", "path")}
        get_db().add_security_event(
            event, severity, ts, ip, path,
            json.dumps(details, default=str) if details else None,
        )
    except Exception:
        pass
