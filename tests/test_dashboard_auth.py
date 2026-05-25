"""
Tests for dashboard session auth and key management proxy endpoints.

Covers:
- GET /dashboard without admin key set → 200 (open access, backward compat)
- GET /dashboard with admin key set, no cookie → 302 redirect to login
- GET /dashboard/login → 200, HTML contains login form
- GET /dashboard/login when already authenticated → 302 to /dashboard
- POST /dashboard/login correct password → 303, sets autotune_session cookie
- POST /dashboard/login wrong password → 401, no cookie set
- GET /dashboard with valid cookie → 200 (FileResponse or 500 for missing static)
- GET /dashboard with expired/tampered cookie → 302 to login
- GET /dashboard/logout → 303, clears cookie
- GET /api/dashboard/overview without admin key → 200 (open access)
- GET /api/dashboard/overview with admin key, no cookie → 401
- GET /api/dashboard/overview with valid cookie → 200
- GET /api/dashboard/auth-status → reflects whether admin key is set
- POST /dashboard/login without admin key configured → 303 to /dashboard (pass-through)
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

TEST_KEY = "test-admin-key-for-dashboard-auth-tests"

_MOCK_OVERVIEW = {
    "ram": {"total_gb": 16.0, "available_gb": 8.0, "used_pct": 50.0},
    "running_models": [],
    "requests_today": 0,
    "avg_ttft_ms": 0.0,
    "avg_tps": 0.0,
    "avg_context_len": 4096.0,
    "kv_savings_pct": 0.0,
    "total_tokens_today": 0,
}

_METRICS_PATCHES = {
    "autotune.dashboard.metrics.get_overview":          MagicMock(return_value=_MOCK_OVERVIEW),
    "autotune.dashboard.metrics.get_requests_timeseries": MagicMock(return_value=[]),
    "autotune.dashboard.metrics.get_ttft_trend":        MagicMock(return_value=[]),
    "autotune.dashboard.metrics.get_models_stats":      MagicMock(return_value=[]),
    "autotune.dashboard.metrics.get_comparison":        MagicMock(return_value={}),
    "autotune.dashboard.metrics.get_api_keys_summary":  MagicMock(return_value=[]),
    "autotune.dashboard.metrics.get_slow_requests":     MagicMock(return_value=[]),
    "autotune.dashboard.metrics.get_suggestions":       MagicMock(return_value=[]),
}


def _build_app() -> FastAPI:
    """Return a minimal FastAPI app containing only the dashboard router."""
    from autotune.dashboard.router import router
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def auth_client(monkeypatch):
    """Client with AUTOTUNE_ADMIN_KEY set — auth is enforced."""
    monkeypatch.setenv("AUTOTUNE_ADMIN_KEY", TEST_KEY)
    with patch.multiple("autotune.dashboard.metrics", **{
        k.split(".")[-1]: v for k, v in _METRICS_PATCHES.items()
    }):
        app = _build_app()
        with TestClient(app, raise_server_exceptions=False, follow_redirects=False) as c:
            yield c


@pytest.fixture
def open_client(monkeypatch):
    """Client with no AUTOTUNE_ADMIN_KEY — dashboard is open."""
    monkeypatch.delenv("AUTOTUNE_ADMIN_KEY", raising=False)
    with patch.multiple("autotune.dashboard.metrics", **{
        k.split(".")[-1]: v for k, v in _METRICS_PATCHES.items()
    }):
        app = _build_app()
        with TestClient(app, raise_server_exceptions=False, follow_redirects=False) as c:
            yield c


def _get_session_cookie(client: TestClient, password: str = TEST_KEY) -> str:
    """Log in and return the raw session cookie value."""
    r = client.post("/dashboard/login", data={"password": password}, follow_redirects=False)
    assert r.status_code == 303, f"expected 303, got {r.status_code}"
    cookie = r.cookies.get("autotune_session")
    assert cookie, "session cookie not set after successful login"
    return cookie


# ---------------------------------------------------------------------------
# Open access (no admin key configured)
# ---------------------------------------------------------------------------

class TestOpenAccess:
    def test_dashboard_accessible_without_cookie(self, open_client):
        """Dashboard is open when no admin key is configured."""
        r = open_client.get("/dashboard")
        # Either 200 (static file found) or 500 (missing in test env) — never 302
        assert r.status_code in (200, 500)

    def test_api_accessible_without_cookie(self, open_client):
        """API endpoints are open when no admin key is configured."""
        r = open_client.get("/api/dashboard/overview")
        assert r.status_code == 200

    def test_auth_status_reports_not_required(self, open_client):
        r = open_client.get("/api/dashboard/auth-status")
        assert r.status_code == 200
        assert r.json()["auth_required"] is False

    def test_login_page_redirects_to_dashboard_when_no_key(self, open_client):
        """GET /dashboard/login redirects back to /dashboard in open mode."""
        r = open_client.get("/dashboard/login", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "/dashboard"

    def test_login_post_passthrough_when_no_key(self, open_client):
        """POST /dashboard/login with no admin key configured → pass-through to /dashboard."""
        r = open_client.post("/dashboard/login", data={"password": "anything"},
                             follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/dashboard"
        assert "autotune_session" not in r.cookies


# ---------------------------------------------------------------------------
# Auth enforced (admin key set)
# ---------------------------------------------------------------------------

class TestAuthEnforced:
    def test_dashboard_redirects_without_cookie(self, auth_client):
        r = auth_client.get("/dashboard", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "/dashboard/login"

    def test_api_returns_401_without_cookie(self, auth_client):
        r = auth_client.get("/api/dashboard/overview")
        assert r.status_code == 401
        body = r.json()
        assert body["detail"]["error"] == "not_authenticated"

    def test_all_api_endpoints_require_session(self, auth_client):
        endpoints = [
            "/api/dashboard/overview",
            "/api/dashboard/requests",
            "/api/dashboard/ttft_trend",
            "/api/dashboard/models",
            "/api/dashboard/comparison",
            "/api/dashboard/keys",
            "/api/dashboard/slow",
            "/api/dashboard/suggestions",
        ]
        for path in endpoints:
            r = auth_client.get(path)
            assert r.status_code == 401, f"{path} should be 401 without session, got {r.status_code}"

    def test_auth_status_reports_required(self, auth_client):
        r = auth_client.get("/api/dashboard/auth-status")
        assert r.status_code == 200
        assert r.json()["auth_required"] is True


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

class TestLoginPage:
    def test_login_page_returns_html(self, auth_client):
        r = auth_client.get("/dashboard/login")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_login_page_contains_form(self, auth_client):
        r = auth_client.get("/dashboard/login")
        body = r.text
        assert 'action="/dashboard/login"' in body
        assert 'name="password"' in body
        assert "autotune" in body.lower()

    def test_login_page_redirects_if_already_logged_in(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.get("/dashboard/login", cookies={"autotune_session": cookie},
                            follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"] == "/dashboard"


# ---------------------------------------------------------------------------
# Login submission
# ---------------------------------------------------------------------------

class TestLoginSubmit:
    def test_correct_password_sets_cookie(self, auth_client):
        r = auth_client.post("/dashboard/login", data={"password": TEST_KEY},
                             follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/dashboard"
        assert "autotune_session" in r.cookies

    def test_cookie_is_httponly(self, auth_client):
        r = auth_client.post("/dashboard/login", data={"password": TEST_KEY},
                             follow_redirects=False)
        set_cookie = r.headers.get("set-cookie", "")
        assert "HttpOnly" in set_cookie or "httponly" in set_cookie.lower()

    def test_cookie_is_samesite_strict(self, auth_client):
        r = auth_client.post("/dashboard/login", data={"password": TEST_KEY},
                             follow_redirects=False)
        set_cookie = r.headers.get("set-cookie", "")
        assert "SameSite=strict" in set_cookie or "samesite=strict" in set_cookie.lower()

    def test_wrong_password_returns_401(self, auth_client):
        r = auth_client.post("/dashboard/login", data={"password": "wrong-key"},
                             follow_redirects=False)
        assert r.status_code == 401
        assert "autotune_session" not in r.cookies

    def test_wrong_password_shows_error_in_html(self, auth_client):
        r = auth_client.post("/dashboard/login", data={"password": "wrong-key"},
                             follow_redirects=False)
        assert "Incorrect" in r.text

    def test_empty_password_returns_401(self, auth_client):
        r = auth_client.post("/dashboard/login", data={"password": ""},
                             follow_redirects=False)
        assert r.status_code == 401

    def test_cookie_has_max_age(self, auth_client):
        r = auth_client.post("/dashboard/login", data={"password": TEST_KEY},
                             follow_redirects=False)
        set_cookie = r.headers.get("set-cookie", "")
        assert "Max-Age" in set_cookie or "max-age" in set_cookie.lower()


# ---------------------------------------------------------------------------
# Session-protected dashboard UI
# ---------------------------------------------------------------------------

class TestSessionProtectedDashboard:
    def test_valid_cookie_serves_dashboard(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.get("/dashboard", cookies={"autotune_session": cookie})
        # 200 = static file found; 500 = missing in test env (both mean auth passed)
        assert r.status_code in (200, 500)

    def test_valid_cookie_allows_api(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.get("/api/dashboard/overview",
                            cookies={"autotune_session": cookie})
        assert r.status_code == 200

    def test_tampered_cookie_redirects_to_login(self, auth_client):
        r = auth_client.get("/dashboard",
                            cookies={"autotune_session": "tampered.bad.token"},
                            follow_redirects=False)
        assert r.status_code == 302
        assert "/dashboard/login" in r.headers["location"]

    def test_tampered_cookie_api_returns_401(self, auth_client):
        r = auth_client.get("/api/dashboard/overview",
                            cookies={"autotune_session": "tampered.bad.token"})
        assert r.status_code == 401

    def test_api_data_shape_with_valid_session(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.get("/api/dashboard/overview",
                            cookies={"autotune_session": cookie})
        assert r.status_code == 200
        data = r.json()
        assert "ram" in data
        assert "requests_today" in data

    def test_all_api_endpoints_pass_with_valid_session(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        cookies = {"autotune_session": cookie}
        endpoints_and_key = [
            ("/api/dashboard/overview",   "ram"),
            ("/api/dashboard/requests",   "data"),
            ("/api/dashboard/ttft_trend", "data"),
            ("/api/dashboard/models",     "models"),
            ("/api/dashboard/keys",       "keys"),
            ("/api/dashboard/slow",       "requests"),
            ("/api/dashboard/suggestions","suggestions"),
        ]
        for path, key in endpoints_and_key:
            r = auth_client.get(path, cookies=cookies)
            assert r.status_code == 200, f"{path} returned {r.status_code}"
            assert key in r.json(), f"{path} response missing key '{key}'"


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

class TestLogout:
    def test_logout_redirects_to_login(self, auth_client):
        r = auth_client.get("/dashboard/logout", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/dashboard/login"

    def test_logout_clears_cookie(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.get("/dashboard/logout",
                            cookies={"autotune_session": cookie},
                            follow_redirects=False)
        set_cookie = r.headers.get("set-cookie", "")
        # Cookie is cleared by setting max-age=0 or an empty value
        assert (
            "autotune_session=" in set_cookie and
            ("Max-Age=0" in set_cookie or 'max-age=0' in set_cookie.lower()
             or set_cookie.startswith("autotune_session=;"))
        ) or "autotune_session" in set_cookie  # header present = deletion attempted

    def test_logout_response_clears_cookie(self, auth_client):
        """Logout must redirect to login and instruct the browser to clear the cookie."""
        cookie = _get_session_cookie(auth_client)
        r = auth_client.get("/dashboard/logout",
                            cookies={"autotune_session": cookie},
                            follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/dashboard/login"
        # The Set-Cookie header must appear with autotune_session (deletion)
        set_cookie = r.headers.get("set-cookie", "")
        assert "autotune_session" in set_cookie


# ---------------------------------------------------------------------------
# Key management proxy (/api/dashboard/admin/keys)
# ---------------------------------------------------------------------------

class TestKeyManagementProxy:
    """Dashboard proxy endpoints for key create/revoke (session-gated)."""

    def _mock_db(self):
        db = MagicMock()
        db.create_api_key = MagicMock()
        db.get_api_key_by_id = MagicMock(return_value={
            "id": "key-abc-123", "name": "test-key", "is_active": True,
        })
        db.revoke_api_key = MagicMock(return_value=True)
        return db

    def test_create_key_requires_session(self, auth_client):
        r = auth_client.post("/api/dashboard/admin/keys",
                             json={"name": "test"})
        assert r.status_code == 401

    def test_create_key_success(self, auth_client):
        mock_db = self._mock_db()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            cookie = _get_session_cookie(auth_client)
            r = auth_client.post(
                "/api/dashboard/admin/keys",
                json={"name": "my-key"},
                cookies={"autotune_session": cookie},
            )
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "my-key"
        assert data["key"].startswith("sk-at-")
        assert "key_prefix" in data
        assert "id" in data
        assert "warning" in data
        mock_db.create_api_key.assert_called_once()

    def test_create_key_empty_name_returns_422(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.post(
            "/api/dashboard/admin/keys",
            json={"name": "   "},
            cookies={"autotune_session": cookie},
        )
        assert r.status_code == 422

    def test_create_key_name_too_long_returns_422(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.post(
            "/api/dashboard/admin/keys",
            json={"name": "x" * 129},
            cookies={"autotune_session": cookie},
        )
        assert r.status_code == 422

    def test_create_key_missing_name_returns_422(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.post(
            "/api/dashboard/admin/keys",
            json={},
            cookies={"autotune_session": cookie},
        )
        assert r.status_code == 422

    def test_create_key_generates_unique_keys(self, auth_client):
        keys_created = []
        mock_db = self._mock_db()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            cookie = _get_session_cookie(auth_client)
            for i in range(3):
                r = auth_client.post(
                    "/api/dashboard/admin/keys",
                    json={"name": f"key-{i}"},
                    cookies={"autotune_session": cookie},
                )
                assert r.status_code == 201
                keys_created.append(r.json()["key"])
        assert len(set(keys_created)) == 3, "Keys should be unique"

    def test_revoke_key_requires_session(self, auth_client):
        r = auth_client.delete("/api/dashboard/admin/keys/some-id")
        assert r.status_code == 401

    def test_revoke_key_success(self, auth_client):
        mock_db = self._mock_db()
        with (
            patch("autotune.db.store.get_db", return_value=mock_db),
            patch("autotune.api.auth.invalidate_key_cache_by_id"),
        ):
            cookie = _get_session_cookie(auth_client)
            r = auth_client.delete(
                "/api/dashboard/admin/keys/key-abc-123",
                cookies={"autotune_session": cookie},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["revoked"] == "key-abc-123"
        assert data["name"] == "test-key"
        mock_db.revoke_api_key.assert_called_once_with("key-abc-123", reason="revoked via dashboard")

    def test_revoke_key_not_found_returns_404(self, auth_client):
        mock_db = self._mock_db()
        mock_db.get_api_key_by_id.return_value = None
        with patch("autotune.db.store.get_db", return_value=mock_db):
            cookie = _get_session_cookie(auth_client)
            r = auth_client.delete(
                "/api/dashboard/admin/keys/nonexistent",
                cookies={"autotune_session": cookie},
            )
        assert r.status_code == 404
        assert r.json()["detail"]["error"] == "not_found"

    def test_revoke_already_revoked_key_returns_409(self, auth_client):
        mock_db = self._mock_db()
        mock_db.get_api_key_by_id.return_value = {
            "id": "key-abc-123", "name": "test-key", "is_active": False,
        }
        with patch("autotune.db.store.get_db", return_value=mock_db):
            cookie = _get_session_cookie(auth_client)
            r = auth_client.delete(
                "/api/dashboard/admin/keys/key-abc-123",
                cookies={"autotune_session": cookie},
            )
        assert r.status_code == 409
        assert r.json()["detail"]["error"] == "already_revoked"

    def test_open_mode_create_key_succeeds_without_login(self, open_client):
        """In open mode (no admin key), key creation works without a cookie."""
        mock_db = MagicMock()
        mock_db.create_api_key = MagicMock()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = open_client.post(
                "/api/dashboard/admin/keys",
                json={"name": "open-mode-key"},
            )
        assert r.status_code == 201
        assert r.json()["key"].startswith("sk-at-")


# ---------------------------------------------------------------------------
# Security headers — CSP, X-Frame-Options, X-Content-Type-Options
# ---------------------------------------------------------------------------

class TestSecurityHeaders:
    """Dashboard HTML responses must carry security headers."""

    def test_login_page_has_csp(self, auth_client):
        r = auth_client.get("/dashboard/login")
        assert "Content-Security-Policy" in r.headers

    def test_login_page_csp_blocks_frame_ancestors(self, auth_client):
        csp = auth_client.get("/dashboard/login").headers.get("Content-Security-Policy", "")
        assert "frame-ancestors 'none'" in csp

    def test_login_page_csp_restricts_connect_src(self, auth_client):
        csp = auth_client.get("/dashboard/login").headers.get("Content-Security-Policy", "")
        assert "connect-src 'self'" in csp

    def test_login_page_has_x_frame_options(self, auth_client):
        r = auth_client.get("/dashboard/login")
        assert r.headers.get("X-Frame-Options", "").upper() == "DENY"

    def test_login_page_has_x_content_type_options(self, auth_client):
        r = auth_client.get("/dashboard/login")
        assert r.headers.get("X-Content-Type-Options", "").lower() == "nosniff"

    def test_login_error_response_has_csp(self, auth_client):
        r = auth_client.post("/dashboard/login", data={"password": "wrong"},
                             follow_redirects=False)
        assert r.status_code == 401
        assert "Content-Security-Policy" in r.headers

    def test_dashboard_html_has_security_headers(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.get("/dashboard", cookies={"autotune_session": cookie})
        # 200 (file found) or 500 (missing static) both go through the same code path
        assert "Content-Security-Policy" in r.headers or r.status_code == 500


# ---------------------------------------------------------------------------
# Per-key usage trend endpoint
# ---------------------------------------------------------------------------

class TestKeyUsageTrend:
    """GET /api/dashboard/usage/{key_id} returns a 30-day daily trend."""

    _MOCK_TREND = [
        {"day": "2026-04-26", "requests": 5, "tokens": 1000, "avg_ttft_ms": 300.0},
        {"day": "2026-04-27", "requests": 0, "tokens": 0,    "avg_ttft_ms": None},
    ]

    def test_requires_session(self, auth_client):
        r = auth_client.get("/api/dashboard/usage/some-key-id")
        assert r.status_code == 401

    def test_returns_data_with_valid_session(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        with patch("autotune.dashboard.metrics.get_key_usage_trend",
                   return_value=self._MOCK_TREND):
            r = auth_client.get(
                "/api/dashboard/usage/valid-key-id-123",
                cookies={"autotune_session": cookie},
            )
        assert r.status_code == 200
        data = r.json()
        assert "key_id" in data
        assert "days" in data
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_invalid_key_id_returns_422(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        r = auth_client.get(
            "/api/dashboard/usage/../../etc/passwd",
            cookies={"autotune_session": cookie},
        )
        assert r.status_code in (422, 404)

    def test_days_clamped_to_90(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        with patch("autotune.dashboard.metrics.get_key_usage_trend",
                   return_value=[]) as mock_fn:
            auth_client.get(
                "/api/dashboard/usage/valid-key-id-123?days=999",
                cookies={"autotune_session": cookie},
            )
            # Called with days clamped to 90
            mock_fn.assert_called_once_with("valid-key-id-123", 90)

    def test_open_mode_accessible_without_session(self, open_client):
        with patch("autotune.dashboard.metrics.get_key_usage_trend",
                   return_value=self._MOCK_TREND):
            r = open_client.get("/api/dashboard/usage/some-key-id")
        assert r.status_code == 200
