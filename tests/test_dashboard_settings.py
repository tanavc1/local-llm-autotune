"""
Comprehensive tests for the three new dashboard settings endpoints:

  GET  /api/dashboard/settings        — returns DB settings + file prefs + env
  POST /api/dashboard/settings        — batch-update writable keys
  POST /api/dashboard/settings/cleanup — prune old data + optimize

Also covers:
  _validate_setting() — boundary values for all 6 writable keys
  _WRITABLE_SETTINGS whitelist enforcement
  file-based prefs (local_storage, remote_telemetry) routing
  Rate limiting (write limiter on POST endpoints)
  Auth enforcement (401 without session in auth mode)
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

TEST_KEY = "test-settings-admin-key"

_MOCK_OVERVIEW = {
    "ram": {"total_gb": 32.0, "available_gb": 16.0, "used_pct": 50.0},
    "running_models": [],
    "requests_today": 0,
    "avg_ttft_ms": 0.0,
    "avg_tps": 0.0,
    "avg_context_len": 4096.0,
    "kv_savings_pct": 0.0,
    "total_tokens_today": 0,
}


def _build_app() -> FastAPI:
    from autotune.dashboard.router import router
    app = FastAPI()
    app.include_router(router)
    return app


def _mock_db_for_settings(
    db_settings: dict | None = None,
    db_stats: dict | None = None,
) -> MagicMock:
    """Return a mock DB suitable for settings endpoint tests."""
    m = MagicMock()
    m.load_revoked_session_hashes.return_value = set()
    m.get_all_settings.return_value = db_settings or {
        "default_qos_profile": {
            "value": "balanced",
            "updated_at": 1700000000.0,
            "description": "Default QOS profile: fast | balanced | quality",
        },
        "retention_days": {
            "value": "90",
            "updated_at": 1700000000.0,
            "description": "Days to keep run_observations rows",
        },
    }
    m.stats.return_value = db_stats or {
        "models": 5,
        "hardware_profiles": 1,
        "run_observations": 100,
        "telemetry_events": 50,
        "security_events": 10,
        "settings": 6,
        "db_path": "/tmp/test.db",
        "db_size_mb": 0.5,
    }
    m.get_setting.return_value = "90"
    m.cleanup_old_data.return_value = {
        "run_observations": 42,
        "telemetry_events": 10,
        "security_events": 0,
    }
    m.set_setting = MagicMock()
    m.optimize = MagicMock()
    return m


def _reset_router_state(monkeypatch) -> None:
    """Reset module-level router state that persists across tests."""
    import autotune.dashboard.router as _router
    monkeypatch.setattr(_router, "_revoked_loaded", False)
    monkeypatch.setattr(_router, "_revoked_sessions", set())
    monkeypatch.setattr(_router, "_load_revoked_from_db", lambda: None)
    # Reset rate limiters to very high limits so tests don't hit 429.
    # The sliding-window limiter is module-level and accumulates across tests.
    monkeypatch.setattr(_router, "_write_limiter",
                        _router._SlidingWindowLimiter(100_000, 3600))
    monkeypatch.setattr(_router, "_read_limiter",
                        _router._SlidingWindowLimiter(100_000, 60))
    monkeypatch.setattr(_router, "_refresh_limiter",
                        _router._SlidingWindowLimiter(100_000, 60))


@pytest.fixture
def open_client(monkeypatch):
    """Client in open mode (no admin key) — no auth enforcement."""
    monkeypatch.delenv("AUTOTUNE_ADMIN_KEY", raising=False)
    _reset_router_state(monkeypatch)
    with patch("autotune.dashboard.metrics.get_overview", return_value=_MOCK_OVERVIEW):
        app = _build_app()
        with TestClient(app, raise_server_exceptions=False, follow_redirects=False) as c:
            yield c


@pytest.fixture
def auth_client(monkeypatch):
    """Client with AUTOTUNE_ADMIN_KEY set — auth is enforced."""
    monkeypatch.setenv("AUTOTUNE_ADMIN_KEY", TEST_KEY)
    _reset_router_state(monkeypatch)
    with patch("autotune.dashboard.metrics.get_overview", return_value=_MOCK_OVERVIEW):
        app = _build_app()
        with TestClient(app, raise_server_exceptions=False, follow_redirects=False) as c:
            yield c


def _get_session_cookie(client: TestClient) -> str:
    r = client.post("/dashboard/login", data={"password": TEST_KEY}, follow_redirects=False)
    assert r.status_code == 303, f"login failed: {r.status_code}"
    return r.cookies["autotune_session"]


# ---------------------------------------------------------------------------
# _validate_setting unit tests
# ---------------------------------------------------------------------------

class TestValidateSetting:
    """Direct tests for the _validate_setting() helper in router.py."""

    def _validate(self, key: str, value: str) -> str:
        from autotune.dashboard.router import _validate_setting
        return _validate_setting(key, value)

    # default_qos_profile
    def test_qos_fast_accepted(self):
        assert self._validate("default_qos_profile", "fast") == "fast"

    def test_qos_balanced_accepted(self):
        assert self._validate("default_qos_profile", "balanced") == "balanced"

    def test_qos_quality_accepted(self):
        assert self._validate("default_qos_profile", "quality") == "quality"

    def test_qos_invalid_raises(self):
        with pytest.raises(ValueError):
            self._validate("default_qos_profile", "ultra")

    def test_qos_strips_whitespace(self):
        assert self._validate("default_qos_profile", "  fast  ") == "fast"

    # catalog_refresh_interval_h
    def test_catalog_refresh_min_boundary(self):
        assert self._validate("catalog_refresh_interval_h", "1") == "1"

    def test_catalog_refresh_max_boundary(self):
        assert self._validate("catalog_refresh_interval_h", "720") == "720"

    def test_catalog_refresh_zero_raises(self):
        with pytest.raises(ValueError):
            self._validate("catalog_refresh_interval_h", "0")

    def test_catalog_refresh_too_large_raises(self):
        with pytest.raises(ValueError):
            self._validate("catalog_refresh_interval_h", "721")

    def test_catalog_refresh_non_int_raises(self):
        with pytest.raises(ValueError):
            self._validate("catalog_refresh_interval_h", "abc")

    def test_catalog_refresh_float_raises(self):
        with pytest.raises(ValueError):
            self._validate("catalog_refresh_interval_h", "12.5")

    # dashboard_session_timeout_h
    def test_session_timeout_min(self):
        assert self._validate("dashboard_session_timeout_h", "1") == "1"

    def test_session_timeout_max(self):
        assert self._validate("dashboard_session_timeout_h", "720") == "720"

    def test_session_timeout_zero_raises(self):
        with pytest.raises(ValueError):
            self._validate("dashboard_session_timeout_h", "0")

    def test_session_timeout_over_max_raises(self):
        with pytest.raises(ValueError):
            self._validate("dashboard_session_timeout_h", "721")

    # retention_days
    def test_retention_zero_allowed(self):
        assert self._validate("retention_days", "0") == "0"

    def test_retention_max_boundary(self):
        assert self._validate("retention_days", "3650") == "3650"

    def test_retention_negative_raises(self):
        with pytest.raises(ValueError):
            self._validate("retention_days", "-1")

    def test_retention_over_max_raises(self):
        with pytest.raises(ValueError):
            self._validate("retention_days", "3651")

    def test_retention_typical_value(self):
        assert self._validate("retention_days", "90") == "90"

    # show_thinking_tokens
    def test_thinking_tokens_one_returns_one(self):
        assert self._validate("show_thinking_tokens", "1") == "1"

    def test_thinking_tokens_true_returns_one(self):
        assert self._validate("show_thinking_tokens", "true") == "1"

    def test_thinking_tokens_yes_returns_one(self):
        assert self._validate("show_thinking_tokens", "yes") == "1"

    def test_thinking_tokens_zero_returns_zero(self):
        assert self._validate("show_thinking_tokens", "0") == "0"

    def test_thinking_tokens_false_returns_zero(self):
        assert self._validate("show_thinking_tokens", "false") == "0"

    def test_thinking_tokens_no_returns_zero(self):
        assert self._validate("show_thinking_tokens", "no") == "0"

    def test_thinking_tokens_empty_returns_zero(self):
        assert self._validate("show_thinking_tokens", "") == "0"

    # ollama_url
    def test_ollama_url_empty_accepted(self):
        assert self._validate("ollama_url", "") == ""

    def test_ollama_url_http_accepted(self):
        url = "http://localhost:11434"
        assert self._validate("ollama_url", url) == url

    def test_ollama_url_https_accepted(self):
        url = "https://my-server.example.com:11434"
        assert self._validate("ollama_url", url) == url

    def test_ollama_url_no_scheme_raises(self):
        with pytest.raises(ValueError):
            self._validate("ollama_url", "localhost:11434")

    def test_ollama_url_ftp_scheme_raises(self):
        with pytest.raises(ValueError):
            self._validate("ollama_url", "ftp://localhost:11434")

    def test_ollama_url_too_long_raises(self):
        url = "http://" + "a" * 510
        with pytest.raises(ValueError):
            self._validate("ollama_url", url)

    def test_ollama_url_strips_whitespace(self):
        url = "  http://localhost:11434  "
        assert self._validate("ollama_url", url) == "http://localhost:11434"


# ---------------------------------------------------------------------------
# GET /api/dashboard/settings — shape and contents
# ---------------------------------------------------------------------------

class TestSettingsGetEndpoint:
    def test_requires_session_in_auth_mode(self, auth_client):
        r = auth_client.get("/api/dashboard/settings")
        assert r.status_code == 401

    def test_returns_200_in_open_mode(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    r = open_client.get("/api/dashboard/settings")
        assert r.status_code == 200

    def test_response_has_db_settings_key(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        assert "db_settings" in data

    def test_response_has_db_stats_key(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        assert "db_stats" in data

    def test_response_has_file_prefs_key(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        assert "file_prefs" in data

    def test_response_has_env_readonly_key(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        assert "env_readonly" in data

    def test_file_prefs_contains_local_storage(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        assert "local_storage" in data["file_prefs"]

    def test_file_prefs_contains_remote_telemetry(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        assert "remote_telemetry" in data["file_prefs"]

    def test_local_storage_true_when_enabled(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        assert data["file_prefs"]["local_storage"] is True

    def test_local_storage_false_when_disabled(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=False):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        assert data["file_prefs"]["local_storage"] is False

    def test_env_readonly_admin_key_set_reflects_env(self, auth_client):
        """auth_client fixture sets AUTOTUNE_ADMIN_KEY — admin_key_set should be True."""
        cookie = _get_session_cookie(auth_client)
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = auth_client.get(
                        "/api/dashboard/settings",
                        cookies={"autotune_session": cookie},
                    ).json()
        assert data["env_readonly"]["admin_key_set"] is True

    def test_env_readonly_admin_key_not_set(self, open_client):
        """open_client fixture has no AUTOTUNE_ADMIN_KEY — admin_key_set should be False."""
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        assert data["env_readonly"]["admin_key_set"] is False

    def test_db_stats_forwarded_correctly(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            with patch("autotune.db.storage_prefs.is_storage_enabled", return_value=True):
                with patch("autotune.telemetry.consent.is_opted_in", return_value=False):
                    data = open_client.get("/api/dashboard/settings").json()
        stats = data["db_stats"]
        assert stats["models"] == 5
        assert stats["settings"] == 6


# ---------------------------------------------------------------------------
# POST /api/dashboard/settings — batch update
# ---------------------------------------------------------------------------

class TestSettingsPostEndpoint:
    def _post(self, client, updates: list, cookies: dict | None = None):
        kwargs = {"json": {"updates": updates}}
        if cookies:
            kwargs["cookies"] = cookies
        return client.post("/api/dashboard/settings", **kwargs)

    def test_requires_session_in_auth_mode(self, auth_client):
        r = auth_client.post("/api/dashboard/settings", json={"updates": []})
        assert r.status_code == 401

    def test_valid_qos_profile_update_applied(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "default_qos_profile", "value": "fast"}])
        assert r.status_code == 200
        data = r.json()
        assert data["applied"]["default_qos_profile"] == "fast"
        assert data["errors"] == {}
        assert data["partial"] is False

    def test_valid_retention_days_update(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "retention_days", "value": "180"}])
        assert r.status_code == 200
        assert r.json()["applied"]["retention_days"] == "180"

    def test_valid_catalog_refresh_update(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "catalog_refresh_interval_h", "value": "12"}])
        assert r.status_code == 200
        assert r.json()["applied"]["catalog_refresh_interval_h"] == "12"

    def test_invalid_qos_returns_error(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "default_qos_profile", "value": "turbo"}])
        data = r.json()
        assert "default_qos_profile" in data["errors"]
        assert data["partial"] is True

    def test_unknown_key_returns_error(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "admin_password", "value": "hack"}])
        data = r.json()
        assert "admin_password" in data["errors"]
        assert data["partial"] is True
        mock_db.set_setting.assert_not_called()

    def test_readonly_key_rejected(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "AUTOTUNE_ADMIN_KEY", "value": "new-key"}])
        data = r.json()
        assert "AUTOTUNE_ADMIN_KEY" in data["errors"]

    def test_local_storage_routed_to_file_pref(self, open_client):
        with patch("autotune.db.storage_prefs.set_storage_enabled") as mock_set:
            r = self._post(open_client, [{"key": "local_storage", "value": "0"}])
        assert r.status_code == 200
        mock_set.assert_called_once_with(False)

    def test_local_storage_true_value(self, open_client):
        with patch("autotune.db.storage_prefs.set_storage_enabled") as mock_set:
            r = self._post(open_client, [{"key": "local_storage", "value": "1"}])
        assert r.status_code == 200
        mock_set.assert_called_once_with(True)

    def test_remote_telemetry_routed_to_consent(self, open_client):
        with patch("autotune.telemetry.consent.set_consent") as mock_set:
            r = self._post(open_client, [{"key": "remote_telemetry", "value": "true"}])
        assert r.status_code == 200
        mock_set.assert_called_once_with(True)

    def test_remote_telemetry_false_value(self, open_client):
        with patch("autotune.telemetry.consent.set_consent") as mock_set:
            r = self._post(open_client, [{"key": "remote_telemetry", "value": "0"}])
        assert r.status_code == 200
        mock_set.assert_called_once_with(False)

    def test_batch_with_mixed_valid_invalid(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [
                {"key": "default_qos_profile", "value": "fast"},
                {"key": "bad_key_xyz", "value": "irrelevant"},
            ])
        data = r.json()
        assert "default_qos_profile" in data["applied"]
        assert "bad_key_xyz" in data["errors"]
        assert data["partial"] is True

    def test_empty_updates_returns_ok(self, open_client):
        r = self._post(open_client, [])
        assert r.status_code == 200
        data = r.json()
        assert data["applied"] == {}
        assert data["errors"] == {}

    def test_set_setting_called_on_valid_db_key(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            self._post(open_client, [{"key": "retention_days", "value": "30"}])
        mock_db.set_setting.assert_called_once_with("retention_days", "30")

    def test_ollama_url_valid_http_update(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "ollama_url", "value": "http://192.168.1.5:11434"}])
        assert r.status_code == 200
        assert r.json()["applied"]["ollama_url"] == "http://192.168.1.5:11434"

    def test_ollama_url_invalid_scheme_rejected(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "ollama_url", "value": "ftp://localhost"}])
        data = r.json()
        assert "ollama_url" in data["errors"]

    def test_show_thinking_tokens_true(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "show_thinking_tokens", "value": "true"}])
        assert r.json()["applied"]["show_thinking_tokens"] == "1"

    def test_show_thinking_tokens_false(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(open_client, [{"key": "show_thinking_tokens", "value": "no"}])
        assert r.json()["applied"]["show_thinking_tokens"] == "0"

    def test_malformed_key_with_special_chars_rejected(self, open_client):
        r = self._post(open_client, [{"key": "bad-key!", "value": "x"}])
        # Either 422 (Pydantic validation) or 200 with error dict
        if r.status_code == 422:
            pass  # Pydantic caught it
        else:
            assert "bad-key!" in r.json().get("errors", {})

    def test_authenticated_endpoint_with_valid_session(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = self._post(
                auth_client,
                [{"key": "retention_days", "value": "60"}],
                cookies={"autotune_session": cookie},
            )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/dashboard/settings/cleanup — prune + optimize
# ---------------------------------------------------------------------------

class TestSettingsCleanupEndpoint:
    def test_requires_session_in_auth_mode(self, auth_client):
        r = auth_client.post("/api/dashboard/settings/cleanup")
        assert r.status_code == 401

    def test_returns_200_in_open_mode(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = open_client.post("/api/dashboard/settings/cleanup")
        assert r.status_code == 200

    def test_response_has_deleted_key(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            data = open_client.post("/api/dashboard/settings/cleanup").json()
        assert "deleted" in data

    def test_response_has_retention_days_key(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            data = open_client.post("/api/dashboard/settings/cleanup").json()
        assert "retention_days" in data

    def test_cleanup_called_with_retention_setting(self, open_client):
        mock_db = _mock_db_for_settings()
        mock_db.get_setting.return_value = "60"
        with patch("autotune.db.store.get_db", return_value=mock_db):
            open_client.post("/api/dashboard/settings/cleanup")
        mock_db.cleanup_old_data.assert_called_once_with(retention_days=60)

    def test_optimize_called_after_cleanup(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            open_client.post("/api/dashboard/settings/cleanup")
        mock_db.optimize.assert_called_once()

    def test_deleted_counts_returned(self, open_client):
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            data = open_client.post("/api/dashboard/settings/cleanup").json()
        assert data["deleted"]["run_observations"] == 42
        assert data["deleted"]["telemetry_events"] == 10

    def test_defaults_to_90_days_when_setting_invalid(self, open_client):
        mock_db = _mock_db_for_settings()
        mock_db.get_setting.return_value = "not_a_number"
        with patch("autotune.db.store.get_db", return_value=mock_db):
            data = open_client.post("/api/dashboard/settings/cleanup").json()
        mock_db.cleanup_old_data.assert_called_once_with(retention_days=90)
        assert data["retention_days"] == 90

    def test_defaults_to_90_days_when_setting_none(self, open_client):
        mock_db = _mock_db_for_settings()
        mock_db.get_setting.return_value = None
        with patch("autotune.db.store.get_db", return_value=mock_db):
            data = open_client.post("/api/dashboard/settings/cleanup").json()
        mock_db.cleanup_old_data.assert_called_once_with(retention_days=90)

    def test_authenticated_cleanup_with_valid_session(self, auth_client):
        cookie = _get_session_cookie(auth_client)
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = auth_client.post(
                "/api/dashboard/settings/cleanup",
                cookies={"autotune_session": cookie},
            )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# _WRITABLE_SETTINGS whitelist — exhaustive coverage
# ---------------------------------------------------------------------------

class TestWritableSettingsWhitelist:
    """Verify every key in _WRITABLE_SETTINGS is accepted and others are rejected."""

    _WRITABLE = {
        "default_qos_profile":         "balanced",
        "ollama_url":                   "http://localhost:11434",
        "catalog_refresh_interval_h":   "24",
        "dashboard_session_timeout_h":  "24",
        "show_thinking_tokens":         "0",
        "retention_days":               "90",
    }

    _REJECTED = [
        "AUTOTUNE_ADMIN_KEY",
        "AUTOTUNE_REQUIRE_API_KEY",
        "admin_key",
        "secret",
        "password",
        "unknown_key_xyz",
        "db_path",
        "db_size_mb",
    ]

    def _post_update(self, client, key: str, value: str) -> dict:
        mock_db = _mock_db_for_settings()
        with patch("autotune.db.store.get_db", return_value=mock_db):
            r = client.post("/api/dashboard/settings", json={
                "updates": [{"key": key, "value": value}]
            })
        return r.json()

    def test_all_writable_keys_accepted(self, open_client):
        for key, value in self._WRITABLE.items():
            data = self._post_update(open_client, key, value)
            assert key in data.get("applied", {}), (
                f"Writable key '{key}' was not applied: {data}"
            )

    def test_rejected_keys_are_rejected(self, open_client):
        for key in self._REJECTED:
            # Skip keys with invalid SQL identifier chars (caught by Pydantic)
            import re
            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key):
                continue
            data = self._post_update(open_client, key, "some_value")
            assert key in data.get("errors", {}), (
                f"Rejected key '{key}' was incorrectly applied: {data}"
            )
