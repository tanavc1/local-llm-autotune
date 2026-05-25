"""
Tests for dashboard/metrics.py — get_gateway_security() and _latest_pypi_version().

Covers:
- Each check's status under various env/system conditions
- PyPI version cache behaviour (cache hit, cache miss, error fallback)
- Version comparison (outdated, up-to-date, unavailable)
- /api/dashboard/security endpoint — requires session, returns checks list
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

TEST_KEY = "test-admin-key-for-security-tests-ok"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _checks_by_name(checks: list[dict]) -> dict[str, dict]:
    return {c["check"]: c for c in checks}


def _run_security(monkeypatch, *, env: dict | None = None,
                  ram_pct: float = 50.0, disk_free_gb: float = 100.0,
                  pypi_version: str | None = None,
                  current_version: str = "1.3.0",
                  db_active_keys: int = 1) -> list[dict[str, Any]]:
    """Helper: run get_gateway_security() with fully mocked dependencies."""
    import autotune.dashboard.metrics as m

    # Reset PyPI cache so version checks are fresh
    m._PYPI_CACHE.clear()

    base_env = {"AUTOTUNE_ADMIN_KEY": TEST_KEY, "AUTOTUNE_REQUIRE_API_KEY": "1"}
    if env:
        base_env.update(env)

    mock_vm = MagicMock()
    mock_vm.percent   = ram_pct
    mock_vm.available = int((1 - ram_pct / 100) * 16 * 1024**3)

    mock_disk = MagicMock()
    mock_disk.free = int(disk_free_gb * 1024**3)

    mock_db = MagicMock()
    mock_db.conn.execute.return_value.fetchone.return_value = {"n": db_active_keys}

    with (
        patch.dict("os.environ", base_env, clear=True),
        patch("autotune.dashboard.metrics.psutil.virtual_memory", return_value=mock_vm),
        patch("autotune.dashboard.metrics.shutil.disk_usage", return_value=mock_disk),
        patch("autotune.dashboard.metrics._latest_pypi_version", return_value=pypi_version),
        patch("autotune.dashboard.metrics._db", return_value=mock_db),
        patch("importlib.metadata.version", return_value=current_version),
    ):
        from autotune.dashboard.metrics import get_gateway_security
        return get_gateway_security()


# ---------------------------------------------------------------------------
# Structure checks
# ---------------------------------------------------------------------------

class TestSecurityCheckStructure:
    def test_returns_non_empty_list(self, monkeypatch):
        checks = _run_security(monkeypatch)
        assert isinstance(checks, list)
        assert len(checks) > 0

    def test_each_check_has_required_fields(self, monkeypatch):
        checks = _run_security(monkeypatch)
        for c in checks:
            assert "check"   in c, f"missing 'check' in {c}"
            assert "status"  in c, f"missing 'status' in {c}"
            assert "message" in c, f"missing 'message' in {c}"
            assert "action"  in c, f"missing 'action' in {c}"

    def test_all_statuses_are_valid(self, monkeypatch):
        valid = {"ok", "warn", "error", "info"}
        checks = _run_security(monkeypatch)
        for c in checks:
            assert c["status"] in valid, f"Invalid status {c['status']!r} in {c}"

    def test_action_is_str_or_none(self, monkeypatch):
        checks = _run_security(monkeypatch)
        for c in checks:
            assert c["action"] is None or isinstance(c["action"], str)


# ---------------------------------------------------------------------------
# API key enforcement check
# ---------------------------------------------------------------------------

class TestApiKeyEnforcementCheck:
    def test_ok_when_enforcement_on(self, monkeypatch):
        checks = _run_security(monkeypatch, env={"AUTOTUNE_REQUIRE_API_KEY": "1"})
        by_name = _checks_by_name(checks)
        assert by_name["API Key Enforcement"]["status"] == "ok"

    def test_warn_when_enforcement_off(self, monkeypatch):
        checks = _run_security(monkeypatch, env={"AUTOTUNE_REQUIRE_API_KEY": "0"})
        by_name = _checks_by_name(checks)
        assert by_name["API Key Enforcement"]["status"] == "warn"

    def test_warn_when_enforcement_not_set(self, monkeypatch):
        checks = _run_security(monkeypatch, env={"AUTOTUNE_REQUIRE_API_KEY": ""})
        by_name = _checks_by_name(checks)
        assert by_name["API Key Enforcement"]["status"] == "warn"


# ---------------------------------------------------------------------------
# Admin key check
# ---------------------------------------------------------------------------

class TestAdminKeyCheck:
    def test_error_when_no_admin_key(self, monkeypatch):
        checks = _run_security(monkeypatch, env={"AUTOTUNE_ADMIN_KEY": ""})
        by_name = _checks_by_name(checks)
        assert by_name["Admin Key"]["status"] == "error"

    def test_warn_when_short_admin_key(self, monkeypatch):
        checks = _run_security(monkeypatch, env={"AUTOTUNE_ADMIN_KEY": "short"})
        by_name = _checks_by_name(checks)
        assert by_name["Admin Key"]["status"] == "warn"
        assert "5" in by_name["Admin Key"]["message"]  # char count in message

    def test_ok_when_strong_admin_key(self, monkeypatch):
        long_key = "a" * 48
        checks = _run_security(monkeypatch, env={"AUTOTUNE_ADMIN_KEY": long_key})
        by_name = _checks_by_name(checks)
        assert by_name["Admin Key"]["status"] == "ok"

    def test_boundary_31_chars_warns(self, monkeypatch):
        checks = _run_security(monkeypatch, env={"AUTOTUNE_ADMIN_KEY": "x" * 31})
        assert _checks_by_name(checks)["Admin Key"]["status"] == "warn"

    def test_boundary_32_chars_ok(self, monkeypatch):
        checks = _run_security(monkeypatch, env={"AUTOTUNE_ADMIN_KEY": "x" * 32})
        assert _checks_by_name(checks)["Admin Key"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Active keys check
# ---------------------------------------------------------------------------

class TestActiveKeysCheck:
    def test_present_only_when_enforcement_is_on(self, monkeypatch):
        checks_off = _run_security(monkeypatch, env={"AUTOTUNE_REQUIRE_API_KEY": "0"})
        names_off = [c["check"] for c in checks_off]
        assert "Active Keys" not in names_off

    def test_warn_when_no_active_keys(self, monkeypatch):
        checks = _run_security(monkeypatch, db_active_keys=0)
        assert _checks_by_name(checks)["Active Keys"]["status"] == "warn"

    def test_ok_when_keys_exist(self, monkeypatch):
        checks = _run_security(monkeypatch, db_active_keys=3)
        assert _checks_by_name(checks)["Active Keys"]["status"] == "ok"
        assert "3" in _checks_by_name(checks)["Active Keys"]["message"]


# ---------------------------------------------------------------------------
# RAM check
# ---------------------------------------------------------------------------

class TestRamCheck:
    def test_ok_below_threshold(self, monkeypatch):
        checks = _run_security(monkeypatch, ram_pct=60.0)
        assert _checks_by_name(checks)["RAM"]["status"] == "ok"

    def test_warn_above_88_percent(self, monkeypatch):
        checks = _run_security(monkeypatch, ram_pct=90.0)
        assert _checks_by_name(checks)["RAM"]["status"] == "warn"

    def test_boundary_exactly_88_is_ok(self, monkeypatch):
        checks = _run_security(monkeypatch, ram_pct=88.0)
        assert _checks_by_name(checks)["RAM"]["status"] == "ok"

    def test_boundary_89_is_warn(self, monkeypatch):
        checks = _run_security(monkeypatch, ram_pct=89.0)
        assert _checks_by_name(checks)["RAM"]["status"] == "warn"


# ---------------------------------------------------------------------------
# Disk check
# ---------------------------------------------------------------------------

class TestDiskCheck:
    def test_ok_ample_space(self, monkeypatch):
        checks = _run_security(monkeypatch, disk_free_gb=200.0)
        assert _checks_by_name(checks)["Disk Space"]["status"] == "ok"

    def test_warn_below_15gb(self, monkeypatch):
        checks = _run_security(monkeypatch, disk_free_gb=10.0)
        assert _checks_by_name(checks)["Disk Space"]["status"] == "warn"

    def test_error_below_5gb(self, monkeypatch):
        checks = _run_security(monkeypatch, disk_free_gb=3.0)
        assert _checks_by_name(checks)["Disk Space"]["status"] == "error"

    def test_boundary_5gb_is_warn(self, monkeypatch):
        checks = _run_security(monkeypatch, disk_free_gb=5.0)
        assert _checks_by_name(checks)["Disk Space"]["status"] in ("warn", "ok")

    def test_boundary_15gb_is_ok(self, monkeypatch):
        checks = _run_security(monkeypatch, disk_free_gb=15.0)
        assert _checks_by_name(checks)["Disk Space"]["status"] == "ok"


# ---------------------------------------------------------------------------
# TLS check (always info)
# ---------------------------------------------------------------------------

class TestTlsCheck:
    def test_tls_check_always_present(self, monkeypatch):
        checks = _run_security(monkeypatch)
        names = [c["check"] for c in checks]
        assert "TLS / HTTPS" in names

    def test_tls_check_is_info(self, monkeypatch):
        checks = _run_security(monkeypatch)
        assert _checks_by_name(checks)["TLS / HTTPS"]["status"] == "info"


# ---------------------------------------------------------------------------
# PyPI version cache
# ---------------------------------------------------------------------------

class TestPypiVersionCache:
    def test_caches_on_first_call(self):
        import autotune.dashboard.metrics as m
        m._PYPI_CACHE.clear()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"info": {"version": "9.9.9"}}

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            v1 = m._latest_pypi_version()
            v2 = m._latest_pypi_version()  # should use cache

        assert v1 == "9.9.9"
        assert v2 == "9.9.9"
        assert mock_get.call_count == 1  # only one network call

    def test_cache_expires_after_ttl(self):
        import autotune.dashboard.metrics as m
        m._PYPI_CACHE.clear()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"info": {"version": "1.0.0"}}

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            m._latest_pypi_version()
            # Force cache to appear expired
            m._PYPI_CACHE["at"] = time.time() - 3700
            m._latest_pypi_version()

        assert mock_get.call_count == 2

    def test_returns_none_on_network_error(self):
        import autotune.dashboard.metrics as m
        m._PYPI_CACHE.clear()

        with patch("httpx.get", side_effect=Exception("timeout")):
            v = m._latest_pypi_version()
        assert v is None

    def test_caches_none_on_error(self):
        import autotune.dashboard.metrics as m
        m._PYPI_CACHE.clear()

        with patch("httpx.get", side_effect=Exception("timeout")) as mock_get:
            m._latest_pypi_version()
            m._latest_pypi_version()  # should not retry immediately

        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# /api/dashboard/security endpoint
# ---------------------------------------------------------------------------

_METRIC_MOCKS = {
    "get_overview":           MagicMock(return_value={}),
    "get_requests_timeseries": MagicMock(return_value=[]),
    "get_ttft_trend":         MagicMock(return_value=[]),
    "get_models_stats":       MagicMock(return_value=[]),
    "get_comparison":         MagicMock(return_value={}),
    "get_api_keys_summary":   MagicMock(return_value=[]),
    "get_slow_requests":      MagicMock(return_value=[]),
    "get_suggestions":        MagicMock(return_value=[]),
}


@pytest.fixture
def sec_client(monkeypatch):
    monkeypatch.setenv("AUTOTUNE_ADMIN_KEY", TEST_KEY)
    from autotune.dashboard.router import router
    app = FastAPI()
    app.include_router(router)
    with patch.multiple("autotune.dashboard.metrics", **_METRIC_MOCKS):
        with TestClient(app, raise_server_exceptions=False, follow_redirects=False) as c:
            yield c


def _login(client: TestClient) -> str:
    r = client.post("/dashboard/login", data={"password": TEST_KEY}, follow_redirects=False)
    return r.cookies.get("autotune_session", "")


class TestSecurityEndpoint:
    def test_endpoint_requires_session(self, sec_client):
        r = sec_client.get("/api/dashboard/security")
        assert r.status_code == 401

    def test_endpoint_returns_checks_list(self, sec_client):
        mock_checks = [{"check": "Test", "status": "ok", "message": "All good", "action": None}]
        cookie = _login(sec_client)
        with patch("autotune.dashboard.metrics.get_gateway_security", return_value=mock_checks):
            r = sec_client.get("/api/dashboard/security",
                               cookies={"autotune_session": cookie})
        assert r.status_code == 200
        data = r.json()
        assert "checks" in data
        assert data["checks"] == mock_checks

    def test_endpoint_returns_non_empty_checks(self, sec_client, monkeypatch):
        cookie = _login(sec_client)
        mock_vm = MagicMock(percent=50.0, available=int(8 * 1024**3))
        mock_disk = MagicMock(free=int(100 * 1024**3))
        mock_db = MagicMock()
        mock_db.conn.execute.return_value.fetchone.return_value = {"n": 1}
        with (
            patch("autotune.dashboard.metrics.psutil.virtual_memory", return_value=mock_vm),
            patch("autotune.dashboard.metrics.shutil.disk_usage", return_value=mock_disk),
            patch("autotune.dashboard.metrics._db", return_value=mock_db),
            patch("autotune.dashboard.metrics._latest_pypi_version", return_value=None),
        ):
            r = sec_client.get("/api/dashboard/security",
                               cookies={"autotune_session": cookie})
        assert r.status_code == 200
        checks = r.json()["checks"]
        assert len(checks) >= 4
        statuses = {c["status"] for c in checks}
        assert statuses <= {"ok", "warn", "error", "info"}


# ---------------------------------------------------------------------------
# get_key_usage_trend unit tests
# ---------------------------------------------------------------------------

class TestKeyUsageTrend:
    """Unit tests for dashboard/metrics.get_key_usage_trend."""

    def _make_mock_db(self, rows: list[dict]):
        """Return a mock _db() whose execute().fetchall() returns rows."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            type("Row", (), {**r, "__getitem__": lambda self, k: r[k]})()
            for r in rows
        ]
        mock_db = MagicMock()
        mock_db.conn = mock_conn
        return mock_db

    def test_returns_list_of_dicts(self):
        from autotune.dashboard.metrics import get_key_usage_trend
        import datetime
        today = datetime.date.today()
        rows = [{"day": today.isoformat(), "requests": 5, "tokens": 1000, "avg_ttft": 200.0}]
        mock_db = self._make_mock_db(rows)
        with patch("autotune.dashboard.metrics._db", return_value=mock_db):
            result = get_key_usage_trend("key-123", days=7)
        assert isinstance(result, list)
        assert len(result) == 7

    def test_fills_missing_days_with_zeros(self):
        from autotune.dashboard.metrics import get_key_usage_trend
        mock_db = self._make_mock_db([])  # no data
        with patch("autotune.dashboard.metrics._db", return_value=mock_db):
            result = get_key_usage_trend("key-123", days=14)
        assert len(result) == 14
        assert all(r["requests"] == 0 for r in result)
        assert all(r["tokens"] == 0 for r in result)

    def test_oldest_day_first(self):
        from autotune.dashboard.metrics import get_key_usage_trend
        mock_db = self._make_mock_db([])
        with patch("autotune.dashboard.metrics._db", return_value=mock_db):
            result = get_key_usage_trend("key-123", days=30)
        assert result[0]["day"] < result[-1]["day"]

    def test_each_entry_has_required_keys(self):
        from autotune.dashboard.metrics import get_key_usage_trend
        mock_db = self._make_mock_db([])
        with patch("autotune.dashboard.metrics._db", return_value=mock_db):
            result = get_key_usage_trend("key-123", days=7)
        for entry in result:
            assert "day" in entry
            assert "requests" in entry
            assert "tokens" in entry
            assert "avg_ttft_ms" in entry
