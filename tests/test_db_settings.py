"""
Comprehensive tests for Database settings CRUD, new schema columns,
and maintenance operations added in v1.5.

Covers:
- Settings table seeded with all defaults on connect
- get_setting / set_setting / get_all_settings
- set_setting with and without description
- _migrate_settings_table idempotency
- New run_observations columns (backend, request_id, conversation_id,
  stress_retry, thinking_tokens, error_code)
- New hardware_profiles columns (metal_gpu_cores, cpu_freq_mhz, storage_type)
- New models columns (ollama_tag, tier, is_local)
- cleanup_old_data: retention enforcement, zero-day retention no-op
- optimize: runs without error
- stats includes settings count
- Thread-safety for settings writes
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from autotune.db.store import Database, _SAFE_COL


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(path=tmp_path / "test.db")
    d.connect()
    yield d
    d.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_data(**extra) -> dict:
    return {
        "model_id": "qwen3:8b",
        "quant": "Q4_K_M",
        "context_len": 4096,
        "n_gpu_layers": 99,
        "observed_at": time.time(),
        **extra,
    }


def _hw_data(**extra) -> dict:
    return {
        "id": "hw-test-001",
        "os_name": "Darwin",
        "cpu_brand": "Apple M3",
        "total_ram_gb": 24.0,
        "first_seen": time.time(),
        "last_seen": time.time(),
        **extra,
    }


def _model_data(**extra) -> dict:
    return {
        "id": "meta-llama/Meta-Llama-3.1-8B",
        "name": "Llama 3.1 8B",
        "total_params_b": 8.0,
        "active_params_b": 8.0,
        "fetched_at": time.time(),
        **extra,
    }


# ---------------------------------------------------------------------------
# Settings table: defaults seeded on connect
# ---------------------------------------------------------------------------

class TestSettingsDefaults:
    def test_all_defaults_seeded(self, db: Database):
        settings = db.get_all_settings()
        expected_keys = {
            "default_qos_profile",
            "ollama_url",
            "catalog_refresh_interval_h",
            "dashboard_session_timeout_h",
            "show_thinking_tokens",
            "retention_days",
        }
        assert expected_keys <= set(settings.keys()), (
            f"Missing keys: {expected_keys - set(settings.keys())}"
        )

    def test_default_qos_profile_value(self, db: Database):
        assert db.get_setting("default_qos_profile") == "balanced"

    def test_catalog_refresh_interval_h_value(self, db: Database):
        assert db.get_setting("catalog_refresh_interval_h") == "24"

    def test_retention_days_value(self, db: Database):
        assert db.get_setting("retention_days") == "90"

    def test_show_thinking_tokens_value(self, db: Database):
        assert db.get_setting("show_thinking_tokens") == "0"

    def test_ollama_url_default_empty(self, db: Database):
        assert db.get_setting("ollama_url") == ""

    def test_settings_not_overwritten_on_reconnect(self, tmp_path: Path):
        """INSERT OR IGNORE: changing a setting persists across reconnects."""
        p = tmp_path / "persist.db"
        with Database(path=p) as d:
            d.set_setting("retention_days", "30")

        with Database(path=p) as d:
            assert d.get_setting("retention_days") == "30"


# ---------------------------------------------------------------------------
# get_setting
# ---------------------------------------------------------------------------

class TestGetSetting:
    def test_returns_default_value_for_known_key(self, db: Database):
        v = db.get_setting("default_qos_profile", default="fast")
        assert v == "balanced"  # seeded default, not the fallback

    def test_returns_passed_default_for_unknown_key(self, db: Database):
        v = db.get_setting("nonexistent_key_xyz", default="fallback")
        assert v == "fallback"

    def test_returns_none_for_unknown_key_no_default(self, db: Database):
        assert db.get_setting("totally_unknown_key") is None

    def test_returns_updated_value(self, db: Database):
        db.set_setting("retention_days", "180")
        assert db.get_setting("retention_days") == "180"


# ---------------------------------------------------------------------------
# set_setting
# ---------------------------------------------------------------------------

class TestSetSetting:
    def test_creates_new_key(self, db: Database):
        db.set_setting("my_custom_key", "my_value")
        assert db.get_setting("my_custom_key") == "my_value"

    def test_updates_existing_key(self, db: Database):
        db.set_setting("retention_days", "60")
        db.set_setting("retention_days", "365")
        assert db.get_setting("retention_days") == "365"

    def test_with_description(self, db: Database):
        db.set_setting("my_key", "v1", description="test description")
        settings = db.get_all_settings()
        assert settings["my_key"]["description"] == "test description"

    def test_without_description_preserves_existing_description(self, db: Database):
        db.set_setting("my_key", "v1", description="keep me")
        db.set_setting("my_key", "v2")
        settings = db.get_all_settings()
        assert settings["my_key"]["value"] == "v2"
        assert settings["my_key"]["description"] == "keep me"

    def test_updated_at_changes(self, db: Database):
        before = db.get_all_settings()["retention_days"]["updated_at"]
        time.sleep(0.01)
        db.set_setting("retention_days", "999")
        after = db.get_all_settings()["retention_days"]["updated_at"]
        assert after >= before

    def test_empty_string_value_allowed(self, db: Database):
        db.set_setting("ollama_url", "")
        assert db.get_setting("ollama_url") == ""

    def test_url_value_stored(self, db: Database):
        url = "http://localhost:11434"
        db.set_setting("ollama_url", url)
        assert db.get_setting("ollama_url") == url


# ---------------------------------------------------------------------------
# get_all_settings
# ---------------------------------------------------------------------------

class TestGetAllSettings:
    def test_returns_dict_with_all_seeded_keys(self, db: Database):
        settings = db.get_all_settings()
        assert len(settings) >= 6

    def test_each_entry_has_required_fields(self, db: Database):
        settings = db.get_all_settings()
        for key, entry in settings.items():
            assert "value" in entry, f"Missing 'value' in {key}"
            assert "updated_at" in entry, f"Missing 'updated_at' in {key}"
            assert "description" in entry, f"Missing 'description' in {key}"

    def test_returned_in_alphabetical_order(self, db: Database):
        keys = list(db.get_all_settings().keys())
        assert keys == sorted(keys)

    def test_custom_key_appears(self, db: Database):
        db.set_setting("zzz_custom", "custom_val")
        settings = db.get_all_settings()
        assert "zzz_custom" in settings
        assert settings["zzz_custom"]["value"] == "custom_val"


# ---------------------------------------------------------------------------
# _migrate_settings_table idempotency
# ---------------------------------------------------------------------------

class TestMigrateSettingsIdempotent:
    def test_double_migrate_is_safe(self, db: Database):
        db._migrate_settings_table()  # already ran in connect()
        db._migrate_settings_table()  # should not raise or duplicate rows
        settings = db.get_all_settings()
        assert len([k for k in settings if k == "retention_days"]) == 1


# ---------------------------------------------------------------------------
# New run_observations columns (v1.5+)
# ---------------------------------------------------------------------------

class TestRunObservationsNewColumns:
    def test_backend_column_stores_value(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data(backend="ollama"))
            runs = db.get_runs()
        assert runs[0]["backend"] == "ollama"

    def test_backend_column_accepts_mlx(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data(backend="mlx"))
            runs = db.get_runs()
        assert runs[0]["backend"] == "mlx"

    def test_request_id_column_stored(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            req_id = "req-abc-123"
            db.log_run(_run_data(request_id=req_id))
            runs = db.get_runs()
        assert runs[0]["request_id"] == req_id

    def test_conversation_id_column_stored(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            conv_id = "conv-xyz-789"
            db.log_run(_run_data(conversation_id=conv_id))
            runs = db.get_runs()
        assert runs[0]["conversation_id"] == conv_id

    def test_stress_retry_defaults_to_zero(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data())
            runs = db.get_runs()
        # SQLite may return 0 or None depending on migration state
        assert runs[0].get("stress_retry") in (0, None)

    def test_stress_retry_stores_one(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data(stress_retry=1))
            runs = db.get_runs()
        assert runs[0]["stress_retry"] == 1

    def test_thinking_tokens_column_stored(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data(thinking_tokens=512))
            runs = db.get_runs()
        assert runs[0]["thinking_tokens"] == 512

    def test_error_code_column_stored(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data(error_code="OOM_KILLED"))
            runs = db.get_runs()
        assert runs[0]["error_code"] == "OOM_KILLED"

    def test_all_new_columns_nullable_by_default(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data())
            runs = db.get_runs()
        row = runs[0]
        # These columns should be present (not KeyError) and nullable
        assert "backend" in row
        assert "request_id" in row
        assert "conversation_id" in row
        assert "thinking_tokens" in row
        assert "error_code" in row

    def test_backend_index_usable(self, db: Database):
        """Verify the partial index on backend doesn't block queries."""
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data(backend="ollama"))
            db.log_run(_run_data(backend="mlx"))
        # Direct SQL query using the index
        rows = db.conn.execute(
            "SELECT * FROM run_observations WHERE backend = ?", ("ollama",)
        ).fetchall()
        assert len(rows) == 1

    def test_request_id_index_usable(self, db: Database):
        req_id = "unique-req-id-abc"
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data(request_id=req_id))
        rows = db.conn.execute(
            "SELECT * FROM run_observations WHERE request_id = ?", (req_id,)
        ).fetchall()
        assert len(rows) == 1

    def test_model_time_index_usable(self, db: Database):
        """Composite index (model_id, observed_at DESC) supports model lookups."""
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data(model_id="qwen3:8b"))
        rows = db.conn.execute(
            "SELECT * FROM run_observations WHERE model_id = ? ORDER BY observed_at DESC",
            ("qwen3:8b",),
        ).fetchall()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# New hardware_profiles columns (v1.5+)
# ---------------------------------------------------------------------------

class TestHardwareProfilesNewColumns:
    def test_metal_gpu_cores_stored(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.upsert_hardware(_hw_data(metal_gpu_cores=30))
            hw = db.get_hardware("hw-test-001")
        assert hw["metal_gpu_cores"] == 30

    def test_cpu_freq_mhz_stored(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.upsert_hardware(_hw_data(cpu_freq_mhz=3200.0))
            hw = db.get_hardware("hw-test-001")
        assert hw["cpu_freq_mhz"] == 3200.0

    def test_storage_type_stored(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.upsert_hardware(_hw_data(storage_type="nvme"))
            hw = db.get_hardware("hw-test-001")
        assert hw["storage_type"] == "nvme"

    def test_new_columns_nullable(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.upsert_hardware(_hw_data())
            hw = db.get_hardware("hw-test-001")
        # Should be present but may be None
        assert "metal_gpu_cores" in hw
        assert "cpu_freq_mhz" in hw
        assert "storage_type" in hw

    def test_all_three_columns_together(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.upsert_hardware(_hw_data(
                metal_gpu_cores=40, cpu_freq_mhz=3400.0, storage_type="ssd"
            ))
            hw = db.get_hardware("hw-test-001")
        assert hw["metal_gpu_cores"] == 40
        assert hw["cpu_freq_mhz"] == 3400.0
        assert hw["storage_type"] == "ssd"


# ---------------------------------------------------------------------------
# New models columns (catalog additions)
# ---------------------------------------------------------------------------

class TestModelsCatalogColumns:
    def test_ollama_tag_stored(self, db: Database):
        db.upsert_model(_model_data(ollama_tag="llama3.1:8b"))
        m = db.get_model("meta-llama/Meta-Llama-3.1-8B")
        assert m["ollama_tag"] == "llama3.1:8b"

    def test_tier_stored(self, db: Database):
        db.upsert_model(_model_data(tier="medium"))
        m = db.get_model("meta-llama/Meta-Llama-3.1-8B")
        assert m["tier"] == "medium"

    def test_is_local_defaults_to_zero(self, db: Database):
        db.upsert_model(_model_data())
        m = db.get_model("meta-llama/Meta-Llama-3.1-8B")
        assert m.get("is_local") in (0, None)

    def test_is_local_set_to_one(self, db: Database):
        db.upsert_model(_model_data(is_local=1))
        m = db.get_model("meta-llama/Meta-Llama-3.1-8B")
        assert m["is_local"] == 1

    def test_all_catalog_columns_together(self, db: Database):
        db.upsert_model(_model_data(
            ollama_tag="llama3.1:8b", tier="medium", is_local=1
        ))
        m = db.get_model("meta-llama/Meta-Llama-3.1-8B")
        assert m["ollama_tag"] == "llama3.1:8b"
        assert m["tier"] == "medium"
        assert m["is_local"] == 1

    def test_tier_values_small_large(self, db: Database):
        for tier in ("tiny", "small", "medium", "large", "xl"):
            db.upsert_model({**_model_data(), "id": f"org/model-{tier}", "tier": tier})
            m = db.get_model(f"org/model-{tier}")
            assert m["tier"] == tier


# ---------------------------------------------------------------------------
# cleanup_old_data
# ---------------------------------------------------------------------------

class TestCleanupOldData:
    def _insert_old_run(self, db: Database, days_ago: float) -> None:
        old_ts = time.time() - days_ago * 86400
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data(observed_at=old_ts))

    def _insert_old_telemetry(self, db: Database, days_ago: float) -> None:
        old_ts = time.time() - days_ago * 86400
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.conn.execute(
                """INSERT INTO telemetry_events
                   (event_type, observed_at) VALUES (?, ?)""",
                ("ram_spike", old_ts),
            )
            db.conn.commit()

    def test_deletes_old_runs_beyond_retention(self, db: Database):
        self._insert_old_run(db, days_ago=100)
        self._insert_old_run(db, days_ago=50)
        result = db.cleanup_old_data(retention_days=90)
        assert result["run_observations"] == 1  # only the 100-day-old one
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            remaining = db.get_runs()
        assert len(remaining) == 1  # 50-day-old still there

    def test_deletes_old_telemetry_beyond_retention(self, db: Database):
        self._insert_old_telemetry(db, days_ago=100)
        self._insert_old_telemetry(db, days_ago=50)
        result = db.cleanup_old_data(retention_days=90)
        assert result["telemetry_events"] == 1

    def test_zero_retention_returns_zero_deleted(self, db: Database):
        self._insert_old_run(db, days_ago=1000)
        result = db.cleanup_old_data(retention_days=0)
        assert result["run_observations"] == 0
        assert result["telemetry_events"] == 0

    def test_negative_retention_returns_zero_deleted(self, db: Database):
        self._insert_old_run(db, days_ago=1000)
        result = db.cleanup_old_data(retention_days=-1)
        assert result["run_observations"] == 0

    def test_returns_zero_when_nothing_to_delete(self, db: Database):
        self._insert_old_run(db, days_ago=1)  # recent, within retention
        result = db.cleanup_old_data(retention_days=90)
        assert result["run_observations"] == 0
        assert result["telemetry_events"] == 0

    def test_returns_dict_with_expected_keys(self, db: Database):
        result = db.cleanup_old_data(retention_days=90)
        assert "run_observations" in result
        assert "telemetry_events" in result
        assert "security_events" in result

    def test_deletes_multiple_old_rows(self, db: Database):
        for _ in range(5):
            self._insert_old_run(db, days_ago=200)
        result = db.cleanup_old_data(retention_days=90)
        assert result["run_observations"] == 5

    def test_security_events_always_zero_in_result(self, db: Database):
        result = db.cleanup_old_data(retention_days=1)
        assert result["security_events"] == 0


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

class TestOptimize:
    def test_optimize_runs_without_error(self, db: Database):
        db.optimize()  # should not raise

    def test_optimize_after_inserts(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            for _ in range(10):
                db.log_run(_run_data())
        db.optimize()  # should not raise

    def test_optimize_and_cleanup_together(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            old_ts = time.time() - 200 * 86400
            db.log_run(_run_data(observed_at=old_ts))
        deleted = db.cleanup_old_data(retention_days=90)
        db.optimize()
        assert deleted["run_observations"] == 1


# ---------------------------------------------------------------------------
# stats includes settings count
# ---------------------------------------------------------------------------

class TestStatsWithSettings:
    def test_stats_includes_settings_key(self, db: Database):
        stats = db.stats()
        assert "settings" in stats

    def test_settings_count_matches_defaults(self, db: Database):
        stats = db.stats()
        assert stats["settings"] >= 6  # at least all 6 defaults seeded

    def test_settings_count_increments_on_new_setting(self, db: Database):
        before = db.stats()["settings"]
        db.set_setting("new_custom_setting", "value")
        after = db.stats()["settings"]
        assert after == before + 1

    def test_stats_includes_all_core_keys(self, db: Database):
        stats = db.stats()
        required = {
            "models", "hardware_profiles", "run_observations",
            "telemetry_events", "security_events", "settings",
            "db_path", "db_size_mb",
        }
        assert required <= set(stats.keys())


# ---------------------------------------------------------------------------
# Thread-safety for settings
# ---------------------------------------------------------------------------

class TestSettingsThreadSafety:
    def test_concurrent_set_setting_all_succeed(self, db: Database):
        errors = []

        def _write(i: int) -> None:
            try:
                db.set_setting(f"thread_key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        for i in range(10):
            assert db.get_setting(f"thread_key_{i}") == f"value_{i}"


# ---------------------------------------------------------------------------
# Schema completeness — verify all new columns exist in SQLite
# ---------------------------------------------------------------------------

class TestSchemaCompleteness:
    def _columns(self, db: Database, table: str) -> set:
        rows = db.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    def test_run_observations_has_v15_columns(self, db: Database):
        cols = self._columns(db, "run_observations")
        expected = {"backend", "request_id", "conversation_id",
                    "stress_retry", "thinking_tokens", "error_code"}
        missing = expected - cols
        assert not missing, f"Missing columns in run_observations: {missing}"

    def test_hardware_profiles_has_v15_columns(self, db: Database):
        cols = self._columns(db, "hardware_profiles")
        expected = {"metal_gpu_cores", "cpu_freq_mhz", "storage_type"}
        missing = expected - cols
        assert not missing, f"Missing columns in hardware_profiles: {missing}"

    def test_models_has_catalog_columns(self, db: Database):
        cols = self._columns(db, "models")
        expected = {"ollama_tag", "tier", "is_local"}
        missing = expected - cols
        assert not missing, f"Missing columns in models: {missing}"

    def test_settings_table_exists_with_columns(self, db: Database):
        cols = self._columns(db, "settings")
        assert {"key", "value", "updated_at", "description"} <= cols

    def test_api_keys_table_exists(self, db: Database):
        tables = {r[0] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "api_keys" in tables
        assert "api_key_usage" in tables
        assert "revoked_sessions" in tables
        assert "security_events" in tables
