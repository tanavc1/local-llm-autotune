"""
Tests for autotune.db.store — Database CRUD and schema correctness.

Covers:
- Database connect/close lifecycle
- upsert_model / get_model / list_models
- upsert_hardware / get_hardware (storage-gated)
- log_run / get_runs (storage-gated)
- log_telemetry_event / get_telemetry / telemetry_summary
- model_perf_history / compare_runs
- stats / model_count
- _safe_cols: rejects unsafe column names
- Storage disabled: log_run / log_telemetry returns -1
- Thread-safety: multiple inserts from threads don't corrupt DB
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from autotune.db.store import Database, _safe_cols

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> Database:
    """Fresh in-memory-style DB backed by a temp file for each test."""
    d = Database(path=tmp_path / "test.db")
    d.connect()
    yield d
    d.close()


def _model_data(model_id: str = "test/model-7b") -> dict:
    return {
        "id": model_id,
        "name": "Model 7B",
        "total_params_b": 7.0,
        "active_params_b": 7.0,
        "fetched_at": time.time(),
    }


def _hw_data(hw_id: str = "abc123") -> dict:
    return {
        "id": hw_id,
        "os_name": "Darwin",
        "cpu_brand": "Apple M3",
        "total_ram_gb": 32.0,
        "first_seen": time.time(),
        "last_seen": time.time(),
    }


def _run_data(model_id: str = "test/model-7b", hw_id: Optional[str] = None) -> dict:
    data: dict = {
        "model_id": model_id,
        "quant": "Q4_K_M",
        "context_len": 4096,
        "n_gpu_layers": 99,
        "tokens_per_sec": 42.5,
        "observed_at": time.time(),
    }
    if hw_id is not None:
        data["hardware_id"] = hw_id
    return data


# ---------------------------------------------------------------------------
# _safe_cols
# ---------------------------------------------------------------------------

class TestSafeCols:
    def test_valid_column_names_pass(self):
        _safe_cols({"model_id": "x", "tokens_per_sec": 1.0})  # no exception

    def test_column_with_semicolon_raises(self):
        with pytest.raises(ValueError, match="Unsafe"):
            _safe_cols({"bad;col": "x"})

    def test_column_with_space_raises(self):
        with pytest.raises(ValueError, match="Unsafe"):
            _safe_cols({"bad col": "x"})

    def test_column_with_dash_raises(self):
        with pytest.raises(ValueError, match="Unsafe"):
            _safe_cols({"bad-col": "x"})

    def test_leading_digit_raises(self):
        with pytest.raises(ValueError, match="Unsafe"):
            _safe_cols({"1col": "x"})

    def test_underscore_prefix_valid(self):
        _safe_cols({"_private": "x"})  # no exception


# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------

class TestDatabaseLifecycle:
    def test_connect_creates_file(self, tmp_path: Path):
        db_path = tmp_path / "autotune.db"
        db = Database(path=db_path)
        db.connect()
        assert db_path.exists()
        db.close()

    def test_context_manager(self, tmp_path: Path):
        with Database(path=tmp_path / "autotune.db") as db:
            assert db._conn is not None

    def test_conn_raises_before_connect(self, tmp_path: Path):
        db = Database(path=tmp_path / "autotune.db")
        with pytest.raises(RuntimeError, match="not connected"):
            _ = db.conn

    def test_schema_tables_created(self, db: Database):
        tables = {
            row[0] for row in
            db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "models" in tables
        assert "hardware_profiles" in tables
        assert "run_observations" in tables
        assert "telemetry_events" in tables


# ---------------------------------------------------------------------------
# Model CRUD
# ---------------------------------------------------------------------------

class TestModelCrud:
    def test_upsert_and_get_model(self, db: Database):
        db.upsert_model(_model_data("org/model-7b"))
        m = db.get_model("org/model-7b")
        assert m is not None
        assert m["name"] == "Model 7B"

    def test_get_nonexistent_model_returns_none(self, db: Database):
        assert db.get_model("does/not-exist") is None

    def test_upsert_replaces_existing(self, db: Database):
        db.upsert_model({**_model_data("org/x"), "total_params_b": 7.0})
        db.upsert_model({**_model_data("org/x"), "total_params_b": 8.0})
        m = db.get_model("org/x")
        assert m["total_params_b"] == 8.0

    def test_available_quants_serialized_as_json(self, db: Database):
        data = {**_model_data("org/q"), "available_quants": ["Q4_K_M", "Q8_0"]}
        db.upsert_model(data)
        m = db.get_model("org/q")
        assert isinstance(m["available_quants"], list)
        assert "Q4_K_M" in m["available_quants"]

    def test_list_models_returns_all(self, db: Database):
        db.upsert_model(_model_data("org/a"))
        db.upsert_model(_model_data("org/b"))
        models = db.list_models()
        ids = [m["id"] for m in models]
        assert "org/a" in ids and "org/b" in ids

    def test_list_models_filter_by_max_params(self, db: Database):
        db.upsert_model({**_model_data("org/small"), "active_params_b": 3.0})
        db.upsert_model({**_model_data("org/big"),   "active_params_b": 70.0})
        small = db.list_models(max_params_b=10.0)
        ids = [m["id"] for m in small]
        assert "org/small" in ids
        assert "org/big" not in ids

    def test_model_count_increments(self, db: Database):
        before = db.model_count()
        db.upsert_model(_model_data("org/new"))
        assert db.model_count() == before + 1


# ---------------------------------------------------------------------------
# Hardware profiles (storage-gated)
# ---------------------------------------------------------------------------

class TestHardwareProfiles:
    def test_upsert_and_get_hardware(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.upsert_hardware(_hw_data("hw-001"))
            hw = db.get_hardware("hw-001")
        assert hw is not None
        assert hw["cpu_brand"] == "Apple M3"

    def test_upsert_hardware_updates_last_seen(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            t0 = time.time()
            db.upsert_hardware({**_hw_data("hw-002"), "last_seen": t0})
            time.sleep(0.01)
            db.upsert_hardware({**_hw_data("hw-002")})
            hw = db.get_hardware("hw-002")
        assert hw["last_seen"] >= t0

    def test_upsert_hardware_disabled_returns_id(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=False):
            result = db.upsert_hardware(_hw_data("hw-skip"))
        assert result == "hw-skip"

    def test_get_nonexistent_hardware_returns_none(self, db: Database):
        assert db.get_hardware("nonexistent-id") is None


# ---------------------------------------------------------------------------
# Run observations (storage-gated)
# ---------------------------------------------------------------------------

class TestRunObservations:
    def test_log_and_get_run(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            run_id = db.log_run(_run_data())
            runs = db.get_runs()
        assert run_id > 0
        assert len(runs) == 1
        assert runs[0]["model_id"] == "test/model-7b"

    def test_log_run_disabled_returns_minus_one(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=False):
            result = db.log_run(_run_data())
        assert result == -1

    def test_get_runs_filter_by_model(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data("model-a"))
            db.log_run(_run_data("model-b"))
            runs = db.get_runs(model_id="model-a")
        assert all(r["model_id"] == "model-a" for r in runs)

    def test_get_runs_respects_limit(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            for _ in range(10):
                db.log_run(_run_data())
            runs = db.get_runs(limit=3)
        assert len(runs) == 3

    def test_model_perf_history(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run({**_run_data("hist-model"), "tokens_per_sec": 35.0})
            history = db.model_perf_history("hist-model")
        assert len(history) == 1
        assert history[0]["tokens_per_sec"] == 35.0

    def test_compare_runs_no_matching_tags(self, db: Database):
        result = db.compare_runs("tag-x", "tag-y")
        assert "error" in result

    def test_get_runs_by_tag(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run({**_run_data(), "notes": "bench_tag=my-bench"})
            runs = db.get_runs_by_tag("my-bench")
        assert len(runs) == 1


# ---------------------------------------------------------------------------
# Telemetry events
# ---------------------------------------------------------------------------

class TestTelemetryEvents:
    def test_log_and_get_telemetry(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            ev_id = db.log_telemetry_event("session_start", model_id="model-x")
            events = db.get_telemetry()
        assert ev_id > 0
        assert len(events) == 1
        assert events[0]["event_type"] == "session_start"

    def test_log_telemetry_disabled_returns_minus_one(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=False):
            result = db.log_telemetry_event("session_start")
        assert result == -1

    def test_get_telemetry_filter_by_event_type(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_telemetry_event("session_start")
            db.log_telemetry_event("oom_near")
            events = db.get_telemetry(event_type="oom_near")
        assert all(e["event_type"] == "oom_near" for e in events)

    def test_telemetry_summary_counts(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_telemetry_event("session_start")
            db.log_telemetry_event("session_start")
            db.log_telemetry_event("oom_near")
        summary = db.telemetry_summary()
        assert summary.get("session_start") == 2
        assert summary.get("oom_near") == 1

    def test_telemetry_value_num_stored(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_telemetry_event("ram_spike", value_num=32.5)
            events = db.get_telemetry(event_type="ram_spike")
        assert events[0]["value_num"] == 32.5


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_counts_and_path(self, db: Database):
        db.upsert_model(_model_data("org/stat-test"))
        stats = db.stats()
        assert "models" in stats
        assert stats["models"] >= 1
        assert "db_path" in stats
        assert "db_size_mb" in stats

    def test_stats_run_count_increments(self, db: Database):
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            db.log_run(_run_data())
        stats = db.stats()
        assert stats["run_observations"] >= 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_inserts_all_succeed(self, db: Database):
        errors = []

        def insert_run(i: int) -> None:
            try:
                with patch("autotune.db.store.is_storage_enabled", return_value=True):
                    db.log_run({**_run_data(), "tokens_per_sec": float(i)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert_run, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        with patch("autotune.db.store.is_storage_enabled", return_value=True):
            runs = db.get_runs(limit=20)
        assert len(runs) == 10
