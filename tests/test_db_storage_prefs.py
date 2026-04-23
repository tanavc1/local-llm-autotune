"""
Tests for autotune.db.storage_prefs — local storage opt-in/out preference.

Covers:
- is_storage_enabled: default True (no file), True when enabled=true, False when disabled
- set_storage_enabled: writes file; reading back gives correct value
- storage_pref_set: False before file exists, True after
- Round-trip: enable → read → disable → read
- File corruption tolerance: invalid JSON → defaults to True
- OS-error tolerance: non-fatal on write failure
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autotune.db.storage_prefs import (
    is_storage_enabled,
    set_storage_enabled,
    storage_pref_set,
)

# ---------------------------------------------------------------------------
# Helpers: redirect config dir to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_prefs(tmp_path: Path, monkeypatch):
    """Redirect all prefs I/O to a temp directory so tests don't touch real config."""
    monkeypatch.setattr(
        "autotune.db.storage_prefs._config_dir",
        lambda: tmp_path,
    )
    yield


# ---------------------------------------------------------------------------
# is_storage_enabled
# ---------------------------------------------------------------------------

class TestIsStorageEnabled:
    def test_default_true_when_no_file(self):
        assert is_storage_enabled() is True

    def test_true_when_file_says_enabled(self):
        set_storage_enabled(True)
        assert is_storage_enabled() is True

    def test_false_when_file_says_disabled(self):
        set_storage_enabled(False)
        assert is_storage_enabled() is False

    def test_invalid_json_defaults_to_true(self, tmp_path: Path):
        prefs_file = tmp_path / "storage_prefs.json"
        prefs_file.write_text("{not valid json}")
        assert is_storage_enabled() is True

    def test_missing_enabled_key_defaults_to_true(self, tmp_path: Path):
        prefs_file = tmp_path / "storage_prefs.json"
        prefs_file.write_text(json.dumps({"set_at": "2025-01-01"}))
        assert is_storage_enabled() is True


# ---------------------------------------------------------------------------
# set_storage_enabled
# ---------------------------------------------------------------------------

class TestSetStorageEnabled:
    def test_writes_file(self, tmp_path: Path):
        set_storage_enabled(True)
        prefs_file = tmp_path / "storage_prefs.json"
        assert prefs_file.exists()

    def test_file_contains_set_at(self, tmp_path: Path):
        set_storage_enabled(True)
        data = json.loads((tmp_path / "storage_prefs.json").read_text())
        assert "set_at" in data

    def test_round_trip_enable(self):
        set_storage_enabled(True)
        assert is_storage_enabled() is True

    def test_round_trip_disable(self):
        set_storage_enabled(False)
        assert is_storage_enabled() is False

    def test_toggle_enable_then_disable(self):
        set_storage_enabled(True)
        assert is_storage_enabled() is True
        set_storage_enabled(False)
        assert is_storage_enabled() is False

    def test_idempotent_double_disable(self):
        set_storage_enabled(False)
        set_storage_enabled(False)
        assert is_storage_enabled() is False

    def test_write_error_non_fatal(self, tmp_path: Path):
        with patch("autotune.db.storage_prefs.Path.write_text", side_effect=OSError("disk full")):
            set_storage_enabled(True)   # should not raise


# ---------------------------------------------------------------------------
# storage_pref_set
# ---------------------------------------------------------------------------

class TestStoragePrefSet:
    def test_false_before_file_created(self):
        assert storage_pref_set() is False

    def test_true_after_set_called(self):
        set_storage_enabled(True)
        assert storage_pref_set() is True

    def test_true_after_disable_called(self):
        set_storage_enabled(False)
        assert storage_pref_set() is True
