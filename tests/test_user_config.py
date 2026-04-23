"""
Tests for autotune.config.user_config — persistent user defaults.

Covers:
- load_config: empty dict when file absent, parsed dict when present
- save_config: writes file with _updated_at field
- get_value: None when not set, correct value when set
- set_value: valid keys, invalid key error, type coercion (int), invalid profile
- reset_config: deletes file, subsequent load returns empty dict
- effective_default: stored value takes precedence over built-in default;
  falls back to built-in default when not set
- File corruption tolerance
- KNOWN_KEYS: all four expected keys present
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotune.config.user_config import (
    KNOWN_KEYS,
    effective_default,
    get_value,
    load_config,
    reset_config,
    save_config,
    set_value,
)

# ---------------------------------------------------------------------------
# Redirect config to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "autotune.config.user_config._config_dir",
        lambda: tmp_path,
    )
    yield


# ---------------------------------------------------------------------------
# KNOWN_KEYS contract
# ---------------------------------------------------------------------------

class TestKnownKeys:
    def test_all_four_keys_present(self):
        assert "default_model"   in KNOWN_KEYS
        assert "default_profile" in KNOWN_KEYS
        assert "serve_host"      in KNOWN_KEYS
        assert "serve_port"      in KNOWN_KEYS

    def test_each_entry_is_three_tuple(self):
        for key, entry in KNOWN_KEYS.items():
            assert len(entry) == 3, f"{key} should be (type, description, default)"

    def test_serve_port_type_is_int(self):
        assert KNOWN_KEYS["serve_port"][0] == "int"

    def test_default_profile_default_is_balanced(self):
        assert KNOWN_KEYS["default_profile"][2] == "balanced"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_empty_dict_when_no_file(self):
        assert load_config() == {}

    def test_returns_stored_values(self, tmp_path: Path):
        (tmp_path / "user_config.json").write_text(
            json.dumps({"default_model": "qwen3:8b"})
        )
        data = load_config()
        assert data["default_model"] == "qwen3:8b"

    def test_invalid_json_returns_empty_dict(self, tmp_path: Path):
        (tmp_path / "user_config.json").write_text("{corrupted}")
        assert load_config() == {}


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------

class TestSaveConfig:
    def test_creates_file(self, tmp_path: Path):
        save_config({"default_model": "phi4"})
        assert (tmp_path / "user_config.json").exists()

    def test_adds_updated_at(self, tmp_path: Path):
        save_config({"default_model": "phi4"})
        data = json.loads((tmp_path / "user_config.json").read_text())
        assert "_updated_at" in data

    def test_persisted_values_readable(self, tmp_path: Path):
        save_config({"serve_port": 9000})
        data = json.loads((tmp_path / "user_config.json").read_text())
        assert data["serve_port"] == 9000


# ---------------------------------------------------------------------------
# get_value
# ---------------------------------------------------------------------------

class TestGetValue:
    def test_returns_none_when_not_set(self):
        assert get_value("default_model") is None

    def test_returns_stored_value(self):
        set_value("serve_host", "0.0.0.0")
        assert get_value("serve_host") == "0.0.0.0"

    def test_returns_none_for_unknown_key(self):
        assert get_value("nonexistent_key") is None


# ---------------------------------------------------------------------------
# set_value
# ---------------------------------------------------------------------------

class TestSetValue:
    def test_set_string_key(self):
        ok, err = set_value("default_model", "llama3:8b")
        assert ok is True
        assert err == ""
        assert get_value("default_model") == "llama3:8b"

    def test_set_int_key_coerces_string(self):
        ok, err = set_value("serve_port", "9999")
        assert ok is True
        assert get_value("serve_port") == 9999
        assert isinstance(get_value("serve_port"), int)

    def test_set_int_key_rejects_non_integer(self):
        ok, err = set_value("serve_port", "not-a-number")
        assert ok is False
        assert "integer" in err

    def test_unknown_key_returns_error(self):
        ok, err = set_value("unknown_key", "value")
        assert ok is False
        assert "Unknown key" in err

    def test_invalid_profile_value_rejected(self):
        ok, err = set_value("default_profile", "turbo")
        assert ok is False
        assert "fast" in err or "balanced" in err

    @pytest.mark.parametrize("profile", ["fast", "balanced", "quality"])
    def test_valid_profiles_accepted(self, profile: str):
        ok, err = set_value("default_profile", profile)
        assert ok is True, f"Profile {profile!r} should be accepted"
        assert get_value("default_profile") == profile

    def test_overwrite_existing_value(self):
        set_value("serve_host", "127.0.0.1")
        set_value("serve_host", "0.0.0.0")
        assert get_value("serve_host") == "0.0.0.0"

    def test_multiple_keys_independent(self):
        set_value("default_profile", "fast")
        set_value("serve_host", "localhost")
        assert get_value("default_profile") == "fast"
        assert get_value("serve_host") == "localhost"


# ---------------------------------------------------------------------------
# reset_config
# ---------------------------------------------------------------------------

class TestResetConfig:
    def test_reset_removes_file(self, tmp_path: Path):
        set_value("default_model", "llama3:8b")
        assert (tmp_path / "user_config.json").exists()
        reset_config()
        assert not (tmp_path / "user_config.json").exists()

    def test_after_reset_load_returns_empty(self):
        set_value("default_model", "llama3:8b")
        reset_config()
        assert load_config() == {}

    def test_reset_idempotent_when_no_file(self):
        reset_config()   # should not raise


# ---------------------------------------------------------------------------
# effective_default
# ---------------------------------------------------------------------------

class TestEffectiveDefault:
    def test_returns_builtin_default_when_not_set(self):
        assert effective_default("default_profile") == "balanced"

    def test_returns_stored_value_over_default(self):
        set_value("default_profile", "quality")
        assert effective_default("default_profile") == "quality"

    def test_returns_none_default_when_not_set(self):
        assert effective_default("default_model") is None

    def test_returns_stored_model_when_set(self):
        set_value("default_model", "qwen3:8b")
        assert effective_default("default_model") == "qwen3:8b"

    def test_serve_port_default(self):
        assert effective_default("serve_port") == 8765
