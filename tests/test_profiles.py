"""Tests for autotune.api.profiles — profile definitions and get_profile."""

import pytest

from autotune.api.profiles import PROFILES, get_profile, Profile


class TestProfileDefinitions:
    def test_all_three_profiles_exist(self):
        for name in ("fast", "balanced", "quality"):
            assert name in PROFILES

    def test_profiles_are_frozen_dataclasses(self):
        for p in PROFILES.values():
            assert isinstance(p, Profile)
            with pytest.raises((AttributeError, TypeError)):
                p.name = "modified"  # type: ignore

    def test_fast_is_lowest_latency(self):
        fast = PROFILES["fast"]
        balanced = PROFILES["balanced"]
        # Fast has shortest context and smallest max_new_tokens
        assert fast.max_context_tokens <= balanced.max_context_tokens
        assert fast.max_new_tokens <= balanced.max_new_tokens

    def test_quality_has_largest_context(self):
        quality = PROFILES["quality"]
        balanced = PROFILES["balanced"]
        assert quality.max_context_tokens >= balanced.max_context_tokens

    def test_fast_uses_q8_kv(self):
        assert PROFILES["fast"].kv_cache_precision == "q8"

    def test_balanced_and_quality_use_f16_kv(self):
        assert PROFILES["balanced"].kv_cache_precision == "f16"
        assert PROFILES["quality"].kv_cache_precision == "f16"

    def test_fast_uses_user_interactive_qos(self):
        assert PROFILES["fast"].qos_class == "USER_INTERACTIVE"

    def test_keep_alive_is_valid_go_duration(self):
        for p in PROFILES.values():
            ka = p.ollama_keep_alive
            assert ka.startswith("-")
            assert any(ka.endswith(u) for u in ["s", "m", "h"]), \
                f"keep_alive={ka!r} has no unit suffix — invalid Go duration"

    def test_all_profiles_have_system_prompt_cache_enabled(self):
        for p in PROFILES.values():
            assert p.system_prompt_cache is True

    def test_fast_has_repetition_penalty_above_1(self):
        # Fast profile uses higher repetition_penalty to prevent loops
        assert PROFILES["fast"].repetition_penalty > 1.0

    def test_temperature_ordering(self):
        # fast < balanced <= quality
        assert PROFILES["fast"].temperature < PROFILES["balanced"].temperature

    def test_mlx_is_first_backend_preference(self):
        for p in PROFILES.values():
            assert p.backend_preference[0] == "mlx"

    def test_timeout_ordering(self):
        # fast < balanced <= quality
        assert PROFILES["fast"].request_timeout_sec <= PROFILES["balanced"].request_timeout_sec
        assert PROFILES["balanced"].request_timeout_sec <= PROFILES["quality"].request_timeout_sec


class TestGetProfile:
    def test_returns_correct_profile(self):
        for name in ("fast", "balanced", "quality"):
            p = get_profile(name)
            assert p.name == name

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("turbo")

    def test_error_message_lists_valid_options(self):
        with pytest.raises(ValueError) as exc_info:
            get_profile("invalid")
        msg = str(exc_info.value)
        assert "fast" in msg or "balanced" in msg or "quality" in msg
