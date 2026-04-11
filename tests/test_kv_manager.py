"""Tests for autotune.api.kv_manager — build_ollama_options and pressure tiers."""

import pytest
from unittest.mock import MagicMock, patch

from autotune.api.kv_manager import (
    build_ollama_options,
    compute_num_keep,
    memory_pressure_snapshot,
    kv_memory_estimate_mb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    max_new_tokens=512,
    max_context_tokens=8192,
    kv_cache_precision="f16",
    system_prompt_cache=True,
):
    p = MagicMock()
    p.max_new_tokens = max_new_tokens
    p.max_context_tokens = max_context_tokens
    p.kv_cache_precision = kv_cache_precision
    p.system_prompt_cache = system_prompt_cache
    return p


def _msgs(user_content="hello"):
    return [{"role": "user", "content": user_content}]


def _build(msgs=None, ram_pct=50.0, profile=None, **kwargs):
    if msgs is None:
        msgs = _msgs()
    if profile is None:
        profile = _make_profile()
    with patch("psutil.virtual_memory") as vm:
        vm.return_value = MagicMock(percent=ram_pct)
        return build_ollama_options(msgs, profile, **kwargs)


# ---------------------------------------------------------------------------
# compute_num_keep
# ---------------------------------------------------------------------------

class TestComputeNumKeep:
    def test_zero_when_cache_disabled(self):
        profile = _make_profile(system_prompt_cache=False)
        msgs = [{"role": "system", "content": "a" * 200}, {"role": "user", "content": "hi"}]
        assert compute_num_keep(msgs, profile) == 0

    def test_counts_leading_system_messages(self):
        profile = _make_profile(system_prompt_cache=True)
        msgs = [{"role": "system", "content": "a" * 200}, {"role": "user", "content": "hi"}]
        result = compute_num_keep(msgs, profile)
        assert result > 0

    def test_stops_at_first_non_system(self):
        profile = _make_profile(system_prompt_cache=True)
        msgs = [
            {"role": "system", "content": "a" * 200},
            {"role": "user", "content": "b" * 200},  # should stop here
            {"role": "system", "content": "c" * 200},  # should NOT be counted
        ]
        only_first = compute_num_keep(msgs, profile)
        # Only the first system message contributes
        profile2 = _make_profile(system_prompt_cache=True)
        only_one = compute_num_keep([{"role": "system", "content": "a" * 200}], profile2)
        assert only_first == only_one

    def test_no_system_message_returns_zero(self):
        profile = _make_profile(system_prompt_cache=True)
        assert compute_num_keep(_msgs(), profile) == 0


# ---------------------------------------------------------------------------
# build_ollama_options — normal pressure
# ---------------------------------------------------------------------------

class TestBuildOllamaOptionsNormal:
    def test_returns_tuple(self):
        opts, notices = _build()
        assert isinstance(opts, dict)
        assert isinstance(notices, list)

    def test_has_flash_attn(self):
        opts, _ = _build()
        assert opts.get("flash_attn") is True

    def test_has_num_batch_1024(self):
        opts, _ = _build()
        assert opts["num_batch"] == 1024

    def test_num_ctx_present(self):
        opts, _ = _build()
        assert "num_ctx" in opts
        assert opts["num_ctx"] >= 512

    def test_f16_kv_true_for_f16_profile(self):
        opts, _ = _build(profile=_make_profile(kv_cache_precision="f16"))
        assert opts["f16_kv"] is True

    def test_f16_kv_false_for_q8_profile(self):
        opts, _ = _build(profile=_make_profile(kv_cache_precision="q8"))
        assert opts["f16_kv"] is False

    def test_no_notices_at_normal_pressure(self):
        _, notices = _build(ram_pct=50.0)
        assert notices == []

    def test_context_ceiling_applied(self):
        opts, _ = _build(
            msgs=[{"role": "user", "content": "a" * 4000}],
            context_ceiling=1024,
        )
        assert opts["num_ctx"] <= 1024

    def test_kv_precision_override_q8(self):
        opts, _ = _build(
            profile=_make_profile(kv_cache_precision="f16"),
            kv_precision_override="Q8_0",
        )
        assert opts["f16_kv"] is False

    def test_kv_precision_override_f16(self):
        opts, _ = _build(
            profile=_make_profile(kv_cache_precision="q8"),
            kv_precision_override="F16",
        )
        assert opts["f16_kv"] is True

    def test_num_keep_set_with_system_prompt(self):
        msgs = [
            {"role": "system", "content": "a" * 200},
            {"role": "user", "content": "hello"},
        ]
        opts, _ = _build(msgs=msgs)
        assert opts.get("num_keep", 0) > 0

    def test_prompt_caching_override(self):
        # Override forces prefix caching even if profile has system_prompt_cache=False
        msgs = [
            {"role": "system", "content": "a" * 200},
            {"role": "user", "content": "hello"},
        ]
        opts, _ = _build(
            msgs=msgs,
            profile=_make_profile(system_prompt_cache=False),
            prompt_caching_override=True,
        )
        assert opts.get("num_keep", 0) > 0


# ---------------------------------------------------------------------------
# build_ollama_options — pressure tiers
# ---------------------------------------------------------------------------

class TestBuildOllamaOptionsModerate:
    def test_notice_emitted(self):
        opts, notices = _build(
            msgs=[{"role": "user", "content": "a" * 800}],
            ram_pct=82.0,
        )
        assert len(notices) > 0
        assert "RAM" in notices[0]

    def test_num_ctx_reduced(self):
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = [{"role": "user", "content": "a" * 800}]
        opts_normal, _ = _build(msgs=msgs, ram_pct=50.0, profile=profile)
        opts_pressure, _ = _build(msgs=msgs, ram_pct=82.0, profile=profile)
        assert opts_pressure["num_ctx"] <= opts_normal["num_ctx"]


class TestBuildOllamaOptionsHigh:
    def test_kv_downgraded(self):
        opts, notices = _build(
            profile=_make_profile(kv_cache_precision="f16"),
            ram_pct=89.0,
        )
        assert opts["f16_kv"] is False
        assert any("KV" in n for n in notices)

    def test_notice_contains_ram_pct(self):
        _, notices = _build(ram_pct=89.0)
        assert any("89" in n for n in notices)


class TestBuildOllamaOptionsCritical:
    def test_kv_downgraded(self):
        opts, _ = _build(
            profile=_make_profile(kv_cache_precision="f16"),
            ram_pct=94.0,
        )
        assert opts["f16_kv"] is False

    def test_num_batch_reduced(self):
        opts, _ = _build(ram_pct=94.0)
        assert opts["num_batch"] == 256

    def test_ctx_approximately_halved(self):
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = [{"role": "user", "content": "a" * 800}]
        opts_normal, _ = _build(msgs=msgs, ram_pct=50.0, profile=profile)
        opts_critical, _ = _build(msgs=msgs, ram_pct=94.0, profile=profile)
        assert opts_critical["num_ctx"] < opts_normal["num_ctx"]

    def test_notice_contains_critical(self):
        _, notices = _build(ram_pct=94.0)
        assert any("critical" in n.lower() for n in notices)


# ---------------------------------------------------------------------------
# memory_pressure_snapshot
# ---------------------------------------------------------------------------

class TestMemoryPressureSnapshot:
    def test_normal_pressure(self):
        with patch("psutil.virtual_memory") as vm, patch("psutil.swap_memory") as sw:
            vm.return_value = MagicMock(percent=50.0, available=8 * 1024**3)
            sw.return_value = MagicMock(percent=0.0, used=0)
            snap = memory_pressure_snapshot()
        assert snap["pressure_level"] == "normal"
        assert snap["ram_pct"] == 50.0

    def test_moderate_pressure(self):
        with patch("psutil.virtual_memory") as vm, patch("psutil.swap_memory") as sw:
            vm.return_value = MagicMock(percent=82.0, available=2 * 1024**3)
            sw.return_value = MagicMock(percent=5.0, used=512 * 1024**2)
            snap = memory_pressure_snapshot()
        assert snap["pressure_level"] == "moderate"

    def test_high_pressure(self):
        with patch("psutil.virtual_memory") as vm, patch("psutil.swap_memory") as sw:
            vm.return_value = MagicMock(percent=89.0, available=1 * 1024**3)
            sw.return_value = MagicMock(percent=20.0, used=2 * 1024**3)
            snap = memory_pressure_snapshot()
        assert snap["pressure_level"] == "high"

    def test_critical_pressure(self):
        with patch("psutil.virtual_memory") as vm, patch("psutil.swap_memory") as sw:
            vm.return_value = MagicMock(percent=94.0, available=500 * 1024**2)
            sw.return_value = MagicMock(percent=40.0, used=4 * 1024**3)
            snap = memory_pressure_snapshot()
        assert snap["pressure_level"] == "critical"

    def test_snapshot_keys(self):
        with patch("psutil.virtual_memory") as vm, patch("psutil.swap_memory") as sw:
            vm.return_value = MagicMock(percent=50.0, available=8 * 1024**3)
            sw.return_value = MagicMock(percent=0.0, used=0)
            snap = memory_pressure_snapshot()
        for key in ("ram_pct", "swap_pct", "available_gb", "swap_used_gb", "pressure_level"):
            assert key in snap


# ---------------------------------------------------------------------------
# kv_memory_estimate_mb
# ---------------------------------------------------------------------------

class TestKvMemoryEstimateMb:
    def test_f16_is_double_q8(self):
        # F16 = 2 bytes/elem, Q8 = 1 byte/elem
        f16 = kv_memory_estimate_mb(2048, 32, 8, 128, f16_kv=True)
        q8 = kv_memory_estimate_mb(2048, 32, 8, 128, f16_kv=False)
        assert abs(f16 / q8 - 2.0) < 0.001

    def test_scales_linearly_with_num_ctx(self):
        small = kv_memory_estimate_mb(1024, 32, 8, 128)
        large = kv_memory_estimate_mb(2048, 32, 8, 128)
        assert abs(large / small - 2.0) < 0.001

    def test_positive_result(self):
        result = kv_memory_estimate_mb(4096, 48, 8, 128, f16_kv=True)
        assert result > 0
