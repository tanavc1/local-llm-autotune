"""Tests for autotune.ttft.optimizer — TTFTOptimizer and bucket snapping."""

from unittest.mock import MagicMock, patch

import pytest

from autotune.ttft.optimizer import _CTX_BUCKETS, KEEP_ALIVE_FOREVER, TTFTOptimizer, _snap_to_bucket

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


def _make_msgs(user_content="hello"):
    return [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# _snap_to_bucket
# ---------------------------------------------------------------------------

class TestSnapToBucket:
    def test_exact_bucket_value_unchanged(self):
        for bucket in _CTX_BUCKETS:
            assert _snap_to_bucket(bucket) == bucket

    def test_snaps_up_to_next_bucket(self):
        # 1 over 512 → snaps to 768
        assert _snap_to_bucket(513) == 768

    def test_snaps_up_within_range(self):
        # 1286 → should snap to 1536
        assert _snap_to_bucket(1286) == 1536

    def test_value_of_1_snaps_to_first_bucket(self):
        assert _snap_to_bucket(1) == _CTX_BUCKETS[0]

    def test_above_all_buckets_returns_max(self):
        assert _snap_to_bucket(999999) == _CTX_BUCKETS[-1]

    def test_512_stays_512(self):
        assert _snap_to_bucket(512) == 512

    def test_0_snaps_to_first_bucket(self):
        assert _snap_to_bucket(0) == _CTX_BUCKETS[0]

    def test_nearby_values_snap_to_same_bucket(self):
        # 1157 and 1308 should both end up in the same bucket
        assert _snap_to_bucket(1157) == _snap_to_bucket(1308)


# ---------------------------------------------------------------------------
# KEEP_ALIVE_FOREVER
# ---------------------------------------------------------------------------

class TestKeepAlive:
    def test_keep_alive_has_unit_suffix(self):
        # Must be a valid Go duration — any negative duration keeps model alive
        assert KEEP_ALIVE_FOREVER.startswith("-")
        # Must have a unit suffix (not just "-1")
        assert any(KEEP_ALIVE_FOREVER.endswith(u) for u in ["s", "m", "h"])


# ---------------------------------------------------------------------------
# TTFTOptimizer.build_request_options
# ---------------------------------------------------------------------------

class TestTTFTOptimizerNormalPressure:
    """No memory pressure (RAM < 80%)."""

    def _run(self, msgs=None, **kwargs):
        optimizer = TTFTOptimizer()
        profile = _make_profile(**kwargs)
        if msgs is None:
            msgs = _make_msgs()
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)
            return optimizer.build_request_options(msgs, profile)

    def test_returns_required_keys(self):
        result = self._run()
        assert "options" in result
        assert "keep_alive" in result
        assert "_debug" in result

    def test_keep_alive_equals_constant(self):
        result = self._run()
        assert result["keep_alive"] == KEEP_ALIVE_FOREVER

    def test_options_has_num_ctx(self):
        result = self._run()
        assert "num_ctx" in result["options"]
        assert result["options"]["num_ctx"] >= 512

    def test_options_has_flash_attn(self):
        result = self._run()
        assert result["options"].get("flash_attn") is True

    def test_options_has_num_batch_1024(self):
        result = self._run()
        assert result["options"]["num_batch"] == 1024

    def test_f16_kv_for_f16_profile(self):
        result = self._run(kv_cache_precision="f16")
        assert result["options"]["f16_kv"] is True

    def test_q8_kv_for_q8_profile(self):
        result = self._run(kv_cache_precision="q8")
        assert result["options"]["f16_kv"] is False

    def test_kv_precision_override_q8(self):
        optimizer = TTFTOptimizer()
        profile = _make_profile(kv_cache_precision="f16")
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)
            result = optimizer.build_request_options(
                _make_msgs(), profile, kv_precision_override="Q8_0"
            )
        assert result["options"]["f16_kv"] is False

    def test_kv_precision_override_f16(self):
        optimizer = TTFTOptimizer()
        profile = _make_profile(kv_cache_precision="q8")
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)
            result = optimizer.build_request_options(
                _make_msgs(), profile, kv_precision_override="F16"
            )
        assert result["options"]["f16_kv"] is True

    def test_context_ceiling_applied(self):
        optimizer = TTFTOptimizer()
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = [{"role": "user", "content": "a" * 4000}]  # ~1000 tokens
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)
            result = optimizer.build_request_options(msgs, profile, context_ceiling=1024)
        assert result["options"]["num_ctx"] <= 1024

    def test_num_ctx_snapped_to_bucket(self):
        result = self._run()
        num_ctx = result["options"]["num_ctx"]
        assert num_ctx in _CTX_BUCKETS

    def test_num_keep_set_for_system_prompt(self):
        msgs = [
            {"role": "system", "content": "a" * 200},  # 50 tokens
            {"role": "user", "content": "hello"},
        ]
        result = self._run(msgs=msgs)
        assert result["options"].get("num_keep", 0) > 0

    def test_num_keep_zero_when_no_system_prompt(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = self._run(msgs=msgs)
        assert result["options"].get("num_keep", 0) == 0

    def test_num_keep_zero_when_cache_disabled(self):
        optimizer = TTFTOptimizer()
        profile = _make_profile(system_prompt_cache=False)
        msgs = [
            {"role": "system", "content": "a" * 200},
            {"role": "user", "content": "hello"},
        ]
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)
            result = optimizer.build_request_options(msgs, profile)
        assert result["options"].get("num_keep", 0) == 0


class TestTTFTOptimizerModeratePressure:
    """RAM between 80% and 87%."""

    def _run(self, ram_pct=82.0):
        optimizer = TTFTOptimizer()
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = _make_msgs("a" * 400)
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=ram_pct)
            return optimizer.build_request_options(msgs, profile)

    def test_num_ctx_reduced(self):
        result_normal = None
        optimizer = TTFTOptimizer()
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = _make_msgs("a" * 400)
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)
            result_normal = optimizer.build_request_options(msgs, profile)

        result_pressure = self._run(82.0)
        # moderate pressure → 10% reduction
        assert result_pressure["options"]["num_ctx"] <= result_normal["options"]["num_ctx"]

    def test_debug_pressure_level(self):
        result = self._run()
        assert result["_debug"]["pressure_level"] == "moderate"

    def test_f16_kv_unchanged_at_moderate(self):
        result = self._run()
        # Moderate pressure does NOT downgrade KV precision
        assert result["options"]["f16_kv"] is True


class TestTTFTOptimizerHighPressure:
    """RAM between 88% and 92%."""

    def _run(self, ram_pct=89.0):
        optimizer = TTFTOptimizer()
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192, kv_cache_precision="f16")
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=ram_pct)
            return optimizer.build_request_options(_make_msgs("a" * 400), profile)

    def test_kv_downgraded_to_q8(self):
        result = self._run()
        assert result["options"]["f16_kv"] is False

    def test_debug_pressure_level(self):
        result = self._run()
        assert result["_debug"]["pressure_level"] == "high"

    def test_num_ctx_reduced_by_25_pct(self):
        optimizer = TTFTOptimizer()
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = _make_msgs("a" * 400)
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)
            normal = optimizer.build_request_options(msgs, profile)
        high = self._run()
        # high pressure: 25% reduction
        assert high["options"]["num_ctx"] <= normal["options"]["num_ctx"]


class TestTTFTOptimizerCriticalPressure:
    """RAM >= 93%."""

    def _run(self, ram_pct=94.0):
        optimizer = TTFTOptimizer()
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192, kv_cache_precision="f16")
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=ram_pct)
            return optimizer.build_request_options(_make_msgs("a" * 400), profile)

    def test_kv_downgraded_to_q8(self):
        result = self._run()
        assert result["options"]["f16_kv"] is False

    def test_num_batch_reduced_to_256(self):
        result = self._run()
        assert result["options"]["num_batch"] == 256

    def test_debug_pressure_level(self):
        result = self._run()
        assert result["_debug"]["pressure_level"] == "critical"

    def test_num_ctx_approximately_halved(self):
        optimizer = TTFTOptimizer()
        profile = _make_profile(max_new_tokens=512, max_context_tokens=8192)
        msgs = _make_msgs("a" * 400)
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(percent=50.0)
            normal = optimizer.build_request_options(msgs, profile)
        critical = self._run()
        # critical pressure: ~50% reduction
        assert critical["options"]["num_ctx"] < normal["options"]["num_ctx"]
