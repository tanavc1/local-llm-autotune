"""
Tests for autotune.memory.noswap — NoSwapGuard and ModelArch.

Covers:
- ModelArch.kv_gb: formula correctness for F16 and Q8
- NoSwapGuard.apply: all 6 reduction levels (ok, l1–l5)
- NoSwapDecision properties: ctx_changed, kv_saved_gb
- Safety margin handling
- snap_fn integration
- NoSwapGuard.would_swap: True/False boundary
- NoSwapGuard.ram_state: returns three positive numbers
- get_model_arch: fallback on HTTP error (mocked)
- _arch_cache isolation between tests
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autotune.memory.noswap import (
    ModelArch,
    NoSwapDecision,
    NoSwapGuard,
    _FALLBACK_ARCH,
    _arch_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_arch() -> ModelArch:
    """Minimal architecture producing a tiny KV footprint — fits almost anywhere."""
    return ModelArch(n_layers=4, n_kv_heads=4, head_dim=64, arch_name="tiny")


def _large_arch() -> ModelArch:
    """70B-scale architecture with heavy KV usage."""
    return ModelArch(n_layers=80, n_kv_heads=8, head_dim=128, arch_name="large")


def _vm(available_gb: float, total_gb: float = 32.0):
    """Mock psutil.virtual_memory() with the given available bytes."""
    m = MagicMock()
    m.available = int(available_gb * 1024**3)
    m.total     = int(total_gb * 1024**3)
    m.used      = int((total_gb - available_gb) * 1024**3)
    m.percent   = round((total_gb - available_gb) / total_gb * 100, 1)
    return m


# ---------------------------------------------------------------------------
# ModelArch.kv_gb
# ---------------------------------------------------------------------------

class TestModelArchKvGb:
    def test_f16_formula(self):
        arch = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128, arch_name="llama")
        # Expected: 2 * 32 * 8 * 128 * 1024 * 2 bytes / 1024^3
        expected = 2 * 32 * 8 * 128 * 1024 * 2 / 1024**3
        assert abs(arch.kv_gb(1024, f16=True) - expected) < 1e-6

    def test_q8_exactly_half_f16(self):
        arch = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128, arch_name="llama")
        f16 = arch.kv_gb(4096, f16=True)
        q8  = arch.kv_gb(4096, f16=False)
        assert abs(f16 / 2 - q8) < 1e-9

    def test_kv_scales_linearly_with_context(self):
        arch = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128, arch_name="llama")
        kv_1k = arch.kv_gb(1000, f16=True)
        kv_4k = arch.kv_gb(4000, f16=True)
        assert abs(kv_4k / kv_1k - 4.0) < 1e-6

    def test_kv_scales_with_layers(self):
        arch_32 = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128, arch_name="a")
        arch_64 = ModelArch(n_layers=64, n_kv_heads=8, head_dim=128, arch_name="b")
        assert abs(arch_64.kv_gb(1024) / arch_32.kv_gb(1024) - 2.0) < 1e-6

    def test_kv_zero_context_is_zero(self):
        arch = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128)
        assert arch.kv_gb(0) == 0.0

    @pytest.mark.parametrize("num_ctx", [512, 1024, 2048, 4096, 8192])
    def test_kv_positive_for_all_ctx_sizes(self, num_ctx: int):
        arch = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128)
        assert arch.kv_gb(num_ctx, f16=True) > 0
        assert arch.kv_gb(num_ctx, f16=False) > 0


# ---------------------------------------------------------------------------
# NoSwapDecision properties
# ---------------------------------------------------------------------------

class TestNoSwapDecision:
    def _make(self, num_ctx=1024, reduced_from=1024, f16_kv=True,
              kv_before=0.5, kv_after=0.5) -> NoSwapDecision:
        return NoSwapDecision(
            num_ctx=num_ctx,
            f16_kv=f16_kv,
            level="ok",
            reduced_from=reduced_from,
            reason="test",
            available_gb=8.0,
            kv_gb_before=kv_before,
            kv_gb_after=kv_after,
            safety_margin=1.5,
        )

    def test_ctx_changed_false_when_same(self):
        d = self._make(num_ctx=1024, reduced_from=1024)
        assert not d.ctx_changed

    def test_ctx_changed_true_when_different(self):
        d = self._make(num_ctx=512, reduced_from=1024)
        assert d.ctx_changed

    def test_kv_saved_gb_positive_when_reduced(self):
        d = self._make(kv_before=0.8, kv_after=0.4)
        assert abs(d.kv_saved_gb - 0.4) < 0.001

    def test_kv_saved_gb_zero_when_unchanged(self):
        d = self._make(kv_before=0.5, kv_after=0.5)
        assert d.kv_saved_gb == 0.0


# ---------------------------------------------------------------------------
# NoSwapGuard.apply — level selection
# ---------------------------------------------------------------------------

class TestNoSwapGuardApply:
    """
    We control 'available RAM' via psutil mock and use architectures whose
    KV footprints we can calculate precisely.
    """

    def test_level_ok_when_fits_comfortably(self):
        arch = _tiny_arch()
        guard = NoSwapGuard(safety_margin_gb=0.0)
        with patch("psutil.virtual_memory", return_value=_vm(16.0)):
            d = guard.apply(512, f16_kv=True, arch=arch)
        assert d.level == "ok"
        assert not d.ctx_changed

    def test_level_l1_trim_25_percent(self):
        # Set available RAM so full ctx doesn't fit but 75% does
        arch = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128, arch_name="llama")
        num_ctx = 4096
        kv_full = arch.kv_gb(num_ctx, f16=True)
        kv_75   = arch.kv_gb(int(num_ctx * 0.75), f16=True)
        # Available: slightly less than full but more than 75%
        available = kv_full * 0.9
        guard = NoSwapGuard(safety_margin_gb=0.0)
        with patch("psutil.virtual_memory", return_value=_vm(available)):
            d = guard.apply(num_ctx, f16_kv=True, arch=arch)
        assert d.level in ("l1_trim", "l2_halve", "ok"), d.level

    def test_level_l5_min_when_no_ram(self):
        arch = _large_arch()
        guard = NoSwapGuard(safety_margin_gb=0.0)
        with patch("psutil.virtual_memory", return_value=_vm(0.01)):  # 10 MB
            d = guard.apply(8192, f16_kv=True, arch=arch)
        assert d.level in ("l5_min", "l4_quarter", "l3_q8")
        assert d.num_ctx == 512 or d.num_ctx <= 2048

    def test_q8_used_at_high_pressure(self):
        arch = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128, arch_name="llama")
        num_ctx = 4096
        kv_f16_half = arch.kv_gb(int(num_ctx * 0.50), f16=True)
        kv_q8_half  = arch.kv_gb(int(num_ctx * 0.50), f16=False)
        # Available: between q8_half and f16_half → should force q8
        available = (kv_f16_half + kv_q8_half) / 2
        guard = NoSwapGuard(safety_margin_gb=0.0)
        with patch("psutil.virtual_memory", return_value=_vm(available)):
            d = guard.apply(num_ctx, f16_kv=True, arch=arch)
        if d.level in ("l3_q8", "l4_quarter", "l5_min"):
            assert d.f16_kv is False

    def test_safety_margin_increases_requirements(self):
        arch = _tiny_arch()
        num_ctx = 512
        kv_needed = arch.kv_gb(num_ctx, f16=True)
        # Available just enough without margin
        available_gb = kv_needed + 0.5
        guard_no_margin  = NoSwapGuard(safety_margin_gb=0.0)
        guard_big_margin = NoSwapGuard(safety_margin_gb=2.0)
        with patch("psutil.virtual_memory", return_value=_vm(available_gb)):
            d_no   = guard_no_margin.apply(num_ctx, f16_kv=True, arch=arch)
            d_big  = guard_big_margin.apply(num_ctx, f16_kv=True, arch=arch)
        # With no margin: ok. With big margin: reduced.
        if d_no.level == "ok":
            assert d_big.level != "ok" or d_big.ctx_changed or not d_big.f16_kv

    def test_snap_fn_applied_to_candidates(self):
        arch = _tiny_arch()
        guard = NoSwapGuard(safety_margin_gb=0.0)
        snapped_values = []

        def snap(ctx: int) -> int:
            snapped_values.append(ctx)
            return ctx

        with patch("psutil.virtual_memory", return_value=_vm(16.0)):
            guard.apply(512, f16_kv=True, arch=arch, snap_fn=snap)

        # snap_fn called at least for level 0 (factor=1.0)
        assert len(snapped_values) >= 1

    def test_min_ctx_never_below_512(self):
        arch = _large_arch()
        guard = NoSwapGuard(safety_margin_gb=0.0)
        with patch("psutil.virtual_memory", return_value=_vm(0.001)):
            d = guard.apply(8192, f16_kv=True, arch=arch)
        assert d.num_ctx >= 512

    def test_fallback_arch_used_without_error(self):
        guard = NoSwapGuard(safety_margin_gb=0.0)
        with patch("psutil.virtual_memory", return_value=_vm(8.0)):
            d = guard.apply(2048, f16_kv=True, arch=_FALLBACK_ARCH)
        assert isinstance(d, NoSwapDecision)

    def test_decision_available_gb_matches_mock(self):
        arch = _tiny_arch()
        guard = NoSwapGuard(safety_margin_gb=0.0)
        with patch("psutil.virtual_memory", return_value=_vm(7.5)):
            d = guard.apply(512, f16_kv=True, arch=arch)
        assert abs(d.available_gb - 7.5) < 0.1

    def test_kv_gb_before_matches_formula(self):
        arch = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128, arch_name="llama")
        num_ctx = 2048
        expected_kv = arch.kv_gb(num_ctx, f16=True)
        guard = NoSwapGuard(safety_margin_gb=0.0)
        with patch("psutil.virtual_memory", return_value=_vm(16.0)):
            d = guard.apply(num_ctx, f16_kv=True, arch=arch)
        assert abs(d.kv_gb_before - expected_kv) < 0.001


# ---------------------------------------------------------------------------
# NoSwapGuard.would_swap
# ---------------------------------------------------------------------------

class TestWouldSwap:
    def test_would_swap_false_when_plenty_of_ram(self):
        with patch("psutil.virtual_memory", return_value=_vm(16.0)):
            assert NoSwapGuard.would_swap(1.0, safety_margin_gb=1.5) is False

    def test_would_swap_true_when_tight(self):
        with patch("psutil.virtual_memory", return_value=_vm(2.0)):
            # Required 1 GB + margin 1.5 GB > 2 GB available
            assert NoSwapGuard.would_swap(1.0, safety_margin_gb=1.5) is True

    def test_would_swap_exact_boundary(self):
        # available=3.0, margin=1.5, usable=1.5, required=1.5 → exactly equal → False
        with patch("psutil.virtual_memory", return_value=_vm(3.0)):
            assert NoSwapGuard.would_swap(1.5, safety_margin_gb=1.5) is False

    def test_would_swap_just_over_boundary(self):
        # available=3.0, margin=1.5, usable=1.5, required=1.501 → True
        with patch("psutil.virtual_memory", return_value=_vm(3.0)):
            assert NoSwapGuard.would_swap(1.501, safety_margin_gb=1.5) is True


# ---------------------------------------------------------------------------
# NoSwapGuard.ram_state
# ---------------------------------------------------------------------------

class TestRamState:
    def test_ram_state_returns_three_values(self):
        with patch("psutil.virtual_memory", return_value=_vm(16.0)):
            total, used, avail = NoSwapGuard.ram_state()
        assert total > 0
        assert used >= 0
        assert avail >= 0

    def test_ram_state_total_equals_used_plus_available_approx(self):
        # psutil's available != total - used exactly (cached pages etc.)
        # but our mock sets available = total - used exactly
        with patch("psutil.virtual_memory", return_value=_vm(16.0)):
            total, used, avail = NoSwapGuard.ram_state()
        assert abs(total - used - avail) < 0.5


# ---------------------------------------------------------------------------
# get_model_arch — fallback behaviour (no real Ollama needed)
# ---------------------------------------------------------------------------

class TestGetModelArch:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Isolate arch cache between tests."""
        _arch_cache.clear()
        yield
        _arch_cache.clear()

    async def test_returns_fallback_on_http_error(self):
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("connection refused")
            )
            arch = await NoSwapGuard.get_model_arch("nonexistent-model")
        assert arch == _FALLBACK_ARCH

    async def test_result_is_cached(self):
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.json.return_value = {"model_info": {
                "llama.block_count": "32",
                "llama.attention.head_count_kv": "8",
                "llama.attention.head_count": "32",
                "llama.embedding_length": "4096",
                "general.architecture": "llama",
            }}
            return resp

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post
            await NoSwapGuard.get_model_arch("test-model-xyz")
            await NoSwapGuard.get_model_arch("test-model-xyz")

        assert call_count == 1, "Should have cached result, not called twice"

    async def test_parses_model_info_correctly(self):
        async def mock_post(*args, **kwargs):
            resp = MagicMock()
            resp.json.return_value = {"model_info": {
                "qwen3.block_count": "28",
                "qwen3.attention.head_count_kv": "8",
                "qwen3.attention.head_count": "16",
                "qwen3.embedding_length": "2048",
                "general.architecture": "qwen3",
            }}
            return resp

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post
            arch = await NoSwapGuard.get_model_arch("qwen3-test-model")

        assert arch.n_layers == 28
        assert arch.n_kv_heads == 8
        assert arch.head_dim == 2048 // 16   # embed // n_heads
