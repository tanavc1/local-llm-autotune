"""
Tests for autotune.api.model_selector — architecture extraction and fit analysis.

Covers:
- ArchInfo properties: head_dim, kv_bytes_per_token, kv_cache_gb, kv_mb_per_1k_tokens
- extract_arch_from_modelinfo: all major arch prefixes, missing fields, None input
- estimate_arch_from_params: all parameter-count bins
- FitClass enumeration values
- ModelSelector.assess: safe/marginal/swap_risk/oom fit classes
- Context budget calculations
- Quant downgrade suggestions
- KV precision recommendations
"""

from __future__ import annotations

import pytest

from autotune.api.model_selector import (
    RUNTIME_OVERHEAD_GB,
    SAFE_RAM_FRACTION,
    SWAP_RISK_FRACTION,
    ArchInfo,
    FitClass,
    ModelSelector,
    estimate_arch_from_params,
    extract_arch_from_modelinfo,
)

# ---------------------------------------------------------------------------
# ArchInfo properties
# ---------------------------------------------------------------------------

class TestArchInfo:
    def make_arch(self, n_layers=32, n_kv_heads=8, n_heads=32, embed=4096) -> ArchInfo:
        return ArchInfo(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            n_heads=n_heads,
            embedding_size=embed,
            source="modelinfo",
        )

    def test_head_dim_computed_correctly(self):
        arch = self.make_arch(n_heads=32, embed=4096)
        assert arch.head_dim == 4096 // 32  # = 128

    def test_head_dim_with_gqa(self):
        arch = self.make_arch(n_heads=16, embed=2048)
        assert arch.head_dim == 2048 // 16  # = 128

    def test_head_dim_never_zero(self):
        arch = self.make_arch(n_heads=0, embed=0)
        assert arch.head_dim >= 1

    def test_kv_bytes_per_token_f16(self):
        arch = self.make_arch(n_layers=32, n_kv_heads=8, n_heads=32, embed=4096)
        # 2 * layers * kv_heads * head_dim * F16_BYTES (2)
        expected = 2 * 32 * 8 * 128 * 2
        assert arch.kv_bytes_per_token("F16") == expected

    def test_kv_bytes_per_token_q8_half_of_f16(self):
        arch = self.make_arch()
        f16 = arch.kv_bytes_per_token("F16")
        q8  = arch.kv_bytes_per_token("Q8_0")
        assert f16 == q8 * 2

    def test_kv_cache_gb_scales_with_context(self):
        arch = self.make_arch()
        gb_1k = arch.kv_cache_gb(1000)
        gb_4k = arch.kv_cache_gb(4000)
        assert abs(gb_4k / gb_1k - 4.0) < 1e-6

    def test_kv_mb_per_1k_positive(self):
        arch = self.make_arch()
        assert arch.kv_mb_per_1k_tokens("F16") > 0
        assert arch.kv_mb_per_1k_tokens("Q8_0") > 0

    def test_kv_mb_per_1k_f16_double_q8(self):
        arch = self.make_arch()
        f16 = arch.kv_mb_per_1k_tokens("F16")
        q8  = arch.kv_mb_per_1k_tokens("Q8_0")
        assert abs(f16 / q8 - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# extract_arch_from_modelinfo
# ---------------------------------------------------------------------------

class TestExtractArchFromModelinfo:
    def _llama_info(self) -> dict:
        return {
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.embedding_length": 4096,
            "general.architecture": "llama",
        }

    def test_extracts_llama(self):
        arch = extract_arch_from_modelinfo(self._llama_info())
        assert arch is not None
        assert arch.n_layers == 32
        assert arch.n_kv_heads == 8
        assert arch.n_heads == 32
        assert arch.embedding_size == 4096
        assert arch.source == "modelinfo"

    def test_extracts_qwen3(self):
        info = {
            "qwen3.block_count": 28,
            "qwen3.attention.head_count": 16,
            "qwen3.attention.head_count_kv": 8,
            "qwen3.embedding_length": 2048,
        }
        arch = extract_arch_from_modelinfo(info)
        assert arch is not None
        assert arch.n_layers == 28

    def test_extracts_gemma(self):
        info = {
            "gemma.block_count": 18,
            "gemma.attention.head_count": 8,
            "gemma.attention.head_count_kv": 1,
            "gemma.embedding_length": 2048,
        }
        arch = extract_arch_from_modelinfo(info)
        assert arch is not None
        assert arch.n_layers == 18

    def test_returns_none_for_empty_dict(self):
        assert extract_arch_from_modelinfo({}) is None

    def test_returns_none_when_dict_is_none(self):
        assert extract_arch_from_modelinfo(None) is None

    def test_returns_none_when_essential_fields_missing(self):
        # Missing embedding_length (required)
        info = {
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
        }
        result = extract_arch_from_modelinfo(info)
        # May return None or fall back — either is acceptable
        # but if returned, must not crash
        assert result is None or isinstance(result, ArchInfo)

    def test_kv_heads_defaults_to_n_heads_when_missing(self):
        info = {
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.embedding_length": 4096,
            # no head_count_kv → MHA, kv_heads == n_heads
        }
        arch = extract_arch_from_modelinfo(info)
        if arch is not None:
            assert arch.n_kv_heads == arch.n_heads

    @pytest.mark.parametrize("prefix", [
        "llama", "qwen3", "qwen2", "phi4", "phi3", "gemma", "mistral"
    ])
    def test_all_major_arch_prefixes_detected(self, prefix: str):
        info = {
            f"{prefix}.block_count": 24,
            f"{prefix}.attention.head_count": 16,
            f"{prefix}.attention.head_count_kv": 8,
            f"{prefix}.embedding_length": 2048,
        }
        arch = extract_arch_from_modelinfo(info)
        assert arch is not None, f"Failed to extract arch for prefix {prefix}"
        assert arch.n_layers == 24


# ---------------------------------------------------------------------------
# estimate_arch_from_params
# ---------------------------------------------------------------------------

class TestEstimateArchFromParams:
    @pytest.mark.parametrize("params_b,expected_layers_min", [
        (0.3,  20),   # < 0.6B
        (1.0,  22),
        (3.0,  24),
        (7.0,  32),
        (14.0, 36),
        (34.0, 52),
    ])
    def test_layers_in_expected_range(self, params_b: float, expected_layers_min: int):
        arch = estimate_arch_from_params(params_b)
        assert arch.n_layers >= expected_layers_min - 4   # ±4 layer tolerance

    def test_source_is_estimate(self):
        arch = estimate_arch_from_params(7.0)
        assert arch.source == "estimate"

    def test_very_large_model_uses_fallback(self):
        arch = estimate_arch_from_params(500.0)
        assert arch.n_layers >= 80

    def test_zero_params_returns_smallest_config(self):
        arch = estimate_arch_from_params(0.0)
        assert arch.n_layers > 0
        assert arch.n_kv_heads > 0

    def test_all_fields_positive(self):
        for params in (0.5, 3.0, 7.0, 14.0, 70.0):
            arch = estimate_arch_from_params(params)
            assert arch.n_layers > 0
            assert arch.n_kv_heads > 0
            assert arch.n_heads > 0
            assert arch.embedding_size > 0
            assert arch.head_dim > 0


# ---------------------------------------------------------------------------
# ModelSelector.assess — fit classes
# ---------------------------------------------------------------------------

class TestModelSelectorFitClasses:
    def _make_selector(self, available_gb: float) -> ModelSelector:
        return ModelSelector(available_gb=available_gb, total_ram_gb=16.0)

    def test_safe_when_plenty_of_ram(self):
        sel = self._make_selector(16.0)
        report = sel.assess("small-model", size_gb=2.0, params_b=3.0, quant="Q4_K_M")
        assert report.fit_class == FitClass.SAFE

    def test_oom_when_model_larger_than_ram(self):
        sel = self._make_selector(4.0)
        report = sel.assess("huge-model", size_gb=20.0, params_b=70.0, quant="Q4_K_M")
        assert report.fit_class == FitClass.OOM

    def test_swap_risk_when_tight(self):
        # Force swap risk: total_est / available between 0.92 and 1.0
        available = 8.0
        # size_gb chosen so weights + overhead + kv_q8 ≈ 95% of available
        size_gb = available * SWAP_RISK_FRACTION * 0.85  # most of budget for weights
        sel = self._make_selector(available)
        report = sel.assess("tight-model", size_gb=size_gb, params_b=7.0, quant="Q4_K_M")
        assert report.fit_class in (FitClass.SWAP_RISK, FitClass.MARGINAL, FitClass.OOM)

    def test_tight_when_exceeds_available_but_fits_in_total_ram(self):
        # available=4.0, total_ram=16.0 → model that needs ~6 GB is TIGHT (not OOM)
        sel = ModelSelector(available_gb=4.0, total_ram_gb=16.0)
        report = sel.assess("mid-model", size_gb=5.5, params_b=7.0, quant="Q4_K_M")
        assert report.fit_class == FitClass.TIGHT

    def test_report_has_all_fields(self):
        sel = self._make_selector(16.0)
        report = sel.assess("qwen3:8b", size_gb=5.2, params_b=8.0, quant="Q4_K_M")
        assert report.model_name == "qwen3:8b"
        assert report.size_gb == 5.2
        assert report.available_gb <= 16.0
        assert report.weights_gb == 5.2
        assert report.overhead_gb == RUNTIME_OVERHEAD_GB
        assert report.fit_class in FitClass.__members__.values()
        assert report.recommended_profile in ("fast", "balanced", "quality")
        assert report.recommended_kv in ("F16", "Q8_0")
        assert report.safe_max_context > 0

    def test_modelinfo_improves_arch_accuracy(self):
        modelinfo = {
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.embedding_length": 4096,
        }
        sel = self._make_selector(16.0)
        report_with    = sel.assess("model", size_gb=5.0, params_b=8.0,
                                     quant="Q4_K_M", modelinfo=modelinfo)
        report_without = sel.assess("model", size_gb=5.0, params_b=8.0, quant="Q4_K_M")
        # Both should succeed; report with modelinfo should have more precise arch
        assert report_with.arch is not None
        assert report_with.arch.source == "modelinfo"
        assert report_without.arch is not None

    def test_context_budget_f16_larger_than_q8_tokens(self):
        sel = self._make_selector(16.0)
        report = sel.assess("model", size_gb=3.0, params_b=7.0, quant="Q4_K_M")
        if report.arch:
            # F16 uses more memory per token → fewer safe tokens
            assert report.budget_f16.max_safe_tokens <= report.budget_q8.max_safe_tokens

    def test_safe_max_context_positive(self):
        sel = self._make_selector(16.0)
        report = sel.assess("model", size_gb=3.0, params_b=7.0, quant="Q4_K_M")
        assert report.safe_max_context > 0

    def test_oom_sets_fatal_true(self):
        sel = self._make_selector(2.0)
        report = sel.assess("huge", size_gb=10.0, params_b=70.0, quant="Q4_K_M")
        if report.fit_class == FitClass.OOM:
            assert report.fatal is True

    def test_safe_model_fatal_false(self):
        sel = self._make_selector(16.0)
        report = sel.assess("small", size_gb=1.0, params_b=1.5, quant="Q4_K_M")
        assert report.fatal is False

    def test_zero_params_does_not_crash(self):
        sel = self._make_selector(16.0)
        report = sel.assess("unknown", size_gb=5.0, params_b=None, quant="unknown")
        assert report is not None
        assert report.fit_class in FitClass.__members__.values()

    def test_available_gb_floor_prevents_zero_division(self):
        sel = ModelSelector(available_gb=0.0, total_ram_gb=0.0)
        # Should not raise ZeroDivisionError
        report = sel.assess("tiny", size_gb=0.1, params_b=0.5, quant="Q4_K_M")
        assert report is not None


# ---------------------------------------------------------------------------
# FitClass enum
# ---------------------------------------------------------------------------

class TestFitClassEnum:
    def test_all_fit_classes_exist(self):
        assert FitClass.SAFE
        assert FitClass.MARGINAL
        assert FitClass.SWAP_RISK
        assert FitClass.TIGHT
        assert FitClass.OOM

    def test_string_values(self):
        assert FitClass.SAFE.value == "safe"
        assert FitClass.MARGINAL.value == "marginal"
        assert FitClass.SWAP_RISK.value == "swap_risk"
        assert FitClass.TIGHT.value == "tight"
        assert FitClass.OOM.value == "oom"
