"""
Tests for autotune.api.model_guard — feasibility guard for model downloads.

Covers:
- Known model size lookup
- Parameter-count parsing (various name formats)
- RAM estimation
- BLOCKED verdicts (disk and RAM)
- WARN verdicts (disk and RAM)
- OK verdict
- Unknown-size fallback (verdict always "ok")
- FeasibilityResult.to_dict() serialisation
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autotune.api.model_guard import (
    FeasibilityResult,
    _KNOWN_SIZES,
    check_feasibility,
    estimate_ram_gb,
    estimate_size_gb,
)


# ---------------------------------------------------------------------------
# estimate_size_gb — known sizes
# ---------------------------------------------------------------------------

class TestKnownSizes:
    def test_exact_key_lookup(self):
        assert estimate_size_gb("qwen3:8b") == 5.2

    def test_lookup_is_case_insensitive(self):
        assert estimate_size_gb("Qwen3:8B") == 5.2

    def test_strip_latest_suffix(self):
        assert estimate_size_gb("llama3.1:8b:latest") == estimate_size_gb("llama3.1:8b")

    def test_mlx_community_entry(self):
        assert estimate_size_gb("mlx-community/qwen3-8b-4bit") == 5.2

    def test_large_model(self):
        assert estimate_size_gb("llama3.1:405b") == 230.0

    def test_tiny_model(self):
        assert estimate_size_gb("smollm2:135m") == 0.3

    def test_all_known_sizes_are_positive(self):
        for name, size in _KNOWN_SIZES.items():
            assert size > 0, f"{name!r} has non-positive size {size}"


# ---------------------------------------------------------------------------
# estimate_size_gb — parameter count parsing
# ---------------------------------------------------------------------------

class TestParamParsing:
    def test_parse_8b_after_colon(self):
        result = estimate_size_gb("unknown-model:8b")
        assert result == pytest.approx(8 * 0.61, abs=0.2)

    def test_parse_14b(self):
        result = estimate_size_gb("myfancymodel:14b")
        assert result == pytest.approx(14 * 0.61, abs=0.2)

    def test_parse_70b(self):
        result = estimate_size_gb("bigmodel:70b")
        assert result == pytest.approx(70 * 0.61, abs=1.0)

    def test_parse_fractional_3point8b(self):
        result = estimate_size_gb("phi-3.8b")
        assert result is not None
        assert 1.5 < result < 3.5

    def test_parse_0point6b(self):
        result = estimate_size_gb("someco/tiny-0.6b-chat")
        assert result is not None
        assert 0.1 < result < 1.0

    def test_instruct_suffix_not_confused(self):
        result = estimate_size_gb("mymodel-7b-instruct")
        assert result == pytest.approx(7 * 0.61, abs=0.2)

    def test_totally_unknown_returns_none(self):
        assert estimate_size_gb("completely-unknown-model-without-size") is None


# ---------------------------------------------------------------------------
# estimate_ram_gb
# ---------------------------------------------------------------------------

class TestRamEstimate:
    def test_multiplier_is_1point25(self):
        assert estimate_ram_gb(8.0) == pytest.approx(10.0)

    def test_small_model(self):
        assert estimate_ram_gb(2.0) == pytest.approx(2.5)

    def test_large_model(self):
        assert estimate_ram_gb(40.0) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Helpers for patching psutil and shutil inside check_feasibility
# ---------------------------------------------------------------------------

def _mock_system(free_disk_gb: float, total_ram_gb: float):
    """Return a context that patches psutil and shutil for check_feasibility."""
    disk_mock = MagicMock()
    disk_mock.free = int(free_disk_gb * 1024 ** 3)

    vm_mock = MagicMock()
    vm_mock.total = int(total_ram_gb * 1024 ** 3)

    import psutil
    return (
        patch("shutil.disk_usage", return_value=disk_mock),
        patch.object(psutil, "virtual_memory", return_value=vm_mock),
    )


# ---------------------------------------------------------------------------
# check_feasibility — BLOCKED verdicts
# ---------------------------------------------------------------------------

class TestBlocked:
    def test_blocked_disk(self):
        disk_p, ram_p = _mock_system(free_disk_gb=5.0, total_ram_gb=64.0)
        with disk_p, ram_p:
            # qwen3:8b is 5.2 GB; free disk is 5.0 GB → 5.2 > 5.0*0.95 = 4.75 → BLOCKED
            result = check_feasibility("qwen3:8b")
        assert result.verdict == "blocked"
        assert "disk" in result.reason.lower()

    def test_blocked_ram(self):
        disk_p, ram_p = _mock_system(free_disk_gb=1000.0, total_ram_gb=4.0)
        with disk_p, ram_p:
            # qwen3:32b is 20 GB → ram_needed ≈ 25 GB; total_ram 4 GB → 25 > 4*1.4=5.6 → BLOCKED
            result = check_feasibility("qwen3:32b")
        assert result.verdict == "blocked"
        assert "ram" in result.reason.lower() or "memory" in result.reason.lower()

    def test_blocked_405b_on_16gb_machine(self):
        disk_p, ram_p = _mock_system(free_disk_gb=1000.0, total_ram_gb=16.0)
        with disk_p, ram_p:
            result = check_feasibility("llama3.1:405b")
        assert result.verdict == "blocked"

    def test_blocked_result_has_correct_sizes(self):
        disk_p, ram_p = _mock_system(free_disk_gb=3.0, total_ram_gb=64.0)
        with disk_p, ram_p:
            result = check_feasibility("qwen3:8b")  # 5.2 GB > 3.0*0.95
        assert result.model_size_gb == pytest.approx(5.2)
        assert result.free_disk_gb  == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------------
# check_feasibility — WARN verdicts
# ---------------------------------------------------------------------------

class TestWarn:
    def test_warn_disk_tight(self):
        # 5.2 GB model, 6.0 GB free: 5.2 > 6*0.80=4.8 → warn; 5.2 < 6*0.95=5.7 → not blocked
        disk_p, ram_p = _mock_system(free_disk_gb=6.0, total_ram_gb=64.0)
        with disk_p, ram_p:
            result = check_feasibility("qwen3:8b")
        assert result.verdict == "warn"

    def test_warn_ram_tight(self):
        # qwen3:8b ram_needed ≈ 6.5 GB; total_ram 8 GB: 6.5 > 8*0.85=6.8? No, 6.5 < 6.8.
        # Use qwen3:14b → ram ≈ 11.25 GB; total 12 GB: 11.25 > 12*0.85=10.2 → warn
        disk_p, ram_p = _mock_system(free_disk_gb=1000.0, total_ram_gb=12.0)
        with disk_p, ram_p:
            result = check_feasibility("qwen3:14b")
        assert result.verdict == "warn"

    def test_warn_reason_is_non_empty(self):
        disk_p, ram_p = _mock_system(free_disk_gb=6.0, total_ram_gb=64.0)
        with disk_p, ram_p:
            result = check_feasibility("qwen3:8b")
        assert result.reason


# ---------------------------------------------------------------------------
# check_feasibility — OK verdict
# ---------------------------------------------------------------------------

class TestOk:
    def test_ok_small_model_large_machine(self):
        disk_p, ram_p = _mock_system(free_disk_gb=500.0, total_ram_gb=64.0)
        with disk_p, ram_p:
            result = check_feasibility("llama3.2:3b")
        assert result.verdict == "ok"

    def test_ok_result_fields_populated(self):
        disk_p, ram_p = _mock_system(free_disk_gb=200.0, total_ram_gb=32.0)
        with disk_p, ram_p:
            result = check_feasibility("qwen3:8b")
        assert result.verdict == "ok"
        assert result.model_size_gb == pytest.approx(5.2)
        assert result.ram_needed_gb == pytest.approx(5.2 * 1.25, abs=0.1)
        assert result.free_disk_gb  > 0
        assert result.total_ram_gb  > 0

    def test_ok_unknown_size_passes_through(self):
        disk_p, ram_p = _mock_system(free_disk_gb=10.0, total_ram_gb=8.0)
        with disk_p, ram_p:
            result = check_feasibility("completely-unknown-model-xyz")
        assert result.verdict == "ok"
        assert "unknown" in result.reason.lower()


# ---------------------------------------------------------------------------
# check_feasibility — estimated_size_gb override
# ---------------------------------------------------------------------------

class TestSizeOverride:
    def test_override_forces_blocked(self):
        disk_p, ram_p = _mock_system(free_disk_gb=5.0, total_ram_gb=64.0)
        with disk_p, ram_p:
            # Override with a huge size
            result = check_feasibility("tiny-model", estimated_size_gb=100.0)
        assert result.verdict == "blocked"

    def test_override_zero_treated_as_none(self):
        disk_p, ram_p = _mock_system(free_disk_gb=5.0, total_ram_gb=64.0)
        with disk_p, ram_p:
            result = check_feasibility("tiny-model", estimated_size_gb=0.0)
        # Falls back to name estimation (unknown → ok)
        assert result.verdict == "ok"


# ---------------------------------------------------------------------------
# FeasibilityResult.to_dict()
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_has_all_keys(self):
        r = FeasibilityResult(
            verdict="ok", reason="fine",
            model_size_gb=5.2, ram_needed_gb=6.5,
            free_disk_gb=100.0, total_ram_gb=32.0,
        )
        d = r.to_dict()
        for key in ("verdict", "reason", "model_size_gb", "ram_needed_gb",
                    "free_disk_gb", "total_ram_gb"):
            assert key in d

    def test_to_dict_rounds_disk_and_ram(self):
        r = FeasibilityResult(
            verdict="ok", reason="",
            model_size_gb=5.2, ram_needed_gb=6.5,
            free_disk_gb=99.876543, total_ram_gb=31.999,
        )
        d = r.to_dict()
        assert d["free_disk_gb"] == round(99.876543, 1)
        assert d["total_ram_gb"] == round(31.999, 1)

    def test_to_dict_verdict_preserved(self):
        for v in ("ok", "warn", "blocked"):
            r = FeasibilityResult(v, "", 0, 0, 100, 32)
            assert r.to_dict()["verdict"] == v
