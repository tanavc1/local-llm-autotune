"""
Tests for scripts/proof_suite.py.

Unit tests cover: statistical functions, RAM sampler logic, RunResult
properties, and report aggregation.  All pure-Python — no Ollama required.

Integration tests (skipped when Ollama is absent) do a single live run
on the smallest installed model to verify the full pipeline end-to-end.
"""
from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── ensure project root is on path ──────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from proof_suite import (
    DEFAULT_MODELS,
    PROMPTS,
    BenchPrompt,
    ModelReport,
    OllamaModelInfo,
    OllamaRamSampler,
    PromptStats,
    RunResult,
    StatResult,
    _find_ollama_runner_pid,
    _pct_cell,
    _stat,
    export_json,
)

_DUMMY_INFO = OllamaModelInfo(model_id="test")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_run(
    condition: str = "raw",
    prefill_ms: float = 200.0,
    eval_tps: float = 20.0,
    total_ms: float = 1000.0,
    ollama_peak_ram_gb: float = 4.0,
    ollama_ram_delta_gb: float = 0.5,
    free_floor_gb: float = 3.0,
    num_ctx: int = 4096,
    load_ms: float = 0.0,
    eval_count: int = 50,
    prompt_eval_count: int = 30,
    error: str | None = None,
    swap_delta_gb: float = 0.0,
    swap_occurred: bool = False,
    kv_cache_mb: float = 400.0,
    reload_detected: bool = False,
) -> RunResult:
    return RunResult(
        condition=condition,
        prompt_id="test",
        num_ctx=num_ctx,
        prefill_ms=prefill_ms,
        load_ms=load_ms,
        eval_tps=eval_tps,
        total_ms=total_ms,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        ollama_peak_ram_gb=ollama_peak_ram_gb,
        ollama_ram_delta_gb=ollama_ram_delta_gb,
        free_floor_gb=free_floor_gb,
        swap_delta_gb=swap_delta_gb,
        swap_occurred=swap_occurred,
        kv_cache_mb=kv_cache_mb,
        reload_detected=reload_detected,
        error=error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RunResult
# ─────────────────────────────────────────────────────────────────────────────

class TestRunResult:
    def test_ttft_is_load_plus_prefill(self):
        r = _make_run(load_ms=100.0, prefill_ms=200.0)
        assert r.ttft_ms == pytest.approx(300.0)

    def test_ok_true_when_no_error(self):
        assert _make_run().ok is True

    def test_ok_false_when_error(self):
        assert _make_run(error="timeout").ok is False

    def test_ok_false_when_zero_eval_count(self):
        assert _make_run(eval_count=0).ok is False

    def test_ok_false_when_error_and_zero_count(self):
        r = _make_run(error="conn refused", eval_count=0)
        assert r.ok is False


# ─────────────────────────────────────────────────────────────────────────────
# Statistical function _stat()
# ─────────────────────────────────────────────────────────────────────────────

class TestStatFunction:
    """_stat() must compute direction, effect size, and CI correctly."""

    def _raw_tuned(self, raw_vals, tuned_vals, higher_is_better=False):
        return _stat("TTFT (ms)", raw_vals, tuned_vals, higher_is_better)

    # Basic pct_change direction
    def test_pct_change_lower_is_better_improved(self):
        sr = self._raw_tuned([200.0]*4, [100.0]*4, higher_is_better=False)
        assert sr.pct_change == pytest.approx(-50.0)
        assert sr.improved is True

    def test_pct_change_lower_is_better_regression(self):
        sr = self._raw_tuned([100.0]*4, [200.0]*4, higher_is_better=False)
        assert sr.pct_change == pytest.approx(100.0)
        assert sr.improved is False

    def test_pct_change_higher_is_better_improved(self):
        sr = self._raw_tuned([10.0]*4, [20.0]*4, higher_is_better=True)
        assert sr.pct_change == pytest.approx(100.0)
        assert sr.improved is True

    # Cohen's d
    def test_cohens_d_large_for_clear_separation(self):
        """50% improvement over low variance should give large Cohen's d."""
        raw   = [200.0, 202.0, 198.0, 201.0, 199.0]
        tuned = [100.0, 101.0,  99.0, 100.0, 100.0]
        sr = self._raw_tuned(raw, tuned)
        assert abs(sr.cohens_d) > 0.8   # at least "large"

    def test_cohens_d_zero_for_identical(self):
        vals = [150.0] * 5
        sr = self._raw_tuned(vals, vals)
        assert sr.cohens_d == pytest.approx(0.0)

    # Direction wins
    def test_wins_all_correct(self):
        sr = self._raw_tuned([200]*5, [100]*5, higher_is_better=False)
        assert sr.wins == 5
        assert sr.direction_consistent is True

    def test_wins_partial(self):
        raw   = [200, 200, 200, 200, 200]
        tuned = [100, 100, 100, 100, 300]   # last one is worse
        sr = self._raw_tuned(raw, tuned, higher_is_better=False)
        assert sr.wins == 4
        assert sr.direction_consistent is False

    # 95% CI — must bracket true mean difference
    def test_ci95_brackets_true_diff(self):
        raw   = [200.0, 202.0, 198.0, 201.0, 203.0]
        tuned = [100.0, 102.0,  98.0, 101.0, 103.0]
        sr = self._raw_tuned(raw, tuned)
        true_mean_diff = -100.0   # tuned - raw
        assert sr.ci95_lo <= true_mean_diff <= sr.ci95_hi

    # Significance stars
    def test_sig_stars_none_for_p_1(self):
        sr = _stat("x", [1.0]*3, [1.0]*3, higher_is_better=True)
        assert sr.sig_stars == ""

    def test_n_reflects_paired_length(self):
        sr = self._raw_tuned([1]*7, [2]*7)
        assert sr.n == 7

    # Effect size labels
    def test_effect_label_very_large(self):
        # Use realistic variability so stdev(diffs) > 0; Cohen's d capped at ±10
        raw   = [200, 205, 195, 202, 198, 201, 199, 203, 197, 200]
        tuned = [10,   12,   8,  11,   9,  10,  11,  13,   9,  10]
        sr = self._raw_tuned(raw, tuned)
        assert sr.effect_label in ("very large", "large")
        assert abs(sr.cohens_d) <= 10.0   # must be capped

    def test_effect_label_negligible(self):
        sr = self._raw_tuned([100.0]*5, [100.1]*5)
        assert sr.effect_label in ("negligible", "small")

    # Insufficient data
    def test_insufficient_data_test_name(self):
        sr = self._raw_tuned([100.0]*2, [50.0]*2)
        assert sr.test_name == "insufficient_data"

    # Test name selection
    def test_wilcoxon_for_small_n(self):
        raw   = [200.0, 201.0, 199.0, 202.0, 198.0]
        tuned = [100.0, 101.0,  99.0, 102.0,  98.0]
        sr = self._raw_tuned(raw, tuned)
        assert sr.test_name in ("wilcoxon", "wilcoxon(tied)", "insufficient_data")


# ─────────────────────────────────────────────────────────────────────────────
# StatResult properties
# ─────────────────────────────────────────────────────────────────────────────

class TestStatResult:
    def _make(self, p: float, pct: float, hib: bool = False) -> StatResult:
        return StatResult(
            metric="x", n=5,
            raw_mean=100.0, tuned_mean=100.0 + pct,
            pct_change=pct,
            cohens_d=1.5,
            p_value=p,
            ci95_lo=pct - 5, ci95_hi=pct + 5,
            wins=5, direction_consistent=True,
            test_name="wilcoxon", higher_is_better=hib,
        )

    def test_sig_stars_three(self):
        assert self._make(0.0005, -30).sig_stars == "***"

    def test_sig_stars_two(self):
        assert self._make(0.005, -30).sig_stars == "**"

    def test_sig_stars_one(self):
        assert self._make(0.03, -30).sig_stars == "*"

    def test_sig_stars_dagger(self):
        assert self._make(0.07, -30).sig_stars == "†"

    def test_sig_stars_none(self):
        assert self._make(0.20, -30).sig_stars == ""

    def test_improved_lower_is_better(self):
        sr = self._make(0.01, -30.0, hib=False)
        assert sr.improved is True

    def test_not_improved_lower_is_better(self):
        sr = self._make(0.01, +30.0, hib=False)
        assert sr.improved is False

    def test_improved_higher_is_better(self):
        sr = self._make(0.01, +30.0, hib=True)
        assert sr.improved is True


# ─────────────────────────────────────────────────────────────────────────────
# Prompt suite completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestPrompts:
    def test_at_least_four_prompts(self):
        assert len(PROMPTS) >= 4

    def test_all_have_required_fields(self):
        for p in PROMPTS:
            assert p.id,       f"prompt {p!r} missing id"
            assert p.label,    f"prompt {p!r} missing label"
            assert p.messages, f"prompt {p!r} has no messages"
            assert all(m.get("role") and m.get("content") for m in p.messages)

    def test_all_ids_unique(self):
        ids = [p.id for p in PROMPTS]
        assert len(ids) == len(set(ids))

    def test_roles_valid(self):
        valid_roles = {"system", "user", "assistant"}
        for p in PROMPTS:
            for m in p.messages:
                assert m["role"] in valid_roles, (
                    f"Prompt {p.id} has invalid role {m['role']!r}"
                )

    def test_prompts_cover_multiple_domains(self):
        domains = {p.domain for p in PROMPTS}
        assert len(domains) >= 3, f"Expected diverse domains, got: {domains}"


# ─────────────────────────────────────────────────────────────────────────────
# OllamaRamSampler
# ─────────────────────────────────────────────────────────────────────────────

class TestOllamaRamSampler:
    def test_peak_zero_before_start(self):
        s = OllamaRamSampler()
        assert s.peak_ollama_gb() == 0.0

    def test_free_floor_zero_before_start(self):
        s = OllamaRamSampler()
        assert s.free_floor_gb() == 0.0

    def test_delta_zero_with_single_sample(self):
        s = OllamaRamSampler()
        s._rss_samples = [4.0]
        assert s.delta_ollama_gb() == 0.0   # need >= 2 samples

    def test_delta_correct_with_samples(self):
        s = OllamaRamSampler()
        s._rss_samples = [4.0, 4.5, 5.0, 4.8]
        assert s.delta_ollama_gb() == pytest.approx(1.0)   # peak(5.0) - first(4.0)

    def test_peak_returns_max(self):
        s = OllamaRamSampler()
        s._rss_samples = [2.0, 3.5, 3.0, 2.5]
        assert s.peak_ollama_gb() == pytest.approx(3.5)

    def test_free_floor_returns_min(self):
        s = OllamaRamSampler()
        s._free_samples = [5.0, 3.0, 4.0, 2.5]
        assert s.free_floor_gb() == pytest.approx(2.5)

    def test_start_stop_without_crash(self):
        """Sampler must start and stop cleanly even if no Ollama process exists."""
        s = OllamaRamSampler()
        s._pid = None   # force no-process path
        s.start()
        import time
        time.sleep(0.2)
        s.stop()
        # free_samples should have accumulated even without a PID
        assert len(s._free_samples) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# ModelReport
# ─────────────────────────────────────────────────────────────────────────────

def _make_stat_result(improved: bool, hib: bool = False) -> StatResult:
    pct = -30.0 if (improved and not hib) else (30.0 if (improved and hib) else 0.0)
    return StatResult(
        metric="x", n=3,
        raw_mean=100.0, tuned_mean=100.0 + pct,
        pct_change=pct, cohens_d=1.0,
        p_value=0.05, ci95_lo=pct-5, ci95_hi=pct+5,
        wins=3, direction_consistent=True,
        test_name="wilcoxon", higher_is_better=hib,
    )


def _make_prompt_stats(all_improved: bool = True) -> PromptStats:
    p = PROMPTS[0]
    raw_runs   = [_make_run("raw")]
    tuned_runs = [_make_run("autotune", num_ctx=1024)]
    return PromptStats(
        prompt=p,
        raw_runs=raw_runs,
        tuned_runs=tuned_runs,
        ttft=_make_stat_result(improved=all_improved, hib=False),
        prefill_ms=_make_stat_result(improved=all_improved, hib=False),
        eval_tps=_make_stat_result(improved=all_improved, hib=True),
        total_ms=_make_stat_result(improved=all_improved, hib=False),
        ollama_ram=_make_stat_result(improved=all_improved, hib=False),
        kv_cache=_make_stat_result(improved=all_improved, hib=False),
        num_ctx_raw=4096.0,
        num_ctx_tuned=1024.0,
        tokens_saved=3072,
        swap_occurred_raw=0,
        swap_occurred_tuned=0,
        reload_count_raw=0,
        reload_count_tuned=0,
    )


class TestModelReport:
    def test_overall_wins_all_improved(self):
        ps = _make_prompt_stats(all_improved=True)
        r = ModelReport("m", "balanced", 3, "hw", 0.0, _DUMMY_INFO, [ps], [], [])
        # ttft, eval_tps, ollama_ram, kv_cache, prefill_ms = 5 metrics per prompt
        assert r.overall_wins() == 5

    def test_total_metrics_per_prompt(self):
        ps_list = [_make_prompt_stats() for _ in range(3)]
        r = ModelReport("m", "balanced", 3, "hw", 0.0, _DUMMY_INFO, ps_list, [], [])
        assert r.total_metrics() == 3 * 5   # 3 prompts × 5 metrics

    def test_skipped_report_has_no_metrics(self):
        r = ModelReport("m", "balanced", 3, "", 0.0, _DUMMY_INFO, [], [], [],
                        skipped=True, skip_reason="not installed")
        assert r.total_metrics() == 0
        assert r.overall_wins() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Export JSON
# ─────────────────────────────────────────────────────────────────────────────

class TestExportJson:
    def test_valid_json_output(self, tmp_path):
        import json
        ps = _make_prompt_stats()
        r = ModelReport("llama3.2:3b", "balanced", 3, "Apple M2", 0.0, _DUMMY_INFO, [ps], [], [])
        out_path = str(tmp_path / "out.json")
        from rich.console import Console
        export_json([r], out_path, Console(quiet=True))

        with open(out_path) as f:
            data = json.load(f)

        assert data["tool"] == "autotune proof suite"
        assert len(data["models"]) == 1
        m = data["models"][0]
        assert m["model_id"] == "llama3.2:3b"
        assert "prompts" in m
        assert len(m["prompts"]) == 1
        p = m["prompts"][0]
        assert "ttft" in p
        assert "eval_tps" in p
        assert "ollama_ram" in p
        assert "raw_runs" in p
        assert "tuned_runs" in p

    def test_skipped_model_in_json(self, tmp_path):
        import json
        r = ModelReport("ghost:7b", "balanced", 3, "", 0.0, _DUMMY_INFO, [], [], [],
                        skipped=True, skip_reason="not installed")
        out_path = str(tmp_path / "skip.json")
        from rich.console import Console
        export_json([r], out_path, Console(quiet=True))

        with open(out_path) as f:
            data = json.load(f)
        assert data["models"][0]["skipped"] is True

    def test_stat_result_fields_present(self, tmp_path):
        import json
        ps = _make_prompt_stats()
        r  = ModelReport("m", "balanced", 3, "hw", 0.0, _DUMMY_INFO, [ps], [], [])
        out_path = str(tmp_path / "fields.json")
        from rich.console import Console
        export_json([r], out_path, Console(quiet=True))

        with open(out_path) as f:
            data = json.load(f)
        ttft = data["models"][0]["prompts"][0]["ttft"]
        required = {"metric","n","raw_mean","tuned_mean","pct_change",
                    "cohens_d","p_value","ci95_lo","ci95_hi","wins",
                    "direction_consistent","test_name","significance",
                    "effect_label","improved"}
        assert required.issubset(ttft.keys())


# ─────────────────────────────────────────────────────────────────────────────
# _pct_cell rendering (no crash)
# ─────────────────────────────────────────────────────────────────────────────

class TestPctCell:
    def test_negative_pct_lower_is_better_green(self):
        cell = _pct_cell(-30.0, higher_is_better=False)
        assert "-30.0%" in cell.plain

    def test_positive_pct_higher_is_better_green(self):
        cell = _pct_cell(+25.0, higher_is_better=True)
        assert "+25.0%" in cell.plain

    def test_zero_pct(self):
        cell = _pct_cell(0.0, higher_is_better=False)
        assert "0.0%" in cell.plain


# ─────────────────────────────────────────────────────────────────────────────
# Default model list
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultModels:
    def test_default_models_contains_three(self):
        assert len(DEFAULT_MODELS) == 3

    def test_target_models_in_defaults(self):
        assert "llama3.2:3b" in DEFAULT_MODELS
        assert "qwen3:8b"    in DEFAULT_MODELS
        assert "gemma4:e2b"  in DEFAULT_MODELS


# ─────────────────────────────────────────────────────────────────────────────
# Integration: live single-model run (skipped when Ollama absent)
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_running() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _smallest_model() -> str | None:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        models = r.json().get("models", [])
        models.sort(key=lambda m: m.get("size", 0))
        return models[0]["name"] if models else None
    except Exception:
        return None


ollama_running = pytest.fixture(scope="session")(
    lambda: _ollama_running() or pytest.skip("Ollama not running")
)


@pytest.fixture(scope="session")
def smallest_model(ollama_running) -> str:
    m = _smallest_model()
    if not m:
        pytest.skip("No Ollama models installed")
    return m


class TestProofSuiteIntegration:
    def test_single_prompt_raw_run(self, smallest_model):
        """Raw run must return a valid RunResult with timing data."""
        from proof_suite import PROMPTS, _fetch_model_info, _run_raw
        prompt     = PROMPTS[0]   # shortest prompt
        model_info = asyncio.run(_fetch_model_info(smallest_model))
        result     = asyncio.run(_run_raw(smallest_model, prompt, model_info, max_tokens=16))
        assert result.ok, f"Raw run failed: {result.error}"
        assert result.prefill_ms > 0, "prefill_ms must be positive"
        assert result.eval_tps > 0,   "eval_tps must be positive"
        assert result.num_ctx == 4096, "Raw run must use Ollama default num_ctx"
        assert result.condition == "raw"

    def test_single_prompt_autotune_run(self, smallest_model):
        """Autotune run must return a valid RunResult with smaller num_ctx."""
        from proof_suite import PROMPTS, _fetch_model_info, _run_autotune
        prompt     = PROMPTS[0]
        model_info = asyncio.run(_fetch_model_info(smallest_model))
        result     = asyncio.run(_run_autotune(smallest_model, prompt, model_info))
        assert result.ok, f"Autotune run failed: {result.error}"
        assert result.prefill_ms > 0
        assert result.eval_tps > 0
        assert result.num_ctx < 4096, (
            f"Autotune num_ctx ({result.num_ctx}) should be < raw default (4096)"
        )
        assert result.condition == "autotune"

    def test_autotune_num_ctx_smaller_than_raw(self, smallest_model):
        """Autotune must always produce a smaller or equal num_ctx than raw."""
        from proof_suite import PROMPTS, _fetch_model_info, _run_autotune, _run_raw
        model_info = asyncio.run(_fetch_model_info(smallest_model))
        for prompt in PROMPTS[:2]:   # first 2 prompts for speed
            raw   = asyncio.run(_run_raw(smallest_model, prompt, model_info, max_tokens=16))
            tuned = asyncio.run(_run_autotune(smallest_model, prompt, model_info))
            assert tuned.num_ctx <= raw.num_ctx, (
                f"Prompt '{prompt.id}': autotune num_ctx ({tuned.num_ctx}) "
                f">= raw ({raw.num_ctx})"
            )

    def test_stat_on_live_paired_runs(self, smallest_model):
        """_stat() must produce finite values on real paired data."""
        from proof_suite import PROMPTS, _fetch_model_info, _run_autotune, _run_raw, _stat
        prompt     = PROMPTS[0]
        model_info = asyncio.run(_fetch_model_info(smallest_model))
        n = 3
        raw_r   = [asyncio.run(_run_raw(smallest_model, prompt, model_info, max_tokens=32))
                   for _ in range(n)]
        tuned_r = [asyncio.run(_run_autotune(smallest_model, prompt, model_info))
                   for _ in range(n)]

        raw_ttft   = [r.ttft_ms for r in raw_r   if r.ok]
        tuned_ttft = [r.ttft_ms for r in tuned_r if r.ok]
        sr = _stat("TTFT (ms)", raw_ttft, tuned_ttft, higher_is_better=False)

        assert math.isfinite(sr.raw_mean)
        assert math.isfinite(sr.tuned_mean)
        assert math.isfinite(sr.cohens_d)
        assert 0.0 <= sr.p_value <= 1.0
        assert sr.n == min(len(raw_ttft), len(tuned_ttft))
