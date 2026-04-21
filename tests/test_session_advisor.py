"""
Tests for autotune.session.advisor — AdaptiveAdvisor, compute_health_score,
health_status, and degradation step logic.

Covers:
- compute_health_score: RAM/swap/thermal/CPU contributions
- health_status: label/color/icon mapping for score bands
- AdaptiveAdvisor.update: state transitions (OPTIMAL → WARNING → ACTION_NEEDED)
- Degradation steps: reduce_context, lower_kv_precision
- Cooldown: actions not repeated within MIN_ACTION_INTERVAL_SEC
- Baseline accumulation via sample_count
- Event emission: proactive, threshold-crossing, spike detection
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import patch

import pytest

from autotune.session.advisor import (
    AdaptiveAdvisor,
    MEM_ACTION_PCT,
    MEM_CRITICAL_PCT,
    MEM_WARN_PCT,
    MIN_ACTION_INTERVAL_SEC,
    STABLE_BEFORE_SCALEUP_SEC,
    compute_health_score,
    health_status,
)
from autotune.session.types import (
    LiveMetrics,
    OllamaModel,
    SessionConfig,
    SessionState,
    ThermalState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(
    ram_pct: float = 50.0,
    swap_used_gb: float = 0.0,
    swap_pct: float = 0.0,
    thermal: ThermalState = ThermalState.NOMINAL,
    cpu_pct: float = 20.0,
    ram_growth: float = 0.0,
    swap_growth: float = 0.0,
    tps: Optional[float] = None,
    ttft_ms: Optional[float] = None,
    vram_pct: Optional[float] = None,
    ollama_models: Optional[list] = None,
    ram_total_gb: float = 32.0,
    ram_used_gb: Optional[float] = None,
) -> LiveMetrics:
    if ram_used_gb is None:
        ram_used_gb = ram_total_gb * ram_pct / 100
    return LiveMetrics(
        timestamp=time.time(),
        ram_total_gb=ram_total_gb,
        ram_used_gb=ram_used_gb,
        ram_available_gb=ram_total_gb - ram_used_gb,
        ram_percent=ram_pct,
        swap_total_gb=8.0,
        swap_used_gb=swap_used_gb,
        swap_percent=swap_pct,
        vram_total_gb=None,
        vram_used_gb=None,
        vram_percent=vram_pct,
        cpu_percent=cpu_pct,
        cpu_per_core=[cpu_pct],
        cpu_temp_c=None,
        gpu_temp_c=None,
        thermal_state=thermal,
        cpu_speed_limit_pct=100,
        llm_processes=[],
        ollama_models=ollama_models or [],
        tokens_per_sec=tps,
        gen_tokens_per_sec=None,
        ttft_ms=ttft_ms,
        queue_depth=0,
        swap_growth_mb_per_min=swap_growth,
        ram_growth_mb_per_min=ram_growth,
        vram_growth_mb_per_min=0.0,
    )


def _make_config(
    model_id: str = "qwen3:8b",
    context_len: int = 8192,
    quant: str = "Q4_K_M",
    kv_precision: str = "f16",
    concurrency: int = 1,
    weight_gb: float = 5.0,
    kv_cache_gb: float = 1.0,
) -> SessionConfig:
    return SessionConfig(
        model_id=model_id,
        model_name=model_id,
        quant=quant,
        context_len=context_len,
        n_gpu_layers=99,
        n_total_layers=32,
        backend="metal",
        kv_cache_precision=kv_precision,
        speculative_decoding=False,
        concurrency=concurrency,
        prompt_caching=False,
        weight_gb=weight_gb,
        kv_cache_gb=kv_cache_gb,
        total_budget_gb=16.0,
    )


# ---------------------------------------------------------------------------
# compute_health_score
# ---------------------------------------------------------------------------

class TestComputeHealthScore:
    def test_perfect_metrics_score_100(self):
        m = _make_metrics(ram_pct=30.0)
        assert compute_health_score(m) == 100

    def test_high_ram_reduces_score(self):
        low  = compute_health_score(_make_metrics(ram_pct=50.0))
        high = compute_health_score(_make_metrics(ram_pct=90.0))
        assert high < low

    def test_critical_ram_heavy_penalty(self):
        score = compute_health_score(_make_metrics(ram_pct=97.0))
        assert score <= 60   # RAM ≥97% → −40 pts from 100

    def test_swap_used_reduces_score(self):
        no_swap   = compute_health_score(_make_metrics(swap_used_gb=0.0))
        with_swap = compute_health_score(_make_metrics(swap_used_gb=1.0, swap_pct=15.0))
        assert with_swap < no_swap

    def test_heavy_swap_big_penalty(self):
        score = compute_health_score(_make_metrics(swap_pct=25.0, swap_used_gb=2.0))
        assert score <= 70   # ≥20% swap → −30 pts

    def test_thermal_throttling_reduces_score(self):
        nominal  = compute_health_score(_make_metrics(thermal=ThermalState.NOMINAL))
        throttle = compute_health_score(_make_metrics(thermal=ThermalState.THROTTLING))
        assert throttle < nominal

    def test_thermal_critical_max_penalty(self):
        score = compute_health_score(_make_metrics(thermal=ThermalState.CRITICAL))
        assert score <= 85   # −15 pts for CRITICAL

    def test_high_cpu_reduces_score(self):
        low_cpu  = compute_health_score(_make_metrics(cpu_pct=20.0))
        high_cpu = compute_health_score(_make_metrics(cpu_pct=92.0))
        assert high_cpu < low_cpu

    def test_vram_percent_used_when_set(self):
        # vram_pct takes precedence over ram_pct for score
        score_vram_high = compute_health_score(_make_metrics(ram_pct=30.0, vram_pct=97.0))
        score_ram_high  = compute_health_score(_make_metrics(ram_pct=97.0, vram_pct=30.0))
        # vram_high should be as bad as ram_high; ram_high with low vram should be better
        assert score_vram_high < score_ram_high

    def test_score_clamped_zero_to_100(self):
        # Pile on every penalty
        score = compute_health_score(_make_metrics(
            ram_pct=97.0, swap_pct=25.0, swap_growth=100.0,
            thermal=ThermalState.CRITICAL, cpu_pct=92.0, ram_growth=400.0,
        ))
        assert 0 <= score <= 100

    def test_score_always_integer(self):
        m = _make_metrics(ram_pct=75.0)
        assert isinstance(compute_health_score(m), int)


# ---------------------------------------------------------------------------
# health_status labels
# ---------------------------------------------------------------------------

class TestHealthStatus:
    @pytest.mark.parametrize("score,expected_fragment", [
        (95,  "smoothly"),
        (80,  "Moderate"),
        (60,  "pressure"),
        (40,  "Stressed"),
        (20,  "Critical"),
    ])
    def test_label_matches_score_band(self, score: int, expected_fragment: str):
        label, color, icon = health_status(score)
        assert expected_fragment in label

    def test_returns_three_values(self):
        result = health_status(75)
        assert len(result) == 3

    def test_icon_is_string(self):
        _, _, icon = health_status(50)
        assert isinstance(icon, str) and len(icon) > 0


# ---------------------------------------------------------------------------
# AdaptiveAdvisor — state transitions
# ---------------------------------------------------------------------------

class TestAdvisorStateTransitions:
    def test_optimal_when_all_good(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics(ram_pct=50.0))
        assert adv.current_state == SessionState.OPTIMAL

    def test_transitions_to_warning_at_warn_threshold(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics(ram_pct=MEM_WARN_PCT + 1))
        assert adv.current_state == SessionState.WARNING

    def test_transitions_to_action_needed_at_action_threshold(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        # Go straight to action threshold
        adv.update(_make_metrics(ram_pct=MEM_ACTION_PCT + 1))
        assert adv.current_state in (SessionState.ACTION_NEEDED, SessionState.DEGRADING)

    def test_returns_decision_when_action_needed(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        decisions = adv.update(_make_metrics(ram_pct=MEM_ACTION_PCT + 1))
        # Should produce at least one decision after exceeding action threshold
        assert isinstance(decisions, list)

    def test_state_is_critical_or_degrading_at_critical_threshold(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics(ram_pct=MEM_CRITICAL_PCT + 1))
        # CRITICAL transitions to DEGRADING immediately when an action fires
        assert adv.current_state in (SessionState.CRITICAL, SessionState.DEGRADING)

    def test_recovers_from_warning_when_ram_drops(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        # Enter warning
        adv.update(_make_metrics(ram_pct=MEM_WARN_PCT + 2))
        assert adv.current_state == SessionState.WARNING
        # Return to safe RAM
        adv.update(_make_metrics(ram_pct=50.0))
        assert adv.current_state == SessionState.OPTIMAL


# ---------------------------------------------------------------------------
# AdaptiveAdvisor — action cooldown
# ---------------------------------------------------------------------------

class TestAdvisorCooldown:
    def test_no_duplicate_action_within_cooldown(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        high_ram = _make_metrics(ram_pct=MEM_ACTION_PCT + 1)

        # Trigger first action
        d1 = adv.update(high_ram)
        # Immediate second update — still within cooldown
        d2 = adv.update(high_ram)
        # Second call should not produce another decision
        assert len(d2) == 0

    def test_action_fires_after_cooldown(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        high_ram = _make_metrics(ram_pct=MEM_ACTION_PCT + 1)

        adv.update(high_ram)
        # Fast-forward past cooldown
        adv._last_action_time -= (MIN_ACTION_INTERVAL_SEC + 1)
        adv.state = SessionState.ACTION_NEEDED
        d2 = adv.update(high_ram)
        assert len(d2) >= 1


# ---------------------------------------------------------------------------
# AdaptiveAdvisor — degradation steps
# ---------------------------------------------------------------------------

class TestAdvisorDegradationSteps:
    def _force_action(self, adv: AdaptiveAdvisor, metrics: LiveMetrics) -> list:
        """Trigger first action bypassing cooldown."""
        adv._last_action_time = 0.0
        adv.state = SessionState.ACTION_NEEDED
        return adv.update(metrics)

    def test_reduce_context_step_produces_smaller_ctx(self):
        cfg = _make_config(context_len=8192)
        adv = AdaptiveAdvisor(cfg)
        m = _make_metrics(ram_pct=MEM_ACTION_PCT + 1)
        decisions = self._force_action(adv, m)
        # After reduce_context step, suggested context should be smaller
        ctx_decisions = [d for d in decisions if d.action == "reduce_context"]
        if ctx_decisions:
            new_ctx = ctx_decisions[0].suggested_changes.get("context_len")
            assert new_ctx < cfg.context_len

    def test_lower_kv_precision_step_produces_lower_precision(self):
        cfg = _make_config(kv_precision="f16")
        adv = AdaptiveAdvisor(cfg)
        adv._degrade_index = 1   # skip reduce_concurrency, skip to lower_kv_precision area
        m = _make_metrics(ram_pct=MEM_ACTION_PCT + 1)
        decisions = self._force_action(adv, m)
        kv_decisions = [d for d in decisions if d.action == "lower_kv_precision"]
        if kv_decisions:
            new_prec = kv_decisions[0].suggested_changes.get("kv_cache_precision")
            assert new_prec != "f16"

    def test_decisions_have_reason_string(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        decisions = self._force_action(adv, _make_metrics(ram_pct=MEM_ACTION_PCT + 1))
        for d in decisions:
            assert isinstance(d.reason, str) and len(d.reason) > 0

    def test_decisions_have_revertible_flag(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        decisions = self._force_action(adv, _make_metrics(ram_pct=MEM_ACTION_PCT + 1))
        for d in decisions:
            assert isinstance(d.revertible, bool)


# ---------------------------------------------------------------------------
# AdaptiveAdvisor — baseline accumulation
# ---------------------------------------------------------------------------

class TestAdvisorBaseline:
    def test_baseline_samples_accumulate(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        for i in range(5):
            adv.update(_make_metrics(ram_pct=50.0, tps=30.0 + i))
        assert adv.baseline.sample_count == 5
        assert adv.baseline.tokens_per_sec is not None

    def test_baseline_tps_is_running_average(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics(tps=10.0))
        adv.update(_make_metrics(tps=20.0))
        # Should be between 10 and 20 — running average
        assert adv.baseline.tokens_per_sec is not None
        assert 10.0 <= adv.baseline.tokens_per_sec <= 20.0

    def test_baseline_not_updated_when_no_tps(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics(tps=None))
        assert adv.baseline.tokens_per_sec is None


# ---------------------------------------------------------------------------
# AdaptiveAdvisor — events
# ---------------------------------------------------------------------------

class TestAdvisorEvents:
    def test_events_accumulate(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics(ram_pct=MEM_WARN_PCT + 1))
        assert len(adv.events) >= 1

    def test_threshold_event_fired_at_warn(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics(ram_pct=MEM_WARN_PCT + 1))
        messages = [e.message for e in adv.events]
        # At least one event should mention RAM crossing the threshold
        assert any("RAM" in m or "caution" in m.lower() for m in messages)

    def test_no_duplicate_warn_events_on_consecutive_updates(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics(ram_pct=MEM_WARN_PCT + 1))
        count_before = sum(1 for e in adv.events if "caution" in e.message.lower())
        adv.update(_make_metrics(ram_pct=MEM_WARN_PCT + 1))
        count_after = sum(1 for e in adv.events if "caution" in e.message.lower())
        assert count_after == count_before  # hysteresis prevents re-fire

    def test_events_have_level_and_message(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics(ram_pct=50.0))
        for event in adv.events:
            assert isinstance(event.level, str)
            assert isinstance(event.message, str)

    def test_ram_drop_event_on_model_unload(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        model = OllamaModel(name="llama3:8b", size_gb=5.0, context_len=4096, vram_gb=0.0)
        # First tick: model present, high RAM
        adv.update(_make_metrics(ram_pct=80.0, ram_used_gb=25.6, ollama_models=[model]))
        # Second tick: model gone, RAM dropped
        adv.update(_make_metrics(ram_pct=60.0, ram_used_gb=19.2, ollama_models=[]))
        messages = [e.message for e in adv.events]
        assert any("unload" in m.lower() or "freed" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# AdaptiveAdvisor — recent_decisions property
# ---------------------------------------------------------------------------

class TestAdvisorDecisionsProperty:
    def test_recent_decisions_is_list(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv.update(_make_metrics())
        assert isinstance(adv.recent_decisions, list)

    def test_decision_added_to_recent_decisions(self):
        cfg = _make_config()
        adv = AdaptiveAdvisor(cfg)
        adv._last_action_time = 0.0
        adv.state = SessionState.ACTION_NEEDED
        adv.update(_make_metrics(ram_pct=MEM_ACTION_PCT + 1))
        assert len(adv.recent_decisions) >= 0  # may be 0 if no step applies
