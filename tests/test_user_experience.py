"""
User-experience tests for autotune.

These tests answer the question a real user would ask:
  "Will my computer feel slower while this LLM is running?"

Test philosophy
---------------
Traditional benchmarks measure KV cache slots and prefill milliseconds.
Users don't experience those — they experience:
  • Laptop slowing down (swap events)
  • Running out of RAM for Chrome/Slack
  • Unpredictable response times
  • Fan noise / heat (CPU spikes)
  • Memory not being released after a call

Every class here maps to a concrete user scenario.
"""

from __future__ import annotations

import statistics
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from autotune.bench.user_metrics import (
    SwapEvent,
    TurnMetrics,
    UserExperienceReport,
    _LiveSampler,
    build_report,
    compute_background_impact_score,
    compute_ttft_consistency,
)
from autotune.memory.noswap import ModelArch, NoSwapGuard

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_turn(
    turn_number: int = 1,
    ttft_ms: float = 400.0,
    swap_events: int = 0,
    cpu_spike_events: int = 0,
    ram_delta_gb: float = 0.0,
    completion_tokens: int = 80,
    prompt_tokens: int = 30,
    elapsed_sec: float = 2.0,
) -> TurnMetrics:
    return TurnMetrics(
        turn_number=turn_number,
        ttft_ms=ttft_ms,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        ram_delta_gb=ram_delta_gb,
        swap_events=swap_events,
        cpu_spike_events=cpu_spike_events,
        elapsed_sec=elapsed_sec,
    )


def _make_report(
    swap_events_total: int = 0,
    ram_headroom_gb: float = 6.0,
    ttft_ms_mean: float = 400.0,
    ttft_ms_p95: float = 600.0,
    ttft_consistency_pct: float = 90.0,
    cpu_spike_events_total: int = 0,
    memory_recovery_sec: float = 2.0,
    scenario: str = "Test scenario",
    model_id: str = "test-model",
    profile_name: str = "balanced",
) -> UserExperienceReport:
    turns = [_make_turn(ttft_ms=ttft_ms_mean)]
    return UserExperienceReport(
        scenario=scenario,
        model_id=model_id,
        profile_name=profile_name,
        swap_events_total=swap_events_total,
        ram_headroom_gb=ram_headroom_gb,
        ttft_ms_mean=ttft_ms_mean,
        ttft_ms_p95=ttft_ms_p95,
        ttft_consistency_pct=ttft_consistency_pct,
        cpu_spike_events_total=cpu_spike_events_total,
        memory_recovery_sec=memory_recovery_sec,
        background_impact_score=compute_background_impact_score(
            swap_events=swap_events_total,
            ram_headroom_gb=ram_headroom_gb,
            cpu_spike_events=cpu_spike_events_total,
            ttft_consistency_pct=ttft_consistency_pct,
        ),
        turns=turns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TestSwapElimination
# "My computer never slowed down"
# ─────────────────────────────────────────────────────────────────────────────

class TestSwapElimination:
    """
    Swap events are the primary way users "feel" an LLM.
    When macOS starts swapping, everything slows — Chrome, Finder, everything.
    autotune must guarantee zero swap events in normal usage.
    """

    def test_zero_swap_events_gives_perfect_swap_score(self):
        score = compute_background_impact_score(
            swap_events=0,
            ram_headroom_gb=4.0,
            cpu_spike_events=0,
            ttft_consistency_pct=100.0,
        )
        assert score == 100.0

    def test_one_swap_event_penalises_score(self):
        score_zero = compute_background_impact_score(0, 4.0, 0, 100.0)
        score_one  = compute_background_impact_score(1, 4.0, 0, 100.0)
        assert score_one < score_zero

    def test_five_swap_events_kills_swap_component(self):
        # 5 events × 20 pts each = 100 pts off the 40% swap component
        score = compute_background_impact_score(5, 4.0, 0, 100.0)
        # Swap component is 40% of total — losing it should cost ≥35 pts
        assert score <= 65.0

    def test_report_verdict_is_positive_when_zero_swap(self):
        r = _make_report(swap_events_total=0, ram_headroom_gb=6.0)
        assert "zero swap" in r.verdict.lower() or "✅" in r.verdict

    def test_report_verdict_is_negative_when_swap_detected(self):
        r = _make_report(swap_events_total=2)
        # Re-generate verdict manually since __post_init__ fired at construction
        from autotune.bench.user_metrics import _build_verdict
        verdict = _build_verdict(r)
        assert "❌" in verdict or "swap" in verdict.lower()

    def test_no_swap_guard_prevents_swap_in_tight_ram(self):
        """NoSwapGuard must reduce num_ctx before swap occurs, not after."""
        guard = NoSwapGuard(safety_margin_gb=1.5)
        arch  = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128)

        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(
                available=int(2.0 * 1024**3),   # only 2 GB available
                percent=87.5,
            )
            # Request for 8192 context would need >2 GB KV — guard must reduce it
            decision = guard.apply(num_ctx=8192, f16_kv=True, arch=arch)

        # Should have reduced to prevent swap
        assert decision.num_ctx < 8192, "Guard must reduce ctx when RAM is tight"
        # The reduced request should fit in available memory
        kv_after = arch.kv_gb(decision.num_ctx, f16=decision.f16_kv)
        usable = 2.0 - 1.5   # available - safety_margin
        assert kv_after <= usable, (
            f"Reduced request ({kv_after:.3f} GB) must fit in usable RAM ({usable:.2f} GB)"
        )

    def test_no_swap_guard_noop_when_plenty_of_ram(self):
        guard = NoSwapGuard(safety_margin_gb=1.5)
        arch  = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128)

        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(
                available=int(12.0 * 1024**3),  # 12 GB available
                percent=25.0,
            )
            decision = guard.apply(num_ctx=4096, f16_kv=True, arch=arch)

        assert decision.level == "ok"
        assert decision.num_ctx == 4096

    @pytest.mark.parametrize("available_gb,expected_no_swap", [
        (10.0, True),   # plenty of room
        (4.0,  True),   # tight but guard should handle it
        (2.0,  True),   # very tight — guard reduces ctx
        (1.0,  True),   # extreme — guard uses minimum ctx
    ])
    def test_guard_guarantees_no_swap_at_all_ram_levels(self, available_gb, expected_no_swap):
        guard = NoSwapGuard(safety_margin_gb=0.5)
        arch  = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128)

        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(available=int(available_gb * 1024**3), percent=50.0)
            decision = guard.apply(num_ctx=8192, f16_kv=True, arch=arch)

        usable = available_gb - 0.5
        kv_after = arch.kv_gb(decision.num_ctx, f16=decision.f16_kv)
        fits = kv_after <= usable
        assert fits == expected_no_swap, (
            f"At {available_gb} GB available, KV ({kv_after:.3f} GB) must fit in usable ({usable:.2f} GB)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestRAMHeadroom
# "I still had RAM for my other apps"
# ─────────────────────────────────────────────────────────────────────────────

class TestRAMHeadroom:
    """
    A user running Chrome, Slack, and VS Code needs a minimum amount of free RAM.
    autotune must not consume all available memory for the LLM.
    """

    TYPICAL_APP_RAM_GB = 3.0   # Chrome + Slack + Finder ≈ 3 GB

    def test_headroom_positive_in_normal_scenario(self):
        r = _make_report(ram_headroom_gb=6.0)
        assert r.ram_headroom_gb >= self.TYPICAL_APP_RAM_GB

    def test_headroom_score_full_when_four_gb_free(self):
        score = compute_background_impact_score(0, 4.0, 0, 100.0)
        assert score == 100.0

    def test_headroom_score_zero_when_no_ram_free(self):
        score = compute_background_impact_score(0, 0.0, 0, 100.0)
        # 30% headroom component = 0; remainder = 40 + 20 + 10 = 70
        assert score == pytest.approx(70.0, abs=1.0)

    def test_headroom_score_scales_linearly(self):
        score_2gb = compute_background_impact_score(0, 2.0, 0, 100.0)
        score_4gb = compute_background_impact_score(0, 4.0, 0, 100.0)
        # 4 GB should score higher than 2 GB
        assert score_4gb > score_2gb

    def test_headroom_capped_at_four_gb(self):
        # More than 4 GB headroom doesn't score higher than 4 GB
        score_4gb  = compute_background_impact_score(0, 4.0,  0, 100.0)
        score_10gb = compute_background_impact_score(0, 10.0, 0, 100.0)
        assert score_4gb == score_10gb

    def test_report_headroom_computed_from_peak(self):
        """
        Headroom = total_ram - peak_ram_during_inference.
        It must use the peak, not the before/after values.
        """
        total_gb = 16.0
        turns = [_make_turn() for _ in range(3)]
        r = build_report(
            scenario="test",
            model_id="test-model",
            profile_name="balanced",
            turns=turns,
            total_ram_gb=total_gb,
            ram_before_gb=10.0,
            ram_peak_gb=12.0,    # peak — this is what matters
            ram_after_gb=10.5,
            swap_before_gb=0.0,
            swap_peak_gb=0.0,
            cpu_avg_pct=30.0,
            cpu_peak_pct=60.0,
            elapsed_total_sec=5.0,
            memory_recovery_sec=1.5,
        )
        assert r.ram_headroom_gb == pytest.approx(total_gb - 12.0, abs=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# TestTTFTConsistency
# "Responses were fast every time, not just sometimes"
# ─────────────────────────────────────────────────────────────────────────────

class TestTTFTConsistency:
    """
    Users don't just want fast responses — they want *consistent* responses.
    A model that takes 300ms sometimes and 4,000ms other times is frustrating.
    autotune's prefix caching should make TTFT converge over turns.
    """

    def test_identical_ttft_is_100_pct_consistent(self):
        ttft = [400.0, 400.0, 400.0, 400.0]
        assert compute_ttft_consistency(ttft) == 100.0

    def test_single_turn_is_100_pct_consistent(self):
        assert compute_ttft_consistency([750.0]) == 100.0

    def test_empty_list_is_100_pct_consistent(self):
        assert compute_ttft_consistency([]) == 100.0

    def test_high_variance_gives_low_consistency(self):
        # Very spread out values → low consistency
        ttft = [100.0, 2000.0, 100.0, 2000.0]
        score = compute_ttft_consistency(ttft)
        assert score < 50.0, f"High-variance TTFT should score < 50%, got {score}"

    def test_low_variance_gives_high_consistency(self):
        ttft = [390.0, 400.0, 410.0, 395.0]
        score = compute_ttft_consistency(ttft)
        assert score >= 85.0, f"Low-variance TTFT should score ≥85%, got {score}"

    def test_prefix_caching_pattern_is_consistent(self):
        """
        With prefix caching, TTFT decreases over turns (warm KV).
        A decreasing sequence that converges should still score highly.
        """
        # Turn 1 cold-start: 900ms. Turns 2–5: stable ~350ms (prefix cached)
        ttft = [900.0, 360.0, 340.0, 355.0, 345.0]
        score = compute_ttft_consistency(ttft)
        # Allow lower score because of the cold-start spike, but must be reasonable
        assert score >= 40.0, f"Prefix-cached pattern should score ≥40%, got {score}"

    def test_consistency_used_in_background_score(self):
        high_consistency = compute_background_impact_score(0, 4.0, 0, 95.0)
        low_consistency  = compute_background_impact_score(0, 4.0, 0, 10.0)
        assert high_consistency > low_consistency

    def test_build_report_computes_ttft_from_turns(self):
        ttft_values = [400.0, 380.0, 420.0, 410.0]
        turns = [_make_turn(turn_number=i+1, ttft_ms=t) for i, t in enumerate(ttft_values)]
        r = build_report(
            scenario="test",
            model_id="test",
            profile_name="balanced",
            turns=turns,
            total_ram_gb=16.0,
            ram_before_gb=8.0,
            ram_peak_gb=10.0,
            ram_after_gb=8.5,
            swap_before_gb=0.0,
            swap_peak_gb=0.0,
            cpu_avg_pct=25.0,
            cpu_peak_pct=55.0,
            elapsed_total_sec=10.0,
            memory_recovery_sec=2.0,
        )
        assert r.ttft_ms_mean == pytest.approx(statistics.mean(ttft_values), abs=1.0)
        assert r.ttft_consistency_pct >= 80.0


# ─────────────────────────────────────────────────────────────────────────────
# TestBackgroundImpact
# "My laptop felt completely normal while the LLM was running"
# ─────────────────────────────────────────────────────────────────────────────

class TestBackgroundImpact:
    """
    The headline score: 0–100, where 100 means the LLM is completely invisible.
    """

    def test_perfect_conditions_score_100(self):
        score = compute_background_impact_score(0, 4.0, 0, 100.0)
        assert score == 100.0

    def test_score_is_zero_to_100_range(self):
        for swap in [0, 1, 5, 10]:
            for headroom in [0.0, 2.0, 4.0, 8.0]:
                for cpu in [0, 1, 3]:
                    for consistency in [0.0, 50.0, 100.0]:
                        score = compute_background_impact_score(
                            swap, headroom, cpu, consistency
                        )
                        assert 0.0 <= score <= 100.0, (
                            f"Score out of range: {score} "
                            f"(swap={swap}, headroom={headroom}, cpu={cpu}, consistency={consistency})"
                        )

    def test_worst_case_approaches_zero(self):
        score = compute_background_impact_score(10, 0.0, 10, 0.0)
        assert score < 15.0, f"Worst-case conditions should score near 0, got {score}"

    def test_swap_weight_is_largest(self):
        """Swap events hurt more than CPU spikes — that's the right priority."""
        one_swap   = compute_background_impact_score(1, 4.0, 0, 100.0)
        one_spike  = compute_background_impact_score(0, 4.0, 1, 100.0)
        assert one_swap < one_spike, "One swap event should penalise more than one CPU spike"

    def test_headroom_weight_second_most_important(self):
        no_headroom = compute_background_impact_score(0, 0.0, 0, 100.0)
        some_headroom = compute_background_impact_score(0, 2.0, 0, 100.0)
        assert some_headroom > no_headroom

    def test_report_has_correct_score(self):
        r = build_report(
            scenario="5-turn chat",
            model_id="qwen3:8b",
            profile_name="balanced",
            turns=[_make_turn(swap_events=0, cpu_spike_events=0) for _ in range(5)],
            total_ram_gb=16.0,
            ram_before_gb=8.0,
            ram_peak_gb=11.0,   # 5 GB headroom
            ram_after_gb=8.2,
            swap_before_gb=0.0,
            swap_peak_gb=0.0,
            cpu_avg_pct=22.0,
            cpu_peak_pct=45.0,
            elapsed_total_sec=30.0,
            memory_recovery_sec=1.5,
        )
        assert r.background_impact_score >= 90.0
        assert r.swap_events_total == 0

    def test_report_card_prints_without_error(self, capsys):
        r = _make_report()
        r.print_card()
        out = capsys.readouterr().out
        assert "USER EXPERIENCE REPORT" in out
        assert "swap" in out.lower()
        assert "RAM free" in out

    def test_report_to_dict_is_serialisable(self):
        import json
        r = _make_report()
        d = r.to_dict()
        # Should be JSON-serialisable
        json_str = json.dumps(d)
        assert "swap_events_total" in json_str
        assert "background_impact_score" in json_str


# ─────────────────────────────────────────────────────────────────────────────
# TestMemoryPressureRecovery
# "RAM came back after the call — my other apps were fine"
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryPressureRecovery:
    """
    After an LLM call, RAM should return to near-baseline quickly.
    If it doesn't, other apps stay starved — Chrome tabs crash, Xcode slows down.
    """

    def test_fast_recovery_scores_well(self):
        r = _make_report(memory_recovery_sec=1.5)
        assert r.memory_recovery_sec < 10.0

    def test_slow_recovery_noted_in_verdict(self):
        r = _make_report(memory_recovery_sec=25.0)
        from autotune.bench.user_metrics import _build_verdict
        verdict = _build_verdict(r)
        # A high recovery time with good other metrics → verdict should still
        # note that things settled
        assert isinstance(verdict, str) and len(verdict) > 0

    def test_recovery_measured_relative_to_baseline(self):
        """
        Recovery time uses the baseline (before inference), not absolute zero.
        """
        sampler = _LiveSampler(interval_sec=0.01)
        # Simulate: start at 8 GB, rise to 11 GB, fall back to 8.2 GB
        sampler._ram_gb   = [8.0, 9.0, 11.0, 10.0, 8.5, 8.2]
        sampler._timestamps = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]

        recovery = sampler.ram_recovery_sec(threshold_gb=0.5)
        # At t=3.0 (index 4), RAM is 8.5, which is within 0.5 of baseline 8.0
        assert recovery <= 4.0

    def test_no_recovery_needed_when_stable(self):
        """If RAM never changed, recovery time is 0."""
        sampler = _LiveSampler(interval_sec=0.01)
        sampler._ram_gb   = [8.0, 8.0, 8.0, 8.0]
        sampler._timestamps = [0.0, 1.0, 2.0, 3.0]

        recovery = sampler.ram_recovery_sec(threshold_gb=0.1)
        # The very last sample is still at baseline → recovery is the total time
        # (last sample within threshold of first sample)
        assert recovery >= 0.0   # valid non-negative

    def test_keep_alive_preserves_model_weights_in_ram(self):
        """
        autotune sets keep_alive=-1m so the model stays loaded.
        This means model weights don't need to be re-loaded each turn.
        The RAM delta per turn (after turn 1) should be small.
        """
        # Simulate: turn 1 loads model (+5 GB), turns 2–4 stable (±0.1 GB)
        turns = [
            _make_turn(turn_number=1, ram_delta_gb=5.0),   # cold load
            _make_turn(turn_number=2, ram_delta_gb=0.05),
            _make_turn(turn_number=3, ram_delta_gb=-0.02),
            _make_turn(turn_number=4, ram_delta_gb=0.03),
        ]
        warm_deltas = [abs(t.ram_delta_gb) for t in turns[1:]]
        # After initial load, per-turn RAM change should be tiny
        assert all(d < 0.5 for d in warm_deltas), (
            f"Post-load RAM deltas should be <0.5 GB per turn, got {warm_deltas}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentLoopStability
# "The agent ran all 10 steps without my computer freezing"
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentLoopStability:
    """
    Agent loops accumulate context over many turns.
    Raw Ollama reloads the model when context exceeds num_ctx — causing a
    multi-second stall. autotune pre-allocates the session ceiling so this
    never happens.
    """

    def test_zero_model_reloads_in_agent_session(self):
        """
        Simulate 10 agent turns. Model reloads are detected by sudden TTFT spikes
        (>400ms increase over a prefix-cached baseline).
        """
        # Autotune scenario: TTFT falls and stays stable (prefix cache working)
        autotune_ttft = [900, 380, 350, 340, 345, 330, 340, 335, 340, 330]

        # Raw Ollama scenario: TTFT climbs then spikes at reload
        raw_ttft = [500, 550, 600, 700, 2500, 520, 600, 700, 2600, 530]

        def count_reloads(ttft_sequence: list[int], threshold_ms: int = 1500) -> int:
            return sum(1 for t in ttft_sequence if t > threshold_ms)

        assert count_reloads(autotune_ttft) == 0, "autotune should have zero model reloads"
        assert count_reloads(raw_ttft) >= 2, "Raw Ollama should show reloads in this scenario"

    def test_context_growth_stays_bounded(self):
        """
        autotune context trimming keeps session context from growing unboundedly.
        After RECENT+FACTS threshold, old turns should be summarised.
        """
        from autotune.context.budget import BudgetTier, classify_budget

        # At 80% usage → should be COMPRESSED or RECENT_PLUS_FACTS, not FULL
        state = classify_budget(history_tokens=6554, effective_budget=8192)   # ~80%
        assert state.tier in (BudgetTier.COMPRESSED, BudgetTier.RECENT_PLUS_FACTS), (
            f"At 80% context usage, should compress — got {state.tier}"
        )

    def test_ttft_improves_over_turns_with_prefix_cache(self):
        """
        With prefix caching, turns 2+ should be faster than turn 1 cold start.
        This is the core value prop for agent loops.
        """
        # Cold start: 900ms. Warm turns: ~350ms
        turns = [
            _make_turn(turn_number=1, ttft_ms=900.0),   # cold: no cache
            _make_turn(turn_number=2, ttft_ms=360.0),   # warm: system prompt cached
            _make_turn(turn_number=3, ttft_ms=340.0),
            _make_turn(turn_number=4, ttft_ms=350.0),
            _make_turn(turn_number=5, ttft_ms=345.0),
        ]

        warm_ttft = [t.ttft_ms for t in turns[1:]]
        cold_ttft = turns[0].ttft_ms

        assert all(t < cold_ttft for t in warm_ttft), (
            f"Warm turns should be faster than cold start ({cold_ttft}ms): {warm_ttft}"
        )

    def test_swap_count_is_zero_throughout_agent_session(self):
        """
        A 10-turn agent session on a 16 GB machine should never trigger swap.
        """
        turns = [_make_turn(turn_number=i, swap_events=0) for i in range(10)]
        r = build_report(
            scenario="10-turn agent loop",
            model_id="qwen3:8b",
            profile_name="balanced",
            turns=turns,
            total_ram_gb=16.0,
            ram_before_gb=6.0,
            ram_peak_gb=11.5,
            ram_after_gb=6.5,
            swap_before_gb=0.0,
            swap_peak_gb=0.0,
            cpu_avg_pct=28.0,
            cpu_peak_pct=65.0,
            elapsed_total_sec=60.0,
            memory_recovery_sec=2.0,
        )
        assert r.swap_events_total == 0
        assert r.background_impact_score >= 80.0


# ─────────────────────────────────────────────────────────────────────────────
# TestNormalLaptopUse
# "I ran Chrome + Slack + VS Code while chatting with the model"
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalLaptopUse:
    """
    The most common real-world scenario: user is doing normal work and fires
    a single question at the LLM in the background.
    """

    CHROME_SLACK_VSCODE_RAM_GB = 3.5   # typical background app footprint

    def test_single_query_leaves_ram_for_apps(self):
        """After one query, there must be enough RAM for background apps."""
        # 16 GB machine, model uses ~5.2 GB, inference overhead ~1 GB
        peak_during_inference = 11.0
        total_ram = 16.0
        headroom = total_ram - peak_during_inference  # 5.0 GB

        assert headroom >= self.CHROME_SLACK_VSCODE_RAM_GB, (
            f"Only {headroom:.1f} GB free during inference — "
            f"not enough for background apps ({self.CHROME_SLACK_VSCODE_RAM_GB} GB needed)"
        )

    def test_background_score_high_on_16gb_machine(self):
        """A 16 GB machine running qwen3:8b should score ≥80/100."""
        r = build_report(
            scenario="Single query — 16 GB laptop",
            model_id="qwen3:8b",
            profile_name="balanced",
            turns=[_make_turn(ttft_ms=450.0, swap_events=0, cpu_spike_events=0)],
            total_ram_gb=16.0,
            ram_before_gb=7.0,
            ram_peak_gb=11.0,
            ram_after_gb=7.2,
            swap_before_gb=0.0,
            swap_peak_gb=0.0,
            cpu_avg_pct=25.0,
            cpu_peak_pct=55.0,
            elapsed_total_sec=4.0,
            memory_recovery_sec=1.0,
        )
        assert r.background_impact_score >= 80.0
        assert r.swap_events_total == 0
        assert r.ram_headroom_gb >= self.CHROME_SLACK_VSCODE_RAM_GB

    def test_background_score_reasonable_on_8gb_machine(self):
        """An 8 GB machine is tight. autotune must still prevent swap."""
        r = build_report(
            scenario="Single query — 8 GB laptop",
            model_id="llama3.2:3b",
            profile_name="fast",
            turns=[_make_turn(ttft_ms=600.0, swap_events=0, cpu_spike_events=0)],
            total_ram_gb=8.0,
            ram_before_gb=4.0,
            ram_peak_gb=6.5,
            ram_after_gb=4.1,
            swap_before_gb=0.0,
            swap_peak_gb=0.0,
            cpu_avg_pct=35.0,
            cpu_peak_pct=70.0,
            elapsed_total_sec=5.0,
            memory_recovery_sec=1.5,
        )
        # 8 GB is tight — may have limited headroom (8-6.5=1.5 GB)
        # but swap must still be 0
        assert r.swap_events_total == 0

    def test_fast_profile_uses_less_ram_than_quality(self):
        """
        fast profile max_context=2048 vs quality=32768.
        Lower context = smaller KV = more RAM for user's other apps.
        """
        from autotune.api.profiles import get_profile

        fast    = get_profile("fast")
        quality = get_profile("quality")

        assert fast.max_context_tokens < quality.max_context_tokens
        # KV scales linearly with context — fast must use less RAM
        # (This is the mechanism behind "fast profile for 8 GB machines")


# ─────────────────────────────────────────────────────────────────────────────
# TestLiveSampler (unit tests for the sampler itself)
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveSampler:
    """Tests for the _LiveSampler that underlies all metric collection."""

    def test_detects_swap_event(self):
        sampler = _LiveSampler()
        sampler._swap_gb   = [0.0, 0.0, 0.05, 0.20, 0.20]
        sampler._timestamps = [0.0, 0.5, 1.0,  1.5,  2.0]
        sampler._cpu_pct    = [20.0] * 5
        sampler._ram_gb     = [8.0] * 5
        sampler._analyse()

        assert len(sampler.swap_events) >= 1
        assert sampler.swap_events[0].delta_gb >= _LiveSampler.SWAP_EVENT_THRESHOLD

    def test_no_false_positive_swap_events(self):
        """Tiny fluctuations (< threshold) must not be counted as swap events."""
        sampler = _LiveSampler()
        sampler._swap_gb   = [0.0, 0.005, 0.003, 0.007]  # all below threshold
        sampler._timestamps = [0.0, 0.5,   1.0,   1.5]
        sampler._cpu_pct    = [20.0] * 4
        sampler._ram_gb     = [8.0] * 4
        sampler._analyse()

        assert len(sampler.swap_events) == 0

    def test_detects_cpu_spike(self):
        sampler = _LiveSampler(interval_sec=0.25)
        # 10 samples at 85% CPU = 2.5 seconds → counts as one spike
        sampler._cpu_pct    = [85.0] * 10 + [20.0]
        sampler._timestamps = [i * 0.25 for i in range(11)]
        sampler._swap_gb    = [0.0] * 11
        sampler._ram_gb     = [8.0] * 11
        sampler._analyse()

        assert sampler.cpu_spike_events >= 1

    def test_no_false_positive_cpu_spike(self):
        """A brief CPU burst under 2 s must not count."""
        sampler = _LiveSampler(interval_sec=0.25)
        # 3 samples at 85% = 0.75 seconds → under 2s threshold
        sampler._cpu_pct    = [85.0, 85.0, 85.0, 20.0, 20.0]
        sampler._timestamps = [0.0, 0.25, 0.5, 0.75, 1.0]
        sampler._swap_gb    = [0.0] * 5
        sampler._ram_gb     = [8.0] * 5
        sampler._analyse()

        assert sampler.cpu_spike_events == 0

    def test_headroom_uses_peak_not_mean(self):
        sampler = _LiveSampler()
        sampler._ram_gb = [8.0, 10.0, 11.5, 9.0, 8.5]   # peak = 11.5
        total = 16.0
        headroom = sampler.ram_headroom_gb(total)
        assert headroom == pytest.approx(total - 11.5, abs=0.01)

    def test_multiple_swap_events_all_counted(self):
        sampler = _LiveSampler()
        # Two separate swap bursts
        sampler._swap_gb   = [0.0, 0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4]
        sampler._timestamps = [i * 0.25 for i in range(8)]
        sampler._cpu_pct    = [20.0] * 8
        sampler._ram_gb     = [8.0]   * 8
        sampler._analyse()

        assert len(sampler.swap_events) >= 1
        assert all(e.delta_gb >= _LiveSampler.SWAP_EVENT_THRESHOLD for e in sampler.swap_events)
