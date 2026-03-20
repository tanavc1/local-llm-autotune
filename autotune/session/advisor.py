"""
Adaptive advisor: watches live metrics and decides when/how to act.

State machine:
  OPTIMAL → WARNING → ACTION_NEEDED → DEGRADING → STABLE_RECOVERING → OPTIMAL

Smooth degradation order (least → most disruptive):
  1. reduce_concurrency
  2. reduce_context
  3. lower_kv_precision
  4. improve_cache_reuse
  5. disable_speculative_decoding
  6. lower_quantization
  7. switch_to_smaller_model
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .types import (
    AdvisorDecision, LiveMetrics, SessionConfig, SessionEvent,
    SessionState, ThermalState,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MEM_WARN_PCT    = 80.0    # start warning
MEM_ACTION_PCT  = 88.0    # must act
MEM_CRITICAL_PCT = 94.0   # emergency

SWAP_GROWTH_WARN_MB_MIN  = 10.0   # MB/min
SWAP_GROWTH_ACTION_MB_MIN = 50.0

TPS_DROP_WARN   = 0.70    # tok/s below 70% of baseline
TPS_DROP_ACTION = 0.50    # below 50%

TTFT_RISE_WARN   = 1.50   # TTFT 50% above baseline
TTFT_RISE_ACTION = 2.00   # 2× baseline

# How long metrics must be stable before scaling back up
STABLE_BEFORE_SCALEUP_SEC = 90.0

# Minimum time between actions (don't thrash)
MIN_ACTION_INTERVAL_SEC = 20.0


# ---------------------------------------------------------------------------
# Degradation steps (ordered least → most disruptive)
# ---------------------------------------------------------------------------

DEGRADE_STEPS = [
    "reduce_concurrency",
    "reduce_context",
    "lower_kv_precision",
    "improve_cache_reuse",
    "disable_speculative_decoding",
    "lower_quantization",
    "switch_to_smaller_model",
]

RECOVER_STEPS = list(reversed(DEGRADE_STEPS))  # reverse order for scale-up

# Context reduction ladder (tokens)
CONTEXT_LADDER = [131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512]

# KV precision ladder
KV_PRECISION_LADDER = ["f16", "q8", "q4"]

# Quantization degradation
QUANT_DEGRADE = {
    "F16": "Q8_0", "Q8_0": "Q6_K", "Q6_K": "Q5_K_M",
    "Q5_K_M": "Q4_K_M", "Q4_K_M": "Q4_K_S",
    "Q4_K_S": "Q3_K_M", "Q3_K_M": "Q2_K",
}


@dataclass
class SessionBaseline:
    tokens_per_sec: Optional[float] = None
    ttft_ms: Optional[float] = None
    ram_percent: float = 0.0
    vram_percent: Optional[float] = None
    established_at: float = field(default_factory=time.time)
    sample_count: int = 0


class AdaptiveAdvisor:
    """
    Watches a stream of LiveMetrics and emits AdvisorDecisions when action is needed.
    """

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.state = SessionState.OPTIMAL
        self.baseline = SessionBaseline()

        self._degrade_index = -1          # index into DEGRADE_STEPS, -1 = none applied
        self._last_action_time = 0.0
        self._stable_since: Optional[float] = None
        self._decisions: deque[AdvisorDecision] = deque(maxlen=50)
        self._events: deque[SessionEvent] = deque(maxlen=100)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def current_state(self) -> SessionState:
        return self.state

    @property
    def recent_decisions(self) -> list[AdvisorDecision]:
        return list(self._decisions)

    @property
    def events(self) -> list[SessionEvent]:
        return list(self._events)

    def update(self, metrics: LiveMetrics) -> list[AdvisorDecision]:
        """
        Feed the latest metrics snapshot. Returns any new decisions made.
        """
        self._update_baseline(metrics)
        signals = self._evaluate_signals(metrics)
        decisions = self._decide(signals, metrics)

        for d in decisions:
            self._decisions.appendleft(d)
            self._log(d.reason, d.severity.value.upper())

        return decisions

    # ------------------------------------------------------------------ #
    # Baseline tracking                                                    #
    # ------------------------------------------------------------------ #

    def _update_baseline(self, m: LiveMetrics) -> None:
        b = self.baseline
        n = b.sample_count

        if m.tokens_per_sec is not None:
            b.tokens_per_sec = (
                m.tokens_per_sec if b.tokens_per_sec is None
                else (b.tokens_per_sec * min(n, 20) + m.tokens_per_sec) / (min(n, 20) + 1)
            )
        if m.ttft_ms is not None:
            b.ttft_ms = (
                m.ttft_ms if b.ttft_ms is None
                else (b.ttft_ms * min(n, 20) + m.ttft_ms) / (min(n, 20) + 1)
            )

        b.ram_percent = (b.ram_percent * min(n, 10) + m.ram_percent) / (min(n, 10) + 1)
        if m.vram_percent is not None:
            b.vram_percent = (
                m.vram_percent if b.vram_percent is None
                else (b.vram_percent * min(n, 10) + m.vram_percent) / (min(n, 10) + 1)
            )
        b.sample_count += 1

    # ------------------------------------------------------------------ #
    # Signal evaluation
    # ------------------------------------------------------------------ #

    def _evaluate_signals(self, m: LiveMetrics) -> dict[str, str]:
        """Return dict of signal_name → 'warn'|'action'|'critical'|'ok'."""
        signals: dict[str, str] = {}

        # Memory
        mem = m.vram_percent if m.vram_percent is not None else m.ram_percent
        if mem >= MEM_CRITICAL_PCT:
            signals["memory"] = "critical"
        elif mem >= MEM_ACTION_PCT:
            signals["memory"] = "action"
        elif mem >= MEM_WARN_PCT:
            signals["memory"] = "warn"
        else:
            signals["memory"] = "ok"

        # Swap growth
        if m.swap_growth_mb_per_min >= SWAP_GROWTH_ACTION_MB_MIN:
            signals["swap"] = "action"
        elif m.swap_growth_mb_per_min >= SWAP_GROWTH_WARN_MB_MIN:
            signals["swap"] = "warn"
        elif m.swap_used_gb > 0.1:
            signals["swap"] = "warn"
        else:
            signals["swap"] = "ok"

        # tok/s vs baseline
        b = self.baseline
        if b.tokens_per_sec and m.tokens_per_sec:
            ratio = m.tokens_per_sec / b.tokens_per_sec
            if ratio < TPS_DROP_ACTION:
                signals["throughput"] = "action"
            elif ratio < TPS_DROP_WARN:
                signals["throughput"] = "warn"
            else:
                signals["throughput"] = "ok"
        else:
            signals["throughput"] = "ok"

        # TTFT vs baseline
        if b.ttft_ms and m.ttft_ms:
            ratio = m.ttft_ms / b.ttft_ms
            if ratio > TTFT_RISE_ACTION:
                signals["ttft"] = "action"
            elif ratio > TTFT_RISE_WARN:
                signals["ttft"] = "warn"
            else:
                signals["ttft"] = "ok"
        else:
            signals["ttft"] = "ok"

        # Thermal
        ts = m.thermal_state
        if ts == ThermalState.CRITICAL:
            signals["thermal"] = "critical"
        elif ts == ThermalState.THROTTLING:
            signals["thermal"] = "action"
        elif ts in (ThermalState.WARNING, ThermalState.WARM):
            signals["thermal"] = "warn"
        else:
            signals["thermal"] = "ok"

        # RAM growth trajectory
        if m.ram_growth_mb_per_min > 200:     # >200 MB/min = heading for trouble
            signals["memory_growth"] = "warn"
        else:
            signals["memory_growth"] = "ok"

        return signals

    # ------------------------------------------------------------------ #
    # Decision logic
    # ------------------------------------------------------------------ #

    def _decide(self, signals: dict[str, str], m: LiveMetrics) -> list[AdvisorDecision]:
        decisions: list[AdvisorDecision] = []
        now = time.time()

        # Determine worst signal level
        levels = list(signals.values())
        worst = (
            "critical" if "critical" in levels else
            "action"   if "action" in levels else
            "warn"     if "warn" in levels else
            "ok"
        )

        # State transitions
        prev_state = self.state

        if worst == "critical":
            self.state = SessionState.CRITICAL
        elif worst == "action":
            if self.state == SessionState.OPTIMAL:
                self.state = SessionState.ACTION_NEEDED
            elif self.state == SessionState.WARNING:
                self.state = SessionState.ACTION_NEEDED
        elif worst == "warn":
            if self.state == SessionState.OPTIMAL:
                self.state = SessionState.WARNING
        elif worst == "ok":
            if self.state in (SessionState.DEGRADING, SessionState.ACTION_NEEDED):
                self.state = SessionState.STABLE_RECOVERING
                if self._stable_since is None:
                    self._stable_since = now
            elif self.state == SessionState.STABLE_RECOVERING:
                if self._stable_since and (now - self._stable_since) > STABLE_BEFORE_SCALEUP_SEC:
                    self.state = SessionState.OPTIMAL
                    self._stable_since = None
                    self._log("Sustained stability — ready to scale up if needed", "OK")
            elif self.state == SessionState.WARNING:
                self.state = SessionState.OPTIMAL

        if prev_state != self.state:
            self._log(f"State: {prev_state.value} → {self.state.value}", "INFO")

        # Emit action if needed and cooldown passed
        if self.state in (SessionState.ACTION_NEEDED, SessionState.CRITICAL):
            if (now - self._last_action_time) > MIN_ACTION_INTERVAL_SEC:
                d = self._next_action(signals, m)
                if d:
                    decisions.append(d)
                    self._last_action_time = now
                    self.state = SessionState.DEGRADING
                    self._degrade_index += 1
                    self._stable_since = None

        return decisions

    def _next_action(self, signals: dict[str, str], m: LiveMetrics) -> Optional[AdvisorDecision]:
        """Choose the least-disruptive available action."""
        cfg = self.config
        mem = m.vram_percent if m.vram_percent is not None else m.ram_percent

        # Work through degradation steps in order
        next_idx = self._degrade_index + 1
        if next_idx >= len(DEGRADE_STEPS):
            self._log("All degradation steps exhausted — cannot act further", "CRITICAL")
            return None

        step = DEGRADE_STEPS[next_idx]

        if step == "reduce_concurrency" and cfg.concurrency > 1:
            new_conc = max(1, cfg.concurrency - 1)
            reason = f"Reducing concurrency {cfg.concurrency}→{new_conc} (memory {mem:.0f}%)"
            return AdvisorDecision(
                action=step, reason=reason,
                severity=SessionState.ACTION_NEEDED,
                suggested_changes={"concurrency": new_conc},
                revertible=True,
            )

        elif step == "reduce_context":
            # Find next smaller context in ladder
            current_ctx = cfg.context_len
            smaller = next((c for c in CONTEXT_LADDER if c < current_ctx), None)
            if smaller:
                reason = (
                    f"Reducing context {current_ctx:,}→{smaller:,} tokens "
                    f"(saves ~{_kv_delta_gb(cfg, current_ctx, smaller):.2f} GB)"
                )
                return AdvisorDecision(
                    action=step, reason=reason,
                    severity=SessionState.ACTION_NEEDED,
                    suggested_changes={"context_len": smaller},
                    revertible=True,
                )

        elif step == "lower_kv_precision":
            prec_idx = KV_PRECISION_LADDER.index(cfg.kv_cache_precision) if cfg.kv_cache_precision in KV_PRECISION_LADDER else 0
            if prec_idx < len(KV_PRECISION_LADDER) - 1:
                new_prec = KV_PRECISION_LADDER[prec_idx + 1]
                reason = f"Lowering KV-cache precision {cfg.kv_cache_precision}→{new_prec} (reduces KV memory ~{'50' if new_prec=='q4' else '30'}%)"
                return AdvisorDecision(
                    action=step, reason=reason,
                    severity=SessionState.ACTION_NEEDED,
                    suggested_changes={"kv_cache_precision": new_prec},
                    revertible=True,
                )

        elif step == "improve_cache_reuse":
            if not cfg.prompt_caching:
                reason = "Enabling prompt caching to reduce redundant KV recomputation"
                return AdvisorDecision(
                    action=step, reason=reason,
                    severity=SessionState.WARNING,
                    suggested_changes={"prompt_caching": True},
                    revertible=True,
                )

        elif step == "disable_speculative_decoding":
            if cfg.speculative_decoding:
                reason = "Disabling speculative decoding to free draft-model memory"
                return AdvisorDecision(
                    action=step, reason=reason,
                    severity=SessionState.ACTION_NEEDED,
                    suggested_changes={"speculative_decoding": False},
                    revertible=True,
                )

        elif step == "lower_quantization":
            next_quant = QUANT_DEGRADE.get(cfg.quant)
            if next_quant:
                reason = f"Lowering quantization {cfg.quant}→{next_quant} (requires model reload)"
                return AdvisorDecision(
                    action=step, reason=reason,
                    severity=SessionState.ACTION_NEEDED,
                    suggested_changes={"quant": next_quant, "reload_required": True},
                    revertible=True,
                )

        elif step == "switch_to_smaller_model":
            reason = f"Memory critically low ({mem:.0f}%). Consider switching to a smaller model."
            return AdvisorDecision(
                action=step, reason=reason,
                severity=SessionState.CRITICAL,
                suggested_changes={"action": "switch_model"},
                revertible=False,
            )

        # Step not applicable, try next
        self._degrade_index += 1
        if self._degrade_index < len(DEGRADE_STEPS) - 1:
            return self._next_action(signals, m)

        return None

    def _log(self, message: str, level: str = "INFO") -> None:
        self._events.appendleft(SessionEvent(
            timestamp=time.time(),
            level=level,
            message=message,
        ))


def _kv_delta_gb(cfg: SessionConfig, old_ctx: int, new_ctx: int) -> float:
    """Estimate VRAM saved by reducing context from old_ctx to new_ctx."""
    if not cfg.n_total_layers:
        return 0.0
    # 2 * layers * n_kv_heads * head_dim * (old-new) * 2 bytes (f16)
    # We don't have n_kv_heads here so approximate
    bytes_per_token = cfg.kv_cache_gb * 1024**3 / max(cfg.context_len, 1)
    saved = bytes_per_token * (old_ctx - new_ctx)
    return saved / 1024**3
