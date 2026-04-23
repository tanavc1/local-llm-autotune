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

Health score (0–100):
  Composite metric giving a single at-a-glance number for LLM inference health.
  90–100 = Running smoothly
  75–89  = Moderate load
  55–74  = Memory pressure building
  35–54  = Stressed — action recommended
  0–34   = Critical — immediate action needed
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .types import (
    AdvisorDecision,
    LiveMetrics,
    SessionConfig,
    SessionEvent,
    SessionState,
    ThermalState,
)

# ---------------------------------------------------------------------------
# Health scoring
# ---------------------------------------------------------------------------

def compute_health_score(metrics: LiveMetrics) -> int:
    """
    Compute a 0–100 health score for LLM inference on this machine.

    Weights:
      RAM pressure   40 pts  (most critical for LLM — OOM kills inference)
      Swap usage     30 pts  (any swap = severe slowdown for LLM)
      Thermal state  15 pts  (throttling = slower tokens)
      CPU overhead   10 pts  (GPU handles generation; CPU = overhead)
      RAM growth      5 pts  (trending toward trouble)
    """
    score = 100.0

    # RAM pressure — up to -40 pts
    ram_pct = metrics.vram_percent if metrics.vram_percent is not None else metrics.ram_percent
    if ram_pct >= 97:
        score -= 40
    elif ram_pct >= 94:
        score -= 30
    elif ram_pct >= 90:
        score -= 18
    elif ram_pct >= 85:
        score -= 10
    elif ram_pct >= 80:
        score -= 5

    # Swap usage — up to -30 pts (any swap is very bad for LLM)
    if metrics.swap_percent >= 20:
        score -= 30
    elif metrics.swap_percent >= 10:
        score -= 22
    elif metrics.swap_percent >= 3:
        score -= 14
    elif metrics.swap_used_gb > 0.5:
        score -= 8
    elif metrics.swap_used_gb > 0.1:
        score -= 3

    # Swap growth — additional penalty for active paging
    if metrics.swap_growth_mb_per_min >= 50:
        score -= 10
    elif metrics.swap_growth_mb_per_min >= 10:
        score -= 5

    # Thermal — up to -15 pts
    if metrics.thermal_state == ThermalState.CRITICAL:
        score -= 15
    elif metrics.thermal_state == ThermalState.THROTTLING:
        score -= 12
    elif metrics.thermal_state in (ThermalState.WARNING,):
        score -= 6
    elif metrics.thermal_state == ThermalState.WARM:
        score -= 2

    # CPU overhead — up to -10 pts (GPU-bound inference; high CPU = competing load)
    if metrics.cpu_percent >= 90:
        score -= 10
    elif metrics.cpu_percent >= 75:
        score -= 5
    elif metrics.cpu_percent >= 60:
        score -= 2

    # RAM growth trajectory — up to -5 pts
    if metrics.ram_growth_mb_per_min > 300:
        score -= 5
    elif metrics.ram_growth_mb_per_min > 150:
        score -= 2

    return max(0, min(100, int(score)))


def health_status(score: int) -> tuple[str, str, str]:
    """Return (label, color, icon) for a health score."""
    if score >= 90:
        return "Running smoothly", "green", "●"
    elif score >= 75:
        return "Moderate load — watching closely", "yellow", "▲"
    elif score >= 55:
        return "Memory pressure building", "bold yellow", "⚠"
    elif score >= 35:
        return "Stressed — action recommended", "red", "⚠"
    else:
        return "Critical — immediate action needed", "bold red", "⛔"

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
        self._events: deque[SessionEvent] = deque(maxlen=200)

        # Health tracking
        self._last_health_score: Optional[int] = None
        self._last_proactive_event_time: float = 0.0
        self._proactive_event_interval: float = 30.0   # seconds

        # Spike detection
        self._prev_ram_used_gb: Optional[float] = None
        self._prev_cpu_pct: Optional[float] = None
        self._prev_swap_gb: Optional[float] = None

        # Threshold crossing detection
        self._prev_ram_pct: Optional[float] = None
        self._ram_warn_fired: bool = False
        self._ram_action_fired: bool = False
        self._swap_warn_fired: bool = False

        # Ollama model change tracking — for attributing RAM spikes
        self._prev_ollama_model_names: Optional[frozenset] = None

        # Performance degradation tracking — hysteresis flags
        self._tps_drop_warned: bool = False
        self._ttft_rise_warned: bool = False

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
        self._emit_proactive_events(metrics)
        self._emit_spike_events(metrics)
        self._emit_threshold_events(metrics)
        self._emit_performance_events(metrics)
        signals = self._evaluate_signals(metrics)
        decisions = self._decide(signals, metrics)

        for d in decisions:
            self._decisions.appendleft(d)
            self._log(d.reason, d.severity.value.upper())

        # Update spike tracking
        self._prev_ram_used_gb = metrics.ram_used_gb
        self._prev_cpu_pct = metrics.cpu_percent
        self._prev_swap_gb = metrics.swap_used_gb
        self._prev_ram_pct = metrics.ram_percent
        self._prev_ollama_model_names = frozenset(m.name for m in metrics.ollama_models)

        return decisions

    # ------------------------------------------------------------------ #
    # Proactive event generation                                           #
    # ------------------------------------------------------------------ #

    def _emit_proactive_events(self, m: LiveMetrics) -> None:
        """Emit a health check event every ~30 seconds regardless of state."""
        now = time.time()
        if (now - self._last_proactive_event_time) < self._proactive_event_interval:
            return

        score = compute_health_score(m)
        label, color, icon = health_status(score)
        self._last_health_score = score

        ram_pct = m.vram_percent if m.vram_percent is not None else m.ram_percent

        # Build a compact one-line status
        swap_note = f"  ·  Swap {m.swap_used_gb:.2f} GB" if m.swap_used_gb > 0.05 else ""
        thermal_note = ""
        if m.thermal_state not in (ThermalState.NOMINAL, ThermalState.WARM):
            thermal_note = f"  ·  Thermal {m.thermal_state.value}"

        level = "OK" if score >= 75 else ("WARN" if score >= 50 else "ACTION")
        self._log(
            f"Health {score}/100 — {label}"
            f"  (RAM {ram_pct:.0f}%  ·  CPU {m.cpu_percent:.0f}%{swap_note}{thermal_note})",
            level,
        )
        self._last_proactive_event_time = now

    def _emit_spike_events(self, m: LiveMetrics) -> None:
        """
        Detect sudden changes in RAM/CPU/swap and log attribution-aware alerts.

        Rather than guessing, we correlate RAM jumps with Ollama model load/unload
        events and with whether an inference session is actively running.
        """
        now_ollama = frozenset(om.name for om in m.ollama_models)
        prev_ollama = self._prev_ollama_model_names   # None on first tick

        # ── RAM spike / drop ────────────────────────────────────────────────
        if self._prev_ram_used_gb is not None:
            delta_ram = m.ram_used_gb - self._prev_ram_used_gb

            if delta_ram > 0.4:
                if prev_ollama is not None:
                    new_models = now_ollama - prev_ollama
                    if new_models:
                        # A model was loaded into Ollama this tick
                        name = sorted(new_models)[0]
                        size_info = ""
                        for om in m.ollama_models:
                            if om.name == name:
                                size_info = f" ({om.size_gb:.1f} GB weights"
                                if om.context_len:
                                    size_info += f", ctx {om.context_len:,}"
                                size_info += ")"
                                break
                        self._log(
                            f"RAM +{delta_ram:.1f} GB "
                            f"({self._prev_ram_used_gb:.1f}→{m.ram_used_gb:.1f} GB) "
                            f"— Ollama loaded {name}{size_info}",
                            "INFO",
                        )
                    elif m.ollama_models:
                        # A model is running but didn't change — KV/context growth
                        running = m.ollama_models[0]
                        ctx_note = (
                            f" (ctx {running.context_len:,} tokens)"
                            if running.context_len else ""
                        )
                        self._log(
                            f"RAM +{delta_ram:.1f} GB "
                            f"({self._prev_ram_used_gb:.1f}→{m.ram_used_gb:.1f} GB) "
                            f"— KV cache growth while {running.name.split(':')[0]} is running{ctx_note}",
                            "WARN",
                        )
                    else:
                        self._log(
                            f"RAM +{delta_ram:.1f} GB "
                            f"({self._prev_ram_used_gb:.1f}→{m.ram_used_gb:.1f} GB) "
                            f"— model loading or background application",
                            "WARN",
                        )
                else:
                    self._log(
                        f"RAM +{delta_ram:.1f} GB "
                        f"({self._prev_ram_used_gb:.1f}→{m.ram_used_gb:.1f} GB)",
                        "WARN",
                    )

            elif delta_ram < -0.4:
                if prev_ollama is not None:
                    gone_models = prev_ollama - now_ollama
                    if gone_models:
                        name = sorted(gone_models)[0]
                        self._log(
                            f"RAM −{abs(delta_ram):.1f} GB "
                            f"({self._prev_ram_used_gb:.1f}→{m.ram_used_gb:.1f} GB) "
                            f"— Ollama unloaded {name}, weights freed",
                            "OK",
                        )
                    else:
                        self._log(
                            f"RAM −{abs(delta_ram):.1f} GB "
                            f"({self._prev_ram_used_gb:.1f}→{m.ram_used_gb:.1f} GB) "
                            f"— cache cleared or application closed",
                            "OK",
                        )
                else:
                    self._log(
                        f"RAM −{abs(delta_ram):.1f} GB "
                        f"({self._prev_ram_used_gb:.1f}→{m.ram_used_gb:.1f} GB) "
                        f"— model unloaded or cache cleared",
                        "OK",
                    )

        # ── CPU spike ──────────────────────────────────────────────────────
        if self._prev_cpu_pct is not None:
            delta_cpu = m.cpu_percent - self._prev_cpu_pct
            if delta_cpu > 30 and self._prev_cpu_pct < 40:
                cause = "inference burst" if m.ollama_models else "background process"
                self._log(
                    f"CPU spike: {self._prev_cpu_pct:.0f}%→{m.cpu_percent:.0f}% — {cause}",
                    "INFO",
                )

        # ── Swap state changes ─────────────────────────────────────────────
        if self._prev_swap_gb is not None:
            if self._prev_swap_gb < 0.05 and m.swap_used_gb >= 0.1:
                model_note = ""
                if m.ollama_models:
                    model_note = (
                        f" — {m.ollama_models[0].name.split(':')[0]} "
                        f"is exceeding available RAM"
                    )
                self._log(
                    f"Swap started: {m.swap_used_gb:.2f} GB now on disk "
                    f"— RAM exhausted, OS is paging to NVMe; "
                    f"inference will slow significantly{model_note}",
                    "WARN",
                )
            elif self._prev_swap_gb >= 0.1 and m.swap_used_gb < 0.05:
                self._log(
                    "Swap cleared — memory pressure resolved, "
                    "inference speed should recover",
                    "OK",
                )

    def _emit_threshold_events(self, m: LiveMetrics) -> None:
        """Fire events when RAM crosses warning/action thresholds."""
        ram_pct = m.vram_percent if m.vram_percent is not None else m.ram_percent

        # RAM warning zone (first time above 80%)
        if not self._ram_warn_fired and ram_pct >= MEM_WARN_PCT:
            self._log(
                f"RAM crossed {MEM_WARN_PCT:.0f}% — entering caution zone  "
                f"({m.ram_used_gb:.1f} / {m.ram_total_gb:.0f} GB)  "
                f"·  Consider closing other apps or reducing context",
                "WARN",
            )
            self._ram_warn_fired = True
        elif self._ram_warn_fired and ram_pct < MEM_WARN_PCT - 3:
            self._log(
                f"RAM back below {MEM_WARN_PCT:.0f}% — pressure eased  "
                f"({m.ram_used_gb:.1f} GB)",
                "OK",
            )
            self._ram_warn_fired = False

        # RAM action zone (first time above 88%)
        if not self._ram_action_fired and ram_pct >= MEM_ACTION_PCT:
            self._log(
                f"RAM crossed {MEM_ACTION_PCT:.0f}% — high pressure  "
                f"({m.ram_used_gb:.1f} / {m.ram_total_gb:.0f} GB)  "
                f"·  Reduce context window or switch to a lighter quant",
                "ACTION",
            )
            self._ram_action_fired = True
        elif self._ram_action_fired and ram_pct < MEM_ACTION_PCT - 3:
            self._log(
                f"RAM dropped below {MEM_ACTION_PCT:.0f}% — pressure eased",
                "OK",
            )
            self._ram_action_fired = False

    def _emit_performance_events(self, m: LiveMetrics) -> None:
        """
        Detect throughput and TTFT degradation vs baseline and emit explanatory events.

        Only fires after baseline is established (≥5 samples) to avoid false positives
        during warm-up.  Uses hysteresis so the same event is not spammed.
        """
        b = self.baseline
        if b.sample_count < 5:
            return

        ram_pct = m.vram_percent if m.vram_percent is not None else m.ram_percent

        # ── TPS degradation / recovery ─────────────────────────────────────
        if b.tokens_per_sec and m.tokens_per_sec:
            ratio = m.tokens_per_sec / b.tokens_per_sec
            pct_drop = (1.0 - ratio) * 100

            if ratio < TPS_DROP_ACTION and not self._tps_drop_warned:
                # Attribute the slowdown to the most likely cause
                if m.swap_used_gb > 0.1:
                    cause = (
                        f" — {m.swap_used_gb:.1f} GB swap in use; "
                        f"model KV cache is being paged to NVMe"
                    )
                elif ram_pct >= MEM_ACTION_PCT:
                    cause = (
                        f" — RAM at {ram_pct:.0f}%; "
                        f"OS page cache is competing with inference"
                    )
                elif m.thermal_state in (ThermalState.THROTTLING, ThermalState.CRITICAL):
                    cause = f" — CPU thermally throttled to {m.cpu_speed_limit_pct}% of max speed"
                elif m.cpu_percent > 85:
                    cause = f" — CPU at {m.cpu_percent:.0f}%; background processes competing"
                else:
                    cause = ""
                self._log(
                    f"Throughput dropped: {m.tokens_per_sec:.1f} tok/s "
                    f"(was {b.tokens_per_sec:.1f}, −{pct_drop:.0f}%){cause}",
                    "WARN",
                )
                self._tps_drop_warned = True

            elif ratio > 0.85 and self._tps_drop_warned:
                self._log(
                    f"Throughput recovered: {m.tokens_per_sec:.1f} tok/s "
                    f"(baseline {b.tokens_per_sec:.1f})",
                    "OK",
                )
                self._tps_drop_warned = False

        # ── TTFT degradation / recovery ────────────────────────────────────
        if b.ttft_ms and m.ttft_ms:
            ratio = m.ttft_ms / b.ttft_ms

            if ratio > TTFT_RISE_ACTION and not self._ttft_rise_warned:
                if m.swap_used_gb > 0.1:
                    cause = " — swap is forcing KV data to load from disk each request"
                elif ram_pct >= MEM_ACTION_PCT:
                    cause = f" — RAM at {ram_pct:.0f}%; KV allocation is slow"
                else:
                    cause = ""
                self._log(
                    f"First-token delay: {m.ttft_ms:.0f} ms "
                    f"(was {b.ttft_ms:.0f} ms, {ratio:.1f}× slower){cause}",
                    "WARN",
                )
                self._ttft_rise_warned = True

            elif ratio < 1.4 and self._ttft_rise_warned:
                self._log(
                    f"TTFT normalized: {m.ttft_ms:.0f} ms (was {b.ttft_ms:.0f} ms baseline)",
                    "OK",
                )
                self._ttft_rise_warned = False

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
                kv_saved = _kv_delta_gb(cfg, current_ctx, smaller)
                kv_str = f" (~{kv_saved:.2f} GB KV freed)" if kv_saved > 0.01 else ""
                reason = (
                    f"Context {current_ctx:,}→{smaller:,} tokens{kv_str} "
                    f"— RAM at {mem:.0f}%, KV cache is the largest lever"
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
                # Q8 = ~50% of F16 footprint; Q4 = ~25%
                savings_pct = 50 if new_prec == "q8" else 75
                savings_gb = cfg.kv_cache_gb * (savings_pct / 100)
                savings_str = (
                    f"~{savings_gb:.1f} GB freed" if savings_gb > 0.1
                    else f"~{savings_pct}% smaller KV"
                )
                reason = (
                    f"KV precision {cfg.kv_cache_precision.upper()}→{new_prec.upper()} "
                    f"({savings_str}, RAM {mem:.0f}%)"
                )
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
