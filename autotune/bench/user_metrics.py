"""
User-experience metrics for autotune benchmarks.

The question we answer is NOT "how many KV slots were freed?" —
it is "did the user feel the LLM running?"

Seven KPIs that map to real user perceptions
--------------------------------------------
1. swap_events          → 0 means "my computer never choked"
2. ram_headroom_gb      → GB free for Chrome, Slack, etc. during inference
3. ttft_ms              → milliseconds until first word appears
4. ttft_consistency_pct → how predictable response times are (100% = never varies)
5. cpu_spike_events     → times CPU went above 80% for >2 s (fans / heat)
6. memory_recovery_sec  → seconds for RAM to settle back after a call
7. background_impact    → composite 0–100 score (100 = zero user impact)

All of these are translatable to plain English:
  "Your computer will never slow down while autotune is running."
  "You keep N GB free for your other apps."
  "Responses arrive in X ms, consistently."
"""

from __future__ import annotations

import statistics
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import psutil


# ---------------------------------------------------------------------------
# Low-level sampler
# ---------------------------------------------------------------------------

class _LiveSampler:
    """
    Samples system state at `interval_sec` in a daemon thread.

    Records every swap event (any increase in swap usage ≥ SWAP_EVENT_THRESHOLD).
    Records every CPU spike (CPU > CPU_SPIKE_THRESHOLD for > CPU_SPIKE_MIN_SEC).
    """

    SWAP_EVENT_THRESHOLD = 0.01   # GB — increases above this count as events
    CPU_SPIKE_THRESHOLD  = 80.0   # percent
    CPU_SPIKE_MIN_SEC    = 2.0    # seconds of sustained spike to count

    def __init__(self, interval_sec: float = 0.25) -> None:
        self.interval = interval_sec
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

        # Raw samples
        self._ram_gb:  list[float] = []
        self._swap_gb: list[float] = []
        self._cpu_pct: list[float] = []
        self._timestamps: list[float] = []

        # Derived events (populated lazily in stop())
        self._swap_events: list[SwapEvent] = []
        self._cpu_spike_events: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._start_time = time.monotonic()
        # Prime cpu_percent (first call always returns 0.0)
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._analyse()

    def _loop(self) -> None:
        while self._running:
            vm  = psutil.virtual_memory()
            sw  = psutil.swap_memory()
            cpu = psutil.cpu_percent(interval=None)
            now = time.monotonic() - self._start_time
            self._ram_gb.append(vm.used     / 1024**3)
            self._swap_gb.append(sw.used    / 1024**3)
            self._cpu_pct.append(cpu)
            self._timestamps.append(now)
            time.sleep(self.interval)

    def _analyse(self) -> None:
        # ── Swap events ───────────────────────────────────────────────────────
        self._swap_events = []
        for i in range(1, len(self._swap_gb)):
            delta = self._swap_gb[i] - self._swap_gb[i - 1]
            if delta >= self.SWAP_EVENT_THRESHOLD:
                self._swap_events.append(SwapEvent(
                    timestamp_sec=round(self._timestamps[i], 2),
                    delta_gb=round(delta, 3),
                    swap_total_gb=round(self._swap_gb[i], 3),
                ))

        # ── CPU spike events ──────────────────────────────────────────────────
        spike_start: Optional[float] = None
        spike_count = 0
        for i, (cpu, ts) in enumerate(zip(self._cpu_pct, self._timestamps)):
            if cpu >= self.CPU_SPIKE_THRESHOLD:
                if spike_start is None:
                    spike_start = ts
            else:
                if spike_start is not None:
                    duration = ts - spike_start
                    if duration >= self.CPU_SPIKE_MIN_SEC:
                        spike_count += 1
                    spike_start = None
        # close any open spike at end of recording
        if spike_start is not None:
            duration = self._timestamps[-1] - spike_start if self._timestamps else 0.0
            if duration >= self.CPU_SPIKE_MIN_SEC:
                spike_count += 1
        self._cpu_spike_events = spike_count

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def swap_events(self) -> list["SwapEvent"]:
        return self._swap_events

    @property
    def cpu_spike_events(self) -> int:
        return self._cpu_spike_events

    def ram_before_gb(self) -> float:
        return self._ram_gb[0] if self._ram_gb else 0.0

    def ram_peak_gb(self) -> float:
        return max(self._ram_gb) if self._ram_gb else 0.0

    def ram_after_gb(self) -> float:
        return self._ram_gb[-1] if self._ram_gb else 0.0

    def swap_before_gb(self) -> float:
        return self._swap_gb[0] if self._swap_gb else 0.0

    def swap_peak_gb(self) -> float:
        return max(self._swap_gb) if self._swap_gb else 0.0

    def cpu_avg_pct(self) -> float:
        return statistics.mean(self._cpu_pct) if self._cpu_pct else 0.0

    def cpu_peak_pct(self) -> float:
        return max(self._cpu_pct) if self._cpu_pct else 0.0

    def ram_headroom_gb(self, total_ram_gb: float) -> float:
        """
        Minimum RAM headroom available for other apps during inference.
        = total - peak_used
        """
        return round(max(0.0, total_ram_gb - self.ram_peak_gb()), 2)

    def ram_recovery_sec(self, threshold_gb: float = 0.1) -> float:
        """
        Seconds after the sampler stopped until RAM settled within
        `threshold_gb` of the pre-inference baseline.

        If we never sampled post-inference, returns 0 (not measured).
        """
        if not self._ram_gb:
            return 0.0
        baseline = self._ram_gb[0]
        # Walk backward to find when RAM first returned to near baseline
        for i in range(len(self._ram_gb) - 1, 0, -1):
            if abs(self._ram_gb[i] - baseline) <= threshold_gb:
                return round(self._timestamps[i] - self._timestamps[0], 1)
        return round(self._timestamps[-1] - self._timestamps[0], 1) if self._timestamps else 0.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SwapEvent:
    """A detected increase in swap usage during inference."""
    timestamp_sec: float   # seconds since inference started
    delta_gb: float        # how much swap increased (GB)
    swap_total_gb: float   # total swap in use at this moment


@dataclass
class TurnMetrics:
    """Per-turn metrics for a multi-turn conversation or agent loop."""
    turn_number: int
    ttft_ms: float
    completion_tokens: int
    prompt_tokens: int
    ram_delta_gb: float       # RAM change from this turn (+= grew, -= shrank)
    swap_events: int          # swap events during this turn
    cpu_spike_events: int
    elapsed_sec: float


@dataclass
class UserExperienceReport:
    """
    The complete user-experience report card for one benchmark scenario.

    All numbers are in units a non-technical user can understand.
    """
    scenario: str               # e.g. "Normal laptop use — 5-turn chat"
    model_id: str
    profile_name: str

    # ── The 7 headline KPIs ───────────────────────────────────────────────────
    swap_events_total: int       # GOAL: 0 — "computer never choked"
    ram_headroom_gb: float       # GB free for Chrome/Slack during inference
    ttft_ms_mean: float          # mean time to first word (ms)
    ttft_ms_p95: float           # 95th-percentile TTFT (worst response)
    ttft_consistency_pct: float  # 100% = never varies; lower = unpredictable
    cpu_spike_events_total: int  # times CPU >80% for >2 s ("fans spun up")
    memory_recovery_sec: float   # how fast RAM settles after inference

    # ── Background impact score ───────────────────────────────────────────────
    background_impact_score: float   # 0–100; 100 = completely invisible to other apps

    # ── Per-turn breakdown ────────────────────────────────────────────────────
    turns: list[TurnMetrics] = field(default_factory=list)

    # ── Raw system snapshots ──────────────────────────────────────────────────
    total_ram_gb: float = 0.0
    ram_before_gb: float = 0.0
    ram_peak_gb: float = 0.0
    ram_after_gb: float = 0.0
    swap_before_gb: float = 0.0
    swap_peak_gb: float = 0.0
    cpu_avg_pct: float = 0.0
    cpu_peak_pct: float = 0.0
    elapsed_total_sec: float = 0.0

    # ── Narrative ─────────────────────────────────────────────────────────────
    verdict: str = ""            # one-sentence human summary

    def __post_init__(self) -> None:
        if not self.verdict:
            self.verdict = _build_verdict(self)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "turns"}
        d["turns"] = [t.__dict__ for t in self.turns]
        return d

    def print_card(self) -> None:
        """Print a user-facing report card to stdout."""
        _print_report_card(self)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_background_impact_score(
    swap_events: int,
    ram_headroom_gb: float,
    cpu_spike_events: int,
    ttft_consistency_pct: float,
    total_ram_gb: float = 16.0,
) -> float:
    """
    Composite 0–100 score of how invisible the LLM is to the user's other apps.

    Scoring breakdown (each sub-score 0–100, weights sum to 1.0):
      40%  swap events         (0 events = 100, every event = −20 pts, floor 0)
      30%  RAM headroom        (≥4 GB = 100, 0 GB = 0, linear)
      20%  CPU spike events    (0 = 100, each event = −25 pts, floor 0)
      10%  TTFT consistency    (pass-through of ttft_consistency_pct)
    """
    # Swap: each event costs 20 points
    swap_score = max(0.0, 100.0 - swap_events * 20.0)

    # RAM headroom: linear 0–4 GB → 0–100
    target_headroom_gb = 4.0
    headroom_score = min(100.0, (ram_headroom_gb / target_headroom_gb) * 100.0)

    # CPU spikes: each event costs 25 points
    cpu_score = max(0.0, 100.0 - cpu_spike_events * 25.0)

    # TTFT consistency: pass-through
    consistency_score = max(0.0, min(100.0, ttft_consistency_pct))

    score = (
        0.40 * swap_score
        + 0.30 * headroom_score
        + 0.20 * cpu_score
        + 0.10 * consistency_score
    )
    return round(score, 1)


def compute_ttft_consistency(ttft_values: list[float]) -> float:
    """
    Consistency percentage: 100% = perfectly stable, lower = unpredictable.

    Uses coefficient of variation (CV = stddev / mean). CV of 0 → 100%,
    CV of 1.0 → 0%, clamped to [0, 100].
    """
    if len(ttft_values) < 2:
        return 100.0
    mean = statistics.mean(ttft_values)
    if mean == 0:
        return 100.0
    cv = statistics.stdev(ttft_values) / mean
    return round(max(0.0, min(100.0, (1.0 - cv) * 100.0)), 1)


# ---------------------------------------------------------------------------
# Verdict builder
# ---------------------------------------------------------------------------

def _build_verdict(r: UserExperienceReport) -> str:
    if r.swap_events_total == 0 and r.background_impact_score >= 85:
        return (
            f"✅ Invisible to your other apps — zero swap events, "
            f"{r.ram_headroom_gb:.1f} GB free for Chrome/Slack, "
            f"responses in {r.ttft_ms_mean:.0f} ms."
        )
    elif r.swap_events_total == 0 and r.background_impact_score >= 60:
        return (
            f"⚠️  No swap, but RAM headroom is tight ({r.ram_headroom_gb:.1f} GB free). "
            f"Close a browser tab or two before a long session."
        )
    elif r.swap_events_total > 0:
        return (
            f"❌ {r.swap_events_total} swap event(s) detected — "
            f"computer may have felt sluggish. "
            f"Try `--profile fast` or a smaller model."
        )
    else:
        return f"Score: {r.background_impact_score:.0f}/100"


# ---------------------------------------------------------------------------
# Report card printer
# ---------------------------------------------------------------------------

def _print_report_card(r: UserExperienceReport) -> None:
    W = 64
    line = "─" * W

    print(f"\n{'━' * W}")
    print(f"  USER EXPERIENCE REPORT  ·  {r.scenario}")
    print(f"  Model: {r.model_id}  ·  Profile: {r.profile_name}")
    print(f"{'━' * W}")

    print(f"\n  {'KPI':<38} {'Value':>10}  {'Goal':>8}")
    print(f"  {line}")

    def row(label: str, value: str, goal: str, ok: bool) -> None:
        flag = "✅" if ok else "❌"
        print(f"  {flag}  {label:<36} {value:>10}  {goal:>8}")

    row(
        "Swap events (computer slowdown)",
        str(r.swap_events_total),
        "0",
        r.swap_events_total == 0,
    )
    row(
        "RAM free for your other apps",
        f"{r.ram_headroom_gb:.1f} GB",
        "≥ 4 GB",
        r.ram_headroom_gb >= 4.0,
    )
    row(
        "Time to first word (avg)",
        f"{r.ttft_ms_mean:.0f} ms",
        "< 2000 ms",
        r.ttft_ms_mean < 2000,
    )
    row(
        "Time to first word (worst)",
        f"{r.ttft_ms_p95:.0f} ms",
        "< 4000 ms",
        r.ttft_ms_p95 < 4000,
    )
    row(
        "Response consistency",
        f"{r.ttft_consistency_pct:.0f}%",
        "≥ 70%",
        r.ttft_consistency_pct >= 70,
    )
    row(
        "CPU spikes (fans / heat)",
        str(r.cpu_spike_events_total),
        "0",
        r.cpu_spike_events_total == 0,
    )
    row(
        "Memory recovery time",
        f"{r.memory_recovery_sec:.1f} s",
        "< 10 s",
        r.memory_recovery_sec < 10,
    )

    print(f"\n  Background impact score:  {r.background_impact_score:>5.0f} / 100")
    print(f"\n  {r.verdict}")
    print(f"{'━' * W}\n")


# ---------------------------------------------------------------------------
# Factory — build a UserExperienceReport from turn data
# ---------------------------------------------------------------------------

def build_report(
    scenario: str,
    model_id: str,
    profile_name: str,
    turns: list[TurnMetrics],
    total_ram_gb: float,
    ram_before_gb: float,
    ram_peak_gb: float,
    ram_after_gb: float,
    swap_before_gb: float,
    swap_peak_gb: float,
    cpu_avg_pct: float,
    cpu_peak_pct: float,
    elapsed_total_sec: float,
    memory_recovery_sec: float,
) -> UserExperienceReport:
    ttft_values = [t.ttft_ms for t in turns if t.ttft_ms > 0]
    ttft_mean   = round(statistics.mean(ttft_values), 1) if ttft_values else 0.0
    ttft_p95    = round(sorted(ttft_values)[int(len(ttft_values) * 0.95)] if ttft_values else 0.0, 1)
    consistency = compute_ttft_consistency(ttft_values)
    swap_total  = sum(t.swap_events for t in turns)
    cpu_spikes  = sum(t.cpu_spike_events for t in turns)

    ram_headroom = max(0.0, total_ram_gb - ram_peak_gb)

    bg_score = compute_background_impact_score(
        swap_events=swap_total,
        ram_headroom_gb=ram_headroom,
        cpu_spike_events=cpu_spikes,
        ttft_consistency_pct=consistency,
        total_ram_gb=total_ram_gb,
    )

    return UserExperienceReport(
        scenario=scenario,
        model_id=model_id,
        profile_name=profile_name,
        swap_events_total=swap_total,
        ram_headroom_gb=round(ram_headroom, 2),
        ttft_ms_mean=ttft_mean,
        ttft_ms_p95=ttft_p95,
        ttft_consistency_pct=consistency,
        cpu_spike_events_total=cpu_spikes,
        memory_recovery_sec=round(memory_recovery_sec, 1),
        background_impact_score=bg_score,
        turns=turns,
        total_ram_gb=total_ram_gb,
        ram_before_gb=round(ram_before_gb, 2),
        ram_peak_gb=round(ram_peak_gb, 2),
        ram_after_gb=round(ram_after_gb, 2),
        swap_before_gb=round(swap_before_gb, 3),
        swap_peak_gb=round(swap_peak_gb, 3),
        cpu_avg_pct=round(cpu_avg_pct, 1),
        cpu_peak_pct=round(cpu_peak_pct, 1),
        elapsed_total_sec=round(elapsed_total_sec, 1),
    )
