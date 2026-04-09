"""
Rich live terminal dashboard for the session controller.

Redesigned to prioritise actionable information:
  - Health score (0-100) in the header — at-a-glance machine state
  - Plain-English status label ("Running smoothly", "Memory pressure", etc.)
  - Active LLM panel: what Ollama has loaded, how much memory it's using
  - Memory panel with trend arrows (↑ growing / ↓ dropping / → stable)
  - CPU / thermal panel
  - Event log with proactive 30-second health checks
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autotune.hardware.profiler import HardwareProfile
from .types import (
    AdvisorDecision, LiveMetrics, SessionConfig,
    SessionEvent, SessionState, ThermalState,
)
from .advisor import compute_health_score, health_status


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _pct_color(pct: float, warn: float = 75.0, crit: float = 88.0) -> str:
    if pct >= crit:
        return "bold red"
    if pct >= warn:
        return "yellow"
    return "green"


def _bar(pct: float, width: int = 20, warn: float = 75.0, crit: float = 88.0) -> Text:
    filled = round(max(0.0, min(1.0, pct / 100)) * width)
    empty = width - filled
    color = _pct_color(pct, warn, crit)
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * empty, style="dim")
    return t


def _health_bar(score: int, width: int = 20) -> Text:
    filled = round(score / 100 * width)
    empty = width - filled
    color = "green" if score >= 90 else "yellow" if score >= 55 else "bold red"
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * empty, style="dim")
    return t


def _trend_arrow(rate_mb_per_min: float) -> Text:
    """Return a colored trend arrow based on growth rate."""
    if abs(rate_mb_per_min) < 2.0:
        return Text("→ stable", style="dim")
    if rate_mb_per_min > 50:
        return Text(f"↑↑ +{rate_mb_per_min:.0f} MB/min", style="bold red")
    if rate_mb_per_min > 10:
        return Text(f"↑ +{rate_mb_per_min:.0f} MB/min", style="yellow")
    if rate_mb_per_min > 0:
        return Text(f"↑ +{rate_mb_per_min:.0f} MB/min", style="dim green")
    return Text(f"↓ {rate_mb_per_min:.0f} MB/min", style="cyan")


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _temp_str(t: Optional[float]) -> str:
    if t is None:
        return "N/A"
    color = "bold red" if t > 90 else "yellow" if t > 75 else "green"
    return f"[{color}]{t:.0f}°C[/{color}]"


def _gb(v: Optional[float]) -> str:
    return f"{v:.2f} GB" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Header panel — health score front and center
# ---------------------------------------------------------------------------

def _header_panel(hw: HardwareProfile, score: int, state: SessionState, start_time: float) -> Panel:
    label, color, icon = health_status(score)

    gpu_label = f"{hw.gpu.name}" if hw.gpu else "CPU-only"
    if hw.gpu and hw.gpu.is_unified_memory:
        gpu_label += " [unified mem]"

    t = Text()
    t.append("  autotune live  ", style="bold white on blue")
    t.append(f"  {hw.cpu.brand.split('@')[0].strip()[:28]}  ·  "
             f"{hw.memory.total_gb:.0f} GB  ·  {gpu_label}  ", style="dim")
    t.append(f"⏱ {_elapsed(start_time)}", style="cyan")
    t.append("  │  ")

    # Health score — the most important single number
    score_color = "bold green" if score >= 90 else "bold yellow" if score >= 55 else "bold red"
    t.append(f"Health ", style="dim")
    t.append(f"{score}/100", style=score_color)
    t.append("  │  ")
    t.append(f"{icon} {label}", style=color)
    t.append("  │  Ctrl-C to exit", style="dim")

    return Panel(t, box=box.HORIZONTALS, padding=(0, 1))


# ---------------------------------------------------------------------------
# Memory panel — with trend arrows
# ---------------------------------------------------------------------------

def _memory_panel(m: LiveMetrics, is_unified: bool) -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(width=7, style="bold")
    t.add_column(width=22)
    t.add_column(width=7, justify="right")
    t.add_column()

    # RAM
    ram_color = _pct_color(m.ram_percent)
    t.add_row(
        "RAM",
        _bar(m.ram_percent),
        f"[{ram_color}]{m.ram_percent:.0f}%[/]",
        Text.assemble(
            (f"{m.ram_used_gb:.1f}/{m.ram_total_gb:.0f} GB  ", "dim"),
            _trend_arrow(m.ram_growth_mb_per_min),
        ),
    )

    # GPU / Unified memory
    if m.vram_percent is not None and not is_unified:
        vram_color = _pct_color(m.vram_percent)
        t.add_row(
            "VRAM",
            _bar(m.vram_percent),
            f"[{vram_color}]{m.vram_percent:.0f}%[/]",
            Text.assemble(
                (_gb(m.vram_used_gb) + " / " + _gb(m.vram_total_gb) + "  ", "dim"),
                _trend_arrow(m.vram_growth_mb_per_min),
            ),
        )
    elif is_unified and m.vram_percent is not None:
        t.add_row(
            "GPU=RAM",
            _bar(m.vram_percent),
            f"[{_pct_color(m.vram_percent)}]{m.vram_percent:.0f}%[/]",
            Text("unified memory pool", style="dim"),
        )

    # Swap
    swap_warn = 3.0 if m.swap_used_gb > 0.05 else 100.0
    swap_color = _pct_color(m.swap_percent, warn=swap_warn, crit=15.0)
    t.add_row(
        "Swap",
        _bar(m.swap_percent, warn=swap_warn, crit=15.0),
        f"[{swap_color}]{m.swap_percent:.0f}%[/]",
        Text.assemble(
            (f"{m.swap_used_gb:.2f}/{m.swap_total_gb:.1f} GB  ", "dim"),
            _trend_arrow(m.swap_growth_mb_per_min),
        ),
    )

    # Free memory line
    avail_color = _pct_color(m.ram_percent)
    free_note = f"{m.ram_available_gb:.1f} GB free"
    if m.swap_used_gb > 0.1:
        free_note += f"  [yellow]⚠ {m.swap_used_gb:.2f} GB swap in use[/yellow]"
    t.add_row("", "", "", f"[{avail_color}]{free_note}[/{avail_color}]")

    return Panel(t, title="[bold]Memory[/bold]", box=box.ROUNDED, padding=(0, 1))


# ---------------------------------------------------------------------------
# CPU / Thermals panel
# ---------------------------------------------------------------------------

def _cpu_thermal_panel(m: LiveMetrics) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(width=14, style="bold dim")
    t.add_column()

    cpu_color = _pct_color(m.cpu_percent, warn=70, crit=90)
    t.add_row("CPU avg", Text.assemble(
        _bar(m.cpu_percent, width=16, warn=70, crit=90),
        (f"  [{cpu_color}]{m.cpu_percent:.0f}%[/{cpu_color}]", ""),
    ))

    # Per-core mini-bar
    cores = m.cpu_per_core
    if cores:
        core_str = Text()
        for c in cores[:16]:
            color = "red" if c > 85 else "yellow" if c > 60 else "green"
            core_str.append("█" if c > 50 else "▄" if c > 25 else "░", style=color)
        t.add_row("Cores", core_str)

    t.add_row("CPU temp", Text.from_markup(_temp_str(m.cpu_temp_c)))
    t.add_row("GPU temp", Text.from_markup(_temp_str(m.gpu_temp_c)))

    ts = m.thermal_state
    if ts == ThermalState.NOMINAL:
        thermal_text = Text("● Nominal — no throttling", style="green")
    elif ts == ThermalState.WARM:
        thermal_text = Text("~ Warm — slight throttle risk", style="yellow")
    elif ts == ThermalState.THROTTLING:
        thermal_text = Text("⚠ Throttling — CPU slowed down!", style="bold red")
    elif ts == ThermalState.CRITICAL:
        thermal_text = Text("⛔ Critical thermal event", style="bold red blink")
    else:
        thermal_text = Text(ts.value, style="dim")
    t.add_row("Thermal", thermal_text)

    if m.cpu_speed_limit_pct < 100:
        t.add_row("Speed limit", Text(f"{m.cpu_speed_limit_pct}% of max", style="yellow"))

    return Panel(t, title="[bold]CPU / Thermals[/bold]", box=box.ROUNDED, padding=(0, 1))


# ---------------------------------------------------------------------------
# Device status panel — replaces the useless "Current Config" panel
# ---------------------------------------------------------------------------

def _device_status_panel(m: LiveMetrics, score: int, start_time: float) -> Panel:
    label, color, icon = health_status(score)

    t = Table.grid(padding=(0, 1))
    t.add_column(width=14, style="bold dim")
    t.add_column()

    # Health score with bar
    t.add_row(
        "Health",
        Text.assemble(
            _health_bar(score),
            (f"  [{color}]{score}/100[/{color}]", ""),
        ),
    )
    t.add_row("Status", Text(f"{icon} {label}", style=color))
    t.add_row("", Text(""))

    # Active LLM
    if m.ollama_models:
        t.add_row("[bold]Active LLM[/bold]", Text(""))
        for om in m.ollama_models[:2]:
            name_str = om.name[:24]
            ctx_str = f"ctx {om.context_len:,}" if om.context_len else ""
            t.add_row(
                f"  {name_str}",
                Text(f"{om.size_gb:.1f} GB weights  {ctx_str}", style="cyan"),
            )
        # Performance stats if available
        if m.tokens_per_sec is not None:
            tps_color = "green" if m.tokens_per_sec > 15 else "yellow"
            t.add_row(
                "  Throughput",
                Text(f"{m.tokens_per_sec:.1f} tok/s", style=tps_color),
            )
        if m.ttft_ms is not None:
            ttft_color = "green" if m.ttft_ms < 500 else "yellow" if m.ttft_ms < 2000 else "red"
            t.add_row(
                "  TTFT",
                Text(f"{m.ttft_ms:.0f} ms", style=ttft_color),
            )
    elif m.llm_processes:
        t.add_row("[bold]Active LLM[/bold]", Text(""))
        for proc in m.llm_processes[:2]:
            t.add_row(
                f"  {proc.runtime[:12]}",
                Text(
                    f"{proc.ram_gb:.1f} GB RAM  {proc.cpu_percent:.0f}% CPU  pid {proc.pid}",
                    style="cyan",
                ),
            )
    else:
        t.add_row("[dim]LLM[/dim]", Text("No active LLM detected", style="dim"))
        t.add_row("", Text("Start one: ollama run qwen3:8b", style="dim"))

    t.add_row("", Text(""))

    # Quick advice based on current state
    ram_pct = m.vram_percent if m.vram_percent is not None else m.ram_percent
    if ram_pct >= 94:
        t.add_row(
            "[red]⚠ Action[/red]",
            Text("Reduce context or switch to lighter quant", style="red"),
        )
    elif ram_pct >= 85:
        t.add_row(
            "[yellow]Tip[/yellow]",
            Text("Close unused apps to free RAM headroom", style="yellow"),
        )
    elif m.swap_used_gb > 0.1:
        t.add_row(
            "[yellow]Tip[/yellow]",
            Text("Swap in use — LLM speed may be degraded", style="yellow"),
        )
    elif score >= 90:
        t.add_row(
            "[green]Tip[/green]",
            Text("Machine is healthy — inference unrestricted", style="dim green"),
        )

    return Panel(t, title="[bold]Device Status[/bold]", box=box.ROUNDED, padding=(0, 1))


# ---------------------------------------------------------------------------
# Recommendations panel
# ---------------------------------------------------------------------------

def _advice_panel(decisions: list[AdvisorDecision]) -> Optional[Panel]:
    if not decisions:
        return None
    t = Table.grid(padding=(0, 1))
    t.add_column(width=14, style="bold")
    t.add_column()

    for d in decisions[:2]:
        severity_color = {
            SessionState.CRITICAL: "bold red",
            SessionState.ACTION_NEEDED: "red",
            SessionState.WARNING: "yellow",
        }.get(d.severity, "cyan")
        action_label = d.action.replace("_", " ").upper()
        t.add_row(f"[{severity_color}]{action_label}[/{severity_color}]", d.reason)
        ch = d.suggested_changes
        if ch:
            cmd_parts = []
            if "context_len" in ch:
                cmd_parts.append(f"set ctx → {ch['context_len']:,}")
            if "kv_cache_precision" in ch:
                cmd_parts.append(f"KV precision → {ch['kv_cache_precision']}")
            if "quant" in ch:
                cmd_parts.append(f"reload with {ch['quant']}")
            if "concurrency" in ch:
                cmd_parts.append(f"concurrency → {ch['concurrency']}")
            if cmd_parts:
                t.add_row("", Text("↳ " + "  ·  ".join(cmd_parts), style="dim"))

    return Panel(t, title="[bold yellow]⚡ Recommended Actions[/bold yellow]", box=box.ROUNDED, padding=(0, 1))


# ---------------------------------------------------------------------------
# Event log panel
# ---------------------------------------------------------------------------

def _events_panel(events: list[SessionEvent]) -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(width=11, style="dim")
    t.add_column(width=7)
    t.add_column()

    _level_styles: dict[str, tuple[str, str]] = {
        "INFO":     ("dim",        "INFO "),
        "OK":       ("green",      " OK  "),
        "WARN":     ("yellow",     "WARN "),
        "WARNING":  ("yellow",     "WARN "),
        "ACTION":   ("bold red",   " ACT "),
        "ACTION_NEEDED": ("bold red", " ACT "),
        "CRITICAL": ("bold red",   "CRIT "),
        "DEGRADING": ("yellow",    " DEG "),
        "OPTIMAL":  ("green",      " OK  "),
        "STABLE_RECOVERING": ("cyan", "RECV "),
    }

    shown = 0
    for ev in events[:14]:
        ts_str = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
        style, tag = _level_styles.get(ev.level, ("white", ev.level[:5]))
        t.add_row(ts_str, f"[{style}]{tag}[/{style}]", ev.message)
        shown += 1

    if shown == 0:
        t.add_row("", "[dim]—[/dim]", "[dim]Collecting first metrics…[/dim]")

    return Panel(t, title="[bold]Event Log[/bold]", box=box.ROUNDED, padding=(0, 1))


# ---------------------------------------------------------------------------
# Main dashboard renderer
# ---------------------------------------------------------------------------

class LiveDashboard:
    def __init__(
        self,
        hw: HardwareProfile,
        config: SessionConfig,
        start_time: float,
    ) -> None:
        self.hw = hw
        self.config = config
        self.start_time = start_time
        self._is_unified = hw.gpu.is_unified_memory if hw.gpu else False

    def render(
        self,
        metrics: Optional[LiveMetrics],
        state: SessionState,
        events: list[SessionEvent],
        decisions: list[AdvisorDecision],
    ) -> Layout:
        # Compute health score
        score = compute_health_score(metrics) if metrics else 85

        layout = Layout()

        has_advice = bool(decisions)
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="advice", size=5) if has_advice else Layout(name="advice", size=0),
            Layout(name="events", size=16),
        )

        layout["header"].update(_header_panel(self.hw, score, state, self.start_time))

        # Body: left (memory + cpu) | right (device status)
        layout["body"].split_row(
            Layout(name="left_col", ratio=3),
            Layout(name="right_col", ratio=2),
        )
        layout["left_col"].split_column(
            Layout(name="memory", ratio=3),
            Layout(name="cpu", ratio=2),
        )

        if metrics:
            layout["memory"].update(_memory_panel(metrics, self._is_unified))
            layout["cpu"].update(_cpu_thermal_panel(metrics))
            layout["right_col"].update(_device_status_panel(metrics, score, self.start_time))
        else:
            loading = Panel("[dim]Collecting first metrics…[/dim]", box=box.ROUNDED)
            layout["memory"].update(loading)
            layout["cpu"].update(loading)
            layout["right_col"].update(
                Panel(
                    "[dim]Initializing…[/dim]",
                    title="[bold]Device Status[/bold]",
                    box=box.ROUNDED,
                )
            )

        if has_advice:
            layout["advice"].update(_advice_panel(decisions) or Panel(""))

        layout["events"].update(_events_panel(events))

        return layout
