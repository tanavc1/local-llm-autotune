"""
Rich live terminal dashboard for the session controller.
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

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _pct_color(pct: float, warn: float = 75.0, crit: float = 88.0) -> str:
    if pct >= crit:
        return "bold red"
    if pct >= warn:
        return "yellow"
    return "green"


def _bar(pct: float, width: int = 18, warn: float = 75.0, crit: float = 88.0) -> Text:
    filled = round(max(0.0, min(1.0, pct / 100)) * width)
    empty = width - filled
    color = _pct_color(pct, warn, crit)
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * empty, style="dim")
    return t


def _state_badge(state: SessionState) -> Text:
    badges = {
        SessionState.OPTIMAL:           ("● OPTIMAL",           "bold green"),
        SessionState.WARNING:           ("▲ WARNING",           "bold yellow"),
        SessionState.ACTION_NEEDED:     ("⚡ ACTION NEEDED",    "bold red"),
        SessionState.DEGRADING:         ("⬇ DEGRADING",         "bold yellow"),
        SessionState.STABLE_RECOVERING: ("↑ RECOVERING",        "cyan"),
        SessionState.CRITICAL:          ("⛔ CRITICAL",          "bold red blink"),
    }
    label, style = badges.get(state, ("? UNKNOWN", "white"))
    return Text(label, style=style)


def _thermal_badge(ts: ThermalState, limit: int) -> Text:
    if ts == ThermalState.NOMINAL:
        return Text(f"● NOMINAL ({limit}%)", style="green")
    if ts == ThermalState.WARM:
        return Text(f"~ WARM ({limit}%)", style="yellow")
    if ts in (ThermalState.WARNING, ThermalState.THROTTLING):
        return Text(f"⚠ THROTTLING ({limit}%)", style="bold red")
    return Text(f"⛔ CRITICAL ({limit}%)", style="bold red blink")


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _temp_str(t: Optional[float]) -> str:
    return f"{t:.0f}°C" if t is not None else "N/A"


def _gb(v: Optional[float]) -> str:
    return f"{v:.2f} GB" if v is not None else "N/A"


def _rate_str(rate: float) -> str:
    if abs(rate) < 0.5:
        return "[dim]stable[/dim]"
    sign = "+" if rate > 0 else ""
    color = "red" if rate > 20 else "yellow" if rate > 5 else "green"
    return f"[{color}]{sign}{rate:.0f} MB/min[/{color}]"


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _header_panel(hw: HardwareProfile, state: SessionState, start_time: float) -> Panel:
    gpu_label = f"{hw.gpu.name} ({hw.gpu.backend.upper()})" if hw.gpu else "CPU-only"
    if hw.gpu and hw.gpu.is_unified_memory:
        gpu_label += " [unified]"

    t = Text()
    t.append("  autotune live  ", style="bold white on blue")
    t.append(f"  {hw.os_version}  │  {hw.cpu.brand[:30]}  │  {hw.memory.total_gb:.0f} GB  │  {gpu_label}  │  ", style="dim")
    t.append(f"⏱ {_elapsed(start_time)}", style="cyan")
    t.append("  │  ")
    t.append_text(_state_badge(state))

    return Panel(t, box=box.HORIZONTALS, padding=(0, 1))


def _memory_panel(m: LiveMetrics, is_unified: bool) -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(width=8, style="bold")
    t.add_column(width=18)
    t.add_column(width=8, justify="right")
    t.add_column()

    # RAM
    t.add_row(
        "RAM",
        _bar(m.ram_percent),
        f"[{_pct_color(m.ram_percent)}]{m.ram_percent:.0f}%[/]",
        f"[dim]{m.ram_used_gb:.1f}/{m.ram_total_gb:.0f} GB  {_rate_str(m.ram_growth_mb_per_min)}[/dim]",
    )

    # Unified / VRAM
    if m.vram_percent is not None and not is_unified:
        t.add_row(
            "VRAM",
            _bar(m.vram_percent),
            f"[{_pct_color(m.vram_percent)}]{m.vram_percent:.0f}%[/]",
            f"[dim]{_gb(m.vram_used_gb)} / {_gb(m.vram_total_gb)}  {_rate_str(m.vram_growth_mb_per_min)}[/dim]",
        )
    elif is_unified and m.vram_percent is not None:
        t.add_row(
            "GPU=RAM",
            _bar(m.vram_percent),
            f"[{_pct_color(m.vram_percent)}]{m.vram_percent:.0f}%[/]",
            f"[dim]unified pool[/dim]",
        )

    # Swap
    swap_warn = 5.0 if m.swap_percent > 0 else 100.0
    t.add_row(
        "Swap",
        _bar(m.swap_percent, warn=swap_warn, crit=15.0),
        f"[{_pct_color(m.swap_percent, warn=swap_warn, crit=15.0)}]{m.swap_percent:.0f}%[/]",
        f"[dim]{m.swap_used_gb:.2f}/{m.swap_total_gb:.1f} GB  {_rate_str(m.swap_growth_mb_per_min)}[/dim]",
    )

    # Compressor hint for macOS
    avail_color = _pct_color(m.ram_percent)
    t.add_row(
        "",
        "",
        "",
        f"[{avail_color}]{m.ram_available_gb:.1f} GB free[/{avail_color}]",
    )

    return Panel(t, title="[bold]Memory[/bold]", box=box.ROUNDED, padding=(0, 1))


def _performance_panel(m: LiveMetrics) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(width=14, style="bold dim")
    t.add_column()

    def maybe(v, fmt=".1f", suffix="") -> str:
        return f"{v:{fmt}}{suffix}" if v is not None else "[dim]—[/dim]"

    tps = m.tokens_per_sec
    tps_color = "green" if tps and tps > 10 else "yellow" if tps else "dim"
    t.add_row("Prompt tok/s", f"[{tps_color}]{maybe(tps)} t/s[/{tps_color}]")

    gen = m.gen_tokens_per_sec
    gen_color = "green" if gen and gen > 10 else "yellow" if gen else "dim"
    t.add_row("Gen tok/s", f"[{gen_color}]{maybe(gen)} t/s[/{gen_color}]")

    ttft = m.ttft_ms
    ttft_color = "green" if ttft and ttft < 500 else "yellow" if ttft and ttft < 2000 else "red" if ttft else "dim"
    t.add_row("TTFT", f"[{ttft_color}]{maybe(ttft, '.0f', ' ms')}[/{ttft_color}]")

    t.add_row("Queue depth", str(m.queue_depth) if m.queue_depth else "[dim]0[/dim]")

    cpu_color = _pct_color(m.cpu_percent, warn=70, crit=90)
    t.add_row("CPU overall", f"[{cpu_color}]{m.cpu_percent:.0f}%[/{cpu_color}]")

    # Per-core mini-bar
    cores = m.cpu_per_core
    if cores:
        core_str = " ".join(
            f"[{'red' if c > 85 else 'yellow' if c > 60 else 'green'}]{'█' if c > 50 else '▄' if c > 25 else '░'}[/]"
            for c in cores[:16]
        )
        t.add_row("CPU cores", core_str)

    return Panel(t, title="[bold]Performance[/bold]", box=box.ROUNDED, padding=(0, 1))


def _thermal_panel(m: LiveMetrics) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(width=12, style="bold dim")
    t.add_column()

    t.add_row("CPU temp", _temp_str(m.cpu_temp_c))
    t.add_row("GPU temp", _temp_str(m.gpu_temp_c))
    t.add_row("Thermal", _thermal_badge(m.thermal_state, m.cpu_speed_limit_pct))

    if m.llm_processes:
        t.add_row("", "")
        t.add_row("[bold]LLM procs[/bold]", "")
        for p in m.llm_processes[:4]:
            t.add_row(
                f"  {p.runtime[:10]}",
                f"[cyan]{p.ram_gb:.1f} GB[/cyan] RAM  [dim]{p.cpu_percent:.0f}% CPU[/dim]"
                f"  [dim]pid {p.pid}[/dim]",
            )
    elif m.ollama_models:
        t.add_row("", "")
        t.add_row("[bold]Ollama[/bold]", "")
        for om in m.ollama_models[:3]:
            t.add_row(f"  {om.name[:20]}", f"[cyan]{om.size_gb:.1f} GB[/cyan]  ctx {om.context_len:,}")
    else:
        t.add_row("", "")
        t.add_row("[dim]No LLM[/dim]", "[dim]No running LLM detected[/dim]")

    return Panel(t, title="[bold]Thermal / Processes[/bold]", box=box.ROUNDED, padding=(0, 1))


def _config_panel(cfg: SessionConfig) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(width=14, style="bold dim")
    t.add_column()

    t.add_row("Model", f"[cyan]{cfg.model_name}[/cyan]")
    t.add_row("Quantization", f"[yellow]{cfg.quant}[/yellow]")
    t.add_row("Context", f"{cfg.context_len:,} tokens")
    t.add_row("GPU layers", f"{cfg.n_gpu_layers}/{cfg.n_total_layers}")
    t.add_row("Backend", cfg.backend.upper())
    t.add_row("KV precision", cfg.kv_cache_precision.upper())
    t.add_row("Concurrency", str(cfg.concurrency))
    t.add_row("Speculative", "on" if cfg.speculative_decoding else "[dim]off[/dim]")
    t.add_row("Prompt cache", "on" if cfg.prompt_caching else "[dim]off[/dim]")
    t.add_row("", "")
    weight_color = _pct_color(cfg.weight_gb / cfg.total_budget_gb * 100, warn=60, crit=80)
    t.add_row("Weights", f"[{weight_color}]{cfg.weight_gb:.2f} GB[/{weight_color}]")
    kv_color = _pct_color(cfg.kv_cache_gb / cfg.total_budget_gb * 100, warn=15, crit=25)
    t.add_row("KV cache", f"[{kv_color}]{cfg.kv_cache_gb:.2f} GB[/{kv_color}]")
    t.add_row("Budget", f"{cfg.total_budget_gb:.1f} GB available")

    return Panel(t, title="[bold]Current Config[/bold]", box=box.ROUNDED, padding=(0, 1))


def _events_panel(events: list[SessionEvent], decisions: list[AdvisorDecision]) -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(width=11, style="dim")
    t.add_column(width=8)
    t.add_column()

    _level_styles = {
        "INFO":     ("dim", "INFO "),
        "OK":       ("green", "OK   "),
        "WARN":     ("yellow", "WARN "),
        "WARNING":  ("yellow", "WARN "),
        "ACTION":   ("bold red", "ACT  "),
        "ACTION_NEEDED": ("bold red", "ACT  "),
        "CRITICAL": ("bold red blink", "CRIT "),
        "DEGRADING": ("yellow", "DEG  "),
        "OPTIMAL":  ("green", "OK   "),
        "STABLE_RECOVERING": ("cyan", "RECOV"),
    }

    shown = 0
    for ev in events[:12]:
        ts_str = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
        style, tag = _level_styles.get(ev.level, ("white", ev.level[:5]))
        t.add_row(ts_str, f"[{style}]{tag}[/{style}]", ev.message)
        shown += 1

    if shown == 0:
        t.add_row("", "[dim]—[/dim]", "[dim]Session started. Monitoring…[/dim]")

    return Panel(t, title="[bold]Event Log[/bold]", box=box.ROUNDED, padding=(0, 1))


def _advice_panel(decisions: list[AdvisorDecision]) -> Optional[Panel]:
    if not decisions:
        return None
    recent = decisions[:3]
    t = Table.grid(padding=(0, 1))
    t.add_column(width=10, style="bold")
    t.add_column()

    for d in recent:
        severity_color = {
            SessionState.CRITICAL: "bold red",
            SessionState.ACTION_NEEDED: "red",
            SessionState.WARNING: "yellow",
        }.get(d.severity, "cyan")
        action_label = d.action.replace("_", " ").upper()
        t.add_row(f"[{severity_color}]{action_label}[/{severity_color}]", d.reason)
        # Show the suggested command
        changes = d.suggested_changes
        if changes:
            cmd_parts = []
            if "context_len" in changes:
                cmd_parts.append(f"--ctx-size {changes['context_len']}")
            if "kv_cache_precision" in changes:
                cmd_parts.append(f"--kv-cache-type {changes['kv_cache_precision']}")
            if "quant" in changes:
                cmd_parts.append(f"→ reload with {changes['quant']}")
            if "concurrency" in changes:
                cmd_parts.append(f"--parallel {changes['concurrency']}")
            if cmd_parts:
                t.add_row("", f"[dim]↳ {' '.join(cmd_parts)}[/dim]")

    return Panel(t, title="[bold yellow]⚡ Recommendations[/bold yellow]", box=box.ROUNDED, padding=(0, 1))


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
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="advice", size=6) if decisions else Layout(name="advice", size=0),
            Layout(name="events", size=14),
        )

        layout["header"].update(_header_panel(self.hw, state, self.start_time))

        # Body: left metrics, right config
        layout["body"].split_row(
            Layout(name="metrics", ratio=3),
            Layout(name="config_col", ratio=2),
        )

        # Metrics column: memory | performance | thermal
        layout["metrics"].split_column(
            Layout(name="memory", ratio=3),
            Layout(name="perf_therm", ratio=2),
        )
        layout["perf_therm"].split_row(
            Layout(name="perf"),
            Layout(name="thermal"),
        )

        if metrics:
            layout["memory"].update(_memory_panel(metrics, self._is_unified))
            layout["perf"].update(_performance_panel(metrics))
            layout["thermal"].update(_thermal_panel(metrics))
        else:
            loading = Panel("[dim]Collecting metrics…[/dim]", box=box.ROUNDED)
            layout["memory"].update(loading)
            layout["perf"].update(loading)
            layout["thermal"].update(loading)

        layout["config_col"].update(_config_panel(self.config))

        if decisions:
            layout["advice"].update(_advice_panel(decisions) or Panel(""))

        layout["events"].update(_events_panel(events, decisions))

        return layout
