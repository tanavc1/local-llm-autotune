"""Rich-based terminal output formatter."""

from __future__ import annotations

from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from autotune.config.generator import Recommendation, ScoredConfig
from autotune.hardware.profiler import HardwareProfile
from autotune.models.registry import QUANTIZATIONS, ModelProfile, list_models

console = Console()

# Mode display labels and colours
MODE_META: dict[str, tuple[str, str]] = {
    "fastest": ("⚡ Fastest", "bold yellow"),
    "balanced": ("⚖  Balanced", "bold cyan"),
    "best_quality": ("✨ Best Quality", "bold magenta"),
}


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------

def print_hardware_profile(hw: HardwareProfile) -> None:
    table = Table(
        title="Hardware Profile",
        box=box.ROUNDED,
        show_header=False,
        min_width=60,
    )
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("OS", hw.os_version or hw.os_name)
    table.add_row("CPU", hw.cpu.brand)
    table.add_row(
        "CPU Cores",
        f"{hw.cpu.physical_cores} physical / {hw.cpu.logical_cores} logical",
    )
    if hw.cpu.freq_max_mhz:
        table.add_row("CPU Freq (max)", f"{hw.cpu.freq_max_mhz / 1000:.2f} GHz")
    table.add_row("Total RAM", f"{hw.memory.total_gb:.1f} GB")
    table.add_row("Available RAM", f"{hw.memory.available_gb:.1f} GB")

    if hw.gpu:
        table.add_row("GPU", hw.gpu.name)
        table.add_row("GPU Backend", hw.gpu.backend.upper())
        if hw.gpu.is_unified_memory:
            table.add_row("VRAM", "Unified (shares system RAM)")
        elif hw.gpu.vram_gb:
            table.add_row("VRAM", f"{hw.gpu.vram_gb:.1f} GB")
        if hw.gpu.driver_version:
            table.add_row("Driver / OS", hw.gpu.driver_version)
    else:
        table.add_row("GPU", "Not detected (CPU-only mode)")

    table.add_row(
        "Effective Memory Budget",
        f"[bold green]{hw.effective_memory_gb:.1f} GB[/bold green]",
    )
    table.add_row("Inference Mode", hw.inference_mode.upper())

    console.print(table)


# ---------------------------------------------------------------------------
# Individual scored config block
# ---------------------------------------------------------------------------

def _config_panel(sc: ScoredConfig, rank: int, mode: str) -> Panel:
    c = sc.candidate
    mem = sc.memory
    q_spec = QUANTIZATIONS[c.quant]

    # Score bars (ASCII progress)
    def bar(val: float, width: int = 12) -> str:
        filled = round(val * width)
        return "█" * filled + "░" * (width - filled)

    lines: list[str] = []
    lines.append(
        f"[bold]{c.model.name}[/bold]  [dim]{c.model.parameters_b:.1f}B params[/dim]"
    )
    lines.append(
        f"Quantization : [yellow]{c.quant}[/yellow]  ({q_spec.description})"
    )
    lines.append(f"Context      : [cyan]{c.context_len:,}[/cyan] tokens")
    gpu_str = (
        f"[green]{c.n_gpu_layers} / {c.model.n_layers} layers on GPU[/green]"
        if c.n_gpu_layers > 0
        else "[dim]CPU only[/dim]"
    )
    lines.append(f"GPU layers   : {gpu_str}")
    lines.append("")
    lines.append(
        f"Memory: weights {mem.weights_gb:.2f} GB  "
        f"+ KV cache {mem.kv_cache_gb:.2f} GB  "
        f"+ overhead {mem.overhead_gb:.2f} GB"
    )
    lines.append(
        f"         = [bold]{mem.peak_gb:.2f} GB[/bold] peak  "
        f"/ [bold green]{mem.headroom_gb:.2f} GB[/bold green] headroom"
    )
    lines.append("")
    lines.append(f"Stability  {bar(sc.stability_score)}  {sc.stability_score:.2f}")
    lines.append(f"Speed      {bar(sc.speed_score)}  {sc.speed_score:.2f}")
    lines.append(f"Quality    {bar(sc.quality_score)}  {sc.quality_score:.2f}")
    lines.append(f"[dim]Composite  {bar(sc.composite)}  {sc.composite:.3f}[/dim]")
    lines.append("")
    lines.append(f"[dim italic]{sc.rationale}[/dim italic]")

    label, colour = MODE_META.get(mode, (mode, "white"))
    title = (
        f"[{colour}]{label}[/{colour}]  —  "
        f"{'Primary' if rank == 0 else f'Alternative #{rank}'}"
    )
    return Panel("\n".join(lines), title=title, box=box.ROUNDED, padding=(0, 2))


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def print_recommendations(
    recs: dict[str, Recommendation],
    modes: Optional[list[str]] = None,
) -> None:
    if modes is None:
        modes = list(recs.keys())

    for mode in modes:
        if mode not in recs:
            console.print(f"[red]No recommendations generated for mode: {mode}[/red]")
            continue

        rec = recs[mode]
        console.rule(f"[bold]{MODE_META.get(mode, (mode,''))[0]} Mode[/bold]")
        console.print(_config_panel(rec.primary, rank=0, mode=mode))

        if rec.alternatives:
            console.print("[dim]Alternatives:[/dim]")
            for i, alt in enumerate(rec.alternatives, start=1):
                console.print(_config_panel(alt, rank=i, mode=mode))

        console.print()


# ---------------------------------------------------------------------------
# Model registry table
# ---------------------------------------------------------------------------

def print_model_table() -> None:
    table = Table(
        title="Model Registry",
        box=box.SIMPLE_HEAD,
        show_lines=False,
    )
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Params", justify="right")
    table.add_column("Layers", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Quantizations")
    table.add_column("Description")

    for m in list_models():
        table.add_row(
            m.id,
            m.name,
            f"{m.parameters_b:.1f}B",
            str(m.n_layers),
            f"{m.context_window // 1024}k",
            ", ".join(m.quantization_options),
            m.description,
        )

    console.print(table)
