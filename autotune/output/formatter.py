"""Rich-based terminal output formatter."""

from __future__ import annotations

import time
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from autotune.config.generator import Recommendation, ScoredConfig
from autotune.hardware.profiler import HardwareProfile, ProcessInfo
from autotune.hardware.ram_advisor import UnlockGroup
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
    lines.append("")

    # Install + use commands — the most important thing for a new user
    if c.model.ollama_tag:
        lines.append(
            f"[bold green]→ Install:[/bold green]  "
            f"[bold cyan]ollama pull {c.model.ollama_tag}[/bold cyan]"
        )
        lines.append(
            f"[bold green]→ Chat:   [/bold green]  "
            f"[bold cyan]autotune chat --model {c.model.id}[/bold cyan]"
        )
    else:
        lines.append(
            f"[bold green]→ Chat:   [/bold green]  "
            f"[bold cyan]autotune chat --model {c.model.id}[/bold cyan]"
        )

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

    # ── Friendly next-steps footer ─────────────────────────────────────────
    console.rule("[dim]Next Steps[/dim]")
    console.print()
    console.print(
        "  [bold]1.[/bold]  Copy the [bold cyan]ollama pull ...[/bold cyan] command above and run it in your terminal."
    )
    console.print(
        "  [bold]2.[/bold]  Once downloaded, start chatting:  "
        "[bold cyan]autotune chat --model <model-id>[/bold cyan]"
    )
    console.print(
        "  [bold]3.[/bold]  Verify autotune is actually helping:  "
        "[bold cyan]autotune proof --model <model-id>[/bold cyan]"
    )
    console.print()
    console.print(
        "  [dim]Not sure which model to pick?  "
        "Start with [bold]balanced[/bold] mode's primary recommendation — "
        "it's the best all-round choice for most computers.[/dim]"
    )
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


# ---------------------------------------------------------------------------
# Running models (autotune ps)
# ---------------------------------------------------------------------------

def print_running_models(models: list) -> None:  # list[RunningModel]
    """Render a table of every LLM currently resident in memory."""
    import psutil

    if not models:
        console.print("[dim]No models are currently loaded in memory.[/dim]")
        console.print(
            "[dim]Start a chat with [bold]autotune chat --model <id>[/bold] "
            "or load a model in Ollama/LM Studio.[/dim]"
        )
        return

    _BACKEND_STYLE = {
        "ollama":   "[cyan]Ollama[/cyan]",
        "mlx":      "[green]MLX[/green]",
        "lmstudio": "[yellow]LM Studio[/yellow]",
    }

    t = Table(
        title=f"Running Models  [dim]({len(models)} loaded)[/dim]",
        box=box.ROUNDED,
        show_header=True,
        min_width=72,
    )
    t.add_column("Model",      style="bold",       no_wrap=True)
    t.add_column("Backend",    justify="center")
    t.add_column("RAM",        justify="right")
    t.add_column("Context",    justify="right")
    t.add_column("Quant",      justify="center",   style="dim")
    t.add_column("Loaded",     justify="right",    style="dim")
    t.add_column("Expires in", justify="right")

    total_ram = 0.0
    for m in models:
        if m.ram_gb:
            ram_str = f"[red]{m.ram_gb:.2f} GB[/red]" if m.ram_gb >= 1.0 else f"{m.ram_gb:.2f} GB"
        else:
            ram_str = "[dim]—[/dim]"
        ctx_str = f"{m.context_len:,}" if m.context_len else "[dim]—[/dim]"
        backend_label = _BACKEND_STYLE.get(m.backend, m.backend)
        quant_str = m.quant if m.quant else "[dim]—[/dim]"
        age_str = m.age_str if m.loaded_since else "[dim]—[/dim]"

        if m.expires_at:
            remaining = m.expires_at - time.time()
            if remaining < 60:
                exp_str = f"[yellow]{m.expires_str}[/yellow]"
            else:
                exp_str = f"[dim]{m.expires_str}[/dim]"
        else:
            exp_str = "[dim]pinned[/dim]"

        t.add_row(m.name, backend_label, ram_str, ctx_str, quant_str, age_str, exp_str)
        total_ram += m.ram_gb or 0.0

    console.print(t)

    # RAM summary bar
    vm = psutil.virtual_memory()
    total_gb = vm.total / 1024 ** 3
    used_gb = vm.used / 1024 ** 3
    avail_gb = vm.available / 1024 ** 3
    model_pct = (total_ram / total_gb * 100) if total_gb else 0

    console.print()
    if total_ram:
        console.print(
            f"  Models hold  [bold]{total_ram:.2f} GB[/bold]"
            f"  of {total_gb:.0f} GB total  "
            f"[dim]({model_pct:.0f}% of RAM)[/dim]"
        )
    console.print(
        f"  System RAM:  {used_gb:.1f} GB used  /  "
        f"[bold green]{avail_gb:.1f} GB free[/bold green]"
    )


# ---------------------------------------------------------------------------
# RAM pressure report
# ---------------------------------------------------------------------------

def print_ram_pressure_report(
    hogs: list[ProcessInfo],
    groups: list[UnlockGroup],
    available_gb: float,
) -> None:
    """Print a RAM-hog table and model-unlock suggestions."""

    # ── Top RAM consumers ──────────────────────────────────────────────────
    if not hogs:
        console.print("[dim]No significant RAM consumers detected.[/dim]")
        return

    hog_table = Table(
        title="Top RAM Consumers",
        box=box.ROUNDED,
        show_header=True,
        min_width=60,
    )
    hog_table.add_column("#", justify="right", style="dim", width=4)
    hog_table.add_column("Process", style="bold")
    hog_table.add_column("PID", justify="right", style="dim")
    hog_table.add_column("RAM Used", justify="right")
    hog_table.add_column("Type", justify="center")

    _KIND_LABEL = {
        "user_app":    "[yellow]app[/yellow]",
        "llm_backend": "[cyan]LLM backend[/cyan]",
        "ide":         "[dim]IDE[/dim]",
        "system":      "[dim]system[/dim]",
    }

    for i, proc in enumerate(hogs, 1):
        ram_str = f"[red]{proc.rss_gb:.2f} GB[/red]" if proc.rss_gb > 1.0 else f"{proc.rss_gb:.2f} GB"
        kind_label = _KIND_LABEL.get(proc.kind, "[dim]?[/dim]")
        hog_table.add_row(str(i), proc.name, str(proc.pid), ram_str, kind_label)

    console.print(hog_table)
    console.print()

    # ── Unlock suggestions ─────────────────────────────────────────────────
    closeable = [p for p in hogs if p.is_closeable]
    if not groups:
        if not closeable:
            console.print(
                "[dim]No closeable user apps detected with significant RAM usage — "
                "nothing to suggest.[/dim]"
            )
        else:
            total_freeable = sum(p.rss_gb for p in closeable)
            console.print(
                f"[dim]Closeable apps hold ~{total_freeable:.1f} GB combined, "
                f"but that's not enough to unlock any additional models "
                f"(or all models already fit at {available_gb:.1f} GB).[/dim]"
            )
        return

    console.print(
        "[bold]RAM Unlock Suggestions[/bold]  "
        "[dim]— close these apps to run a bigger model[/dim]\n"
    )

    for grp in groups:
        # Processes to close — shown once per group
        close_lines: list[str] = []
        for proc in grp.processes:
            close_lines.append(
                f"  [yellow]✕[/yellow] [bold]{proc.name}[/bold] "
                f"[dim](PID {proc.pid})[/dim]  frees ~{proc.rss_gb:.2f} GB"
            )
        close_block = "Close:\n" + "\n".join(close_lines)

        # Models unlocked — listed once beneath the closure action
        model_lines: list[str] = []
        for s in grp.models:
            mmlu = f"  MMLU {s.model.bench_mmlu:.0%}" if s.model.bench_mmlu else ""
            model_lines.append(
                f"  [green]→[/green] [bold]{s.model.name}[/bold]  "
                f"[dim]{s.model.parameters_b:.1f}B · {s.quant}[/dim]{mmlu}  "
                f"[dim](needs {s.required_gb:.1f} GB)[/dim]"
            )
        unlock_block = "Unlocks:\n" + "\n".join(model_lines)

        header = (
            f"Free ~{grp.freed_gb:.1f} GB  "
            f"[dim]({available_gb:.1f} GB → "
            f"[green]{grp.available_after_gb:.1f} GB[/green])[/dim]"
        )
        body = header + "\n\n" + close_block + "\n\n" + unlock_block

        console.print(Panel(body, border_style="green", expand=False))
        console.print()
