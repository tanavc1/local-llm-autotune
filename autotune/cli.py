"""
autotune CLI – entry point.

Commands
--------
  autotune recommend            Run full hardware profiling + generate recommendations
  autotune recommend --mode fastest|balanced|best_quality
  autotune hardware             Show detected hardware profile only
  autotune models               List the model registry
  autotune fetch <model_id>     Fetch model specs from HuggingFace and store locally
  autotune fetch-many           Bulk-fetch a curated list of popular OSS models
  autotune db                   Show database stats
  autotune db-models            List all models cached in the local DB
  autotune log-run              Manually log a real inference observation
  autotune mlx list             List locally cached MLX models (Apple Silicon)
  autotune mlx pull <model>     Download MLX-quantized model from mlx-community
  autotune mlx resolve <model>  Show which MLX model ID would be used
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import click
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="llm-autotune")
def cli() -> None:
    """Local-LLM autotune – recommends the best inference config for your hardware."""


# ---------------------------------------------------------------------------
# `autotune recommend`
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fastest", "balanced", "best_quality", "all"]),
    default="all",
    show_default=True,
    help="Optimisation target.",
)
@click.option(
    "--model",
    "model_filter",
    default=None,
    metavar="MODEL_ID",
    help="Restrict search to a single model (use `autotune models` for IDs).",
)
@click.option(
    "--show-hardware/--no-show-hardware",
    default=True,
    show_default=True,
    help="Print the hardware profile before recommendations.",
)
@click.option(
    "--top",
    default=3,
    show_default=True,
    metavar="N",
    help="Number of alternatives to show per mode.",
)
def recommend(
    mode: str,
    model_filter: Optional[str],
    show_hardware: bool,
    top: int,
) -> None:
    """Profile this machine and recommend the best LLM inference configuration."""
    from autotune.telemetry import maybe_prompt_consent
    maybe_prompt_consent()
    from autotune.hardware.profiler import profile_hardware
    from autotune.config.generator import generate_recommendations, MODE_WEIGHTS
    from autotune.output.formatter import print_hardware_profile, print_recommendations
    from autotune.models.registry import MODEL_REGISTRY

    # ── Hardware profiling ──────────────────────────────────────────────
    console.rule("[bold blue]autotune recommend[/bold blue]")
    console.print()
    console.print("[bold]Step 1 of 2[/bold]  Detecting your hardware…")
    console.print("  [dim]Scanning CPU model and core count, available RAM, GPU backend…[/dim]")
    with console.status(
        "  [dim]Reading /proc/cpuinfo, sysctl, Metal/CUDA device info…[/dim]",
        spinner="dots",
    ):
        hw = profile_hardware()

    gpu_str = (
        f"{hw.gpu.name} ({hw.gpu.backend.upper()})"
        if hw.gpu else "CPU-only (no GPU detected)"
    )
    mem_avail = hw.memory.available_gb
    console.print(
        f"  [green]✓[/green]  {hw.cpu.brand[:42]}  /  "
        f"{hw.memory.total_gb:.0f} GB RAM ({mem_avail:.1f} GB free)  /  {gpu_str}"
    )
    console.print()

    if show_hardware:
        print_hardware_profile(hw)
        console.print()

    # ── Optional model filter ───────────────────────────────────────────
    if model_filter:
        if model_filter not in MODEL_REGISTRY:
            console.print(
                f"[red]Unknown model ID: {model_filter!r}. "
                f"Run `autotune models` to see valid IDs.[/red]"
            )
            sys.exit(1)
        # Temporarily restrict the registry inside the generator by patching
        # global – we do a minimal monkeypatch here to keep generator pure.
        import autotune.models.registry as _reg
        _orig = dict(_reg.MODEL_REGISTRY)
        _reg.MODEL_REGISTRY = {model_filter: _reg.MODEL_REGISTRY[model_filter]}  # type: ignore[assignment]

    # ── Generate recommendations ────────────────────────────────────────
    modes = list(MODE_WEIGHTS.keys()) if mode == "all" else [mode]

    # Count candidates we'll evaluate
    from autotune.config.generator import CONTEXT_LENGTHS, GPU_LAYER_FRACTIONS
    n_models = len(MODEL_REGISTRY) if not model_filter else 1
    n_quants_avg = 5  # rough average across models
    n_ctx = len(CONTEXT_LENGTHS)
    n_gpu = len(GPU_LAYER_FRACTIONS) if hw.has_gpu else 1
    n_candidates = n_models * n_quants_avg * n_ctx * n_gpu * len(modes)

    console.print(
        f"[bold]Step 2 of 2[/bold]  Scoring candidate configurations…\n"
        f"  [dim]Evaluating ~{n_candidates:,} combos "
        f"(model × quant × context × GPU layers × mode)  "
        f"against {mem_avail:.1f} GB available…[/dim]"
    )
    with console.status(
        "  [dim]Fitting, stability-scoring, speed-scoring, quality-scoring…[/dim]",
        spinner="dots",
    ):
        try:
            recs = generate_recommendations(hw, modes=modes, top_n=top)
        finally:
            if model_filter:
                import autotune.models.registry as _reg  # noqa: F811
                _reg.MODEL_REGISTRY = _orig  # type: ignore[assignment]

    if not recs:
        console.print(
            "[bold red]No configuration fits within the available memory budget.[/bold red]\n"
            "Try closing other applications to free RAM, or add more memory/VRAM."
        )
        sys.exit(1)

    console.print(
        f"  [green]✓[/green]  Done — found recommendations for "
        f"{len(recs)} mode(s)\n"
    )
    print_recommendations(recs, modes=modes)


# ---------------------------------------------------------------------------
# `autotune hardware`
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--ram-tips/--no-ram-tips",
    default=True,
    show_default=True,
    help="Show top RAM consumers and model-unlock suggestions.",
)
def hardware(ram_tips: bool) -> None:
    """Detect and display the hardware profile without generating recommendations.

    Also shows which apps are consuming the most RAM and which models you could
    run if you closed them (use --no-ram-tips to skip this section).
    """
    from autotune.hardware.profiler import profile_hardware, get_ram_hogs
    from autotune.hardware.ram_advisor import compute_unlock_suggestions
    from autotune.output.formatter import print_hardware_profile, print_ram_pressure_report

    console.print("[dim]Scanning CPU, RAM, GPU, OS version…[/dim]")
    with console.status("[cyan]Profiling hardware…[/cyan]", spinner="dots"):
        hw = profile_hardware()
        if ram_tips:
            hogs = get_ram_hogs(top_n=10)
            suggestions = compute_unlock_suggestions(hw.effective_memory_gb, hogs)

    print_hardware_profile(hw)

    if ram_tips:
        console.print()
        console.rule("[bold blue]RAM Pressure & Model Unlock Tips[/bold blue]")
        console.print()
        print_ram_pressure_report(hogs, suggestions, hw.effective_memory_gb)


# ---------------------------------------------------------------------------
# `autotune ps`
# ---------------------------------------------------------------------------

@cli.command()
def ps() -> None:
    """Show all LLMs currently loaded in memory across Ollama, MLX, and LM Studio."""
    from autotune.api.running_models import get_running_models
    from autotune.output.formatter import print_running_models

    with console.status("[cyan]Querying backends…[/cyan]", spinner="dots"):
        models = get_running_models()

    print_running_models(models)


# ---------------------------------------------------------------------------
# `autotune models`
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--registry", is_flag=True, default=False,
    help="Show the internal model registry instead of locally downloaded models.",
)
def models(registry: bool) -> None:
    """List all models available on this machine (Ollama, MLX, LM Studio).

    Shows size on disk, architecture, quantization, and quality tier based on
    public benchmarks (MMLU, HumanEval) so you can pick the right model.

    Use --registry to show autotune's internal pre-configured model list.
    """
    if registry:
        from autotune.output.formatter import print_model_table
        print_model_table()
        return

    from autotune.api.local_models import list_local_models, is_ollama_running
    from autotune.models.quality import tier_badge, tier_markup
    from rich.table import Table
    from rich import box as _box
    from rich.text import Text

    with console.status("[cyan]Scanning for local models…[/cyan]", spinner="dots"):
        local = list_local_models()

    if not local:
        if not is_ollama_running():
            console.print(
                "[yellow]No models found.[/yellow]\n"
                "Ollama does not appear to be running.  Start it first:\n"
                "  [bold]ollama serve[/bold]\n\n"
                "Or pull a model directly:\n"
                "  [bold]autotune pull qwen3:8b[/bold]"
            )
        else:
            console.print(
                "[yellow]No models found.[/yellow]\n"
                "Pull one to get started:\n"
                "  [bold]autotune pull[/bold]  (browse popular models)\n"
                "  [bold]autotune pull qwen3:8b[/bold]"
            )
        return

    # Group by source
    by_source: dict[str, list] = {}
    for m in local:
        by_source.setdefault(m.source, []).append(m)

    source_order = ["ollama", "mlx", "lmstudio"]
    source_labels = {
        "ollama":   "Ollama  (via Ollama runtime)",
        "mlx":      "MLX  (Apple Silicon native — fastest)",
        "lmstudio": "LM Studio",
    }

    for source in source_order:
        group = by_source.get(source, [])
        if not group:
            continue

        label = source_labels.get(source, source)
        t = Table(
            title=f"[bold]{label}[/bold]",
            box=_box.SIMPLE_HEAD,
            show_lines=False,
            title_justify="left",
            pad_edge=False,
            min_width=72,
        )
        t.add_column("Model",    style="cyan bold", no_wrap=True)
        t.add_column("Size",     justify="right",   no_wrap=True,  style="dim", min_width=6)
        t.add_column("Params",   justify="right",   no_wrap=True,  style="dim", min_width=5)
        t.add_column("Quant",    justify="center",  no_wrap=True,  style="dim", min_width=7)
        t.add_column("Ctx",      justify="right",   no_wrap=True,  style="dim", min_width=5)
        t.add_column("Tier",     justify="center",  no_wrap=True,  min_width=4)
        t.add_column("MMLU",     justify="right",   no_wrap=True,  min_width=5)
        t.add_column("Code",     justify="right",   no_wrap=True,  min_width=5)
        t.add_column("Note",     no_wrap=True,      style="dim")

        for m in sorted(group, key=lambda x: x.id):
            q = m.quality

            tier_cell = Text.from_markup(tier_badge(q.tier)) if q else Text("?", style="dim")
            mmlu_str  = f"{q.mmlu:.0f}%" if (q and q.mmlu)      else "—"
            code_str  = f"{q.humaneval:.0f}%" if (q and q.humaneval) else "—"
            note_str  = (q.note[:52] + "…") if (q and len(q.note) > 53) else (q.note if q else "")
            size_str  = f"{m.size_gb:.1f}G"  if m.size_gb         else "—"
            param_str = m.parameter_size      or "—"
            quant_str = m.quantization        or "—"
            ctx_str   = (
                f"{m.context_length // 1000}K" if (m.context_length and m.context_length >= 1000)
                else (str(m.context_length) if m.context_length else "—")
            )

            model_cell = m.id
            if m.mlx_available and source == "ollama":
                model_cell += "  [dim green]✦MLX[/dim green]"

            t.add_row(model_cell, size_str, param_str, quant_str, ctx_str,
                      tier_cell, mmlu_str, code_str, note_str)

        console.print(t)
        console.print()

    total = len(local)
    console.print(
        f"[dim]{total} model(s) on device  ·  "
        f"[bold]autotune pull[/bold] to add more  ·  "
        f"[bold]autotune chat --model <id>[/bold] to chat[/dim]"
    )
    console.print(
        "[dim]Tier S→D, MMLU = broad knowledge %, Code = HumanEval pass@1  "
        "(public benchmarks, ~4-bit quant unless noted)[/dim]\n"
    )


# ---------------------------------------------------------------------------
# `autotune pull`
# ---------------------------------------------------------------------------

@cli.command("pull")
@click.argument("model", required=False, default=None)
@click.option(
    "--list", "show_list",
    is_flag=True, default=False,
    help="Show popular models you can pull instead of downloading one.",
)
def pull(model: Optional[str], show_list: bool) -> None:
    """Download an Ollama model directly from within autotune.

    MODEL is an Ollama model tag, e.g. qwen3:8b, llama3.2, qwen2.5:14b.
    After downloading you can chat with it immediately:

    \b
      autotune pull llama3.2
      autotune chat --model llama3.2

    Run without arguments (or with --list) to browse popular models.
    """
    from autotune.api.ollama_pull import (
        OllamaNotRunningError, PullError,
        print_popular_models, pull_model,
    )

    if show_list or not model:
        print_popular_models(console)
        if not model:
            return

    console.print(f"[dim]Checking if Ollama is running…[/dim]")
    try:
        pull_model(model, console)
        console.print(
            f"[dim]Start chatting:  [bold]autotune chat --model {model}[/bold]\n"
            f"           or list models: [bold]autotune ls[/bold][/dim]"
        )
    except OllamaNotRunningError as e:
        console.print(f"[red]Ollama not running:[/red] {e}")
        raise SystemExit(1)
    except PullError as e:
        console.print(f"[red]Pull failed:[/red] {e}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")


# ---------------------------------------------------------------------------
# `autotune delete`
# ---------------------------------------------------------------------------

@cli.command("delete")
@click.argument("model", required=False, default=None)
@click.option(
    "--yes", "-y",
    is_flag=True, default=False,
    help="Skip confirmation prompt.",
)
def delete(model: Optional[str], yes: bool) -> None:
    """Delete a locally cached Ollama model.

    MODEL is an Ollama model tag, e.g. qwen3:8b or gemma4:e4b.
    Run without arguments to pick from a list of downloaded models.

    \b
      autotune delete qwen3:8b
      autotune delete               # interactive picker
    """
    from autotune.api.local_models import list_local_models, is_ollama_running
    from autotune.api.ollama_pull import OllamaNotRunningError, PullError, delete_model

    # If no model given, show interactive picker
    if not model:
        if not is_ollama_running():
            console.print("[red]Ollama is not running.[/red] Start it with: [bold]ollama serve[/bold]")
            raise SystemExit(1)
        local = [m for m in list_local_models() if m.source == "ollama"]
        if not local:
            console.print("[yellow]No Ollama models found.[/yellow]")
            raise SystemExit(0)

        from rich.table import Table
        from rich import box as _box
        tbl = Table(box=_box.SIMPLE, show_header=True, header_style="bold")
        tbl.add_column("#", style="dim", width=4)
        tbl.add_column("Model")
        tbl.add_column("Size", justify="right")
        tbl.add_column("Family")
        for i, m in enumerate(local, 1):
            size_str = f"{m.size_gb:.1f} GB" if m.size_gb else "—"
            tbl.add_row(str(i), m.id, size_str, m.family or "—")
        console.print(tbl)

        try:
            choice = input("Enter number or model name to delete (blank to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Cancelled.[/yellow]")
            raise SystemExit(0)

        if not choice:
            console.print("[yellow]Cancelled.[/yellow]")
            raise SystemExit(0)

        if choice.isdigit():
            idx = int(choice) - 1
            if idx < 0 or idx >= len(local):
                console.print(f"[red]Invalid selection:[/red] {choice}")
                raise SystemExit(1)
            model = local[idx].id
        else:
            model = choice

    # Confirm
    if not yes:
        try:
            ans = input(f"Delete [bold]{model}[/bold]? This cannot be undone. [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Cancelled.[/yellow]")
            raise SystemExit(0)
        if ans not in ("y", "yes"):
            console.print("[yellow]Cancelled.[/yellow]")
            raise SystemExit(0)

    try:
        delete_model(model, console)
        console.print(
            f"[dim]List remaining models: [bold]autotune ls[/bold][/dim]"
        )
    except OllamaNotRunningError as e:
        console.print(f"[red]Ollama not running:[/red] {e}")
        raise SystemExit(1)
    except PullError as e:
        console.print(f"[red]Delete failed:[/red] {e}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")


# ---------------------------------------------------------------------------
# `autotune benchmark`
# ---------------------------------------------------------------------------

@cli.command("benchmark")
@click.argument("model")
@click.option(
    "--runs", "-n",
    type=int, default=2,
    help="Runs per prompt per condition (default: 2; use 1 for a quick check).",
    show_default=True,
)
@click.option(
    "--profile", "-p",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced",
    help="autotune profile to compare against raw Ollama defaults.",
    show_default=True,
)
@click.option(
    "--output", "-o",
    default=None,
    metavar="FILE.json",
    help="Save full results to a JSON file.",
)
@click.option(
    "--no-save",
    is_flag=True, default=False,
    help="Do not persist results to the autotune DB.",
)
def benchmark(model: str, runs: int, profile: str, output: Optional[str], no_save: bool) -> None:
    """Honest 1-vs-1: raw Ollama defaults vs autotune optimizer.

    Runs a curated 6-prompt suite (factual Q&A, code, math, long output,
    multi-turn conversation, code review) through both conditions and reports
    tok/s, TTFT, peak RAM, CPU load, and swap — with color-coded deltas.

    If the optimizer makes something WORSE it shows that too.

    \b
    Examples:
      autotune benchmark qwen3:8b            # 2 runs × 6 prompts × 2 conditions
      autotune benchmark llama3.2:3b -n 1   # quick single-pass
      autotune benchmark qwen2.5:7b -p fast --output results.json
    """
    import asyncio

    from autotune.bench.compare import run_comparison, print_report, export_json

    # Estimate duration so user isn't surprised
    est_min = runs * 6 * 2 * 1.5 / 60   # rough: 1.5 min per inference call
    console.print(
        f"\n[bold]autotune benchmark[/bold]  ·  [cyan]{model}[/cyan]  ·  "
        f"profile=[bold]{profile}[/bold]  ·  {runs} run(s) per condition\n"
        f"[dim]Estimated time: {est_min:.0f}–{est_min*2:.0f} min  "
        f"(depends on model size and hardware)[/dim]\n"
    )

    try:
        report = asyncio.run(
            run_comparison(model, runs, profile, console, save_db=not no_save)
        )
        print_report(report, console)
        if output:
            export_json(report, output, console)
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark cancelled.[/yellow]")
    except Exception as e:
        console.print(f"[red]Benchmark failed:[/red] {e}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# `autotune session`
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "model_id", default=None, metavar="HF_MODEL_ID",
              help="Specific model to optimize for (HF repo ID, must be in DB).")
@click.option("--mode", "-m",
              type=click.Choice(["fastest", "balanced", "best_quality"]),
              default="balanced", show_default=True)
@click.option("--interval", default=1.0, show_default=True, metavar="SEC",
              help="Metrics polling interval in seconds.")
@click.option("--json", "json_only", is_flag=True, default=False,
              help="Print initial recommendation JSON and exit (no live UI).")
def session(
    model_id: Optional[str],
    mode: str,
    interval: float,
    json_only: bool,
) -> None:
    """
    Start a live session controller.

    Continuously monitors hardware, detects running LLMs, and recommends
    adaptive configuration changes to keep inference smooth and stable.

    Controls: context, KV precision, quantization, concurrency, caching.
    Logs all observations to the local database.
    """
    from autotune.session.controller import SessionController

    ctrl = SessionController(
        model_id=model_id,
        mode=mode,
        interval=interval,
        json_only=json_only,
    )
    ctrl.run()


# ---------------------------------------------------------------------------
# `autotune fetch`
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("model_id")
@click.option("--force", is_flag=True, default=False, help="Re-fetch even if already cached.")
def fetch(model_id: str, force: bool) -> None:
    """Fetch a model's specs from HuggingFace and store in the local DB.

    MODEL_ID is the HuggingFace repo ID, e.g. meta-llama/Meta-Llama-3.1-8B
    """
    from autotune.hub.fetcher import fetch_model
    from autotune.db.store import get_db

    db = get_db()

    if not force:
        existing = db.get_model(model_id)
        if existing:
            console.print(f"[yellow]Already cached:[/yellow] {model_id}  (use --force to refresh)")
            _print_model_summary(existing)
            return

    with console.status(f"[cyan]Fetching {model_id} from HuggingFace…[/cyan]", spinner="dots"):
        spec = fetch_model(model_id)

    db.upsert_model(spec.to_db_dict())
    console.print(f"[green]✓ Stored:[/green] {model_id}")
    _print_model_summary(spec.to_db_dict())


def _print_model_summary(d: dict) -> None:
    from rich.table import Table
    from rich import box

    t = Table(box=box.ROUNDED, show_header=False, min_width=64)
    t.add_column("Field", style="bold")
    t.add_column("Value")

    def row(label: str, val) -> None:
        if val is not None and val != "" and val != []:
            t.add_row(label, str(val))

    row("Name", d.get("name"))
    row("Org", d.get("organization"))
    row("Family", d.get("family"))
    row("License", d.get("license"))
    row("Params (total)", f"{d.get('total_params_b')} B")
    row("Params (active)", f"{d.get('active_params_b')} B" if d.get("is_moe") else None)
    row("MoE", f"{d.get('num_experts')} experts, {d.get('experts_per_token')} active" if d.get("is_moe") else None)
    row("Architecture", d.get("arch_type"))
    row("Layers", d.get("n_layers"))
    row("Hidden size", d.get("hidden_size"))
    row("Attention heads", f"{d.get('n_heads')} total / {d.get('n_kv_heads')} KV ({d.get('attention_type','').upper()})")
    row("Head dim", d.get("head_dim"))
    row("FFN size", d.get("intermediate_size"))
    row("Vocab size", d.get("vocab_size"))
    row("Context window", f"{d.get('max_context_window'):,} tokens" if d.get("max_context_window") else None)
    row("RoPE theta", d.get("rope_theta"))
    row("Activation", d.get("activation"))
    row("Normalization", d.get("normalization"))
    row("Sliding window", d.get("sliding_window_size"))
    row("KV latent dim (MLA)", d.get("kv_latent_dim"))
    row("Attn logit cap", d.get("attn_logit_softcapping"))

    mem_parts = []
    for label, key in [("F16", "mem_f16_gb"), ("Q8", "mem_q8_0_gb"),
                       ("Q6K", "mem_q6_k_gb"), ("Q5KM", "mem_q5_k_m_gb"),
                       ("Q4KM", "mem_q4_k_m_gb"), ("Q3KM", "mem_q3_k_m_gb"),
                       ("Q2K", "mem_q2_k_gb")]:
        if d.get(key):
            mem_parts.append(f"{label}:{d[key]:.1f}GB")
    if mem_parts:
        t.add_row("Weight mem", "  ".join(mem_parts))

    row("Recommended quant", d.get("recommended_quant"))
    row("HuggingFace", d.get("hf_url"))
    row("GGUF download", d.get("gguf_url"))

    console.print(t)


# ---------------------------------------------------------------------------
# `autotune fetch-many`
# ---------------------------------------------------------------------------

# Curated list of important OSS models with real HF repo IDs
CURATED_MODELS: list[str] = [
    # ── Tiny / on-device ──────────────────────────────────────────────
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "google/gemma-3-1b",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/phi-4-mini",
    # ── 3–4 B ─────────────────────────────────────────────────────────
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3-4B",
    "google/gemma-2-2b",
    "google/gemma-3-4b",
    "ibm-granite/granite-3.1-2b-instruct",
    # ── 7–9 B ─────────────────────────────────────────────────────────
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Meta-Llama-3.1-8B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "google/gemma-2-9b",
    "google/gemma-3-12b",
    "01-ai/Yi-1.5-6B",
    "01-ai/Yi-1.5-9B",
    "internlm/internlm2_5-7b",
    "allenai/OLMo-2-1124-7B",
    "ibm-granite/granite-3.1-8b-instruct",
    # ── 12–20 B ───────────────────────────────────────────────────────
    "mistralai/Mistral-Nemo-Base-2407",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-Coder-14B",
    "Qwen/Qwen3-14B",
    "microsoft/phi-4",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "internlm/internlm2_5-20b",
    "allenai/OLMo-2-1124-13B",
    # ── 24–40 B ───────────────────────────────────────────────────────
    "mistralai/Mistral-Small-3.1-24B-Base-2503",
    "google/gemma-2-27b",
    "google/gemma-3-27b",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-Coder-32B",
    "Qwen/Qwen3-32B",
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "CohereForAI/c4ai-command-r-v01",
    # ── 70 B+ ─────────────────────────────────────────────────────────
    "meta-llama/Meta-Llama-3.1-70B",
    "Qwen/Qwen2.5-72B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "01-ai/Yi-1.5-34B",
    # ── MoE ───────────────────────────────────────────────────────────
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "Qwen/Qwen3-30B-A3B",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
]


@cli.command("fetch-many")
@click.option("--force", is_flag=True, default=False, help="Re-fetch even if already cached.")
@click.option(
    "--filter", "name_filter", default=None, metavar="SUBSTR",
    help="Only fetch models whose ID contains this substring."
)
def fetch_many(force: bool, name_filter: Optional[str]) -> None:
    """Bulk-fetch all curated OSS models from HuggingFace into the local DB."""
    from autotune.hub.fetcher import fetch_model
    from autotune.db.store import get_db

    db = get_db()
    targets = CURATED_MODELS
    if name_filter:
        targets = [m for m in targets if name_filter.lower() in m.lower()]

    console.print(f"[bold]Fetching {len(targets)} models…[/bold] (Ctrl-C to stop)\n")

    ok = skip = fail = 0
    for model_id in targets:
        if not force and db.get_model(model_id):
            console.print(f"  [dim]skip[/dim]  {model_id}")
            skip += 1
            continue
        try:
            with console.status(f"  [cyan]fetch[/cyan] {model_id}", spinner="dots"):
                spec = fetch_model(model_id)
            db.upsert_model(spec.to_db_dict())
            params_str = f"{spec.total_params_b}B" if spec.total_params_b else "?"
            layers_str = str(spec.n_layers) if spec.n_layers else "?"
            ctx_str = f"{spec.max_context_window//1024}k" if spec.max_context_window else "?"
            console.print(
                f"  [green]✓[/green]    {model_id:<55} "
                f"[dim]{params_str:>7}  {layers_str:>3}L  {ctx_str:>5} ctx[/dim]"
            )
            ok += 1
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            break
        except Exception as e:
            console.print(f"  [red]✗[/red]    {model_id}  ({e})")
            fail += 1

    console.print(
        f"\n[bold]Done.[/bold]  fetched={ok}  skipped={skip}  failed={fail}  "
        f"total in DB={db.model_count()}"
    )


# ---------------------------------------------------------------------------
# `autotune db`
# ---------------------------------------------------------------------------

@cli.command("db")
def db_stats() -> None:
    """Show local database statistics."""
    from autotune.db.store import get_db
    from rich.table import Table
    from rich import box

    db = get_db()
    s = db.stats()

    t = Table(box=box.ROUNDED, show_header=False)
    t.add_column("", style="bold")
    t.add_column("")
    t.add_row("DB path", s["db_path"])
    t.add_row("DB size", f"{s['db_size_mb']} MB")
    t.add_row("Models cached", str(s["models"]))
    t.add_row("Hardware profiles", str(s["hardware_profiles"]))
    t.add_row("Run observations", str(s["run_observations"]))
    console.print(t)


# ---------------------------------------------------------------------------
# `autotune db-models`
# ---------------------------------------------------------------------------

@cli.command("db-models")
@click.option("--family", default=None, help="Filter by model family.")
@click.option("--max-params", default=None, type=float, metavar="B", help="Max active params (billions).")
def db_models(family: Optional[str], max_params: Optional[float]) -> None:
    """List all models cached in the local database."""
    from autotune.db.store import get_db
    from rich.table import Table
    from rich import box

    db = get_db()
    rows = db.list_models(family=family, max_params_b=max_params)

    if not rows:
        console.print("[yellow]No models in DB. Run `autotune fetch-many` to populate.[/yellow]")
        return

    t = Table(box=box.SIMPLE_HEAD, show_lines=False)
    t.add_column("ID", style="cyan", no_wrap=True)
    t.add_column("Params", justify="right")
    t.add_column("Layers", justify="right")
    t.add_column("Hidden", justify="right")
    t.add_column("KV heads", justify="right")
    t.add_column("Head dim", justify="right")
    t.add_column("Context", justify="right")
    t.add_column("Attn", justify="center")
    t.add_column("Q4KM GB", justify="right")
    t.add_column("Rec quant")

    for m in rows:
        params = f"{m.get('total_params_b') or '?'}B"
        if m.get("is_moe") and m.get("active_params_b") != m.get("total_params_b"):
            params += f" ({m['active_params_b']}B act)"
        t.add_row(
            m["id"],
            params,
            str(m.get("n_layers") or "?"),
            str(m.get("hidden_size") or "?"),
            str(m.get("n_kv_heads") or "?"),
            str(m.get("head_dim") or "?"),
            f"{m['max_context_window']//1024}k" if m.get("max_context_window") else "?",
            (m.get("attention_type") or "?").upper(),
            f"{m['mem_q4_k_m_gb']:.1f}" if m.get("mem_q4_k_m_gb") else "?",
            m.get("recommended_quant") or "?",
        )

    console.print(t)
    console.print(f"[dim]{len(rows)} models[/dim]")


# ---------------------------------------------------------------------------
# `autotune log-run`
# ---------------------------------------------------------------------------

@cli.command("log-run")
@click.option("--model", "model_id", required=True, help="HF model ID (must be in DB).")
@click.option("--quant", required=True, help="Quantization used e.g. Q4_K_M.")
@click.option("--context", "context_len", required=True, type=int)
@click.option("--gpu-layers", "n_gpu_layers", required=True, type=int)
@click.option("--tps", "tokens_per_sec", type=float, default=None, help="Prompt eval tok/s.")
@click.option("--gen-tps", "gen_tokens_per_sec", type=float, default=None, help="Generation tok/s.")
@click.option("--peak-ram", "peak_ram_gb", type=float, default=None)
@click.option("--peak-vram", "peak_vram_gb", type=float, default=None)
@click.option("--load-time", "load_time_sec", type=float, default=None)
@click.option("--ttft", "ttft_ms", type=float, default=None, help="Time to first token (ms).")
@click.option("--oom", is_flag=True, default=False)
@click.option("--notes", default="")
def log_run(
    model_id: str,
    quant: str,
    context_len: int,
    n_gpu_layers: int,
    tokens_per_sec: Optional[float],
    gen_tokens_per_sec: Optional[float],
    peak_ram_gb: Optional[float],
    peak_vram_gb: Optional[float],
    load_time_sec: Optional[float],
    ttft_ms: Optional[float],
    oom: bool,
    notes: str,
) -> None:
    """Log a real inference observation to the database."""
    from autotune.hardware.profiler import profile_hardware
    from autotune.db.store import get_db
    from autotune.db.fingerprint import hardware_id, hardware_to_db_dict

    db = get_db()

    if not db.get_model(model_id):
        console.print(f"[red]Model {model_id!r} not in DB. Run `autotune fetch {model_id}` first.[/red]")
        raise SystemExit(1)

    with console.status("Profiling hardware…", spinner="dots"):
        hw = profile_hardware()

    hw_dict = hardware_to_db_dict(hw)
    db.upsert_hardware(hw_dict)

    row_id = db.log_run({
        "model_id": model_id,
        "hardware_id": hw_dict["id"],
        "quant": quant,
        "context_len": context_len,
        "n_gpu_layers": n_gpu_layers,
        "tokens_per_sec": tokens_per_sec,
        "gen_tokens_per_sec": gen_tokens_per_sec,
        "peak_ram_gb": peak_ram_gb,
        "peak_vram_gb": peak_vram_gb,
        "load_time_sec": load_time_sec,
        "ttft_ms": ttft_ms,
        "completed": 0 if oom else 1,
        "oom": int(oom),
        "notes": notes,
    })

    console.print(f"[green]✓ Logged run #{row_id}[/green]  {model_id} @ {quant}  {context_len}ctx  "
                  f"hardware={hw_dict['id']}")


# ---------------------------------------------------------------------------
# `autotune bench`
# ---------------------------------------------------------------------------

INTENSIVE_PROMPT = """\
You are a senior systems engineer and performance optimization expert.

I need a COMPLETE multi-part answer — do not truncate or summarize. Work through every part.

**Scenario**: A Python web service receives 1,000 concurrent requests per second. \
The service parses JSON, queries a PostgreSQL database with ORM calls, processes \
results with nested loops, caches nothing, and re-instantiates DB connections per \
request. It runs on a 16 GB unified-memory Apple Silicon machine also running \
an LLM inference server.

**PART 1 — Bottleneck Analysis**
Identify and rank the top 5 performance bottlenecks in order of severity. \
For each: explain the root cause, the memory access pattern it creates, \
and how it interacts with Apple Silicon's unified memory architecture.

**PART 2 — Optimized Python Rewrite**
Write fully working Python code that fixes bottlenecks #1 and #2. \
Include: async I/O, connection pooling, and an LRU cache layer. \
Add inline comments explaining each optimization choice.

**PART 3 — Complexity Analysis**
For each function you wrote: state its time complexity O(n), space complexity O(n), \
and explain why those bounds hold.

**PART 4 — Memory Pressure on Apple Silicon**
Explain how running this service alongside an LLM (e.g., a 14B parameter Q4_K_M \
quantized model) affects both workloads on a 16 GB unified-memory system. \
What happens at 80% RAM utilization? At 94%? How does the memory compressor behave?

**PART 5 — Concrete Recommendations**
Give 3 specific, actionable configuration changes (with exact values) to run \
both the web service and LLM inference on this machine without OOM-killing either.

Be thorough and precise. This is a real production system."""


@cli.command("bench")
@click.option("--model", "-m", default="qwen3:8b", show_default=True,
              help="Model to benchmark.")
@click.option(
    "--profile", "-p",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced", show_default=True,
    help="autotune optimization profile to use.",
)
@click.option("--tag", default=None, metavar="NAME",
              help="Label this run (auto-generated if omitted).")
@click.option("--prompt-file", default=None, type=click.Path(exists=True),
              help="Use a custom prompt from a file instead of the built-in intensive prompt.")
@click.option("--no-hw-tuning", is_flag=True, default=False,
              help="Skip OS-level hardware optimizations (for comparison runs).")
@click.option("--raw", is_flag=True, default=False,
              help="Run ONLY raw Ollama defaults — no autotune. "
                   "Good for establishing a baseline before comparing.")
@click.option("--duel", is_flag=True, default=False,
              help="Run BOTH raw Ollama AND autotune back-to-back on the same prompt, "
                   "then show an immediate side-by-side comparison. "
                   "This is the recommended way to see the full before/after picture.")
@click.option("--save/--no-save", default=True,
              help="Save result to DB (default: save).")
@click.option("--compare", default=None, metavar="TAG_A,TAG_B",
              help="Compare two previously saved bench tags and show delta table.")
def bench(
    model: str,
    profile: str,
    tag: Optional[str],
    prompt_file: Optional[str],
    no_hw_tuning: bool,
    raw: bool,
    duel: bool,
    save: bool,
    compare: Optional[str],
) -> None:
    """Run an intensive benchmark and measure hardware strain.

    There are three ways to use this command:

    \b
    1. AUTOTUNE ONLY (default) — run autotune with the chosen profile:
         autotune bench --model qwen3:8b --profile balanced

    \b
    2. RAW OLLAMA ONLY — run pure Ollama defaults, no autotune:
         autotune bench --model qwen3:8b --raw

    \b
    3. DUEL (recommended) — run both raw and autotune, show comparison:
         autotune bench --model qwen3:8b --duel
         autotune bench --model qwen3:8b --duel --profile fast

    \b
    4. COMPARE SAVED RUNS — diff two previously saved tags:
         autotune bench --compare baseline,fast_optimized

    \b
    Save a run to a named tag and compare later:
      autotune bench --model qwen3:8b --raw --tag my_baseline
      autotune bench --model qwen3:8b --profile fast --tag my_fast
      autotune bench --compare my_baseline,my_fast
    """
    import asyncio
    import psutil as _psutil
    from autotune.bench.runner import run_bench, run_bench_ollama_only, run_raw_ollama, save_result

    # ── Compare mode ─────────────────────────────────────────────────────
    if compare:
        from autotune.db.store import get_db
        from rich.table import Table
        from rich import box

        parts = compare.split(",")
        if len(parts) != 2:
            console.print("[red]--compare requires exactly two tags: TAG_A,TAG_B[/red]")
            raise SystemExit(1)

        db = get_db()
        result = db.compare_runs(parts[0].strip(), parts[1].strip())

        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise SystemExit(1)

        console.print(f"\n[bold]Comparing[/bold]  [cyan]{result['tag_a']}[/cyan]  vs  [green]{result['tag_b']}[/green]")
        console.print(f"[dim]{result['runs_a']} run(s) vs {result['runs_b']} run(s)[/dim]\n")

        t = Table(box=box.ROUNDED)
        t.add_column("Metric", style="bold")
        t.add_column(result["tag_a"], justify="right", style="cyan")
        t.add_column(result["tag_b"], justify="right", style="green")
        t.add_column("Δ (%)", justify="right")

        metric_labels = {
            "tokens_per_sec": "Throughput (tok/s)  ↑ better",
            "ttft_ms":         "TTFT (ms)           ↓ better",
            "peak_ram_gb":     "Peak RAM (GB)       ↓ better",
            "peak_vram_gb":    "Peak VRAM/UMem (GB) ↓ better",
        }
        improvement_direction = {
            "tokens_per_sec": +1,   # higher = better
            "ttft_ms": -1,          # lower = better
            "peak_ram_gb": -1,
            "peak_vram_gb": -1,
        }
        for key, label in metric_labels.items():
            if key not in result["deltas"]:
                continue
            d = result["deltas"][key]
            pct = d["delta_pct"]
            direction = improvement_direction[key]
            improved = (pct * direction) > 0
            pct_str = f"[green]{pct:+.1f}%[/green]" if improved else f"[red]{pct:+.1f}%[/red]"
            t.add_row(label, str(d["a"]), str(d["b"]), pct_str)

        console.print(t)
        return

    # ── Build prompt ──────────────────────────────────────────────────────
    if prompt_file:
        with open(prompt_file) as f:
            prompt_text = f.read()
    else:
        prompt_text = INTENSIVE_PROMPT

    messages = [
        {"role": "system", "content": "You are a senior systems engineer and performance expert. Answer thoroughly and completely. Do not truncate."},
        {"role": "user",   "content": prompt_text},
    ]
    prompt_tokens_est = sum(len(m["content"]) // 4 for m in messages)

    ts = int(time.time())
    vm_before = _psutil.virtual_memory()
    sw_before = _psutil.swap_memory()

    # ── DUEL MODE — run both raw and autotune, then show comparison ────────
    if duel:
        from rich.table import Table
        from rich.panel import Panel
        from rich.rule import Rule
        from rich import box

        raw_tag   = tag + "_raw"   if tag else f"{model.replace(':', '_').replace('/', '_')}_duel_raw_{ts}"
        tuned_tag = tag + "_tuned" if tag else f"{model.replace(':', '_').replace('/', '_')}_duel_{profile}_{ts}"

        console.print()
        console.print(Panel(
            f"[bold]DUEL MODE[/bold]  [cyan]{model}[/cyan]  ·  autotune/{profile} vs. raw Ollama\n"
            f"[dim]Running the same intensive prompt through both configurations.\n"
            f"Prompt: ~{prompt_tokens_est} tokens  ·  "
            f"RAM before: {vm_before.used/1024**3:.2f} GB / {vm_before.total/1024**3:.1f} GB[/dim]",
            border_style="cyan",
        ))
        console.print()

        # ── Round 1: Raw Ollama ──────────────────────────────────────────
        console.print("[bold]Round 1 of 2[/bold]  [red]Raw Ollama[/red]  (zero autotune — factory defaults)")
        console.print("  [dim]num_ctx=4096, temp=0.8, keep_alive=5m, no HW tuning, no prefix cache[/dim]")
        with console.status("  [dim]Running raw Ollama inference… (measuring TTFT, tok/s, RAM, CPU…)[/dim]", spinner="dots"):
            raw_result = asyncio.run(run_raw_ollama(
                model_id=model,
                messages=messages,
                tag=raw_tag,
            ))
        if raw_result.error:
            console.print(f"  [red]✗  Raw Ollama failed:[/red] {raw_result.error}")
            raise SystemExit(1)
        console.print(
            f"  [green]✓[/green]  Done in {raw_result.elapsed_sec:.1f}s — "
            f"TTFT [yellow]{raw_result.ttft_ms:.0f} ms[/yellow]  "
            f"tok/s [yellow]{raw_result.tokens_per_sec:.1f}[/yellow]  "
            f"peak RAM [yellow]{raw_result.ram_peak_gb:.2f} GB[/yellow]  "
            f"CPU [yellow]{raw_result.cpu_avg_pct:.0f}%[/yellow]"
        )
        if save:
            save_result(raw_result)
            console.print(f"  [dim]Saved to DB as tag: {raw_tag}[/dim]")

        console.print()
        console.print("[dim]Cooling down 5 seconds to let RAM settle…[/dim]")
        import asyncio as _asyncio
        _asyncio.run(_asyncio.sleep(5))
        console.print()

        # ── Round 2: Autotune ────────────────────────────────────────────
        console.print(f"[bold]Round 2 of 2[/bold]  [cyan]autotune/{profile}[/cyan]  (full optimizer stack)")
        console.print(
            f"  [dim]Dynamic num_ctx, prefix caching (num_keep), keep_alive=-1, "
            f"repeat_penalty, QoS={profile.upper()}, GC suspend[/dim]"
        )
        with console.status(f"  [dim]Running autotune/{profile} inference…[/dim]", spinner="dots"):
            tuned_result = asyncio.run(run_bench_ollama_only(
                model_id=model,
                messages=messages,
                profile_name=profile,
                tag=tuned_tag,
                apply_hw_tuning=not no_hw_tuning,
            ))
        if tuned_result.error:
            console.print(f"  [red]✗  autotune/{profile} failed:[/red] {tuned_result.error}")
            raise SystemExit(1)
        console.print(
            f"  [green]✓[/green]  Done in {tuned_result.elapsed_sec:.1f}s — "
            f"TTFT [cyan]{tuned_result.ttft_ms:.0f} ms[/cyan]  "
            f"tok/s [cyan]{tuned_result.tokens_per_sec:.1f}[/cyan]  "
            f"peak RAM [cyan]{tuned_result.ram_peak_gb:.2f} GB[/cyan]  "
            f"CPU [cyan]{tuned_result.cpu_avg_pct:.0f}%[/cyan]"
        )
        if save:
            save_result(tuned_result)
            console.print(f"  [dim]Saved to DB as tag: {tuned_tag}[/dim]")

        # ── Duel comparison table ────────────────────────────────────────
        console.print()
        console.print(Rule("[bold]Duel Results[/bold]", style="dim"))
        console.print()

        def _pct_change(raw_val: float, tuned_val: float, higher_better: bool) -> str:
            if raw_val == 0:
                return "[dim]—[/dim]"
            pct = (tuned_val - raw_val) / abs(raw_val) * 100
            improved = (pct > 1 and higher_better) or (pct < -1 and not higher_better)
            degraded = (pct < -1 and higher_better) or (pct > 1 and not higher_better)
            sign = "+" if pct >= 0 else ""
            if improved:
                return f"[bold green]{sign}{pct:.1f}%[/bold green]"
            elif degraded:
                return f"[bold red]{sign}{pct:.1f}%[/bold red]"
            return f"[dim]{sign}{pct:.1f}%[/dim]"

        t = Table(box=box.ROUNDED, show_header=True, header_style="bold", expand=False)
        t.add_column("Metric",           style="bold", min_width=26)
        t.add_column("Raw Ollama",        justify="right", style="yellow")
        t.add_column(f"autotune/{profile}", justify="right", style="cyan")
        t.add_column("Change",            justify="right")
        t.add_column("Better?",           justify="center")

        metrics_cfg = [
            ("TTFT (ms)       ↓ lower=better",  raw_result.ttft_ms,       tuned_result.ttft_ms,       False, ".0f"),
            ("Throughput (tok/s)  ↑ higher=better", raw_result.tokens_per_sec, tuned_result.tokens_per_sec, True, ".1f"),
            ("Total time (s)",                  raw_result.elapsed_sec,   tuned_result.elapsed_sec,   False, ".2f"),
            ("Peak RAM (GB)   ↓ lower=better",  raw_result.ram_peak_gb,   tuned_result.ram_peak_gb,   False, ".3f"),
            ("RAM delta (GB)  ↓ lower=better",  raw_result.delta_ram_gb,  tuned_result.delta_ram_gb,  False, "+.3f"),
            ("Swap peak (GB)  ↓ lower=better",  raw_result.swap_peak_gb,  tuned_result.swap_peak_gb,  False, ".3f"),
            ("CPU avg (%)     ↓ lower=better",  raw_result.cpu_avg_pct,   tuned_result.cpu_avg_pct,   False, ".1f"),
            ("num_ctx used",                    float(raw_result.num_ctx_used or 4096),
                                                float(tuned_result.num_ctx_used or 0),               False, ".0f"),
        ]

        for label, rv, tv, hb, fmt in metrics_cfg:
            if rv == 0 and tv == 0:
                continue
            rv_str = f"{rv:{fmt}}"
            tv_str = f"{tv:{fmt}}" if tv != 0 else "—"
            delta  = _pct_change(rv, tv, hb)
            if rv == 0:
                better = "[dim]—[/dim]"
            else:
                pct = (tv - rv) / abs(rv) * 100
                improved = (pct > 1 and hb) or (pct < -1 and not hb)
                degraded = (pct < -1 and hb) or (pct > 1 and not hb)
                better = "[green]✓ better[/green]" if improved else "[red]✗ worse[/red]" if degraded else "[dim]≈ same[/dim]"
            t.add_row(label, rv_str, tv_str, delta, better)

        console.print(t)

        # KV settings used by autotune
        if tuned_result.num_ctx_used and tuned_result.num_keep_used is not None:
            console.print(
                f"\n[dim]autotune used: num_ctx={tuned_result.num_ctx_used:,}  "
                f"num_keep={tuned_result.num_keep_used}  "
                f"f16_kv={'yes' if tuned_result.f16_kv_used else 'no (Q8)'}[/dim]"
            )
        if save:
            console.print(
                f"\n[dim]Both runs saved. Compare later with:\n"
                f"  autotune bench --compare {raw_tag},{tuned_tag}[/dim]"
            )
        return

    # ── SINGLE MODE — run either raw or autotune ──────────────────────────
    mode_label = "raw_ollama" if raw else profile
    auto_tag = tag or f"{model.replace(':', '_').replace('/', '_')}_{mode_label}_{ts}"

    console.print()
    if raw:
        console.print(
            f"[bold]autotune bench[/bold]  [cyan]{model}[/cyan]  "
            f"profile=[red]RAW OLLAMA (no autotune)[/red]  tag=[dim]{auto_tag}[/dim]"
        )
        console.print(
            "[dim]Running with Ollama factory defaults: num_ctx=4096, temp=0.8, "
            "keep_alive=5m, no HW tuning, no prefix cache[/dim]"
        )
    else:
        console.print(
            f"[bold]autotune bench[/bold]  [cyan]{model}[/cyan]  "
            f"profile=[yellow]{profile}[/yellow]  tag=[dim]{auto_tag}[/dim]"
        )
        console.print(
            f"[dim]Running autotune/{profile}: dynamic num_ctx, prefix caching, keep_alive=-1, "
            f"repeat_penalty, QoS tuning  ·  Prompt: ~{prompt_tokens_est} tokens  "
            f"·  HW tuning: {'off (--no-hw-tuning)' if no_hw_tuning else 'on'}[/dim]"
        )
    console.print()

    console.print(
        f"[dim]RAM before: {vm_before.used/1024**3:.2f} GB / {vm_before.total/1024**3:.1f} GB  "
        f"·  Swap: {sw_before.used/1024**3:.2f} GB[/dim]\n"
    )
    console.print("[dim]Running inference (measuring TTFT, tok/s, RAM, CPU every 250ms)…[/dim]")

    with console.status("[bold cyan]Inference in progress…[/bold cyan]", spinner="dots"):
        if raw:
            result = asyncio.run(run_raw_ollama(
                model_id=model,
                messages=messages,
                tag=auto_tag,
            ))
        else:
            result = asyncio.run(run_bench_ollama_only(
                model_id=model,
                messages=messages,
                profile_name=profile,
                tag=auto_tag,
                apply_hw_tuning=not no_hw_tuning,
            ))

    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
        raise SystemExit(1)

    # ── Single-run results table ─────────────────────────────────────────
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console.print(f"[bold green]✓ Done[/bold green]  {result.elapsed_sec:.1f}s total\n")

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")

    t.add_row("Model", result.model_id)
    t.add_row("Profile", result.profile_name)
    t.add_row("num_ctx (KV window)", f"{result.num_ctx_used:,} tokens")
    t.add_row("Prompt tokens (est)", f"{result.prompt_tokens:,}")
    t.add_row("Completion tokens", f"{result.completion_tokens:,}")
    t.add_section()
    t.add_row("[yellow]TTFT[/yellow]",            f"[yellow]{result.ttft_ms:.0f} ms[/yellow]")
    t.add_row("[yellow]Throughput[/yellow]",       f"[yellow]{result.tokens_per_sec:.1f} tok/s[/yellow]")
    t.add_row("[yellow]Total time[/yellow]",       f"[yellow]{result.elapsed_sec:.2f} s[/yellow]")
    t.add_section()

    ram_color = "green" if result.ram_peak_gb < 12 else ("yellow" if result.ram_peak_gb < 14 else "red")
    swap_color = "green" if result.swap_peak_gb < 1 else ("yellow" if result.swap_peak_gb < 3 else "red")

    t.add_row("RAM before",  f"{result.ram_before_gb:.3f} GB")
    t.add_row("RAM peak",    f"[{ram_color}]{result.ram_peak_gb:.3f} GB[/{ram_color}]")
    t.add_row("RAM after",   f"{result.ram_after_gb:.3f} GB")
    t.add_row("RAM delta",   (f"[red]+{result.delta_ram_gb:.3f} GB[/red]"
                              if result.delta_ram_gb > 0.1 else
                              f"[green]{result.delta_ram_gb:+.3f} GB[/green]"))
    t.add_section()
    t.add_row("Swap before", f"{result.swap_before_gb:.3f} GB")
    t.add_row("Swap peak",   f"[{swap_color}]{result.swap_peak_gb:.3f} GB[/{swap_color}]")
    t.add_row("Swap after",  f"{result.swap_after_gb:.3f} GB")
    t.add_row("Swap delta",  (f"[red]+{result.delta_swap_gb:.3f} GB[/red]"
                              if result.delta_swap_gb > 0.05 else
                              f"[green]{result.delta_swap_gb:+.3f} GB[/green]"))
    t.add_section()
    t.add_row("CPU avg",     f"{result.cpu_avg_pct:.1f}%")
    t.add_row("CPU peak",    f"{result.cpu_peak_pct:.1f}%")

    console.print(t)

    # Show first 600 chars of response
    console.print()
    preview = result.response_text[:800].strip()
    if len(result.response_text) > 800:
        preview += f"\n[dim]... ({result.completion_tokens} tokens total)[/dim]"
    console.print(Panel(
        preview,
        title=f"[bold]Model response[/bold]  [dim](tag: {auto_tag})[/dim]",
        border_style="dim",
    ))

    if save:
        row_id = save_result(result)
        console.print(f"\n[dim]✓ Saved to DB as run #{row_id}  (tag: {auto_tag})[/dim]")
    else:
        console.print(f"\n[dim]Not saved (--no-save)[/dim]")


# ---------------------------------------------------------------------------
# `autotune ls`  — list Ollama models with hardware fitness scores
# ---------------------------------------------------------------------------

@cli.command("ls")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit JSON instead of a table.")
def ls(as_json: bool) -> None:
    """List locally downloaded Ollama models with hardware fitness scores.

    Shows which models fit in available memory, their safe context limit,
    recommended profile, and quantization warnings based on your hardware.
    KV cache is included in all memory estimates.
    """
    import json as _json
    import httpx
    from autotune.hardware.profiler import profile_hardware
    from autotune.api.model_selector import ModelSelector
    from rich.table import Table
    from rich import box

    # ── 1. Probe Ollama ─────────────────────────────────────────────────
    try:
        tags_resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        tags_resp.raise_for_status()
        ollama_models = tags_resp.json().get("models", [])
    except Exception:
        console.print(
            "[red]Ollama is not running.[/red]  Start it with: [cyan]ollama serve[/cyan]"
        )
        raise SystemExit(1)

    if not ollama_models:
        console.print(
            "[yellow]No models downloaded.[/yellow]  "
            "Pull one with: [cyan]ollama pull qwen3:8b[/cyan]"
        )
        return

    # ── 2. Hardware snapshot ─────────────────────────────────────────────
    with console.status("[cyan]Profiling hardware…[/cyan]", spinner="dots"):
        hw = profile_hardware()

    available_gb = hw.effective_memory_gb
    total_gb     = hw.memory.total_gb
    sel          = ModelSelector(available_gb=available_gb, total_ram_gb=total_gb)

    # ── 3. Enrich each model from /api/show ──────────────────────────────
    def _show(model_name: str) -> dict:
        try:
            r = httpx.post(
                "http://localhost:11434/api/show",
                json={"name": model_name},
                timeout=5.0,
            )
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    rows: list[dict] = []
    for m in ollama_models:
        name    = m.get("name", "")
        size_gb = m.get("size", 0) / 1024**3

        details      = _show(name)
        detail_block = details.get("details", {}) or {}
        modelinfo    = details.get("model_info") or details.get("modelinfo") or {}

        param_str   = detail_block.get("parameter_size", "")   # e.g. "3.8B"
        quant_level = detail_block.get("quantization_level", "unknown")

        params_b: Optional[float] = None
        if param_str:
            try:
                params_b = float(param_str.rstrip("Bb").strip())
                if "M" in param_str.upper():
                    params_b /= 1000
            except ValueError:
                pass

        # ── KV-aware fit analysis ────────────────────────────────────────
        report = sel.assess(
            model_name=name,
            size_gb=size_gb,
            params_b=params_b,
            quant=quant_level,
            modelinfo=modelinfo,
        )

        # Score: 10 = perfect fit; penalise swap/OOM heavily
        fc = report.fit_class
        from autotune.api.model_selector import FitClass
        if fc == FitClass.OOM:
            score = 0.0
        elif fc == FitClass.SWAP_RISK:
            score = max(1.0, 3.0 - (report.ram_util_pct - 92) * 0.3)
        elif fc == FitClass.MARGINAL:
            score = 5.0 + (92 - report.ram_util_pct) * 0.3
        else:   # SAFE
            util = report.ram_util_pct / 100
            if util < 0.15:
                score = 6.0 + util * 15
            elif util <= 0.70:
                score = 10.0 - abs(util - 0.50) * 5
            else:
                score = max(5.0, 10.0 - (util - 0.70) * 12)

        # Status label
        if fc == FitClass.OOM:
            status = "[red]⛔ OOM[/red]"
        elif fc == FitClass.SWAP_RISK:
            status = "[red]⚠ swap risk[/red]"
        elif fc == FitClass.MARGINAL:
            status = "[yellow]~ marginal[/yellow]"
        else:
            status = "[green]✓ fits[/green]"

        # Safe context string
        safe_ctx = report.safe_max_context
        if safe_ctx >= 32768:
            ctx_str = f"[green]{safe_ctx//1024}k[/green]"
        elif safe_ctx >= 8192:
            ctx_str = f"[yellow]{safe_ctx//1024}k[/yellow]"
        elif safe_ctx >= 1024:
            ctx_str = f"[red]{safe_ctx//1024}k[/red]"
        else:
            ctx_str = "[red]—[/red]"

        # Quant warning
        quant_note = ""
        if report.quant_too_heavy and report.suggested_quant:
            quant_note = f"→ try {report.suggested_quant}"

        rows.append({
            "name":        name,
            "size_gb":     round(size_gb, 2),
            "params":      param_str or "?",
            "quant":       quant_level,
            "total_gb":    report.total_est_gb,
            "util_pct":    report.ram_util_pct,
            "safe_ctx":    ctx_str,
            "status":      status,
            "rec_profile": report.recommended_profile,
            "rec_kv":      report.recommended_kv,
            "score":       round(min(10.0, score), 1),
            "quant_note":  quant_note,
            "warning":     report.warning or "",
            "fatal":       report.fatal,
            # raw for JSON
            "fit_class":   fc.value,
            "safe_ctx_tokens": safe_ctx,
            "arch_source": report.arch.source if report.arch else "none",
        })

    rows.sort(key=lambda r: -r["score"])

    if as_json:
        import re
        clean = []
        for r in rows:
            cr = dict(r)
            for k in ("status", "safe_ctx", "score"):
                cr[k] = re.sub(r"\[.*?\]", "", str(cr[k])).strip()
            clean.append(cr)
        console.print(_json.dumps(clean, indent=2))
        return

    # ── 4. Rich table ────────────────────────────────────────────────────
    console.print()
    console.print(
        f"[bold]Ollama models[/bold]  "
        f"[dim]available: {available_gb:.1f} GB / {total_gb:.0f} GB  "
        f"(safe limit: {available_gb * 0.85:.1f} GB)[/dim]"
        f"  [dim]{hw.cpu.brand.split('@')[0].strip()}[/dim]"
    )
    console.print()

    t = Table(box=box.SIMPLE_HEAD, show_lines=False)
    t.add_column("Model",       style="cyan",  no_wrap=True)
    t.add_column("Size",        justify="right")
    t.add_column("Params",      justify="right")
    t.add_column("Quant",       justify="center")
    t.add_column("Total+KV",    justify="right")
    t.add_column("RAM%",        justify="right")
    t.add_column("Safe ctx",    justify="center")
    t.add_column("Fits?",       justify="center")
    t.add_column("Profile",     justify="center", style="yellow")
    t.add_column("KV prec",     justify="center")
    t.add_column("Score",       justify="right")

    for r in rows:
        score_str = (
            f"[green]{r['score']}/10[/green]"   if r["score"] >= 8 else
            f"[yellow]{r['score']}/10[/yellow]" if r["score"] >= 5 else
            f"[red]{r['score']}/10[/red]"
        )
        util_str = f"{r['util_pct']:.0f}%"
        if r["util_pct"] > 92:
            util_str = f"[red]{util_str}[/red]"
        elif r["util_pct"] > 85:
            util_str = f"[yellow]{util_str}[/yellow]"

        quant_display = r["quant"]
        if r["quant_note"]:
            quant_display = f"{r['quant']} [dim]{r['quant_note']}[/dim]"

        t.add_row(
            r["name"],
            f"{r['size_gb']:.1f} GB",
            r["params"],
            quant_display,
            f"{r['total_gb']:.1f} GB",
            util_str,
            r["safe_ctx"],
            r["status"],
            r["rec_profile"],
            r["rec_kv"],
            score_str,
        )

    console.print(t)
    console.print(
        "[dim]Total+KV = weights + KV cache (8k ctx) + runtime overhead.  "
        "Safe ctx = max context before swap.  "
        "Run: [cyan]autotune run <model>[/cyan][/dim]\n"
    )

    # Print warnings for problematic models
    for r in rows:
        if r["warning"]:
            icon = "[red]✗[/red]" if r["fatal"] else "[yellow]⚠[/yellow]"
            console.print(f"  {icon} [bold]{r['name']}[/bold]: {r['warning']}")


# ---------------------------------------------------------------------------
# `autotune run`  — pick best profile and launch optimised chat
# ---------------------------------------------------------------------------

@cli.command("run")
@click.argument("model_name")
@click.option("--profile", "-p",
              type=click.Choice(["fast", "balanced", "quality", "auto"]),
              default="auto", show_default=True,
              help="Profile to use. 'auto' selects based on memory fit analysis.")
@click.option("--system", "-s", default=None, metavar="TEXT",
              help="System prompt to use for the session.")
@click.option("--force", is_flag=True, default=False,
              help="Start even if memory analysis predicts swap risk (not recommended).")
@click.option("--recall", is_flag=True, default=False,
              help="Inject relevant context from past conversations. Off by default.")
def run(model_name: str, profile: str, system: Optional[str], force: bool, recall: bool) -> None:
    """Pre-flight analysis + optimized chat for a locally downloaded Ollama model.

    Difference from `autotune chat`:
      run  = pre-flight (memory fit, swap risk, auto-profile) + chat
      chat = chat only (connects directly, optimization still active by default)

    Use `run` when you want autotune to analyze the model's memory requirements
    before loading it — it will warn you about swap risk and automatically pick
    the safest profile and context window.  Use `chat` for HuggingFace/MLX models
    or when you already know the profile you want.

    Both commands run the same real-time optimizer during inference (adaptive RAM
    monitor, KV manager, context optimizer).

    \b
    Examples:
      autotune run qwen3:8b
      autotune run qwen2.5-coder:14b --profile balanced
      autotune run llama3.2 --system "You are a concise coding assistant"
    """
    import httpx
    from autotune.hardware.profiler import profile_hardware
    from autotune.api.chat import start_chat
    from autotune.api.model_selector import ModelSelector, FitClass

    console.print(f"\n[bold]Pre-flight check for[/bold] [cyan]{model_name}[/cyan]\n")

    console.print("  [dim]Detecting hardware…[/dim]")
    hw           = profile_hardware()
    available_gb = hw.effective_memory_gb
    total_gb     = hw.memory.total_gb
    sel          = ModelSelector(available_gb=available_gb, total_ram_gb=total_gb)

    console.print(
        f"  [green]✓[/green]  {hw.cpu.brand[:40]}  /  "
        f"{total_gb:.0f} GB RAM  /  {available_gb:.1f} GB available"
    )

    # ── Fetch model info from Ollama ─────────────────────────────────────
    size_gb:   float          = 0.0
    params_b:  Optional[float] = None
    quant_str: str            = "unknown"
    modelinfo: dict           = {}

    console.print("  [dim]Querying Ollama for model info…[/dim]")
    try:
        tags_resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        for m in tags_resp.json().get("models", []):
            if m.get("name", "").lower() == model_name.lower():
                size_gb = m.get("size", 0) / 1024**3
                break
    except Exception:
        console.print("  [red]✗  Ollama is not running.[/red]  Start with: [bold]ollama serve[/bold]")
        raise SystemExit(1)

    try:
        show_resp = httpx.post(
            "http://localhost:11434/api/show",
            json={"name": model_name},
            timeout=5.0,
        )
        if show_resp.status_code == 200:
            show_data    = show_resp.json()
            detail_block = show_data.get("details", {}) or {}
            modelinfo    = show_data.get("model_info") or show_data.get("modelinfo") or {}
            param_str    = detail_block.get("parameter_size", "")
            quant_str    = detail_block.get("quantization_level", "unknown")
            if param_str:
                try:
                    params_b = float(param_str.rstrip("Bb").strip())
                    if "M" in param_str.upper():
                        params_b /= 1000
                except ValueError:
                    pass
    except Exception:
        pass

    if size_gb == 0.0:
        console.print(
            f"  [red]✗[/red]  Model [cyan]{model_name!r}[/cyan] not found in Ollama.\n"
            f"  Pull it first: [bold]ollama pull {model_name}[/bold]"
        )
        raise SystemExit(1)

    console.print(
        f"  [green]✓[/green]  Found [cyan]{model_name}[/cyan]  "
        f"({size_gb:.1f} GB on disk, quant={quant_str})"
    )

    # ── Pre-flight fit analysis ──────────────────────────────────────────
    console.print("  [dim]Running memory fit analysis (weights + KV cache + runtime overhead)…[/dim]")
    report = sel.assess(
        model_name=model_name,
        size_gb=size_gb,
        params_b=params_b,
        quant=quant_str,
        modelinfo=modelinfo,
    )

    arch_note = f"arch from {report.arch.source}" if report.arch else "arch: estimated"

    if report.fatal and not force:
        console.print(
            f"\n[bold red]✗ Cannot load model safely[/bold red]\n"
            f"  {report.warning}\n"
            f"  Use --force to override (will likely OOM or swap severely)."
        )
        raise SystemExit(1)

    if report.fit_class == FitClass.SWAP_RISK and not force:
        console.print(
            f"\n[bold yellow]⚠ Swap risk detected[/bold yellow]\n"
            f"  {report.warning}\n"
            f"  Use --force to proceed anyway."
        )
        raise SystemExit(1)

    if report.warning:
        console.print(f"[yellow]⚠[/yellow] {report.warning}")

    # ── Select profile ───────────────────────────────────────────────────
    if profile == "auto":
        chosen = report.recommended_profile
        if chosen == "—":
            chosen = "fast"
        console.print(
            f"[dim]Pre-flight:  {size_gb:.1f} GB weights  "
            f"+ {report.kv_q8_gb:.2f} GB KV (Q8, 8k ctx)  "
            f"+ {report.overhead_gb:.2f} GB overhead  "
            f"= {report.total_est_gb:.1f} GB / {available_gb:.1f} GB available "
            f"({report.ram_util_pct:.0f}%)  [{arch_note}][/dim]"
        )
        console.print(
            f"[dim]Auto-profile: [yellow]{chosen}[/yellow]  "
            f"safe context: {report.safe_max_context:,} tokens  "
            f"KV precision: {report.recommended_kv}[/dim]"
        )
    else:
        chosen = profile

    # ── Warn about quant downgrade opportunity ───────────────────────────
    if report.quant_too_heavy and report.suggested_quant:
        console.print(
            f"[dim]Tip: pull [cyan]{report.suggested_quant}[/cyan] "
            f"(~{report.suggested_quant_gb:.1f} GB) for "
            f"+{report.suggested_headroom_gb:.1f} GB headroom.[/dim]"
        )

    console.print()
    console.print(
        f"[bold green]✓ Pre-flight passed[/bold green]  —  "
        f"launching [cyan]autotune chat[/cyan] "
        f"(profile=[yellow]{chosen}[/yellow], optimize=on)\n"
        f"[dim]Equivalent command: autotune chat --model {model_name} --profile {chosen}"
        + (f" --system \"{system}\"" if system else "")
        + "[/dim]\n"
    )
    start_chat(model_id=model_name, profile=chosen, system_prompt=system, recall=recall)


# ---------------------------------------------------------------------------
# `autotune telemetry`  — show persisted telemetry history
# ---------------------------------------------------------------------------

@cli.command("telemetry")
@click.option("--model", "model_id", default=None,
              help="Filter to a specific model ID.")
@click.option("--limit", default=20, show_default=True,
              help="Number of recent runs to show.")
@click.option("--events", is_flag=True, default=False,
              help="Show individual telemetry events instead of run history.")
@click.option("--enable", is_flag=True, default=False,
              help="Opt in to anonymous telemetry collection.")
@click.option("--disable", is_flag=True, default=False,
              help="Opt out of all telemetry. No data will be sent.")
@click.option("--status", is_flag=True, default=False,
              help="Show current telemetry consent status.")
def telemetry(model_id: Optional[str], limit: int, events: bool,
              enable: bool, disable: bool, status: bool) -> None:
    """Show persisted performance telemetry for local LLM runs.

    Displays run history with structured metrics — TTFT, throughput, RAM/swap
    pressure, CPU load — all queryable for trend analysis.

    Consent management:
      autotune telemetry --enable    Opt in to anonymous telemetry
      autotune telemetry --disable   Opt out (stop all data collection)
      autotune telemetry --status    Show current consent status

    \b
    Examples:
      autotune telemetry
      autotune telemetry --model qwen3:8b
      autotune telemetry --events --model qwen3:8b
      autotune telemetry --disable
    """
    # ── Consent management ───────────────────────────────────────────────
    if enable and disable:
        console.print("[red]Cannot use --enable and --disable together.[/red]")
        raise SystemExit(1)

    if enable:
        from autotune.telemetry.consent import set_consent, is_opted_in
        from autotune.telemetry.events import EventType
        from autotune.telemetry import emit, register_install
        already = is_opted_in()
        set_consent(True)
        if not already:
            console.print("[green]✓ Telemetry enabled — thank you for helping improve autotune![/green]")
            console.print("[dim]Hardware fingerprint and anonymous performance data will be collected.[/dim]")
            import threading
            threading.Thread(target=register_install, daemon=True).start()
            emit(EventType.OPT_IN)
        else:
            console.print("[green]✓ Telemetry is already enabled.[/green]")
        return

    if disable:
        from autotune.telemetry.consent import set_consent, is_opted_in
        from autotune.telemetry.events import EventType
        from autotune.telemetry import emit
        was_opted_in = is_opted_in()
        if was_opted_in:
            emit(EventType.OPT_OUT)   # send one last event before disabling
        set_consent(False)
        console.print("[yellow]✓ Telemetry disabled — no further data will be sent.[/yellow]")
        console.print("[dim]Run `autotune telemetry --enable` to re-enable at any time.[/dim]")
        return

    if status:
        from autotune.telemetry.consent import is_opted_in, consent_answered, get_install_key
        answered = consent_answered()
        opted_in = is_opted_in()
        install_key = get_install_key() if answered else None
        console.print()
        if not answered:
            console.print("[yellow]Telemetry: not yet configured[/yellow]")
            console.print("[dim]Run `autotune serve` to see the opt-in prompt.[/dim]")
        elif opted_in:
            console.print("[green]Telemetry: enabled[/green]")
            console.print(f"[dim]Install key: {install_key}[/dim]")
            console.print("[dim]Run `autotune telemetry --disable` to opt out.[/dim]")
        else:
            console.print("[yellow]Telemetry: disabled[/yellow]")
            console.print("[dim]Run `autotune telemetry --enable` to opt in.[/dim]")
        console.print()
        return
    from autotune.db.store import get_db
    from rich.table import Table
    from rich import box
    import datetime

    db = get_db()

    if events:
        rows = db.get_telemetry(model_id=model_id, limit=limit)
        if not rows:
            console.print("[yellow]No telemetry events recorded yet.[/yellow]")
            return

        t = Table(box=box.SIMPLE_HEAD)
        t.add_column("Time",       style="dim", no_wrap=True)
        t.add_column("Model",      style="cyan")
        t.add_column("Event",      style="yellow")
        t.add_column("Value",      justify="right")
        t.add_column("Detail")

        for r in rows:
            ts = datetime.datetime.fromtimestamp(r["observed_at"]).strftime("%m-%d %H:%M")
            evt = r["event_type"]
            evt_styled = (
                f"[red]{evt}[/red]"   if evt in ("error", "oom_near", "swap_spike") else
                f"[yellow]{evt}[/yellow]" if evt in ("ram_spike", "slow_token", "pressure_high") else
                f"[green]{evt}[/green]"
            )
            val = f"{r['value_num']:.2f}" if r.get("value_num") is not None else "—"
            t.add_row(ts, r.get("model_id") or "—", evt_styled, val,
                      (r.get("value_text") or "")[:60])

        console.print(t)
        console.print(f"[dim]{len(rows)} event(s)[/dim]")

        # Summary by type
        summary = db.telemetry_summary(model_id=model_id)
        if summary:
            console.print()
            console.print("[bold]Event counts:[/bold]  " +
                          "  ".join(f"{k}={v}" for k, v in summary.items()))
        return

    # ── Run history ──────────────────────────────────────────────────────
    if model_id:
        rows = db.model_perf_history(model_id, limit=limit)
    else:
        rows = db.get_runs(limit=limit)

    if not rows:
        console.print("[yellow]No runs recorded yet. Run `autotune bench` to start.[/yellow]")
        return

    t = Table(box=box.SIMPLE_HEAD, show_lines=False)
    t.add_column("#",          justify="right", style="dim")
    t.add_column("Time",       style="dim", no_wrap=True)
    t.add_column("Model",      style="cyan")
    t.add_column("Profile",    justify="center", style="yellow")
    t.add_column("ctx",        justify="right")
    t.add_column("TTFT (ms)",  justify="right")
    t.add_column("tok/s",      justify="right")
    t.add_column("Peak RAM",   justify="right")
    t.add_column("Swap peak",  justify="right")
    t.add_column("CPU avg",    justify="right")
    t.add_column("OK?",        justify="center")

    for r in rows:
        ts = datetime.datetime.fromtimestamp(r["observed_at"]).strftime("%m-%d %H:%M")

        ttft = r.get("ttft_ms")
        ttft_str = f"{ttft:.0f}" if ttft else "—"
        if ttft and ttft > 3000:
            ttft_str = f"[red]{ttft_str}[/red]"
        elif ttft and ttft < 800:
            ttft_str = f"[green]{ttft_str}[/green]"

        tps = r.get("tokens_per_sec")
        tps_str = f"{tps:.1f}" if tps else "—"

        ram = r.get("peak_ram_gb")
        ram_str = f"{ram:.2f}" if ram else "—"
        if ram and ram > 13:
            ram_str = f"[red]{ram_str}[/red]"
        elif ram and ram > 10:
            ram_str = f"[yellow]{ram_str}[/yellow]"

        swap = r.get("swap_peak_gb")
        swap_str = f"{swap:.2f}" if swap else "—"
        if swap and swap > 3:
            swap_str = f"[red]{swap_str}[/red]"
        elif swap and swap > 1:
            swap_str = f"[yellow]{swap_str}[/yellow]"

        cpu = r.get("cpu_avg_pct")
        cpu_str = f"{cpu:.0f}%" if cpu else "—"

        ok = r.get("completed", 1)
        ok_str = "[green]✓[/green]" if ok else "[red]✗[/red]"

        notes = r.get("notes") or ""
        profile_name = r.get("profile_name") or ""
        if not profile_name:
            for part in notes.split():
                if part.startswith("profile="):
                    profile_name = part[8:]

        tag = r.get("bench_tag") or ""
        if not tag:
            for part in notes.split():
                if part.startswith("bench_tag="):
                    tag = part[10:]

        ctx = r.get("context_len", 0)

        t.add_row(
            str(r["id"]),
            ts,
            (r.get("model_id") or "")[:28],
            profile_name[:10] or "[dim]—[/dim]",
            str(ctx) if ctx else "—",
            ttft_str,
            tps_str,
            ram_str,
            swap_str,
            cpu_str,
            ok_str,
        )

    console.print()
    console.print(f"[bold]Telemetry history[/bold]"
                  + (f"  [dim]{model_id}[/dim]" if model_id else "")
                  + f"  [dim](last {len(rows)} runs)[/dim]")
    console.print()
    console.print(t)
    console.print(
        "[dim]TTFT: [green]green[/green]=fast (<800ms)  "
        "[red]red[/red]=slow (>3s)  │  "
        "RAM: [yellow]yellow[/yellow]=high  [red]red[/red]=critical[/dim]\n"
    )


# ---------------------------------------------------------------------------
# `autotune serve`
# ---------------------------------------------------------------------------

@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host.")
@click.option("--port", default=8765, show_default=True, type=int, help="Bind port.")
@click.option(
    "--reload", is_flag=True, default=False,
    help="Auto-reload on code changes (dev mode).",
)
@click.option(
    "--mlx", "enable_mlx", is_flag=True, default=False,
    help=(
        "Enable MLX backend on Apple Silicon (opt-in). "
        "Routes requests to mlx_lm for ~10–40%% higher throughput. "
        "Costs ~370 MB extra RAM and disables tool/function calling. "
        "Default is Ollama-only (~94 MB, full tool support)."
    ),
)
def serve(host: str, port: int, reload: bool, enable_mlx: bool) -> None:
    """Start the autotune OpenAI-compatible API server.

    Any OpenAI client can use it via base_url=http://HOST:PORT/v1
    """
    import os as _os
    if not enable_mlx:
        _os.environ["AUTOTUNE_DISABLE_MLX"] = "1"

    # Warn when binding to a non-loopback address: the server has no built-in
    # auth and CORS allow_origins=["*"], so any machine on the network (or any
    # website, via CORS) can drive local LLM inference.
    if host not in ("127.0.0.1", "::1", "localhost"):
        console.print(
            f"\n[bold yellow]⚠  Security warning[/bold yellow]\n"
            f"  Binding to [bold]{host}[/bold] exposes the autotune API to all "
            f"network interfaces.\n"
            f"  The server has no authentication — any client on your network "
            f"can send inference requests.\n"
            f"  Use [bold]--host 127.0.0.1[/bold] (default) for local-only access.\n"
            f"  If LAN access is intentional, consider a firewall rule or a reverse\n"
            f"  proxy (nginx/caddy) with TLS and an API key.\n"
        )

    # Opt-in telemetry prompt — shown exactly once on first run
    from autotune.telemetry import maybe_prompt_consent
    maybe_prompt_consent()

    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install 'uvicorn[standard]'[/red]")
        raise SystemExit(1)

    base = f"http://localhost:{port}"
    console.print(
        f"\n[bold green]autotune[/bold green] server running at [bold cyan]{base}/v1[/bold cyan]\n"
        f"\n[bold]Endpoints[/bold]\n"
        f"  [cyan]POST /v1/chat/completions[/cyan]   streaming · OpenAI-compatible\n"
        f"  [cyan]POST /v1/completions[/cyan]         FIM autocomplete (Continue.dev)\n"
        f"  [cyan]GET  /v1/models[/cyan]              available local models\n"
        f"  [cyan]GET  /health[/cyan]                 backend + memory status\n"
        f"\n[bold]Connect Open WebUI[/bold]  (easiest — use the launch command):\n"
        f"  [cyan]autotune webui launch[/cyan]   ← starts Open WebUI pre-wired to THIS server\n"
        f"\n  Or, if Open WebUI is already running:\n"
        f"  [cyan]autotune webui login[/cyan]    ← authenticate with Open WebUI\n"
        f"  [cyan]autotune webui connect[/cyan]  ← configure Open WebUI to route through autotune\n"
        f"\n  Manual (Admin Panel → Settings → Connections → OpenAI API):\n"
        f"    [dim]URL: {base}/v1    Key: autotune[/dim]\n"
        f"\n[bold]Other tools[/bold]\n"
        f"  [bold]Continue.dev[/bold]  config.json → models → add:\n"
        f'    [dim]{{"provider":"openai","model":"qwen3:8b","apiBase":"{base}","apiKey":"autotune"}}[/dim]\n'
        f"\n"
        f"  [bold]Python SDK[/bold]\n"
        f'    [dim]client = OpenAI(base_url="{base}/v1", api_key="autotune")[/dim]\n'
        f"\n"
        f"  [bold]curl[/bold]\n"
        f'    [dim]curl {base}/v1/models[/dim]\n'
    )

    uvicorn.run(
        "autotune.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )


# ---------------------------------------------------------------------------
# `autotune chat`
# ---------------------------------------------------------------------------

@cli.command("chat")
@click.option("--model", "-m", required=True, help="Model ID (e.g. llama3.2 or meta-llama/Meta-Llama-3.1-8B).")
@click.option(
    "--profile", "-p",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced",
    show_default=True,
    help="Optimization profile.",
)
@click.option("--system", "-s", default=None, metavar="TEXT", help="System prompt.")
@click.option("--conv-id", default=None, metavar="ID", help="Resume an existing conversation.")
@click.option(
    "--no-optimize", "no_optimize",
    is_flag=True, default=False,
    help="Disable real-time hardware/context optimization (use static profile settings).",
)
@click.option(
    "--no-swap", "no_swap",
    is_flag=True, default=False,
    help=(
        "Guarantee that inference will not trigger macOS swap.  "
        "Before each request, autotune measures available RAM, computes the exact "
        "KV cache size needed, and reduces context size / KV precision until it fits — "
        "keeping a 1.5 GB safety margin.  Prevents the catastrophic perf drop that "
        "occurs when Metal buffers spill to NVMe."
    ),
)
@click.option(
    "--recall",
    is_flag=True, default=False,
    help=(
        "Inject relevant context from past conversations into the system prompt. "
        "Requires an embedding model (run `autotune memory setup`). "
        "Off by default — conversations are always saved regardless."
    ),
)
def chat(
    model: str,
    profile: str,
    system: Optional[str],
    conv_id: Optional[str],
    no_optimize: bool,
    no_swap: bool,
    recall: bool,
) -> None:
    """Start an optimized terminal chat session with any model.

    Connects directly to Ollama / LM Studio / HuggingFace / MLX (whichever is
    available) — no `autotune serve` needed.

    Real-time optimization is ON by default:
      adaptive-RAM    monitors RAM each request, reduces context if pressure builds
      KV-manager      dynamically sizes and precision-tunes the KV cache
      context-optimizer  clips context to minimum needed, not profile maximum

    Conversations are always saved to local memory (search with `autotune memory search`).
    Use --recall to also inject relevant past context into the system prompt.

    Use --no-optimize to disable adaptive overrides (static profile settings only).
    Use `autotune run <model>` instead if you want a pre-flight memory analysis
    before loading (swap risk warnings, auto-profile selection).

    \b
    Examples:
      autotune chat --model llama3.2
      autotune chat --model qwen3:8b --recall
      autotune chat --model llama3.2 --system "You are a concise assistant"
      autotune chat --model qwen3:8b --no-optimize
      autotune chat --model qwen3:8b --no-swap
    """
    from autotune.telemetry import maybe_prompt_consent
    maybe_prompt_consent()
    from autotune.api.chat import start_chat
    start_chat(
        model_id=model,
        profile=profile,
        system_prompt=system,
        conv_id=conv_id,
        optimize=not no_optimize,
        no_swap=no_swap,
        recall=recall,
    )


# ---------------------------------------------------------------------------
# `autotune memory` — persistent conversation recall
# ---------------------------------------------------------------------------

@cli.group("memory")
def memory_group() -> None:
    """Search, browse, and manage your persistent conversation memory.

    autotune stores past conversations locally (SQLite + optional vector
    embeddings via Ollama) so future chat sessions can surface relevant
    context automatically.

    \b
    Examples:
      autotune memory search "postgres migration"
      autotune memory list --days 7
      autotune memory stats
      autotune memory forget 42
      autotune memory forget --all
      autotune memory setup           Pull nomic-embed-text for semantic search
    """


@memory_group.command("search")
@click.argument("query")
@click.option("--top", "-n", default=5, show_default=True, metavar="N",
              help="Number of results to return.")
@click.option("--min-score", default=0.20, show_default=True, metavar="SCORE",
              help="Minimum similarity score (0–1). Ignored for FTS5 fallback.")
def memory_search(query: str, top: int, min_score: float) -> None:
    """Search past conversations by semantic or keyword similarity.

    Uses vector search (cosine similarity) when an embedding model is
    available in Ollama, otherwise falls back to FTS5 keyword search.

    \b
    Examples:
      autotune memory search "postgres migration"
      autotune memory search "FastAPI authentication" --top 10
    """
    import asyncio
    from autotune.recall.manager import get_recall_manager
    from rich.table import Table
    from rich import box

    mgr = get_recall_manager()

    with console.status(f"[cyan]Searching memories for:[/cyan] {query!r}", spinner="dots"):
        results = asyncio.run(mgr.search(query, top_k=top, min_score=min_score))

    if not results:
        console.print(f"[dim]No memories found for: {query!r}[/dim]")
        return

    console.print(f"\n[bold]Memory search:[/bold] {query!r}  [dim]({len(results)} result(s))[/dim]\n")

    for r in results:
        score_str = f"score={r.score:.3f}" if r.score else "fts"
        model_label = (r.model_id or "unknown").split(":")[0]
        console.print(
            f"  [dim][#{r.id}  {r.age_str}  ·  {model_label}  ·  {score_str}][/dim]"
        )
        # Show chunk with some formatting
        text = r.chunk_text
        if len(text) > 400:
            text = text[:400].rsplit(" ", 1)[0] + "…"
        console.print(f"  {text}\n")


@memory_group.command("list")
@click.option("--limit", "-n", default=20, show_default=True, metavar="N",
              help="Number of memories to show.")
@click.option("--days", default=None, type=int, metavar="DAYS",
              help="Only show memories from the last N days.")
@click.option("--model", "model_id", default=None, metavar="MODEL",
              help="Filter by model (e.g. qwen3:8b).")
def memory_list(limit: int, days: Optional[int], model_id: Optional[str]) -> None:
    """List recently stored conversation memories.

    \b
    Examples:
      autotune memory list
      autotune memory list --days 7
      autotune memory list --model qwen3:8b --limit 50
    """
    from autotune.recall.manager import get_recall_manager
    from rich.table import Table
    from rich import box

    mgr = get_recall_manager()
    results = mgr.get_recent(limit=limit, model_id=model_id, days=days)

    if not results:
        console.print("[dim]No memories found. Start chatting to build memory![/dim]")
        return

    t = Table(box=box.SIMPLE_HEAD, show_lines=False)
    t.add_column("#",       style="dim", justify="right", width=5)
    t.add_column("Age",     style="dim", no_wrap=True)
    t.add_column("Model",   style="cyan", no_wrap=True)
    t.add_column("Preview", no_wrap=False)

    for r in results:
        model_label = (r.model_id or "unknown").split(":")[0]
        # First line of the chunk as preview
        preview = r.chunk_text.replace("\n", " ")[:80]
        if len(r.chunk_text) > 80:
            preview += "…"
        t.add_row(str(r.id), r.age_str, model_label, preview)

    header = "[bold]Recent memories[/bold]"
    if days:
        header += f"  [dim](last {days} days)[/dim]"
    if model_id:
        header += f"  [dim](model: {model_id})[/dim]"
    console.print()
    console.print(header)
    console.print(t)
    console.print(f"[dim]{len(results)} chunk(s)  ·  autotune memory forget <id> to remove[/dim]\n")


@memory_group.command("stats")
def memory_stats() -> None:
    """Show statistics about the local memory store."""
    import asyncio
    from autotune.recall.manager import get_recall_manager
    import datetime

    mgr = get_recall_manager()
    stats = mgr.stats()

    with console.status("[dim]Checking embedding availability…[/dim]", spinner="dots"):
        embed_status = asyncio.run(mgr.embedder_status())

    console.print()
    console.print("[bold]Memory store[/bold]")
    console.print(f"  Total chunks   : {stats['total_chunks']}")
    console.print(f"  With vectors   : {stats['with_embeddings']}")
    console.print(f"  DB size        : {stats['size_mb']} MB")

    if stats.get("oldest_at"):
        oldest = datetime.datetime.fromtimestamp(stats["oldest_at"]).strftime("%Y-%m-%d")
        newest = datetime.datetime.fromtimestamp(stats["newest_at"]).strftime("%Y-%m-%d")
        console.print(f"  Date range     : {oldest} → {newest}")

    if stats.get("by_model"):
        console.print(f"  By model:")
        for model, cnt in stats["by_model"].items():
            console.print(f"    [cyan]{model}[/cyan]  {cnt} chunk(s)")

    embed_str = (
        f"[green]{embed_status['model']}[/green]"
        if embed_status["available"]
        else "[yellow]not available — run: autotune memory setup[/yellow]"
    )
    console.print(f"  Embedding      : {embed_str}")
    console.print()


@memory_group.command("forget")
@click.argument("memory_id", required=False, type=int, default=None)
@click.option("--all", "forget_all", is_flag=True, default=False,
              help="Delete ALL memories (cannot be undone).")
@click.option("--conv-id", "conv_id", default=None, metavar="ID",
              help="Delete all memories for a specific conversation ID.")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation prompt.")
def memory_forget(
    memory_id: Optional[int],
    forget_all: bool,
    conv_id: Optional[str],
    yes: bool,
) -> None:
    """Delete memories from the local store.

    \b
    Examples:
      autotune memory forget 42               Delete memory chunk #42
      autotune memory forget --conv-id abc123  Delete all chunks for a conversation
      autotune memory forget --all            Wipe everything (with confirmation)
    """
    from autotune.recall.manager import get_recall_manager

    mgr = get_recall_manager()

    if forget_all:
        if not yes:
            try:
                ans = input("Delete ALL memories? This cannot be undone. [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Cancelled.[/yellow]")
                raise SystemExit(0)
            if ans not in ("y", "yes"):
                console.print("[yellow]Cancelled.[/yellow]")
                raise SystemExit(0)
        removed = mgr.delete_all()
        console.print(f"[green]✓ Deleted {removed} memory chunk(s).[/green]")

    elif conv_id:
        removed = mgr.delete_conversation(conv_id)
        if removed:
            console.print(f"[green]✓ Deleted {removed} chunk(s) for conv {conv_id}.[/green]")
        else:
            console.print(f"[yellow]No memories found for conv ID: {conv_id}[/yellow]")

    elif memory_id is not None:
        ok = mgr.delete(memory_id)
        if ok:
            console.print(f"[green]✓ Deleted memory #{memory_id}.[/green]")
        else:
            console.print(f"[yellow]Memory #{memory_id} not found.[/yellow]")

    else:
        console.print(
            "[dim]Specify a memory ID, --conv-id, or --all.\n"
            "Use [bold]autotune memory list[/bold] to see IDs.[/dim]"
        )


@memory_group.command("setup")
def memory_setup() -> None:
    """Pull nomic-embed-text from Ollama to enable semantic (vector) search.

    Without an embedding model, autotune falls back to FTS5 keyword search.
    nomic-embed-text is fast, small (~274 MB), and works well for
    conversation retrieval.

    \b
      autotune memory setup         Pull nomic-embed-text (recommended)
    """
    import subprocess

    console.print(
        "[bold]Memory setup[/bold]\n\n"
        "This will pull [cyan]nomic-embed-text[/cyan] from Ollama (~274 MB).\n"
        "It enables semantic (vector) search across your conversation history.\n"
    )

    try:
        ans = input("Pull nomic-embed-text now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Cancelled.[/yellow]")
        raise SystemExit(0)

    if ans not in ("", "y", "yes"):
        console.print("[yellow]Cancelled.[/yellow]")
        raise SystemExit(0)

    console.print("[dim]Running: ollama pull nomic-embed-text[/dim]\n")
    try:
        subprocess.run(["ollama", "pull", "nomic-embed-text"], check=True)
        console.print(
            "\n[green]✓ Done.[/green]  Semantic search is now active.\n"
            "[dim]Future conversations will be embedded automatically.[/dim]"
        )
    except subprocess.CalledProcessError:
        console.print("[red]Pull failed.[/red] Make sure Ollama is running: [bold]ollama serve[/bold]")
        raise SystemExit(1)
    except FileNotFoundError:
        console.print("[red]ollama not found.[/red] Install it from https://ollama.ai")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# `autotune mlx` — MLX model management (Apple Silicon only)
# ---------------------------------------------------------------------------

@cli.group("mlx")
def mlx_group() -> None:
    """Manage MLX models for Apple Silicon acceleration.

    MLX runs LLMs entirely on-chip using Apple's unified memory and Metal GPU
    kernels — typically 10–40% faster than Ollama on the same model.

    \b
    Examples:
      autotune mlx list               Show cached MLX models
      autotune mlx pull qwen3:8b      Pull the MLX version of qwen3:8b
      autotune mlx resolve llama3.2   Show which MLX model would be used
    """


@mlx_group.command("list")
def mlx_list() -> None:
    """List MLX models available locally (already downloaded)."""
    from autotune.api.backends.mlx_backend import (
        IS_APPLE_SILICON, mlx_available, list_cached_mlx_models,
    )

    if not IS_APPLE_SILICON:
        console.print("[yellow]MLX is only available on Apple Silicon (arm64) Macs.[/yellow]")
        return

    if not mlx_available():
        console.print(
            "[yellow]mlx-lm is not installed.[/yellow]\n"
            "Install it with:  [bold]pip install mlx-lm[/bold]"
        )
        return

    models = list_cached_mlx_models()
    if not models:
        console.print(
            "[dim]No MLX models cached locally.[/dim]\n"
            "Pull one with:  [bold]autotune mlx pull <model>[/bold]"
        )
        return

    from rich.table import Table
    table = Table(title="Cached MLX Models", header_style="bold magenta")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right", style="green")

    for m in sorted(models, key=lambda x: x["id"]):
        size = f"{m['size_gb']:.1f} GB" if m["size_gb"] else "–"
        table.add_row(m["id"], size)

    console.print(table)
    console.print(f"[dim]{len(models)} model(s) cached locally[/dim]")


@mlx_group.command("pull")
@click.argument("model")
@click.option(
    "--quant", "-q",
    default="4bit",
    show_default=True,
    type=click.Choice(["4bit", "8bit", "bf16"]),
    help="Quantization level to pull.",
)
def mlx_pull(model: str, quant: str) -> None:
    """Pull an MLX-quantized model from mlx-community on HuggingFace.

    MODEL can be an Ollama model name (e.g. qwen3:8b, llama3.2:3b) or a
    full HuggingFace model ID (e.g. mlx-community/Qwen3-8B-4bit).

    \b
    Examples:
      autotune mlx pull qwen3:8b
      autotune mlx pull llama3.2:3b
      autotune mlx pull qwen2.5-coder:14b --quant 8bit
    """
    from autotune.api.backends.mlx_backend import (
        IS_APPLE_SILICON, mlx_available, resolve_mlx_model_id,
    )

    if not IS_APPLE_SILICON:
        console.print("[yellow]MLX is only available on Apple Silicon Macs.[/yellow]")
        raise SystemExit(1)

    if not mlx_available():
        console.print(
            "[yellow]mlx-lm is not installed.[/yellow]\n"
            "Install it with:  [bold]pip install mlx-lm[/bold]"
        )
        raise SystemExit(1)

    # Resolve model ID
    mlx_id = resolve_mlx_model_id(model)
    if mlx_id is None:
        # Build a best-guess ID from the model name + quant
        base = model.split(":")[0].split("/")[-1]
        # Normalise common names: capitalise first char of each word
        words = [w.capitalize() for w in base.replace("-", " ").replace("_", " ").split()]
        guess = f"mlx-community/{''.join(words)}-instruct-{quant}"
        console.print(
            f"[yellow]No known MLX mapping for '{model}'.[/yellow]\n"
            f"Trying:  [cyan]{guess}[/cyan]\n"
            "[dim](If this fails, browse https://huggingface.co/mlx-community for the exact name)[/dim]"
        )
        mlx_id = guess
    else:
        console.print(f"Resolved  [cyan]{model}[/cyan]  →  [cyan]{mlx_id}[/cyan]")

    console.print(f"[bold]Downloading {mlx_id}…[/bold]  (this may take a while)")

    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=mlx_id, ignore_patterns=["*.md", "*.txt"])
        console.print(f"[green]✓ Downloaded to:[/green] {local_dir}")
        console.print(
            f"\nRun inference with:  [bold]autotune chat --model {model}[/bold]"
        )
    except Exception as exc:
        console.print(f"[red]Download failed:[/red] {exc}")
        console.print(
            "[dim]Tip: You may need to accept the model's license on HuggingFace first.[/dim]"
        )
        raise SystemExit(1)


@mlx_group.command("resolve")
@click.argument("model")
def mlx_resolve(model: str) -> None:
    """Show which MLX model ID would be used for MODEL."""
    from autotune.api.backends.mlx_backend import (
        IS_APPLE_SILICON, mlx_available, resolve_mlx_model_id,
    )

    if not IS_APPLE_SILICON:
        console.print("[yellow]Not on Apple Silicon — MLX not active.[/yellow]")
        return

    mlx_id = resolve_mlx_model_id(model)
    if mlx_id:
        console.print(f"[cyan]{model}[/cyan]  →  [green]{mlx_id}[/green]")
    else:
        console.print(
            f"[yellow]No MLX mapping for '{model}'.[/yellow]\n"
            f"Will fall back to Ollama.\n"
            f"Pull an MLX version with:  [bold]autotune mlx pull {model}[/bold]"
        )


# ---------------------------------------------------------------------------
# `autotune stress-test`  — multi-model proof-of-improvement benchmark
# ---------------------------------------------------------------------------

_STRESS_PROMPTS = [
    {
        "id": "short",
        "label": "Short/Direct",
        "messages": [
            {"role": "user", "content": "List three benefits of regular exercise. Be concise."}
        ],
    },
    {
        "id": "code",
        "label": "Code Generation",
        "messages": [
            {"role": "user",
             "content": (
                 "Write a Python function called `two_sum` that takes a list of integers "
                 "and a target integer, and returns the indices of the two numbers that "
                 "add up to the target. Include type hints and a brief docstring."
             )}
        ],
    },
    {
        "id": "reasoning",
        "label": "Reasoning/Comparison",
        "messages": [
            {"role": "user",
             "content": (
                 "Explain the difference between supervised learning, unsupervised learning, "
                 "and reinforcement learning. Give one concrete real-world application of each."
             )}
        ],
    },
    {
        "id": "analysis",
        "label": "Analysis/Review",
        "messages": [
            {"role": "user",
             "content": (
                 "You are a senior software engineer reviewing the following code. "
                 "Identify all security vulnerabilities, performance issues, and design problems. "
                 "Be specific:\n\n"
                 "def get_user(user_id, db):\n"
                 "    result = db.execute('SELECT * FROM users WHERE id=' + str(user_id))\n"
                 "    return result[0]\n\n"
                 "def login(username, password, db):\n"
                 "    user = db.execute(f'SELECT * FROM users WHERE name={username}').fetchone()\n"
                 "    if user['password'] == password:\n"
                 "        return {'token': user['id'], 'admin': user.get('is_admin', False)}\n"
             )}
        ],
    },
]

# Models to recommend for auto-pull, in priority order, with min RAM (GB) required
_RECOMMENDED_PULL_MODELS = [
    ("qwen3:8b",      5.5, "Best general reasoning — Qwen3 text-only 8B"),
    ("llama3.2:3b",   2.5, "Fast baseline — Llama 3.2 3B"),
]

# Models to exclude from automated multi-model benchmarks.
# Add models here that produce unreliable results or that you want to exclude
# from comparative runs (e.g. models with known quirks or that need special flags).
_SKIP_MODELS: set[str] = set()


def _ollama_list_models() -> list[dict]:
    """Return list of {name, size_gb} from ollama list."""
    import subprocess, re
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, timeout=10)
        models = []
        for line in out.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            size_gb = 0.0
            for i, p in enumerate(parts):
                if p in ("GB", "MB") and i > 0:
                    try:
                        val = float(parts[i-1])
                        size_gb = val if p == "GB" else val / 1024
                    except ValueError:
                        pass
            models.append({"name": name, "size_gb": size_gb})
        return models
    except Exception:
        return []


def _ollama_pull_model(model: str) -> bool:
    """Pull a model via ollama pull. Returns True on success."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            timeout=600,
            capture_output=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _select_stress_models(
    available_ram_gb: float,
    force_models: list[str] | None,
    auto_pull: bool,
) -> list[str]:
    """Choose which models to run. Returns list of model IDs."""
    pulled = {m["name"] for m in _ollama_list_models()}

    if force_models:
        # Validate they're pulled
        missing = [m for m in force_models if m not in pulled]
        if missing:
            console.print(f"[yellow]Warning: these models are not pulled: {', '.join(missing)}[/yellow]")
        return force_models

    # Start with what's already pulled, filtered by RAM and skip list
    selected = []
    for m in _ollama_list_models():
        name = m["name"]
        if any(skip in name for skip in _SKIP_MODELS):
            continue
        # Model should fit with ~4GB OS headroom
        if m["size_gb"] > 0 and m["size_gb"] > available_ram_gb - 4.0:
            console.print(
                f"  [yellow]Skipping {name} ({m['size_gb']:.1f} GB) — "
                f"too large for {available_ram_gb:.0f} GB RAM[/yellow]"
            )
            continue
        selected.append(name)

    # Auto-pull recommended models that fit and aren't already pulled
    if auto_pull:
        for model_id, min_ram, desc in _RECOMMENDED_PULL_MODELS:
            if model_id in pulled:
                continue
            if available_ram_gb < min_ram + 4.0:
                console.print(f"  [dim]Skipping auto-pull {model_id} — not enough RAM[/dim]")
                continue
            console.print(f"\n  [bold cyan]Auto-pulling {model_id}[/bold cyan]  ({desc})")
            ok = _ollama_pull_model(model_id)
            if ok:
                console.print(f"  [green]✓  {model_id} pulled[/green]")
                selected.append(model_id)
            else:
                console.print(f"  [red]✗  Failed to pull {model_id}[/red]")

    return selected


@cli.command("stress-test")
@click.option(
    "--models", "model_list", default=None, metavar="MODEL,MODEL,...",
    help=(
        "Comma-separated Ollama model IDs to test. "
        "If omitted, auto-selects all pulled models that fit in RAM."
    ),
)
@click.option(
    "--runs", default=3, show_default=True, type=int,
    help="Inference runs per model per mode (raw + autotune). Min 1, recommended 3.",
)
@click.option(
    "--profile", "-p",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced", show_default=True,
    help="autotune optimization profile to benchmark against raw Ollama.",
)
@click.option(
    "--output", "-o", default=None, type=click.Path(),
    help="Path to save full JSON results (auto-named if omitted).",
)
@click.option(
    "--auto-pull", "auto_pull", is_flag=True, default=False,
    help="Automatically pull recommended models (qwen3:8b, llama3.2:3b) if not present.",
)
@click.option(
    "--fast", is_flag=True, default=False,
    help="1 run per cell instead of 3 — quick proof-of-concept (less statistically robust).",
)
@click.option(
    "--prompts", "prompt_filter", default=None, metavar="ID,ID,...",
    help=f"Run only specific prompt IDs: {', '.join(p['id'] for p in _STRESS_PROMPTS)}. "
         "Default: all 4.",
)
def stress_test(
    model_list: str | None,
    runs: int,
    profile: str,
    output: str | None,
    auto_pull: bool,
    fast: bool,
    prompt_filter: str | None,
) -> None:
    """Multi-model stress test: raw Ollama vs autotune, side-by-side proof.

    Runs every pulled model (or the ones you specify) through 4 diverse
    prompts × N runs each, sampling CPU/RAM every 250 ms. Produces
    per-model comparison tables and a final aggregate verdict.

    \b
    Quick start (uses all pulled models, 3 runs each):
      autotune stress-test

    \b
    With auto-pull of best models:
      autotune stress-test --auto-pull

    \b
    Test specific models:
      autotune stress-test --models qwen3-vl:8b,qwen2.5-coder:14b

    \b
    Fast 1-run proof-of-concept:
      autotune stress-test --fast

    \b
    Save results to a specific file:
      autotune stress-test --output results/stress_$(date +%Y%m%d).json
    """
    import asyncio
    import json
    import statistics
    import time as _time
    from datetime import datetime
    from pathlib import Path

    import psutil as _psutil
    from rich import box
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    from autotune.bench.runner import run_bench_ollama_only, run_raw_ollama, save_result

    n_runs = 1 if fast else max(1, runs)
    active_prompts = _STRESS_PROMPTS
    if prompt_filter:
        ids = {p.strip() for p in prompt_filter.split(",")}
        active_prompts = [p for p in _STRESS_PROMPTS if p["id"] in ids]
        if not active_prompts:
            console.print(f"[red]No prompts matched filter '{prompt_filter}'[/red]")
            raise SystemExit(1)

    # ── System check ─────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]autotune stress-test[/bold]", style="cyan"))
    console.print()

    vm = _psutil.virtual_memory()
    total_ram_gb = vm.total / 1024**3
    avail_ram_gb = vm.available / 1024**3

    console.print(f"[bold]Step 1 of 5[/bold]  Checking system resources…")
    console.print(f"  RAM: [cyan]{total_ram_gb:.1f} GB[/cyan] total  /  "
                  f"[green]{avail_ram_gb:.1f} GB[/green] available")

    swap = _psutil.swap_memory()
    if swap.used > 0.5 * 1024**3:
        console.print(
            f"  [yellow]⚠  Swap already in use ({swap.used/1024**3:.1f} GB) — "
            f"results may be noisier.[/yellow]"
        )

    # ── Model selection ───────────────────────────────────────────────────
    console.print(f"\n[bold]Step 2 of 5[/bold]  Selecting models to test…")
    force_models = [m.strip() for m in model_list.split(",")] if model_list else None
    selected_models = _select_stress_models(total_ram_gb, force_models, auto_pull)

    if not selected_models:
        console.print(
            "[red]No models available to test.[/red]\n"
            "Pull a model first:  [bold]ollama pull qwen3-vl:8b[/bold]\n"
            "Or use --auto-pull to download recommended models automatically."
        )
        raise SystemExit(1)

    console.print(f"  [green]✓  {len(selected_models)} model(s) selected:[/green]  "
                  f"{', '.join(selected_models)}")

    # ── Plan overview ─────────────────────────────────────────────────────
    console.print(f"\n[bold]Step 3 of 5[/bold]  Planning test matrix…")
    total_calls = len(selected_models) * len(active_prompts) * n_runs * 2  # raw + autotune
    console.print(f"  Models:   [cyan]{len(selected_models)}[/cyan]")
    console.print(f"  Prompts:  [cyan]{len(active_prompts)}[/cyan]  "
                  f"({', '.join(p['label'] for p in active_prompts)})")
    console.print(f"  Runs/cell:[cyan]{n_runs}[/cyan]  (raw + autotune = {n_runs*2} calls per prompt)")
    console.print(f"  Total inference calls: [bold cyan]{total_calls}[/bold cyan]")
    if fast:
        console.print("  [dim]Fast mode: 1 run per cell — less noise averaging but faster.[/dim]")
    console.print()
    console.print(f"[bold]Step 4 of 5[/bold]  Running benchmarks…")
    console.print(f"  [dim]psutil sampling every 250 ms · 3 s cooldown between runs · "
                  f"10 s cooldown between models[/dim]")
    console.print()

    # ── Main benchmark loop ───────────────────────────────────────────────
    all_results: list[dict] = []   # raw storage for JSON export
    model_summaries: list[dict] = []  # per-model aggregate for final table

    run_start_wall = _time.time()

    for m_idx, model_id in enumerate(selected_models):
        console.print(Rule(
            f"[bold cyan]Model {m_idx+1}/{len(selected_models)}: {model_id}[/bold cyan]",
            style="cyan",
        ))

        raw_cells: dict[str, list[float]] = {
            "ttft": [], "tps": [], "cpu": [], "ram_delta": [], "elapsed": []
        }
        tune_cells: dict[str, list[float]] = {
            "ttft": [], "tps": [], "cpu": [], "ram_delta": [], "elapsed": []
        }

        for p_idx, prompt in enumerate(active_prompts):
            console.print(
                f"\n  [bold]Prompt {p_idx+1}/{len(active_prompts)}:[/bold]  "
                f"[dim]{prompt['label']}[/dim]"
            )

            # ── RAW OLLAMA ────────────────────────────────────────────────
            console.print(f"    [white]▷ Raw Ollama[/white]  ({n_runs} run{'s' if n_runs>1 else ''})…")
            for r in range(n_runs):
                with console.status(
                    f"      [dim]Raw run {r+1}/{n_runs}…[/dim]", spinner="dots"
                ):
                    res = asyncio.run(
                        run_raw_ollama(
                            model_id,
                            prompt["messages"],
                            tag=f"stress_raw_{model_id}_{prompt['id']}",
                        )
                    )
                if res.error:
                    console.print(f"      [red]✗ Error: {res.error[:80]}[/red]")
                else:
                    raw_cells["ttft"].append(res.ttft_ms)
                    raw_cells["tps"].append(res.tokens_per_sec)
                    raw_cells["cpu"].append(res.cpu_avg_pct)
                    raw_cells["ram_delta"].append(res.delta_ram_gb)
                    raw_cells["elapsed"].append(res.elapsed_sec)
                    save_result(res)
                    console.print(
                        f"      [green]✓[/green]  TTFT [bold]{res.ttft_ms:.0f}ms[/bold]  "
                        f"· {res.tokens_per_sec:.1f} tok/s  "
                        f"· CPU {res.cpu_avg_pct:.0f}%  "
                        f"· ΔRAM {res.delta_ram_gb:+.2f}GB"
                    )
                    all_results.append({
                        "model": model_id, "prompt": prompt["id"],
                        "mode": "raw", "run": r+1,
                        "ttft_ms": res.ttft_ms, "tps": res.tokens_per_sec,
                        "cpu_avg_pct": res.cpu_avg_pct, "delta_ram_gb": res.delta_ram_gb,
                        "elapsed_sec": res.elapsed_sec, "error": res.error,
                    })
                if r < n_runs - 1:
                    _time.sleep(3)

            _time.sleep(3)  # cooldown before autotune

            # ── AUTOTUNE ──────────────────────────────────────────────────
            console.print(
                f"    [bold green]▷ autotune/{profile}[/bold green]  "
                f"({n_runs} run{'s' if n_runs>1 else ''})…"
            )
            for r in range(n_runs):
                with console.status(
                    f"      [dim]Autotune run {r+1}/{n_runs}…[/dim]", spinner="dots"
                ):
                    res = asyncio.run(
                        run_bench_ollama_only(
                            model_id,
                            prompt["messages"],
                            profile_name=profile,
                            tag=f"stress_autotune_{model_id}_{prompt['id']}",
                        )
                    )
                if res.error:
                    console.print(f"      [red]✗ Error: {res.error[:80]}[/red]")
                else:
                    tune_cells["ttft"].append(res.ttft_ms)
                    tune_cells["tps"].append(res.tokens_per_sec)
                    tune_cells["cpu"].append(res.cpu_avg_pct)
                    tune_cells["ram_delta"].append(res.delta_ram_gb)
                    tune_cells["elapsed"].append(res.elapsed_sec)
                    save_result(res)
                    console.print(
                        f"      [green]✓[/green]  TTFT [bold]{res.ttft_ms:.0f}ms[/bold]  "
                        f"· {res.tokens_per_sec:.1f} tok/s  "
                        f"· CPU {res.cpu_avg_pct:.0f}%  "
                        f"· ΔRAM {res.delta_ram_gb:+.2f}GB"
                    )
                    all_results.append({
                        "model": model_id, "prompt": prompt["id"],
                        "mode": "autotune", "run": r+1, "profile": profile,
                        "ttft_ms": res.ttft_ms, "tps": res.tokens_per_sec,
                        "cpu_avg_pct": res.cpu_avg_pct, "delta_ram_gb": res.delta_ram_gb,
                        "elapsed_sec": res.elapsed_sec, "error": res.error,
                    })
                if r < n_runs - 1:
                    _time.sleep(3)

            _time.sleep(3)  # cooldown before next prompt

        # ── Per-model comparison table ─────────────────────────────────
        def _avg(lst: list[float]) -> float:
            return statistics.mean(lst) if lst else float("nan")

        def _win(raw_v: float, tune_v: float, lower_is_better: bool = True) -> str:
            if lower_is_better:
                diff_pct = (tune_v - raw_v) / max(raw_v, 0.01) * 100
                if tune_v < raw_v * 0.97:
                    return f"[green]✓ {diff_pct:+.1f}%[/green]"
                elif tune_v > raw_v * 1.03:
                    return f"[red]✗ {diff_pct:+.1f}%[/red]"
                return f"[dim]≈ {diff_pct:+.1f}%[/dim]"
            else:
                diff_pct = (tune_v - raw_v) / max(raw_v, 0.01) * 100
                if tune_v > raw_v * 1.03:
                    return f"[green]✓ +{diff_pct:.1f}%[/green]"
                elif tune_v < raw_v * 0.97:
                    return f"[red]✗ {diff_pct:+.1f}%[/red]"
                return f"[dim]≈ {diff_pct:+.1f}%[/dim]"

        r_ttft  = _avg(raw_cells["ttft"])
        t_ttft  = _avg(tune_cells["ttft"])
        r_tps   = _avg(raw_cells["tps"])
        t_tps   = _avg(tune_cells["tps"])
        r_cpu   = _avg(raw_cells["cpu"])
        t_cpu   = _avg(tune_cells["cpu"])
        r_ram   = _avg(raw_cells["ram_delta"])
        t_ram   = _avg(tune_cells["ram_delta"])
        r_elap  = _avg(raw_cells["elapsed"])
        t_elap  = _avg(tune_cells["elapsed"])

        tbl = Table(
            title=f"  {model_id}  —  {n_runs * len(active_prompts)} total runs per mode",
            box=box.ROUNDED, show_header=True, header_style="bold",
            title_style="bold cyan",
        )
        tbl.add_column("Metric",         style="bold", min_width=18)
        tbl.add_column("Raw Ollama",     style="white",  justify="right", min_width=14)
        tbl.add_column(f"autotune/{profile}", style="cyan", justify="right", min_width=14)
        tbl.add_column("Result",         justify="center", min_width=14)

        tbl.add_row(
            "TTFT (ms)", f"{r_ttft:.0f}", f"{t_ttft:.0f}",
            _win(r_ttft, t_ttft, lower_is_better=True),
        )
        tbl.add_row(
            "Throughput (tok/s)", f"{r_tps:.1f}", f"{t_tps:.1f}",
            _win(r_tps, t_tps, lower_is_better=False),
        )
        tbl.add_row(
            "CPU avg (%)", f"{r_cpu:.1f}", f"{t_cpu:.1f}",
            _win(r_cpu, t_cpu, lower_is_better=True),
        )
        tbl.add_row(
            "Elapsed (s)", f"{r_elap:.2f}", f"{t_elap:.2f}",
            _win(r_elap, t_elap, lower_is_better=True),
        )
        tbl.add_row(
            "ΔRAM (GB)", f"{r_ram:+.3f}", f"{t_ram:+.3f}",
            _win(r_ram, t_ram, lower_is_better=True),
        )

        console.print()
        console.print(tbl)

        # Count autotune wins for this model
        wins = 0
        total_metrics = 5
        if t_ttft  < r_ttft  * 0.97: wins += 1
        if t_tps   > r_tps   * 1.03: wins += 1
        if t_cpu   < r_cpu   * 0.97: wins += 1
        if t_elap  < r_elap  * 0.97: wins += 1
        if t_ram   < r_ram   * 0.97: wins += 1

        model_summaries.append({
            "model": model_id,
            "raw_ttft":  r_ttft,  "tune_ttft":  t_ttft,
            "raw_tps":   r_tps,   "tune_tps":   t_tps,
            "raw_cpu":   r_cpu,   "tune_cpu":   t_cpu,
            "raw_elap":  r_elap,  "tune_elap":  t_elap,
            "raw_ram":   r_ram,   "tune_ram":   t_ram,
            "wins": wins, "total_metrics": total_metrics,
        })

        if m_idx < len(selected_models) - 1:
            console.print(f"\n  [dim]Cooling down 10 s before next model…[/dim]")
            _time.sleep(10)

    # ── Step 5: Aggregate verdict ──────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Step 5 of 5 — Final Verdict[/bold]", style="cyan"))
    console.print()

    agg = Table(
        title="Aggregate Results — All Models",
        box=box.HEAVY_EDGE, show_header=True, header_style="bold",
        title_style="bold white",
    )
    agg.add_column("Model",       style="cyan",  min_width=22)
    agg.add_column("TTFT raw→tune",  justify="right", min_width=16)
    agg.add_column("tok/s raw→tune", justify="right", min_width=16)
    agg.add_column("CPU raw→tune",   justify="right", min_width=14)
    agg.add_column("Wins",           justify="center", min_width=10)

    total_wins = 0
    total_possible = 0

    for s in model_summaries:
        ttft_arrow  = "→" if abs(s["tune_ttft"] - s["raw_ttft"]) / max(s["raw_ttft"], 1) < 0.03 else (
            "↓" if s["tune_ttft"] < s["raw_ttft"] else "↑"
        )
        tps_arrow   = "→" if abs(s["tune_tps"]  - s["raw_tps"])  / max(s["raw_tps"],  0.01) < 0.03 else (
            "↑" if s["tune_tps"] > s["raw_tps"] else "↓"
        )
        cpu_arrow   = "→" if abs(s["tune_cpu"]  - s["raw_cpu"])  / max(s["raw_cpu"],  0.01) < 0.03 else (
            "↓" if s["tune_cpu"] < s["raw_cpu"] else "↑"
        )

        ttft_color  = "green" if ttft_arrow == "↓" else ("red" if ttft_arrow == "↑" else "dim")
        tps_color   = "green" if tps_arrow  == "↑" else ("red" if tps_arrow  == "↓" else "dim")
        cpu_color   = "green" if cpu_arrow  == "↓" else ("red" if cpu_arrow  == "↑" else "dim")

        wins_str = f"{s['wins']}/{s['total_metrics']}"
        wins_color = "green" if s["wins"] >= 4 else ("yellow" if s["wins"] >= 2 else "red")

        agg.add_row(
            s["model"],
            f"[{ttft_color}]{s['raw_ttft']:.0f} {ttft_arrow} {s['tune_ttft']:.0f} ms[/{ttft_color}]",
            f"[{tps_color}]{s['raw_tps']:.1f} {tps_arrow} {s['tune_tps']:.1f}[/{tps_color}]",
            f"[{cpu_color}]{s['raw_cpu']:.0f} {cpu_arrow} {s['tune_cpu']:.0f}%[/{cpu_color}]",
            f"[{wins_color}]{wins_str}[/{wins_color}]",
        )
        total_wins += s["wins"]
        total_possible += s["total_metrics"]

    console.print(agg)
    console.print()

    # Verdict banner
    win_rate = total_wins / max(total_possible, 1)
    total_wall = _time.time() - run_start_wall

    if win_rate >= 0.70:
        verdict_color = "bold green"
        verdict_icon  = "🏆"
        verdict_text  = (
            f"autotune OUTPERFORMS raw Ollama on [bold]{total_wins}/{total_possible}[/bold] metrics "
            f"across {len(selected_models)} model(s) — "
            f"strong proof of improvement"
        )
    elif win_rate >= 0.50:
        verdict_color = "bold yellow"
        verdict_icon  = "▲"
        verdict_text  = (
            f"autotune IMPROVES raw Ollama on [bold]{total_wins}/{total_possible}[/bold] metrics "
            f"across {len(selected_models)} model(s)"
        )
    else:
        verdict_color = "bold red"
        verdict_icon  = "⚠"
        verdict_text  = (
            f"Mixed results: autotune wins {total_wins}/{total_possible} metrics. "
            f"Consider tuning profile or reviewing system state."
        )

    console.print(Panel(
        f"[{verdict_color}]{verdict_icon}  {verdict_text}[/{verdict_color}]\n\n"
        f"[dim]{len(selected_models)} model(s) · "
        f"{len(active_prompts)} prompt(s) · "
        f"{n_runs} run(s)/cell · "
        f"{total_calls} total inference calls · "
        f"wall time {total_wall/60:.1f} min[/dim]",
        title="[bold]Stress-Test Verdict[/bold]",
        border_style=verdict_color,
        padding=(1, 2),
    ))
    console.print()

    # ── Save JSON ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output or f"stress_test_{ts}.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "profile": profile,
        "runs_per_cell": n_runs,
        "models": selected_models,
        "prompts": [p["id"] for p in active_prompts],
        "total_inference_calls": total_calls,
        "wall_time_sec": round(total_wall, 1),
        "win_rate": round(win_rate, 3),
        "total_wins": total_wins,
        "total_possible": total_possible,
        "model_summaries": model_summaries,
        "raw_results": all_results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    console.print(f"  [green]✓  Full results saved →[/green] [bold]{out_path}[/bold]")
    console.print(
        f"\n  [dim]Reproduce this run:[/dim]\n"
        f"  [bold]autotune stress-test "
        f"--models {','.join(selected_models)} "
        f"--runs {n_runs} "
        f"--profile {profile}[/bold]\n"
    )


# ---------------------------------------------------------------------------
# `autotune proof`  — honest, traceable benchmark
# ---------------------------------------------------------------------------

@cli.command("proof")
@click.option("--model", "-m", default="qwen3:8b",
              help="Ollama model to benchmark (default: qwen3:8b)")
@click.option("--runs", "-r", type=int, default=3,
              help="Warm inference runs per prompt per config (default: 3)")
@click.option("--cold-runs", type=int, default=3,
              help="Cold-start calls per config (default: 3)")
@click.option("--output", "-o", default="proof_results.json",
              help="JSON output path (default: proof_results.json)")
@click.option("--with-cold",   is_flag=True, help="Include cold-start phase")
@click.option("--with-noswap", is_flag=True, help="Include no-swap mode demonstration")
@click.option("--list-models", is_flag=True, help="List available Ollama models and exit")
def proof(
    model: str,
    runs: int,
    cold_runs: int,
    output: str,
    with_cold: bool,
    with_noswap: bool,
    list_models: bool,
) -> None:
    """
    Does autotune actually help? Run this to find out.

    Measures three things on your model:\n
      Speed    — time before first word appears (what we improve)\n
      Memory   — RAM used while model is loaded (what we reduce)\n
      Honesty  — generation speed (GPU-bound, we don't touch this)\n

    Add --with-noswap to see how autotune prevents your Mac from\n
    swapping under memory pressure.\n

    All numbers from Ollama's own internal timers. Nothing estimated.
    """
    import asyncio as _asyncio
    import argparse as _argparse
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent.parent / "scripts"))
    from proof import main as _proof_main

    ns = _argparse.Namespace(
        model=model,
        runs=runs,
        cold_runs=cold_runs,
        output=output,
        with_cold=with_cold,
        with_noswap=with_noswap,
        list_models=list_models,
    )
    _asyncio.run(_proof_main(ns))


# ---------------------------------------------------------------------------
# `autotune proof-suite`
# ---------------------------------------------------------------------------

@cli.command("proof-suite")
@click.option(
    "--models", "-m", multiple=True, metavar="MODEL",
    help=(
        "Ollama model IDs to benchmark.  Repeat for multiple: "
        "-m llama3.2:3b -m qwen3:8b.  "
        "Defaults to llama3.2:3b gemma4:e2b qwen3:8b."
    ),
)
@click.option("--runs", "-n", type=int, default=3, show_default=True,
              help="Inference runs per condition per prompt.  Min 3 for statistics.")
@click.option(
    "--profile", "-p",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced", show_default=True,
    help="autotune profile to compare against raw Ollama defaults.",
)
@click.option("--output", "-o", default=None, metavar="PATH",
              help="Save full results to a JSON file.")
@click.option("--list-models", is_flag=True,
              help="List locally installed Ollama models and exit.")
def proof_suite(
    models: tuple,
    runs: int,
    profile: str,
    output: Optional[str],
    list_models: bool,
) -> None:
    """Multi-model scientific benchmark: raw Ollama vs autotune.

    Runs a curated 5-prompt suite (factual, code, analysis, conversation,
    long output) through both conditions and reports Ollama-internal timing,
    Ollama-process-isolated RAM, and statistical significance (Wilcoxon
    signed-rank + Cohen's d effect size + 95% CI).

    \b
    Examples:
      autotune proof-suite
      autotune proof-suite -m llama3.2:3b -m gemma4:e2b -m qwen3:8b
      autotune proof-suite -m qwen3:8b --runs 5 --output results.json
    """
    import argparse as _argparse
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).parent.parent / "scripts"))
    from proof_suite import main as _suite_main  # type: ignore

    # Patch sys.argv so proof_suite's argparse picks up our values
    _argv = ["proof_suite"]
    if models:
        _argv += ["--models"] + list(models)
    _argv += ["--runs", str(runs), "--profile", profile]
    if output:
        _argv += ["--output", output]
    if list_models:
        _argv += ["--list-models"]

    _sys.argv = _argv
    _suite_main()


# ---------------------------------------------------------------------------
# `autotune agent-bench`  — Agentic multi-turn benchmark
# ---------------------------------------------------------------------------

@cli.command("agent-bench")
@click.option(
    "--models", "-m", multiple=True, metavar="MODEL",
    help=(
        "Ollama model IDs to benchmark.  Repeat for multiple: "
        "-m llama3.2:3b -m qwen3:8b.  "
        "Defaults to llama3.2:3b gemma4:e2b qwen3:8b."
    ),
)
@click.option("--trials", "-n", type=int, default=5, show_default=True,
              help="Trials per condition per task.  Min 3 recommended for statistics.")
@click.option(
    "--tasks", "-t", default="", metavar="TASK_IDS",
    help=(
        "Comma-separated task IDs to run.  Default: all five tasks.  "
        "Options: code_debugger,research_synth,step_planner,"
        "adversarial_context,extended_session"
    ),
)
@click.option(
    "--profile", "-p",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced", show_default=True,
    help="autotune profile to benchmark against raw Ollama defaults.",
)
@click.option("--quick", "-q", is_flag=True,
              help="Quick mode: 3 tasks, 2 trials (~20-30 min).")
@click.option("--output", "-o", default=None, metavar="PATH",
              help="Save full results JSON to this path.")
def agent_bench(
    models: tuple,
    trials: int,
    tasks: str,
    profile: str,
    quick: bool,
    output: Optional[str],
) -> None:
    """Agentic multi-turn benchmark: raw Ollama vs autotune.

    Runs 5 realistic agentic tasks (code debugging, research synthesis,
    step planning, adversarial context, extended session) through both
    raw Ollama defaults and autotune.  Measures per-turn TTFT, RAM,
    KV cache size, tool-call reliability, and task success rate.

    The TTFT growth curves reveal the core story: in raw Ollama, TTFT grows
    linearly with context because the full 4096-token KV cache is filled on
    every prefill step.  autotune's dynamic num_ctx keeps TTFT flat by
    sizing the context window to actual usage.

    \b
    Examples:
      autotune agent-bench
      autotune agent-bench -m llama3.2:3b -m qwen3:8b -n 3
      autotune agent-bench --tasks code_debugger,extended_session --quick
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).parent.parent / "scripts"))
    import agent_bench as _ab  # type: ignore

    import argparse as _argparse

    # Build a Namespace that matches agent_bench's argparse schema
    ns = _argparse.Namespace(
        models=list(models) or None,
        trials=trials,
        tasks=tasks,
        profile=profile,
        quick=quick,
        output=output,
    )

    import asyncio as _asyncio
    rc = _asyncio.run(_ab._async_main(ns))
    raise SystemExit(rc)


# ---------------------------------------------------------------------------
# `autotune webui`  — Open WebUI chat management
# ---------------------------------------------------------------------------

@cli.group("webui")
def webui_group() -> None:
    """Install, launch, and browse Open WebUI — wired to autotune.

    \b
    HOW AUTOTUNE INTEGRATES WITH OPEN WEBUI
    ─────────────────────────────────────────
    autotune runs its own OpenAI-compatible API server on port 8765.
    Every chat request from Open WebUI flows through autotune first,
    where context sizing, KV-cache tuning, and hardware optimization
    happen automatically — then autotune forwards to Ollama/MLX.

    Without this wiring, Open WebUI talks directly to Ollama and all
    autotune optimizations are bypassed.

    \b
    FIRST-TIME SETUP  (recommended — starts everything together):
      autotune webui launch    ← starts autotune server + Open WebUI
                                 (Open WebUI is pre-configured to route
                                  through autotune automatically)
      autotune webui login     ← sign in and save your API key
      autotune webui open      ← open in browser

    \b
    CONNECT AN ALREADY-RUNNING OPEN WEBUI:
      autotune serve           ← start autotune server (separate terminal)
      autotune webui login     ← sign in to Open WebUI
      autotune webui connect   ← configure Open WebUI to use autotune

    \b
    AFTER SETUP:
      autotune webui status    ← shows whether autotune is connected
      autotune webui models    ← list models (tagged autotune vs direct)
      autotune webui chats     ← list your chat history
    """


def _webui_headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _webui_get(url: str, path: str, key: str, params: dict | None = None) -> dict | list:
    """GET from Open WebUI. Raises SystemExit on auth/connection errors."""
    import httpx
    try:
        r = httpx.get(
            url.rstrip("/") + path,
            headers=_webui_headers(key),
            params=params or {},
            timeout=8.0,
            follow_redirects=True,
        )
    except httpx.ConnectError:
        console.print(
            f"[red]Cannot connect to Open WebUI at {url}[/red]\n"
            "[dim]Is it running?  Default: http://localhost:3000[/dim]"
        )
        raise SystemExit(1)
    except httpx.TimeoutException:
        console.print(f"[red]Request timed out connecting to {url}[/red]")
        raise SystemExit(1)

    if r.status_code == 401:
        console.print(
            "[red]Authentication failed.[/red]  "
            "Get a key from Open WebUI → Settings → Account → API Keys\n"
            "[dim]Pass it with --key or set OPEN_WEBUI_API_KEY[/dim]"
        )
        raise SystemExit(1)
    if r.status_code != 200:
        console.print(f"[red]Open WebUI returned HTTP {r.status_code}:[/red] {r.text[:200]}")
        raise SystemExit(1)
    return r.json()


def _resolve_key(key: Optional[str]) -> str:
    import os
    resolved = key or os.environ.get("OPEN_WEBUI_API_KEY", "")
    if not resolved:
        console.print(
            "[red]No API key.[/red]  "
            "Set [cyan]OPEN_WEBUI_API_KEY=sk-...[/cyan] or pass [cyan]--key sk-...[/cyan]\n"
            "[dim]Get your key: Open WebUI → Settings → Account → API Keys[/dim]"
        )
        raise SystemExit(1)
    return resolved


def _fmt_time(ts: int | float | None) -> str:
    """Format a Unix timestamp to a human-readable relative string."""
    import datetime
    if not ts:
        return "—"
    dt = datetime.datetime.fromtimestamp(float(ts))
    now = datetime.datetime.now()
    delta = now - dt
    if delta.days == 0:
        h = delta.seconds // 3600
        m = (delta.seconds % 3600) // 60
        if h == 0:
            return f"{m}m ago" if m > 0 else "just now"
        return f"{h}h ago"
    if delta.days == 1:
        return "yesterday"
    if delta.days < 7:
        return f"{delta.days}d ago"
    if delta.days < 30:
        return f"{delta.days // 7}w ago"
    return dt.strftime("%b %d")


@webui_group.command("chats")
@click.option("--url", default=None, envvar="OPEN_WEBUI_URL",
              metavar="URL", help="Open WebUI base URL (auto-detected if omitted).")
@click.option("--key", default=None, envvar="OPEN_WEBUI_API_KEY",
              metavar="SK", help="API key (or set OPEN_WEBUI_API_KEY).")
@click.option("--limit", "-n", default=30, show_default=True,
              help="Max chats to show.")
@click.option("--model", default=None, metavar="NAME",
              help="Filter to chats that used a specific model.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit raw JSON instead of a table.")
@click.option("--open", "open_id", default=None, metavar="ID",
              help="Open a specific chat in your browser by ID.")
def webui_chats(
    url: str,
    key: Optional[str],
    limit: int,
    model: Optional[str],
    as_json: bool,
    open_id: Optional[str],
) -> None:
    """List your Open WebUI chat history.

    Shows chat title, model used, last-updated time, and chat ID.
    Paginates automatically to respect --limit.

    \b
    Examples:
      autotune webui chats
      autotune webui chats --limit 50
      autotune webui chats --model qwen3:8b
      autotune webui chats --open <chat-id>
      autotune webui chats --json | jq '.[].title'
    """
    import json as _json
    import webbrowser
    import httpx as _httpx
    from rich.table import Table
    from rich import box

    resolved_key = _resolve_key(key)

    # ── Auto-detect URL ──────────────────────────────────────────────────
    if not url:
        for candidate in ("http://localhost:8080", "http://localhost:3000"):
            try:
                r = _httpx.get(candidate, timeout=2.0, follow_redirects=True)
                if r.status_code < 500:
                    url = candidate
                    break
            except Exception:
                continue
    if not url:
        console.print(
            "[red]Open WebUI is not running.[/red]\n"
            "[dim]Start it:  autotune webui install[/dim]"
        )
        raise SystemExit(1)

    # ── If --open, launch browser and exit ──────────────────────────────
    if open_id:
        chat_url = f"{url.rstrip('/')}/c/{open_id}"
        console.print(f"[cyan]Opening[/cyan] {chat_url}")
        webbrowser.open(chat_url)
        return

    # ── Paginate chat list until we have `limit` items ──────────────────
    chats: list[dict] = []
    page = 1
    with console.status("[cyan]Fetching chats from Open WebUI…[/cyan]", spinner="dots"):
        while len(chats) < limit:
            batch = _webui_get(url, "/api/v1/chats/", resolved_key, {"page": page})
            if not isinstance(batch, list) or not batch:
                break
            chats.extend(batch)
            if len(batch) < 60:   # Open WebUI returns 60/page; fewer = last page
                break
            page += 1

    chats = chats[:limit]

    if not chats:
        console.print("[yellow]No chats found.[/yellow]  Start a conversation in Open WebUI first.")
        return

    # ── For each chat, fetch the model used (from the chat detail) ──────
    # The list endpoint returns: id, title, updated_at, created_at
    # The model is in the full chat object under chat.models[] or chat.model
    # We fetch details for up to 40 chats; beyond that skip model column.
    FETCH_MODEL_LIMIT = 40
    fetch_models = len(chats) <= FETCH_MODEL_LIMIT

    def _get_model_for_chat(chat_id: str) -> str:
        try:
            detail = _webui_get(url, f"/api/v1/chats/{chat_id}", resolved_key)
            if isinstance(detail, dict):
                chat_body = detail.get("chat") or {}
                # Open WebUI stores model in `models` list or `model` string
                models_list = chat_body.get("models") or []
                if models_list:
                    return ", ".join(str(m) for m in models_list)
                single = chat_body.get("model") or ""
                if single:
                    return str(single)
                # Fallback: look at first message that has a model field
                for msg in chat_body.get("messages") or []:
                    if isinstance(msg, dict) and msg.get("model"):
                        return str(msg["model"])
        except Exception:
            pass
        return "—"

    if fetch_models:
        with console.status("[cyan]Fetching model info for each chat…[/cyan]", spinner="dots"):
            for chat in chats:
                chat["_model"] = _get_model_for_chat(chat.get("id", ""))
    else:
        for chat in chats:
            chat["_model"] = "—"

    # ── Filter by model if requested ────────────────────────────────────
    if model:
        chats = [c for c in chats if model.lower() in c.get("_model", "").lower()]
        if not chats:
            console.print(f"[yellow]No chats found using model '{model}'.[/yellow]")
            return

    # ── JSON output ──────────────────────────────────────────────────────
    if as_json:
        output = [
            {
                "id":         c.get("id"),
                "title":      c.get("title"),
                "model":      c.get("_model"),
                "updated_at": c.get("updated_at"),
                "created_at": c.get("created_at"),
            }
            for c in chats
        ]
        console.print(_json.dumps(output, indent=2))
        return

    # ── Rich table ───────────────────────────────────────────────────────
    console.print()
    model_filter_note = f"  [dim]model: {model}[/dim]" if model else ""
    console.print(
        f"[bold]Open WebUI chats[/bold]  "
        f"[dim]{url}[/dim]  "
        f"[dim]{len(chats)} shown[/dim]"
        f"{model_filter_note}"
    )
    console.print()

    t = Table(box=box.SIMPLE_HEAD, show_lines=False, expand=False)
    t.add_column("#",        width=4,  justify="right",  style="dim")
    t.add_column("Title",    min_width=28, max_width=52, no_wrap=True)
    t.add_column("Model",    min_width=12, max_width=22, style="cyan", no_wrap=True)
    t.add_column("Updated",  width=11, justify="right",  style="dim")
    t.add_column("ID",       width=10, style="dim",      no_wrap=True)

    for i, chat in enumerate(chats, 1):
        chat_id    = chat.get("id", "")
        title      = chat.get("title") or "[dim](untitled)[/dim]"
        chat_model = chat.get("_model") or "—"
        updated    = _fmt_time(chat.get("updated_at"))
        short_id   = chat_id[:8] + "…" if len(chat_id) > 8 else chat_id

        t.add_row(str(i), title, chat_model, updated, short_id)

    console.print(t)
    console.print(
        "[dim]Open a chat in browser:  "
        "[cyan]autotune webui chats --open <ID>[/cyan]  "
        "(copy the full ID from above)[/dim]\n"
    )


@webui_group.command("models")
@click.option("--url", default=None, envvar="OPEN_WEBUI_URL",
              metavar="URL", help="Open WebUI base URL (auto-detected if omitted).")
@click.option("--key", default=None, envvar="OPEN_WEBUI_API_KEY",
              metavar="SK", help="API key (or set OPEN_WEBUI_API_KEY).")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit raw JSON.")
def webui_models(url: str, key: Optional[str], as_json: bool) -> None:
    """List models in Open WebUI, tagged by whether they route through autotune.

    Models labelled [autotune] are routed through autotune's hardware-
    optimization middleware (context sizing, KV-cache tuning, profiling).
    Models labelled [ollama-direct] bypass autotune entirely.

    \b
    Examples:
      autotune webui models
      autotune webui models --json
    """
    import json as _json
    import httpx as _httpx
    from rich.table import Table
    from rich import box

    resolved_key = _resolve_key(key)

    if not url:
        for candidate in ("http://localhost:8080", "http://localhost:3000"):
            try:
                r = _httpx.get(candidate, timeout=2.0, follow_redirects=True)
                if r.status_code < 500:
                    url = candidate
                    break
            except Exception:
                continue
    if not url:
        console.print("[red]Open WebUI is not running.[/red]")
        raise SystemExit(1)

    with console.status("[cyan]Fetching models from Open WebUI…[/cyan]", spinner="dots"):
        data = _webui_get(url, "/api/models", resolved_key)

    # /api/models returns {"data": [...]} or a list directly
    model_list: list[dict] = []
    if isinstance(data, dict):
        model_list = data.get("data") or []
    elif isinstance(data, list):
        model_list = data

    if not model_list:
        console.print(
            "[yellow]No models found in Open WebUI.[/yellow]\n"
            "[dim]If autotune is not connected yet, run:  "
            "[cyan]autotune webui connect[/cyan][/dim]"
        )
        return

    if as_json:
        console.print(_json.dumps(model_list, indent=2))
        return

    # Open WebUI labels all models from OpenAI-compatible connections as
    # owned_by="openai", regardless of what the upstream server reports.
    # Models pulled directly from Ollama keep owned_by="ollama".
    # So: openai = routed through autotune, ollama = bypasses autotune.
    n_autotune = sum(1 for m in model_list if m.get("owned_by") == "openai")
    n_direct   = sum(1 for m in model_list if m.get("owned_by") == "ollama")
    n_other    = len(model_list) - n_autotune - n_direct

    console.print()
    console.print(
        f"[bold]Open WebUI models[/bold]  [dim]{url}[/dim]  "
        f"[dim]{len(model_list)} total[/dim]  "
        + (f"[green]{n_autotune} via autotune[/green]  " if n_autotune else "[yellow]0 via autotune[/yellow]  ")
        + (f"[dim]{n_direct} direct-Ollama[/dim]" if n_direct else "")
    )
    console.print()

    t = Table(box=box.SIMPLE_HEAD, show_lines=False)
    t.add_column("Route",     width=18)
    t.add_column("Model ID",  style="cyan", min_width=22)
    t.add_column("Name",      min_width=16)
    t.add_column("Context",   width=10, justify="right")

    for m in model_list:
        model_id  = m.get("id") or m.get("name") or "—"
        name      = m.get("name") or model_id
        owned_by  = m.get("owned_by") or m.get("provider") or ""
        ctx       = m.get("context_length") or ""
        ctx_str   = f"{ctx:,}" if isinstance(ctx, int) and ctx else "—"

        # De-dup name == id
        if name == model_id:
            name = "—"

        # "openai" = came through autotune's OpenAI-compat endpoint
        # "ollama" = came through Open WebUI's direct Ollama integration
        if owned_by == "openai":
            route_str = "[green]● autotune[/green]"
        elif owned_by in ("ollama",):
            route_str = "[yellow]○ ollama-direct[/yellow]"
        else:
            route_str = f"[dim]{owned_by or '—'}[/dim]"

        t.add_row(route_str, model_id, name, ctx_str)

    console.print(t)

    if n_autotune == 0:
        console.print(
            "[yellow]No autotune models visible.[/yellow]  "
            "Is the autotune server running?  "
            "Run [cyan]autotune webui connect[/cyan] to wire it in.\n"
        )
    elif n_direct > 0:
        console.print(
            "[dim]● autotune      = routed via autotune (context sizing, KV-cache tuning, profiling)\n"
            "○ ollama-direct = bypasses autotune, goes straight to Ollama\n\n"
            "Tip: disable direct-Ollama in Open WebUI to remove the duplicates:\n"
            "  Admin Panel → Settings → Connections → Ollama → disable[/dim]\n"
        )
    else:
        console.print(
            "[dim]All models are routed through autotune — no duplicates.[/dim]\n"
        )


@webui_group.command("open")
@click.option("--url", default=None, envvar="OPEN_WEBUI_URL",
              metavar="URL", help="Open WebUI base URL (auto-detected if omitted).")
@click.option("--chat", default=None, metavar="ID",
              help="Open a specific chat by ID instead of the home page.")
def webui_open(url: Optional[str], chat: Optional[str]) -> None:
    """Open Open WebUI in your browser.

    Auto-detects whether Open WebUI is running on port 8080 (pip install)
    or port 3000 (Docker).

    \b
    Examples:
      autotune webui open
      autotune webui open --chat <chat-id>
      autotune webui open --url http://myserver:3000
    """
    import httpx
    import webbrowser

    # ── Auto-detect port if URL not given ────────────────────────────────
    if not url:
        for candidate in ("http://localhost:8080", "http://localhost:3000"):
            try:
                r = httpx.get(candidate, timeout=2.0, follow_redirects=True)
                if r.status_code < 500:
                    url = candidate
                    break
            except Exception:
                continue

    if not url:
        console.print(
            "[red]Open WebUI is not running.[/red]\n\n"
            "Start it:  [cyan]autotune webui install[/cyan]\n"
            "or:        [cyan]open-webui serve[/cyan]"
        )
        raise SystemExit(1)

    target = f"{url.rstrip('/')}/c/{chat}" if chat else url
    console.print(f"[cyan]Opening[/cyan] {target}")
    webbrowser.open(target)


@webui_group.command("login")
@click.option("--url", default=None, envvar="OPEN_WEBUI_URL",
              metavar="URL", help="Open WebUI base URL (auto-detected if omitted).")
@click.option("--email", default=None, prompt="Email", help="Your Open WebUI account email.")
@click.option("--password", default=None,
              help="Password (prompted securely if omitted).")
@click.option("--save/--no-save", default=True,
              help="Save the API key to ~/.config/autotune/webui.env (default: save).")
@click.option("--new-key", is_flag=True, default=False,
              help="Generate a fresh API key even if one already exists.")
def webui_login(
    url: Optional[str],
    email: str,
    password: Optional[str],
    save: bool,
    new_key: bool,
) -> None:
    """Sign in to Open WebUI and retrieve your API key.

    Authenticates with your email + password, then fetches (or generates) your
    API key. Optionally saves it to ~/.config/autotune/webui.env so you never
    need to pass --key again.

    \b
    Examples:
      autotune webui login
      autotune webui login --email me@example.com
      autotune webui login --no-save          # print key only
      autotune webui login --new-key          # rotate to a fresh API key
    """
    import httpx
    from pathlib import Path as _Path
    from rich.panel import Panel

    # ── Prompt for password securely ─────────────────────────────────────
    if not password:
        import click as _click
        password = _click.prompt("Password", hide_input=True)

    # ── Auto-detect URL ──────────────────────────────────────────────────
    if not url:
        for candidate in ("http://localhost:8080", "http://localhost:3000"):
            try:
                r = httpx.get(candidate, timeout=2.0, follow_redirects=True)
                if r.status_code < 500:
                    url = candidate
                    break
            except Exception:
                continue
    if not url:
        console.print(
            "[red]Open WebUI is not running.[/red]\n"
            "Start it with: [cyan]autotune webui install[/cyan]"
        )
        raise SystemExit(1)

    base = url.rstrip("/")

    # ── Step 1: sign in → JWT ────────────────────────────────────────────
    with console.status("[cyan]Signing in…[/cyan]", spinner="dots"):
        try:
            resp = httpx.post(
                f"{base}/api/v1/auths/signin",
                json={"email": email, "password": password},
                timeout=8.0,
            )
        except httpx.ConnectError:
            console.print(f"[red]Cannot connect to {base}[/red]")
            raise SystemExit(1)

    if resp.status_code == 401 or resp.status_code == 400:
        console.print("[red]Login failed.[/red]  Check your email and password.")
        raise SystemExit(1)
    if resp.status_code != 200:
        console.print(f"[red]Unexpected response {resp.status_code}:[/red] {resp.text[:200]}")
        raise SystemExit(1)

    data     = resp.json()
    jwt      = data.get("token", "")
    username = data.get("name") or data.get("email") or email
    role     = data.get("role", "user")
    if not jwt:
        console.print("[red]No token in response. Open WebUI may have changed its API.[/red]")
        raise SystemExit(1)

    console.print(f"[green]✓[/green]  Signed in as [bold]{username}[/bold]  [dim]({role})[/dim]")

    auth_headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}

    # ── Step 2: ensure API key auth is enabled (admin only, silent) ──────
    if role == "admin":
        try:
            cfg_r = httpx.get(f"{base}/api/v1/auths/admin/config",
                               headers=auth_headers, timeout=8.0)
            if cfg_r.status_code == 200:
                cfg = cfg_r.json()
                if not cfg.get("ENABLE_API_KEYS", True):
                    cfg["ENABLE_API_KEYS"] = True
                    httpx.post(f"{base}/api/v1/auths/admin/config",
                               json=cfg, headers=auth_headers, timeout=8.0)
                    console.print("[green]✓[/green]  API key authentication enabled")
        except Exception:
            pass   # non-fatal — try fetching the key anyway

    # ── Step 3: fetch or generate API key ────────────────────────────────
    with console.status("[cyan]Fetching API key…[/cyan]", spinner="dots"):
        try:
            if new_key:
                kr = httpx.post(f"{base}/api/v1/auths/api_key",
                                headers=auth_headers, timeout=8.0)
            else:
                kr = httpx.get(f"{base}/api/v1/auths/api_key",
                               headers=auth_headers, timeout=8.0)

            # If not found or empty, generate a fresh one
            if kr.status_code == 404 or (kr.status_code == 200 and not kr.json().get("api_key")):
                kr = httpx.post(f"{base}/api/v1/auths/api_key",
                                headers=auth_headers, timeout=8.0)
        except Exception as exc:
            console.print(f"[red]Failed to fetch API key:[/red] {exc}")
            raise SystemExit(1)

    if kr.status_code != 200:
        console.print(
            f"[red]Could not retrieve API key (HTTP {kr.status_code}).[/red]\n"
            "[dim]If this is a new install, make sure you created an account first.[/dim]"
        )
        raise SystemExit(1)

    api_key = kr.json().get("api_key", "")
    if not api_key:
        console.print("[yellow]No API key returned.[/yellow]  Try --new-key to generate one.")
        raise SystemExit(1)

    console.print(f"[green]✓[/green]  API key retrieved")

    # ── Step 3: save to file ─────────────────────────────────────────────
    env_path = _Path.home() / ".config" / "autotune" / "webui.env"
    if save:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(
            f"# Open WebUI — written by autotune webui login\n"
            f"export OPEN_WEBUI_URL={base}\n"
            f"export OPEN_WEBUI_API_KEY={api_key}\n"
        )
        console.print(f"[green]✓[/green]  Saved to [dim]{env_path}[/dim]")

    # ── Print summary panel ──────────────────────────────────────────────
    lines = [
        f"[bold green]✓  Authenticated as {username}[/bold green]",
        "",
        f"  [dim]URL:[/dim]  {base}",
        f"  [dim]Key:[/dim]  {api_key}",
        "",
    ]
    if save:
        lines += [
            "  [bold]Activate in your shell (add to ~/.zshrc or ~/.bash_profile):[/bold]",
            f"    [cyan]source {env_path}[/cyan]",
            "",
            "  Or for this terminal session only:",
            f"    [cyan]source {env_path}[/cyan]",
        ]
    else:
        lines += [
            "  [bold]Activate for this session:[/bold]",
            f"    [cyan]export OPEN_WEBUI_API_KEY={api_key}[/cyan]",
            f"    [cyan]export OPEN_WEBUI_URL={base}[/cyan]",
        ]
    lines += [
        "",
        "  [bold]Next — wire autotune as the chat backend:[/bold]",
        "    [cyan]autotune webui connect[/cyan]",
        "",
        "  [dim]Then:[/dim]",
        "  [dim]  autotune webui open     ← open in browser[/dim]",
        "  [dim]  autotune webui models   ← see which models route through autotune[/dim]",
        "  [dim]  autotune webui status   ← full integration status[/dim]",
    ]
    console.print(Panel("\n".join(lines), title="[bold]Open WebUI authenticated[/bold]",
                        border_style="green"))


@webui_group.command("install")
@click.option("--start/--no-start", default=True,
              help="Start Open WebUI after installing (default: yes).")
@click.option("--port", default=8080, show_default=True, type=int,
              help="Port to bind Open WebUI on.")
@click.option("--autotune-port", default=8765, show_default=True, type=int,
              help="Port where autotune server is (or will be) running.")
@click.option("--with-ollama", is_flag=True, default=False,
              help="Also enable direct Ollama access in Open WebUI (shows duplicate models).")
def webui_install(start: bool, port: int, autotune_port: int, with_ollama: bool) -> None:
    """Install Open WebUI and start it wired to the autotune server.

    Open WebUI is launched with autotune's OpenAI-compatible API
    (http://localhost:AUTOTUNE_PORT/v1) as its backend, so every chat
    request is routed through autotune's hardware optimizations before
    reaching Ollama.  Direct Ollama access is disabled by default to
    prevent duplicate model entries.

    \b
    IMPORTANT — autotune server must be running first:
      autotune serve           ← start in a separate terminal

    Or use the one-command shortcut that starts both together:
      autotune webui launch

    \b
    What this does:
      1. Checks if Open WebUI is already running
      2. Checks if open-webui pip package is installed
      3. Installs it (pip install open-webui) if missing
      4. Starts Open WebUI with autotune pre-configured as the API backend
         (OPENAI_API_BASE_URL=http://localhost:AUTOTUNE_PORT/v1)
      5. Direct Ollama API disabled (pass --with-ollama to keep it)

    \b
    After Open WebUI starts, run:
      autotune webui login     ← sign in + save API key
      autotune webui connect   ← verify autotune connection is active
      autotune webui open      ← open in browser
    """
    import os as _os
    import httpx
    import shutil
    import subprocess as _sp
    from rich.panel import Panel

    autotune_url = f"http://localhost:{autotune_port}/v1"
    webui_url    = f"http://localhost:{port}"
    url_3000     = "http://localhost:3000"

    # ── Check if already running ─────────────────────────────────────────
    def _is_running(u: str) -> bool:
        try:
            r = httpx.get(u, timeout=2.0, follow_redirects=True)
            return r.status_code < 500
        except Exception:
            return False

    for check_url in (webui_url, url_3000):
        if _is_running(check_url):
            console.print(
                f"[green]✓[/green]  Open WebUI is already running at "
                f"[cyan]{check_url}[/cyan]"
            )
            console.print(
                "[dim]Run  [cyan]autotune webui login[/cyan]   to authenticate\n"
                "     [cyan]autotune webui connect[/cyan] to wire autotune as the backend\n"
                "     [cyan]autotune webui open[/cyan]    to open in browser[/dim]"
            )
            return

    # ── Check if autotune server is running ──────────────────────────────
    autotune_running = _is_running(f"http://localhost:{autotune_port}/health")
    if not autotune_running:
        console.print(
            f"[yellow]⚠[/yellow]  autotune server is not running on port {autotune_port}.\n"
            f"[dim]  Open WebUI will start but cannot serve models until autotune is up.\n"
            f"  Start it now:  [cyan]autotune serve[/cyan]  (in a separate terminal)\n"
            f"  Or use:        [cyan]autotune webui launch[/cyan]  (starts both together)[/dim]\n"
        )
    else:
        console.print(
            f"[green]✓[/green]  autotune server running at "
            f"[cyan]http://localhost:{autotune_port}[/cyan]"
        )

    # ── Check if pip package is installed ────────────────────────────────
    installed = shutil.which("open-webui") is not None
    if not installed:
        try:
            import importlib.util
            installed = importlib.util.find_spec("open_webui") is not None
        except Exception:
            installed = False

    if not installed:
        console.print("[yellow]open-webui not found.[/yellow]  Installing via pip…\n")
        console.print("[dim]pip install open-webui[/dim]\n")
        try:
            _sp.run(
                [sys.executable, "-m", "pip", "install", "open-webui"],
                check=True,
            )
            console.print("\n[green]✓  open-webui installed.[/green]")
        except _sp.CalledProcessError:
            console.print(
                "\n[red]Installation failed.[/red]  "
                "Try manually:  [cyan]pip install open-webui[/cyan]"
            )
            raise SystemExit(1)
    else:
        console.print("[green]✓[/green]  open-webui is already installed")

    # ── Start ────────────────────────────────────────────────────────────
    if not start:
        console.print(
            f"\n[dim]Start with autotune wired in:\n"
            f"  OPENAI_API_BASE_URL={autotune_url} \\\n"
            f"  OPENAI_API_KEY=autotune \\\n"
            f"  ENABLE_OLLAMA_API={'true' if with_ollama else 'false'} \\\n"
            f"  open-webui serve --port {port}[/dim]"
        )
        return

    bin_path = shutil.which("open-webui")
    cmd = (
        [bin_path, "serve", "--port", str(port)]
        if bin_path else
        [sys.executable, "-m", "open_webui", "serve", "--port", str(port)]
    )

    # Inject autotune as the OpenAI-compatible backend.
    # ENABLE_OLLAMA_API=false prevents direct Ollama models appearing alongside
    # autotune models (which are already backed by Ollama under the hood).
    env = dict(_os.environ)
    env["OPENAI_API_BASE_URL"]  = autotune_url
    env["OPENAI_API_BASE_URLS"] = autotune_url   # Open WebUI ≥0.5 plural form
    env["OPENAI_API_KEY"]       = "autotune"
    env["OPENAI_API_KEYS"]      = "autotune"
    if not with_ollama:
        env["ENABLE_OLLAMA_API"] = "false"

    ollama_note = (
        "[dim]  Direct Ollama also enabled (--with-ollama)[/dim]\n"
        if with_ollama else
        "[dim]  Direct Ollama disabled — all models route through autotune[/dim]\n"
    )

    console.print(
        f"\n[bold green]Starting Open WebUI[/bold green] on port [cyan]{port}[/cyan]"
        f"  [dim](Ctrl+C to stop)[/dim]\n\n"
        f"[bold]autotune integration[/bold]\n"
        f"  API backend:  [cyan]{autotune_url}[/cyan]\n"
        f"  API key:      [dim]autotune[/dim]\n"
        f"{ollama_note}\n"
        f"[dim]Once ready, run in a new terminal:[/dim]\n"
        f"  [cyan]autotune webui login[/cyan]    ← authenticate\n"
        f"  [cyan]autotune webui connect[/cyan]  ← verify autotune is wired\n"
        f"  [cyan]autotune webui open[/cyan]     ← open in browser\n"
    )
    try:
        _sp.run(cmd, check=False, env=env)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


# ---------------------------------------------------------------------------
# `autotune webui launch`  — start autotune server + Open WebUI together
# ---------------------------------------------------------------------------

@webui_group.command("launch")
@click.option("--port", default=8080, show_default=True, type=int,
              help="Port for Open WebUI.")
@click.option("--autotune-port", default=8765, show_default=True, type=int,
              help="Port for the autotune API server.")
@click.option("--with-ollama", is_flag=True, default=False,
              help="Also enable direct Ollama in Open WebUI (shows duplicate model names).")
@click.option("--mlx", "enable_mlx", is_flag=True, default=False,
              help="Enable MLX backend in autotune (Apple Silicon, opt-in).")
def webui_launch(port: int, autotune_port: int, with_ollama: bool, enable_mlx: bool) -> None:
    """Start the autotune server AND Open WebUI together, fully wired.

    This is the recommended first-time setup command.  It:

    \b
      1. Starts autotune serve in the background (port AUTOTUNE_PORT)
      2. Waits until autotune's /health endpoint responds
      3. Starts Open WebUI in the foreground (port PORT), pre-configured
         to route ALL chat requests through autotune's API middleware
      4. Open WebUI models labelled "autotune" in the model picker are
         the ones with hardware optimization active

    \b
    After Open WebUI is ready:
      autotune webui login     ← sign in + save API key (new terminal)
      autotune webui open      ← open in browser (new terminal)

    Press Ctrl+C to stop both servers.
    """
    import os as _os
    import shutil
    import subprocess as _sp
    import time as _time
    import httpx
    from rich.panel import Panel

    autotune_url   = f"http://localhost:{autotune_port}/v1"
    autotune_health = f"http://localhost:{autotune_port}/health"
    webui_url       = f"http://localhost:{port}"

    # ── Check if autotune server already running ─────────────────────────
    def _http_ok(u: str, timeout: float = 2.0) -> bool:
        try:
            return httpx.get(u, timeout=timeout, follow_redirects=True).status_code < 500
        except Exception:
            return False

    autotune_proc = None
    if _http_ok(autotune_health):
        console.print(
            f"[green]✓[/green]  autotune server already running at "
            f"[cyan]http://localhost:{autotune_port}[/cyan]"
        )
    else:
        # ── Start autotune serve in background ───────────────────────────
        autotune_cmd = [sys.executable, "-m", "autotune", "serve",
                        "--port", str(autotune_port)]
        if not enable_mlx:
            env_at = dict(_os.environ)
            env_at["AUTOTUNE_DISABLE_MLX"] = "1"
        else:
            env_at = _os.environ.copy()

        console.print(
            f"[cyan]Starting autotune server[/cyan] on port "
            f"[cyan]{autotune_port}[/cyan]  [dim](background)[/dim]"
        )
        autotune_proc = _sp.Popen(
            autotune_cmd,
            env=env_at,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
        )

        # Wait up to 15s for autotune /health to respond
        deadline = _time.time() + 15
        with console.status(
            "[cyan]Waiting for autotune server to start…[/cyan]",
            spinner="dots",
        ):
            while _time.time() < deadline:
                if _http_ok(autotune_health, timeout=1.0):
                    break
                _time.sleep(0.5)
            else:
                console.print(
                    "[red]autotune server did not start in 15 s.[/red]\n"
                    "[dim]Check for errors with:  autotune serve[/dim]"
                )
                autotune_proc.terminate()
                raise SystemExit(1)

        console.print(
            f"[green]✓[/green]  autotune server ready at "
            f"[cyan]http://localhost:{autotune_port}[/cyan]"
        )

    # ── Check if Open WebUI already running ──────────────────────────────
    if _http_ok(webui_url):
        console.print(
            f"[green]✓[/green]  Open WebUI already running at [cyan]{webui_url}[/cyan]\n"
            f"[yellow]⚠[/yellow]  It may not be wired to autotune.\n"
            f"[dim]  Run  [cyan]autotune webui connect[/cyan]  to configure the autotune connection.[/dim]"
        )
        if autotune_proc:
            autotune_proc.terminate()
        return

    # ── Build Open WebUI command with autotune env vars ───────────────────
    bin_path = shutil.which("open-webui")
    if not bin_path:
        # Check if pip package exists at all
        try:
            import importlib.util
            has_pkg = importlib.util.find_spec("open_webui") is not None
        except Exception:
            has_pkg = False
        if not has_pkg:
            console.print(
                "[red]open-webui is not installed.[/red]\n"
                "[dim]  Install it first:  [cyan]autotune webui install --no-start[/cyan][/dim]"
            )
            if autotune_proc:
                autotune_proc.terminate()
            raise SystemExit(1)
        webui_cmd = [sys.executable, "-m", "open_webui", "serve", "--port", str(port)]
    else:
        webui_cmd = [bin_path, "serve", "--port", str(port)]

    webui_env = dict(_os.environ)
    webui_env["OPENAI_API_BASE_URL"]  = autotune_url
    webui_env["OPENAI_API_BASE_URLS"] = autotune_url   # Open WebUI ≥0.5 plural form
    webui_env["OPENAI_API_KEY"]       = "autotune"
    webui_env["OPENAI_API_KEYS"]      = "autotune"
    if not with_ollama:
        webui_env["ENABLE_OLLAMA_API"] = "false"

    ollama_note = (
        "  Direct Ollama:  enabled (--with-ollama)\n"
        if with_ollama else
        "  Direct Ollama:  disabled  ← all models route through autotune\n"
    )

    console.print(Panel(
        f"[bold green]autotune + Open WebUI ready[/bold green]\n\n"
        f"  autotune API:   [cyan]http://localhost:{autotune_port}/v1[/cyan]\n"
        f"  Open WebUI:     [cyan]{webui_url}[/cyan]  (starting…)\n"
        f"{ollama_note}\n"
        f"[dim]In a new terminal once Open WebUI finishes loading:[/dim]\n"
        f"  [cyan]autotune webui login[/cyan]   ← sign in and save API key\n"
        f"  [cyan]autotune webui open[/cyan]    ← open browser\n\n"
        f"[dim]Models shown as \"owned by autotune\" in Open WebUI's model\n"
        f"picker are routed through autotune's hardware optimization.[/dim]\n\n"
        f"[dim]Press Ctrl+C to stop both servers.[/dim]",
        title="[bold]autotune webui launch[/bold]",
        border_style="green",
    ))

    webui_proc = None
    try:
        webui_proc = _sp.Popen(webui_cmd, env=webui_env)
        webui_proc.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping…[/yellow]")
    finally:
        if webui_proc and webui_proc.poll() is None:
            webui_proc.terminate()
            try:
                webui_proc.wait(timeout=5)
            except Exception:
                webui_proc.kill()
        if autotune_proc and autotune_proc.poll() is None:
            autotune_proc.terminate()
            try:
                autotune_proc.wait(timeout=5)
            except Exception:
                autotune_proc.kill()
        console.print("[yellow]Both servers stopped.[/yellow]")


# ---------------------------------------------------------------------------
# `autotune webui connect`  — wire autotune into an already-running Open WebUI
# ---------------------------------------------------------------------------

@webui_group.command("connect")
@click.option("--url", default=None, envvar="OPEN_WEBUI_URL",
              metavar="URL", help="Open WebUI base URL (auto-detected if omitted).")
@click.option("--key", default=None, envvar="OPEN_WEBUI_API_KEY",
              metavar="SK",
              help="Open WebUI API key (from 'autotune webui login').")
@click.option("--autotune-url", default="http://localhost:8765/v1",
              show_default=True, metavar="URL",
              help="autotune server URL to register in Open WebUI.")
@click.option("--autotune-key", default="autotune", show_default=True,
              metavar="KEY",
              help="API key sent to autotune (any non-empty value works).")
def webui_connect(
    url: Optional[str],
    key: Optional[str],
    autotune_url: str,
    autotune_key: str,
) -> None:
    """Wire autotune into an already-running Open WebUI instance.

    Reads Open WebUI's current OpenAI-connection list via its admin API,
    adds (or updates) the autotune server entry, and writes it back.
    Existing connections (e.g. real OpenAI) are preserved untouched.

    After this runs, every model in Open WebUI whose owner shows as
    "openai" is routed through autotune's hardware-optimization layer.
    Models shown as "ollama" still bypass autotune (direct Ollama).

    \b
    Requirements:
      • autotune webui login   must have been run first (or pass --key)
      • autotune serve         must be running on port 8765

    \b
    Examples:
      autotune webui connect
      autotune webui connect --autotune-url http://localhost:8765/v1
      autotune webui connect --key sk-... --url http://myserver:8080
    """
    import httpx
    from rich.panel import Panel

    resolved_key = _resolve_key(key)

    # ── Auto-detect Open WebUI URL ────────────────────────────────────────
    if not url:
        for candidate in ("http://localhost:8080", "http://localhost:3000"):
            try:
                r = httpx.get(candidate, timeout=2.0, follow_redirects=True)
                if r.status_code < 500:
                    url = candidate
                    break
            except Exception:
                continue
    if not url:
        console.print(
            "[red]Open WebUI is not running.[/red]\n"
            "[dim]Start it:  autotune webui launch[/dim]"
        )
        raise SystemExit(1)

    base = url.rstrip("/")
    auth_headers = {"Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json"}

    # ── Check autotune server is reachable ───────────────────────────────
    # Derive health URL: strip trailing "/v1" then add "/health"
    at_base_url = autotune_url.rstrip("/")
    if at_base_url.endswith("/v1"):
        at_base_url = at_base_url[:-3]
    autotune_health = at_base_url + "/health"
    try:
        hr = httpx.get(autotune_health, timeout=3.0)
        if hr.status_code == 200:
            hdata = hr.json()
            ollama_ok = hdata.get("backends", {}).get("ollama", False)
            console.print(
                f"[green]✓[/green]  autotune server running at [cyan]{autotune_url}[/cyan]"
                + ("  [dim](Ollama: up)[/dim]" if ollama_ok else "  [yellow](Ollama: down)[/yellow]")
            )
        else:
            console.print(
                f"[yellow]⚠[/yellow]  autotune returned HTTP {hr.status_code} — continuing."
            )
    except Exception:
        console.print(
            f"[yellow]⚠[/yellow]  Cannot reach autotune at {autotune_url}.\n"
            f"[dim]  Start it:  [cyan]autotune serve[/cyan]  "
            f"(connection will activate once it starts)[/dim]"
        )

    # ── Read current OpenAI connection config from Open WebUI ─────────────
    # API: GET  /openai/config          → OpenAIConfigForm
    #      POST /openai/config/update   → updated OpenAIConfigForm
    #
    # OpenAIConfigForm schema (confirmed from Open WebUI's /openapi.json):
    #   {
    #     "ENABLE_OPENAI_API":   bool | null,
    #     "OPENAI_API_BASE_URLS": ["https://api.openai.com/v1", ...],
    #     "OPENAI_API_KEYS":     ["sk-real-key", "autotune", ...],
    #     "OPENAI_API_CONFIGS":  {"0": {...}, "1": {...}}  ← keyed by string index
    #   }
    #   Each OPENAI_API_CONFIGS entry:
    #   {
    #     "enable": true,
    #     "prefix_id": "",     ← optional model ID prefix shown in picker
    #     "model_ids": [],     ← empty = all models from this connection
    #     "tags": [],
    #     "connection_type": "external",
    #     "auth_type": "bearer"
    #   }

    configured = False

    with console.status("[cyan]Reading Open WebUI connection config…[/cyan]", spinner="dots"):
        try:
            cfg_r = httpx.get(f"{base}/openai/config",
                               headers=auth_headers, timeout=8.0)
        except httpx.ConnectError:
            console.print(f"[red]Cannot connect to Open WebUI at {base}[/red]")
            raise SystemExit(1)

    if cfg_r.status_code == 401:
        console.print(
            "[red]Authentication failed.[/red]  "
            "Run [cyan]autotune webui login[/cyan] to refresh your API key."
        )
        raise SystemExit(1)
    if cfg_r.status_code != 200:
        console.print(
            f"[red]GET /openai/config returned HTTP {cfg_r.status_code}[/red]\n"
            f"[dim]{cfg_r.text[:200]}[/dim]"
        )
        raise SystemExit(1)

    cfg = cfg_r.json()
    existing_urls    = list(cfg.get("OPENAI_API_BASE_URLS") or [])
    existing_keys    = list(cfg.get("OPENAI_API_KEYS")      or [])
    existing_configs = dict(cfg.get("OPENAI_API_CONFIGS")   or {})

    # ── Locate existing autotune entry (match by port or keyword) ────────
    at_url_norm = autotune_url.rstrip("/")
    at_indices  = [
        i for i, u in enumerate(existing_urls)
        if u.rstrip("/") == at_url_norm
        or "8765" in u
        or "autotune" in u.lower()
    ]

    if at_indices:
        idx = at_indices[0]
        existing_urls[idx]  = at_url_norm
        while len(existing_keys) <= idx:
            existing_keys.append("")
        existing_keys[idx] = autotune_key
        action = "updated"
    else:
        idx = len(existing_urls)
        existing_urls.append(at_url_norm)
        while len(existing_keys) < len(existing_urls):
            existing_keys.append("")
        existing_keys[idx] = autotune_key
        action = "added"

    # ── Build / update OPENAI_API_CONFIGS entry for this index ───────────
    # prefix_id="autotune" makes every model from this connection appear in
    # Open WebUI's picker as "autotune.<model_name>" (e.g. autotune.qwen3:8b).
    # This:
    #   1. Makes it unmistakably clear which models are autotune-routed
    #   2. Eliminates ID conflicts with direct-Ollama entries (same model,
    #      different picker entries = user can choose either path)
    # The autotune server strips the "autotune." prefix before routing to Ollama.
    prev = existing_configs.get(str(idx), {})
    existing_configs[str(idx)] = {
        "enable":          True,
        "prefix_id":       "autotune",      # ← this is the key labeling change
        "tags":            prev.get("tags",            []),
        "model_ids":       prev.get("model_ids",       []),
        "connection_type": prev.get("connection_type", "external"),
        "auth_type":       prev.get("auth_type",       "bearer"),
    }

    payload = {
        "ENABLE_OPENAI_API":    True,
        "OPENAI_API_BASE_URLS": existing_urls,
        "OPENAI_API_KEYS":      existing_keys,
        "OPENAI_API_CONFIGS":   existing_configs,
    }

    with console.status("[cyan]Writing autotune connection to Open WebUI…[/cyan]", spinner="dots"):
        try:
            post_r = httpx.post(f"{base}/openai/config/update",
                                 json=payload, headers=auth_headers, timeout=8.0)
        except Exception as exc:
            console.print(f"[red]POST /openai/config/update failed: {exc}[/red]")
            raise SystemExit(1)

    if post_r.status_code == 200:
        console.print(f"[green]✓[/green]  autotune connection {action} (index {idx})")
        configured = True
    else:
        console.print(
            f"[red]POST /openai/config/update returned HTTP {post_r.status_code}[/red]\n"
            f"[dim]{post_r.text[:300]}[/dim]"
        )

    # ── Verify: count openai-owned models (= autotune-routed) in Open WebUI
    if configured:
        try:
            models_r = httpx.get(f"{base}/api/models",
                                  headers=auth_headers, timeout=8.0)
            if models_r.status_code == 200:
                mdata  = models_r.json()
                mlist  = mdata.get("data", mdata) if isinstance(mdata, dict) else mdata
                # Open WebUI labels all models from OpenAI-compatible connections
                # as owned_by="openai", regardless of what the upstream server returns.
                n_at   = sum(1 for m in mlist
                             if isinstance(m, dict) and m.get("owned_by") == "openai")
                n_dir  = sum(1 for m in mlist
                             if isinstance(m, dict) and m.get("owned_by") == "ollama")
                if n_at > 0:
                    console.print(
                        f"[green]✓[/green]  [bold]{n_at}[/bold] model(s) visible via autotune  "
                        f"[dim]({n_dir} still direct-Ollama)[/dim]"
                    )
                else:
                    console.print(
                        "[yellow]⚠[/yellow]  No autotune models visible yet — "
                        "try refreshing your browser."
                    )
        except Exception:
            pass

    # ── Summary panel ─────────────────────────────────────────────────────
    if configured:
        n_total = len(existing_urls)
        console.print(Panel(
            f"[bold green]✓  autotune wired into Open WebUI[/bold green]\n\n"
            f"  Open WebUI:    [cyan]{base}[/cyan]\n"
            f"  autotune API:  [cyan]{autotune_url}[/cyan]  [dim](connection {idx + 1} of {n_total})[/dim]\n\n"
            f"[bold]How models appear in Open WebUI's picker:[/bold]\n"
            f"  [green]autotune.qwen3:8b[/green]   ← prefixed = routed through autotune\n"
            f"  [dim]qwen3:8b[/dim]              ← no prefix = direct Ollama (bypasses autotune)\n\n"
            f"  Refresh your browser — autotune-labelled models will appear.\n\n"
            f"  [dim]autotune webui models[/dim]   ← see routing labels in the terminal\n"
            f"  [dim]autotune webui status[/dim]   ← full integration check\n"
            f"  [dim]autotune webui open[/dim]     ← open browser",
            title="[bold]autotune connected[/bold]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[yellow]Automatic configuration failed.[/yellow]\n\n"
            "Add the connection manually in Open WebUI:\n\n"
            f"  1. Open [cyan]{base}[/cyan] in your browser\n"
            f"  2. Admin Panel → Settings → Connections\n"
            f"  3. Add a new OpenAI-compatible connection:\n"
            f"       URL:  [cyan]{autotune_url}[/cyan]\n"
            f"       Key:  [cyan]{autotune_key}[/cyan]\n"
            f"  4. Save and refresh the page",
            title="[bold]Manual setup[/bold]",
            border_style="yellow",
        ))


@webui_group.command("status")
@click.option("--url", default=None, envvar="OPEN_WEBUI_URL",
              metavar="URL", help="Open WebUI base URL (auto-detected if omitted).")
@click.option("--key", default=None, envvar="OPEN_WEBUI_API_KEY",
              metavar="SK", help="API key (or set OPEN_WEBUI_API_KEY).")
@click.option("--autotune-port", default=8765, show_default=True, type=int,
              help="Port to check for the autotune API server.")
def webui_status(url: Optional[str], key: Optional[str], autotune_port: int) -> None:
    """Show autotune + Open WebUI connectivity — the full integration picture.

    Checks three things:
      1. Is the autotune API server running? (required for routing)
      2. Is Open WebUI running?
      3. Is Open WebUI actually wired to autotune, or talking to raw Ollama?

    \b
    Example:
      autotune webui status
    """
    import os as _os
    import shutil
    import httpx
    from rich.panel import Panel

    lines: list[str] = []
    border_style = "green"

    def _http_ok(u: str, timeout: float = 2.0) -> bool:
        try:
            return httpx.get(u, timeout=timeout, follow_redirects=True).status_code < 500
        except Exception:
            return False

    # ── 1. autotune server ───────────────────────────────────────────────
    lines.append("[bold]autotune server[/bold]")
    autotune_health = f"http://localhost:{autotune_port}/health"
    if _http_ok(autotune_health):
        try:
            hdata = httpx.get(autotune_health, timeout=3.0).json()
            ollama_up  = hdata.get("backends", {}).get("ollama", False)
            version    = hdata.get("version", "")
            mem_pct    = hdata.get("memory", {}).get("ram_pct", 0)
            active_q   = hdata.get("queue", {}).get("active", 0)
            lines.append(
                f"  [green]✓[/green]  Running at [cyan]http://localhost:{autotune_port}/v1[/cyan]"
                + (f"  [dim]v{version}[/dim]" if version else "")
            )
            ollama_str = "[green]up[/green]" if ollama_up else "[yellow]down[/yellow]"
            lines.append(
                f"  [dim]Ollama backend: {ollama_str}  │  "
                f"RAM: {mem_pct:.0f}%  │  "
                f"Active requests: {active_q}[/dim]"
            )
        except Exception:
            lines.append(
                f"  [green]✓[/green]  Running at [cyan]http://localhost:{autotune_port}/v1[/cyan]"
            )
    else:
        border_style = "yellow"
        lines.append(f"  [red]✗[/red]  Not running on port {autotune_port}")
        lines.append(
            "  [dim]Start it:  [cyan]autotune serve[/cyan]"
            "  or  [cyan]autotune webui launch[/cyan][/dim]"
        )

    lines.append("")

    # ── 2. Open WebUI ────────────────────────────────────────────────────
    lines.append("[bold]Open WebUI[/bold]")
    detected_url: Optional[str] = url
    if not detected_url:
        for candidate in ("http://localhost:8080", "http://localhost:3000"):
            if _http_ok(candidate):
                detected_url = candidate
                break

    if not detected_url:
        border_style = "red"
        installed = shutil.which("open-webui") is not None
        lines.append("  [red]✗[/red]  Not running")
        if installed:
            lines.append("  [dim]Start:   [cyan]autotune webui launch[/cyan]  (starts both)[/dim]")
        else:
            lines.append("  [dim]Install: [cyan]autotune webui install --no-start[/cyan]  then  [cyan]autotune webui launch[/cyan][/dim]")
        console.print(Panel("\n".join(lines), title="[bold]autotune + Open WebUI status[/bold]",
                            border_style=border_style))
        return

    lines.append(f"  [green]✓[/green]  Running at [cyan]{detected_url}[/cyan]")

    # ── 3. autotune wired into Open WebUI? ───────────────────────────────
    # Detection strategy:
    #   a) GET /openai/config  — check OPENAI_API_BASE_URLS for the autotune URL
    #      This is the ground-truth check. Works regardless of model count.
    #   b) GET /api/models     — count owned_by="openai" (autotune-routed) vs "ollama" (direct)
    #      Note: Open WebUI overrides owned_by to "openai" for all OpenAI-compat connections,
    #      so we use that as the proxy for "going through autotune".
    lines.append("")
    lines.append("[bold]autotune integration[/bold]")

    resolved_key = key or _os.environ.get("OPEN_WEBUI_API_KEY", "")
    if not resolved_key:
        lines.append("  [yellow]⚠[/yellow]  No API key — cannot check connection status")
        lines.append(
            "  [dim]Run  [cyan]autotune webui login[/cyan]  to authenticate[/dim]"
        )
    else:
        try:
            auth_hdrs = {"Authorization": f"Bearer {resolved_key}",
                         "Content-Type": "application/json"}

            # ── a) Check OpenAI connection config ────────────────────────
            at_port     = str(autotune_port)
            at_url_frag = f"localhost:{autotune_port}"
            connection_found = False
            connection_idx   = -1
            n_connections    = 0

            cfg_r = httpx.get(f"{detected_url}/openai/config",
                               headers=auth_hdrs, timeout=8.0)
            if cfg_r.status_code == 200:
                cfg  = cfg_r.json()
                urls = cfg.get("OPENAI_API_BASE_URLS") or []
                n_connections = len(urls)
                for i, u in enumerate(urls):
                    if at_url_frag in u or "autotune" in u.lower():
                        connection_found = True
                        connection_idx   = i
                        break

                if connection_found:
                    lines.append(
                        f"  [green]✓[/green]  autotune connection present "
                        f"[dim](entry {connection_idx + 1} of {n_connections} OpenAI connections)[/dim]"
                    )
                    # Show the actual URL stored
                    lines.append(
                        f"  [dim]  → {urls[connection_idx]}[/dim]"
                    )
                else:
                    border_style = "yellow"
                    lines.append("  [yellow]✗[/yellow]  autotune NOT in Open WebUI's connection list")
                    lines.append(
                        "  [dim]  Fix:  [cyan]autotune webui connect[/cyan][/dim]"
                    )
            elif cfg_r.status_code == 401:
                border_style = "yellow"
                lines.append("  [red]✗[/red]  API key rejected")
                lines.append("  [dim]Re-authenticate:  [cyan]autotune webui login[/cyan][/dim]")
            else:
                lines.append(f"  [yellow]⚠[/yellow]  /openai/config returned HTTP {cfg_r.status_code}")

            # ── b) Model routing breakdown ───────────────────────────────
            models_r = httpx.get(f"{detected_url}/api/models",
                                   headers=auth_hdrs, timeout=8.0)
            if models_r.status_code == 200:
                mdata  = models_r.json()
                mlist  = mdata.get("data", mdata) if isinstance(mdata, dict) else mdata
                # "openai" owned_by = came from an OpenAI-compat connection (i.e. autotune)
                # "ollama" owned_by = came from direct Ollama API (bypasses autotune)
                n_at  = sum(1 for m in mlist if isinstance(m, dict)
                            and m.get("owned_by") == "openai")
                n_dir = sum(1 for m in mlist if isinstance(m, dict)
                            and m.get("owned_by") == "ollama")
                n_tot = len([m for m in mlist if isinstance(m, dict)])

                if n_at > 0:
                    lines.append(
                        f"  [green]✓[/green]  [bold]{n_at}[/bold] model(s) routed via autotune  "
                        f"[dim]({n_dir} direct-Ollama, {n_tot} total)[/dim]"
                    )
                elif connection_found:
                    lines.append(
                        "  [yellow]⚠[/yellow]  Connection configured but 0 models visible — "
                        "try refreshing your browser or check autotune serve is up."
                    )
                else:
                    lines.append(
                        f"  [dim]{n_dir} Ollama-direct models, 0 via autotune[/dim]"
                    )

                if n_dir > 0 and connection_found:
                    lines.append(
                        f"  [dim]  {n_dir} model(s) are direct-Ollama (bypass autotune).[/dim]\n"
                        f"  [dim]  These are the same models via a different path — both are available.[/dim]"
                    )

            # Chat count
            chats_r = httpx.get(f"{detected_url}/api/v1/chats/",
                                  headers=auth_hdrs, params={"page": 1}, timeout=8.0)
            if chats_r.status_code == 200:
                cdata = chats_r.json()
                n_chats = len(cdata) if isinstance(cdata, list) else 0
                suffix = "+" if n_chats == 60 else ""
                lines.append(f"  [dim]{n_chats}{suffix} chats in history[/dim]")

        except Exception as _exc:
            lines.append(f"  [yellow]⚠[/yellow]  Could not check integration: {_exc}")

    lines += [
        "",
        "[dim]"
        "  autotune webui launch   ← start both servers together\n"
        "  autotune webui connect  ← wire autotune into running Open WebUI\n"
        "  autotune webui login    ← authenticate\n"
        "  autotune webui open     ← open in browser\n"
        "  autotune webui models   ← list models with routing labels"
        "[/dim]",
    ]

    console.print(Panel(
        "\n".join(lines),
        title="[bold]autotune + Open WebUI status[/bold]",
        border_style=border_style,
    ))


# ---------------------------------------------------------------------------
# Entrypoint (for `python -m autotune`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
