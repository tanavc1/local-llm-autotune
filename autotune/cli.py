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
@click.version_option(package_name="autotune")
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
    from autotune.hardware.profiler import profile_hardware
    from autotune.config.generator import generate_recommendations, MODE_WEIGHTS
    from autotune.output.formatter import print_hardware_profile, print_recommendations
    from autotune.models.registry import MODEL_REGISTRY

    # ── Hardware profiling ──────────────────────────────────────────────
    console.rule("[bold blue]autotune[/bold blue]")
    with console.status("[cyan]Profiling hardware…[/cyan]", spinner="dots"):
        hw = profile_hardware()

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
    with console.status("[cyan]Scoring candidates…[/cyan]", spinner="dots"):
        recs = generate_recommendations(hw, modes=modes, top_n=top)

    if model_filter:
        import autotune.models.registry as _reg  # noqa: F811
        _reg.MODEL_REGISTRY = _orig  # type: ignore[assignment]

    if not recs:
        console.print(
            "[bold red]No configuration fits within the available memory budget.[/bold red]\n"
            "Try closing other applications to free RAM, or add more memory/VRAM."
        )
        sys.exit(1)

    print_recommendations(recs, modes=modes)


# ---------------------------------------------------------------------------
# `autotune hardware`
# ---------------------------------------------------------------------------

@cli.command()
def hardware() -> None:
    """Detect and display the hardware profile without generating recommendations."""
    from autotune.hardware.profiler import profile_hardware
    from autotune.output.formatter import print_hardware_profile

    with console.status("[cyan]Profiling hardware…[/cyan]", spinner="dots"):
        hw = profile_hardware()

    print_hardware_profile(hw)


# ---------------------------------------------------------------------------
# `autotune models`
# ---------------------------------------------------------------------------

@cli.command()
def models() -> None:
    """List all models in the registry with their specifications."""
    from autotune.output.formatter import print_model_table

    print_model_table()


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

    MODEL is an Ollama model tag, e.g. llama3.2, phi4-mini, qwen2.5:14b.
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

    try:
        pull_model(model, console)
        console.print(
            f"[dim]Start chatting:  [bold]autotune chat --model {model}[/bold][/dim]"
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
@click.option("--model", "-m", default="phi4-mini:latest", show_default=True,
              help="Model to benchmark.")
@click.option(
    "--profile", "-p",
    type=click.Choice(["fast", "balanced", "quality"]),
    default="balanced", show_default=True,
)
@click.option("--tag", default=None, metavar="NAME",
              help="Label this run (auto-generated if omitted).")
@click.option("--prompt-file", default=None, type=click.Path(exists=True),
              help="Use a custom prompt from a file instead of the built-in intensive prompt.")
@click.option("--no-hw-tuning", is_flag=True, default=False,
              help="Skip OS-level hardware optimizations (for comparison runs).")
@click.option("--raw", is_flag=True, default=False,
              help="Hit Ollama with ZERO autotune settings — pure Ollama defaults. "
                   "Use this to establish a true raw baseline for comparison.")
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
    save: bool,
    compare: Optional[str],
) -> None:
    """Run an intensive benchmark prompt and measure real hardware strain.

    \b
    Examples:
      # Run baseline
      autotune bench --model phi4-mini:latest --profile balanced --tag baseline

      # Run with fast profile (less RAM pressure)
      autotune bench --model phi4-mini:latest --profile fast --tag fast_optimized

      # Compare the two
      autotune bench --compare baseline,fast_optimized
    """
    import asyncio
    from autotune.bench.runner import run_bench, save_result

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

    # ── Benchmark run ─────────────────────────────────────────────────────
    mode_label = "raw_ollama" if raw else profile
    auto_tag = tag or f"{model.replace(':', '_').replace('/', '_')}_{mode_label}_{int(time.time())}"

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

    console.print()
    if raw:
        console.print(f"[bold]autotune bench[/bold]  [cyan]{model}[/cyan]  profile=[red]RAW OLLAMA (no autotune)[/red]  tag=[dim]{auto_tag}[/dim]")
        console.print(f"[dim]Ollama defaults: num_ctx=4096, temp=0.8, no HW tuning, no keep_alive override[/dim]")
    else:
        console.print(f"[bold]autotune bench[/bold]  [cyan]{model}[/cyan]  profile=[yellow]{profile}[/yellow]  tag=[dim]{auto_tag}[/dim]")
        console.print(f"[dim]Prompt: ~{prompt_tokens_est} tokens  │  HW tuning: {'off' if no_hw_tuning else 'on'}[/dim]")
    console.print()

    import psutil
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    console.print(
        f"[dim]System before:  RAM {vm.percent:.1f}%  "
        f"({vm.used/1024**3:.2f} GB used / {vm.total/1024**3:.1f} GB)  "
        f"swap {sw.used/1024**3:.2f} GB[/dim]"
    )
    console.print()

    with console.status("[bold cyan]Running inference…[/bold cyan]", spinner="dots"):
        if raw:
            from autotune.bench.runner import run_raw_ollama
            result = asyncio.run(run_raw_ollama(
                model_id=model,
                messages=messages,
                tag=auto_tag,
            ))
        else:
            result = asyncio.run(run_bench(
                model_id=model,
                messages=messages,
                profile_name=profile,
                tag=auto_tag,
                apply_hw_tuning=not no_hw_tuning,
            ))

    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
        raise SystemExit(1)

    # ── Results table ────────────────────────────────────────────────────
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
            "Pull one with: [cyan]ollama pull phi4-mini[/cyan]"
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
def run(model_name: str, profile: str, system: Optional[str], force: bool) -> None:
    """Launch an optimised chat session with a locally downloaded Ollama model.

    Runs a pre-flight memory analysis (weights + KV cache + runtime) to select
    the correct profile and safe context window before loading the model.

    \b
    Examples:
      autotune run phi4-mini:latest
      autotune run qwen2.5-coder:14b --profile balanced
      autotune run llama3.2 --system "You are a concise coding assistant"
    """
    import httpx
    from autotune.hardware.profiler import profile_hardware
    from autotune.api.chat import start_chat
    from autotune.api.model_selector import ModelSelector, FitClass

    hw           = profile_hardware()
    available_gb = hw.effective_memory_gb
    total_gb     = hw.memory.total_gb
    sel          = ModelSelector(available_gb=available_gb, total_ram_gb=total_gb)

    # ── Fetch model info from Ollama ─────────────────────────────────────
    size_gb:   float          = 0.0
    params_b:  Optional[float] = None
    quant_str: str            = "unknown"
    modelinfo: dict           = {}

    try:
        tags_resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        for m in tags_resp.json().get("models", []):
            if m.get("name", "").lower() == model_name.lower():
                size_gb = m.get("size", 0) / 1024**3
                break
    except Exception:
        console.print("[red]Ollama is not running.[/red]  Start with: ollama serve")
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
            f"[yellow]Model {model_name!r} not found in Ollama.[/yellow]  "
            f"Pull it first: [cyan]ollama pull {model_name}[/cyan]"
        )
        raise SystemExit(1)

    # ── Pre-flight fit analysis ──────────────────────────────────────────
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

    start_chat(model_id=model_name, profile=chosen, system_prompt=system)


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
def telemetry(model_id: Optional[str], limit: int, events: bool) -> None:
    """Show persisted performance telemetry for local LLM runs.

    Displays run history with structured metrics — TTFT, throughput, RAM/swap
    pressure, CPU load — all queryable for trend analysis.

    \b
    Examples:
      autotune telemetry
      autotune telemetry --model phi4-mini:latest
      autotune telemetry --events --model phi4-mini:latest
    """
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
def serve(host: str, port: int, reload: bool) -> None:
    """Start the autotune OpenAI-compatible API server.

    Any OpenAI client can use it via base_url=http://HOST:PORT/v1
    """
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install 'uvicorn[standard]'[/red]")
        raise SystemExit(1)

    console.print(
        f"[bold green]autotune API server[/bold green]  "
        f"[dim]http://{host}:{port}/v1[/dim]\n"
        f"  [cyan]/v1/chat/completions[/cyan]  [dim]streaming · OpenAI-compatible[/dim]\n"
        f"  [cyan]/v1/models[/cyan]             [dim]discover available models[/dim]\n"
        f"  [cyan]/health[/cyan]                [dim]backend status + hardware[/dim]\n"
        f"  [cyan]/api/conversations[/cyan]      [dim]persistent conversation CRUD[/dim]\n"
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
def chat(
    model: str,
    profile: str,
    system: Optional[str],
    conv_id: Optional[str],
    no_optimize: bool,
) -> None:
    """Start an optimized terminal chat session with any model.

    The chat connects directly to Ollama / LM Studio / HuggingFace Inference API
    (whichever is available) without needing `autotune serve` to be running.

    Real-time optimization runs by default: the session monitors RAM, thermals,
    and token throughput, and adjusts context size and KV precision automatically
    when pressure builds.  Use --no-optimize to disable this.

    \b
    Examples:
      autotune chat --model llama3.2
      autotune chat --model Qwen/Qwen2.5-7B-Instruct --profile fast
      autotune chat --model llama3.2 --system "You are a concise assistant"
      autotune chat --model phi4-mini --no-optimize
    """
    from autotune.api.chat import start_chat
    start_chat(
        model_id=model,
        profile=profile,
        system_prompt=system,
        conv_id=conv_id,
        optimize=not no_optimize,
    )


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
      autotune mlx pull phi4-mini     Pull the MLX version of phi4-mini
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

    MODEL can be an Ollama model name (e.g. phi4-mini, llama3.2:3b) or a
    full HuggingFace model ID (e.g. mlx-community/Phi-4-mini-instruct-4bit).

    \b
    Examples:
      autotune mlx pull phi4-mini
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
# Entrypoint (for `python -m autotune`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
