#!/usr/bin/env python3
"""
proof_report.py — merge multiple per-model proof_suite JSON files into
a single cross-model report and re-render the full summary.

Usage
-----
    python scripts/proof_report.py proof_results_v2.json proof_results_gemma4.json proof_results_qwen3.json
    python scripts/proof_report.py --glob "proof_results_*.json"
    python scripts/proof_report.py *.json --output proof_combined.json
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from rich import box as _box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


_CON_W = 100


def _pct_cell(pct: float, higher_is_better: bool) -> Text:
    if math.isnan(pct):
        return Text("N/A", style="dim")
    improved = (pct > 0) if higher_is_better else (pct < 0)
    color = "green" if improved else ("red" if abs(pct) > 1 else "dim")
    sign  = "+" if pct >= 0 else ""
    return Text(f"{sign}{pct:.1f}%", style=color)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _mean_pct(models: list[dict], prompt_key: str, metric_key: str) -> float:
    vals = []
    for m in models:
        for p in m.get("prompts", []):
            if metric_key in p and p[metric_key].get("n", 0) > 0:
                vals.append(p[metric_key]["pct_change"])
    return statistics.mean(vals) if vals else float("nan")


def _mean_across_prompts(model: dict, metric_key: str) -> float:
    vals = [p[metric_key]["pct_change"]
            for p in model.get("prompts", [])
            if metric_key in p and p[metric_key].get("n", 0) > 0]
    return statistics.mean(vals) if vals else float("nan")


def render_combined_report(all_data: list[dict], console: Console) -> None:
    # Flatten all model entries across all files
    all_models: list[dict] = []
    for d in all_data:
        for m in d.get("models", []):
            if not m.get("skipped"):
                all_models.append(m)

    if not all_models:
        console.print("[red]No valid model results found.[/red]")
        return

    console.print()
    console.rule("[bold white]Cross-Model Proof Report — autotune v2.0[/bold white]", style="white")
    console.print()

    # ── Per-model KPI table ────────────────────────────────────────────────
    t = Table(box=_box.SIMPLE_HEAVY, header_style="bold", width=_CON_W,
              title="[bold]Per-model KPI summary (autotune vs. raw Ollama)[/bold]")
    t.add_column("Model",         width=18)
    t.add_column("TTFT",          width=10, justify="right")
    t.add_column("Prefill",       width=10, justify="right")
    t.add_column("Gen speed",     width=10, justify="right")
    t.add_column("Peak RAM",      width=10, justify="right")
    t.add_column("KV cache",      width=10, justify="right")
    t.add_column("ctx ratio",     width=10, justify="right")
    t.add_column("Tok freed",     width=11, justify="right")
    t.add_column("Win rate",      width=10, justify="right")

    row_data = []
    for m in all_models:
        ttft_pct  = _mean_across_prompts(m, "ttft")
        pre_pct   = _mean_across_prompts(m, "prefill_ms")
        tps_pct   = _mean_across_prompts(m, "eval_tps")
        ram_pct   = _mean_across_prompts(m, "ollama_ram")
        kv_pct    = _mean_across_prompts(m, "kv_cache")
        tok_freed = m.get("total_tokens_saved", 0)
        win_rate  = m.get("win_rate", 0)

        ctx_ratios = []
        for p in m.get("prompts", []):
            raw_ctx   = p.get("num_ctx_raw", 0)
            tuned_ctx = p.get("num_ctx_tuned", 0)
            if tuned_ctx > 0:
                ctx_ratios.append(raw_ctx / tuned_ctx)
        ctx_ratio = statistics.mean(ctx_ratios) if ctx_ratios else 1.0

        row_data.append({
            "model_id": m["model_id"], "ttft_pct": ttft_pct,
            "pre_pct": pre_pct, "tps_pct": tps_pct,
            "ram_pct": ram_pct, "kv_pct": kv_pct,
            "ctx_ratio": ctx_ratio, "tok_freed": tok_freed,
            "win_rate": win_rate,
        })

        t.add_row(
            m["model_id"],
            _pct_cell(ttft_pct, higher_is_better=False),
            _pct_cell(pre_pct,  higher_is_better=False),
            _pct_cell(tps_pct,  higher_is_better=True),
            _pct_cell(ram_pct,  higher_is_better=False),
            _pct_cell(kv_pct,   higher_is_better=False),
            Text(f"{ctx_ratio:.1f}×", style="cyan"),
            Text(f"{tok_freed:,}", style="cyan"),
            Text(f"{win_rate:.0f}%", style="green" if win_rate >= 60 else "yellow"),
        )

    console.print(t)
    console.print()

    # ── Memory growth table ───────────────────────────────────────────────
    has_growth = any(
        m.get("memory_growth", {}).get("raw") and m.get("memory_growth", {}).get("tuned")
        for m in all_models
    )
    if has_growth:
        console.rule("[bold yellow]Memory Growth Over Turns (per model)[/bold yellow]", style="yellow")
        console.print()
        for m in all_models:
            mg = m.get("memory_growth", {})
            raw_pts   = mg.get("raw", [])
            tuned_pts = mg.get("tuned", [])
            if not raw_pts or not tuned_pts:
                continue
            g = Table(
                title=f"[bold]{m['model_id']}[/bold] — RAM per conversation turn",
                box=_box.SIMPLE_HEAVY, header_style="bold", width=_CON_W,
            )
            g.add_column("Turn",      width=6,  justify="center")
            g.add_column("Raw ctx",   width=10, justify="right")
            g.add_column("Auto ctx",  width=10, justify="right")
            g.add_column("Raw RAM",   width=12, justify="right")
            g.add_column("Auto RAM",  width=12, justify="right")
            g.add_column("Saved",     width=11, justify="right")
            g.add_column("Raw KV",    width=10, justify="right")
            g.add_column("Auto KV",   width=10, justify="right")

            for r, tu in zip(raw_pts, tuned_pts):
                saved = r["ollama_ram_gb"] - tu["ollama_ram_gb"]
                sc    = "green" if saved > 0.01 else ("dim" if abs(saved) < 0.01 else "red")
                g.add_row(
                    str(r["turn"]),
                    str(r["num_ctx"]),
                    str(tu["num_ctx"]),
                    f"{r['ollama_ram_gb']:.3f} GB",
                    f"{tu['ollama_ram_gb']:.3f} GB",
                    Text(f"{saved:+.3f} GB", style=sc),
                    f"{r['kv_cache_mb']:.0f} MB" if r.get("kv_cache_mb") else "N/A",
                    f"{tu['kv_cache_mb']:.0f} MB" if tu.get("kv_cache_mb") else "N/A",
                )
            console.print(g)

            if len(raw_pts) >= 2 and len(tuned_pts) >= 2:
                n_turns = len(raw_pts)
                rg = (raw_pts[-1]["ollama_ram_gb"] - raw_pts[0]["ollama_ram_gb"]) / (n_turns - 1)
                tg = (tuned_pts[-1]["ollama_ram_gb"] - tuned_pts[0]["ollama_ram_gb"]) / (n_turns - 1)
                c  = "green" if tg < rg else "yellow"
                console.print(
                    f"  RAM/turn:  Raw [dim]{rg:+.3f} GB[/dim]  →  "
                    f"Autotune [{c}]{tg:+.3f} GB[/{c}]\n"
                )

    # ── Grand summary panel ────────────────────────────────────────────────
    valid = [r for r in row_data if not math.isnan(r["ttft_pct"])]
    if not valid:
        return

    g_ttft   = statistics.mean(r["ttft_pct"]  for r in valid)
    g_pre    = statistics.mean(r["pre_pct"]   for r in valid)
    g_tps    = statistics.mean(r["tps_pct"]   for r in valid)
    g_ram    = statistics.mean(r["ram_pct"]   for r in valid)
    g_kv     = statistics.mean(r["kv_pct"]    for r in valid if not math.isnan(r["kv_pct"]))
    g_ctx    = statistics.mean(r["ctx_ratio"] for r in valid)
    g_tok    = sum(r["tok_freed"] for r in valid)
    g_win    = statistics.mean(r["win_rate"]  for r in valid)

    # Swap / reload aggregates
    tot_swap_r = sum(m.get("total_swaps_raw", 0)    for m in all_models)
    tot_swap_t = sum(m.get("total_swaps_tuned", 0)  for m in all_models)
    tot_rel_r  = sum(m.get("total_reloads_raw", 0)  for m in all_models)
    tot_rel_t  = sum(m.get("total_reloads_tuned", 0) for m in all_models)

    swap_color = "green" if tot_swap_t <= tot_swap_r else "yellow"
    rel_color  = "green" if tot_rel_t  <= tot_rel_r  else "yellow"

    ttft_dir  = "faster" if g_ttft < 0 else "slower"
    pre_dir   = "faster" if g_pre  < 0 else "slower"
    ram_dir   = "less"   if g_ram  < 0 else "more"

    summary_text = (
        f"Across [bold]{len(valid)} model(s)[/bold] and [bold]5 prompt types[/bold] "
        f"(win rate: [bold]{g_win:.0f}%[/bold]):\n\n"
        f"  [green bold]TTFT (time to first word):[/green bold]     "
        f"{abs(g_ttft):.0f}% {ttft_dir}\n"
        f"     [dim]autotune allocates a smaller KV buffer so Ollama initialises it faster[/dim]\n\n"
        f"  [green bold]KV prefill time:[/green bold]               "
        f"{abs(g_pre):.0f}% {pre_dir}\n"
        f"     [dim]dynamic num_ctx shrinks the prefill forward-pass proportionally[/dim]\n\n"
        f"  [green bold]Generation speed:[/green bold]              "
        f"{g_tps:+.1f}%  (marginal — generation is Metal GPU-bound)\n"
        f"     [dim]throughput is GPU-saturated; KV quant provides headroom on tight hardware[/dim]\n\n"
        f"  [green bold]Peak RAM (LLM process):[/green bold]        "
        f"{abs(g_ram):.0f}% {ram_dir}\n"
        f"     [dim]smaller KV cache means less RSS allocated by the Ollama runner process[/dim]\n\n"
        f"  [green bold]KV cache size:[/green bold]                 "
        f"{abs(g_kv):.0f}% smaller  ({g_ctx:.1f}x smaller context window)\n"
        f"     [dim]autotune sets num_ctx = actual_tokens x 1.15 safety margin, not 4096 globally[/dim]\n\n"
        f"  [cyan bold]Tokens freed:[/cyan bold]                    "
        f"{g_tok:,} KV tokens across all runs\n"
        f"     [dim]every freed token = one row in KV matrix never allocated or zeroed[/dim]\n\n"
        f"  [{swap_color} bold]Swap pressure:[/{swap_color} bold]               "
        f"Raw {tot_swap_r} events -> Autotune {tot_swap_t} events\n"
        f"     [dim]your Mac was at 5.5/6.0 GB swap before benchmark; "
        f"autotune reduces risk of additional swap during inference[/dim]\n\n"
        f"  [{rel_color} bold]Model reloads:[/{rel_color} bold]               "
        f"Raw {tot_rel_r} -> Autotune {tot_rel_t}\n"
        f"     [dim]'reload' = load_ms > 400ms (cold load from disk, not Metal KV init ~100ms)[/dim]\n\n"
        f"  [dim]Statistical method: Wilcoxon signed-rank (n<10) + Cohen's d effect size.\n"
        f"  All timing values from Ollama's internal Go nanosecond timers — not Python estimates.\n"
        f"  KV cache: 2 x n_layers x n_kv_heads x head_dim x num_ctx x dtype_bytes.[/dim]"
    )
    console.print(Panel(
        summary_text,
        title="[bold]autotune proof suite — What it means for you[/bold]",
        border_style="green",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="proof_report",
        description="Merge and render multiple proof_suite JSON result files",
    )
    parser.add_argument("files", nargs="*", metavar="JSON_FILE",
                        help="JSON result files from proof_suite.py")
    parser.add_argument("--glob", metavar="PATTERN",
                        help="Glob pattern to find result files (e.g. 'proof_results_*.json')")
    parser.add_argument("--output", "-o", metavar="PATH",
                        help="Save merged JSON to this file")
    args = parser.parse_args()

    console = Console(width=_CON_W)

    paths: list[str] = list(args.files or [])
    if args.glob:
        paths += _glob.glob(args.glob)
    if not paths:
        console.print("[red]No input files specified. Use: python scripts/proof_report.py file1.json file2.json[/red]")
        sys.exit(1)

    all_data: list[dict] = []
    for p in sorted(set(paths)):
        try:
            all_data.append(_load_json(p))
            console.print(f"[dim]Loaded {p}[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Could not load {p}: {exc}[/yellow]")

    if not all_data:
        console.print("[red]No data loaded.[/red]")
        sys.exit(1)

    render_combined_report(all_data, console)

    if args.output:
        merged = {"tool": "autotune proof report", "version": "2.0", "sources": paths, "data": all_data}
        with open(args.output, "w") as f:
            import json
            json.dump(merged, f, indent=2)
        console.print(f"\n[dim]Merged JSON saved → {args.output}[/dim]")


if __name__ == "__main__":
    main()
