#!/usr/bin/env python3
"""
autotune full benchmark suite — the complete tell-all.

Runs three benchmark layers sequentially, saves every result,
and sends a desktop notification when done.

  Layer 1 — proof-suite (statistical)  : TTFT, KV cache, RAM, swap
                                          Wilcoxon + Cohen's d
  Layer 2 — user-bench (real workflows) : swap events, RAM headroom,
                                          TTFT consistency, CPU spikes,
                                          background impact score
  Layer 3 — agent-bench (agentic tasks) : TTFT per turn, context growth,
                                          model reloads, tool-call errors

Usage
-----
  # Auto-detect models, run everything in the background:
  python scripts/run_full_benchmark.py --background

  # Specific models only:
  python scripts/run_full_benchmark.py --models qwen3:8b llama3.2:3b

  # Foreground (see live output):
  python scripts/run_full_benchmark.py

Output
------
  benchmark_results/
    proof_<model>.json      — proof-suite statistical results
    user_bench_<model>.json — user-experience KPIs
    agent_bench_results.json — agentic task metrics
    summary.json            — cross-layer summary
    summary.txt             — human-readable report card
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODELS = ["llama3.2:3b", "gemma4:e2b", "qwen3:8b"]
OUTPUT_DIR = _ROOT / "benchmark_results"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_ollama() -> list[str]:
    import httpx
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=4.0)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _notify(title: str, message: str) -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            script = f'display notification "{message}" with title "{title}" sound name "Glass"'
            subprocess.run(["osascript", "-e", script], check=False, capture_output=True, timeout=5)
        elif system == "Linux":
            subprocess.run(["notify-send", title, message], check=False, capture_output=True, timeout=5)
    except Exception:
        pass


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}]  {msg}", flush=True)


def _run_cmd(cmd: list[str], label: str) -> int:
    _log(f"Starting: {label}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_ROOT))
    elapsed = time.time() - t0
    status = "✓" if result.returncode == 0 else "✗"
    _log(f"{status}  {label}  ({elapsed:.0f}s)")
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def _generate_summary(models: list[str], output_dir: Path) -> None:
    summary: dict = {"models": {}, "highlights": []}
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("  autotune FULL BENCHMARK RESULTS")
    lines.append("=" * 70)

    for model in models:
        safe = model.replace(":", "_").replace("/", "_")
        entry: dict = {"model": model}

        # Proof-suite results
        proof_path = output_dir / f"proof_{safe}.json"
        if proof_path.exists():
            try:
                proof = json.loads(proof_path.read_text())
                # Find best TTFT improvement across prompts
                best_ttft_pct = 0.0
                best_kv_pct = 0.0
                swap_events = 0
                for model_entry in proof.get("models", []):
                    if model_entry.get("model_id") != model:
                        continue
                    for ps in model_entry.get("prompt_stats", []):
                        ttft_pct = ps.get("ttft_ms", {}).get("pct_improvement", 0)
                        kv_pct = ps.get("kv_cache_mb", {}).get("pct_improvement", 0)
                        best_ttft_pct = max(best_ttft_pct, ttft_pct)
                        best_kv_pct = max(best_kv_pct, kv_pct)
                        swap_events += ps.get("swap_autotune", 0)
                entry["proof"] = {
                    "best_ttft_pct": round(best_ttft_pct, 1),
                    "best_kv_pct": round(best_kv_pct, 1),
                    "swap_events": swap_events,
                }
            except Exception:
                pass

        # User-bench results
        ub_path = output_dir / f"user_bench_{safe}.json"
        if ub_path.exists():
            try:
                ub = json.loads(ub_path.read_text())
                scenarios = ub.get("scenarios", [])
                if scenarios:
                    avg_score_at  = sum(s["autotune"]["background_impact_score"] for s in scenarios) / len(scenarios)
                    avg_score_raw = sum(s["raw"]["background_impact_score"] for s in scenarios) / len(scenarios)
                    total_swap_at  = sum(s["autotune"]["swap_events_total"] for s in scenarios)
                    best_ttft_pct  = max(s["improvement"]["ttft_pct"] for s in scenarios)
                    entry["user_bench"] = {
                        "background_score_raw":    round(avg_score_raw, 0),
                        "background_score_tuned":  round(avg_score_at, 0),
                        "swap_events_autotune":    total_swap_at,
                        "best_ttft_pct":           round(best_ttft_pct, 1),
                    }
            except Exception:
                pass

        summary["models"][model] = entry

        # Write per-model section
        lines.append(f"\n  ── {model} ──")
        if "proof" in entry:
            p = entry["proof"]
            lines.append(f"  TTFT improvement:   up to −{p['best_ttft_pct']:.0f}%")
            lines.append(f"  KV cache freed:     up to −{p['best_kv_pct']:.0f}%")
            lines.append(f"  Swap events:        {p['swap_events']} (autotune)")
        if "user_bench" in entry:
            u = entry["user_bench"]
            lines.append(f"  Background score:   Raw {u['background_score_raw']:.0f}/100 → autotune {u['background_score_tuned']:.0f}/100")
            lines.append(f"  Swap events (UX):   {u['swap_events_autotune']} total")
            lines.append(f"  TTFT (UX bench):    up to −{u['best_ttft_pct']:.0f}%")

    # Agent bench (shared across models)
    ab_path = output_dir / "agent_bench_results.json"
    if ab_path.exists():
        try:
            ab = json.loads(ab_path.read_text())
            lines.append("\n  ── Agent benchmark ──")
            for model_res in ab.get("model_results", []):
                mid = model_res.get("model_id", "")
                ttft_trend = model_res.get("ttft_trend_ms_per_turn", {})
                reload_diff = model_res.get("model_reload_diff", 0)
                ctx_reduction = model_res.get("context_reduction_pct", 0)
                lines.append(f"  {mid}")
                if ttft_trend:
                    lines.append(f"    TTFT trend:   Raw {ttft_trend.get('raw', '?')} ms/turn  →  autotune {ttft_trend.get('tuned', '?')} ms/turn")
                if reload_diff:
                    lines.append(f"    Reloads saved: {reload_diff} per session")
                if ctx_reduction:
                    lines.append(f"    Context at end: −{ctx_reduction:.0f}%")
        except Exception:
            pass

    lines.append("\n" + "=" * 70)
    lines.append(f"  Results directory: {output_dir}")
    lines.append("=" * 70)

    report = "\n".join(lines)
    (output_dir / "summary.txt").write_text(report)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(report)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="autotune full benchmark suite — proof + user-bench + agent-bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models", nargs="+", metavar="MODEL", default=None,
                   help="Models to benchmark. Default: auto-detect from installed.")
    p.add_argument("--background", action="store_true",
                   help="Fork to background before running (survives terminal close).")
    p.add_argument("--skip-proof",  action="store_true", help="Skip proof-suite layer.")
    p.add_argument("--skip-user",   action="store_true", help="Skip user-bench layer.")
    p.add_argument("--skip-agents", action="store_true", help="Skip agent-bench layer.")
    p.add_argument("--output-dir",  default=str(OUTPUT_DIR), help="Output directory.")
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Ollama check ──────────────────────────────────────────────────────────
    installed = _check_ollama()
    if not installed:
        print(
            "\n❌  No models found.\n"
            "    Pull one with: autotune pull qwen3:8b\n"
            "    (autotune starts Ollama automatically)\n"
        )
        sys.exit(1)

    # ── Resolve models ────────────────────────────────────────────────────────
    if args.models:
        models = [m for m in args.models if m in installed]
        missing = [m for m in args.models if m not in installed]
        if missing:
            print(f"⚠  Not installed (skipping): {', '.join(missing)}")
    else:
        models = [m for m in DEFAULT_MODELS if m in installed] or [installed[0]]

    print(f"\n  Models to benchmark: {', '.join(models)}")

    # ── Background fork ───────────────────────────────────────────────────────
    if args.background:
        if not hasattr(os, "fork"):
            print("Background mode not supported on Windows. Running in foreground.")
        else:
            pid = os.fork()
            if pid != 0:
                log_path = Path(args.output_dir) / "full_benchmark.log"
                print(
                    f"\n✓  Full benchmark running in background (PID {pid})\n"
                    f"   Log: {log_path}\n"
                    f"   Results: {args.output_dir}/\n"
                    f"   You'll get a desktop notification when done.\n"
                )
                sys.exit(0)
            # Child: redirect to log
            log_path = Path(args.output_dir) / "full_benchmark.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lf = open(log_path, "w", buffering=1)
            sys.stdout = lf
            sys.stderr = lf

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _log(f"Starting full benchmark — models: {', '.join(models)}")
    _log(f"Output: {output_dir}")

    python = sys.executable

    # ── Layer 1: proof-suite ──────────────────────────────────────────────────
    if not args.skip_proof:
        _log("=" * 50)
        _log("LAYER 1: Statistical proof-suite (TTFT, KV, RAM, swap)")
        _log("=" * 50)
        for model in models:
            safe = model.replace(":", "_").replace("/", "_")
            out_path = str(output_dir / f"proof_{safe}.json")
            cmd = [
                python, str(_ROOT / "scripts" / "proof_suite.py"),
                "--models", model,
                "--complete",
                "--runs", "3",
                "--output", out_path,
            ]
            _run_cmd(cmd, f"proof-suite: {model}")
    else:
        _log("Skipping proof-suite (--skip-proof)")

    # ── Layer 2: user-bench ───────────────────────────────────────────────────
    if not args.skip_user:
        _log("=" * 50)
        _log("LAYER 2: User experience benchmark (swap events, RAM headroom, TTFT consistency)")
        _log("=" * 50)
        for model in models:
            safe = model.replace(":", "_").replace("/", "_")
            cmd = [
                python, str(_ROOT / "scripts" / "user_bench.py"),
                "--model", model,
                "--runs", "3",
                "--output-dir", str(output_dir),
            ]
            _run_cmd(cmd, f"user-bench: {model}")
    else:
        _log("Skipping user-bench (--skip-user)")

    # ── Layer 3: agent-bench ──────────────────────────────────────────────────
    if not args.skip_agents:
        _log("=" * 50)
        _log("LAYER 3: Agentic benchmark (multi-turn, tool-call, TTFT per turn)")
        _log("=" * 50)
        cmd = [
            python, str(_ROOT / "scripts" / "agent_bench.py"),
        ]
        if models:
            # Pass models one at a time as -m flags
            for m in models:
                cmd += ["-m", m]
        cmd += ["--trials", "3"]
        # Save to output dir
        import os as _os
        _os.chdir(str(output_dir))
        _run_cmd(cmd, f"agent-bench: {', '.join(models)}")
    else:
        _log("Skipping agent-bench (--skip-agents)")

    elapsed = time.time() - t0
    _log(f"All layers complete in {elapsed:.0f}s")

    # ── Summary report ────────────────────────────────────────────────────────
    _log("Generating summary report…")
    _generate_summary(models, output_dir)

    # ── Desktop notification ──────────────────────────────────────────────────
    _notify(
        title="autotune benchmark complete ✅",
        message=f"All 3 layers done for {', '.join(models)} in {elapsed:.0f}s. See {output_dir}/summary.txt",
    )
    _log(f"Results saved to {output_dir}/")
    _log("Done.")


if __name__ == "__main__":
    main()
