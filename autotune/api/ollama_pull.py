"""
Ollama model pull with streaming progress.

Streams the Ollama /api/pull endpoint and renders a Rich progress bar
showing layer-by-layer download progress.  Works from both the CLI and
the in-chat /pull command.

The Ollama pull API emits NDJSON:
  {"status":"pulling manifest"}
  {"status":"pulling <hash>","digest":"sha256:...","total":4661211808,"completed":0}
  {"status":"pulling <hash>","digest":"sha256:...","total":4661211808,"completed":123456}
  ...
  {"status":"verifying sha256 digest"}
  {"status":"writing manifest"}
  {"status":"success"}
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich import box as _box

_OLLAMA_BASE = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OllamaNotRunningError(Exception):
    pass


class PullError(Exception):
    pass


# ---------------------------------------------------------------------------
# Popular model catalogue
# ---------------------------------------------------------------------------

@dataclass
class PopularModel:
    id: str          # Ollama model tag
    desc: str        # one-line description
    size: str        # approximate size string


POPULAR_MODELS: list[PopularModel] = [
    # ── Tiny (< 2 GB) — any laptop, even older hardware ────────────────
    PopularModel("smollm2:135m",      "SmolLM2 135M — ultra-tiny, fits on anything",    "~0.3 GB"),
    PopularModel("smollm2:1.7b",      "SmolLM2 1.7B — best tiny model for 4 GB RAM",    "~1.1 GB"),
    PopularModel("qwen3:0.6b",        "Qwen3 0.6B — thinking model, on-device tasks",   "~0.5 GB"),
    PopularModel("qwen3:1.7b",        "Qwen3 1.7B — strong reasoning for its size",     "~1.1 GB"),
    PopularModel("llama3.2:1b",       "Llama 3.2 1B — ultra-fast on any Mac",           "~1.3 GB"),
    # ── Small (2–5 GB) — 8 GB+ RAM ─────────────────────────────────────
    PopularModel("llama3.2:3b",       "Llama 3.2 3B — excellent small model",           "~2.0 GB"),
    PopularModel("qwen3:4b",          "Qwen3 4B — hybrid thinking, best 4B available",  "~2.6 GB"),
    PopularModel("gemma4:e2b",        "Gemma 4 2B — multimodal, 128k context",          "~2.5 GB"),
    PopularModel("qwen2.5:3b",        "Qwen 2.5 3B — great multilingual",               "~2.0 GB"),
    # ── Medium (5–12 GB) — 16 GB RAM ───────────────────────────────────
    PopularModel("qwen3:8b",          "Qwen3 8B — best general model for 16 GB laptops","~5.2 GB"),
    PopularModel("gemma4:e4b",        "Gemma 4 4B — top compact model, multimodal",     "~5.8 GB"),
    PopularModel("llama3.1:8b",       "Llama 3.1 8B — reliable 128k-context model",     "~4.7 GB"),
    PopularModel("qwen2.5:7b",        "Qwen 2.5 7B — excellent all-rounder",            "~4.7 GB"),
    PopularModel("qwen2.5-coder:7b",  "Qwen 2.5 Coder 7B — best small coding model",   "~4.7 GB"),
    PopularModel("deepseek-r1:7b",    "DeepSeek R1 7B — chain-of-thought reasoning",    "~4.7 GB"),
    PopularModel("gemma3:12b",        "Gemma 3 12B — strong multilingual model",        "~8.1 GB"),
    # ── Large (12–20 GB) — 24–32 GB RAM ────────────────────────────────
    PopularModel("qwen3:14b",         "Qwen3 14B — frontier-quality for 24 GB+",        "~9.0 GB"),
    PopularModel("qwen2.5:14b",       "Qwen 2.5 14B — strong all-around",               "~9.0 GB"),
    PopularModel("qwen2.5-coder:14b", "Qwen 2.5 Coder 14B — top coding model",         "~9.0 GB"),
    PopularModel("deepseek-r1:14b",   "DeepSeek R1 14B distill — top reasoning",        "~9.0 GB"),
    # ── XL (20–40 GB) — 32+ GB RAM ─────────────────────────────────────
    PopularModel("qwen3:30b-a3b",     "Qwen3 30B MoE (3.3B active) — near-frontier",   "~17 GB"),
    PopularModel("qwen3:32b",         "Qwen3 32B dense — top open model",               "~20 GB"),
    PopularModel("qwen2.5:32b",       "Qwen 2.5 32B — premium quality",                "~20 GB"),
    PopularModel("qwen2.5-coder:32b", "Qwen 2.5 Coder 32B — best open coding model",   "~20 GB"),
    PopularModel("deepseek-r1:32b",   "DeepSeek R1 32B distill — excellent reasoning",  "~20 GB"),
    PopularModel("qwq:32b",           "QwQ 32B — best open maths/science reasoning",    "~20 GB"),
]


def print_popular_models(console: Optional[Console] = None) -> None:
    """Print the popular model catalogue as a Rich table."""
    con = console or Console()
    t = Table(
        title="Popular Ollama Models  •  pull with [bold]autotune pull <id>[/bold]",
        box=_box.SIMPLE_HEAD,
        show_lines=False,
        title_style="bold",
    )
    t.add_column("Model ID", style="cyan", no_wrap=True)
    t.add_column("Size", justify="right", style="dim")
    t.add_column("Description")

    for m in POPULAR_MODELS:
        t.add_row(m.id, m.size, m.desc)

    con.print(t)
    con.print(
        "[dim]Tip: Models with matching MLX versions run natively on Apple Silicon "
        "(10–40 % faster).  autotune handles this automatically.[/dim]\n"
    )


# ---------------------------------------------------------------------------
# Ollama availability check
# ---------------------------------------------------------------------------

def is_ollama_running() -> bool:
    """Return True if the Ollama daemon is reachable."""
    try:
        req = urllib.request.Request(f"{_OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Streaming pull
# ---------------------------------------------------------------------------

def _stream_pull_events(model_id: str) -> Iterator[dict]:
    """Yield parsed JSON events from the Ollama /api/pull stream."""
    body = json.dumps({"model": model_id, "stream": True}).encode()
    req = urllib.request.Request(
        f"{_OLLAMA_BASE}/api/pull",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")[:300]
        raise PullError(f"Ollama returned HTTP {e.code}: {body_text}")
    except urllib.error.URLError as e:
        raise PullError(f"Cannot reach Ollama: {e.reason}")


def pull_model(model_id: str, console: Optional[Console] = None) -> bool:
    """
    Pull an Ollama model with a Rich progress display.

    Parameters
    ----------
    model_id : Ollama model tag, e.g. "llama3.2:3b" or "phi4-mini"
    console  : Rich Console to render into (defaults to a new one)

    Returns
    -------
    True on success.  Raises OllamaNotRunningError or PullError on failure.
    """
    con = console or Console()

    if not is_ollama_running():
        raise OllamaNotRunningError(
            "Ollama is not running.\n"
            "Start it with:  [bold]ollama serve[/bold]\n"
            "or open the Ollama desktop app."
        )

    con.print(f"\n[bold]Pulling[/bold] [cyan]{model_id}[/cyan] via Ollama…\n")

    # Track layers: digest → (total_bytes, completed_bytes)
    layer_totals:    dict[str, int] = {}
    layer_completed: dict[str, int] = {}
    layer_done:      set[str]       = set()
    layer_tasks:     dict[str, TaskID] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=32),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=con,
        transient=False,
    ) as progress:
        status_task = progress.add_task("[cyan]Connecting…", total=None)

        try:
            for event in _stream_pull_events(model_id):
                status  = event.get("status", "")
                digest  = event.get("digest", "")
                total   = event.get("total", 0)
                comp    = event.get("completed", 0)

                # ── Non-download status messages ───────────────────────
                if not digest:
                    label = status.capitalize() if status else "Working"
                    progress.update(status_task, description=f"[cyan]{label}[/cyan]")
                    if status == "success":
                        progress.update(status_task, description="[green]Complete[/green]")
                        break
                    continue

                # ── Layer download progress ────────────────────────────
                short = digest[7:19] if digest.startswith("sha256:") else digest[:12]

                # First event for this layer: create its task
                if digest not in layer_tasks:
                    layer_totals[digest] = total or 0
                    layer_completed[digest] = 0
                    t = progress.add_task(
                        f"  [dim]Layer {short}[/dim]",
                        total=total or None,
                    )
                    layer_tasks[digest] = t
                    progress.update(
                        status_task,
                        description=f"[cyan]{status.capitalize()}[/cyan]",
                        total=None,
                    )

                # Update layer task
                if total:
                    layer_totals[digest] = total
                    progress.update(layer_tasks[digest], total=total, completed=comp)
                    layer_completed[digest] = comp

                # Mark layer complete when fully downloaded
                if total and comp >= total and digest not in layer_done:
                    layer_done.add(digest)
                    progress.update(
                        layer_tasks[digest],
                        completed=total,
                        description=f"  [dim green]✓ {short}[/dim green]",
                    )

        except PullError:
            progress.stop()
            raise
        except KeyboardInterrupt:
            progress.stop()
            con.print("\n[yellow]Pull cancelled.[/yellow]")
            raise

    con.print(f"[green]✓[/green] [bold]{model_id}[/bold] is ready.\n")
    return True
