"""
Terminal chat REPL.

Can run in two modes:
  1. Direct  — talks to backends directly (no server needed)
  2. Client  — connects to a running autotune server

Supports persistent conversations, streaming, and live metrics display.

Commands:
  /help               Show this help
  /new                Start a new conversation
  /history            Show conversation history
  /profile <name>     Switch profile: fast | balanced | quality
  /model <id>         Switch model
  /system <text>      Set system prompt
  /export             Export conversation as Markdown
  /metrics            Show session metrics
  /quit or Ctrl-C     Exit
"""

from __future__ import annotations

import asyncio
import os
import platform
import sys
import time
from typing import Optional

import psutil
import rich.box as _rich_box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .backends.chain import get_chain
from .backends.openai_compat import AuthError, BackendError, ModelNotAvailableError
from .conversation import get_conv_manager
from .ctx_utils import estimate_tokens, estimate_messages_tokens
from .kv_manager import build_ollama_options
from .hardware_tuner import get_tuner
from .profiles import PROFILES, get_profile

# Enable readline history and line-editing in input() calls.
# On macOS the stdlib "readline" is backed by libedit, which uses
# editline syntax for parse_and_bind.  We bind just enough to make
# Up/Down arrow work without breaking anything else.
try:
    import readline as _rl
    # libedit (macOS) uses "bind" syntax; GNU readline uses "Control-p: ..."
    if "libedit" in getattr(_rl, "__doc__", "") or sys.platform == "darwin":
        _rl.parse_and_bind("bind ^[[A ed-prev-history")
        _rl.parse_and_bind("bind ^[[B ed-next-history")
    else:
        _rl.parse_and_bind("tab: complete")
        _rl.parse_and_bind('"\\e[A": previous-history')
        _rl.parse_and_bind('"\\e[B": next-history')
    del _rl
except Exception:
    pass

console = Console()

_IS_APPLE_SILICON: bool = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)

HELP_TEXT = """
[bold]autotune chat commands[/bold]

  [cyan]/new[/cyan]                Start a new conversation (keeps model/profile)
  [cyan]/history[/cyan]            Show full conversation history
  [cyan]/profile[/cyan] [yellow]<name>[/yellow]     Switch profile: [yellow]fast[/yellow] | [yellow]balanced[/yellow] | [yellow]quality[/yellow]
  [cyan]/model[/cyan] [yellow]<id>[/yellow]         Switch to a different model (HF ID or local name)
  [cyan]/pull[/cyan] [yellow][model][/yellow]       Download an Ollama model (omit model to browse popular ones)
  [cyan]/delete[/cyan] [yellow][model][/yellow]     Delete a locally cached Ollama model
  [cyan]/system[/cyan] [yellow]<text>[/yellow]      Set / replace the system prompt
  [cyan]/export[/cyan]             Export conversation as Markdown
  [cyan]/metrics[/cyan]            Show session performance stats
  [cyan]/backends[/cyan]           Show available backends
  [cyan]/models[/cyan]             List locally available models
  [cyan]/quit[/cyan]               Exit

[dim]Tip: /pull with no argument lists popular models you can download.[/dim]
[dim]Tip: Up arrow cycles through input history (readline)[/dim]
"""


def _clear_line() -> None:
    """Overwrite the current terminal line (used to clear loading hints)."""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


class ChatSession:
    def __init__(
        self,
        model_id: str,
        profile_name: str = "balanced",
        system_prompt: Optional[str] = None,
        conv_id: Optional[str] = None,
        optimize: bool = True,
        no_swap: bool = False,
    ) -> None:
        self.model_id = model_id
        self.profile_name = profile_name
        self.system_prompt = system_prompt
        self.optimize = optimize
        self.no_swap = no_swap
        self.conv_mgr = get_conv_manager()
        self.chain = get_chain()
        self.tuner = get_tuner()

        # Session metrics
        self._total_tokens = 0
        self._total_time = 0.0
        self._request_count = 0
        self._ttft_sum = 0.0
        self._tps_baseline: Optional[float] = None  # rolling average for slow-token detection

        # Real-time optimizer (set up lazily in _start_optimizer)
        self._monitor = None
        self._advisor = None
        self._sess_cfg = None
        self._ctx_ceiling: Optional[int] = None           # lowered by advisor under pressure
        self._kv_precision_override: Optional[str] = None  # "Q8_0" or "F16" from advisor

        # No-swap: model architecture, fetched lazily on first request
        self._no_swap_arch = None   # autotune.memory.noswap.ModelArch, populated lazily

        # Telemetry / hardware
        self._hw_id: Optional[str] = None
        self._hw_profile = None
        self._last_run_id: Optional[int] = None

        # Conversation ID
        if conv_id:
            self.conv_id = conv_id
        else:
            self.conv_id = self.conv_mgr.create(
                model_id=model_id,
                profile=profile_name,
                system_prompt=system_prompt,
            )

    # ------------------------------------------------------------------ #
    # Display helpers                                                      #
    # ------------------------------------------------------------------ #

    def _print_header(self) -> None:
        profile = PROFILES[self.profile_name]
        opt_tag = "" if self.optimize else "  [dim]│  optimize=off[/dim]"
        swap_tag = "  [dim]│[/dim]  [green]no-swap[/green]" if self.no_swap else ""
        console.print()
        console.print(Panel(
            f"[bold cyan]{self.model_id}[/bold cyan]  │  "
            f"[yellow]{profile.label}[/yellow]  │  "
            f"[dim]conv:{self.conv_id}[/dim]  │  "
            f"[dim]Type [/dim][cyan]/help[/cyan][dim] for commands[/dim]{opt_tag}{swap_tag}",
            box=_rich_box.HORIZONTALS,
            padding=(0, 1),
        ))
        console.print()

    def _print_metrics(self) -> None:
        if self._request_count == 0:
            console.print("[dim]No requests yet.[/dim]")
            return
        avg_tps = self._total_tokens / max(self._total_time, 0.01)
        avg_ttft = self._ttft_sum / self._request_count
        opt_info = ""
        if self._ctx_ceiling:
            opt_info += f"  │  ctx-ceiling {self._ctx_ceiling:,}"
        if self._kv_precision_override:
            opt_info += f"  │  KV {self._kv_precision_override}"
        console.print(
            f"\n[dim]Session:  {self._request_count} requests  │  "
            f"{self._total_tokens} tokens  │  "
            f"avg {avg_tps:.1f} tok/s  │  "
            f"avg TTFT {avg_ttft:.0f} ms{opt_info}[/dim]\n"
        )

    async def _show_history(self) -> None:
        messages = self.conv_mgr.get_messages(self.conv_id)
        if not messages:
            console.print("[dim]No history yet.[/dim]")
            return
        console.print()
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                console.print(Panel(content, title="[dim]System[/dim]", style="dim"))
            elif role == "user":
                console.print(f"[bold blue]You:[/bold blue] {content}")
            else:
                metrics = ""
                if m.get("tokens_per_sec"):
                    metrics = f" [dim]({m['tokens_per_sec']:.1f} tok/s)[/dim]"
                console.print(f"[bold green]Assistant:[/bold green]{metrics}")
                console.print(Markdown(content))
            console.print()

    # ------------------------------------------------------------------ #
    # Model preload — fixes the silent-hang / broken-startup bug           #
    # ------------------------------------------------------------------ #

    async def _preload_model(self) -> None:
        """
        Resolve and pre-load the model before the first prompt.

        Without this, the first message silently blocks while downloading
        or loading model weights — users see "Assistant: " with no output
        and think the UI is broken.  By loading eagerly with a visible
        spinner, users understand something is happening.

        Also profiles hardware here (needed for telemetry and optimizer).
        """
        # ── Hardware profile + DB registration ──────────────────────────
        if self._hw_profile is None:
            try:
                from autotune.hardware.profiler import profile_hardware
                from autotune.db.store import get_db
                from autotune.db.fingerprint import hardware_to_db_dict
                hw = profile_hardware()
                self._hw_profile = hw
                db = get_db()
                hw_dict = hardware_to_db_dict(hw)
                db.upsert_hardware(hw_dict)
                self._hw_id = hw_dict["id"]
            except Exception:
                pass

        # ── MLX preload (Apple Silicon) ──────────────────────────────────
        if _IS_APPLE_SILICON:
            try:
                from .backends.mlx_backend import (
                    mlx_available, resolve_mlx_model_id,
                    _load_model_sync, is_mlx_model_loaded,
                    list_cached_mlx_models,
                )
                if mlx_available():
                    mlx_id = resolve_mlx_model_id(self.model_id)
                    cached_ids = {m["id"] for m in list_cached_mlx_models()} if mlx_id else set()
                    if mlx_id and mlx_id in cached_ids:
                        if is_mlx_model_loaded(mlx_id):
                            console.print("[dim]Model already in memory.[/dim]\n")
                            return
                        name = mlx_id.split("/")[-1]
                        loop = asyncio.get_event_loop()
                        with console.status(
                            f"[cyan]Loading[/cyan] [bold]{name}[/bold]…",
                            spinner="dots",
                        ):
                            try:
                                await loop.run_in_executor(
                                    None, _load_model_sync, mlx_id
                                )
                            except Exception as exc:
                                console.print(
                                    f"[yellow]⚠ Preload failed — will retry on first message[/yellow]"
                                    f"  [dim]({exc})[/dim]\n"
                                )
                                return
                        console.print("[green]✓ Model ready[/green]\n")
                        return
            except Exception:
                pass

        # ── Ollama probe ─────────────────────────────────────────────────
        try:
            running = await self.chain.ollama_running()
            if not running:
                console.print(
                    "[yellow]⚠ Ollama not detected.[/yellow]  "
                    "[dim]Model will load on first message.[/dim]\n"
                )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Real-time optimizer                                                  #
    # ------------------------------------------------------------------ #

    def _start_optimizer(self) -> None:
        """
        Start background metrics collection + adaptive advisor.

        The advisor watches RAM, swap, thermal state, and per-request
        tok/s / TTFT.  When pressure builds it reduces context ceiling
        or downgrades KV precision — changes take effect on the next
        request without interrupting the current one.
        """
        try:
            from autotune.session.monitor import MetricsCollector
            from autotune.session.advisor import AdaptiveAdvisor
            from autotune.session.types import SessionConfig

            profile = get_profile(self.profile_name)

            # Best-effort model architecture info for KV estimate
            n_layers, n_kv_heads, head_dim = 32, 8, 128
            try:
                from autotune.db.store import get_db
                model_info = get_db().get_model(self.model_id)
                if model_info:
                    n_layers   = model_info.get("n_layers")   or n_layers
                    n_kv_heads = model_info.get("n_kv_heads") or n_kv_heads
                    head_dim   = model_info.get("head_dim")   or head_dim
            except Exception:
                pass

            ctx = profile.max_context_tokens
            kv_gb = 2 * n_layers * n_kv_heads * head_dim * ctx * 2 / 1024**3
            total_gb = (
                self._hw_profile.effective_memory_gb
                if self._hw_profile else 16.0
            )

            self._sess_cfg = SessionConfig(
                model_id=self.model_id,
                model_name=self.model_id.split("/")[-1],
                quant="4bit",
                context_len=ctx,
                n_gpu_layers=n_layers,
                n_total_layers=n_layers,
                backend="mlx" if _IS_APPLE_SILICON else "ollama",
                kv_cache_precision=profile.kv_cache_precision,
                speculative_decoding=profile.speculative_decoding,
                concurrency=1,
                prompt_caching=profile.system_prompt_cache,
                weight_gb=0.0,
                kv_cache_gb=kv_gb,
                total_budget_gb=total_gb,
            )

            self._monitor = MetricsCollector(interval_sec=1.0)
            self._monitor.start()
            self._advisor = AdaptiveAdvisor(self._sess_cfg)
        except Exception:
            # Optimizer is optional — never let it crash the chat
            self._monitor = None
            self._advisor = None
            self._sess_cfg = None

    def _update_optimizer(self, tps: float, ttft_ms: float) -> None:
        """Feed the latest inference metrics to the advisor and apply any decisions."""
        if not (self._monitor and self._advisor):
            return
        try:
            from dataclasses import replace as _dc_replace
            sys_m = self._monitor.latest
            if sys_m is None:
                return
            # Inject actual inference performance into the system snapshot
            updated = _dc_replace(
                sys_m,
                tokens_per_sec=tps,
                gen_tokens_per_sec=tps,
                ttft_ms=ttft_ms,
            )
            decisions = self._advisor.update(updated)
            if decisions:
                self._apply_optimizer_decisions(decisions)
        except Exception:
            pass

    def _apply_optimizer_decisions(self, decisions: list) -> None:
        """Apply advisor-recommended config changes and notify the user."""
        for d in decisions:
            ch = d.suggested_changes
            if "context_len" in ch:
                self._ctx_ceiling = ch["context_len"]
                if self._sess_cfg:
                    self._sess_cfg.context_len = ch["context_len"]
                # Extract just the action part of the reason for a clean message
                reason_short = d.reason.split("(")[0].strip()
                console.print(
                    f"[dim]⚙ Optimizer: context → {ch['context_len']:,} tokens"
                    f"  ({reason_short})[/dim]"
                )
            if "kv_cache_precision" in ch:
                prec = ch["kv_cache_precision"]
                self._kv_precision_override = "Q8_0" if prec in ("q8", "q4") else "F16"
                if self._sess_cfg:
                    self._sess_cfg.kv_cache_precision = prec
                console.print(
                    f"[dim]⚙ Optimizer: KV precision → {prec.upper()}[/dim]"
                )

    # ------------------------------------------------------------------ #
    # Telemetry                                                            #
    # ------------------------------------------------------------------ #

    def _log_run(
        self,
        tps: float,
        ttft_ms: float,
        elapsed: float,
        prompt_tokens: int,
        comp_tokens: int,
        backend: str,
        ollama_opts: dict,
        error_msg: Optional[str] = None,
    ) -> None:
        """
        Persist this inference run to the telemetry DB.

        Logged for every chat turn regardless of whether the session monitor
        is running.  Failures are silently swallowed — telemetry must never
        crash the chat.
        """
        try:
            from autotune.db.store import get_db
            db = get_db()

            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()

            # Ensure model stub exists so the FK constraint is satisfied
            if not db.get_model(self.model_id):
                db.upsert_model({
                    "id": self.model_id,
                    "name": self.model_id.split("/")[-1].split(":")[0],
                    "fetched_at": time.time(),
                })

            quant = "4bit" if backend == "mlx" else "unknown"

            # Ensure a valid hardware_id exists (FK in run_observations)
            hw_id = self._hw_id
            if not hw_id:
                hw_id = "autotune_unknown_hw"
                if not db.get_hardware(hw_id):
                    db.upsert_hardware({
                        "id": hw_id,
                        "os_name": platform.system(),
                        "cpu_brand": "unknown",
                        "cpu_physical_cores": 0,
                        "cpu_logical_cores": 0,
                        "cpu_arch": platform.machine(),
                        "total_ram_gb": 0.0,
                        "gpu_backend": "none",
                        "is_unified_memory": 0,
                    })

            run_data: dict = {
                "model_id": self.model_id,
                "hardware_id": hw_id,
                "quant": quant,
                "context_len": ollama_opts.get("num_ctx", 0),
                "n_gpu_layers": -1,
                "profile_name": self.profile_name,
                "gen_tokens_per_sec": round(tps, 2) if tps else None,
                "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                "peak_ram_gb": round(vm.used / 1024**3, 2),
                "elapsed_sec": round(elapsed, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tokens,
                "f16_kv": int(ollama_opts.get("f16_kv", True)),
                "num_keep": ollama_opts.get("num_keep", 0),
                "completed": 0 if error_msg else 1,
                "notes": f"backend={backend} conv={self.conv_id}",
            }
            if error_msg:
                run_data["error_msg"] = str(error_msg)[:500]

            run_id = db.log_run(run_data)
            self._last_run_id = run_id

            # ── Telemetry events ──────────────────────────────────────
            # hw_id was already resolved above (with fallback stub)
            model = self.model_id
            ram_pct = vm.percent

            if ram_pct >= 93.0:
                db.log_telemetry_event(
                    "pressure_high", value_num=ram_pct,
                    run_id=run_id, hardware_id=hw_id, model_id=model,
                )
            elif ram_pct >= 80.0:
                db.log_telemetry_event(
                    "ram_spike", value_num=ram_pct,
                    run_id=run_id, hardware_id=hw_id, model_id=model,
                )

            if sw.percent > 10.0:
                db.log_telemetry_event(
                    "swap_spike", value_num=sw.percent,
                    run_id=run_id, hardware_id=hw_id, model_id=model,
                )

            if tps and self._tps_baseline and tps < self._tps_baseline * 0.5:
                db.log_telemetry_event(
                    "slow_token", value_num=round(tps, 2),
                    value_text=f"baseline={self._tps_baseline:.1f}",
                    run_id=run_id, hardware_id=hw_id, model_id=model,
                )

            if error_msg:
                db.log_telemetry_event(
                    "error", value_text=str(error_msg)[:200],
                    run_id=run_id, hardware_id=hw_id, model_id=model,
                )

            # Update rolling TPS baseline (first run sets it; subsequent runs EMA)
            if tps and comp_tokens > 10:
                if self._tps_baseline is None:
                    self._tps_baseline = tps
                else:
                    self._tps_baseline = self._tps_baseline * 0.8 + tps * 0.2

        except Exception:
            pass  # telemetry must never crash the chat

    # ------------------------------------------------------------------ #
    # Model download                                                       #
    # ------------------------------------------------------------------ #

    async def _pull_model(self, model_id: str) -> bool:
        """
        Pull an Ollama model in a thread pool so the event loop stays free.

        Returns True on success, False on failure (errors are displayed inline).
        """
        from .ollama_pull import OllamaNotRunningError, PullError, pull_model

        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: pull_model(model_id, console)
            )
        except OllamaNotRunningError as e:
            console.print(f"[red]Ollama not running:[/red] {e}")
            return False
        except PullError as e:
            console.print(f"[red]Pull failed:[/red] {e}")
            return False

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    async def _chat(self, user_input: str, _retried: bool = False) -> None:
        profile = get_profile(self.profile_name)

        # Add user message to conversation — but ONLY on the first attempt.
        # When _retried=True the message was already saved in the first call;
        # saving it again would corrupt history with a duplicate user turn,
        # which causes the model to re-process the old message on subsequent
        # turns — the "auto-send" bug.
        if not _retried:
            self.conv_mgr.add_message(self.conv_id, "user", user_input)

        # Build context via the intelligent ContextWindow manager.
        # new_user_message=None because the user turn was just saved to DB above.
        # reserved_for_output ensures the model has headroom to reply.
        msgs, _ = self.conv_mgr.build_context(
            self.conv_id,
            profile.max_context_tokens,
            new_user_message=None,
            reserved_for_output=profile.max_new_tokens,
        )

        prompt_tokens = estimate_messages_tokens(msgs)

        # Apply hardware optimization
        self.tuner._apply(self.profile_name)

        console.print()

        t_start = time.time()
        first_token_t: Optional[float] = None
        collected: list[str] = []
        backend_used = "?"
        header_shown = False
        error_msg: Optional[str] = None

        # Lazily fetch model architecture for no-swap guarantee.
        # Done once per session — cached in self._no_swap_arch.
        if self.no_swap and self._no_swap_arch is None:
            from autotune.memory.noswap import NoSwapGuard
            self._no_swap_arch = await NoSwapGuard.get_model_arch(self.model_id)

        # Compute dynamic num_ctx and KV precision, applying any live
        # optimizer overrides (context ceiling / KV precision downgrade).
        ollama_opts = build_ollama_options(
            msgs, profile,
            context_ceiling=self._ctx_ceiling,
            kv_precision_override=self._kv_precision_override,
            no_swap_arch=self._no_swap_arch if self.no_swap else None,
        )

        # Show a subtle loading hint.  Printed with \r so the first token
        # can overwrite it cleanly — this avoids the old "Assistant: [blank]"
        # prompt that made users think they needed to type the response.
        sys.stdout.write("  \033[2m[generating…]\033[0m\r")
        sys.stdout.flush()

        try:
            async for chunk in self.chain.stream(
                self.model_id,
                msgs,
                max_new_tokens=profile.max_new_tokens,
                temperature=profile.temperature,
                top_p=profile.top_p,
                repetition_penalty=profile.repetition_penalty,
                timeout=profile.request_timeout_sec,
                num_ctx=ollama_opts["num_ctx"],
                ollama_options=ollama_opts,
            ):
                backend_used = chunk.backend
                if chunk.content:
                    if not header_shown:
                        # First token: clear the loading hint then print header
                        _clear_line()
                        console.print("[bold green]Assistant:[/bold green] ", end="")
                        header_shown = True
                    if first_token_t is None:
                        first_token_t = time.time()
                    print(chunk.content, end="", flush=True)
                    collected.append(chunk.content)

            # If the stream completed without any content (model returned empty
            # response, no error), the loading hint is still on the terminal line.
            # Clear it so it doesn't bleed into the next "You: " prompt.
            if not header_shown:
                _clear_line()

        except ModelNotAvailableError as e:
            if not header_shown:
                _clear_line()
            error_msg = str(e)
            # For Ollama-style names (no "/"), offer to pull the model right now
            looks_like_ollama = "/" not in self.model_id and "hf_api" not in str(e).lower()
            if looks_like_ollama and not _retried:
                console.print(
                    f"\n[yellow]Model [bold]{self.model_id}[/bold] not found locally.[/yellow]"
                )
                try:
                    console.file.flush()
                    sys.stdout.flush()
                    answer = input("  → Download it from Ollama now? [Y/n]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                if answer in ("", "y", "yes"):
                    pulled = await self._pull_model(self.model_id)
                    if pulled:
                        # Invalidate backend cache so next request picks up the new model
                        self.chain._ollama_ok = None
                        self.chain._ollama_probed_at = 0.0
                        console.print(
                            "[dim]Model downloaded — your message has been kept. "
                            "Sending it now…[/dim]"
                        )
                        # Re-run the same message now that the model exists
                        await self._chat(user_input, _retried=True)
                        return
                else:
                    console.print(
                        f"[dim]Tip: pull later with  "
                        f"[bold]autotune pull {self.model_id}[/bold][/dim]"
                    )
            else:
                console.print(f"\n[red]Model not available:[/red] {e}")
        except AuthError as e:
            if not header_shown:
                _clear_line()
            error_msg = str(e)
            console.print(f"\n[red]Auth error:[/red] {e}")
        except BackendError as e:
            if not header_shown:
                _clear_line()
            error_msg = str(e)
            console.print(f"\n[red]Backend error:[/red] {e}")
        except Exception as e:
            if not header_shown:
                _clear_line()
            error_msg = str(e)
            console.print(f"\n[red]Unexpected error:[/red] {e}")
        finally:
            self.tuner._restore()

        content = "".join(collected)
        elapsed = time.time() - t_start
        ttft_ms = (first_token_t - t_start) * 1000 if first_token_t else 0.0
        comp_tokens = estimate_tokens(content)
        tps = comp_tokens / max(elapsed, 0.01)

        self._total_tokens += comp_tokens
        self._total_time += elapsed
        self._request_count += 1
        self._ttft_sum += ttft_ms

        if content:
            # Store assistant response
            self.conv_mgr.add_message(
                self.conv_id, "assistant", content,
                ttft_ms=ttft_ms, tokens_per_sec=tps, backend=backend_used,
            )
            # Auto-title from first exchange
            conv = self.conv_mgr.get(self.conv_id)
            if conv and not conv.get("title"):
                title = user_input[:50].strip().replace("\n", " ")
                self.conv_mgr.update_title(self.conv_id, title)
        elif not _retried:
            # Inference failed with no output.  The user message we saved at
            # the top of this function produced no assistant reply, leaving
            # an orphaned user turn in the DB.  Roll it back so the next
            # request doesn't replay a failed turn the user never got a
            # response for.  (On _retried=True the message was added by the
            # first call and should stay — the retry IS the recovery path.)
            try:
                self.conv_mgr.delete_last_message(self.conv_id, role="user")
            except Exception:
                pass  # rollback is best-effort

        print()  # newline after streaming
        console.print(
            f"[dim]  ⚡ {tps:.1f} tok/s  │  TTFT {ttft_ms:.0f} ms  │  "
            f"{comp_tokens} tokens  │  [{backend_used}]  │  "
            f"{elapsed:.1f}s[/dim]"
        )
        console.print()

        # Persist telemetry and feed inference metrics to the live optimizer
        self._log_run(tps, ttft_ms, elapsed, prompt_tokens, comp_tokens,
                      backend_used, ollama_opts, error_msg)
        self._update_optimizer(tps, ttft_ms)

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        self._print_header()

        # Pre-warm: load model weights and profile hardware before the user
        # types anything.  Fixes the "assistant doesn't respond" startup bug
        # caused by silent model loading during the first request.
        await self._preload_model()

        # Start background hardware monitor + adaptive advisor
        if self.optimize:
            self._start_optimizer()

        # Show existing history if resuming
        conv = self.conv_mgr.get(self.conv_id)
        if conv and conv.get("message_count", 0) > 0:
            console.print(
                f"[dim]Resuming conversation with {conv['message_count']} messages.[/dim]"
            )
            await self._show_history()

        try:
            while True:
                try:
                    # Flush both Rich's internal buffer and raw stdout before
                    # reading input.  Rich and print() write to the same fd but
                    # through different buffers; if either is un-flushed when
                    # input() is called the prompt can appear before the last
                    # assistant line, making it look like the terminal glitched.
                    console.file.flush()
                    sys.stdout.flush()
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]Bye![/dim]")
                    break

                if not user_input:
                    continue

                # Commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""

                    if cmd in ("/quit", "/exit"):
                        self._print_metrics()
                        console.print("[dim]Bye![/dim]")
                        break

                    elif cmd == "/help":
                        console.print(HELP_TEXT)

                    elif cmd == "/new":
                        self.conv_id = self.conv_mgr.create(
                            model_id=self.model_id,
                            profile=self.profile_name,
                            system_prompt=self.system_prompt,
                        )
                        console.print(f"[green]New conversation started: {self.conv_id}[/green]")
                        self._print_header()

                    elif cmd == "/history":
                        await self._show_history()

                    elif cmd == "/profile":
                        if arg in PROFILES:
                            self.profile_name = arg
                            # Reset optimizer overrides when profile changes
                            self._ctx_ceiling = None
                            self._kv_precision_override = None
                            console.print(f"[green]Profile: {PROFILES[arg].label}[/green]")
                        else:
                            console.print(
                                f"[red]Unknown profile. Choose: {list(PROFILES.keys())}[/red]"
                            )

                    elif cmd == "/model":
                        if arg:
                            self.model_id = arg
                            # Reset all per-model state for the new model
                            self._ctx_ceiling = None
                            self._kv_precision_override = None
                            self._no_swap_arch = None   # arch is model-specific
                            console.print(f"[green]Model: {arg}[/green]")
                            self._print_header()
                            # Pre-warm the new model
                            await self._preload_model()
                        else:
                            console.print(f"[dim]Current model: {self.model_id}[/dim]")

                    elif cmd == "/pull":
                        from .ollama_pull import print_popular_models
                        if not arg:
                            # No model given — show popular models to choose from
                            print_popular_models(console)
                            try:
                                console.file.flush()
                                sys.stdout.flush()
                                choice = input("  Model to pull (Enter to cancel): ").strip()
                            except (EOFError, KeyboardInterrupt):
                                choice = ""
                            if not choice:
                                continue
                            arg = choice
                        pulled = await self._pull_model(arg)
                        if pulled:
                            # Invalidate Ollama backend cache so it picks up the new model
                            self.chain._ollama_ok = None
                            self.chain._ollama_probed_at = 0.0
                            # Ask if user wants to switch to the newly pulled model
                            try:
                                console.file.flush()
                                sys.stdout.flush()
                                switch = input(
                                    f"  Switch to {arg} now? [Y/n]: "
                                ).strip().lower()
                            except (EOFError, KeyboardInterrupt):
                                switch = "n"
                            if switch in ("", "y", "yes"):
                                self.model_id = arg
                                self._ctx_ceiling = None
                                self._kv_precision_override = None
                                self._no_swap_arch = None
                                self._print_header()
                                await self._preload_model()

                    elif cmd == "/system":
                        if arg:
                            self.system_prompt = arg
                            self.conv_mgr.update_system_prompt(self.conv_id, arg)
                            console.print("[green]System prompt updated.[/green]")
                        else:
                            console.print(f"[dim]{self.system_prompt or '(none)'}[/dim]")

                    elif cmd == "/export":
                        md = self.conv_mgr.export_markdown(self.conv_id)
                        fname = f"conv_{self.conv_id}.md"
                        with open(fname, "w") as f:
                            f.write(md)
                        console.print(f"[green]Exported to {fname}[/green]")

                    elif cmd == "/metrics":
                        self._print_metrics()

                    elif cmd == "/backends":
                        ollama = await self.chain.ollama_running()
                        lms = await self.chain.lmstudio_running()
                        hf = bool(os.environ.get("HF_TOKEN"))
                        console.print(
                            f"  Ollama:    {'[green]running[/green]' if ollama else '[dim]not running[/dim]'}\n"
                            f"  LM Studio: {'[green]running[/green]' if lms else '[dim]not running[/dim]'}\n"
                            f"  HF API:    {'[green]token set[/green]' if hf else '[yellow]no token (set HF_TOKEN)[/yellow]'}"
                        )

                    elif cmd == "/models":
                        models = await self.chain.discover_all()
                        if not models:
                            console.print(
                                "[yellow]No local models found. Start Ollama or set HF_TOKEN.[/yellow]"
                            )
                        for m in models[:30]:
                            size = f"  {m.size_gb:.1f} GB" if m.size_gb else ""
                            console.print(
                                f"  [cyan]{m.id}[/cyan][dim]  {m.source}{size}[/dim]"
                            )

                    elif cmd == "/delete":
                        from .ollama_pull import OllamaNotRunningError, PullError, delete_model
                        from .local_models import list_local_models
                        target = arg
                        if not target:
                            # Interactive picker: list Ollama models
                            ollama_models = [m for m in list_local_models() if m.source == "ollama"]
                            if not ollama_models:
                                console.print("[yellow]No Ollama models found.[/yellow]")
                                continue
                            for i, m in enumerate(ollama_models, 1):
                                size = f"  {m.size_gb:.1f} GB" if m.size_gb else ""
                                console.print(f"  [dim]{i}.[/dim] [cyan]{m.id}[/cyan][dim]{size}[/dim]")
                            try:
                                console.file.flush()
                                sys.stdout.flush()
                                choice = input("  Model to delete (Enter to cancel): ").strip()
                            except (EOFError, KeyboardInterrupt):
                                choice = ""
                            if not choice:
                                continue
                            if choice.isdigit():
                                idx = int(choice) - 1
                                if 0 <= idx < len(ollama_models):
                                    target = ollama_models[idx].id
                                else:
                                    console.print("[red]Invalid selection.[/red]")
                                    continue
                            else:
                                target = choice
                        # Confirm
                        try:
                            console.file.flush()
                            sys.stdout.flush()
                            ans = input(f"  Delete {target}? This cannot be undone. [y/N] ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            ans = ""
                        if ans not in ("y", "yes"):
                            console.print("[dim]Cancelled.[/dim]")
                            continue
                        try:
                            delete_model(target, console)
                            # If we deleted the current model, warn the user
                            if target == self.model_id:
                                console.print(
                                    "[yellow]Warning:[/yellow] You just deleted the active model. "
                                    "Use [cyan]/model <id>[/cyan] to switch to another."
                                )
                        except OllamaNotRunningError as exc:
                            console.print(f"[red]Ollama not running:[/red] {exc}")
                        except PullError as exc:
                            console.print(f"[red]Delete failed:[/red] {exc}")

                    else:
                        console.print(f"[dim]Unknown command: {cmd}. Type /help.[/dim]")

                    continue

                # Regular chat
                await self._chat(user_input)

        finally:
            # Clean up background monitor thread
            if self._monitor:
                try:
                    self._monitor.stop()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def _run_chat(
    model_id: str,
    profile: str,
    system_prompt: Optional[str],
    conv_id: Optional[str],
    optimize: bool = True,
    no_swap: bool = False,
) -> None:
    session = ChatSession(
        model_id=model_id,
        profile_name=profile,
        system_prompt=system_prompt,
        conv_id=conv_id,
        optimize=optimize,
        no_swap=no_swap,
    )
    await session.run()


def start_chat(
    model_id: str,
    profile: str = "balanced",
    system_prompt: Optional[str] = None,
    conv_id: Optional[str] = None,
    optimize: bool = True,
    no_swap: bool = False,
) -> None:
    """Entry point called from the CLI."""
    asyncio.run(_run_chat(model_id, profile, system_prompt, conv_id, optimize, no_swap))
