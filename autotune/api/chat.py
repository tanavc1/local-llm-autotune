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
import sys
import time
import uuid
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .backends.chain import get_chain
from .backends.openai_compat import AuthError, BackendError, ModelNotAvailableError
from .conversation import get_conv_manager
from .kv_manager import build_ollama_options
from .hardware_tuner import get_tuner
from .profiles import PROFILES, get_profile

# Enable readline history on Unix
try:
    import readline
    readline.parse_and_bind("")
except ImportError:
    pass

console = Console()

HELP_TEXT = """
[bold]autotune chat commands[/bold]

  [cyan]/new[/cyan]                Start a new conversation (keeps model/profile)
  [cyan]/history[/cyan]            Show full conversation history
  [cyan]/profile[/cyan] [yellow]<name>[/yellow]     Switch profile: [yellow]fast[/yellow] | [yellow]balanced[/yellow] | [yellow]quality[/yellow]
  [cyan]/model[/cyan] [yellow]<id>[/yellow]         Switch to a different model (HF ID or local name)
  [cyan]/system[/cyan] [yellow]<text>[/yellow]      Set / replace the system prompt
  [cyan]/export[/cyan]             Export conversation as Markdown
  [cyan]/metrics[/cyan]            Show session performance stats
  [cyan]/backends[/cyan]           Show available backends
  [cyan]/models[/cyan]             List locally available models
  [cyan]/quit[/cyan]               Exit

[dim]Tip: Up arrow cycles through input history (readline)[/dim]
"""


class ChatSession:
    def __init__(
        self,
        model_id: str,
        profile_name: str = "balanced",
        system_prompt: Optional[str] = None,
        conv_id: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.profile_name = profile_name
        self.system_prompt = system_prompt
        self.conv_mgr = get_conv_manager()
        self.chain = get_chain()
        self.tuner = get_tuner()

        # Session metrics
        self._total_tokens = 0
        self._total_time = 0.0
        self._request_count = 0
        self._ttft_sum = 0.0

        # Conversation ID
        if conv_id:
            self.conv_id = conv_id
        else:
            self.conv_id = self.conv_mgr.create(
                model_id=model_id,
                profile=profile_name,
                system_prompt=system_prompt,
            )

    def _print_header(self) -> None:
        profile = PROFILES[self.profile_name]
        console.print()
        console.print(Panel(
            f"[bold cyan]{self.model_id}[/bold cyan]  │  "
            f"[yellow]{profile.label}[/yellow]  │  "
            f"[dim]conv:{self.conv_id}[/dim]  │  "
            f"[dim]Type [/dim][cyan]/help[/cyan][dim] for commands[/dim]",
            box=__import__("rich.box", fromlist=["HORIZONTALS"]).HORIZONTALS,
            padding=(0, 1),
        ))
        console.print()

    def _print_metrics(self) -> None:
        if self._request_count == 0:
            console.print("[dim]No requests yet.[/dim]")
            return
        avg_tps = self._total_tokens / max(self._total_time, 0.01)
        avg_ttft = self._ttft_sum / self._request_count
        console.print(
            f"\n[dim]Session:  {self._request_count} requests  │  "
            f"{self._total_tokens} tokens  │  "
            f"avg {avg_tps:.1f} tok/s  │  "
            f"avg TTFT {avg_ttft:.0f} ms[/dim]\n"
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

    async def _chat(self, user_input: str) -> None:
        profile = get_profile(self.profile_name)

        # Add user message to conversation
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

        # Apply hardware optimization
        self.tuner._apply(self.profile_name)

        console.print()
        console.print("[bold green]Assistant:[/bold green] ", end="")

        t_start = time.time()
        first_token_t: Optional[float] = None
        collected: list[str] = []
        backend_used = "?"

        # Compute dynamic num_ctx and KV precision for this specific conversation
        ollama_opts = build_ollama_options(msgs, profile)

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
                    if first_token_t is None:
                        first_token_t = time.time()
                    print(chunk.content, end="", flush=True)
                    collected.append(chunk.content)

        except ModelNotAvailableError as e:
            console.print(f"\n[red]Model not available:[/red] {e}")
        except AuthError as e:
            console.print(f"\n[red]Auth error:[/red] {e}")
        except BackendError as e:
            console.print(f"\n[red]Backend error:[/red] {e}")
        finally:
            self.tuner._restore()

        content = "".join(collected)
        elapsed = time.time() - t_start
        ttft_ms = (first_token_t - t_start) * 1000 if first_token_t else 0
        comp_tokens = max(1, len(content) // 4)
        tps = comp_tokens / max(elapsed, 0.01)

        self._total_tokens += comp_tokens
        self._total_time += elapsed
        self._request_count += 1
        self._ttft_sum += ttft_ms

        # Store assistant response
        if content:
            self.conv_mgr.add_message(
                self.conv_id, "assistant", content,
                ttft_ms=ttft_ms, tokens_per_sec=tps, backend=backend_used,
            )
            # Auto-title from first exchange
            conv = self.conv_mgr.get(self.conv_id)
            if conv and not conv.get("title"):
                title = user_input[:50].strip().replace("\n", " ")
                self.conv_mgr.update_title(self.conv_id, title)

        print()  # newline after streaming
        console.print(
            f"[dim]  ⚡ {tps:.1f} tok/s  │  TTFT {ttft_ms:.0f} ms  │  "
            f"{comp_tokens} tokens  │  [{backend_used}]  │  "
            f"{elapsed:.1f}s[/dim]"
        )
        console.print()

    async def run(self) -> None:
        self._print_header()

        # Show existing history if resuming
        conv = self.conv_mgr.get(self.conv_id)
        if conv and conv.get("message_count", 0) > 0:
            console.print(f"[dim]Resuming conversation with {conv['message_count']} messages.[/dim]")
            await self._show_history()

        while True:
            try:
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

                if cmd == "/quit" or cmd == "/exit":
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
                        console.print(f"[green]Profile: {PROFILES[arg].label}[/green]")
                    else:
                        console.print(f"[red]Unknown profile. Choose: {list(PROFILES.keys())}[/red]")

                elif cmd == "/model":
                    if arg:
                        self.model_id = arg
                        console.print(f"[green]Model: {arg}[/green]")
                        self._print_header()
                    else:
                        console.print(f"[dim]Current model: {self.model_id}[/dim]")

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
                        console.print("[yellow]No local models found. Start Ollama or set HF_TOKEN.[/yellow]")
                    for m in models[:30]:
                        size = f"  {m.size_gb:.1f} GB" if m.size_gb else ""
                        console.print(f"  [cyan]{m.id}[/cyan][dim]  {m.source}{size}[/dim]")

                else:
                    console.print(f"[dim]Unknown command: {cmd}. Type /help.[/dim]")

                continue

            # Regular chat
            await self._chat(user_input)


async def _run_chat(
    model_id: str,
    profile: str,
    system_prompt: Optional[str],
    conv_id: Optional[str],
) -> None:
    session = ChatSession(
        model_id=model_id,
        profile_name=profile,
        system_prompt=system_prompt,
        conv_id=conv_id,
    )
    await session.run()


def start_chat(
    model_id: str,
    profile: str = "balanced",
    system_prompt: Optional[str] = None,
    conv_id: Optional[str] = None,
) -> None:
    """Entry point called from the CLI."""
    asyncio.run(_run_chat(model_id, profile, system_prompt, conv_id))
