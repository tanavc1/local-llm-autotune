"""
Session controller: orchestrates monitor, advisor, and dashboard.

The controller is the single entry point for `autotune session`.
It:
  1. Profiles hardware and recommends initial config
  2. Starts live metrics collection
  3. Runs the adaptive advisor each tick
  4. Renders the Rich live dashboard
  5. Logs observations to the DB
  6. Outputs JSON state on demand
"""

from __future__ import annotations

import json
import signal
import time
from typing import Optional

from rich.console import Console
from rich.live import Live

from autotune.config.generator import generate_recommendations
from autotune.hardware.profiler import HardwareProfile, profile_hardware
from autotune.memory.estimator import estimate_memory
from autotune.models.registry import QUANTIZATIONS, list_models

from .advisor import AdaptiveAdvisor
from .dashboard import LiveDashboard
from .monitor import MetricsCollector
from .types import AdvisorDecision, SessionConfig, SessionEvent, SessionState

console = Console()

# How often the main loop ticks (seconds)
TICK_INTERVAL = 1.0

# How often to write a DB observation (seconds)
DB_LOG_INTERVAL = 30.0


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_session_config(
    hw: HardwareProfile,
    model_id: Optional[str],
    mode: str,
) -> SessionConfig:
    """
    Build initial SessionConfig from hardware profile and recommendation engine.
    """
    recs = generate_recommendations(hw, modes=[mode], top_n=1)
    rec = recs.get(mode)

    if rec is None:
        # No config fits — use smallest possible fallback
        models = list_models()
        if not models:
            raise RuntimeError("No models in registry. Run `autotune fetch-many` first.")
        model = models[0]
        quant = model.quantization_options[0]
        ctx = 512
        gpu_layers = model.n_layers if hw.has_gpu else 0
    else:
        cand = rec.primary.candidate
        model = cand.model
        quant = cand.quant
        ctx = cand.context_len
        gpu_layers = cand.n_gpu_layers

    # If a specific model was requested, try to use it
    if model_id:
        from autotune.db.store import get_db
        db = get_db()
        db_model = db.get_model(model_id)
        if db_model and db_model.get("active_params_b"):
            # Build synthetic config from DB entry
            params_b = db_model["active_params_b"]
            quant_key = db_model.get("recommended_quant") or "Q4_K_M"
            weight_gb = params_b * QUANTIZATIONS.get(quant_key, QUANTIZATIONS["Q4_K_M"]).bytes_per_param
            kv_gb = _estimate_kv(
                db_model.get("n_layers") or 32,
                db_model.get("n_kv_heads") or 8,
                db_model.get("head_dim") or 128,
                ctx,
            )
            return SessionConfig(
                model_id=model_id,
                model_name=db_model["name"],
                quant=quant_key,
                context_len=ctx,
                n_gpu_layers=db_model.get("n_layers") or 32 if hw.has_gpu else 0,
                n_total_layers=db_model.get("n_layers") or 32,
                backend=hw.gpu.backend if hw.gpu else "cpu",
                kv_cache_precision="f16",
                speculative_decoding=False,
                concurrency=1,
                prompt_caching=True,
                weight_gb=weight_gb,
                kv_cache_gb=kv_gb,
                total_budget_gb=hw.effective_memory_gb,
            )

    mem = estimate_memory(model, quant, ctx, gpu_layers, hw.effective_memory_gb)

    return SessionConfig(
        model_id=model.id,
        model_name=model.name,
        quant=quant,
        context_len=ctx,
        n_gpu_layers=gpu_layers,
        n_total_layers=model.n_layers,
        backend=hw.gpu.backend if hw.gpu else "cpu",
        kv_cache_precision="f16",
        speculative_decoding=False,
        concurrency=1,
        prompt_caching=True,
        weight_gb=mem.weights_gb,
        kv_cache_gb=mem.kv_cache_gb,
        total_budget_gb=hw.effective_memory_gb,
    )


def _estimate_kv(n_layers: int, n_kv_heads: int, head_dim: int, ctx: int) -> float:
    return 2 * n_layers * n_kv_heads * head_dim * ctx * 2 / 1024**3


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _build_json_snapshot(
    hw: HardwareProfile,
    config: SessionConfig,
    advisor: AdaptiveAdvisor,
    start_time: float,
    metrics,
) -> dict:
    state = advisor.current_state
    decisions = advisor.recent_decisions[:3]

    recommended_config = {
        "model":            config.model_name,
        "model_id":         config.model_id,
        "quantization":     config.quant,
        "context_length":   config.context_len,
        "n_gpu_layers":     config.n_gpu_layers,
        "backend":          config.backend,
        "kv_cache_precision": config.kv_cache_precision,
        "speculative_decoding": config.speculative_decoding,
        "concurrency":      config.concurrency,
        "prompt_caching":   config.prompt_caching,
        "estimated_weight_gb": round(config.weight_gb, 2),
        "estimated_kv_gb":  round(config.kv_cache_gb, 2),
    }

    rationale = [e.message for e in advisor.events[-5:]]

    live_monitoring_plan = {
        "interval_sec":     TICK_INTERVAL,
        "thresholds": {
            "memory_warn_pct":    80,
            "memory_action_pct":  88,
            "memory_critical_pct": 94,
            "swap_growth_warn_mb_per_min": 10,
            "tps_drop_warn_pct":  30,
            "ttft_rise_warn_pct": 50,
        },
        "monitors": [
            "ram_percent", "vram_percent", "swap_growth_mb_per_min",
            "cpu_percent", "gpu_temp_c", "cpu_temp_c",
            "tokens_per_sec", "ttft_ms", "thermal_state",
            "llm_processes", "queue_depth",
        ],
        "runtime_apis": ["ollama:11434", "lmstudio:1234"],
    }

    adaptive_loop = {
        "current_state": state.value,
        "degradation_order": [
            "reduce_concurrency",
            "reduce_context",
            "lower_kv_precision",
            "improve_cache_reuse",
            "disable_speculative_decoding",
            "lower_quantization",
            "switch_to_smaller_model",
        ],
        "pending_actions": [
            {"action": d.action, "reason": d.reason, "changes": d.suggested_changes}
            for d in decisions
        ],
        "min_action_interval_sec": 20,
        "stable_before_scaleup_sec": 90,
    }

    session_status = {
        "uptime_sec": round(time.time() - start_time, 1),
        "state": state.value,
        "hardware": {
            "os":   hw.os_version,
            "cpu":  hw.cpu.brand,
            "ram_total_gb": round(hw.memory.total_gb, 1),
            "ram_available_gb": round(hw.memory.available_gb, 1),
            "gpu":  hw.gpu.name if hw.gpu else None,
            "gpu_backend": hw.gpu.backend if hw.gpu else None,
            "effective_budget_gb": round(hw.effective_memory_gb, 1),
        },
    }

    if metrics:
        session_status["live"] = {
            "ram_pct":  round(metrics.ram_percent, 1),
            "swap_gb":  round(metrics.swap_used_gb, 2),
            "cpu_pct":  round(metrics.cpu_percent, 1),
            "thermal":  metrics.thermal_state.value,
            "llm_proc_count": len(metrics.llm_processes),
        }

    return {
        "recommended_config": recommended_config,
        "rationale": rationale,
        "live_monitoring_plan": live_monitoring_plan,
        "adaptive_loop": adaptive_loop,
        "session_status": session_status,
    }


# ---------------------------------------------------------------------------
# Session controller
# ---------------------------------------------------------------------------

class SessionController:
    def __init__(
        self,
        model_id: Optional[str] = None,
        mode: str = "balanced",
        interval: float = TICK_INTERVAL,
        json_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.mode = mode
        self.interval = interval
        self.json_only = json_only
        self._stop = False

    def run(self) -> None:
        import psutil as _psutil

        # ── 1. Profile hardware ─────────────────────────────────────────
        console.print()
        console.print("[bold]Starting autotune live session[/bold]\n")
        console.print("[dim]Step 1 of 4[/dim]  Detecting hardware…")
        with console.status("  [dim]Scanning CPU, RAM, GPU, OS…[/dim]", spinner="dots"):
            hw = profile_hardware()

        gpu_str = (
            f"{hw.gpu.name} ({hw.gpu.backend.upper()})"
            if hw.gpu else "CPU-only"
        )
        vm = _psutil.virtual_memory()
        console.print(
            f"  [green]✓[/green]  {hw.cpu.brand[:40]}  ·  "
            f"{hw.memory.total_gb:.0f} GB RAM ({vm.available/1024**3:.1f} GB free)  ·  {gpu_str}"
        )

        # ── 2. Detect active LLMs ───────────────────────────────────────
        console.print("\n[dim]Step 2 of 4[/dim]  Scanning for running LLMs…")
        from .monitor import _detect_llm_processes, _query_ollama
        with console.status("  [dim]Checking Ollama API, process list…[/dim]", spinner="dots"):
            ollama_models = _query_ollama()
            llm_procs = _detect_llm_processes()

        if ollama_models:
            model_names = ", ".join(m.name for m in ollama_models[:3])
            console.print(
                f"  [green]✓[/green]  Ollama running — loaded: [cyan]{model_names}[/cyan]"
            )
        elif llm_procs:
            proc_names = ", ".join(f"{p.runtime} (pid {p.pid})" for p in llm_procs[:2])
            console.print(f"  [green]✓[/green]  LLM processes detected: {proc_names}")
        else:
            console.print(
                "  [dim]✓[/dim]  No active LLM detected  "
                "[dim](start one and session will auto-detect it)[/dim]"
            )

        # ── 3. Build initial config ─────────────────────────────────────
        console.print("\n[dim]Step 3 of 4[/dim]  Loading recommendation engine…")
        with console.status("  [dim]Scoring candidates for this hardware…[/dim]", spinner="dots"):
            try:
                config = _build_session_config(hw, self.model_id, self.mode)
            except Exception as e:
                console.print(f"  [red]✗  Config error: {e}[/red]")
                console.print("  [dim]Run `autotune fetch-many` to populate the model DB first.[/dim]")
                return

        console.print(
            f"  [green]✓[/green]  Session config ready  "
            f"(mode={self.mode},  "
            f"budget={hw.effective_memory_gb:.1f} GB)"
        )

        start_time = time.time()

        # ── 3b. JSON-only mode ──────────────────────────────────────────
        if self.json_only:
            advisor = AdaptiveAdvisor(config)
            snapshot = _build_json_snapshot(hw, config, advisor, start_time, None)
            console.print_json(json.dumps(snapshot, indent=2))
            return

        # ── 4. Start metrics collector ──────────────────────────────────
        console.print("\n[dim]Step 4 of 4[/dim]  Starting live monitor…")
        monitor = MetricsCollector(interval_sec=self.interval)
        monitor.start()
        console.print(
            f"  [green]✓[/green]  Sampling every {self.interval:.1f}s  ·  "
            f"Health checks every 30s  ·  Spike detection active  ·  Logging to DB"
        )
        console.print()
        console.print("[dim]Dashboard starting in 1 second… (Ctrl-C to exit)[/dim]")
        time.sleep(1.0)

        # ── 5. Advisor ──────────────────────────────────────────────────
        advisor = AdaptiveAdvisor(config)
        advisor._log(
            f"Session started — {hw.cpu.brand.split('@')[0].strip()[:32]}  ·  "
            f"{hw.memory.total_gb:.0f} GB RAM  ·  "
            f"{vm.available/1024**3:.1f} GB free  ·  {gpu_str}",
            "INFO",
        )
        if ollama_models:
            for om in ollama_models[:2]:
                advisor._log(
                    f"Ollama: {om.name} loaded  ({om.size_gb:.1f} GB weights"
                    + (f"  ctx {om.context_len:,}" if om.context_len else "")
                    + ")",
                    "INFO",
                )
        # Emit first health check immediately
        advisor._last_proactive_event_time = 0.0  # forces first health event on first tick

        # ── 6. Dashboard ────────────────────────────────────────────────
        dashboard = LiveDashboard(hw, config, start_time)

        # DB setup
        last_db_log = 0.0
        hw_id: Optional[str] = None
        db = None
        try:
            from autotune.db.fingerprint import hardware_to_db_dict
            from autotune.db.store import get_db
            db = get_db()
            hw_dict = hardware_to_db_dict(hw)
            db.upsert_hardware(hw_dict)
            hw_id = hw_dict["id"]
        except Exception:
            pass

        # ── 7. Main live loop ───────────────────────────────────────────
        all_decisions: list[AdvisorDecision] = []
        applied_optimizations: list[str] = []   # human-readable applied changes

        def _handle_sigint(sig, frame):
            self._stop = True

        signal.signal(signal.SIGINT, _handle_sigint)

        with Live(
            dashboard.render(None, advisor.current_state, advisor.events, all_decisions, stats={}),
            refresh_per_second=2,
            screen=True,
            console=Console(stderr=False),
        ) as live:
            while not self._stop:
                metrics = monitor.latest

                if metrics:
                    # Run advisor (emits proactive events, spike detection, etc.)
                    new_decisions = advisor.update(metrics)
                    if new_decisions:
                        all_decisions = new_decisions + all_decisions
                        for d in new_decisions:
                            self._apply_decision(config, d)
                            # Record applied optimizations for dashboard display
                            ch = d.suggested_changes
                            if "context_len" in ch:
                                applied_optimizations.insert(
                                    0, f"ctx→{ch['context_len']:,} tokens"
                                )
                            if "kv_cache_precision" in ch:
                                applied_optimizations.insert(
                                    0,
                                    f"KV {ch['kv_cache_precision'].upper()} precision",
                                )
                            if "prompt_caching" in ch:
                                applied_optimizations.insert(0, "prompt caching on")
                            if "concurrency" in ch:
                                applied_optimizations.insert(
                                    0, f"concurrency→{ch['concurrency']}"
                                )
                        # Keep list bounded
                        del applied_optimizations[6:]

                    # DB logging every 30 seconds
                    now = time.time()
                    if db and hw_id and (now - last_db_log) > DB_LOG_INTERVAL:
                        try:
                            db.log_run({
                                "model_id": config.model_id,
                                "hardware_id": hw_id,
                                "quant": config.quant,
                                "context_len": config.context_len,
                                "n_gpu_layers": config.n_gpu_layers,
                                "peak_ram_gb": round(metrics.ram_used_gb, 2),
                                "peak_vram_gb": round(metrics.vram_used_gb, 2) if metrics.vram_used_gb else None,
                                "tokens_per_sec": metrics.tokens_per_sec,
                                "gen_tokens_per_sec": metrics.gen_tokens_per_sec,
                                "ttft_ms": metrics.ttft_ms,
                                "oom": int(metrics.swap_growth_mb_per_min > 100),
                                "notes": f"session_monitor state={advisor.current_state.value}",
                            })
                            last_db_log = now
                        except Exception:
                            pass

                # Re-render dashboard at 2 fps
                live.update(
                    dashboard.render(
                        metrics,
                        advisor.current_state,
                        advisor.events,
                        all_decisions[:3],
                        stats={
                            "baseline_tps": advisor.baseline.tokens_per_sec,
                            "baseline_ttft": advisor.baseline.ttft_ms,
                            "applied_changes": list(applied_optimizations),
                        },
                    )
                )

                time.sleep(self.interval)

        monitor.stop()
        console.print("\n[cyan]Session ended.[/cyan]")

        # Brief summary on exit
        if monitor.latest:
            m = monitor.latest
            from .advisor import compute_health_score, health_status
            score = compute_health_score(m)
            label, color, icon = health_status(score)
            uptime = int(time.time() - start_time)
            h, rem = divmod(uptime, 3600)
            mins, sec = divmod(rem, 60)
            console.print(
                f"[dim]Session ran for {h:02d}:{mins:02d}:{sec:02d}  ·  "
                f"Final health: {score}/100 ({label})  ·  "
                f"Final RAM: {m.ram_percent:.0f}%  ·  "
                f"Final CPU: {m.cpu_percent:.0f}%[/dim]"
            )

    def _apply_decision(self, config: SessionConfig, decision: AdvisorDecision) -> None:
        """Update the live config based on an advisor decision."""
        ch = decision.suggested_changes
        if "context_len" in ch:
            config.context_len = ch["context_len"]
            # Recompute KV cache estimate
            from autotune.models.registry import MODEL_REGISTRY
            model = MODEL_REGISTRY.get(config.model_id)
            if model:
                config.kv_cache_gb = model.kv_cache_gb(ch["context_len"])
        if "kv_cache_precision" in ch:
            config.kv_cache_precision = ch["kv_cache_precision"]
        if "concurrency" in ch:
            config.concurrency = ch["concurrency"]
        if "speculative_decoding" in ch:
            config.speculative_decoding = ch["speculative_decoding"]
        if "prompt_caching" in ch:
            config.prompt_caching = ch["prompt_caching"]
        if "quant" in ch:
            config.quant = ch["quant"]
