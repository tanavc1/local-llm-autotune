"""
running_models.py — snapshot every model currently resident in memory.

Polls all local backends (Ollama, MLX, LM Studio) and returns a unified
list of RunningModel entries.  Designed to be fast (all requests have short
timeouts) and safe to call at any time without side-effects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx


@dataclass
class RunningModel:
    name: str               # model identifier as reported by the backend
    backend: str            # "ollama" | "mlx" | "lmstudio"
    ram_gb: float           # unified/VRAM memory held right now
    context_len: int        # context window the model was loaded with (0 = unknown)
    loaded_since: float     # unix timestamp when it was loaded (0 = unknown)
    expires_at: float       # unix timestamp when backend will auto-unload (0 = never / unknown)
    quant: str              # quantization label, e.g. "Q4_K_M" (empty = unknown)
    family: str             # model family, e.g. "llama" (empty = unknown)

    # ------------------------------------------------------------------ #
    # Derived helpers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def age_str(self) -> str:
        """Human-readable time since the model was loaded."""
        if not self.loaded_since:
            return "unknown"
        secs = int(time.time() - self.loaded_since)
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        return f"{secs // 3600}h {(secs % 3600) // 60}m"

    @property
    def expires_str(self) -> str:
        """Human-readable time until auto-unload, or 'pinned' if keep_alive=-1."""
        if not self.expires_at:
            return "pinned"
        remaining = int(self.expires_at - time.time())
        if remaining <= 0:
            return "unloading…"
        if remaining < 60:
            return f"{remaining}s"
        if remaining < 3600:
            return f"{remaining // 60}m {remaining % 60}s"
        return f"{remaining // 3600}h {(remaining % 3600) // 60}m"


# ---------------------------------------------------------------------------
# Backend probes
# ---------------------------------------------------------------------------

def _probe_ollama() -> list[RunningModel]:
    """Query Ollama /api/ps — returns all currently loaded models."""
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get("http://localhost:11434/api/ps")
            if r.status_code != 200:
                return []
            data = r.json()
    except Exception:
        return []

    results: list[RunningModel] = []
    for m in data.get("models", []):
        details = m.get("details", {})

        # Parse expiry time from RFC-3339 string
        expires_ts = 0.0
        raw_exp = m.get("expires_at", "")
        if raw_exp and not raw_exp.startswith("0001"):  # zero-value = pinned
            try:
                dt = datetime.fromisoformat(raw_exp.replace("Z", "+00:00"))
                expires_ts = dt.timestamp()
            except Exception:
                pass

        results.append(RunningModel(
            name=m.get("name", "?"),
            backend="ollama",
            ram_gb=m.get("size_vram", m.get("size", 0)) / 1024 ** 3,
            context_len=m.get("context_length", 0),
            loaded_since=0.0,    # Ollama /api/ps doesn't expose load time
            expires_at=expires_ts,
            quant=details.get("quantization_level", ""),
            family=details.get("family", ""),
        ))
    return results


def _pid_alive(pid: int) -> bool:
    """Return True if a process with the given PID is currently running."""
    try:
        import os as _os
        _os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # EPERM: process exists, we just lack permission to signal it
    except Exception:
        return True  # unknown state — assume alive


def _probe_mlx() -> list[RunningModel]:
    """Check for a loaded MLX model, cross-process safe via a state file."""
    import json
    from pathlib import Path

    state_file = Path.home() / ".autotune" / "mlx_running.json"

    # Primary path: read state file written by the process that loaded the model.
    # This works whether ps runs in a separate process or the same one.
    try:
        if state_file.exists():
            data = json.loads(state_file.read_text())
            pid = data.get("pid", 0)
            if pid and not _pid_alive(pid):
                # Stale file — the loading process is gone; clean it up.
                state_file.unlink(missing_ok=True)
            else:
                return [RunningModel(
                    name=data["model_id"],
                    backend="mlx",
                    ram_gb=0.0,          # MLX doesn't expose per-model byte counts
                    context_len=0,
                    loaded_since=data.get("loaded_at", 0.0),
                    expires_at=0.0,      # MLX stays loaded until explicitly unloaded
                    quant="",
                    family="",
                )]
    except Exception:
        pass

    # Fallback: in-process import (only non-None when ps is called from within
    # the same server/chat process — e.g. from the /api/running_models endpoint).
    try:
        from autotune.api.backends.mlx_backend import _model_cache
        if _model_cache is not None:
            return [RunningModel(
                name=_model_cache.model_id,
                backend="mlx",
                ram_gb=0.0,
                context_len=0,
                loaded_since=_model_cache.loaded_at,
                expires_at=0.0,
                quant="",
                family="",
            )]
    except Exception:
        pass

    return []


def _probe_lmstudio() -> list[RunningModel]:
    """Query LM Studio's loaded model via /api/v0/models (v0 API exposes status)."""
    try:
        with httpx.Client(timeout=3.0) as client:
            # Try v0 API which includes loaded/status info
            r = client.get("http://localhost:1234/api/v0/models")
            if r.status_code != 200:
                # Fall back to OpenAI-compat endpoint (just names, no status)
                r = client.get("http://localhost:1234/v1/models")
                if r.status_code != 200:
                    return []
            data = r.json()
    except Exception:
        return []

    results: list[RunningModel] = []
    for m in data.get("data", []):
        # v0 API: state field indicates if actually loaded in memory
        state = m.get("state", "")
        if state and state not in ("loaded", "loading"):
            continue
        results.append(RunningModel(
            name=m.get("id", "?"),
            backend="lmstudio",
            ram_gb=0.0,
            context_len=m.get("context_length", 0),
            loaded_since=0.0,
            expires_at=0.0,
            quant="",
            family="",
        ))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_running_models() -> list[RunningModel]:
    """
    Return every LLM currently resident in memory across all local backends.

    Calls each backend probe sequentially (all have short timeouts).
    Returns an empty list if no backends are reachable.
    """
    models: list[RunningModel] = []
    models.extend(_probe_ollama())
    models.extend(_probe_mlx())
    models.extend(_probe_lmstudio())
    return models
