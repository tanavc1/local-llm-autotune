"""
Telemetry event type constants and a lightweight dataclass for event payloads.

EventType values
----------------
session_start       autotune server came online
session_end         autotune server shut down cleanly
run_complete        a chat/completion request finished
model_loaded        a model was loaded into memory
recommendation_run  `autotune recommend` command executed
error               an unhandled exception was caught
oom_near            RAM pressure crossed the warning threshold (>90 %)
ram_spike           instantaneous RAM delta > 1 GB in 250 ms window
swap_spike          swap usage climbed > 500 MB
cpu_peak            CPU hit > 95 % sustained for > 5 s
kv_reduced          KV context was trimmed to avoid OOM
opt_in              user enabled telemetry
opt_out             user disabled telemetry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


class EventType:
    SESSION_START        = "session_start"
    SESSION_END          = "session_end"
    RUN_COMPLETE         = "run_complete"
    MODEL_LOADED         = "model_loaded"
    RECOMMENDATION_RUN   = "recommendation_run"
    ERROR                = "error"
    OOM_NEAR             = "oom_near"
    RAM_SPIKE            = "ram_spike"
    SWAP_SPIKE           = "swap_spike"
    CPU_PEAK             = "cpu_peak"
    KV_REDUCED           = "kv_reduced"
    OPT_IN               = "opt_in"
    OPT_OUT              = "opt_out"


@dataclass
class TelemetryEvent:
    """Typed container for a single telemetry event."""

    install_key:         str
    event_type:          str
    autotune_version:    Optional[str] = None
    session_id:          Optional[str] = None

    # Model context
    model_id:            Optional[str] = None

    # Inference performance (run_complete)
    tokens_per_sec:      Optional[float] = None
    gen_tokens_per_sec:  Optional[float] = None
    ttft_ms:             Optional[float] = None
    prompt_tokens:       Optional[int]   = None
    completion_tokens:   Optional[int]   = None
    context_len:         Optional[int]   = None
    peak_ram_gb:         Optional[float] = None
    peak_vram_gb:        Optional[float] = None
    delta_ram_gb:        Optional[float] = None
    cpu_avg_pct:         Optional[float] = None
    cpu_peak_pct:        Optional[float] = None
    load_time_sec:       Optional[float] = None
    elapsed_sec:         Optional[float] = None
    profile_name:        Optional[str]   = None
    quant:               Optional[str]   = None
    completed:           bool            = True
    oom:                 bool            = False

    # System event (pressure / spike events)
    value_num:           Optional[float] = None
    value_text:          Optional[str]   = None

    # Error info
    error_type:          Optional[str]   = None
    error_msg:           Optional[str]   = None

    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Return a dict ready for insertion into telemetry_events."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
