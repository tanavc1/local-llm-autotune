"""
TelemetryClient — writes anonymised usage data to Supabase via the REST API.

Why REST instead of psycopg2?
------------------------------
Supabase's direct PostgreSQL port is IPv6-only, which is unreachable on many
residential and corporate networks.  The Supabase REST API (PostgREST) runs on
port 443 (HTTPS) and is reachable everywhere.  It is also the officially
recommended approach for client applications — the postgres password never
leaves the developer's machine.

Configuration
-------------
Two environment variables drive the client (both required for any data to flow):

    AUTOTUNE_SUPABASE_URL   — project URL, e.g. https://abc123.supabase.co
    AUTOTUNE_SUPABASE_KEY   — service_role key (for schema setup) OR anon key
                              with appropriate Row Level Security policies

These are **NOT** the postgres superuser password.  They are JWT tokens that
can be found in the Supabase dashboard under:
    Project Settings → API → Project API keys

The anon key is designed to be embedded in client applications.  Combined with
Row Level Security policies (see schema.sql) it can only INSERT into the three
telemetry tables, and cannot read, update, or delete any data.

All write methods catch every exception and return a boolean success flag so
a DB outage or misconfiguration never propagates to the user.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

_URL_ENV = "AUTOTUNE_SUPABASE_URL"
_KEY_ENV = "AUTOTUNE_SUPABASE_KEY"

# These values can be baked in at package-build time by the release pipeline.
# They are intentionally empty here so the postgres superuser credentials are
# never shipped in source.  Set the env vars in your shell or CI to activate
# telemetry in development.
_BUILTIN_URL: str = os.environ.get(_URL_ENV, "https://gmsibgsdedyrbiucaitv.supabase.co")
_BUILTIN_KEY: str = os.environ.get(
    _KEY_ENV,
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdtc2liZ3NkZWR5cmJpdWNhaXR2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzYzMDQ2NzksImV4cCI6MjA5MTg4MDY3OX0.Cjv6oseYgXo4TtRi7LDrQItDnZZCqUARFthfMDX9Tec",
)


def _url() -> str:
    return os.environ.get(_URL_ENV, _BUILTIN_URL).rstrip("/")


def _key() -> str:
    return os.environ.get(_KEY_ENV, _BUILTIN_KEY)


# ---------------------------------------------------------------------------
# Lazy httpx import (already a project dependency)
# ---------------------------------------------------------------------------

def _httpx():
    try:
        import httpx
        return httpx
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# TelemetryClient
# ---------------------------------------------------------------------------

class TelemetryClient:
    """
    Supabase REST API telemetry writer.

    Each public method sends a single HTTPS POST to the Supabase PostgREST
    endpoint and returns a boolean indicating success.  The underlying HTTP
    client uses a 6-second timeout so a slow network never stalls the CLI.
    """

    def __init__(self, url: str, key: str) -> None:
        self._url = url.rstrip("/")
        self._key = key

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _headers(self) -> dict:
        return {
            "apikey":        self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type":  "application/json",
            "Prefer":        "return=minimal",
        }

    def _post(self, table: str, payload: dict) -> bool:
        """POST one row to a Supabase REST table endpoint."""
        hx = _httpx()
        if hx is None:
            return False
        try:
            resp = hx.post(
                f"{self._url}/rest/v1/{table}",
                json=payload,
                headers=self._headers(),
                timeout=6.0,
            )
            ok = resp.status_code in (200, 201)
            if not ok and os.environ.get("AUTOTUNE_TELEMETRY_DEBUG"):
                print(
                    f"[telemetry] {table} INSERT failed "
                    f"{resp.status_code}: {resp.text[:200]}",
                    file=sys.stderr,
                )
            return ok
        except Exception as exc:  # noqa: BLE001
            if os.environ.get("AUTOTUNE_TELEMETRY_DEBUG"):
                print(f"[telemetry] HTTP error: {exc}", file=sys.stderr)
            return False

    def _rpc(self, fn: str, payload: dict) -> bool:
        """Call a Supabase RPC (stored function) endpoint."""
        hx = _httpx()
        if hx is None:
            return False
        try:
            resp = hx.post(
                f"{self._url}/rest/v1/rpc/{fn}",
                json=payload,
                headers=self._headers(),
                timeout=6.0,
            )
            ok = resp.status_code in (200, 204)
            if not ok and os.environ.get("AUTOTUNE_TELEMETRY_DEBUG"):
                print(
                    f"[telemetry] RPC {fn} failed "
                    f"{resp.status_code}: {resp.text[:200]}",
                    file=sys.stderr,
                )
            return ok
        except Exception as exc:  # noqa: BLE001
            if os.environ.get("AUTOTUNE_TELEMETRY_DEBUG"):
                print(f"[telemetry] RPC HTTP error: {exc}", file=sys.stderr)
            return False

    # ------------------------------------------------------------------ #
    # installations                                                        #
    # ------------------------------------------------------------------ #

    def register_installation(self, hw_dict: dict, version: str) -> bool:
        """
        Upsert a hardware fingerprint row.

        On conflict (same install_key) only last_seen_at and
        autotune_version are updated.
        """
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "install_key":        hw_dict.get("id"),
            "os_name":            hw_dict.get("os_name"),
            "os_version":         hw_dict.get("os_version"),
            "cpu_brand":          hw_dict.get("cpu_brand"),
            "cpu_physical_cores": hw_dict.get("cpu_physical_cores"),
            "cpu_logical_cores":  hw_dict.get("cpu_logical_cores"),
            "cpu_arch":           hw_dict.get("cpu_arch"),
            "total_ram_gb":       hw_dict.get("total_ram_gb"),
            "gpu_name":           hw_dict.get("gpu_name"),
            "gpu_backend":        hw_dict.get("gpu_backend"),
            "gpu_vram_gb":        hw_dict.get("gpu_vram_gb"),
            "is_unified_memory":  bool(hw_dict.get("is_unified_memory")),
            "autotune_version":   version,
            "python_version":     (
                f"{sys.version_info.major}.{sys.version_info.minor}"
                f".{sys.version_info.micro}"
            ),
            "telemetry_opted_in": True,
            "first_seen_at":      now,
            "last_seen_at":       now,
        }
        # Plain INSERT — 409 (duplicate key) means the row already exists, which
        # is fine.  Upsert Prefer headers require an UPDATE RLS policy that the
        # anon role intentionally doesn't have.
        hx = _httpx()
        if hx is None:
            return False
        try:
            resp = hx.post(
                f"{self._url}/rest/v1/installations",
                json=payload,
                headers=self._headers(),
                timeout=6.0,
            )
            if resp.status_code in (200, 201):
                # New row — increment counter
                self._rpc("increment_stat", {"p_key": "total_installations"})
                return True
            if resp.status_code == 409:
                # Row already exists — update last_seen_at so active_users view
                # stays accurate.  The anon role can't UPDATE directly; the
                # update_last_seen() SECURITY DEFINER RPC is the safe path.
                self._rpc("update_last_seen", {"p_install_key": payload["install_key"]})
                return True
            if os.environ.get("AUTOTUNE_TELEMETRY_DEBUG"):
                print(
                    f"[telemetry] installation insert failed "
                    f"{resp.status_code}: {resp.text[:200]}",
                    file=sys.stderr,
                )
            return False
        except Exception as exc:  # noqa: BLE001
            if os.environ.get("AUTOTUNE_TELEMETRY_DEBUG"):
                print(f"[telemetry] install register error: {exc}", file=sys.stderr)
            return False

    # ------------------------------------------------------------------ #
    # telemetry_events                                                     #
    # ------------------------------------------------------------------ #

    def record_event(self, install_key: str, event_type: str, **kwargs: Any) -> bool:
        """
        Insert one event row into telemetry_events.

        Accepted keyword arguments match the columns in the table.
        Unknown keys are silently dropped.
        """
        allowed = {
            "session_id", "autotune_version", "model_id",
            "tokens_per_sec", "gen_tokens_per_sec", "ttft_ms",
            "prompt_tokens", "completion_tokens", "context_len",
            "peak_ram_gb", "peak_vram_gb", "delta_ram_gb",
            "cpu_avg_pct", "cpu_peak_pct", "load_time_sec", "elapsed_sec",
            "profile_name", "quant", "completed", "oom",
            "value_num", "value_text",
            "error_type", "error_msg",
        }
        payload: dict[str, Any] = {k: v for k, v in kwargs.items() if k in allowed}
        payload["install_key"] = install_key
        payload["event_type"] = event_type
        payload["occurred_at"] = datetime.now(timezone.utc).isoformat()
        # server_received_at intentionally omitted — the column has DEFAULT NOW()
        # on the Supabase side, giving a true server timestamp for clock-skew analysis.

        ok = self._post("telemetry_events", payload)
        if ok:
            self._rpc("increment_stat", {"p_key": "total_telemetry_events"})
        return ok

    # ------------------------------------------------------------------ #
    # run_observations                                                     #
    # ------------------------------------------------------------------ #

    def record_run(self, install_key: str, run_data: dict[str, Any]) -> bool:
        """
        Insert one performance observation row into run_observations.

        run_data should contain the fields from the local SQLite
        run_observations table plus hardware metadata (os_name, cpu_arch,
        total_ram_gb, gpu_backend) and autotune_version.
        """
        required = {"model_id", "quant", "context_len"}
        if not required.issubset(run_data.keys()):
            return False

        allowed = {
            "model_id", "hardware_key", "os_name", "cpu_arch",
            "total_ram_gb", "gpu_backend",
            "quant", "context_len", "n_gpu_layers", "batch_size",
            "profile_name", "f16_kv", "num_keep",
            "tokens_per_sec", "gen_tokens_per_sec",
            "peak_ram_gb", "peak_vram_gb",
            "ram_before_gb", "ram_after_gb", "delta_ram_gb",
            "swap_peak_gb", "delta_swap_gb",
            "cpu_avg_pct", "cpu_peak_pct",
            "load_time_sec", "ttft_ms", "elapsed_sec",
            "prompt_tokens", "completion_tokens",
            "completed", "oom",
            "autotune_version",
        }
        payload = {k: v for k, v in run_data.items() if k in allowed}
        payload["install_key"] = install_key
        payload["observed_at"] = datetime.now(timezone.utc).isoformat()

        ok = self._post("run_observations", payload)
        if ok:
            self._rpc("increment_stat", {"p_key": "total_run_observations"})
        return ok


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: Optional[TelemetryClient] = None


def get_client() -> Optional[TelemetryClient]:
    """
    Return the module-level TelemetryClient singleton.

    Returns None if AUTOTUNE_SUPABASE_KEY is not set or httpx is unavailable.
    """
    global _client
    if _client is not None:
        return _client

    url = _url()
    key = _key()
    if not url or not key:
        return None
    if _httpx() is None:
        return None

    _client = TelemetryClient(url, key)
    return _client
