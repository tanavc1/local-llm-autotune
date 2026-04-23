"""
autotune.telemetry — opt-in anonymous usage telemetry.

This module is entirely opt-in.  Nothing is sent unless the user explicitly
agrees to participate.  No personally-identifiable information is collected;
the sole identifier is a sha-256 hardware fingerprint generated from
CPU brand, core count, total RAM, and GPU name.

Public API
----------
    from autotune.telemetry import maybe_prompt_consent, emit, register_install

    # Call once at the start of any CLI command to surface the opt-in prompt:
    maybe_prompt_consent()

    # Fire-and-forget event (silently skipped if not opted in or DB unavailable):
    emit("session_start", model_id="qwen3:8b", autotune_version="0.1.1")
"""

from __future__ import annotations

import sys
from typing import Any

from autotune.telemetry.client import get_client
from autotune.telemetry.consent import get_install_key, is_opted_in, prompt_opt_in
from autotune.telemetry.events import EventType

__all__ = [
    "maybe_prompt_consent",
    "emit",
    "register_install",
    "EventType",
    "is_opted_in",
    "get_install_key",
]


def maybe_prompt_consent() -> None:
    """
    Display the opt-in prompt exactly once — on the first run of any command.

    After the user answers (Y/N) the choice is persisted to disk and this
    function becomes a no-op on all future invocations.
    """
    from autotune.telemetry.consent import consent_answered

    if consent_answered():
        return

    # Only show the prompt when running interactively (not piped / in tests)
    if not sys.stdin.isatty():
        return

    opted_in = prompt_opt_in()

    if opted_in:
        _register_install_background()


def emit(event_type: str, **kwargs: Any) -> None:
    """
    Fire-and-forget telemetry event.

    Silently dropped if:
      • the user has not opted in
      • psycopg2 is not installed
      • the Supabase DB is unreachable

    Parameters
    ----------
    event_type : str
        One of the EventType constants (e.g. "session_start", "run_complete").
    **kwargs
        Event payload fields — see TelemetryClient.record_event for the full
        set of accepted keyword arguments.
    """
    if not is_opted_in():
        return
    try:
        client = get_client()
        if client is None:
            return
        install_key = get_install_key()
        client.record_event(install_key, event_type, **kwargs)
    except Exception:  # noqa: BLE001 — telemetry must never crash the host process
        pass


def register_install() -> None:
    """
    Push installation metadata to Supabase (called once, after consent).

    Safe to call multiple times — INSERT ignores duplicate install_key.
    Silently skipped if the user has not opted in.
    """
    if not is_opted_in():
        return
    try:
        client = get_client()
        if client is None:
            return
        import autotune
        from autotune.db.fingerprint import hardware_to_db_dict
        from autotune.hardware.profiler import profile_hardware

        hw = profile_hardware()
        hw_dict = hardware_to_db_dict(hw)
        client.register_installation(hw_dict, autotune.__version__)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _register_install_background() -> None:
    """Register install in a daemon thread so the CLI prompt returns quickly."""
    import threading
    t = threading.Thread(target=register_install, daemon=True)
    t.start()
