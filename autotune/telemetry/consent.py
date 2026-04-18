"""
Opt-in consent management — reads and writes the user's telemetry preference.

The preference is stored in:
  macOS:   ~/Library/Application Support/autotune/telemetry_consent.json
  Linux:   ~/.local/share/autotune/telemetry_consent.json
  Windows: %APPDATA%/autotune/telemetry_consent.json

File contents (JSON):
  {
    "opted_in": true,
    "install_key": "<16-char hex fingerprint>",
    "answered_at": "<ISO 8601 timestamp>"
  }
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _config_dir() -> Path:
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "autotune"
    elif platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home())) / "autotune"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "autotune"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _consent_file() -> Path:
    return _config_dir() / "telemetry_consent.json"


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _load() -> Optional[dict]:
    path = _consent_file()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def consent_answered() -> bool:
    """Return True if the user has already answered the opt-in prompt."""
    return _load() is not None


def is_opted_in() -> bool:
    """Return True only if the user has explicitly consented."""
    data = _load()
    return bool(data and data.get("opted_in"))


def get_install_key() -> Optional[str]:
    """Return the cached hardware fingerprint, or derive it fresh."""
    data = _load()
    if data and data.get("install_key"):
        return data["install_key"]
    return _derive_install_key()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def set_consent(opted_in: bool) -> None:
    """Persist the user's answer. Idempotent."""
    install_key = _derive_install_key()
    payload = {
        "opted_in": opted_in,
        "install_key": install_key,
        "answered_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _consent_file()
    try:
        path.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass  # non-fatal — consent just won't persist


# ---------------------------------------------------------------------------
# Interactive prompt
# ---------------------------------------------------------------------------

_BANNER = """\
─────────────────────────────────────────────────────────────────
  autotune  ·  Anonymous Usage Telemetry  (opt-in)
─────────────────────────────────────────────────────────────────
  Help improve autotune by sharing anonymous performance data:

    • Which LLM models you run and their speed / memory usage
    • Your hardware class (CPU arch, RAM, GPU type) — no serial
      numbers, hostnames, usernames, or file paths
    • Crash and OOM events so we can fix them faster

  Data is sent to a private Supabase database.  We never sell or
  share it.  Opt out at any time:
      autotune telemetry --disable

  Check status:
      autotune telemetry --status

  Full details: https://github.com/tanavc1/local-llm-autotune
─────────────────────────────────────────────────────────────────"""


def prompt_opt_in() -> bool:
    """
    Show the opt-in banner and ask the user Y/N.

    Returns True if the user consents, False otherwise.
    Also persists the answer via set_consent().
    """
    print(_BANNER)
    try:
        answer = input("  Share anonymous telemetry? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    opted_in = answer in ("y", "yes")
    set_consent(opted_in)

    if opted_in:
        print("  ✓ Telemetry enabled — thank you!\n")
    else:
        print("  ✗ Telemetry disabled — no data will be sent.\n")

    return opted_in


# ---------------------------------------------------------------------------
# Hardware fingerprint helper
# ---------------------------------------------------------------------------

def _derive_install_key() -> str:
    """
    Return a stable 16-char hex fingerprint for this machine.

    Falls back to a machine-architecture hash if the hardware profiler is
    unavailable.  The fallback never includes hostnames, usernames, or any
    PII — only CPU architecture, OS family, and machine type.
    """
    try:
        from autotune.hardware.profiler import profile_hardware
        from autotune.db.fingerprint import hardware_id
        hw = profile_hardware()
        return hardware_id(hw)
    except Exception:  # noqa: BLE001
        import hashlib
        # Use only anonymous hardware/OS identifiers — no hostname, no username.
        seed = "|".join([
            platform.system(),       # "Darwin", "Linux", "Windows"
            platform.machine(),      # "arm64", "x86_64"
            platform.processor(),    # "arm", "Intel Core i9-..."
        ])
        return hashlib.sha256(seed.encode()).hexdigest()[:16]
