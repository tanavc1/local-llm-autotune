"""
Local storage preference — controls whether autotune writes performance data
and telemetry events to the local SQLite database.

Default: ENABLED (opt-out, unlike cloud telemetry which is opt-in).
Model metadata fetched from HuggingFace is always stored regardless of this
setting — it is reference data, not behavioral data.

Preference file location:
  macOS:   ~/Library/Application Support/autotune/storage_prefs.json
  Linux:   ~/.local/share/autotune/storage_prefs.json
  Windows: %APPDATA%/autotune/storage_prefs.json

File contents (JSON):
  {"enabled": true, "set_at": "<ISO 8601 timestamp>"}
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path helpers (mirrors autotune/telemetry/consent.py layout)
# ---------------------------------------------------------------------------

def _config_dir() -> Path:
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "autotune"
    elif platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home())) / "autotune"
    else:
        base = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        ) / "autotune"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _prefs_file() -> Path:
    return _config_dir() / "storage_prefs.json"


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _load() -> Optional[dict]:
    path = _prefs_file()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def is_storage_enabled() -> bool:
    """
    Return True if local SQLite storage is enabled.

    Defaults to True — the preference file only needs to exist when the user
    has explicitly disabled storage.
    """
    data = _load()
    if data is None:
        return True          # default: on
    return bool(data.get("enabled", True))


def storage_pref_set() -> bool:
    """Return True if the user has explicitly set a storage preference."""
    return _prefs_file().exists()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def set_storage_enabled(enabled: bool) -> None:
    """Persist the storage preference. Idempotent."""
    payload = {
        "enabled": enabled,
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _prefs_file()
    try:
        path.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass  # non-fatal
