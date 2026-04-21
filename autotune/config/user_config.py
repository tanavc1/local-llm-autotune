"""
User-facing persistent configuration for autotune.

Stores defaults like the model to use, inference profile, server port, etc.
Located alongside storage_prefs.json in the platform config dir.

  macOS:   ~/Library/Application Support/autotune/user_config.json
  Linux:   ~/.local/share/autotune/user_config.json
  Windows: %APPDATA%/autotune/user_config.json
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Keys the user is allowed to set, with (type_hint, description, default)
KNOWN_KEYS: dict[str, tuple[str, str, Any]] = {
    "default_model":   ("str",  "Default model for `autotune chat` / `autotune run`", None),
    "default_profile": ("str",  "Default profile: fast | balanced | quality",         "balanced"),
    "serve_host":      ("str",  "Default host for `autotune serve`",                  "127.0.0.1"),
    "serve_port":      ("int",  "Default port for `autotune serve`",                  8765),
}


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


def _config_file() -> Path:
    return _config_dir() / "user_config.json"


def load_config() -> dict:
    path = _config_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(data: dict) -> None:
    data["_updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        _config_file().write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def get_value(key: str) -> Optional[Any]:
    """Return the stored value for key, or None if not set."""
    return load_config().get(key)


def set_value(key: str, value: str) -> tuple[bool, str]:
    """
    Validate and store a key=value pair.

    Returns (success, error_message).  On success error_message is "".
    """
    if key not in KNOWN_KEYS:
        known = ", ".join(KNOWN_KEYS)
        return False, f"Unknown key {key!r}. Known keys: {known}"

    type_hint, _, _ = KNOWN_KEYS[key]
    coerced: Any = value

    if type_hint == "int":
        try:
            coerced = int(value)
        except ValueError:
            return False, f"{key} must be an integer, got {value!r}"

    if key == "default_profile" and value not in ("fast", "balanced", "quality"):
        return False, "default_profile must be one of: fast, balanced, quality"

    data = load_config()
    data[key] = coerced
    save_config(data)
    return True, ""


def reset_config() -> None:
    """Remove the config file (resets all keys to defaults)."""
    path = _config_file()
    if path.exists():
        path.unlink()


def effective_default(key: str) -> Any:
    """Return stored value for key if set, otherwise the built-in default."""
    val = get_value(key)
    if val is not None:
        return val
    _, _, default = KNOWN_KEYS[key]
    return default
