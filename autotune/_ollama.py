"""
Shared Ollama base URL resolution.

Set AUTOTUNE_OLLAMA_URL to override the default localhost connection.
This is needed for Docker Compose (multi-container) and remote Ollama setups.

Examples
--------
  # Docker Compose — autotune container reaching a separate ollama service:
  AUTOTUNE_OLLAMA_URL=http://ollama:11434

  # Remote Ollama on another machine:
  AUTOTUNE_OLLAMA_URL=http://192.168.1.50:11434
"""

import os
import platform


def ollama_base() -> str:
    """Return the Ollama base URL (no trailing slash).

    Reads AUTOTUNE_OLLAMA_URL; falls back to http://127.0.0.1:11434 on
    Windows and http://localhost:11434 elsewhere.

    Windows note: localhost can resolve to ::1 (IPv6) before 127.0.0.1
    (IPv4).  Ollama on Windows listens on IPv4 only, so using the explicit
    IPv4 address avoids silent connection failures when no env var is set.
    """
    url = os.environ.get("AUTOTUNE_OLLAMA_URL", "")
    if not url:
        url = (
            "http://127.0.0.1:11434"
            if platform.system() == "Windows"
            else "http://localhost:11434"
        )
    return url.rstrip("/")
