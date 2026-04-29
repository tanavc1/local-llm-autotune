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


def ollama_base() -> str:
    """Return the Ollama base URL (no trailing slash).

    Reads AUTOTUNE_OLLAMA_URL; falls back to http://localhost:11434.
    """
    url = os.environ.get("AUTOTUNE_OLLAMA_URL", "http://localhost:11434")
    return url.rstrip("/")
