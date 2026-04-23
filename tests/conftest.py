"""
Shared fixtures and helpers for the autotune test suite.

Every test file can import from this module via the standard pytest conftest
mechanism — no explicit import needed.

Fixture inventory
-----------------
make_profile        — factory for Profile objects with overrideable fields
mock_vm             — patch psutil.virtual_memory to a given RAM % level
mock_swap           — patch psutil.swap_memory to a given swap state
sample_messages     — short conversation list (system + 2 turns)
long_messages       — 10-turn conversation for context-window tests
tool_messages       — conversation containing tool/function call turns
code_messages       — conversation with code blocks and stack traces
temp_db             — isolated SQLite DB path in tmp_path
temp_recall_db      — fresh RecallStore backed by tmp_path
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Profile factory
# ---------------------------------------------------------------------------

def make_profile(
    *,
    name: str = "balanced",
    label: str = "Balanced",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    max_context_tokens: int = 8192,
    system_prompt_cache: bool = True,
    qos_class: str = "USER_INITIATED",
    preferred_quants: Optional[list] = None,
    kv_cache_precision: str = "f16",
    backend_preference: Optional[list] = None,
    ollama_keep_alive: str = "-1m",
    request_timeout_sec: float = 120.0,
    speculative_decoding: bool = False,
    flash_attention: bool = True,
):
    """Build a Profile dataclass with sensible defaults."""
    from autotune.api.profiles import Profile
    return Profile(
        name=name,
        label=label,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_context_tokens=max_context_tokens,
        system_prompt_cache=system_prompt_cache,
        qos_class=qos_class,
        preferred_quants=preferred_quants or ["Q4_K_M", "Q5_K_M"],
        kv_cache_precision=kv_cache_precision,
        backend_preference=backend_preference or ["mlx", "ollama"],
        ollama_keep_alive=ollama_keep_alive,
        request_timeout_sec=request_timeout_sec,
        speculative_decoding=speculative_decoding,
        flash_attention=flash_attention,
    )


# ---------------------------------------------------------------------------
# psutil mocks
# ---------------------------------------------------------------------------

def _mock_vm(total_gb: float, used_pct: float, swap_gb: float = 0.0):
    """
    Return a mock psutil virtual_memory() namedtuple-like object.

    used_pct is the percent field (0–100).  available is derived from total.
    """
    total_b = int(total_gb * 1024**3)
    used_b  = int(total_b * used_pct / 100)
    avail_b = total_b - used_b
    m = MagicMock()
    m.total     = total_b
    m.used      = used_b
    m.available = avail_b
    m.percent   = used_pct
    return m


def _mock_sw(used_gb: float = 0.0, total_gb: float = 8.0):
    """Return a mock psutil swap_memory() object."""
    total_b = int(total_gb * 1024**3)
    used_b  = int(used_gb * 1024**3)
    m = MagicMock()
    m.total   = total_b
    m.used    = used_b
    m.percent = (used_b / max(total_b, 1)) * 100
    return m


@pytest.fixture
def mock_ram_normal():
    """RAM at 60% — no pressure."""
    with patch("psutil.virtual_memory", return_value=_mock_vm(16, 60.0)):
        with patch("psutil.swap_memory", return_value=_mock_sw(0.0)):
            yield


@pytest.fixture
def mock_ram_moderate():
    """RAM at 82% — moderate pressure threshold."""
    with patch("psutil.virtual_memory", return_value=_mock_vm(16, 82.0)):
        with patch("psutil.swap_memory", return_value=_mock_sw(0.0)):
            yield


@pytest.fixture
def mock_ram_high():
    """RAM at 89% — high pressure threshold."""
    with patch("psutil.virtual_memory", return_value=_mock_vm(16, 89.0)):
        with patch("psutil.swap_memory", return_value=_mock_sw(0.5)):
            yield


@pytest.fixture
def mock_ram_critical():
    """RAM at 94% — critical pressure threshold."""
    with patch("psutil.virtual_memory", return_value=_mock_vm(16, 94.0)):
        with patch("psutil.swap_memory", return_value=_mock_sw(2.0)):
            yield


# ---------------------------------------------------------------------------
# Message fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_messages():
    """Short 2-turn conversation (system + 1 user/assistant pair)."""
    return [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
    ]


@pytest.fixture
def long_messages():
    """10-turn conversation for context-window budget tests."""
    msgs = [{"role": "system", "content": "You are a helpful Python expert."}]
    for i in range(10):
        msgs.append({"role": "user",      "content": f"Question {i}: How do I do X in Python? " * 8})
        msgs.append({"role": "assistant", "content": f"Answer {i}: Here is a detailed explanation " * 20})
    return msgs


@pytest.fixture
def tool_messages():
    """Conversation with tool call results."""
    return [
        {"role": "system",    "content": "You are an assistant with tools."},
        {"role": "user",      "content": "Check the weather in New York."},
        {"role": "assistant", "content": "I'll check the weather for you."},
        {"role": "tool",      "content": "temperature: 72°F, humidity: 65%, sky: partly cloudy"},
        {"role": "assistant", "content": "The weather in New York is 72°F and partly cloudy."},
    ]


@pytest.fixture
def code_messages():
    """Conversation with code blocks, stack traces, and technical content."""
    return [
        {"role": "system", "content": "You are a debugging assistant."},
        {
            "role": "user",
            "content": (
                "I'm getting this error:\n"
                "Traceback (most recent call last):\n"
                "  File 'app.py', line 42, in main\n"
                "    result = db.execute(query)\n"
                "sqlite3.OperationalError: no such table: users\n"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The error means the `users` table doesn't exist.\n\n"
                "```python\n"
                "import sqlite3\n"
                "conn = sqlite3.connect('app.db')\n"
                "conn.execute('''\n"
                "    CREATE TABLE IF NOT EXISTS users (\n"
                "        id INTEGER PRIMARY KEY,\n"
                "        name TEXT NOT NULL\n"
                "    )\n"
                "''')\n"
                "conn.commit()\n"
                "```\n\n"
                "Run this migration before your main app starts."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Isolated SQLite DB file path in a temp directory."""
    return tmp_path / "test_autotune.db"


@pytest.fixture
def temp_recall_db(tmp_path: Path):
    """Fresh RecallStore backed by a temp SQLite file."""
    from autotune.recall.store import RecallStore
    store = RecallStore(tmp_path / "recall.db")
    yield store


@pytest.fixture
def temp_user_config(tmp_path: Path, monkeypatch):
    """
    Redirect user_config._config_file() to a temp path so tests
    never touch the real ~/Library/… config.
    """
    cfg_path = tmp_path / "user_config.json"
    monkeypatch.setattr(
        "autotune.config.user_config._config_file",
        lambda: cfg_path,
    )
    return cfg_path


# ---------------------------------------------------------------------------
# Ollama integration fixtures (session-scoped, skips when Ollama absent)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ollama_running():
    """
    Skip the entire test module if Ollama is not reachable.
    Returns the list of available models.
    """
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        r.raise_for_status()
        return r.json().get("models", [])
    except Exception:
        pytest.skip("Ollama not running — skipping integration tests")


@pytest.fixture(scope="session")
def smallest_ollama_model(ollama_running):
    """Return the name of the smallest installed Ollama model."""
    models = ollama_running
    if not models:
        pytest.skip("No Ollama models installed")
    return min(models, key=lambda m: m.get("size", float("inf")))["name"]
