"""
autotune — local-LLM inference optimiser.

Package layout
--------------
ttft/           ← START HERE if you care about latency
                  Owns all three TTFT-reduction mechanisms:
                  dynamic num_ctx, keep_alive=-1, num_keep prefix caching.
                  Benchmark: −39% average TTFT vs raw Ollama defaults
                  (cross-model avg: qwen3:8b −53%, gemma3:4b −29%, llama3.2:3b −35%).

api/            Inference pipeline: profiles, FastAPI server, terminal chat,
                KV manager, hardware tuner, model selector, backends.

context/        Intelligent context window management: token budget tiers,
                message compression, fact extraction for long conversations.

bench/          Benchmarking framework: 250ms hardware sampler, DB persistence,
                raw vs autotune comparison runners.

db/             SQLite persistence: run_observations, telemetry_events, hardware
                fingerprints, model stubs.

hardware/       CPU/GPU/RAM detection (psutil + py-cpuinfo + subprocess).

memory/         Model memory estimation: weights + KV cache + runtime overhead.

models/         Model registry: 9 OSS models with real MMLU/HumanEval scores.

config/         Recommendation engine: candidate enumeration, multi-objective
                scoring (stability × speed × quality × context).

session/        Real-time session monitor, adaptive advisor state machine.

hub/            HuggingFace model hub fetcher.

output/         Rich-based terminal formatting (tables, panels, score bars).

Programmatic API (for application developers)
---------------------------------------------
    import autotune

    autotune.start()                   # starts server if not running, blocks until ready
    autotune.is_running()              # True/False — is the server up?
    autotune.stop()                    # stop the server autotune.start() launched
    autotune.client_kwargs()           # {"base_url": ..., "api_key": "local"}

    # Minimal example with any OpenAI-compatible client:
    from openai import OpenAI
    autotune.start()
    client = OpenAI(**autotune.client_kwargs())
    response = client.chat.completions.create(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

__version__ = "0.2.0"

import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level state for the managed subprocess
# ---------------------------------------------------------------------------

_managed_proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
_managed_host: str = "localhost"
_managed_port: int = 8765


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(
    host: str = "localhost",
    port: int = 8765,
    timeout: float = 30.0,
    *,
    profile: Optional[str] = None,
    use_mlx: bool = False,
    log_level: str = "warning",
) -> str:
    """Start the autotune server if it is not already running.

    Blocks until the server is accepting requests or ``timeout`` seconds
    elapse (raises ``TimeoutError`` in that case).

    Parameters
    ----------
    host : str
        Interface to bind. Default ``"localhost"`` — local-only.
    port : int
        Port to bind. Default ``8765``.
    timeout : float
        Seconds to wait for the server to become ready. Default ``30``.
    profile : str | None
        Default optimization profile (``"fast"``, ``"balanced"``, ``"quality"``).
        Sets the ``AUTOTUNE_DEFAULT_PROFILE`` env var for the spawned process.
    use_mlx : bool
        Whether to allow the MLX backend on Apple Silicon. Default ``True``.

        Set to ``False`` for the lightest possible memory footprint (~150 MB
        vs ~470 MB).  When disabled, all requests are routed through Ollama.
        The trade-off is ~10-40% lower token throughput vs native MLX, but
        autotune's KV-cache and TTFT optimisations still apply in full.

        Why ``use_mlx=False`` saves RAM: ``mlx_lm`` loads the HuggingFace
        ``transformers`` tokenizer on first use, which transitively imports
        PyTorch (~250-300 MB RSS).  Disabling MLX routing prevents that import
        entirely.
    log_level : str
        uvicorn log level for the spawned process. Default ``"warning"``
        (quiet — won't pollute your application's stdout).

    Returns
    -------
    str
        Base URL of the server, e.g. ``"http://localhost:8765/v1"``.

    Raises
    ------
    TimeoutError
        If the server does not become ready within ``timeout`` seconds.

    Examples
    --------
    >>> import autotune
    >>> from openai import OpenAI
    >>> autotune.start()                       # default: Ollama only (~94 MB RAM)
    'http://localhost:8765/v1'
    >>> autotune.start(use_mlx=True)           # opt in to MLX on Apple Silicon
    'http://localhost:8765/v1'
    >>> client = OpenAI(**autotune.client_kwargs())
    """
    global _managed_proc, _managed_host, _managed_port

    _managed_host = host
    _managed_port = port
    base_url = f"http://{host}:{port}/v1"

    # Fast path: server already up (started externally or in a previous call).
    if _health_check(host, port):
        return base_url

    # Build the command.  Prefer the installed `autotune` script; fall back to
    # running the CLI module directly so this works in editable-install / venv
    # setups where the script may not be on PATH.
    autotune_bin = shutil.which("autotune")
    if autotune_bin:
        cmd = [autotune_bin, "serve", "--host", host, "--port", str(port)]
    else:
        cmd = [
            sys.executable, "-c",
            "from autotune.cli import cli; cli(standalone_mode=False)",
            "serve", "--host", host, "--port", str(port),
        ]

    import os
    env = os.environ.copy()
    if profile:
        env["AUTOTUNE_DEFAULT_PROFILE"] = profile
    if not use_mlx:
        env["AUTOTUNE_DISABLE_MLX"] = "1"

    # Launch server as a background subprocess.  stdout/stderr are suppressed
    # by default (log_level=warning) so the caller's output isn't polluted.
    _managed_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL if log_level == "warning" else None,
        stderr=subprocess.DEVNULL if log_level == "warning" else None,
    )

    # Poll until ready.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _health_check(host, port):
            return base_url
        # Check if the process died immediately (bad install, port conflict, etc.)
        if _managed_proc.poll() is not None:
            raise RuntimeError(
                f"autotune server process exited with code {_managed_proc.returncode}. "
                f"Run `autotune serve` manually to see the error."
            )
        time.sleep(0.5)

    # Timed out — kill the process we started.
    _managed_proc.terminate()
    _managed_proc = None
    raise TimeoutError(
        f"autotune server did not become ready within {timeout:.0f}s. "
        f"Check that port {port} is free and Ollama is running."
    )


def stop() -> None:
    """Stop the server that was started by :func:`start`.

    Does nothing if the server was not started by this process (e.g. it was
    started externally via ``autotune serve``).
    """
    global _managed_proc
    if _managed_proc is not None and _managed_proc.poll() is None:
        _managed_proc.terminate()
        try:
            _managed_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _managed_proc.kill()
    _managed_proc = None


def is_running(host: str = "localhost", port: int = 8765) -> bool:
    """Return ``True`` if the autotune server is accepting requests.

    Parameters
    ----------
    host, port : str / int
        Where to check. Defaults match :func:`start`.
    """
    return _health_check(host, port)


def client_kwargs(host: str = "localhost", port: int = 8765) -> dict:
    """Return keyword arguments to pass to any OpenAI-compatible client.

    The returned dict has ``base_url`` and ``api_key`` keys.  Pass it with
    ``**`` unpacking to construct your client:

        client = OpenAI(**autotune.client_kwargs())

    The ``api_key`` is set to ``"local"`` — autotune does not validate API
    keys, but most OpenAI client libraries require the field to be non-empty.
    """
    return {
        "base_url": f"http://{host}:{port}/v1",
        "api_key": "local",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _health_check(host: str, port: int) -> bool:
    """Return True if the server /health endpoint responds successfully."""
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/health", timeout=1.0
        ) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False
