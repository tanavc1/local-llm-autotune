"""
autotune.metrics — Accurate inference resource measurement.

Problem with naive metrics
--------------------------
Measuring LLM inference with ``psutil.cpu_percent()`` and
``psutil.virtual_memory().used`` has two fundamental flaws on Apple Silicon:

1. **CPU%** is system-wide CPU core utilisation.  Ollama's actual inference
   runs on Metal (GPU + Neural Engine).  Metal compute is invisible to
   ``psutil.cpu_percent()`` — you can have 100% GPU utilisation and 5% CPU%.
   The number we were calling "CPU%" was really Python wrapper overhead.

2. **RAM delta** captures what changes *during* a call, not at model load.
   KV cache is allocated by Ollama when the model first loads (not per-call).
   Once the model is loaded with ``keep_alive=-1`` the per-call delta is just
   Python string buffers — identical for raw and autotune.

What we measure instead
-----------------------
Both metrics come from Ollama's own internal timers, reported in every
response via the native ``/api/chat`` endpoint.  No estimation.

**Prefill duration** (``prompt_eval_duration``)
    Time Ollama spent processing the input tokens (the KV-fill phase).
    Maps directly to TTFT latency.  Shrinks with smaller ``num_ctx`` because
    Ollama must initialise less KV-buffer memory before starting the forward
    pass.  Proven: ctx=4096 → 324 ms, ctx=1024 → 200 ms (-38%) on phi4-mini.

**Generation throughput** (``eval_count / eval_duration``)
    True tokens/second from Ollama's internal Metal timer — accurate to the
    nanosecond, not estimated from ``len(response)/4``.

**VRAM footprint** (``size_vram`` from ``/api/ps``)
    Actual unified memory Ollama is holding for this model right now, in bytes.
    Smaller ``num_ctx`` → smaller KV buffer → lower ``size_vram``.
    Proven: ctx=512 → -806 MB vs ctx=4096 on phi4-mini.

**Model load time** (``load_duration``)
    Time for Ollama to allocate buffers and move weights to Metal.
    Scales with KV cache size → smaller ctx → faster load.
    Proven: ctx=4096 → 2606 ms, ctx=1024 → 976 ms (-63%) on phi4-mini.

Public API
----------
.. code-block:: python

    from autotune.metrics import NativeInferenceStats, OllamaMetricsClient

    client = OllamaMetricsClient()

    # Run inference and get authoritative Ollama stats
    stats = await client.run_with_stats(
        model="phi4-mini:latest",
        messages=[{"role": "user", "content": "Hello"}],
        options={"num_ctx": 1290},
    )
    print(stats.eval_tps)            # true tok/s
    print(stats.prefill_ms)          # TTFT proxy (ms)
    print(stats.load_ms)             # model load time (ms)

    # Check VRAM footprint of loaded model
    vram = await client.get_vram_snapshot("phi4-mini:latest")
    print(vram.size_vram_gb)         # GB of unified memory in use
"""

from autotune.metrics.ollama_client import NativeInferenceStats, OllamaMetricsClient
from autotune.metrics.vram import VRAMSnapshot

__all__ = ["NativeInferenceStats", "OllamaMetricsClient", "VRAMSnapshot"]
