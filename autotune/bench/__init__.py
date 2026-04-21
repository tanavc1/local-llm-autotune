"""
autotune.bench — Inference benchmarking with hardware sampling.

Modules
-------
runner.py
    Core benchmark loop.  Two call paths:

    run_raw_ollama()          — Ollama defaults, zero autotune settings.
                                Use as the baseline in every comparison.

    run_bench_ollama_only()   — autotune/balanced settings applied directly to
                                Ollama HTTP.  Uses TTFTOptimizer for TTFT options
                                + HardwareTuner for OS-level scheduling.
                                Skips BackendChain so MLX is never attempted —
                                correct for benchmarking Ollama-native models.

    run_bench()               — Full BackendChain path (MLX → Ollama → LM Studio).
                                Use when testing the complete autotune stack.

    BenchResult               — Dataclass capturing all metrics per call:
                                TTFT, tok/s, RAM before/peak/after, swap,
                                CPU avg/peak, num_ctx, KV precision.

compare.py
    Side-by-side comparison helpers for DB-stored runs.

user_metrics.py
    User-experience KPI framework.  Reports what users actually feel:
    swap_events, ram_headroom_gb, ttft_consistency_pct, cpu_spike_events,
    memory_recovery_sec, background_impact_score.

Scripts (not in this package — see /scripts/)
---------------------------------------------
scripts/user_bench.py      User-experience benchmark (7 user KPIs, head-to-head
                           autotune vs raw Ollama, desktop notification when done).
scripts/stress_test.py     Comprehensive multi-phase stress test (6 phases,
                           63+ calls, raw vs autotune comparison).
scripts/proof_suite.py     11-KPI technical benchmark (TTFT, KV cache, RAM, swap).
scripts/agent_bench.py     Multi-turn agentic task benchmark.
scripts/benchmark.py       Quick 4-prompt × 3-run single-model benchmark.
"""
