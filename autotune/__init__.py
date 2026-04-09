"""
autotune — local-LLM inference optimiser.

Package layout
--------------
ttft/           ← START HERE if you care about latency
                  Owns all three TTFT-reduction mechanisms:
                  dynamic num_ctx, keep_alive=-1, num_keep prefix caching.
                  Benchmark: −44% average TTFT vs raw Ollama defaults.

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
"""

__version__ = "0.1.0"
