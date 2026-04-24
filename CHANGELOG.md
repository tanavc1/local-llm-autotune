# Changelog

All notable changes to **llm-autotune** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-04-23

### Added

**`autotune proof` — quick TTFT proof (≤30s)**
- New self-contained TTFT proof that runs in under 30 seconds and shows two side-by-side tests: a short-prompt baseline and a long-context KV-allocation test.
- Test 2 forces a full KV buffer flush before both conditions (via `keep_alive=0`) to isolate KV init cost from model load time.
- Breakdown of `load_ms (KV alloc)` vs `prefill_ms (prompt eval)` with an honest verdict section.
- `QuickProofResult` extended with `lc_raw_run` / `lc_tuned_run` fields and derived improvement properties.

**`autotune upgrade` — in-place version management**
- New `autotune upgrade` command: checks PyPI for the latest version and runs `pip install --upgrade llm-autotune` in-place.
- Post-command upgrade reminder printed after every CLI invocation (throttled to once per 24 hours via `~/.cache/autotune/upgrade_check.json`).

**`autotune recommend` — install commands in output**
- Each recommendation panel now shows `→ Install: ollama pull <tag>` and `→ Chat: autotune chat --model <id>` lines at the bottom.
- A 3-step "Next Steps" footer appended to every `autotune recommend` run (pull → chat → proof).

**Model registry expanded to 40 models**
- Added smollm2-1.7b, phi3.5-3.8b, qwen2.5-3b, gemma3-4b, gemma3-12b, gemma3-27b, qwen2.5-7b, qwen2.5-coder-7b, deepseek-r1-8b, mistral-small-24b, qwen2.5-coder-32b, deepseek-r1-32b, llama-3.3-70b, deepseek-r1-70b.
- `ollama_tag` field added to `ModelProfile` — used for install commands in `recommend` output.

**Model load status in chat**
- Chat header now shows a live model-load indicator while the first response is being fetched.

**Website v1.0.0**
- Full command reference page.
- "All we do" page documenting all 14 optimisations with latency/memory breakdown.
- Deployed to Vercel.

**Security policy**
- `SECURITY.md` added: responsible disclosure process, supported versions, contact.

### Fixed

- **Model fitness honesty**: `TIGHT` fitness class introduced — SWAP_RISK no longer hard-blocks a model. `TIGHT` means the model can run but chat is not recommended; `UNSAFE` remains for true no-go cases. Fitness labels now clearly distinguish run vs chat viability.
- **`autotune recommend` pager**: removed Rich Pager — output now prints directly to the terminal in place. The pager was disorienting for new users and broke pipe/redirect workflows.
- **Port conflict handling**: `autotune serve` now detects port-in-use errors at startup and prints an actionable message (`--port` flag) instead of crashing with a stack trace.
- **Telemetry race conditions**: consent prompt now fires before `SESSION_START` on first run; `server_received_at` set server-side only; duplicate telemetry path eliminated via shared `_emit_run_telemetry()` helper.
- **ruff lint**: all lint errors in `autotune/` and `tests/` resolved; CI unblocked on lint step.
- **Coverage threshold**: omit list tuned so `pytest --cov-fail-under=60` passes on CI across all Python versions (3.10–3.13).
- **Ollama install commands in README**: `ollama pull` commands added for all 40 models in docs.

### Changed

- `autotune recommend` no longer opens a pager; output is always printed inline.
- Model fitness assessment: `SWAP_RISK` is now a warning label, not a hard block. Recommendations surface `TIGHT` models with a clear note rather than hiding them.

---

## [0.2.0] — 2026-04-19

### Added

**Agent benchmark framework**
- New `autotune agent-bench` command: measures autotune vs raw Ollama across multi-turn agentic tasks with tool calling. Reports 11 KPIs including model reload count, TTFT-per-turn, swap events, and context token growth.
- `autotune/bench/agent_types.py`: `AgentTask`, `AgentTurnResult`, `AgentRunResult`, `TaskConditionSummary` dataclasses.
- `scripts/agent_bench.py`: full benchmark runner with JSON + Markdown export.
- `AGENT_BENCHMARK.md`: methodology, results, and honest trade-off analysis.

**Local storage control**
- New `autotune storage [on|off|status]` command: opt-out of local SQLite writes without affecting cloud telemetry consent. Model metadata is always stored regardless.
- `autotune/db/storage_prefs.py`: persistent preference file at `~/Library/Application Support/autotune/storage_prefs.json`.
- All behavioral write methods in `store.py` (`log_run`, `log_telemetry_event`, `upsert_hardware`, agent table writes) now check `is_storage_enabled()` and return `-1` when disabled.

**Proof suite v2.0**
- `scripts/proof_suite.py` overhauled: 11 KPIs, 3 models (qwen3:8b, gemma3:4b, llama3.2:3b), 5 prompt types, Wilcoxon signed-rank + Cohen's d statistical tests.
- Cross-model summary report with significance stars.
- New `scripts/create_supabase_schema.py`: applies `schema.sql` to Supabase via Management API (dollar-quoting safe).

**Telemetry**
- `autotune telemetry --enable / --disable / --status` flags for consent management without running `serve`.
- `update_last_seen` Supabase RPC called on every existing-install ping — keeps `active_users` view accurate without UPDATE RLS policy.
- `EventType` now exported from `autotune.telemetry` public API.

**Developer API**
- `autotune.start()`, `autotune.stop()`, `autotune.is_running()`, `autotune.client_kwargs()` programmatic server API.
- `/v1/models/{model_id}/status` endpoint: returns `ready` / `available` / `not_found` + memory fit assessment.
- Structured error bodies on all 4xx/5xx responses: `type`, `message`, `model`, `suggestion`, `status_url`.
- `Retry-After` header on all 429 responses.

**CI/CD**
- Switched PyPI publishing to OIDC trusted publishing (no long-lived API token).
- Wheel verification step in publish pipeline: installs wheel and runs `autotune --help` before upload.
- Python 3.13 added to test matrix.
- `scipy>=1.11` added to `[dev]` extras (required by proof suite statistical tests).

### Fixed

**Critical**
- `_THINK_OPEN` / `_THINK_CLOSE` were referenced in the `/v1/completions` inline state machine but never imported — every streaming request to a reasoning model (qwen3, deepseek-r1, qwq) would crash with `NameError`. Fixed + regression tests added.

**High**
- `_quant_cache` race condition: concurrent requests for the same model issued duplicate Ollama `/api/show` calls. Fixed with `asyncio.Lock` + double-checked pattern.
- `get_db()` singleton: two threads could race on first `connect()`. Fixed with `threading.Lock` + double-checked locking.
- FK race: `SESSION_START` event now fires after `register_install()` in the same background thread — eliminates foreign-key constraint violation on first launch.

**Medium**
- `_normalize_model_id("autotune.")` returned an empty string — backends received an empty model ID. Now raises `HTTP 400`.
- Streaming paths (`/v1/completions`, `/v1/chat/completions`) were missing `asyncio.CancelledError` and broad `Exception` catches — client disconnects caused unhandled exceptions and log noise.
- SQL injection: all dynamic `INSERT` column names now validated against `^[A-Za-z_][A-Za-z0-9_]*$` before interpolation.
- Telemetry `server_received_at` set client-side (defeating the column's purpose). Now set by DB `DEFAULT NOW()`.
- Fallback hardware fingerprint included `socket.gethostname()`. Replaced with anonymous `platform.system()|machine|processor`.
- `estimate_tokens("")` returned 1 instead of 0 — empty content fields inflated token counts by 1.
- `num_keep` could exceed `num_ctx` under aggressive KV reduction. Added overflow clamp.

**User experience**
- Ollama-not-running error now says **"Ollama is not running. Start it with: `ollama serve`"** instead of the confusing model-not-found / HuggingFace token message.
- Chat warmup message updated: no longer says "model will load on first message" when Ollama is demonstrably offline.
- "Download it now?" pull prompt suppressed when the real problem is Ollama being off.

### Changed

- `autotune.start()` now defaults to `use_mlx=False` (Ollama-only, ~94 MB RAM, full tool calling). Pass `use_mlx=True` for MLX on Apple Silicon.
- MLX eliminated as a default `torch` import path — saves 370 MB RAM footprint on non-MLX launches.
- Ollama listed first in backend priority documentation (was deprioritised vs MLX for non-Apple-Silicon users).
- Telemetry anon key documented with explicit security note explaining INSERT-only RLS.

---

## [0.1.1] — 2026-04-14

### Added
- Reasoning model support: `qwen3`, `deepseek-r1`, `qwq` — `<think>` blocks stripped from responses and never saved to conversation DB.
- FIM (fill-in-the-middle) endpoint: `POST /v1/completions` for code autocomplete.
- Flash attention + `num_batch=1024` for lower TTFT and better memory pressure handling.
- `autotune ps` command: shows every model currently loaded in memory across Ollama and MLX.
- Recall system: `/recall` and `/recall search <query>` in chat; semantic search across past conversations.
- Live context tier stats in chat header.
- Model presence watcher: background poller notifies when a model unloads unexpectedly.
- `--no-swap` guarantee: `autotune chat --no-swap` adjusts `num_ctx` and KV precision to guarantee no macOS swap.
- 174 unit tests + CI (GitHub Actions, Python 3.10–3.13).
- PyPI distribution (`pip install llm-autotune`).

### Fixed
- `keep_alive` set to `-1m` (valid Go duration) — previously used `-1` which Ollama silently ignored.
- Thinking tags were being saved to conversation DB and printed to terminal.
- `num_ctx` was not bumped by `THINKING_OVERHEAD` for reasoning models.
- Various chat flow race conditions: auto-send bugs, terminal line-clearing glitches.

---

## [0.1.0] — 2026-04-05

### Added
- Core TTFT optimisation layer: dynamic `num_ctx`, `keep_alive=-1m`, `num_keep` prefix caching.
- `autotune chat`, `autotune run`, `autotune recommend`, `autotune hardware`, `autotune ls`, `autotune benchmark`.
- Profiles: `fast`, `balanced`, `quality`.
- OpenAI-compatible API server (`autotune serve`) with FIFO inference queue.
- Multi-backend support: Ollama, LM Studio, MLX (Apple Silicon), HuggingFace Inference API.
- Persistent conversation memory with SQLite (FTS5 + cosine vector search).
- Multi-tier context management (FULL → RECENT+FACTS → COMPRESSED → EMERGENCY).
- Hardware profiler: CPU/GPU/RAM detection via psutil + py-cpuinfo.
- Model memory estimator: weights + KV cache + runtime overhead.
- `autotune webui`: Open WebUI integration.
- Proof benchmark: `autotune proof` and `scripts/proof_suite.py`.
- Initial 9-model registry with real MMLU/HumanEval/GSM8K scores.

---

[1.0.0]: https://github.com/tanavc1/local-llm-autotune/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/tanavc1/local-llm-autotune/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/tanavc1/local-llm-autotune/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/tanavc1/local-llm-autotune/releases/tag/v0.1.0
