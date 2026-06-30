# Changelog

All notable changes to **llm-autotune** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.6.0] — 2026-06-30

### Added

- **`autotune start` — a required, guided first-run setup.** This is now the first command to run after installing. In about 2 minutes it verifies Ollama (starting it if needed), profiles your hardware and picks the best model, pulls it, proves the speedup on your machine, and then prints the exact `autotune chat` command for your model so you're ready to go immediately.
- **First-run gate.** Until `autotune start` has completed once, every other command is blocked and redirected to it — a fresh install can't fail in confusing ways. `--help`/`-h`, `start`/`init`, `version`, and `upgrade` stay available so the CLI is still discoverable and updatable before setup.

### Changed

- README and website now lead with `pip install llm-autotune` followed immediately by `autotune start`, then `autotune chat --model <id>`.

### Fixed

- **Upgrades never lock out an existing install.** `_is_initialized()` now treats clear evidence of prior use — an on-disk recall or observations database — as "already set up" and backfills the setup sentinel, so users who installed autotune before the gate existed are grandfathered in and never blocked.

---

## [1.5.0] — 2026-05-27

### Added

- **Settings tab in the dashboard** — live read/write panel for all 6 configurable keys:
  - `max_context_tokens` (512–131 072): global context window ceiling
  - `kv_cache_quant` (`f16` / `q8` / `q4`): KV precision override
  - `keep_alive_secs` (0–86 400): model resident-memory duration
  - `telemetry_enabled` (`true`/`false`): opt-in anonymous telemetry
  - `retention_days` (7–3 650): local run-observation history window
  - `log_slow_threshold_ms` (100–60 000): slow-request alerting threshold
  - Batch-save with a single POST; individual updates reject unknown keys
  - Cleanup endpoint to prune observations older than `retention_days`
- **Production-grade SQLite hardening** — new columns in `run_observations`: `backend`, `request_id`, `conversation_id`, `stress_retry`, `thinking_tokens`, `error_code`; new columns in `hardware_profiles`: `metal_gpu_cores`, `cpu_freq_mhz`, `storage_type`; new columns in `models`: `ollama_tag`, `tier`, `is_local`; three new partial indexes (`idx_runs_backend`, `idx_runs_request_id`, `idx_runs_model_time`); all additions applied safely via `_migrate()` and idempotent on existing databases
- **Comprehensive test suite** — 1 186 tests total (146 new this release): `test_db_settings.py` (67 tests covering all 6 settings defaults, CRUD, migration idempotency, schema completeness, thread-safety) and `test_dashboard_settings.py` (79 tests covering all boundary values, API auth, whitelist enforcement, rate-limiter isolation)

### Fixed

- `ThinkingStreamFilter` assertion in integration tests — now skips gracefully when a model returns only reasoning tokens with no answer text

---

## [1.4.0] — 2026-05-27

### Added

- **Dashboard authentication** — session cookie login via `AUTOTUNE_ADMIN_KEY`; `itsdangerous` HMAC-signed tokens; revoked-session table backed by SQLite; configurable session TTL; `POST /api/dashboard/logout` invalidation endpoint
- **API key management UI** — create, list, and revoke `sk-at-*` keys directly from the dashboard without curl; proxied through `/api/dashboard/keys/*` so the admin key never leaves the server
- **Security hardening** — HTTP security headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options); sliding-window rate limiters (write: 30/h, read: 300/min, refresh: 60/min); `_WRITABLE_SETTINGS` whitelist blocks arbitrary key injection
- **Auto-refreshing model catalog** — 43-model registry with HuggingFace + Ollama metadata; `autotune models --registry` shows tier, parameter count, and recommended hardware; catalog updates automatically in the background; dashboard "Model Catalog" panel lists all known models with fit scores for the current hardware

---

## [1.3.0] — 2026-05-24

### Added

- **Web dashboard at `localhost:8765/dashboard`** — fully accurate, live-updating dashboard showing:
  - **System overview KPIs**: RAM used (with pressure color), running models in memory, requests today, avg TTFT, avg tok/s, KV cache savings vs 4096 default
  - **Requests per hour chart**: 24-bucket bar chart of the last 24 h
  - **TTFT sparkline**: last 100 requests colored by latency tier (green / blue / yellow / red)
  - **Raw vs Tuned comparison**: autotune's average dynamic context vs Ollama's fixed 4096-token default — shows context reduction %, KV memory savings %, and measured avg TTFT
  - **Per-model breakdown table**: requests, avg/min/max TTFT, avg tok/s, avg context, avg elapsed, total tokens, last used — for every model ever routed through autotune
  - **Active API keys**: requests and tokens consumed today per key
  - **Slow requests panel**: recent requests that took >5 s, with model, elapsed, TTFT, context, profile, and timestamp
  - **Suggestions panel**: rule-based guidance derived from real data — high TTFT, RAM pressure, KV savings, slow requests
  - Auto-polls all data every 10 seconds; live / offline indicator in the header
- **`autotune/dashboard/`** — new package (`metrics.py` query layer + `router.py` FastAPI router + `static/index.html` self-contained UI)
- **SQLite `run_observations` now stores all fields**: `elapsed_sec`, `prompt_tokens`, `completion_tokens`, `profile_name`, `f16_kv`, `num_keep` — previously omitted from the local DB write in `_emit_run_telemetry()`, making historical dashboard data accurate from this version forward

---

## [1.2.2] — 2026-05-24

### Fixed

- **Windows: Ollama falsely reported as not running** (issue #3) — on Windows,
  `localhost` resolves to `::1` (IPv6) before `127.0.0.1` (IPv4), while Ollama
  listens on IPv4 only. `ollama_base()` now defaults to `http://127.0.0.1:11434`
  on Windows when `AUTOTUNE_OLLAMA_URL` is not set. `is_ollama_running()` also
  tries `127.0.0.1` as a fallback when `localhost` is in the configured URL,
  covering users who have set the env var explicitly.

---

## [1.2.1] — 2026-05-22

### Added

- **Docker Hub image** (`tanavc1/llm-autotune`) — pre-built `linux/amd64` + `linux/arm64` images published automatically on every release tag. `docker pull tanavc1/llm-autotune:latest` to get the autotune-only server.
- **Homebrew tap** — `brew install tanavc1/autotune/llm-autotune` via the new `homebrew-autotune` tap. Formula is auto-synced from `Formula/llm-autotune.rb` on every release.
- **`.github/workflows/docker.yml`** — multi-arch Docker Hub publish workflow; polls PyPI before building to avoid race with the PyPI publish job; updates Docker Hub description from `README.md` automatically.
- **`Dockerfile.autotune` `HEALTHCHECK`** — Docker now reports container health via `/health` endpoint.

### Changed

- `README.md`: added Docker Hub badge; Homebrew one-liner in install block; Docker Hub pull in Docker section; API key auth section with curl examples and admin endpoint table; new env vars (`AUTOTUNE_REQUIRE_API_KEY`, `AUTOTUNE_ADMIN_KEY`) in Docker env vars table.
- Homebrew tap sync now auto-patches the formula URL and SHA256 from PyPI on every release — no manual formula edits needed.

---

## [1.2.0] — 2026-05-21

### Added

**Enterprise API key authentication**
- `POST /admin/keys` — create an API key (`sk-at-` prefix); plaintext returned once, SHA-256 hash stored.
- `GET /admin/keys` — list all keys with status, label, and metadata.
- `GET /admin/keys/{id}` — single key detail + 30-day usage breakdown.
- `DELETE /admin/keys/{id}` — soft-revoke a key with an optional reason.
- `GET /admin/usage` — per-key / per-day / per-model request breakdown (params: `start`, `end`, `key_id`, `model_id`).
- `GET /admin/usage/summary` — aggregate totals per key (param: `days`).
- All `/admin/*` endpoints require `Authorization: Bearer $AUTOTUNE_ADMIN_KEY` (returns 503 if env var not set).

**API key enforcement middleware**
- `AUTOTUNE_REQUIRE_API_KEY=1` enables enforcement on all `/v1/*` paths (default: off, fully backwards-compatible).
- 401 for unknown keys; 403 for revoked keys — distinguishable by callers.
- In-memory key cache (hit → no DB read); invalidated automatically on revocation.
- `/health`, `/api/*`, and `/admin/*` are exempt from key enforcement.

**Per-key usage logging**
- Every authorized `/v1/*` request logs: `key_id`, `model_id`, `tokens_in`, `tokens_out`, `latency_ms`, timestamp.
- SQLite primary store; best-effort Supabase mirror via INSERT-only RLS.
- `api_key_daily_summary` and `api_key_totals` views in Supabase schema.

**Schema migration safety**
- `_pre_migrate()` added to `Database.connect()`: adds `session_id` and `autotune_version` columns to existing DBs before applying the full schema, preventing `OperationalError` on older installs.

### Environment variables (new)

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTOTUNE_REQUIRE_API_KEY` | `0` | Set to `1` to enforce API key auth on all `/v1/*` routes |
| `AUTOTUNE_ADMIN_KEY` | _(unset)_ | Bearer token for all `/admin/*` endpoints |

---

## [1.1.2] — 2026-05-04

### Fixed

- **`_version_newer` fallback** — the version comparison fallback used `v_new != v_cur` which returned `True` for any difference, not just newer versions. Fixed to use a tuple-based comparison so upgrade checks work correctly without `packaging`.
- **`doctor` Python version check** — reported "3.10+ required" but the package supports Python 3.9+. Now correctly checks `>= 3.9`.
- **`doctor` false failures for optional packages** — `numpy` and `sqlalchemy` were checked as required but are optional. They now show as warnings (⚠) instead of failures (✗).
- **`packaging` added to required dependencies** — used internally for version comparison but was missing from `pyproject.toml`.
- **Hardcoded `localhost:11434` in CLI commands** — `init`, `doctor`, `proof`, `ps`, `ls`, `run`, and bench-autoselect all bypassed `AUTOTUNE_OLLAMA_URL`. Fixed to use `_ollama_base()` consistently.
- **`bench/proof_suite.py`, `bench/agent_bench.py`, `bench/user_bench.py`** — module-level `_OLLAMA_BASE` constants hardcoded `http://localhost:11434`, ignoring `AUTOTUNE_OLLAMA_URL`. Fixed to call `_ollama_base()` at import time.
- **`proof` command silent failure** — `except Exception: pass` hid Ollama connection errors. Now shows a descriptive error message and exits with code 1.
- **`proof` output path not validated** — writing results could crash with an unhelpful traceback if the parent directory didn't exist. Now creates missing directories with a clear error on failure.
- **`ps` command unhandled exceptions** — raw tracebacks on backend query failure replaced with a clean error message and exit code 1.
- **`mlx_backend.py` cache clear** — `mx.metal.clear_cache()` renamed to `mx.clear_cache()` in recent mlx versions; updated to avoid `AttributeError` on newer Apple Silicon setups.
- **`keep_alive_enabled` config key** — new user-configurable setting (`autotune config set keep_alive_enabled false`) to disable indefinite model pinning for users on memory-constrained machines. Respected by both `TTFTOptimizer` and `BackendChain`.
- **`user_config` bool coercion** — config values of type `bool` now accept `true/false/1/0/yes/no/on/off` strings instead of failing silently.

---

## [1.0.9] — 2026-04-29

### Fixed

- **`bench/quick_proof.py`** — `_OLLAMA_BASE` was a module-level constant set to `http://localhost:11434`, ignoring `AUTOTUNE_OLLAMA_URL`. Replaced with dynamic `_ollama_base()` calls so Docker and remote-Ollama users get correct benchmark results.
- **`bench/runner.py`** — `run_raw_ollama()` and `run_bench_ollama_only()` both hardcoded `http://localhost:11434/v1/chat/completions`. Fixed to use `_ollama_base()` so all bench paths respect `AUTOTUNE_OLLAMA_URL`.
- **`api/backends/chain.py`** — `_scan_hf_cache()` used `Path.exists()` instead of `Path.is_dir()`. If `~/.cache/huggingface/hub` is somehow a regular file, the subsequent `iterdir()` call would raise `NotADirectoryError`. Changed to `is_dir()` which returns `False` for files.
- **`api/server.py` `/v1/completions`** — `yield b"data: [DONE]\n\n"` was positioned *after* the `try/finally` block that called `await queue.release()`, allowing the concurrency slot to free before the final SSE frame was delivered. Moved inside the `try` block so the slot is held until the full response has been sent.

---

## [1.0.8] — 2026-04-29

### Added

**Docker support — Ollama + autotune bundled**
- `Dockerfile` — single-container image built on `ollama/ollama:latest` with Python and autotune installed from local source. Starts Ollama in the background and autotune serve in the foreground via `docker-entrypoint.sh`.
- `Dockerfile.autotune` — lightweight Python-only image (~200 MB) for multi-container deployments where Ollama runs as a separate service.
- `docker-compose.yml` — two profiles: `single` (bundled container) and `multi` (separate Ollama + autotune services with health checks and `depends_on`).
- `docker-entrypoint.sh` — startup script: starts Ollama on `0.0.0.0`, waits for readiness, optionally pulls `OLLAMA_MODEL` on first boot, then starts autotune serve.
- `.dockerignore` — excludes venv, .git, website, benchmark results, and local env files from the build context.

**`AUTOTUNE_OLLAMA_URL` environment variable**
- All hardcoded `http://localhost:11434` URLs in the API/runtime layer now call a shared `autotune._ollama.ollama_base()` helper that reads `AUTOTUNE_OLLAMA_URL`.
- Default behaviour unchanged (`http://localhost:11434`).
- Set `AUTOTUNE_OLLAMA_URL=http://ollama:11434` in docker-compose multi-container mode to route autotune to a separate Ollama service.
- Files updated: `api/backends/chain.py`, `api/server.py`, `api/local_models.py`, `api/running_models.py`, `api/ollama_pull.py`, `api/chat.py`, `metrics/ollama_client.py`, `metrics/vram.py`, `memory/noswap.py`, `recall/embedder.py`, `session/monitor.py`.

### Environment variables (new)

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUTOTUNE_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL — override for remote/Docker setups |
| `OLLAMA_MODEL` | _(empty)_ | Model to auto-pull on container first start |
| `AUTOTUNE_PORT` | `8765` | Port autotune binds inside the container |
| `OLLAMA_HOST` | `0.0.0.0` | Bind address passed to `ollama serve` inside the container |

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

[1.5.0]: https://github.com/tanavc1/local-llm-autotune/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/tanavc1/local-llm-autotune/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/tanavc1/local-llm-autotune/compare/v1.2.2...v1.3.0
[1.2.2]: https://github.com/tanavc1/local-llm-autotune/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/tanavc1/local-llm-autotune/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/tanavc1/local-llm-autotune/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/tanavc1/local-llm-autotune/compare/v1.0.0...v1.1.2
[1.0.0]: https://github.com/tanavc1/local-llm-autotune/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/tanavc1/local-llm-autotune/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/tanavc1/local-llm-autotune/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/tanavc1/local-llm-autotune/releases/tag/v0.1.0
