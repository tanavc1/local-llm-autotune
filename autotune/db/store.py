"""
Persistent SQLite database for model specs, hardware profiles, and run observations.

Tables
------
  models            – architecture + memory data fetched from HuggingFace
  hardware_profiles – fingerprint of each machine that has run autotune
  run_observations  – real token/sec + memory measurements logged over time
  telemetry_events  – local system-pressure and lifecycle events
  agent_runs        – agentic benchmark run records
  agent_turns       – per-turn data within an agent run
  tool_calls        – individual tool invocations within an agent turn

The DB lives at ~/.local/share/autotune/autotune.db (XDG) or
~/Library/Application Support/autotune/autotune.db on macOS.

Local storage can be disabled with `autotune storage off` — writes to
run_observations, telemetry_events, hardware_profiles, and agent tables are
silently skipped while storage is off.  Model metadata (models table) is
always stored; it is reference data, not behavioral data.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from autotune.db.storage_prefs import is_storage_enabled


# ---------------------------------------------------------------------------
# DB location
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "autotune"
    elif platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home())) / "autotune"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "autotune"
    base.mkdir(parents=True, exist_ok=True)
    return base / "autotune.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ---------------------------------------------------------------------------
-- models: HuggingFace architecture + memory data (always stored)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS models (
    -- identity
    id                      TEXT PRIMARY KEY,   -- "meta-llama/Meta-Llama-3.1-8B"
    name                    TEXT NOT NULL,
    organization            TEXT,
    family                  TEXT,
    license                 TEXT,
    description             TEXT,
    hf_url                  TEXT,
    gguf_url                TEXT,
    paper_url               TEXT,
    release_date            TEXT,

    -- parameters
    total_params_b          REAL,
    active_params_b         REAL,               -- same as total for dense models
    is_moe                  INTEGER DEFAULT 0,
    num_experts             INTEGER,
    experts_per_token       INTEGER,

    -- architecture (everything that affects hardware)
    arch_type               TEXT,               -- "decoder-only", "encoder-decoder"
    n_layers                INTEGER,
    hidden_size             INTEGER,
    n_heads                 INTEGER,
    n_kv_heads              INTEGER,
    head_dim                INTEGER,            -- explicit, may differ from hidden/n_heads
    intermediate_size       INTEGER,
    vocab_size              INTEGER,
    max_context_window      INTEGER,
    rope_theta              REAL,
    positional_encoding     TEXT,               -- "rope", "alibi", "none"
    activation              TEXT,               -- "silu", "gelu", etc.
    normalization           TEXT,               -- "rms_norm", "layer_norm"
    attention_type          TEXT,               -- "mha", "gqa", "mqa", "mla"
    sliding_window_size     INTEGER,            -- NULL = full attention
    sliding_window_pattern  TEXT,               -- e.g. "every_other" (Gemma 2)
    kv_latent_dim           INTEGER,            -- DeepSeek MLA compressed KV dim
    logit_softcapping       REAL,               -- Gemma 2: 30.0
    attn_logit_softcapping  REAL,               -- Gemma 2: 50.0
    tie_word_embeddings     INTEGER DEFAULT 0,
    num_shared_experts      INTEGER,            -- MoE shared experts

    -- memory estimates (GB at common quants, weights only)
    mem_f16_gb              REAL,
    mem_q8_0_gb             REAL,
    mem_q6_k_gb             REAL,
    mem_q5_k_m_gb           REAL,
    mem_q4_k_m_gb           REAL,
    mem_q4_k_s_gb           REAL,
    mem_q3_k_m_gb           REAL,
    mem_q2_k_gb             REAL,

    -- quantization support
    available_quants        TEXT,               -- JSON array
    recommended_quant       TEXT,
    supports_awq            INTEGER DEFAULT 0,
    supports_gptq           INTEGER DEFAULT 0,
    supports_exl2           INTEGER DEFAULT 0,
    quant_notes             TEXT,

    -- benchmarks (nullable)
    bench_mmlu              REAL,
    bench_humaneval         REAL,
    bench_gsm8k             REAL,
    bench_hellaswag         REAL,
    bench_mt_bench          REAL,
    bench_source            TEXT,

    -- use cases
    use_cases               TEXT,               -- JSON array

    -- metadata
    fetched_at              REAL NOT NULL,      -- unix timestamp
    raw_config              TEXT                -- full config.json as JSON string
);

-- ---------------------------------------------------------------------------
-- hardware_profiles: one row per unique machine fingerprint
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hardware_profiles (
    id                      TEXT PRIMARY KEY,   -- sha256[:16] of (os+cpu+ram+gpu)
    os_name                 TEXT,
    os_version              TEXT,
    cpu_brand               TEXT,
    cpu_physical_cores      INTEGER,
    cpu_logical_cores       INTEGER,
    cpu_arch                TEXT,
    total_ram_gb            REAL,
    gpu_name                TEXT,
    gpu_backend             TEXT,               -- "cuda" | "metal" | "rocm" | "none"
    gpu_vram_gb             REAL,
    is_unified_memory       INTEGER DEFAULT 0,
    first_seen              REAL NOT NULL,
    last_seen               REAL NOT NULL
);

-- ---------------------------------------------------------------------------
-- run_observations: one row per completed LLM inference call
-- NOTE: model_id is NOT a foreign key — the server logs Ollama model IDs
-- like "qwen3:8b" which differ from HuggingFace IDs in the models table.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_observations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id                TEXT NOT NULL,
    hardware_id             TEXT REFERENCES hardware_profiles(id),
    session_id              TEXT,               -- server session UUID
    autotune_version        TEXT,               -- autotune version that produced this row

    -- inference config
    quant                   TEXT NOT NULL,
    context_len             INTEGER NOT NULL,
    n_gpu_layers            INTEGER NOT NULL DEFAULT 0,
    batch_size              INTEGER DEFAULT 1,
    profile_name            TEXT,               -- "fast" | "balanced" | "quality"
    bench_tag               TEXT,               -- label for bench run comparisons
    f16_kv                  INTEGER,            -- 1=F16 KV, 0=Q8 KV
    num_keep                INTEGER,            -- tokens pinned in KV prefix cache

    -- throughput
    tokens_per_sec          REAL,               -- prompt-eval throughput (tok/s)
    gen_tokens_per_sec      REAL,               -- generation throughput (tok/s)

    -- memory
    peak_ram_gb             REAL,               -- peak RSS during inference
    peak_vram_gb            REAL,               -- peak VRAM during inference
    ram_before_gb           REAL,
    ram_after_gb            REAL,
    delta_ram_gb            REAL,               -- ram_after − ram_before
    swap_before_gb          REAL,
    swap_peak_gb            REAL,
    swap_after_gb           REAL,
    delta_swap_gb           REAL,

    -- CPU
    cpu_avg_pct             REAL,
    cpu_peak_pct            REAL,

    -- latency
    load_time_sec           REAL,               -- model cold-start time
    ttft_ms                 REAL,               -- time-to-first-token (ms)
    elapsed_sec             REAL,               -- total wall time (s)

    -- tokens
    prompt_tokens           INTEGER,
    completion_tokens       INTEGER,

    -- outcome
    completed               INTEGER DEFAULT 1,  -- 0 = aborted / OOM
    oom                     INTEGER DEFAULT 0,
    error_msg               TEXT,
    notes                   TEXT,               -- legacy free-form field

    observed_at             REAL NOT NULL
);

-- ---------------------------------------------------------------------------
-- telemetry_events: local system-pressure and lifecycle events
-- Mirrors the Supabase telemetry_events schema so both stores stay in sync.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS telemetry_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER REFERENCES run_observations(id),
    hardware_id         TEXT,
    session_id          TEXT,
    autotune_version    TEXT,
    model_id            TEXT,

    -- event type: "session_start" | "session_end" | "run_complete"
    --           | "ram_spike" | "swap_spike" | "cpu_peak" | "oom_near"
    --           | "kv_reduced" | "error" | "model_loaded" | "opt_in" | "opt_out"
    event_type          TEXT NOT NULL,

    -- inference metrics (populated for run_complete)
    tokens_per_sec      REAL,
    gen_tokens_per_sec  REAL,
    ttft_ms             REAL,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    context_len         INTEGER,
    peak_ram_gb         REAL,
    cpu_avg_pct         REAL,
    elapsed_sec         REAL,
    profile_name        TEXT,
    completed           INTEGER,
    oom                 INTEGER DEFAULT 0,

    -- system-pressure event payload
    value_num           REAL,
    value_text          TEXT,

    -- error event payload
    error_type          TEXT,
    error_msg           TEXT,

    observed_at         REAL NOT NULL
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_runs_model       ON run_observations(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_hardware    ON run_observations(hardware_id);
CREATE INDEX IF NOT EXISTS idx_runs_quant       ON run_observations(quant);
CREATE INDEX IF NOT EXISTS idx_runs_observed    ON run_observations(observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_session     ON run_observations(session_id)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_telemetry_run    ON telemetry_events(run_id)
    WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_telemetry_model  ON telemetry_events(model_id)
    WHERE model_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_telemetry_type   ON telemetry_events(event_type, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_session ON telemetry_events(session_id)
    WHERE session_id IS NOT NULL;
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _db_path()
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------ #
    # Connection management                                                #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        # check_same_thread=False: the singleton is accessed from the FastAPI
        # async event loop and background threads; SQLite handles this safely
        # when operations are serialised via the transaction() context manager.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._migrate()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Database":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Use `with Database()` or call .connect()")
        return self._conn

    def _migrate(self) -> None:
        """Add any columns introduced after the initial schema was deployed.

        SQLite does not support ADD COLUMN IF NOT EXISTS — we attempt each
        ALTER and silently ignore the OperationalError raised when the column
        already exists.
        """
        new_cols = [
            ("run_observations", "profile_name",      "TEXT"),
            ("run_observations", "bench_tag",          "TEXT"),
            ("run_observations", "f16_kv",             "INTEGER"),
            ("run_observations", "num_keep",           "INTEGER"),
            ("run_observations", "ram_before_gb",      "REAL"),
            ("run_observations", "ram_after_gb",       "REAL"),
            ("run_observations", "delta_ram_gb",       "REAL"),
            ("run_observations", "swap_before_gb",     "REAL"),
            ("run_observations", "swap_peak_gb",       "REAL"),
            ("run_observations", "swap_after_gb",      "REAL"),
            ("run_observations", "delta_swap_gb",      "REAL"),
            ("run_observations", "cpu_avg_pct",        "REAL"),
            ("run_observations", "cpu_peak_pct",       "REAL"),
            ("run_observations", "elapsed_sec",        "REAL"),
            ("run_observations", "prompt_tokens",      "INTEGER"),
            ("run_observations", "completion_tokens",  "INTEGER"),
            ("run_observations", "error_msg",          "TEXT"),
        ]
        for table, col, col_type in new_cols:
            try:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                )
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

        # Indexes on new columns — created after the columns exist
        new_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_runs_tag     ON run_observations(bench_tag)",
            "CREATE INDEX IF NOT EXISTS idx_runs_profile ON run_observations(profile_name)",
        ]
        for idx_sql in new_indexes:
            try:
                self._conn.execute(idx_sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------------ #
    # Model CRUD                                                           #
    # ------------------------------------------------------------------ #

    def upsert_model(self, data: dict[str, Any]) -> None:
        """Insert or replace a model record."""
        data = dict(data)
        if "available_quants" in data and isinstance(data["available_quants"], list):
            data["available_quants"] = json.dumps(data["available_quants"])
        if "use_cases" in data and isinstance(data["use_cases"], list):
            data["use_cases"] = json.dumps(data["use_cases"])
        if "raw_config" in data and isinstance(data["raw_config"], dict):
            data["raw_config"] = json.dumps(data["raw_config"])
        data.setdefault("fetched_at", time.time())

        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        sql = f"INSERT OR REPLACE INTO models ({cols}) VALUES ({placeholders})"
        with self.transaction():
            self.conn.execute(sql, list(data.values()))

    def get_model(self, model_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM models WHERE id = ?", (model_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        for field in ("available_quants", "use_cases"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def list_models(
        self,
        family: Optional[str] = None,
        max_params_b: Optional[float] = None,
        min_params_b: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM models WHERE 1=1"
        args: list[Any] = []
        if family:
            q += " AND family = ?"
            args.append(family)
        if max_params_b is not None:
            q += " AND active_params_b <= ?"
            args.append(max_params_b)
        if min_params_b is not None:
            q += " AND active_params_b >= ?"
            args.append(min_params_b)
        q += " ORDER BY active_params_b ASC"
        rows = self.conn.execute(q, args).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for field in ("available_quants", "use_cases"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result

    # ------------------------------------------------------------------ #
    # Hardware profiles                                                    #
    # ------------------------------------------------------------------ #

    def upsert_hardware(self, data: dict[str, Any]) -> str:
        """Insert or update a hardware profile. Returns the profile ID."""
        if not is_storage_enabled():
            return data.get("id", "")
        hw_id = data["id"]
        existing = self.conn.execute(
            "SELECT id FROM hardware_profiles WHERE id = ?", (hw_id,)
        ).fetchone()

        now = time.time()
        if existing:
            with self.transaction():
                self.conn.execute(
                    "UPDATE hardware_profiles SET last_seen = ? WHERE id = ?",
                    (now, hw_id),
                )
        else:
            data = dict(data)
            data.setdefault("first_seen", now)
            data["last_seen"] = now
            cols = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            with self.transaction():
                self.conn.execute(
                    f"INSERT INTO hardware_profiles ({cols}) VALUES ({placeholders})",
                    list(data.values()),
                )
        return hw_id

    def get_hardware(self, hw_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM hardware_profiles WHERE id = ?", (hw_id,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    # Run observations                                                     #
    # ------------------------------------------------------------------ #

    def log_run(self, data: dict[str, Any]) -> int:
        """Log a performance observation. Returns the new row ID, or -1 if storage disabled."""
        if not is_storage_enabled():
            return -1
        data = dict(data)
        data.setdefault("observed_at", time.time())
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        with self.transaction():
            cur = self.conn.execute(
                f"INSERT INTO run_observations ({cols}) VALUES ({placeholders})",
                list(data.values()),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def get_runs(
        self,
        model_id: Optional[str] = None,
        hardware_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM run_observations WHERE 1=1"
        args: list[Any] = []
        if model_id:
            q += " AND model_id = ?"
            args.append(model_id)
        if hardware_id:
            q += " AND hardware_id = ?"
            args.append(hardware_id)
        q += " ORDER BY observed_at DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def get_runs_by_tag(self, tag_substr: str, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch runs whose notes field contains a bench_tag= substring."""
        rows = self.conn.execute(
            "SELECT * FROM run_observations WHERE notes LIKE ? ORDER BY observed_at DESC LIMIT ?",
            (f"%bench_tag={tag_substr}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Telemetry events                                                     #
    # ------------------------------------------------------------------ #

    def log_telemetry_event(
        self,
        event_type: str,
        value_num: Optional[float] = None,
        value_text: Optional[str] = None,
        run_id: Optional[int] = None,
        hardware_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> int:
        """Append a system-level telemetry event. Returns the new row ID, or -1 if storage disabled."""
        if not is_storage_enabled():
            return -1
        with self.transaction():
            cur = self.conn.execute(
                """INSERT INTO telemetry_events
                   (run_id, hardware_id, model_id, event_type, value_num, value_text, observed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, hardware_id, model_id, event_type,
                 value_num, value_text, time.time()),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def get_telemetry(
        self,
        model_id: Optional[str] = None,
        event_type: Optional[str] = None,
        run_id: Optional[int] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM telemetry_events WHERE 1=1"
        args: list[Any] = []
        if model_id:
            q += " AND model_id = ?"
            args.append(model_id)
        if event_type:
            q += " AND event_type = ?"
            args.append(event_type)
        if run_id is not None:
            q += " AND run_id = ?"
            args.append(run_id)
        q += " ORDER BY observed_at DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def telemetry_summary(self, model_id: Optional[str] = None) -> dict[str, Any]:
        """Aggregate telemetry event counts by type for a quick overview."""
        q = "SELECT event_type, COUNT(*) as cnt FROM telemetry_events"
        args: list[Any] = []
        if model_id:
            q += " WHERE model_id = ?"
            args.append(model_id)
        q += " GROUP BY event_type ORDER BY cnt DESC"
        rows = self.conn.execute(q, args).fetchall()
        return {r["event_type"]: r["cnt"] for r in rows}

    def model_perf_history(self, model_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Return performance history for a model across all runs.

        Includes structured columns (ttft_ms, tokens_per_sec, peak_ram_gb,
        elapsed_sec, profile_name, bench_tag) ordered newest first.
        """
        rows = self.conn.execute(
            """SELECT id, observed_at, profile_name, bench_tag,
                      tokens_per_sec, ttft_ms, peak_ram_gb, swap_peak_gb,
                      cpu_avg_pct, elapsed_sec, context_len, f16_kv, num_keep,
                      delta_ram_gb, delta_swap_gb, error_msg, completed
               FROM run_observations
               WHERE model_id = ?
               ORDER BY observed_at DESC
               LIMIT ?""",
            (model_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def compare_runs(self, tag_a: str, tag_b: str) -> dict[str, Any]:
        """
        Compare two bench tags.  Returns delta metrics (B relative to A).
        Positive delta = B is higher/worse for RAM/swap; higher/better for tok/s.
        """
        def _avg(rows: list[dict], key: str) -> Optional[float]:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        rows_a = self.get_runs_by_tag(tag_a)
        rows_b = self.get_runs_by_tag(tag_b)

        if not rows_a or not rows_b:
            return {"error": f"No runs found for tag '{tag_a}' or '{tag_b}'"}

        metrics = ["tokens_per_sec", "ttft_ms", "peak_ram_gb", "peak_vram_gb"]
        result: dict[str, Any] = {
            "tag_a": tag_a,
            "tag_b": tag_b,
            "runs_a": len(rows_a),
            "runs_b": len(rows_b),
            "deltas": {},
        }
        for m in metrics:
            a = _avg(rows_a, m)
            b = _avg(rows_b, m)
            if a is not None and b is not None and a != 0:
                pct = round((b - a) / a * 100, 1)
                result["deltas"][m] = {"a": a, "b": b, "delta_pct": pct}
        return result

    # ------------------------------------------------------------------ #
    # Agent benchmark tables (migration + CRUD)                           #
    # ------------------------------------------------------------------ #

    _AGENT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS agent_runs (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id              TEXT NOT NULL,
        condition            TEXT NOT NULL,
        model_id             TEXT NOT NULL,
        trial_idx            INTEGER NOT NULL,
        profile              TEXT,
        task_success         INTEGER NOT NULL,
        exit_reason          TEXT NOT NULL,
        total_wall_sec       REAL NOT NULL,
        total_tool_calls     INTEGER NOT NULL,
        tool_error_count     INTEGER NOT NULL,
        backtrack_count      INTEGER NOT NULL,
        total_turns          INTEGER NOT NULL,
        reload_count         INTEGER NOT NULL,
        peak_ram_gb          REAL NOT NULL,
        swap_occurred        INTEGER NOT NULL,
        free_floor_gb        REAL NOT NULL,
        final_context_tokens INTEGER NOT NULL,
        observed_at          REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS agent_turns (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id               INTEGER NOT NULL REFERENCES agent_runs(id),
        turn_idx             INTEGER NOT NULL,
        role                 TEXT NOT NULL,
        prefill_ms           REAL NOT NULL,
        ttft_ms              REAL NOT NULL,
        eval_tps             REAL NOT NULL,
        total_ms             REAL NOT NULL,
        ollama_ram_gb        REAL NOT NULL,
        kv_cache_mb          REAL,
        swap_delta_gb        REAL NOT NULL,
        tokens_in_context    INTEGER NOT NULL,
        num_ctx              INTEGER,
        tool_name            TEXT,
        tool_success         INTEGER,
        tool_latency_ms      REAL
    );

    CREATE TABLE IF NOT EXISTS tool_calls (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id               INTEGER NOT NULL REFERENCES agent_runs(id),
        turn_id              INTEGER NOT NULL REFERENCES agent_turns(id),
        tool_name            TEXT NOT NULL,
        args_json            TEXT,
        result_preview       TEXT,
        success              INTEGER NOT NULL,
        latency_ms           REAL NOT NULL,
        observed_at          REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_agent_runs_task      ON agent_runs(task_id);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_condition ON agent_runs(condition);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_model     ON agent_runs(model_id);
    CREATE INDEX IF NOT EXISTS idx_agent_turns_run      ON agent_turns(run_id);
    CREATE INDEX IF NOT EXISTS idx_tool_calls_run       ON tool_calls(run_id);
    """

    def migrate_agent_tables(self) -> None:
        """Create agent benchmark tables if they don't already exist."""
        self.conn.executescript(self._AGENT_SCHEMA)
        self.conn.commit()

    def log_agent_run(self, data: dict[str, Any]) -> int:
        """Insert one AgentRunResult row. Returns the new row ID, or -1 if storage disabled."""
        if not is_storage_enabled():
            return -1
        data = dict(data)
        data.setdefault("observed_at", time.time())
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        with self.transaction():
            cur = self.conn.execute(
                f"INSERT INTO agent_runs ({cols}) VALUES ({placeholders})",
                list(data.values()),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def log_agent_turn(self, run_id: int, data: dict[str, Any]) -> int:
        """Insert one AgentTurnResult row. Returns the new row ID, or -1 if storage disabled."""
        if not is_storage_enabled():
            return -1
        data = dict(data)
        data["run_id"] = run_id
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        with self.transaction():
            cur = self.conn.execute(
                f"INSERT INTO agent_turns ({cols}) VALUES ({placeholders})",
                list(data.values()),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def log_tool_call(self, run_id: int, turn_id: int, data: dict[str, Any]) -> int:
        """Insert one tool_calls row. Returns the new row ID, or -1 if storage disabled."""
        if not is_storage_enabled():
            return -1
        data = dict(data)
        data["run_id"]  = run_id
        data["turn_id"] = turn_id
        data.setdefault("observed_at", time.time())
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        with self.transaction():
            cur = self.conn.execute(
                f"INSERT INTO tool_calls ({cols}) VALUES ({placeholders})",
                list(data.values()),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def get_agent_runs(
        self,
        model_id: Optional[str] = None,
        task_id: Optional[str] = None,
        condition: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM agent_runs WHERE 1=1"
        args: list[Any] = []
        if model_id:
            q += " AND model_id = ?"
            args.append(model_id)
        if task_id:
            q += " AND task_id = ?"
            args.append(task_id)
        if condition:
            q += " AND condition = ?"
            args.append(condition)
        q += " ORDER BY observed_at DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def get_agent_turns(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM agent_turns WHERE run_id = ? ORDER BY turn_idx ASC",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def model_count(self) -> int:
        """Return the number of models in the DB (used by CLI fetch-many)."""
        return self.conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]

    def stats(self) -> dict[str, Any]:
        return {
            "models": self.conn.execute("SELECT COUNT(*) FROM models").fetchone()[0],
            "hardware_profiles": self.conn.execute("SELECT COUNT(*) FROM hardware_profiles").fetchone()[0],
            "run_observations": self.conn.execute("SELECT COUNT(*) FROM run_observations").fetchone()[0],
            "db_path": str(self.path),
            "db_size_mb": round(self.path.stat().st_size / 1024**2, 3) if self.path.exists() else 0,
        }


# ---------------------------------------------------------------------------
# Module-level singleton helper
# ---------------------------------------------------------------------------

_db: Optional[Database] = None


def get_db() -> Database:
    """Return the open module-level DB singleton (connects on first call)."""
    global _db
    if _db is None:
        _db = Database()
        _db.connect()
    return _db
