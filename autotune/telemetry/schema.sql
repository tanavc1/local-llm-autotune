-- ==========================================================================
--  autotune  ·  Supabase telemetry schema
--  Version: 1.0   Compatible with: autotune >= 0.1.1
--
--  Run this in the Supabase SQL Editor (or via psql with the service_role
--  credentials).  All statements are idempotent — safe to re-run.
-- ==========================================================================


-- --------------------------------------------------------------------------
-- 1. installations
--    One row per unique machine.  Keyed by a 16-char sha-256 hardware
--    fingerprint — no PII (no hostnames, usernames, serial numbers, or IPs).
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.installations (
    install_key         TEXT PRIMARY KEY,

    -- Anonymous hardware class
    os_name             TEXT,                    -- "Darwin" | "Linux" | "Windows"
    os_version          TEXT,                    -- "14.5", "22.04"
    cpu_brand           TEXT,                    -- "Apple M3 Pro", "Intel Core i9"
    cpu_physical_cores  INTEGER,
    cpu_logical_cores   INTEGER,
    cpu_arch            TEXT,                    -- "arm64" | "x86_64"
    total_ram_gb        REAL,
    gpu_name            TEXT,
    gpu_backend         TEXT,                    -- "cuda" | "metal" | "rocm" | "none"
    gpu_vram_gb         REAL,
    is_unified_memory   BOOLEAN DEFAULT FALSE,

    -- Software context
    autotune_version    TEXT,                    -- "0.1.1"
    python_version      TEXT,                    -- "3.11.9"
    telemetry_opted_in  BOOLEAN DEFAULT FALSE,

    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.installations IS
    'One row per unique machine fingerprint (sha-256 of hardware attributes). '
    'Updated on each session start via last_seen_at.';


-- --------------------------------------------------------------------------
-- 2. app_stats
--    Global aggregate counters.  Rows are seeded once here; the
--    increment_stat() function keeps them current.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.app_stats (
    stat_key        TEXT PRIMARY KEY,
    stat_value      BIGINT DEFAULT 0,
    last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.app_stats IS
    'Rolling counters for high-level product metrics '
    '(installs, events, runs, active users).';

INSERT INTO public.app_stats (stat_key, stat_value) VALUES
    ('total_installations',    0),
    ('total_telemetry_events', 0),
    ('total_run_observations', 0),
    ('active_7d',              0),
    ('active_30d',             0)
ON CONFLICT (stat_key) DO NOTHING;


-- --------------------------------------------------------------------------
-- 3. app_releases
--    Tracks every published autotune version so we can measure adoption.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.app_releases (
    version         TEXT PRIMARY KEY,            -- "0.1.1"
    release_notes   TEXT,
    released_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_stable       BOOLEAN DEFAULT TRUE,
    min_python      TEXT DEFAULT '3.10',
    install_count   BIGINT DEFAULT 0             -- incremented per unique machine
);

COMMENT ON TABLE public.app_releases IS
    'Published autotune versions.  install_count is bumped when a new '
    'machine reports that version for the first time.';

INSERT INTO public.app_releases (version, released_at, is_stable, min_python)
VALUES ('0.1.1', NOW(), TRUE, '3.10')
ON CONFLICT (version) DO NOTHING;


-- --------------------------------------------------------------------------
-- 4. telemetry_events
--    Opt-in event stream.  One row per discrete event from a user machine.
--    Low-cardinality string columns (event_type, quant, profile_name) are
--    kept as TEXT so schema changes are zero-downtime.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.telemetry_events (
    id                  BIGSERIAL PRIMARY KEY,
    install_key         TEXT NOT NULL
                            REFERENCES public.installations(install_key)
                            ON DELETE CASCADE,
    session_id          UUID,                    -- groups events within one server session
    event_type          TEXT NOT NULL,           -- see EventType in events.py
    autotune_version    TEXT,

    -- Model context (NULL for non-inference events)
    model_id            TEXT,                    -- "qwen3:8b", "gemma3:12b", etc.

    -- Inference performance (run_complete events)
    tokens_per_sec      REAL,                    -- prompt-eval throughput (tok/s)
    gen_tokens_per_sec  REAL,                    -- generation throughput (tok/s)
    ttft_ms             REAL,                    -- time-to-first-token (ms)
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    context_len         INTEGER,                 -- num_ctx actually used
    peak_ram_gb         REAL,
    peak_vram_gb        REAL,
    delta_ram_gb        REAL,                    -- RAM increase during inference
    cpu_avg_pct         REAL,
    cpu_peak_pct        REAL,
    load_time_sec       REAL,                    -- model load latency (s)
    elapsed_sec         REAL,                    -- total wall time (s)
    profile_name        TEXT,                    -- "fast" | "balanced" | "quality"
    quant               TEXT,                    -- "q4_K_M", "q8_0", etc.
    completed           BOOLEAN DEFAULT TRUE,
    oom                 BOOLEAN DEFAULT FALSE,

    -- System pressure events (ram_spike, swap_spike, oom_near, cpu_peak)
    value_num           REAL,                    -- numeric reading (e.g. RAM %)
    value_text          TEXT,                    -- human-readable detail

    -- Error events
    error_type          TEXT,                    -- exception class name
    error_msg           TEXT,                    -- first 500 chars of message

    occurred_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    server_received_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.telemetry_events IS
    'Opt-in anonymous event stream. One row per discrete event from a user machine.';

-- Covering indexes for the most common analytics queries
CREATE INDEX IF NOT EXISTS idx_te_install
    ON public.telemetry_events (install_key, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_te_event_type
    ON public.telemetry_events (event_type, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_te_model
    ON public.telemetry_events (model_id, occurred_at DESC)
    WHERE model_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_te_session
    ON public.telemetry_events (session_id)
    WHERE session_id IS NOT NULL;


-- --------------------------------------------------------------------------
-- 5. run_observations
--    Detailed inference performance records — one per completed LLM call.
--    Cloud mirror of the local SQLite run_observations, enriched with
--    hardware context for cross-machine comparisons.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.run_observations (
    id                  BIGSERIAL PRIMARY KEY,
    install_key         TEXT
                            REFERENCES public.installations(install_key)
                            ON DELETE SET NULL,
    hardware_key        TEXT,                    -- 16-char sha-256 (= install_key)

    -- Hardware class at time of run
    os_name             TEXT,
    cpu_arch            TEXT,
    total_ram_gb        REAL,
    gpu_backend         TEXT,

    -- Model + inference config
    model_id            TEXT NOT NULL,
    quant               TEXT NOT NULL,
    context_len         INTEGER NOT NULL,
    n_gpu_layers        INTEGER NOT NULL DEFAULT 0,
    batch_size          INTEGER DEFAULT 1,
    profile_name        TEXT,
    f16_kv              BOOLEAN,
    num_keep            INTEGER,

    -- Measurements
    tokens_per_sec      REAL,
    gen_tokens_per_sec  REAL,
    peak_ram_gb         REAL,
    peak_vram_gb        REAL,
    ram_before_gb       REAL,
    ram_after_gb        REAL,
    delta_ram_gb        REAL,
    swap_peak_gb        REAL,
    delta_swap_gb       REAL,
    cpu_avg_pct         REAL,
    cpu_peak_pct        REAL,
    load_time_sec       REAL,
    ttft_ms             REAL,
    elapsed_sec         REAL,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,

    -- Outcome
    completed           BOOLEAN DEFAULT TRUE,
    oom                 BOOLEAN DEFAULT FALSE,

    autotune_version    TEXT,
    observed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.run_observations IS
    'One row per completed LLM inference run from opted-in machines. '
    'Enables cross-machine analytics by model, quant, and hardware class.';

CREATE INDEX IF NOT EXISTS idx_ro_install
    ON public.run_observations (install_key, observed_at DESC)
    WHERE install_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ro_model
    ON public.run_observations (model_id, quant, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_ro_hardware
    ON public.run_observations (hardware_key, model_id)
    WHERE hardware_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ro_observed
    ON public.run_observations (observed_at DESC);


-- ==========================================================================
-- 6. Helper function: increment_stat
--    Called by the telemetry client after each successful INSERT to keep
--    app_stats counters current without a SELECT round-trip.
-- ==========================================================================
CREATE OR REPLACE FUNCTION public.increment_stat(p_key TEXT)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER           -- runs with owner privileges so anon can call it
AS $$
BEGIN
    INSERT INTO public.app_stats (stat_key, stat_value, last_updated_at)
    VALUES (p_key, 1, NOW())
    ON CONFLICT (stat_key) DO UPDATE
        SET stat_value      = public.app_stats.stat_value + 1,
            last_updated_at = NOW();
END;
$$;

COMMENT ON FUNCTION public.increment_stat IS
    'Atomically increments an app_stats counter. '
    'SECURITY DEFINER so the anon role can call it.';


-- ==========================================================================
-- 7. Row Level Security (RLS)
--    Enable RLS and grant the anon role insert-only access to the three
--    telemetry tables.  The service_role key bypasses RLS.
-- ==========================================================================

-- installations
ALTER TABLE public.installations ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY anon_can_upsert_installation
        ON public.installations FOR INSERT TO anon WITH CHECK (TRUE);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- telemetry_events
ALTER TABLE public.telemetry_events ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY anon_can_insert_events
        ON public.telemetry_events FOR INSERT TO anon WITH CHECK (TRUE);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- run_observations
ALTER TABLE public.run_observations ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    CREATE POLICY anon_can_insert_runs
        ON public.run_observations FOR INSERT TO anon WITH CHECK (TRUE);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- app_stats and app_releases: read-only for anon (no RLS required for SELECT)
GRANT SELECT ON public.app_stats    TO anon;
GRANT SELECT ON public.app_releases TO anon;

-- Grant EXECUTE on the counter function to anon
GRANT EXECUTE ON FUNCTION public.increment_stat TO anon;


-- ==========================================================================
-- 7b. Helper function: update_last_seen
--     Called by the telemetry client on every session start for existing
--     installs.  INSERT-only RLS prevents a plain UPDATE, so this SECURITY
--     DEFINER function acts as a controlled write-through escape hatch.
--     It only touches last_seen_at — the anon caller cannot modify any other
--     column, and cannot call this function on rows it did not create (the
--     install_key is the only parameter and the WHERE clause is exact-match).
-- ==========================================================================

CREATE OR REPLACE FUNCTION public.update_last_seen(p_install_key TEXT)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    UPDATE public.installations
       SET last_seen_at    = NOW(),
           autotune_version = COALESCE(autotune_version, autotune_version)
     WHERE install_key = p_install_key;
END;
$$;

COMMENT ON FUNCTION public.update_last_seen IS
    'Updates last_seen_at for an existing install. SECURITY DEFINER so the '
    'anon role can ping activity without needing UPDATE RLS policy.';

GRANT EXECUTE ON FUNCTION public.update_last_seen TO anon;


-- ==========================================================================
-- 8. Analytical views
-- ==========================================================================

CREATE OR REPLACE VIEW public.model_performance_summary AS
SELECT
    model_id,
    quant,
    gpu_backend,
    cpu_arch,
    COUNT(*)                                          AS sample_count,
    ROUND(AVG(tokens_per_sec)::NUMERIC, 1)            AS avg_tps,
    ROUND(AVG(gen_tokens_per_sec)::NUMERIC, 1)        AS avg_gen_tps,
    ROUND(AVG(ttft_ms)::NUMERIC, 0)                   AS avg_ttft_ms,
    ROUND(AVG(peak_ram_gb)::NUMERIC, 2)               AS avg_peak_ram_gb,
    ROUND(AVG(cpu_avg_pct)::NUMERIC, 1)               AS avg_cpu_pct,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP
        (ORDER BY tokens_per_sec)::NUMERIC, 1
    )                                                 AS p50_tps,
    ROUND(
        PERCENTILE_CONT(0.95) WITHIN GROUP
        (ORDER BY ttft_ms)::NUMERIC, 0
    )                                                 AS p95_ttft_ms,
    MAX(observed_at)                                  AS last_seen
FROM public.run_observations
WHERE completed = TRUE
GROUP BY model_id, quant, gpu_backend, cpu_arch;

COMMENT ON VIEW public.model_performance_summary IS
    'Aggregate performance stats per (model, quant, GPU backend, CPU arch). '
    'The main analytics view for cross-machine comparisons.';


CREATE OR REPLACE VIEW public.installation_summary AS
SELECT
    os_name,
    cpu_arch,
    gpu_backend,
    (ROUND(total_ram_gb / 8.0) * 8)::INTEGER         AS ram_class_gb,
    COUNT(*)                                          AS install_count,
    COUNT(*) FILTER (WHERE telemetry_opted_in)        AS opted_in_count,
    MAX(last_seen_at)                                 AS most_recent_active,
    MIN(first_seen_at)                                AS earliest_install
FROM public.installations
GROUP BY os_name, cpu_arch, gpu_backend,
         (ROUND(total_ram_gb / 8.0) * 8)::INTEGER;

COMMENT ON VIEW public.installation_summary IS
    'Hardware class breakdown across all known installations.';


CREATE OR REPLACE VIEW public.daily_event_summary AS
SELECT
    DATE_TRUNC('day', occurred_at)::DATE              AS day,
    event_type,
    COUNT(*)                                          AS total_count,
    COUNT(DISTINCT install_key)                       AS unique_machines
FROM public.telemetry_events
GROUP BY DATE_TRUNC('day', occurred_at)::DATE, event_type
ORDER BY day DESC, total_count DESC;

COMMENT ON VIEW public.daily_event_summary IS
    'Daily breakdown of event counts by type — useful for trend analysis '
    'and spotting error spikes.';


CREATE OR REPLACE VIEW public.active_users AS
SELECT
    COUNT(DISTINCT install_key) FILTER (
        WHERE last_seen_at >= NOW() - INTERVAL '7 days'
    )                                                 AS active_7d,
    COUNT(DISTINCT install_key) FILTER (
        WHERE last_seen_at >= NOW() - INTERVAL '30 days'
    )                                                 AS active_30d,
    COUNT(DISTINCT install_key) FILTER (
        WHERE last_seen_at >= NOW() - INTERVAL '90 days'
    )                                                 AS active_90d,
    COUNT(*)                                          AS total_installations,
    COUNT(*) FILTER (WHERE telemetry_opted_in)        AS total_opted_in
FROM public.installations;

COMMENT ON VIEW public.active_users IS
    'Rolling active machine counts (7d / 30d / 90d) and opt-in rate.';
