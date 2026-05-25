"""Dashboard metrics aggregation — SQLite queries + live psutil stats."""
from __future__ import annotations

import datetime
import os
import shutil
import time
from typing import Any

import psutil

# ---------------------------------------------------------------------------
# PyPI version cache (1-hour TTL, shared across all requests in a process)
# ---------------------------------------------------------------------------
_PYPI_CACHE: dict = {}


def _latest_pypi_version() -> str | None:
    """Return the latest published llm-autotune version from PyPI, or None on error.

    Results are cached for 1 hour so the dashboard never blocks on a network call.
    """
    now = time.time()
    if _PYPI_CACHE.get("at", 0) + 3600 > now:
        return _PYPI_CACHE.get("v")
    try:
        import httpx
        r = httpx.get("https://pypi.org/pypi/llm-autotune/json", timeout=3.0)
        v: str = r.json()["info"]["version"]
        _PYPI_CACHE.update({"at": now, "v": v})
        return v
    except Exception:
        _PYPI_CACHE.update({"at": now, "v": None})
        return None


def _db():
    from autotune.db.store import get_db
    return get_db()


def get_overview() -> dict[str, Any]:
    """RAM, running models, request counts, avg perf for the last 24 h."""
    db = _db()
    now = time.time()
    day_ago = now - 86400

    vm = psutil.virtual_memory()

    running_models: list[dict] = []
    try:
        from autotune.api.running_models import get_running_models
        running_models = [
            {
                "name": m.name,
                "backend": m.backend,
                "ram_gb": round(m.ram_gb, 2),
                "loaded_since": m.loaded_since,
            }
            for m in get_running_models()
        ]
    except Exception:
        pass

    row = db.conn.execute(
        """SELECT
               COUNT(*)                         AS total_requests,
               AVG(ttft_ms)                     AS avg_ttft,
               AVG(gen_tokens_per_sec)          AS avg_tps,
               AVG(context_len)                 AS avg_ctx,
               SUM(COALESCE(prompt_tokens, 0))      AS total_prompt_tokens,
               SUM(COALESCE(completion_tokens, 0))  AS total_comp_tokens
           FROM run_observations
           WHERE observed_at > ?""",
        (day_ago,),
    ).fetchone()

    avg_ctx = (row["avg_ctx"] or 4096) if row else 4096
    kv_savings_pct = round((4096 - avg_ctx) / 4096 * 100, 1) if avg_ctx < 4096 else 0.0

    # TTFT percentiles — P50 (median) and P95 from last 24 h
    ttft_rows = db.conn.execute(
        "SELECT ttft_ms FROM run_observations WHERE observed_at > ? AND ttft_ms IS NOT NULL ORDER BY ttft_ms",
        (day_ago,),
    ).fetchall()
    ttft_vals = [r["ttft_ms"] for r in ttft_rows]
    n = len(ttft_vals)
    p50_ttft = round(ttft_vals[n // 2], 1) if ttft_vals else 0
    p95_ttft = round(ttft_vals[min(int(n * 0.95), n - 1)], 1) if ttft_vals else 0

    # Average total latency (elapsed time = full end-to-end response time)
    elapsed_row = db.conn.execute(
        "SELECT AVG(elapsed_sec) AS avg_el FROM run_observations WHERE observed_at > ? AND elapsed_sec IS NOT NULL AND elapsed_sec > 0",
        (day_ago,),
    ).fetchone()
    avg_elapsed_ms = round((elapsed_row["avg_el"] or 0) * 1000, 1) if elapsed_row else 0

    # Token breakdown
    prompt_tokens = int(row["total_prompt_tokens"] or 0) if row else 0
    comp_tokens   = int(row["total_comp_tokens"] or 0) if row else 0

    return {
        "ram": {
            "total_gb":    round(vm.total / 1024**3, 1),
            "available_gb": round(vm.available / 1024**3, 1),
            "used_pct":    round(vm.percent, 1),
        },
        "running_models":    running_models,
        "requests_today":    int(row["total_requests"] or 0) if row else 0,
        "avg_ttft_ms":       round(row["avg_ttft"] or 0, 1) if row else 0,
        "avg_tps":           round(row["avg_tps"] or 0, 1) if row else 0,
        "avg_context_len":   round(avg_ctx, 0),
        "kv_savings_pct":    kv_savings_pct,
        "total_tokens_today": prompt_tokens + comp_tokens,
        "prompt_tokens_today": prompt_tokens,
        "comp_tokens_today":   comp_tokens,
        "p50_ttft_ms":       p50_ttft,
        "p95_ttft_ms":       p95_ttft,
        "avg_elapsed_ms":    avg_elapsed_ms,
    }


def get_requests_timeseries() -> list[dict]:
    """Hourly request counts for the last 24 h, always returning 24 buckets."""
    db = _db()
    now = time.time()
    day_ago = now - 86400

    rows = db.conn.execute(
        """SELECT
               CAST(((observed_at - ?) / 3600) AS INTEGER) AS hour_bucket,
               COUNT(*)               AS count,
               AVG(ttft_ms)           AS avg_ttft,
               AVG(gen_tokens_per_sec) AS avg_tps
           FROM run_observations
           WHERE observed_at > ?
           GROUP BY hour_bucket
           ORDER BY hour_bucket""",
        (day_ago, day_ago),
    ).fetchall()

    buckets: dict[int, dict] = {}
    for r in rows:
        h = int(r["hour_bucket"])
        buckets[h] = {
            "count":    r["count"],
            "avg_ttft": round(r["avg_ttft"] or 0, 1),
            "avg_tps":  round(r["avg_tps"] or 0, 1),
        }

    result = []
    for i in range(24):
        ts = day_ago + i * 3600
        label = datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
        b = buckets.get(i, {"count": 0, "avg_ttft": 0, "avg_tps": 0})
        result.append({"hour": label, **b})
    return result


def get_ttft_trend() -> list[dict]:
    """Last 100 completed requests, chronological, for the TTFT sparkline."""
    db = _db()
    rows = db.conn.execute(
        """SELECT model_id, ttft_ms, gen_tokens_per_sec, elapsed_sec, context_len, observed_at
           FROM run_observations
           WHERE ttft_ms IS NOT NULL
           ORDER BY observed_at DESC
           LIMIT 100""",
    ).fetchall()

    result = []
    for r in reversed(rows):
        result.append({
            "model":       r["model_id"],
            "ttft_ms":     round(r["ttft_ms"] or 0, 1),
            "tps":         round(r["gen_tokens_per_sec"] or 0, 1),
            "elapsed_sec": round(r["elapsed_sec"] or 0, 2),
            "context_len": r["context_len"],
            "time": datetime.datetime.fromtimestamp(r["observed_at"]).strftime("%H:%M:%S"),
            "observed_at_ts": r["observed_at"],  # raw Unix float
        })
    return result


def get_models_stats() -> list[dict]:
    """Per-model aggregate stats with proper P50/P95 latency, sorted by request count desc."""
    db = _db()

    # Fetch all TTFT values per model (sorted) for percentile computation in Python
    all_ttft = db.conn.execute(
        "SELECT model_id, ttft_ms FROM run_observations WHERE ttft_ms IS NOT NULL ORDER BY model_id, ttft_ms"
    ).fetchall()
    from collections import defaultdict
    ttft_by_model: dict[str, list[float]] = defaultdict(list)
    for r in all_ttft:
        ttft_by_model[r["model_id"]].append(r["ttft_ms"])

    def _pct(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        i = min(int(len(vals) * p / 100), len(vals) - 1)
        return round(vals[i], 1)

    rows = db.conn.execute(
        """SELECT
               model_id,
               COUNT(*)                                      AS requests,
               AVG(ttft_ms)                                  AS avg_ttft,
               AVG(gen_tokens_per_sec)                       AS avg_tps,
               MAX(gen_tokens_per_sec)                       AS max_tps,
               AVG(context_len)                              AS avg_ctx,
               AVG(elapsed_sec)                              AS avg_elapsed,
               SUM(COALESCE(prompt_tokens, 0))               AS total_prompt,
               SUM(COALESCE(completion_tokens, 0))           AS total_comp,
               MAX(observed_at)                              AS last_seen
           FROM run_observations
           GROUP BY model_id
           ORDER BY requests DESC""",
    ).fetchall()

    result = []
    for r in rows:
        mid   = r["model_id"]
        vals  = ttft_by_model.get(mid, [])
        last  = r["last_seen"]
        result.append({
            "model_id":        mid,
            "requests":        r["requests"],
            "avg_ttft_ms":     round(r["avg_ttft"] or 0, 1),
            "p50_ttft_ms":     _pct(vals, 50),
            "p95_ttft_ms":     _pct(vals, 95),
            "avg_tps":         round(r["avg_tps"] or 0, 1),
            "max_tps":         round(r["max_tps"] or 0, 1),
            "avg_context_len": round(r["avg_ctx"] or 0, 0),
            "avg_elapsed_sec": round(r["avg_elapsed"] or 0, 2),
            "total_tokens":    int((r["total_prompt"] or 0) + (r["total_comp"] or 0)),
            "total_prompt":    int(r["total_prompt"] or 0),
            "total_comp":      int(r["total_comp"] or 0),
            "last_seen_ts":    last,
        })
    return result


def get_token_timeseries() -> list[dict]:
    """Hourly prompt + completion token counts for the last 24 h, always 24 buckets."""
    db = _db()
    now = time.time()
    day_ago = now - 86400

    rows = db.conn.execute(
        """SELECT
               CAST(((observed_at - ?) / 3600) AS INTEGER) AS hour_bucket,
               SUM(COALESCE(prompt_tokens, 0))              AS prompt_tokens,
               SUM(COALESCE(completion_tokens, 0))          AS comp_tokens
           FROM run_observations
           WHERE observed_at > ?
           GROUP BY hour_bucket
           ORDER BY hour_bucket""",
        (day_ago, day_ago),
    ).fetchall()

    buckets: dict[int, dict] = {int(r["hour_bucket"]): r for r in rows}
    result = []
    for i in range(24):
        ts    = day_ago + i * 3600
        label = datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
        b = buckets.get(i)
        result.append({
            "hour":             label,
            "prompt_tokens":    int(b["prompt_tokens"] or 0) if b else 0,
            "completion_tokens": int(b["comp_tokens"] or 0) if b else 0,
        })
    return result


def get_usage_summary() -> dict[str, Any]:
    """Aggregated usage for the Usage tab KPIs and per-model share chart."""
    db = _db()
    now = time.time()
    day_ago = now - 86400

    # Per-model requests in last 24 h
    model_rows = db.conn.execute(
        """SELECT model_id, COUNT(*) AS requests,
                  SUM(COALESCE(prompt_tokens, 0)) AS prompt,
                  SUM(COALESCE(completion_tokens, 0)) AS comp
           FROM run_observations
           WHERE observed_at > ?
           GROUP BY model_id
           ORDER BY requests DESC""",
        (day_ago,),
    ).fetchall()

    model_share = [
        {
            "model":    r["model_id"],
            "requests": r["requests"],
            "tokens":   int((r["prompt"] or 0) + (r["comp"] or 0)),
        }
        for r in model_rows
    ]

    # Peak requests per hour in last 24 h
    hour_counts = db.conn.execute(
        """SELECT COUNT(*) AS n
           FROM run_observations
           WHERE observed_at > ?
           GROUP BY CAST(((observed_at - ?) / 3600) AS INTEGER)""",
        (day_ago, day_ago),
    ).fetchall()
    peak_rph = max((r["n"] for r in hour_counts), default=0)

    # Active API key count
    try:
        active_keys = db.conn.execute(
            "SELECT COUNT(*) AS n FROM api_keys WHERE is_active=1"
        ).fetchone()["n"]
    except Exception:
        active_keys = 0

    # 7-day rolling request total
    week_ago  = now - 7 * 86400
    week_row  = db.conn.execute(
        "SELECT COUNT(*) AS n FROM run_observations WHERE observed_at > ?",
        (week_ago,),
    ).fetchone()
    requests_7d = int(week_row["n"] or 0) if week_row else 0

    total_24h = sum(m["requests"] for m in model_share)
    total_tokens_24h = sum(m["tokens"] for m in model_share)

    return {
        "requests_today":   total_24h,
        "tokens_today":     total_tokens_24h,
        "requests_7d":      requests_7d,
        "peak_rph":         peak_rph,
        "active_keys":      active_keys,
        "model_share":      model_share,
    }


def get_comparison() -> dict[str, Any]:
    """Autotune dynamic context vs Ollama's 4096-token fixed default."""
    db = _db()
    row = db.conn.execute(
        """SELECT
               COUNT(*)                                                         AS total_runs,
               AVG(context_len)                                                 AS avg_tuned_ctx,
               SUM(CASE WHEN context_len < 4096 THEN 1 ELSE 0 END)             AS runs_below_4096,
               AVG(CASE WHEN context_len < 4096
                        THEN (4096.0 - context_len) / 4096 * 100
                        ELSE 0 END)                                             AS avg_kv_savings_pct,
               AVG(ttft_ms)                                                     AS avg_ttft
           FROM run_observations""",
    ).fetchone()

    avg_ctx    = row["avg_tuned_ctx"] or 4096
    baseline   = 4096
    kv_ratio   = avg_ctx / baseline if baseline > 0 else 1.0
    kv_savings = row["avg_kv_savings_pct"] or 0.0

    return {
        "baseline_context":     baseline,
        "avg_tuned_context":    round(avg_ctx, 0),
        "context_reduction_pct": round((1 - kv_ratio) * 100, 1),
        "kv_memory_ratio":      round(kv_ratio, 3),
        "avg_kv_savings_pct":   round(kv_savings, 1),
        "runs_optimized":       int(row["runs_below_4096"] or 0),
        "total_runs":           int(row["total_runs"] or 0),
        "avg_ttft_ms":          round(row["avg_ttft"] or 0, 1),
        "note": (
            "Baseline is Ollama's fixed 4096-token default context. "
            "autotune sizes num_ctx dynamically per request — "
            "smaller context = smaller KV cache = lower TTFT and less RAM pressure."
        ),
    }


def get_api_keys_summary() -> list[dict]:
    """Active API keys with today's request and token counts."""
    db = _db()
    today = datetime.date.today().isoformat()

    rows = db.conn.execute(
        """SELECT
               k.id,
               k.name,
               k.key_prefix,
               k.is_active,
               k.created_at,
               k.last_used_at,
               COALESCE(u.requests_today, 0) AS requests_today,
               COALESCE(u.tokens_today,   0) AS tokens_today
           FROM api_keys k
           LEFT JOIN (
               SELECT key_id,
                      COUNT(*)                                     AS requests_today,
                      SUM(prompt_tokens + completion_tokens)       AS tokens_today
               FROM api_key_usage
               WHERE day = ?
               GROUP BY key_id
           ) u ON u.key_id = k.id
           WHERE k.is_active = 1
           ORDER BY k.created_at DESC""",
        (today,),
    ).fetchall()

    result = []
    for r in rows:
        result.append({
            "id":             r["id"],
            "name":           r["name"],
            "key_prefix":     r["key_prefix"],
            "requests_today": int(r["requests_today"]),
            "tokens_today":   int(r["tokens_today"]),
            "last_used_at": (
                datetime.datetime.fromtimestamp(r["last_used_at"]).strftime("%Y-%m-%d %H:%M")
                if r["last_used_at"] else None
            ),
            "created_at": (
                datetime.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d")
                if r["created_at"] else None
            ),
            "last_used_ts": r["last_used_at"],   # raw Unix float, may be None
            "created_ts":   r["created_at"],     # raw Unix float
        })
    return result


def get_slow_requests(threshold_sec: float = 5.0) -> list[dict]:
    """Most recent requests where total elapsed time exceeded threshold_sec."""
    db = _db()
    rows = db.conn.execute(
        """SELECT model_id, elapsed_sec, ttft_ms, context_len,
                  prompt_tokens, completion_tokens, profile_name, observed_at
           FROM run_observations
           WHERE elapsed_sec > ?
           ORDER BY observed_at DESC
           LIMIT 50""",
        (threshold_sec,),
    ).fetchall()

    result = []
    for r in rows:
        result.append({
            "model_id":         r["model_id"],
            "elapsed_sec":      round(r["elapsed_sec"] or 0, 2),
            "ttft_ms":          round(r["ttft_ms"] or 0, 1),
            "context_len":      r["context_len"],
            "prompt_tokens":    r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "profile_name":     r["profile_name"],
            "time": datetime.datetime.fromtimestamp(r["observed_at"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "observed_at_ts": r["observed_at"],  # raw Unix float
        })
    return result


def get_suggestions() -> list[dict]:
    """Rule-based suggestions derived entirely from real observed data."""
    db = _db()
    suggestions: list[dict] = []
    now = time.time()

    # ── TTFT check (last hour, ≥3 requests) ──────────────────────────────
    row = db.conn.execute(
        """SELECT AVG(ttft_ms) AS avg_ttft, COUNT(*) AS n
           FROM run_observations
           WHERE observed_at > ? AND ttft_ms IS NOT NULL""",
        (now - 3600,),
    ).fetchone()

    if row and row["n"] >= 3:
        avg_ttft = row["avg_ttft"] or 0
        if avg_ttft > 2000:
            suggestions.append({
                "level": "warning",
                "title": "High TTFT detected",
                "body": (
                    f"Average first-token latency is {avg_ttft:.0f} ms (last hour). "
                    "This is above the 2 s threshold. Switching to the 'fast' profile "
                    "or a more quantized model can reduce it significantly."
                ),
                "action": 'Use X-Autotune-Profile: fast header in your requests',
            })
        elif avg_ttft < 300:
            suggestions.append({
                "level": "ok",
                "title": "TTFT is excellent",
                "body": f"Average {avg_ttft:.0f} ms — your model is well-tuned for your hardware.",
                "action": None,
            })

    # ── RAM pressure ──────────────────────────────────────────────────────
    vm = psutil.virtual_memory()
    if vm.percent > 88:
        suggestions.append({
            "level": "warning",
            "title": "High system RAM usage",
            "body": (
                f"RAM is {vm.percent:.0f}% full ({vm.available / 1024**3:.1f} GB free). "
                "The model may be competing with OS memory. Consider a smaller model "
                "or Q4_K_M quantization to reduce footprint."
            ),
            "action": "Try: autotune pull qwen3:4b",
        })
    elif vm.percent > 75:
        suggestions.append({
            "level": "info",
            "title": "RAM is getting full",
            "body": (
                f"System RAM is {vm.percent:.0f}% used. "
                "autotune's dynamic context sizing is already limiting KV cache growth — "
                "no action needed unless you see slowdowns."
            ),
            "action": None,
        })

    # ── Context / KV savings info ─────────────────────────────────────────
    ctx_row = db.conn.execute(
        """SELECT AVG(context_len) AS avg_ctx, COUNT(*) AS n
           FROM run_observations
           WHERE observed_at > ?""",
        (now - 86400,),
    ).fetchone()

    if ctx_row and ctx_row["n"] and ctx_row["avg_ctx"]:
        avg_ctx = ctx_row["avg_ctx"]
        if avg_ctx < 3500:
            savings = round((1 - avg_ctx / 4096) * 100, 1)
            suggestions.append({
                "level": "info",
                "title": "Dynamic context is saving KV memory",
                "body": (
                    f"Average allocated context: {avg_ctx:.0f} tokens vs Ollama's 4096 default — "
                    f"{savings}% KV cache reduction. This directly lowers TTFT and RAM usage."
                ),
                "action": None,
            })

    # ── Slow requests in last hour ────────────────────────────────────────
    slow = db.conn.execute(
        """SELECT COUNT(*) AS n
           FROM run_observations
           WHERE elapsed_sec > 10 AND observed_at > ?""",
        (now - 3600,),
    ).fetchone()

    if slow and slow["n"] > 0:
        suggestions.append({
            "level": "warning",
            "title": f"{slow['n']} slow request(s) in the last hour",
            "body": (
                "Requests taking >10 s indicate memory pressure or very long generations. "
                "Check the Slow Requests panel for details, and consider the 'fast' profile."
            ),
            "action": "See slow requests panel below",
        })

    # ── No data yet ───────────────────────────────────────────────────────
    if not suggestions:
        total = db.conn.execute(
            "SELECT COUNT(*) AS n FROM run_observations"
        ).fetchone()
        if not total or not total["n"]:
            suggestions.append({
                "level": "info",
                "title": "No requests recorded yet",
                "body": (
                    "Send requests through autotune (port 8765) to see "
                    "personalized optimization suggestions here."
                ),
                "action": None,
            })
        else:
            suggestions.append({
                "level": "ok",
                "title": "Everything looks good",
                "body": "No anomalies detected in recent requests.",
                "action": None,
            })

    return suggestions


def _make_session(reqs: list[dict]) -> dict[str, Any]:
    models = list(dict.fromkeys(r["model"] for r in reqs))
    ttfts  = [r["ttft_ms"] for r in reqs if r["ttft_ms"]]
    elapsed = [r["elapsed_sec"] for r in reqs if r["elapsed_sec"]]
    return {
        "start_ts":      reqs[0]["ts"],
        "end_ts":        reqs[-1]["ts"],
        "request_count": len(reqs),
        "models":        models,
        "total_tokens":  sum(r["total_tokens"] for r in reqs),
        "avg_ttft_ms":   round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
        "avg_elapsed_sec": round(sum(elapsed) / len(elapsed), 2) if elapsed else None,
    }


def get_recent_activity(limit: int = 100) -> dict[str, Any]:
    """Recent request feed + auto-grouped sessions (gap > 5 min = new session)."""
    db = _db()
    now = time.time()
    rows = db.conn.execute(
        """SELECT model_id, ttft_ms, gen_tokens_per_sec, elapsed_sec,
                  context_len, prompt_tokens, completion_tokens,
                  profile_name, observed_at
           FROM run_observations
           ORDER BY observed_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()

    requests: list[dict] = []
    for r in rows:
        requests.append({
            "model":         r["model_id"],
            "ttft_ms":       round(r["ttft_ms"], 1) if r["ttft_ms"] else None,
            "elapsed_sec":   round(r["elapsed_sec"] or 0, 2),
            "tps":           round(r["gen_tokens_per_sec"] or 0, 1),
            "context_len":   r["context_len"],
            "total_tokens":  (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0),
            "profile":       r["profile_name"],
            "ts":            r["observed_at"],
        })

    # Active now: requests in last 10 minutes
    ten_min_ago = now - 600
    active_models: list[str] = []
    active_count = 0
    for req in requests:
        if req["ts"] >= ten_min_ago:
            active_count += 1
            if req["model"] not in active_models:
                active_models.append(req["model"])

    # Group requests into sessions (ascending order, gap > 5 min = new session)
    sessions: list[dict] = []
    if requests:
        asc = sorted(requests, key=lambda x: x["ts"])
        GAP = 300  # 5 minutes
        buf: list[dict] = [asc[0]]
        for req in asc[1:]:
            if req["ts"] - buf[-1]["ts"] > GAP:
                sessions.append(_make_session(buf))
                buf = []
            buf.append(req)
        if buf:
            sessions.append(_make_session(buf))
        sessions.reverse()  # newest first

    return {
        "requests":      requests,
        "sessions":      sessions[:30],
        "active_now":    {"count": active_count, "models": active_models},
    }


def get_installed_models() -> list[dict[str, Any]]:
    """Installed Ollama models from the /api/tags endpoint, or [] if unavailable."""
    try:
        import httpx
        ollama_url = os.environ.get("AUTOTUNE_OLLAMA_URL", "http://localhost:11434").rstrip("/")
        r = httpx.get(f"{ollama_url}/api/tags", timeout=3.0)
        result = []
        for m in r.json().get("models", []):
            details = m.get("details", {})
            mod_ts: float | None = None
            try:
                mod_raw = m.get("modified_at", "")
                if mod_raw:
                    mod_ts = datetime.datetime.fromisoformat(
                        mod_raw.replace("Z", "+00:00")
                    ).timestamp()
            except Exception:
                pass
            result.append({
                "name":           m.get("name", ""),
                "size_gb":        round(m.get("size", 0) / 1024 ** 3, 2),
                "modified_ts":    mod_ts,
                "family":         details.get("family", ""),
                "parameter_size": details.get("parameter_size", ""),
                "quantization":   details.get("quantization_level", ""),
            })
        result.sort(key=lambda x: x["name"])
        return result
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Gateway security / health checker
# ---------------------------------------------------------------------------

def get_gateway_security() -> list[dict[str, Any]]:
    """Return a list of gateway security and health check results.

    Each item is a dict with keys: check, status (ok|warn|error|info),
    message, action (str or None).
    """
    checks: list[dict[str, Any]] = []

    def _check(name: str, status: str, message: str, action: str | None = None) -> None:
        checks.append({"check": name, "status": status, "message": message, "action": action})

    # ── 1. API key enforcement ────────────────────────────────────────────
    api_key_on = os.environ.get("AUTOTUNE_REQUIRE_API_KEY", "").strip() in ("1", "true", "yes")
    if api_key_on:
        _check(
            "API Key Enforcement", "ok",
            "AUTOTUNE_REQUIRE_API_KEY=1 — all /v1/* requests require a valid API key.",
        )
    else:
        _check(
            "API Key Enforcement", "warn",
            "Enforcement is OFF. Any client that can reach this port can use your models.",
            "Set AUTOTUNE_REQUIRE_API_KEY=1 in your .env file.",
        )

    # ── 2. Admin key strength ─────────────────────────────────────────────
    admin_key = os.environ.get("AUTOTUNE_ADMIN_KEY", "").strip()
    if not admin_key:
        _check(
            "Admin Key", "error",
            "AUTOTUNE_ADMIN_KEY is not set. The admin API and dashboard are unprotected.",
            "Run ./setup.sh to generate a strong key.",
        )
    elif len(admin_key) < 32:
        _check(
            "Admin Key", "warn",
            f"Admin key is only {len(admin_key)} characters. Use at least 32 random characters.",
            "python3 -c \"import secrets; print(secrets.token_urlsafe(48))\"",
        )
    else:
        _check(
            "Admin Key", "ok",
            f"Admin key is set ({len(admin_key)} chars) and dashboard login is active.",
        )

    # ── 3. Active API keys (only checked when enforcement is on) ──────────
    if api_key_on:
        try:
            n = _db().conn.execute(
                "SELECT COUNT(*) AS n FROM api_keys WHERE is_active=1"
            ).fetchone()["n"]
            if n == 0:
                _check(
                    "Active Keys", "warn",
                    "Enforcement is ON but no active API keys exist — every request will be rejected.",
                    "Use the New Key button in the dashboard to create a key.",
                )
            else:
                _check(
                    "Active Keys", "ok",
                    f"{n} active API key{'s' if n != 1 else ''} configured.",
                )
        except Exception:
            pass

    # ── 4. RAM pressure ───────────────────────────────────────────────────
    vm = psutil.virtual_memory()
    if vm.percent > 88:
        _check(
            "RAM", "warn",
            f"System RAM is {vm.percent:.0f}% used ({vm.available / 1024**3:.1f} GB free). "
            "Models may be competing with OS memory.",
            "Consider a smaller model or Q4_K_M quantization.",
        )
    else:
        _check(
            "RAM", "ok",
            f"RAM at {vm.percent:.0f}% — {vm.available / 1024**3:.1f} GB free.",
        )

    # ── 5. Disk space ─────────────────────────────────────────────────────
    try:
        disk = shutil.disk_usage("/")
        free_gb = disk.free / 1024 ** 3
        if free_gb < 5:
            _check(
                "Disk Space", "error",
                f"Only {free_gb:.1f} GB free on root disk. Model pulls and logs may fail.",
                "Free space before pulling new models.",
            )
        elif free_gb < 15:
            _check(
                "Disk Space", "warn",
                f"{free_gb:.1f} GB free — may be tight for large models (7B+ need 4-8 GB each).",
                "ollama rm <model-name>  to remove unused models.",
            )
        else:
            _check("Disk Space", "ok", f"{free_gb:.0f} GB free on root disk.")
    except Exception:
        pass

    # ── 6. Version currency ───────────────────────────────────────────────
    try:
        from importlib.metadata import version as _pkgver
        from packaging.version import Version

        current = _pkgver("llm-autotune")
        latest = _latest_pypi_version()
        if latest and Version(latest) > Version(current):
            _check(
                "Version", "info",
                f"Update available: v{current} → v{latest}",
                "pip install --upgrade llm-autotune",
            )
        elif latest:
            _check("Version", "ok", f"autotune v{current} is up to date.")
        else:
            _check("Version", "info", f"Running autotune v{current} (PyPI check unavailable).")
    except Exception:
        pass

    # ── 7. TLS reminder ───────────────────────────────────────────────────
    _check(
        "TLS / HTTPS", "info",
        "Gateway is running on plain HTTP. For team deployments, add Nginx or Caddy in front.",
        "See docs/team-tls.md for a Nginx + Let's Encrypt setup guide.",
    )

    return checks


def get_key_usage_trend(key_id: str, days: int = 30) -> list[dict]:
    """Return daily request and token counts for a specific API key.

    Returns one dict per calendar day (oldest first) — days with no usage
    are included with counts of zero so the chart timeline is continuous.
    """
    db = _db()
    today = datetime.date.today()
    since = (today - datetime.timedelta(days=days - 1)).isoformat()

    rows = db.conn.execute(
        """SELECT day,
                  COUNT(*)                                  AS requests,
                  SUM(prompt_tokens + completion_tokens)    AS tokens,
                  AVG(ttft_ms)                              AS avg_ttft
           FROM api_key_usage
           WHERE key_id = ? AND day >= ?
           GROUP BY day
           ORDER BY day ASC""",
        (key_id, since),
    ).fetchall()

    day_map = {r["day"]: r for r in rows}
    result = []
    for i in range(days):
        day = (today - datetime.timedelta(days=days - 1 - i)).isoformat()
        r = day_map.get(day)
        result.append({
            "day": day,
            "requests": int(r["requests"]) if r else 0,
            "tokens": int(r["tokens"] or 0) if r else 0,
            "avg_ttft_ms": round(r["avg_ttft"], 1) if r and r["avg_ttft"] else None,
        })
    return result
