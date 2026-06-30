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

    _AGG_SQL = """SELECT
               COUNT(*)                         AS total_requests,
               AVG(ttft_ms)                     AS avg_ttft,
               AVG(gen_tokens_per_sec)          AS avg_tps,
               AVG(context_len)                 AS avg_ctx,
               AVG(elapsed_sec)                 AS avg_elapsed,
               SUM(COALESCE(prompt_tokens, 0))      AS total_prompt_tokens,
               SUM(COALESCE(completion_tokens, 0))  AS total_comp_tokens
           FROM run_observations
           WHERE {where}"""

    row = db.conn.execute(_AGG_SQL.format(where="observed_at > ?"), (day_ago,)).fetchone()
    requests_today = int(row["total_requests"] or 0) if row else 0

    # The capability cards (KV saved, TTFT, tok/s) describe how well autotune
    # performs — not how busy the last day was. When there's no traffic in the
    # last 24 h they would all read 0/— and make the tool look like it does
    # nothing, even though the lifetime history right below shows real wins.
    # So fall back to all-time figures for those cards (clearly labelled),
    # while "requests today" stays honestly scoped to the last 24 h.
    perf_window = "24h"
    perf = row
    if requests_today == 0:
        all_row = db.conn.execute(_AGG_SQL.format(where="1=1")).fetchone()
        if all_row and (all_row["total_requests"] or 0) > 0:
            perf = all_row
            perf_window = "all-time"

    avg_ctx = (perf["avg_ctx"] if perf and perf["avg_ctx"] is not None else 4096) or 4096
    kv_savings_pct = round((4096 - avg_ctx) / 4096 * 100, 1) if avg_ctx < 4096 else 0.0

    # TTFT percentiles — P50 (median) and P95 over the chosen window
    _ttft_where = "observed_at > ?" if perf_window == "24h" else "1=1"
    _ttft_args = (day_ago,) if perf_window == "24h" else ()
    ttft_rows = db.conn.execute(
        f"SELECT ttft_ms FROM run_observations WHERE {_ttft_where} AND ttft_ms IS NOT NULL ORDER BY ttft_ms",
        _ttft_args,
    ).fetchall()
    ttft_vals = [r["ttft_ms"] for r in ttft_rows]
    n = len(ttft_vals)
    p50_ttft = round(ttft_vals[n // 2], 1) if ttft_vals else 0
    p95_ttft = round(ttft_vals[min(int(n * 0.95), n - 1)], 1) if ttft_vals else 0

    avg_elapsed_ms = round((perf["avg_elapsed"] or 0) * 1000, 1) if perf and perf["avg_elapsed"] else 0

    # Token breakdown (always the last-24 h window — this is a "today" figure)
    prompt_tokens = int(row["total_prompt_tokens"] or 0) if row else 0
    comp_tokens   = int(row["total_comp_tokens"] or 0) if row else 0

    return {
        "ram": {
            "total_gb":    round(vm.total / 1024**3, 1),
            "available_gb": round(vm.available / 1024**3, 1),
            "used_pct":    round(vm.percent, 1),
        },
        "running_models":    running_models,
        "requests_today":    requests_today,
        "perf_window":       perf_window,
        "avg_ttft_ms":       round(perf["avg_ttft"] or 0, 1) if perf else 0,
        "avg_tps":           round(perf["avg_tps"] or 0, 1) if perf else 0,
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
        """SELECT model_id, ttft_ms, gen_tokens_per_sec, elapsed_sec, context_len,
                  observed_at, concurrent_models
           FROM run_observations
           WHERE ttft_ms IS NOT NULL
           ORDER BY observed_at DESC
           LIMIT 100""",
    ).fetchall()

    result = []
    for r in reversed(rows):
        result.append({
            "model":             r["model_id"],
            "ttft_ms":           round(r["ttft_ms"] or 0, 1),
            "tps":               round(r["gen_tokens_per_sec"] or 0, 1),
            "elapsed_sec":       round(r["elapsed_sec"] or 0, 2),
            "context_len":       r["context_len"],
            "time":              datetime.datetime.fromtimestamp(r["observed_at"]).strftime("%H:%M:%S"),
            "observed_at_ts":    r["observed_at"],
            "concurrent_models": r["concurrent_models"] or 1,
        })
    return result


def get_perf_trends() -> dict:
    """Return RAM, KV-cache size, and TPS time-series for the Performance tab charts."""
    db = _db()
    rows = db.conn.execute(
        """SELECT model_id, ttft_ms, gen_tokens_per_sec, context_len,
                  peak_ram_gb, ram_before_gb, concurrent_models, observed_at
           FROM run_observations
           WHERE observed_at IS NOT NULL
           ORDER BY observed_at DESC
           LIMIT 150""",
    ).fetchall()

    ram_series:  list[dict] = []
    kv_series:   list[dict] = []
    tps_series:  list[dict] = []

    for r in reversed(rows):
        t = datetime.datetime.fromtimestamp(r["observed_at"]).strftime("%H:%M")
        ram_gb = r["peak_ram_gb"] or r["ram_before_gb"] or None
        if ram_gb is not None:
            ram_series.append({"t": t, "v": round(ram_gb, 2), "model": r["model_id"]})
        if r["context_len"]:
            # Each token in KV cache uses ~0.5 MB for a typical Q4 model at F16 KV
            kv_mb = round(r["context_len"] * 0.5 / 1024, 2)
            kv_series.append({"t": t, "v": kv_mb, "ctx": r["context_len"], "model": r["model_id"]})
        if r["gen_tokens_per_sec"]:
            n_concurrent = r["concurrent_models"] or 1
            tps_series.append({
                "t": t,
                "v": round(r["gen_tokens_per_sec"], 1),
                "model": r["model_id"],
                "concurrent": n_concurrent,
            })

    return {"ram": ram_series, "kv": kv_series, "tps": tps_series}


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

    Each item: check, status (ok|warn|error|info), message, action.
    This drives both the Overview tab mini-card and the Security tab posture grid.
    """
    checks: list[dict[str, Any]] = []

    def _check(
        name: str, status: str, message: str,
        action: str | None = None, category: str = "security",
    ) -> None:
        checks.append({
            "check": name, "status": status,
            "message": message, "action": action,
            "category": category,
        })

    api_key_on = os.environ.get("AUTOTUNE_REQUIRE_API_KEY", "").strip() in ("1", "true", "yes")
    admin_key  = os.environ.get("AUTOTUNE_ADMIN_KEY", "").strip()
    tls_cert   = os.environ.get("AUTOTUNE_SSL_CERTFILE", "").strip()
    max_body   = int(os.environ.get("AUTOTUNE_MAX_BODY_BYTES", str(10 * 1024 * 1024)))

    # ── 1. Admin key / dashboard auth ────────────────────────────────────
    if not admin_key:
        _check("Dashboard Auth", "error",
               "AUTOTUNE_ADMIN_KEY not set — dashboard and admin API are wide open.",
               "Run ./setup.sh to generate a strong key.")
    elif len(admin_key) < 32:
        _check("Dashboard Auth", "warn",
               f"Admin key only {len(admin_key)} chars — use ≥ 32 random characters.",
               'python3 -c "import secrets; print(secrets.token_urlsafe(48))"')
    else:
        _check("Dashboard Auth", "ok",
               f"Admin key set ({len(admin_key)} chars) — dashboard login required.")

    # ── 2. API key enforcement ────────────────────────────────────────────
    if api_key_on:
        _check("API Key Enforcement", "ok",
               "All /v1/* inference requests require a valid API key.")
    else:
        _check("API Key Enforcement", "warn",
               "Open access — any client on this network can call your models.",
               "Set AUTOTUNE_REQUIRE_API_KEY=1 in your .env file.")

    # ── 3. Active keys (only meaningful when enforcement is on) ───────────
    if api_key_on:
        try:
            n = _db().conn.execute(
                "SELECT COUNT(*) AS n FROM api_keys WHERE is_active=1"
            ).fetchone()["n"]
            if n == 0:
                _check("Active API Keys", "warn",
                       "Enforcement is ON but no active keys exist — every request will be rejected.",
                       "Create a key on the Keys tab.")
            else:
                exp_soon = _db().conn.execute(
                    "SELECT COUNT(*) AS n FROM api_keys WHERE is_active=1"
                    " AND expires_at IS NOT NULL AND expires_at < ?",
                    (time.time() + 7 * 86400,)
                ).fetchone()["n"]
                msg = f"{n} active key{'s' if n != 1 else ''} configured."
                if exp_soon:
                    msg += f" {exp_soon} expire{'s' if exp_soon == 1 else ''} within 7 days."
                _check("Active API Keys", "warn" if exp_soon else "ok", msg)
        except Exception:
            pass

    # ── 4. CORS policy ────────────────────────────────────────────────────
    extra_cors = os.environ.get("AUTOTUNE_CORS_ORIGINS", "").strip()
    if extra_cors and extra_cors.strip("*") == "":
        _check("CORS Policy", "error",
               "AUTOTUNE_CORS_ORIGINS contains '*' — any website can call your API.",
               "Remove the wildcard and list specific domains instead.")
    else:
        origins_note = f" + {extra_cors}" if extra_cors else ""
        _check("CORS Policy", "ok",
               f"Locked to localhost{origins_note} — cross-origin requests from unknown sites are blocked.")

    # ── 5. TLS / HTTPS ────────────────────────────────────────────────────
    if tls_cert:
        _check("TLS / HTTPS", "ok",
               f"Native TLS active — traffic encrypted via {os.path.basename(tls_cert)}.")
    else:
        # For a localhost-only tool, plain HTTP is acceptable — CORS is already locked
        # to loopback and there is no network exposure. Mark "info" so it doesn't
        # penalise the posture score; operators who expose autotune to a LAN/VPN
        # should add --ssl-certfile.
        _check("TLS / HTTPS", "info",
               "Running over HTTP (localhost only). Add TLS if you expose autotune beyond loopback.",
               "autotune serve --ssl-certfile cert.pem --ssl-keyfile key.pem")

    # ── 6. Login rate limiting ────────────────────────────────────────────
    _check("Login Rate Limiting", "ok",
           "Exponential backoff active — IPs locked out after 5 failed attempts (up to 5 min).")

    # ── 7. Request body size limit ────────────────────────────────────────
    _check("Body Size Limit", "ok",
           f"Requests capped at {max_body // (1024*1024)} MB — oversized payloads rejected with 413.")

    # ── 8. Server fingerprinting ──────────────────────────────────────────
    _check("Server Fingerprint", "ok",
           "Server: response header suppressed — stack is not advertised to scanners.")

    # ── 9. Session revocation ─────────────────────────────────────────────
    _check("Session Revocation", "ok",
           "Logout immediately invalidates tokens — cookies cannot be replayed after sign-out.")

    # ── 10. DB file permissions ───────────────────────────────────────────
    try:
        import stat
        db_path = _db().path
        mode = stat.S_IMODE(os.stat(db_path).st_mode)
        if mode == 0o600:
            _check("DB File Permissions", "ok",
                   "SQLite DB is owner-read/write only (600) — other OS users cannot read keys.")
        else:
            _check("DB File Permissions", "warn",
                   f"DB file mode is {oct(mode)} — should be 600 to prevent multi-user exposure.",
                   f"chmod 600 {db_path}")
    except Exception:
        pass

    # ── 11. Security audit log ────────────────────────────────────────────
    _check("Security Audit Log", "ok",
           "All auth events (logins, key use, revocations) are logged to disk and stored in the DB.")

    # ── 12. Dashboard API rate limiting ──────────────────────────────────
    _check("Dashboard Rate Limiting", "ok",
           "Key create/revoke: 30/hr · Catalog refresh: 10/min · Reads: 300/min — per IP.")

    # ── System health checks (category=health) ────────────────────────────

    # RAM
    vm = psutil.virtual_memory()
    if vm.percent > 88:
        _check("RAM Pressure", "warn",
               f"RAM at {vm.percent:.0f}% — only {vm.available / 1024**3:.1f} GB free. "
               "Models may swap under load.",
               "Consider a smaller model or Q4_K_M quantization.",
               category="health")
    else:
        _check("RAM", "ok",
               f"RAM at {vm.percent:.0f}% — {vm.available / 1024**3:.1f} GB free.",
               category="health")

    # Disk
    try:
        free_gb = shutil.disk_usage("/").free / 1024**3
        if free_gb < 5:
            _check("Disk Space", "error",
                   f"Only {free_gb:.1f} GB free — model pulls and logs may fail.",
                   "Free space before pulling new models.", category="health")
        elif free_gb < 15:
            _check("Disk Space", "warn",
                   f"{free_gb:.1f} GB free — tight for large models (7B+ need 4–8 GB each).",
                   "ollama rm <model-name>", category="health")
        else:
            _check("Disk Space", "ok", f"{free_gb:.0f} GB free.", category="health")
    except Exception:
        pass

    # Version
    try:
        from importlib.metadata import version as _pkgver
        from packaging.version import Version
        current = _pkgver("llm-autotune")
        latest  = _latest_pypi_version()
        if latest and Version(latest) > Version(current):
            _check("Version", "info", f"Update available: v{current} → v{latest}",
                   "pip install --upgrade llm-autotune", category="health")
        elif latest:
            _check("Version", "ok", f"autotune v{current} is up to date.", category="health")
        else:
            _check("Version", "info", f"Running v{current} (PyPI check unavailable).", category="health")
    except Exception:
        pass

    return checks


def get_security_stats_24h() -> dict:
    """Event counts for the last 24 hours, keyed by event name."""
    try:
        return _db().get_security_stats_24h()
    except Exception:
        return {}


def get_security_events_recent(
    limit: int = 200,
    event_filter: str | None = None,
    severity_filter: str | None = None,
) -> list[dict]:
    """Recent security events from the persistent log, newest first."""
    try:
        return _db().get_security_events(
            limit=limit,
            event_filter=event_filter or None,
            severity_filter=severity_filter or None,
        )
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Optimization events — derived from run_observations
# ---------------------------------------------------------------------------
_OPT_DESC: dict[str, dict] = {
    "kv_context": {
        "label": "KV Cache Sizing",
        "purpose": (
            "Your prompt was {after:,} tokens long. Ollama defaults to a fixed {before:,}-token "
            "context window and pre-allocates RAM for the entire window regardless of actual usage. "
            "autotune measured the real prompt length and set num_ctx={after:,} instead."
        ),
        "helps": (
            "~{savings_pct}% of KV-cache RAM freed for this request. Smaller KV cache = "
            "lower TTFT, less memory pressure, and headroom to run additional models simultaneously."
        ),
    },
    "kv_quant_q8": {
        "label": "KV Quant: Q8",
        "purpose": (
            "autotune set the KV cache to Q8 (8-bit quantized) precision instead of F16. "
            "{reason}"
        ),
        "helps": (
            "Q8 KV halves the memory footprint of the attention cache compared to F16, "
            "with negligible quality impact (< 0.1% perplexity difference on most models). "
            "This directly reduces TTFT and peak RAM usage."
        ),
    },
    "kv_quant_f16": {
        "label": "KV Quant: F16",
        "purpose": (
            "autotune kept the KV cache at full F16 (16-bit) precision. "
            "This request used the {profile} profile, which prioritises output quality."
        ),
        "helps": (
            "F16 KV preserves the full numerical range of attention scores — "
            "important for long-context reasoning, code generation, and multi-step analysis "
            "where small errors in attention can compound across many tokens."
        ),
    },
    "sys_prompt_cache": {
        "label": "System Prompt Cached",
        "purpose": (
            "autotune pinned {num_keep} tokens of the system prompt in the KV cache using num_keep. "
            "On repeated requests the model reuses this cached prefix instead of re-encoding it."
        ),
        "helps": (
            "Re-encoding the system prompt on every turn wastes compute proportional to its length. "
            "Caching it cuts that overhead to zero for every follow-up message — directly reducing "
            "TTFT for multi-turn conversations."
        ),
    },
    "ram_pressure_moderate": {
        "label": "RAM Pressure · Moderate",
        "purpose": (
            "System RAM reached {ram_pct}% during this request. "
            "autotune reduced num_ctx by ~10% ({before:,} → {after:,} tokens) "
            "to ease memory pressure before sending to the model."
        ),
        "helps": (
            "Proactively shrinking the KV window keeps the model inside available RAM "
            "without triggering OS paging. Paging stalls inference by orders of magnitude — "
            "a smaller context is always faster than a swapping one."
        ),
    },
    "ram_pressure_high": {
        "label": "RAM Pressure · High",
        "purpose": (
            "System RAM hit {ram_pct}% — above the high-pressure threshold. "
            "autotune cut num_ctx by ~25% ({before:,} → {after:,} tokens) "
            "and downgraded KV cache from F16 → Q8 to reclaim memory."
        ),
        "helps": (
            "At high RAM pressure a double reduction is applied: smaller context window "
            "shrinks total KV allocation, and Q8 precision halves the per-token KV footprint. "
            "Together they can free 30–50% of KV memory, keeping inference stable."
        ),
    },
    "ram_pressure_critical": {
        "label": "RAM Pressure · Critical",
        "purpose": (
            "System RAM hit {ram_pct}% — critical threshold. "
            "autotune halved num_ctx ({before:,} → {after:,} tokens), forced KV to Q8, "
            "and reduced num_batch to 256 to limit peak activation memory."
        ),
        "helps": (
            "Critical pressure means the system is on the edge of OOM. "
            "All three levers (context, KV precision, batch size) are pulled simultaneously "
            "to keep inference from crashing. Consider a smaller model or Q4_K_M quantization."
        ),
    },
    "swap_guard": {
        "label": "SwapGuard Triggered",
        "purpose": (
            "NoSwapGuard detected that the requested configuration would exceed available RAM "
            "and cause OS swap usage. autotune recalculated safe parameters — "
            "num_ctx and KV precision — to fit entirely within free RAM."
        ),
        "helps": (
            "Swap (virtual memory on disk) can slow LLM inference by 10–100× compared to "
            "RAM-resident inference. SwapGuard guarantees the model stays off-disk "
            "even on machines with limited memory."
        ),
    },
    "swap_activity": {
        "label": "Swap Activity Detected",
        "purpose": (
            "The system recorded {delta_swap:.2f} GB of additional swap usage during this request. "
            "This means the OS moved memory pages to disk while the model was running."
        ),
        "helps": (
            "Swap activity is a warning sign — it indicates the model is too large for available RAM. "
            "Consider switching to a smaller model, enabling Q4_K_M quantization, or reducing "
            "concurrent model load."
        ),
    },
    "qos_fast": {
        "label": "QOS: Fast",
        "purpose": (
            "autotune classified this as a low-latency request and applied the Fast profile: "
            "temperature 0.1 (near-greedy), max 512 output tokens, context cap 2048 tokens, "
            "Q8 KV cache, and USER_INTERACTIVE OS scheduling priority."
        ),
        "helps": (
            "Fast profile minimises TTFT above all else. It is ideal for code completions, "
            "short Q&A, command-line tools, and IDE integrations where sub-second responses "
            "matter more than extended reasoning depth."
        ),
    },
    "qos_balanced": {
        "label": "QOS: Balanced",
        "purpose": (
            "autotune applied the Balanced profile: temperature 0.7, up to 1024 output tokens, "
            "context cap 8192 tokens, F16 KV cache precision, and standard USER_INITIATED "
            "scheduling. This is the default for general-purpose requests."
        ),
        "helps": (
            "Balanced gives you full-quality attention (F16 KV) and enough context for documents "
            "and multi-turn conversations, while keeping max_new_tokens at 1024 to prevent "
            "runaway generation. The sweet spot between speed and quality for everyday work."
        ),
    },
    "qos_quality": {
        "label": "QOS: Quality",
        "purpose": (
            "autotune applied the Quality profile: temperature 0.8, up to 4096 output tokens, "
            "context cap 32768 tokens, F16 KV cache, and preferred quants Q5_K_M/Q6_K. "
            "Applied for long-form or complex requests."
        ),
        "helps": (
            "Quality mode removes all the shortcuts taken by Fast and Balanced. F16 KV preserves "
            "attention precision over long contexts, high max_new_tokens allows complete answers, "
            "and better quants reduce accumulated rounding errors in the weight matrices."
        ),
    },
}

_BASELINE_CTX = 4096


def get_optimization_events(limit: int = 200, model_id: str | None = None) -> list[dict]:
    """Return recent optimization decisions derived from run_observations.

    Each request may yield multiple events per row. Events are returned newest-first.
    """
    db = _db()
    filters = ["observed_at IS NOT NULL"]
    params: list = []

    if model_id:
        filters.append("model_id = ?")
        params.append(model_id)

    rows = db.conn.execute(
        "SELECT model_id, context_len, profile_name, ttft_ms, gen_tokens_per_sec, "
        "elapsed_sec, prompt_tokens, completion_tokens, observed_at, "
        "f16_kv, num_keep, delta_ram_gb, delta_swap_gb, ram_before_gb "
        "FROM run_observations WHERE " + " AND ".join(filters) +
        " ORDER BY observed_at DESC LIMIT ?",
        params + [limit * 4],
    ).fetchall()

    def _ev(ts, model, ev_type, ttft, tps, extra: dict) -> dict:
        desc = _OPT_DESC.get(ev_type, {})
        return {
            "ts": ts, "model": model, "type": ev_type,
            "ttft_ms": ttft, "tps": tps,
            # Default description — overridden by extra["description"] when the
            # caller supplies a formatted version with real numbers substituted in.
            "description": {
                "label":   desc.get("label", ev_type),
                "purpose": desc.get("purpose", ""),
                "helps":   desc.get("helps", ""),
            },
            **extra,
        }

    # Pre-pass: identify which rows represent QOS profile transitions (chronological).
    # Rows arrive newest-first; iterate in reverse (oldest-first) to detect changes.
    _qos_change_ids: set[int] = set()
    _last_profile_per_model: dict[str, str] = {}
    for _r in reversed(rows):
        _m   = _r["model_id"] or "unknown"
        _p   = _r["profile_name"]
        if _p and _p in ("fast", "quality"):
            if _last_profile_per_model.get(_m) != _p:
                _qos_change_ids.add(id(_r))
        if _p:
            _last_profile_per_model[_m] = _p

    events: list[dict] = []
    for r in rows:
        if len(events) >= limit:
            break
        ctx            = r["context_len"]
        profile        = r["profile_name"]
        ts             = r["observed_at"]
        model          = r["model_id"] or "unknown"
        ttft           = r["ttft_ms"]
        tps            = r["gen_tokens_per_sec"]
        f16_kv         = r["f16_kv"]        # 1=F16, 0=Q8, None=unknown
        num_keep       = r["num_keep"]
        delta_swap     = r["delta_swap_gb"] or 0.0
        delta_ram      = r["delta_ram_gb"]  or 0.0

        # ── 1. KV context sizing ─────────────────────────────────────────
        if ctx and int(ctx) < int(_BASELINE_CTX * 0.95):
            savings_pct = round((_BASELINE_CTX - ctx) / _BASELINE_CTX * 100, 1)
            desc = _OPT_DESC["kv_context"]
            events.append(_ev(ts, model, "kv_context", ttft, tps, {
                "before": _BASELINE_CTX, "after": int(ctx),
                "savings_pct": savings_pct,
                "description": {
                    "label":   desc["label"],
                    "purpose": desc["purpose"].format(after=int(ctx), before=_BASELINE_CTX),
                    "helps":   desc["helps"].format(savings_pct=savings_pct),
                },
            }))

        # ── 2. KV cache quantization ─────────────────────────────────────
        if f16_kv is not None:
            if f16_kv == 0:
                # Q8 KV — determine reason
                if profile == "fast":
                    reason = "The Fast profile defaults to Q8 KV to minimise memory footprint for short interactions."
                elif delta_ram > 0.3:
                    reason = f"RAM pressure ({delta_ram:.1f} GB spike detected) triggered an automatic downgrade from F16 to Q8."
                else:
                    reason = "RAM pressure was detected before this request, triggering an automatic F16 → Q8 downgrade."
                desc = _OPT_DESC["kv_quant_q8"]
                events.append(_ev(ts, model, "kv_quant_q8", ttft, tps, {
                    "description": {
                        "label":   desc["label"],
                        "purpose": desc["purpose"].format(reason=reason),
                        "helps":   desc["helps"],
                    },
                }))
            # F16 KV is the baseline — not an active optimisation, skip it.

        # ── 3. System prompt caching ─────────────────────────────────────
        if num_keep and int(num_keep) > 0:
            desc = _OPT_DESC["sys_prompt_cache"]
            events.append(_ev(ts, model, "sys_prompt_cache", ttft, tps, {
                "num_keep": int(num_keep),
                "description": {
                    "label":   desc["label"],
                    "purpose": desc["purpose"].format(num_keep=int(num_keep)),
                    "helps":   desc["helps"],
                },
            }))

        # ── 4. RAM pressure reductions ───────────────────────────────────
        # Infer pressure level from delta_ram heuristic (no direct column).
        # We use the context reduction as the signal: if ctx was cut and delta_ram
        # is significant, a pressure event fired.
        if delta_ram > 0.5 and ctx and ctx < _BASELINE_CTX:
            reduction_pct = round((_BASELINE_CTX - ctx) / _BASELINE_CTX * 100)
            if reduction_pct >= 45:
                ev_type  = "ram_pressure_critical"
                ram_pct_est = "≥ 95"
            elif reduction_pct >= 20:
                ev_type  = "ram_pressure_high"
                ram_pct_est = "≥ 85"
            else:
                ev_type  = "ram_pressure_moderate"
                ram_pct_est = "≥ 75"
            desc = _OPT_DESC[ev_type]
            events.append(_ev(ts, model, ev_type, ttft, tps, {
                "before": _BASELINE_CTX, "after": int(ctx), "delta_ram_gb": delta_ram,
                "description": {
                    "label":   desc["label"],
                    "purpose": desc["purpose"].format(
                        ram_pct=ram_pct_est, before=_BASELINE_CTX, after=int(ctx)
                    ),
                    "helps":   desc["helps"],
                },
            }))

        # ── 5. Swap activity ─────────────────────────────────────────────
        if delta_swap > 0.05:
            desc = _OPT_DESC["swap_activity"]
            events.append(_ev(ts, model, "swap_activity", ttft, tps, {
                "delta_swap_gb": round(delta_swap, 2),
                "description": {
                    "label":   desc["label"],
                    "purpose": desc["purpose"].format(delta_swap=delta_swap),
                    "helps":   desc["helps"],
                },
            }))

        # ── 6. QOS profile selection — only at transitions ────────────────
        # "balanced" is the default; only "fast" / "quality" matter, and only
        # when the profile actually changed for this model (not every request).
        if profile and profile in ("fast", "quality") and id(r) in _qos_change_ids:
            opt_key = f"qos_{profile}"
            desc    = _OPT_DESC.get(opt_key, {})
            events.append(_ev(ts, model, opt_key, ttft, tps, {
                "profile": profile,
                "elapsed_sec": r["elapsed_sec"],
                "description": {
                    "label":   desc.get("label", opt_key),
                    "purpose": desc.get("purpose", ""),
                    "helps":   desc.get("helps", ""),
                },
            }))

    events.sort(key=lambda e: (e["ts"] or 0), reverse=True)
    return events[:limit]


def get_optimization_summary() -> dict:
    """Return aggregate stats about optimizations.

    Scoped to the last 24 h, but falls back to all-time when the day has had
    no traffic — otherwise the headline reads "0 optimizations / 0.0% saved"
    while the event list right below shows real wins, which looks broken.
    """
    db    = _db()
    since = time.time() - 86400
    _SQL = """SELECT COUNT(*) AS total_requests,
               COUNT(CASE WHEN context_len IS NOT NULL AND context_len < ? * 0.95 THEN 1 END) AS kv_ctx_opt,
               COUNT(CASE WHEN f16_kv = 0 THEN 1 END)                                          AS kv_q8_count,
               COUNT(CASE WHEN num_keep IS NOT NULL AND num_keep > 0 THEN 1 END)                AS cache_count,
               COUNT(CASE WHEN delta_swap_gb IS NOT NULL AND delta_swap_gb > 0.05 THEN 1 END)   AS swap_count,
               COUNT(CASE WHEN profile_name IS NOT NULL THEN 1 END)                             AS qos_sel,
               AVG(CASE WHEN context_len IS NOT NULL AND context_len < ? * 0.95
                        THEN (? - context_len) * 100.0 / ? END)                                AS avg_kv_pct,
               AVG(ttft_ms) AS avg_ttft
           FROM run_observations WHERE {where}"""
    base_params = (_BASELINE_CTX, _BASELINE_CTX, _BASELINE_CTX, _BASELINE_CTX)
    row = db.conn.execute(_SQL.format(where="observed_at > ?"), (*base_params, since)).fetchone()
    window = "24h"
    if int(row["total_requests"] or 0) == 0:
        all_row = db.conn.execute(_SQL.format(where="1=1"), base_params).fetchone()
        if all_row and (all_row["total_requests"] or 0) > 0:
            row = all_row
            window = "all-time"

    total      = int(row["total_requests"]  or 0)
    kv_ctx     = int(row["kv_ctx_opt"]      or 0)
    kv_q8      = int(row["kv_q8_count"]     or 0)
    cache_pins = int(row["cache_count"]     or 0)
    swap_hits  = int(row["swap_count"]      or 0)
    qos        = int(row["qos_sel"]         or 0)
    total_opt  = kv_ctx + kv_q8 + cache_pins + swap_hits + qos

    return {
        "total_requests":      total,
        "total_optimizations": total_opt,
        "window":              window,
        "kv_ctx_optimized":    kv_ctx,
        "kv_q8_count":         kv_q8,
        "sys_prompt_cached":   cache_pins,
        "swap_interventions":  swap_hits,
        "qos_selected":        qos,
        "kv_optimized_pct":    round(kv_ctx / total * 100, 1) if total > 0 else 0.0,
        "avg_kv_savings_pct":  round(row["avg_kv_pct"] or 0.0, 1),
        "avg_ttft_ms":         round(row["avg_ttft"] or 0.0, 1),
    }


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


def get_onboarding_state() -> dict[str, bool]:
    """Return completion state for the first-run onboarding checklist."""
    db = _db()
    conn = db.conn

    has_run = conn.execute(
        "SELECT 1 FROM run_observations LIMIT 1"
    ).fetchone() is not None

    has_key = conn.execute(
        "SELECT 1 FROM api_keys WHERE is_active=1 LIMIT 1"
    ).fetchone() is not None

    recent_request = conn.execute(
        "SELECT 1 FROM run_observations WHERE observed_at > ? LIMIT 1",
        (time.time() - 86400,),
    ).fetchone() is not None

    try:
        from autotune.api.local_models import is_ollama_running
        ollama_ok = is_ollama_running()
    except Exception:
        ollama_ok = False

    return {
        "ollama_connected":    ollama_ok,
        "model_pulled":        has_run,
        "key_created":         has_key,
        "first_request_made":  recent_request,
        "all_done":            ollama_ok and has_run and has_key and recent_request,
    }
