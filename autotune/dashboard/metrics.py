"""Dashboard metrics aggregation — SQLite queries + live psutil stats."""
from __future__ import annotations

import datetime
import time
from typing import Any

import psutil


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
        "total_tokens_today": int(
            (row["total_prompt_tokens"] or 0) + (row["total_comp_tokens"] or 0)
        ) if row else 0,
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
        })
    return result


def get_models_stats() -> list[dict]:
    """Per-model aggregate stats, all time, sorted by request count desc."""
    db = _db()
    rows = db.conn.execute(
        """SELECT
               model_id,
               COUNT(*)                                      AS requests,
               AVG(ttft_ms)                                  AS avg_ttft,
               MIN(ttft_ms)                                  AS min_ttft,
               MAX(ttft_ms)                                  AS max_ttft,
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
        last = r["last_seen"]
        result.append({
            "model_id":       r["model_id"],
            "requests":       r["requests"],
            "avg_ttft_ms":    round(r["avg_ttft"] or 0, 1),
            "min_ttft_ms":    round(r["min_ttft"] or 0, 1),
            "max_ttft_ms":    round(r["max_ttft"] or 0, 1),
            "avg_tps":        round(r["avg_tps"] or 0, 1),
            "max_tps":        round(r["max_tps"] or 0, 1),
            "avg_context_len": round(r["avg_ctx"] or 0, 0),
            "avg_elapsed_sec": round(r["avg_elapsed"] or 0, 2),
            "total_tokens":   int((r["total_prompt"] or 0) + (r["total_comp"] or 0)),
            "last_seen": (
                datetime.datetime.fromtimestamp(last).strftime("%Y-%m-%d %H:%M")
                if last else None
            ),
        })
    return result


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
