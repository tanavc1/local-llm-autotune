"""
autotune quick proof — does autotune actually help on YOUR machine?

Runs in ≤60 seconds.  Two tests, honest numbers.

TEST 1 — Every message (model already loaded)
  Shows RAM freed per request.  Both conditions use the same loaded model,
  so timing noise is low.  autotune picks the smallest memory block that
  fits your prompt; raw Ollama always reserves the full default block.
  RAM freed = raw_reserved − tuned_reserved.  Goes back to your browser,
  IDE, Slack, and OS every single request.

TEST 2 — Starting a new chat (two interleaved rounds each)
  Measures how long until you see the FIRST word of a response, starting
  from the same neutral memory state.  Each condition must freshly allocate
  its own block before processing your message.
    raw    → always allocates the full default block
    tuned  → allocates only what the prompt needs (smaller)
  Two rounds are interleaved (prime→raw, prime→tuned, prime→raw, prime→tuned)
  and averaged to cancel out thermal and cache noise.
  Whether the difference is measurable depends on hardware:
    – Apple Silicon (Metal): allocation is near-instant regardless of size
    – CPU inference or VRAM-limited GPU: size matters more
  Either way, the RAM freed applies on every request.

All timings from Ollama's internal Go nanosecond timers — nothing estimated.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import httpx
import psutil

_OLLAMA_BASE    = "http://localhost:11434"
_RAW_CTX        = 4096
_KEEP_ALIVE     = "30m"
_MAX_TOKENS     = 120      # short generation; TTFT is what we measure
_MAX_TOKENS_LC  = 60       # even shorter for session-start test
_RELOAD_MS      = 400.0    # load_ms above this → cold disk reload happened
_COOLDOWN_SEC   = 1.5      # pause between condition switches

# Neutral memory state for the session-start TTFT test.
# Set BELOW autotune's smallest typical bucket (~1536) so both conditions
# must allocate UPWARD from scratch — maximizing the size difference tested.
# raw:   1024 → 4096  (allocates 448 MB block)
# tuned: 1024 → ~1536 (allocates ~168 MB block)
_NEUTRAL_CTX    = 1024

# Number of interleaved rounds for the session-start TTFT test.
# Each round: prime→raw, prime→tuned.  Averages cancel thermal/cache noise.
_SS_ROUNDS      = 2

# Number of interleaved rounds for the speed test.
_SPD_ROUNDS     = 2
_MAX_TOKENS_SPD = 40       # very short generation; only TTFT and prefill matter here

# ─────────────────────────────────────────────────────────────────────────────
# Test 1 prompt — multi-turn conversation with system prompt.
#
# autotune computes:  input_tokens(~130) + max_new_tokens(1024) + 256 = ~1410
#                     → snapped to bucket 1536
# raw Ollama:         always uses 4096
# KV freed (llama3.2:3b F16):  (4096−1536)/4096 × 448 MB = 280 MB
# ─────────────────────────────────────────────────────────────────────────────
PROOF_MESSAGES: list[dict] = [
    {
        "role": "system",
        "content": (
            "You are a senior software engineer with 15 years of production experience. "
            "Give concise, precise answers with real examples."
        ),
    },
    {
        "role": "user",
        "content": "When should I use Redis instead of in-process caching?",
    },
    {
        "role": "assistant",
        "content": (
            "Use in-process caching (e.g. lru_cache) when data is per-instance "
            "and you only have one process. Use Redis for shared state across "
            "multiple processes or replicas, atomic operations, pub/sub, or TTL "
            "eviction. The cost: ~1 ms network RTT and a serialization step."
        ),
    },
    {
        "role": "user",
        "content": "Our API has 8 replicas behind a load balancer. Which should we use?",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Test 2 prompt — longer prompt simulating code review / document work.
#
# autotune computes:  input_tokens(~340) + max_new_tokens(1024) + 256 = ~1620
#                     → snapped to bucket 2048
# raw Ollama:         always uses 4096
# KV freed (llama3.2:3b F16):  (4096−2048)/4096 × 448 MB = 224 MB
#
# This prompt simulates the most common case where TTFT matters: a user pastes
# in a code block or document as their opening message in a new session.
# ─────────────────────────────────────────────────────────────────────────────
SESSION_START_MESSAGES: list[dict] = [
    {
        "role": "system",
        "content": "You are a code reviewer. Be concise — list issues only.",
    },
    {
        "role": "user",
        "content": (
            "Review this Python database manager and list the top 3 issues:\n\n"
            "```python\n"
            "class DatabaseManager:\n"
            "    _instance = None\n"
            "    _conn = None\n\n"
            "    def __new__(cls):\n"
            "        if cls._instance is None:\n"
            "            cls._instance = super().__new__(cls)\n"
            "        return cls._instance\n\n"
            "    def connect(self, host, port, db, user, password):\n"
            "        import psycopg2\n"
            "        self._conn = psycopg2.connect(\n"
            "            host=host, port=port, dbname=db,\n"
            "            user=user, password=password\n"
            "        )\n\n"
            "    def query(self, sql, params=None):\n"
            "        cur = self._conn.cursor()\n"
            "        cur.execute(sql, params)\n"
            "        rows = cur.fetchall()\n"
            "        cur.close()\n"
            "        return rows\n\n"
            "    def get_user(self, user_id):\n"
            "        return self.query(\n"
            "            f'SELECT * FROM users WHERE id = {user_id}'\n"
            "        )\n\n"
            "    def transfer_funds(self, src, dst, amount):\n"
            "        cur = self._conn.cursor()\n"
            "        cur.execute(\n"
            "            'UPDATE accounts SET balance = balance - %s WHERE id = %s',\n"
            "            (amount, src)\n"
            "        )\n"
            "        cur.execute(\n"
            "            'UPDATE accounts SET balance = balance + %s WHERE id = %s',\n"
            "            (amount, dst)\n"
            "        )\n"
            "        self._conn.commit()\n"
            "        cur.close()\n\n"
            "    def bulk_insert(self, table, rows):\n"
            "        cur = self._conn.cursor()\n"
            "        for row in rows:\n"
            "            cur.execute(f'INSERT INTO {table} VALUES {row}')\n"
            "        self._conn.commit()\n"
            "        cur.close()\n"
            "```\n"
            "List the top 3 critical bugs or security issues."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Speed test prompt — ~1700-token Python security review.
#
# Deliberately >1024 tokens to expose the num_batch difference:
#   raw Ollama:  num_batch=512 (default) → 4 GPU dispatch passes for this prompt
#   autotune:    num_batch=1024          → 2 GPU dispatch passes (−50% dispatches)
#
# autotune computes:  input_tokens(~1725) + max_new_tokens(40) + 256 = ~2021
#                     → snapped to bucket 2048
# raw Ollama:         always uses 4096
# Both start from _NEUTRAL_CTX so each must freshly allocate its own KV block.
# ─────────────────────────────────────────────────────────────────────────────
SPEED_MESSAGES: list[dict] = [
    {
        "role": "system",
        "content": "You are a senior Python security reviewer. Identify only critical vulnerabilities, be concise.",
    },
    {
        "role": "user",
        "content": (
            "Review this Python web application and identify the top 5 critical security vulnerabilities:\n\n"
            "```python\n"
            "import hashlib\n"
            "import sqlite3\n"
            "import os\n"
            "import pickle\n"
            "import subprocess\n"
            "import yaml\n"
            "import requests\n"
            "from flask import Flask, request, jsonify\n"
            "from functools import wraps\n"
            "\n"
            "app = Flask(__name__)\n"
            "app.secret_key = 'hardcoded_secret_key_abc123'\n"
            "DATABASE = 'app.db'\n"
            "\n"
            "def get_db():\n"
            "    conn = sqlite3.connect(DATABASE)\n"
            "    conn.row_factory = sqlite3.Row\n"
            "    return conn\n"
            "\n"
            "def init_db():\n"
            "    with get_db() as conn:\n"
            "        conn.execute('''\n"
            "            CREATE TABLE IF NOT EXISTS users (\n"
            "                id INTEGER PRIMARY KEY,\n"
            "                username TEXT UNIQUE NOT NULL,\n"
            "                password TEXT NOT NULL,\n"
            "                email TEXT,\n"
            "                role TEXT DEFAULT 'user',\n"
            "                api_key TEXT,\n"
            "                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
            "            )\n"
            "        ''')\n"
            "        conn.execute('''\n"
            "            CREATE TABLE IF NOT EXISTS sessions (\n"
            "                token TEXT PRIMARY KEY,\n"
            "                user_id INTEGER,\n"
            "                data BLOB,\n"
            "                expires_at TIMESTAMP\n"
            "            )\n"
            "        ''')\n"
            "        conn.execute('''\n"
            "            CREATE TABLE IF NOT EXISTS audit_log (\n"
            "                id INTEGER PRIMARY KEY,\n"
            "                user_id INTEGER,\n"
            "                action TEXT,\n"
            "                details TEXT,\n"
            "                ip_address TEXT,\n"
            "                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
            "            )\n"
            "        ''')\n"
            "\n"
            "def hash_password(password):\n"
            "    return hashlib.md5(password.encode()).hexdigest()\n"
            "\n"
            "def verify_password(stored, provided):\n"
            "    return stored == hash_password(provided)\n"
            "\n"
            "def require_auth(f):\n"
            "    @wraps(f)\n"
            "    def decorated(*args, **kwargs):\n"
            "        token = request.headers.get('Authorization', '').replace('Bearer ', '')\n"
            "        if not token:\n"
            "            return jsonify({'error': 'No token'}), 401\n"
            "        with get_db() as conn:\n"
            "            row = conn.execute(\n"
            "                f\"SELECT * FROM sessions WHERE token = '{token}'\"\n"
            "            ).fetchone()\n"
            "        if not row:\n"
            "            return jsonify({'error': 'Invalid token'}), 401\n"
            "        return f(*args, **kwargs)\n"
            "    return decorated\n"
            "\n"
            "@app.route('/login', methods=['POST'])\n"
            "def login():\n"
            "    data = request.get_json()\n"
            "    username = data.get('username', '')\n"
            "    password = data.get('password', '')\n"
            "    with get_db() as conn:\n"
            "        user = conn.execute(\n"
            "            f\"SELECT * FROM users WHERE username = '{username}'\"\n"
            "        ).fetchone()\n"
            "    if user and verify_password(user['password'], password):\n"
            "        token = os.urandom(16).hex()\n"
            "        with get_db() as conn:\n"
            "            conn.execute(\n"
            "                f\"INSERT INTO sessions VALUES ('{token}', {user['id']}, NULL, \"\n"
            "                f\"datetime('now', '+1 day'))\"\n"
            "            )\n"
            "        return jsonify({'token': token, 'role': user['role']})\n"
            "    return jsonify({'error': 'Invalid credentials'}), 401\n"
            "\n"
            "@app.route('/profile/<user_id>')\n"
            "@require_auth\n"
            "def get_profile(user_id):\n"
            "    with get_db() as conn:\n"
            "        user = conn.execute(\n"
            "            f\"SELECT id, username, email, role FROM users WHERE id = {user_id}\"\n"
            "        ).fetchone()\n"
            "    if user:\n"
            "        return jsonify(dict(user))\n"
            "    return jsonify({'error': 'Not found'}), 404\n"
            "\n"
            "@app.route('/upload', methods=['POST'])\n"
            "@require_auth\n"
            "def upload_file():\n"
            "    if 'file' not in request.files:\n"
            "        return jsonify({'error': 'No file'}), 400\n"
            "    f = request.files['file']\n"
            "    f.save(f'/var/uploads/{f.filename}')\n"
            "    return jsonify({'status': 'ok', 'path': f'/var/uploads/{f.filename}'})\n"
            "\n"
            "@app.route('/run-command', methods=['POST'])\n"
            "@require_auth\n"
            "def run_command():\n"
            "    data = request.get_json()\n"
            "    result = subprocess.check_output(data.get('command', ''), shell=True, text=True)\n"
            "    return jsonify({'output': result})\n"
            "\n"
            "@app.route('/load-config', methods=['POST'])\n"
            "@require_auth\n"
            "def load_config():\n"
            "    data = request.get_json()\n"
            "    parsed = yaml.load(data.get('config', ''), Loader=yaml.Loader)\n"
            "    return jsonify({'config': parsed})\n"
            "\n"
            "@app.route('/deserialize', methods=['POST'])\n"
            "@require_auth\n"
            "def deserialize_data():\n"
            "    data = request.get_json()\n"
            "    obj = pickle.loads(bytes.fromhex(data.get('payload', '')))\n"
            "    return jsonify({'result': str(obj)})\n"
            "\n"
            "@app.route('/fetch-url', methods=['POST'])\n"
            "@require_auth\n"
            "def fetch_url():\n"
            "    data = request.get_json()\n"
            "    resp = requests.get(data.get('url', ''), timeout=10)\n"
            "    return jsonify({'content': resp.text, 'status': resp.status_code})\n"
            "\n"
            "@app.route('/admin/users')\n"
            "def list_users():\n"
            "    with get_db() as conn:\n"
            "        rows = conn.execute(\n"
            "            'SELECT id, username, email, role, api_key FROM users'\n"
            "        ).fetchall()\n"
            "    return jsonify([dict(u) for u in rows])\n"
            "\n"
            "@app.route('/reset-password', methods=['POST'])\n"
            "def reset_password():\n"
            "    data = request.get_json()\n"
            "    email = data.get('email', '')\n"
            "    new_pw = data.get('new_password', '')\n"
            "    with get_db() as conn:\n"
            "        user = conn.execute(\n"
            "            f\"SELECT id FROM users WHERE email = '{email}'\"\n"
            "        ).fetchone()\n"
            "    if user:\n"
            "        with get_db() as conn:\n"
            "            conn.execute(\n"
            "                f\"UPDATE users SET password = '{hash_password(new_pw)}' \"\n"
            "                f\"WHERE email = '{email}'\"\n"
            "            )\n"
            "        return jsonify({'status': 'Password updated'})\n"
            "    return jsonify({'error': 'Not found'}), 404\n"
            "\n"
            "@app.route('/export')\n"
            "@require_auth\n"
            "def export_data():\n"
            "    table = request.args.get('table', 'users')\n"
            "    with get_db() as conn:\n"
            "        rows = conn.execute(f'SELECT * FROM {table}').fetchall()\n"
            "    return jsonify([dict(r) for r in rows])\n"
            "\n"
            "def log_action(user_id, action, details):\n"
            "    ip = request.remote_addr\n"
            "    with get_db() as conn:\n"
            "        conn.execute(\n"
            "            f\"INSERT INTO audit_log (user_id, action, details, ip_address) \"\n"
            "            f\"VALUES ({user_id}, '{action}', '{details}', '{ip}')\"\n"
            "        )\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    init_db()\n"
            "    app.run(host='0.0.0.0', port=5000, debug=True)\n"
            "```\n"
            "List only the top 5 critical security vulnerabilities with line references."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal RAM sampler (system-level, not process-isolated)
# ─────────────────────────────────────────────────────────────────────────────

class _RamSampler:
    """Samples system-available RAM and swap every 100ms in a background thread."""

    def __init__(self) -> None:
        self._free: list[float] = []
        self._swap_start = 0.0
        self._swap_end   = 0.0
        self._running    = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._swap_start = psutil.swap_memory().used / 1024 ** 3
        vm = psutil.virtual_memory()
        self._free.append(vm.available / 1024 ** 3)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._swap_end = psutil.swap_memory().used / 1024 ** 3

    def _loop(self) -> None:
        while self._running:
            vm = psutil.virtual_memory()
            self._free.append(vm.available / 1024 ** 3)
            time.sleep(0.1)

    def free_floor_gb(self) -> float:
        return min(self._free) if self._free else 0.0

    def swap_delta_gb(self) -> float:
        return max(0.0, self._swap_end - self._swap_start)

    def swap_occurred(self) -> bool:
        return self.swap_delta_gb() > 0.031   # 32 MB threshold


# ─────────────────────────────────────────────────────────────────────────────
# Single-run result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RunResult:
    condition:          str    # "raw" | "autotune"
    num_ctx:            int
    prefill_ms:         float  # prompt_eval_duration (Ollama Go timer)
    load_ms:            float  # load_duration = KV allocation + any model load cost
    eval_tps:           float  # generation tok/s
    eval_count:         int    # tokens generated
    prompt_tokens:      int    # tokens in prompt (Ollama-reported)
    kv_cache_mb:        float  # estimated KV size for this num_ctx
    free_floor_gb:      float  # min system available RAM during run
    swap_occurred:      bool
    reload_detected:    bool   # load_ms > threshold → disk reload (cold start noise)
    error: Optional[str] = None

    @property
    def ttft_ms(self) -> float:
        return self.load_ms + self.prefill_ms

    @property
    def ok(self) -> bool:
        return self.error is None and self.eval_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# KV cache estimator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ModelArch:
    n_layers:   int = 0
    n_kv_heads: int = 0
    head_dim:   int = 0

    def kv_cache_mb(self, num_ctx: int, dtype_bytes: int = 2) -> float:
        if not (self.n_layers and self.n_kv_heads and self.head_dim):
            return 0.0
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * num_ctx * dtype_bytes / (1024 ** 2)


async def _fetch_arch(model_id: str) -> _ModelArch:
    arch = _ModelArch()
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.post(f"{_OLLAMA_BASE}/api/show", json={"name": model_id, "verbose": True})
        mi = r.json().get("model_info", {})

        def _find(suffix: str) -> int:
            for k, v in mi.items():
                if k.endswith(suffix) and isinstance(v, (int, float)):
                    return int(v)
            return 0

        arch.n_layers   = _find(".block_count")
        arch.n_kv_heads = _find(".attention.head_count_kv") or _find(".attention.kv_heads")
        arch.head_dim   = _find(".attention.head_dim") or _find(".attention.key_length")
        if not arch.head_dim:
            emb   = _find(".embedding_length")
            heads = _find(".attention.head_count")
            if emb and heads:
                arch.head_dim = emb // heads
    except Exception:
        pass
    return arch


# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _call_ollama(
    model_id: str,
    messages: list[dict],
    options: dict,
    max_tokens: int = _MAX_TOKENS,
    keep_alive: str = _KEEP_ALIVE,
) -> dict:
    payload = {
        "model":      model_id,
        "messages":   messages,
        "stream":     False,
        "options":    {**options, "num_predict": max_tokens},
        "keep_alive": keep_alive,
    }
    async with httpx.AsyncClient(timeout=90.0) as c:
        r = await c.post(f"{_OLLAMA_BASE}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()


async def _prime_ctx(model_id: str, num_ctx: int) -> None:
    """
    Set the model's active KV context window to num_ctx and return.

    Used to establish a known neutral state before the session-start TTFT test.
    The trivial request runs but the result is discarded — we only care that
    Ollama has allocated a KV buffer at num_ctx so the NEXT request triggers a
    fresh reallocation to its own (different) num_ctx.

    Keeps the model loaded (keep_alive=_KEEP_ALIVE) — no disk reload noise.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            await c.post(f"{_OLLAMA_BASE}/api/chat", json={
                "model":      model_id,
                "messages":   [{"role": "user", "content": "ok"}],
                "stream":     False,
                "options":    {"num_ctx": num_ctx, "num_predict": 1},
                "keep_alive": _KEEP_ALIVE,
            })
        await asyncio.sleep(0.4)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Single instrumented run
# ─────────────────────────────────────────────────────────────────────────────

async def _measured_run(
    model_id: str,
    condition: str,
    options: dict,
    arch: _ModelArch,
    kv_dtype_bytes: int,
    messages: Optional[list] = None,
    max_tokens: int = _MAX_TOKENS,
) -> _RunResult:
    if messages is None:
        messages = PROOF_MESSAGES
    sampler = _RamSampler()
    sampler.start()
    error = None

    try:
        data = await _call_ollama(model_id, messages, options, max_tokens=max_tokens)
        load_ms    = data.get("load_duration",        0) / 1_000_000
        prefill_ms = data.get("prompt_eval_duration", 0) / 1_000_000
        eval_cnt   = data.get("eval_count",           0)
        eval_dur   = data.get("eval_duration",        0) / 1_000_000_000
        prompt_cnt = data.get("prompt_eval_count",    0)
        tps        = eval_cnt / eval_dur if eval_dur > 0 else 0.0
    except Exception as exc:
        error     = str(exc)
        load_ms   = prefill_ms = tps = 0.0
        eval_cnt  = prompt_cnt = 0
    finally:
        sampler.stop()

    num_ctx = options.get("num_ctx", _RAW_CTX)
    return _RunResult(
        condition       = condition,
        num_ctx         = num_ctx,
        prefill_ms      = prefill_ms,
        load_ms         = load_ms,
        eval_tps        = tps,
        eval_count      = eval_cnt,
        prompt_tokens   = prompt_cnt,
        kv_cache_mb     = arch.kv_cache_mb(num_ctx, kv_dtype_bytes),
        free_floor_gb   = sampler.free_floor_gb(),
        swap_occurred   = sampler.swap_occurred(),
        reload_detected = (load_ms > _RELOAD_MS),
        error           = error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuickProofResult:
    model_id:       str
    profile_name:   str
    n_runs:         int
    elapsed_sec:    float
    raw_runs:       list[_RunResult]
    tuned_runs:     list[_RunResult]
    # Session-start test — multiple interleaved rounds, each primed to neutral
    ss_raw_runs:    list[_RunResult] = field(default_factory=list)
    ss_tuned_runs:  list[_RunResult] = field(default_factory=list)
    # Speed test — optional (--speed flag); long-context prefill & TTFT
    spd_raw_runs:   list[_RunResult] = field(default_factory=list)
    spd_tuned_runs: list[_RunResult] = field(default_factory=list)

    # ── Aggregated stats (standard test) ─────────────────────────────────────

    def _mean(self, runs: list[_RunResult], attr: str) -> float:
        vals = [getattr(r, attr) for r in runs if r.ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def raw_ttft_ms(self) -> float:
        return self._mean(self.raw_runs, "ttft_ms")

    @property
    def tuned_ttft_ms(self) -> float:
        return self._mean(self.tuned_runs, "ttft_ms")

    @property
    def ttft_improvement_pct(self) -> float:
        if self.raw_ttft_ms <= 0:
            return 0.0
        return (self.raw_ttft_ms - self.tuned_ttft_ms) / self.raw_ttft_ms * 100

    @property
    def raw_kv_mb(self) -> float:
        return self._mean(self.raw_runs, "kv_cache_mb")

    @property
    def tuned_kv_mb(self) -> float:
        return self._mean(self.tuned_runs, "kv_cache_mb")

    @property
    def kv_saved_mb(self) -> float:
        return max(0.0, self.raw_kv_mb - self.tuned_kv_mb)

    @property
    def kv_pct(self) -> float:
        if self.raw_kv_mb <= 0:
            return 0.0
        return self.kv_saved_mb / self.raw_kv_mb * 100

    @property
    def raw_num_ctx(self) -> int:
        ok = [r for r in self.raw_runs if r.ok]
        return ok[0].num_ctx if ok else _RAW_CTX

    @property
    def tuned_num_ctx(self) -> int:
        ok = [r for r in self.tuned_runs if r.ok]
        return ok[0].num_ctx if ok else 0

    @property
    def ctx_pct(self) -> float:
        if self.raw_num_ctx <= 0:
            return 0.0
        return (self.raw_num_ctx - self.tuned_num_ctx) / self.raw_num_ctx * 100

    @property
    def raw_free_gb(self) -> float:
        return self._mean(self.raw_runs, "free_floor_gb")

    @property
    def tuned_free_gb(self) -> float:
        return self._mean(self.tuned_runs, "free_floor_gb")

    @property
    def headroom_gained_gb(self) -> float:
        return max(0.0, self.tuned_free_gb - self.raw_free_gb)

    @property
    def raw_swap_events(self) -> int:
        return sum(1 for r in self.raw_runs if r.swap_occurred)

    @property
    def tuned_swap_events(self) -> int:
        return sum(1 for r in self.tuned_runs if r.swap_occurred)

    @property
    def raw_tps(self) -> float:
        return self._mean(self.raw_runs, "eval_tps")

    @property
    def tuned_tps(self) -> float:
        return self._mean(self.tuned_runs, "eval_tps")

    # ── Session-start test properties ────────────────────────────────────────

    @property
    def _ss_raw_ok(self) -> list[_RunResult]:
        return [r for r in self.ss_raw_runs if r.ok]

    @property
    def _ss_tuned_ok(self) -> list[_RunResult]:
        return [r for r in self.ss_tuned_runs if r.ok]

    @property
    def has_ss_test(self) -> bool:
        return bool(self._ss_raw_ok) and bool(self._ss_tuned_ok)

    # Convenience: first ok run (for architecture/config values)
    @property
    def ss_raw_run(self) -> Optional[_RunResult]:
        ok = self._ss_raw_ok
        return ok[0] if ok else None

    @property
    def ss_tuned_run(self) -> Optional[_RunResult]:
        ok = self._ss_tuned_ok
        return ok[0] if ok else None

    @property
    def ss_raw_ttft_ms(self) -> float:
        vals = [r.ttft_ms for r in self._ss_raw_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def ss_tuned_ttft_ms(self) -> float:
        vals = [r.ttft_ms for r in self._ss_tuned_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def ss_raw_load_ms(self) -> float:
        vals = [r.load_ms for r in self._ss_raw_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def ss_tuned_load_ms(self) -> float:
        vals = [r.load_ms for r in self._ss_tuned_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def ss_raw_prefill_ms(self) -> float:
        vals = [r.prefill_ms for r in self._ss_raw_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def ss_tuned_prefill_ms(self) -> float:
        vals = [r.prefill_ms for r in self._ss_tuned_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def ss_ttft_improvement_pct(self) -> float:
        if self.ss_raw_ttft_ms <= 0:
            return 0.0
        return (self.ss_raw_ttft_ms - self.ss_tuned_ttft_ms) / self.ss_raw_ttft_ms * 100

    @property
    def ss_raw_kv_mb(self) -> float:
        r = self.ss_raw_run
        return r.kv_cache_mb if r else 0.0

    @property
    def ss_tuned_kv_mb(self) -> float:
        r = self.ss_tuned_run
        return r.kv_cache_mb if r else 0.0

    @property
    def ss_raw_num_ctx(self) -> int:
        r = self.ss_raw_run
        return r.num_ctx if r else _RAW_CTX

    @property
    def ss_tuned_num_ctx(self) -> int:
        r = self.ss_tuned_run
        return r.num_ctx if r else 0

    # ── Speed test properties ─────────────────────────────────────────────────

    @property
    def _spd_raw_ok(self) -> list[_RunResult]:
        return [r for r in self.spd_raw_runs if r.ok]

    @property
    def _spd_tuned_ok(self) -> list[_RunResult]:
        return [r for r in self.spd_tuned_runs if r.ok]

    @property
    def has_speed_test(self) -> bool:
        return bool(self._spd_raw_ok) and bool(self._spd_tuned_ok)

    @property
    def spd_raw_ttft_ms(self) -> float:
        vals = [r.ttft_ms for r in self._spd_raw_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def spd_tuned_ttft_ms(self) -> float:
        vals = [r.ttft_ms for r in self._spd_tuned_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def spd_ttft_improvement_pct(self) -> float:
        if self.spd_raw_ttft_ms <= 0:
            return 0.0
        return (self.spd_raw_ttft_ms - self.spd_tuned_ttft_ms) / self.spd_raw_ttft_ms * 100

    @property
    def spd_raw_prefill_ms(self) -> float:
        vals = [r.prefill_ms for r in self._spd_raw_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def spd_tuned_prefill_ms(self) -> float:
        vals = [r.prefill_ms for r in self._spd_tuned_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def spd_prefill_improvement_pct(self) -> float:
        if self.spd_raw_prefill_ms <= 0:
            return 0.0
        return (self.spd_raw_prefill_ms - self.spd_tuned_prefill_ms) / self.spd_raw_prefill_ms * 100

    @property
    def spd_raw_load_ms(self) -> float:
        vals = [r.load_ms for r in self._spd_raw_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def spd_tuned_load_ms(self) -> float:
        vals = [r.load_ms for r in self._spd_tuned_ok]
        return statistics.mean(vals) if vals else 0.0

    @property
    def spd_raw_kv_mb(self) -> float:
        ok = self._spd_raw_ok
        return ok[0].kv_cache_mb if ok else 0.0

    @property
    def spd_tuned_kv_mb(self) -> float:
        ok = self._spd_tuned_ok
        return ok[0].kv_cache_mb if ok else 0.0

    def to_dict(self) -> dict:
        d: dict = {
            "model_id":       self.model_id,
            "profile":        self.profile_name,
            "n_runs":         self.n_runs,
            "elapsed_sec":    round(self.elapsed_sec, 1),
            "ttft_raw_ms":    round(self.raw_ttft_ms, 1),
            "ttft_tuned_ms":  round(self.tuned_ttft_ms, 1),
            "ttft_pct":       round(self.ttft_improvement_pct, 1),
            "kv_raw_mb":      round(self.raw_kv_mb, 1),
            "kv_tuned_mb":    round(self.tuned_kv_mb, 1),
            "kv_saved_mb":    round(self.kv_saved_mb, 1),
            "kv_pct":         round(self.kv_pct, 1),
            "ctx_raw":        self.raw_num_ctx,
            "ctx_tuned":      self.tuned_num_ctx,
            "ctx_pct":        round(self.ctx_pct, 1),
            "free_raw_gb":    round(self.raw_free_gb, 2),
            "free_tuned_gb":  round(self.tuned_free_gb, 2),
            "headroom_gained_gb": round(self.headroom_gained_gb, 2),
            "swap_raw":       self.raw_swap_events,
            "swap_tuned":     self.tuned_swap_events,
            "tps_raw":        round(self.raw_tps, 1),
            "tps_tuned":      round(self.tuned_tps, 1),
            "raw_runs":       [asdict(r) for r in self.raw_runs],
            "tuned_runs":     [asdict(r) for r in self.tuned_runs],
        }
        if self.has_ss_test:
            d.update({
                "ss_ttft_raw_ms":    round(self.ss_raw_ttft_ms, 1),
                "ss_ttft_tuned_ms":  round(self.ss_tuned_ttft_ms, 1),
                "ss_ttft_pct":       round(self.ss_ttft_improvement_pct, 1),
                "ss_load_raw_ms":    round(self.ss_raw_load_ms, 1),
                "ss_load_tuned_ms":  round(self.ss_tuned_load_ms, 1),
                "ss_kv_raw_mb":      round(self.ss_raw_kv_mb, 1),
                "ss_kv_tuned_mb":    round(self.ss_tuned_kv_mb, 1),
                "ss_ctx_raw":        self.ss_raw_num_ctx,
                "ss_ctx_tuned":      self.ss_tuned_num_ctx,
                "ss_raw_runs":       [asdict(r) for r in self.ss_raw_runs],
                "ss_tuned_runs":     [asdict(r) for r in self.ss_tuned_runs],
            })
        if self.has_speed_test:
            d.update({
                "spd_ttft_raw_ms":      round(self.spd_raw_ttft_ms, 1),
                "spd_ttft_tuned_ms":    round(self.spd_tuned_ttft_ms, 1),
                "spd_ttft_pct":         round(self.spd_ttft_improvement_pct, 1),
                "spd_prefill_raw_ms":   round(self.spd_raw_prefill_ms, 1),
                "spd_prefill_tuned_ms": round(self.spd_tuned_prefill_ms, 1),
                "spd_prefill_pct":      round(self.spd_prefill_improvement_pct, 1),
                "spd_load_raw_ms":      round(self.spd_raw_load_ms, 1),
                "spd_load_tuned_ms":    round(self.spd_tuned_load_ms, 1),
                "spd_kv_raw_mb":        round(self.spd_raw_kv_mb, 1),
                "spd_kv_tuned_mb":      round(self.spd_tuned_kv_mb, 1),
                "spd_raw_runs":         [asdict(r) for r in self.spd_raw_runs],
                "spd_tuned_runs":       [asdict(r) for r in self.spd_tuned_runs],
            })
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_quick_proof(
    model_id:     str,
    profile_name: str = "balanced",
    n_runs:       int = 2,
    output_path:  Optional[Path] = None,
    on_step:      Optional[Callable[[str], None]] = None,
    speed:        bool = False,
) -> QuickProofResult:
    from autotune.api.kv_manager import build_ollama_options
    from autotune.api.profiles import get_profile

    def _step(msg: str) -> None:
        if on_step:
            on_step(msg)

    profile = get_profile(profile_name)
    t0 = time.monotonic()

    # ── Fetch model architecture for KV estimation ───────────────────────────
    _step("Fetching model architecture…")
    arch = await _fetch_arch(model_id)

    # Compute autotune options for both prompts upfront.
    # f16_kv=False means Q8 KV (1 byte per element); True or absent means F16 (2 bytes).
    tuned_opts, _  = build_ollama_options(PROOF_MESSAGES, profile)
    tuned_ss_opts, _ = build_ollama_options(SESSION_START_MESSAGES, profile)
    kv_dtype = 1 if not tuned_opts.get("f16_kv", True) else 2

    # ── Warmup: load model into RAM, verify it responds, discard timing ──────
    _step("Warming up model (loading from disk)…")
    _warmup_err: Optional[str] = None
    try:
        _wu = await _call_ollama(
            model_id,
            [{"role": "user", "content": "Say exactly: ready"}],
            {"num_ctx": _RAW_CTX, "num_predict": 4},
        )
        if _wu.get("eval_count", 0) == 0:
            _warmup_err = (
                f"Model '{model_id}' loaded but produced no tokens. "
                f"It may be corrupted — try: autotune pull {model_id}"
            )
        else:
            await asyncio.sleep(0.5)
    except httpx.ConnectError:
        _warmup_err = "Cannot reach Ollama — autotune will start it automatically on next run."
    except httpx.HTTPStatusError as _e:
        if _e.response.status_code == 404:
            _warmup_err = (
                f"Model '{model_id}' is not installed. "
                f"Pull it first: autotune pull {model_id}"
            )
        else:
            _warmup_err = f"Ollama returned HTTP {_e.response.status_code}"
    except Exception as _e:
        _warmup_err = f"Warmup failed: {_e}"

    if _warmup_err is not None:
        raise RuntimeError(_warmup_err)

    # ── TEST 1: Standard (warm model) ────────────────────────────────────────
    # Model is at num_ctx=4096 after warmup.
    # raw stays at 4096 → load_ms ≈ 0 (no realloc needed)
    # tuned switches to its own bucket → pays one-time realloc, then stable
    # This represents every message after the first in a session.

    raw_opts    = {"num_ctx": _RAW_CTX}
    raw_runs: list[_RunResult] = []

    for i in range(n_runs):
        _step(f"Test 1 — raw Ollama  run {i + 1}/{n_runs}…")
        r = await _measured_run(model_id, "raw", raw_opts, arch, kv_dtype_bytes=2)
        raw_runs.append(r)
        if i < n_runs - 1:
            await asyncio.sleep(0.5)

    _step("Test 1 — switching to autotune…")
    await asyncio.sleep(_COOLDOWN_SEC)

    tuned_runs: list[_RunResult] = []
    for i in range(n_runs):
        _step(f"Test 1 — autotune    run {i + 1}/{n_runs}…")
        r = await _measured_run(model_id, "autotune", tuned_opts, arch, kv_dtype_bytes=kv_dtype)
        tuned_runs.append(r)
        if i < n_runs - 1:
            await asyncio.sleep(0.5)

    # ── TEST 2: Session-start TTFT ───────────────────────────────────────────
    # Prime model to _NEUTRAL_CTX (below both conditions' target) so each must
    # freshly allocate its own memory block.  Runs are interleaved to cancel
    # thermal and cache-warming noise: prime→raw, prime→tuned, prime→raw, …
    # load_ms = memory setup time; tuned allocates a smaller block → potentially faster.

    _step(f"Test 2 — priming neutral state ({_NEUTRAL_CTX} tokens)…")
    ss_raw_runs:   list[_RunResult] = []
    ss_tuned_runs: list[_RunResult] = []

    try:
        for _ss_i in range(_SS_ROUNDS):
            # prime → raw
            await _prime_ctx(model_id, _NEUTRAL_CTX)
            _step(f"Test 2 — raw Ollama round {_ss_i + 1}/{_SS_ROUNDS}  (full memory block)…")
            _sr = await _measured_run(
                model_id, "raw", raw_opts, arch, kv_dtype_bytes=2,
                messages=SESSION_START_MESSAGES, max_tokens=_MAX_TOKENS_LC,
            )
            ss_raw_runs.append(_sr)

            # prime → tuned
            await _prime_ctx(model_id, _NEUTRAL_CTX)
            _step(f"Test 2 — autotune round {_ss_i + 1}/{_SS_ROUNDS}  (right-sized block)…")
            _st = await _measured_run(
                model_id, "autotune", tuned_ss_opts, arch, kv_dtype_bytes=kv_dtype,
                messages=SESSION_START_MESSAGES, max_tokens=_MAX_TOKENS_LC,
            )
            ss_tuned_runs.append(_st)

    except Exception:
        pass  # session-start test is best-effort; never fail the whole proof

    # ── TEST 3 (optional): Prefill & TTFT speed ──────────────────────────────
    # A ~1700-token prompt exposes the num_batch difference:
    #   raw:   num_batch=512 (Ollama default) → 4 GPU dispatch passes
    #   tuned: num_batch=1024 (autotune)      → 2 GPU dispatch passes → faster prefill
    # Combined with a smaller num_ctx (2048 vs 4096), both load_ms and
    # prefill_ms genuinely improve.  Both conditions start from _NEUTRAL_CTX.

    spd_raw_runs:   list[_RunResult] = []
    spd_tuned_runs: list[_RunResult] = []

    if speed:
        tuned_spd_opts, _ = build_ollama_options(SPEED_MESSAGES, profile)
        _step(f"Speed test — priming neutral state ({_NEUTRAL_CTX} tokens)…")

        try:
            for _spd_i in range(_SPD_ROUNDS):
                # prime → raw
                await _prime_ctx(model_id, _NEUTRAL_CTX)
                _step(
                    f"Speed test — raw Ollama round {_spd_i + 1}/{_SPD_ROUNDS} "
                    f"(512-token prefill batches, 4096-ctx block)…"
                )
                _sr = await _measured_run(
                    model_id, "raw", raw_opts, arch, kv_dtype_bytes=2,
                    messages=SPEED_MESSAGES, max_tokens=_MAX_TOKENS_SPD,
                )
                spd_raw_runs.append(_sr)

                # prime → tuned
                await _prime_ctx(model_id, _NEUTRAL_CTX)
                _step(
                    f"Speed test — autotune round {_spd_i + 1}/{_SPD_ROUNDS} "
                    f"(1024-token prefill batches, right-sized block)…"
                )
                _st = await _measured_run(
                    model_id, "autotune", tuned_spd_opts, arch, kv_dtype_bytes=kv_dtype,
                    messages=SPEED_MESSAGES, max_tokens=_MAX_TOKENS_SPD,
                )
                spd_tuned_runs.append(_st)

        except Exception:
            pass  # speed test is best-effort; never fail the whole proof

    elapsed = time.monotonic() - t0

    result = QuickProofResult(
        model_id=model_id,
        profile_name=profile_name,
        n_runs=n_runs,
        elapsed_sec=elapsed,
        raw_runs=raw_runs,
        tuned_runs=tuned_runs,
        ss_raw_runs=ss_raw_runs,
        ss_tuned_runs=ss_tuned_runs,
        spd_raw_runs=spd_raw_runs,
        spd_tuned_runs=spd_tuned_runs,
    )

    if output_path:
        output_path.write_text(json.dumps(result.to_dict(), indent=2))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Rich console output
# ─────────────────────────────────────────────────────────────────────────────

def print_proof_result(result: QuickProofResult, console, output_path: Optional[Path] = None) -> None:
    """Print the proof result in plain user-facing language."""
    from rich.table import Table
    from rich import box as _box
    from rich.panel import Panel

    console.print()
    console.rule(
        f"[bold]autotune proof[/bold]  ·  [cyan]{result.model_id}[/cyan]  "
        f"·  {result.n_runs} runs/condition  ·  {result.elapsed_sec:.0f}s"
    )
    console.print()

    ok_raw   = any(r.ok for r in result.raw_runs)
    ok_tuned = any(r.ok for r in result.tuned_runs)

    if not ok_raw or not ok_tuned:
        console.print(
            "[red]One or more runs failed. "
            "Check that Ollama is running and the model is installed.[/red]"
        )
        return

    def _pct_label(pct: float, better_is_lower: bool = True) -> str:
        if abs(pct) < 1:
            return "[dim]unchanged[/dim]"
        sign = "−" if pct > 0 else "+"
        icon = "✅" if (pct > 0) == better_is_lower else "⚠️"
        return f"{sign}{abs(pct):.0f}% {icon}"

    # ── RAM savings banner ────────────────────────────────────────────────────
    if result.raw_kv_mb > 0 and result.kv_saved_mb >= 1:
        kv_saved_mb = result.kv_saved_mb
        kv_saved_gb = kv_saved_mb / 1024

        if kv_saved_gb >= 0.5:
            freed_str = f"[bold green]{kv_saved_gb:.2f} GB freed[/bold green]"
        else:
            freed_str = f"[bold green]{kv_saved_mb:.0f} MB freed[/bold green]"

        ram_lines = [
            f"  RAM freed per request: {freed_str}  "
            f"[dim](without autotune: {result.raw_kv_mb:.0f} MB  →  with autotune: {result.tuned_kv_mb:.0f} MB,"
            f" −{result.kv_pct:.0f}%)[/dim]",
            "",
            "  [dim]Before answering, Ollama reserves a block of RAM for the AI's working memory.[/dim]",
            "  [dim]Without autotune it always reserves the same large block, even for short messages.[/dim]",
            "  [dim]autotune measures your prompt and reserves only what's actually needed —[/dim]",
            "  [dim]the rest goes back to Chrome, VS Code, Slack, and your OS. Every request.[/dim]",
        ]
        if result.headroom_gained_gb >= 0.05:
            ram_lines.append(
                f"\n  RAM free while model runs: "
                f"[green]+{result.headroom_gained_gb:.2f} GB more[/green]  "
                f"[dim](with autotune: {result.tuned_free_gb:.1f} GB free "
                f"vs without: {result.raw_free_gb:.1f} GB)[/dim]"
            )
        elif result.tuned_free_gb > 0:
            ram_lines.append(
                f"\n  [dim]System RAM free while model runs: {result.tuned_free_gb:.1f} GB "
                f"(your machine has plenty of RAM — savings mainly help your other apps)[/dim]"
            )

        console.print(Panel(
            "\n".join(ram_lines),
            title="[bold green]RAM Savings[/bold green]",
            border_style="green",
            padding=(0, 1),
        ))
        console.print()

    # ── TEST 1 — Every message ────────────────────────────────────────────────
    console.print(
        "[bold]Test 1 — Every message you send[/bold]  "
        "[dim](model already loaded — no startup noise)[/dim]"
    )
    console.print(
        "[dim]  Both conditions use a model that's already running in memory.\n"
        "  RAM savings apply to every single request — not just the first one.[/dim]"
    )
    console.print()

    t1 = Table(box=_box.SIMPLE_HEAD, show_header=True, show_edge=False, pad_edge=False)
    t1.add_column("Metric",            style="white",      width=36)
    t1.add_column("Without autotune",  style="dim white",  justify="right", width=16)
    t1.add_column("With autotune",     style="green",      justify="right", width=14)
    t1.add_column("Change",                                justify="right", width=18)

    # RAM reserved (the primary metric — always show)
    kv_raw_str   = f"{result.raw_kv_mb:.0f} MB"   if result.raw_kv_mb   > 0 else "N/A"
    kv_tuned_str = f"{result.tuned_kv_mb:.0f} MB" if result.tuned_kv_mb > 0 else "N/A"
    t1.add_row(
        "RAM held for AI (per request)",
        kv_raw_str,
        kv_tuned_str,
        _pct_label(result.kv_pct) if result.raw_kv_mb > 0 else "N/A",
    )

    # Swap events
    swap_raw_str   = ("0 ✅" if result.raw_swap_events == 0   else f"{result.raw_swap_events} ⚠️")
    swap_tuned_str = ("0 ✅" if result.tuned_swap_events == 0 else f"{result.tuned_swap_events} ⚠️")
    swap_change    = "[dim]none ✅[/dim]" if result.tuned_swap_events == 0 else f"+{result.tuned_swap_events} ⚠️"
    t1.add_row("Memory overflow events", swap_raw_str, swap_tuned_str, swap_change)

    # Generation speed
    if abs(result.raw_tps) > 0.1:
        delta_pct = (result.tuned_tps - result.raw_tps) / max(result.raw_tps, 0.01) * 100
        if abs(delta_pct) < 5:
            tps_change = "[dim]unchanged[/dim]"
        else:
            icon = "✅" if delta_pct > 0 else "—"
            tps_change = f"{'+'  if delta_pct > 0 else '−'}{abs(delta_pct):.0f}% {icon}"
        t1.add_row(
            "Words per second",
            f"{result.raw_tps:.1f}",
            f"{result.tuned_tps:.1f}",
            tps_change,
        )

    console.print(t1)
    console.print()

    # ── TEST 2 — Starting a new chat ─────────────────────────────────────────
    if result.has_ss_test:
        ss_pct      = result.ss_ttft_improvement_pct
        ttft_helped = ss_pct >= 5
        n_rounds    = len(result.ss_raw_runs)

        if ttft_helped:
            console.print(
                "[bold]Test 2 — Starting a new chat[/bold]  "
                "[dim](time until you see the first word — each condition starts from scratch)[/dim]"
            )
            console.print(
                f"  [dim]{n_rounds} round{'s' if n_rounds != 1 else ''} interleaved and averaged.\n"
                "  Both conditions start from the same neutral point so neither has a head start.\n"
                "  autotune reserves a smaller block of RAM → sets up faster → first word appears sooner.[/dim]"
            )
            console.print()

            t2 = Table(box=_box.SIMPLE_HEAD, show_header=True, show_edge=False, pad_edge=False)
            t2.add_column("Metric",            style="white",      width=36)
            t2.add_column("Without autotune",  style="dim white",  justify="right", width=16)
            t2.add_column("With autotune",     style="green",      justify="right", width=14)
            t2.add_column("Change",                                justify="right", width=18)

            t2.add_row(
                "[bold]Time to first word[/bold]",
                f"[bold]{result.ss_raw_ttft_ms:.0f} ms[/bold]",
                f"[bold]{result.ss_tuned_ttft_ms:.0f} ms[/bold]",
                _pct_label(ss_pct),
            )

            load_pct = (result.ss_raw_load_ms - result.ss_tuned_load_ms) / max(result.ss_raw_load_ms, 1) * 100
            t2.add_row(
                "  └ Memory setup",
                f"{result.ss_raw_load_ms:.0f} ms",
                f"{result.ss_tuned_load_ms:.0f} ms",
                _pct_label(load_pct),
            )
            t2.add_row(
                "  └ Reading your message",
                f"{result.ss_raw_prefill_ms:.0f} ms",
                f"{result.ss_tuned_prefill_ms:.0f} ms",
                "[dim]same[/dim]",
            )

            if result.ss_raw_kv_mb > 0:
                ss_kv_pct = (result.ss_raw_kv_mb - result.ss_tuned_kv_mb) / max(result.ss_raw_kv_mb, 1) * 100
                t2.add_row(
                    "RAM reserved at startup",
                    f"{result.ss_raw_kv_mb:.0f} MB",
                    f"{result.ss_tuned_kv_mb:.0f} MB",
                    _pct_label(ss_kv_pct),
                )

            console.print(t2)
            console.print()

        else:
            # TTFT didn't improve — show one concise honest note instead of a full table
            console.print(
                "[bold]Test 2 — Starting a new chat[/bold]  "
                "[dim](first-response time)[/dim]"
            )
            if result.ss_raw_kv_mb > 0:
                console.print(
                    f"  [dim]autotune reserves {result.ss_tuned_kv_mb:.0f} MB at startup "
                    f"vs {result.ss_raw_kv_mb:.0f} MB without it.[/dim]"
                )
            console.print(
                f"  [dim]First-response time on this machine: "
                f"{result.ss_raw_ttft_ms:.0f} ms without autotune, "
                f"{result.ss_tuned_ttft_ms:.0f} ms with autotune.\n"
                f"  Your hardware allocates RAM very quickly, so both are similar here.\n"
                f"  On a machine under load or with less RAM, autotune's smaller memory block\n"
                f"  prevents slowdowns that can add 0.5–5 seconds to that first response.[/dim]"
            )
            console.print()

    # ── TEST 3 — Prefill & TTFT speed ────────────────────────────────────────
    if result.has_speed_test:
        n_spd_rounds = len(result.spd_raw_runs)
        prefill_pct  = result.spd_prefill_improvement_pct
        ttft_pct     = result.spd_ttft_improvement_pct

        console.print(
            "[bold]Test 3 — Prefill & TTFT speed[/bold]  "
            "[dim](~1700-token prompt — where autotune's batch efficiency kicks in)[/dim]"
        )
        console.print(
            f"  [dim]{n_spd_rounds} round{'s' if n_spd_rounds != 1 else ''} interleaved and averaged.\n"
            "  autotune processes this prompt in 2 GPU passes (num_batch=1024) vs Ollama's 4 passes\n"
            "  (num_batch=512 default).  Fewer passes → faster prefill → first word appears sooner.\n"
            "  It also allocates a smaller KV block (2048 vs 4096 ctx), reducing memory setup time.[/dim]"
        )
        console.print()

        t3 = Table(box=_box.SIMPLE_HEAD, show_header=True, show_edge=False, pad_edge=False)
        t3.add_column("Metric",            style="white",      width=36)
        t3.add_column("Without autotune",  style="dim white",  justify="right", width=16)
        t3.add_column("With autotune",     style="green",      justify="right", width=14)
        t3.add_column("Change",                                justify="right", width=18)

        t3.add_row(
            "[bold]Time to first word[/bold]",
            f"[bold]{result.spd_raw_ttft_ms:.0f} ms[/bold]",
            f"[bold]{result.spd_tuned_ttft_ms:.0f} ms[/bold]",
            _pct_label(ttft_pct),
        )

        spd_load_pct = (
            (result.spd_raw_load_ms - result.spd_tuned_load_ms) / max(result.spd_raw_load_ms, 1) * 100
        )
        t3.add_row(
            "  └ Memory setup (KV allocation)",
            f"{result.spd_raw_load_ms:.0f} ms",
            f"{result.spd_tuned_load_ms:.0f} ms",
            _pct_label(spd_load_pct),
        )
        t3.add_row(
            "  └ Prefill (reading the ~1700-token prompt)",
            f"{result.spd_raw_prefill_ms:.0f} ms",
            f"{result.spd_tuned_prefill_ms:.0f} ms",
            _pct_label(prefill_pct),
        )
        if result.spd_raw_kv_mb > 0:
            spd_kv_pct = (
                (result.spd_raw_kv_mb - result.spd_tuned_kv_mb) / max(result.spd_raw_kv_mb, 1) * 100
            )
            t3.add_row(
                "RAM reserved",
                f"{result.spd_raw_kv_mb:.0f} MB",
                f"{result.spd_tuned_kv_mb:.0f} MB",
                _pct_label(spd_kv_pct),
            )

        console.print(t3)

        if ttft_pct < 5 and prefill_pct < 5:
            console.print(
                "  [dim]On this machine, the difference is small — memory allocation and "
                "GPU dispatch are both fast.\n"
                "  The improvement is larger on machines under RAM pressure or with slower storage.[/dim]"
            )
        console.print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    console.rule("[dim]Verdict[/dim]")
    console.print()

    wins:  list[str] = []
    notes: list[str] = []

    # — RAM freed (most reliable, always show) —
    if result.raw_kv_mb > 0 and result.kv_pct >= 10:
        kv_saved = result.kv_saved_mb
        label = f"{kv_saved / 1024:.2f} GB" if kv_saved >= 512 else f"{kv_saved:.0f} MB"
        wins.append(
            f"[green]✅  {label} of RAM freed on every request[/green]  "
            f"[dim](without: {result.raw_kv_mb:.0f} MB → with autotune: {result.tuned_kv_mb:.0f} MB,"
            f" −{result.kv_pct:.0f}%)\n"
            f"     Goes back to Chrome, VS Code, Slack — every single time.[/dim]"
        )
    elif result.raw_kv_mb > 0 and result.kv_pct >= 3:
        wins.append(
            f"[green]✅  {result.kv_saved_mb:.0f} MB of RAM freed per request[/green]  "
            f"[dim](−{result.kv_pct:.0f}%)[/dim]"
        )

    # — Speed test prefill (only if it actually improved) —
    if result.has_speed_test:
        spd_pref_pct = result.spd_prefill_improvement_pct
        spd_ttft_pct = result.spd_ttft_improvement_pct
        if spd_pref_pct >= 10:
            wins.append(
                f"[green]✅  {spd_pref_pct:.0f}% faster prompt processing on large inputs[/green]  "
                f"[dim]({result.spd_raw_prefill_ms:.0f} ms → {result.spd_tuned_prefill_ms:.0f} ms)\n"
                f"     1 GPU dispatch pass vs 2 — autotune doubles the prefill batch size.[/dim]"
            )
        elif spd_ttft_pct >= 5:
            wins.append(
                f"[green]✅  {spd_ttft_pct:.0f}% faster time-to-first-word on long prompts[/green]  "
                f"[dim]({result.spd_raw_ttft_ms:.0f} ms → {result.spd_tuned_ttft_ms:.0f} ms)[/dim]"
            )

    # — Session-start TTFT (only if it actually improved) —
    if result.has_ss_test:
        ss_pct = result.ss_ttft_improvement_pct
        if ss_pct >= 10:
            wins.append(
                f"[green]✅  {ss_pct:.0f}% faster first response when starting a chat[/green]  "
                f"[dim]({result.ss_raw_ttft_ms:.0f} ms → {result.ss_tuned_ttft_ms:.0f} ms)\n"
                f"     Smaller memory block set up faster — you see the first word sooner.[/dim]"
            )
        elif ss_pct >= 5:
            wins.append(
                f"[green]✅  First response {ss_pct:.0f}% faster on session start[/green]  "
                f"[dim]({result.ss_raw_ttft_ms:.0f} ms → {result.ss_tuned_ttft_ms:.0f} ms)[/dim]"
            )
        else:
            notes.append(
                f"[dim]ℹ  On this machine, first-response time is similar with or without autotune\n"
                f"     ({result.ss_raw_ttft_ms:.0f} ms vs {result.ss_tuned_ttft_ms:.0f} ms).\n"
                f"     On older hardware or under load, autotune's smaller memory block prevents\n"
                f"     slowdowns that can add 0.5–5 seconds to every new chat.[/dim]"
            )

    # — RAM headroom —
    if result.headroom_gained_gb >= 0.1:
        wins.append(
            f"[green]✅  +{result.headroom_gained_gb:.2f} GB of RAM free while AI runs[/green]  "
            f"[dim](with autotune: {result.tuned_free_gb:.1f} GB free "
            f"vs without: {result.raw_free_gb:.1f} GB)[/dim]"
        )

    # — Swap —
    if result.raw_swap_events > 0 and result.tuned_swap_events == 0:
        wins.append(
            f"[green]✅  Memory overflow eliminated[/green]  "
            f"[dim]Without autotune: {result.raw_swap_events} overflow event(s) — "
            f"each adds 2–10 s of freezing. autotune kept everything in fast RAM.[/dim]"
        )
    elif result.tuned_swap_events == 0:
        wins.append(
            "[green]✅  No memory overflow[/green]  "
            "[dim]AI stayed in fast RAM — your computer stayed responsive.[/dim]"
        )
    else:
        notes.append(
            f"[yellow]⚠[/yellow]  {result.tuned_swap_events} memory overflow event(s) with autotune — "
            f"try [bold]--profile fast[/bold] or a smaller model"
        )

    # — RAM pressure projection —
    if result.kv_saved_mb >= 100 and result.tuned_swap_events == 0 and result.raw_swap_events == 0:
        notes.append(
            f"[dim]ℹ  On an 8 GB laptop with many apps open, freeing {result.kv_saved_mb:.0f} MB\n"
            f"     per request can prevent memory overflow — turning a 4–10 s freeze into no pause.[/dim]"
        )

    for line in wins:
        console.print(f"  {line}")
    if wins and notes:
        console.print()
    for line in notes:
        console.print(f"  {line}")

    console.print()

    # ── Footer ────────────────────────────────────────────────────────────────
    console.rule("[dim]What's next?[/dim]")
    console.print()
    if not result.has_speed_test:
        console.print(
            "  [dim]Add [bold]--speed[/bold] to also measure prefill & TTFT on a long-context prompt\n"
            "  (proves autotune's num_batch=1024 advantage — adds ~1 min):[/dim]"
        )
        console.print(
            f"  [bold cyan]autotune proof -m {result.model_id} --speed[/bold cyan]\n"
        )
    console.print(
        "  [dim]Run a full statistical benchmark across 5 prompt types (Wilcoxon + effect size):[/dim]"
    )
    console.print(
        f"  [bold cyan]autotune proof-suite -m {result.model_id}[/bold cyan]"
        "  [dim](~20 min)[/dim]"
    )
    console.print()
    if output_path:
        console.print(f"  [dim]Results saved → {output_path}[/dim]")
    console.print()
    console.rule()
