"""
Model download feasibility guard.

Before any model pull (Ollama, MLX/HuggingFace) autotune checks that the
download will actually fit on disk and that the model can run in the
machine's RAM.  Three verdicts are returned:

  "ok"      — safe to proceed
  "warn"    — close call; user is warned but the pull is allowed
  "blocked" — egregious mismatch; the pull is refused

Callers decide how to surface the result:
  - CLI: print rich text, prompt on "warn", exit on "blocked"
  - Dashboard API: return 409 on "blocked", include warning in response on "warn"

Size estimation
---------------
Known models are looked up in _KNOWN_SIZES.  For unknown models the
parameter count is parsed from the name (e.g. "14b" → 14B params) and a
Q4_K_M download-size approximation is applied (~0.61 bytes/param).

RAM estimate
------------
runtime_ram ≈ download_size × 1.25
(model weights + ~25% for KV cache and framework overhead)

Thresholds
----------
BLOCKED — disk:  model_size  > free_disk × 0.95
BLOCKED — RAM:   ram_needed  > total_ram × 1.4   (will never fit)
WARN    — disk:  model_size  > free_disk × 0.80
WARN    — RAM:   ram_needed  > total_ram × 0.85  (tight, expect swapping)
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Known model sizes (download size in GB, Q4_K_M unless noted)
# ---------------------------------------------------------------------------

_KNOWN_SIZES: dict[str, float] = {
    # ── Tiny ──────────────────────────────────────────────────────────────
    "smollm2:135m":               0.3,
    "smollm2:1.7b":               1.1,
    "qwen3:0.6b":                 0.5,
    "qwen3:1.7b":                 1.1,
    "llama3.2:1b":                1.3,
    # ── Small ─────────────────────────────────────────────────────────────
    "llama3.2:3b":                2.0,
    "qwen3:4b":                   2.6,
    "qwen3.5:4b":                 2.6,
    "gemma3n:e2b":                1.6,
    "gemma3n:e4b":                2.5,
    "qwen2.5:3b":                 2.0,
    "phi4-mini":                  2.5,
    "phi3:3.8b":                  2.3,
    # ── Medium ────────────────────────────────────────────────────────────
    "qwen3:8b":                   5.2,
    "qwen3.5:9b":                 5.6,
    "gpt-oss:20b":               14.0,
    "gemma4:12b":                 8.1,
    "llama3.1:8b":                4.7,
    "llama3.2:8b":                4.7,
    "qwen2.5:7b":                 4.7,
    "qwen2.5-coder:7b":           4.7,
    "deepseek-r1:7b":             4.7,
    "mistral:7b":                 4.1,
    "mistral:7b-instruct":        4.1,
    "gemma3:12b":                 8.1,
    "phi4:14b":                   9.1,
    # ── Large ─────────────────────────────────────────────────────────────
    "qwen3:14b":                  9.0,
    "qwen2.5:14b":                9.0,
    "qwen2.5-coder:14b":          9.0,
    "deepseek-r1:14b":            9.0,
    "devstral:24b":              14.0,
    "qwen3:30b-a3b":             17.0,
    # ── XL ────────────────────────────────────────────────────────────────
    "qwen3:32b":                 20.0,
    "qwen2.5:32b":               20.0,
    "qwen2.5-coder:32b":         20.0,
    "deepseek-r1:32b":           20.0,
    "qwq:32b":                   20.0,
    "qwen3.6:27b":               17.0,
    "qwen3-coder:30b":           19.0,
    "gemma4:26b":                17.0,
    "qwen3.5:35b":               21.0,
    "llama3.1:70b":              40.0,
    "llama3.3:70b":              40.0,
    "qwen2.5:72b":               43.0,
    # ── Absurdly large ────────────────────────────────────────────────────
    "gpt-oss:120b":              65.0,
    "llama3.1:405b":            230.0,
    "llama3.1:405b-instruct":   230.0,
    # ── Popular MLX / HF (approx) ─────────────────────────────────────────
    "mlx-community/qwen3-8b-4bit":                     5.2,
    "mlx-community/qwen3-14b-4bit":                    9.0,
    "mlx-community/qwen3-32b-4bit":                   20.0,
    "mlx-community/llama-3.1-8b-instruct-4bit":        4.7,
    "mlx-community/llama-3.2-3b-instruct-4bit":        2.0,
    "mlx-community/mistral-7b-instruct-v0.3-4bit":     4.1,
    "mlx-community/phi-4-4bit":                        9.1,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityResult:
    verdict:       str    # "ok" | "warn" | "blocked"
    reason:        str    # human-readable explanation
    model_size_gb: float  # estimated download size in GB
    ram_needed_gb: float  # estimated runtime RAM needed in GB
    free_disk_gb:  float  # current free disk in GB
    total_ram_gb:  float  # total system RAM in GB

    def to_dict(self) -> dict:
        return {
            "verdict":       self.verdict,
            "reason":        self.reason,
            "model_size_gb": self.model_size_gb,
            "ram_needed_gb": self.ram_needed_gb,
            "free_disk_gb":  round(self.free_disk_gb, 1),
            "total_ram_gb":  round(self.total_ram_gb, 1),
        }


def estimate_size_gb(model_name: str) -> float | None:
    """
    Return estimated download size in GB for *model_name*.

    Lookup order:
      1. _KNOWN_SIZES (hard-coded, fast)
      2. Model catalog (~/.autotune/model_catalog.json — richer, auto-updated)
      3. Parameter-count parsing + Q4_K_M approximation
    """
    tag = model_name.lower().strip()

    if tag in _KNOWN_SIZES:
        return _KNOWN_SIZES[tag]

    # Strip trailing :latest
    base = re.sub(r":latest$", "", tag)
    if base in _KNOWN_SIZES:
        return _KNOWN_SIZES[base]

    # Try the catalog (loaded without network — only reads local JSON)
    try:
        from autotune.models.catalog import load_catalog, get_entries
        _cat = load_catalog()
        for entry in get_entries(_cat):
            if entry.id.lower() == tag or entry.id.lower() == base:
                return entry.size_gb
            if entry.ollama_tag.lower() == tag or entry.ollama_tag.lower() == base:
                return entry.size_gb
    except Exception:
        pass

    # Parse parameter count (e.g. "8b", "14b", "72b", "0.6b", "3.8b")
    patterns = [
        r"[:\-_/](\d+\.?\d*)[bB](?:[:\-_\s.]|$)",   # after separator: qwen3:8b, phi-3.8b
        r"^(\d+\.?\d*)[bB](?:[:\-_\s.]|$)",           # at start of string
        r"-(\d+\.?\d*)[bB](?:-instruct|-chat|-base)",  # before variant suffix
        r"(\d{1,3}\.?\d*)[bB]\b",                      # loose match
    ]
    for pat in patterns:
        m = re.search(pat, tag)
        if m:
            params_b = float(m.group(1))
            if 0 < params_b < 1000:
                return round(params_b * 0.61, 1)

    return None


def estimate_ram_gb(model_size_gb: float) -> float:
    """Estimate runtime RAM: model weights + ~25% for KV cache and OS overhead."""
    return round(model_size_gb * 1.25, 1)


def check_feasibility(
    model_name: str,
    source: str = "ollama",
    estimated_size_gb: float | None = None,
) -> FeasibilityResult:
    """
    Check whether downloading and running *model_name* is feasible.

    Parameters
    ----------
    model_name        : Ollama tag, HF repo ID, or GGUF path
    source            : "ollama" | "hf" | "mlx"  (informational only)
    estimated_size_gb : Exact size if known; otherwise estimated from name

    Returns
    -------
    FeasibilityResult with verdict "ok", "warn", or "blocked"
    """
    import psutil

    # ── Disk ───────────────────────────────────────────────────────────────
    try:
        disk = shutil.disk_usage("/")
        free_disk_gb = disk.free / 1024 ** 3
    except Exception:
        free_disk_gb = 9_999.0  # cannot read → don't block

    # ── RAM ────────────────────────────────────────────────────────────────
    try:
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / 1024 ** 3
    except Exception:
        total_ram_gb = 8.0  # conservative fallback

    # ── Model size ─────────────────────────────────────────────────────────
    size_gb: float | None = (
        estimated_size_gb if estimated_size_gb and estimated_size_gb > 0 else None
    )
    if size_gb is None:
        size_gb = estimate_size_gb(model_name)

    if size_gb is None or size_gb <= 0:
        # Size unknown — cannot make a definitive call; allow through
        return FeasibilityResult(
            verdict="ok",
            reason="Model size unknown — skipping feasibility check.",
            model_size_gb=0.0,
            ram_needed_gb=0.0,
            free_disk_gb=free_disk_gb,
            total_ram_gb=total_ram_gb,
        )

    ram_needed_gb = estimate_ram_gb(size_gb)

    # ── BLOCKED ────────────────────────────────────────────────────────────
    if size_gb > free_disk_gb * 0.95:
        return FeasibilityResult(
            verdict="blocked",
            reason=(
                f"Not enough disk space: {model_name!r} needs ~{size_gb:.1f} GB "
                f"but only {free_disk_gb:.1f} GB is free. "
                "Remove unused models (autotune delete) or free up disk space first."
            ),
            model_size_gb=size_gb,
            ram_needed_gb=ram_needed_gb,
            free_disk_gb=free_disk_gb,
            total_ram_gb=total_ram_gb,
        )

    if ram_needed_gb > total_ram_gb * 1.4:
        return FeasibilityResult(
            verdict="blocked",
            reason=(
                f"{model_name!r} needs ~{ram_needed_gb:.1f} GB RAM to load, "
                f"but this machine only has {total_ram_gb:.1f} GB total. "
                "This model will never fit in memory here — "
                "choose a smaller or more quantized version."
            ),
            model_size_gb=size_gb,
            ram_needed_gb=ram_needed_gb,
            free_disk_gb=free_disk_gb,
            total_ram_gb=total_ram_gb,
        )

    # ── WARN ───────────────────────────────────────────────────────────────
    disk_tight = size_gb > free_disk_gb * 0.80
    ram_tight  = ram_needed_gb > total_ram_gb * 0.85

    if disk_tight or ram_tight:
        parts: list[str] = []
        if ram_tight:
            parts.append(
                f"needs ~{ram_needed_gb:.1f} GB RAM on a {total_ram_gb:.1f} GB machine "
                "(expect memory pressure and slower inference)"
            )
        if disk_tight:
            pct = size_gb / free_disk_gb * 100
            parts.append(
                f"~{size_gb:.1f} GB download uses {pct:.0f}% of your "
                f"{free_disk_gb:.1f} GB free disk"
            )
        return FeasibilityResult(
            verdict="warn",
            reason=f"{model_name!r}: " + "; ".join(parts) + ".",
            model_size_gb=size_gb,
            ram_needed_gb=ram_needed_gb,
            free_disk_gb=free_disk_gb,
            total_ram_gb=total_ram_gb,
        )

    # ── OK ─────────────────────────────────────────────────────────────────
    return FeasibilityResult(
        verdict="ok",
        reason=(
            f"{model_name!r}: ~{size_gb:.1f} GB download, "
            f"~{ram_needed_gb:.1f} GB RAM — feasible on this machine."
        ),
        model_size_gb=size_gb,
        ram_needed_gb=ram_needed_gb,
        free_disk_gb=free_disk_gb,
        total_ram_gb=total_ram_gb,
    )
