"""
Catalog auto-updater.

On startup (or on `autotune catalog update`), checks whether the local
model_catalog.json is stale and, if so, fetches fresh data from:

  1. Hugging Face Hub API  — trending GGUF + MLX models
  2. Ollama registry API   — checks known model families for new tags

New models are merged in with `is_new=True`.  Existing entries keep all
their curated fields; only the `is_new` flag and `added_at` are written
for brand-new entries.  The update runs in a daemon thread so it never
blocks the CLI.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from autotune.models.catalog import (
    CATALOG_PATH,
    CatalogEntry,
    _BASELINE_ENTRIES,
    _fmt_ts,
    _parse_ts,
    get_entries,
    is_stale,
    load_catalog,
    save_catalog,
)

_FETCH_TIMEOUT = 8          # seconds per HTTP request
_HF_API_BASE   = "https://huggingface.co/api"
_OLLAMA_REG    = "https://registry.ollama.ai/v2/library"
_UA = "autotune-catalog-updater/2 (+https://github.com/tanav2202/local-llm-optimizer)"

# Families to check for new tags on the Ollama registry
_OLLAMA_FAMILIES = [
    "qwen3", "qwen2.5", "qwen2.5-coder", "deepseek-r1",
    "llama3.3", "llama3.2", "llama3.1", "llama4",
    "phi4", "phi3.5", "phi4-mini",
    "gemma3", "gemma4", "gemma2",
    "mistral", "mistral-small", "mistral-nemo",
    "smollm2", "qwq",
]

# HuggingFace model card tags we consider when looking for new models
_HF_LLM_TAGS = ["gguf", "mlx", "text-generation"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refresh_if_stale(catalog: dict, background: bool = True) -> None:
    """
    Check staleness and, if needed, refresh in a daemon thread (or inline).
    Safe to call from any CLI entry point — never raises.
    """
    if not is_stale(catalog):
        return
    if background:
        t = threading.Thread(target=_do_refresh, args=(catalog,), daemon=True)
        t.start()
    else:
        _do_refresh(catalog)


def force_refresh() -> tuple[dict, int, int]:
    """
    Synchronously refresh the catalog, regardless of staleness.

    Returns (updated_catalog, new_count, updated_count).
    """
    catalog = load_catalog()
    new_count, updated_count = _do_refresh(catalog)
    return load_catalog(), new_count, updated_count


# ---------------------------------------------------------------------------
# Core refresh logic
# ---------------------------------------------------------------------------

def _do_refresh(catalog: dict) -> tuple[int, int]:
    """
    Fetch fresh model data and merge into *catalog* (mutates in place + saves).
    Returns (new_count, updated_count).  Never raises.
    """
    new_count = 0
    updated_count = 0
    existing_ids: set[str] = {m["id"] for m in catalog.get("models", [])}

    try:
        fresh: list[dict] = []
        fresh.extend(_fetch_hf_gguf_trending())
        fresh.extend(_fetch_hf_mlx_trending())
        fresh.extend(_fetch_ollama_tag_check(existing_ids))

        for entry in fresh:
            eid = entry.get("id", "")
            if not eid:
                continue
            if eid not in existing_ids:
                entry["is_new"]   = True
                entry["added_at"] = _fmt_ts(time.time())
                catalog["models"].append(entry)
                existing_ids.add(eid)
                new_count += 1
            else:
                # Update size/bench data if the entry looks more complete
                idx = next(i for i, m in enumerate(catalog["models"]) if m["id"] == eid)
                existing = catalog["models"][idx]
                changed = _merge_entry(existing, entry)
                if changed:
                    updated_count += 1
    except Exception:
        pass

    # Stamp the update time
    now = time.time()
    interval = catalog.get("update_interval_days", 4)
    catalog["last_updated"] = _fmt_ts(now)
    catalog["next_update"]  = _fmt_ts(now + interval * 86400)

    try:
        save_catalog(catalog)
    except Exception:
        pass

    return new_count, updated_count


def _merge_entry(existing: dict, fresh: dict) -> bool:
    """
    Merge benchmark scores and sizes from *fresh* into *existing*.
    Returns True if anything changed.
    """
    changed = False
    for key in ("bench_mmlu", "bench_humaneval", "bench_gsm8k", "bench_math",
                "size_gb", "ram_gb", "context_k", "description", "highlight",
                "hf_repo", "mlx_repo"):
        fresh_val = fresh.get(key)
        if fresh_val is not None and fresh_val != "" and existing.get(key) != fresh_val:
            # Only update if the fresh value is non-trivially different
            if isinstance(fresh_val, float) and existing.get(key) is not None:
                if abs(fresh_val - existing.get(key, 0)) < 0.001:
                    continue
            existing[key] = fresh_val
            changed = True
    return changed


# ---------------------------------------------------------------------------
# HuggingFace Hub trending
# ---------------------------------------------------------------------------

def _fetch_hf_gguf_trending() -> list[dict]:
    """
    Fetch top trending GGUF models from the HF Hub API.
    Only includes models that look like instruction-tuned LLMs.
    """
    url = f"{_HF_API_BASE}/models?sort=trending&limit=50&filter=gguf&full=false"
    raw = _get_json(url)
    if not raw or not isinstance(raw, list):
        return []

    results: list[dict] = []
    for item in raw:
        entry = _hf_item_to_entry(item, source="hf")
        if entry:
            results.append(entry)
    return results


def _fetch_hf_mlx_trending() -> list[dict]:
    """Fetch top trending MLX models from the HF Hub API."""
    url = f"{_HF_API_BASE}/models?sort=trending&limit=50&filter=mlx&full=false"
    raw = _get_json(url)
    if not raw or not isinstance(raw, list):
        return []

    results: list[dict] = []
    for item in raw:
        entry = _hf_item_to_entry(item, source="mlx")
        if entry:
            results.append(entry)
    return results


def _hf_item_to_entry(item: dict, source: str) -> Optional[dict]:
    """Convert an HF Hub API model item to a catalog entry dict (or None)."""
    model_id: str = item.get("modelId") or item.get("id") or ""
    if not model_id:
        return None

    # Skip non-instruct models to avoid base weights
    lower_id = model_id.lower()
    instruct_hints = ["instruct", "chat", "it", "-4bit", "-q4", "gguf"]
    if not any(h in lower_id for h in instruct_hints):
        return None

    # Parse params from model name
    params_b = _parse_params_b(model_id)
    if params_b is None:
        return None
    if params_b > 500:  # skip gigantic models
        return None

    size_gb = round(params_b * 0.61, 1)
    ram_gb  = round(size_gb * 1.25, 1)
    tier    = _classify_tier(params_b)
    name    = _humanize_name(model_id)
    tags    = _infer_tags(model_id)

    return {
        "id":           model_id,
        "name":         name,
        "family":       _infer_family(model_id),
        "tier":         tier,
        "parameters_b": params_b,
        "size_gb":      size_gb,
        "ram_gb":       ram_gb,
        "context_k":    32,  # unknown; conservative
        "ollama_tag":   "",
        "hf_repo":      model_id if source != "mlx" else "",
        "mlx_repo":     model_id if source == "mlx" else "",
        "description":  f"Trending on HuggingFace ({source.upper()})",
        "highlight":    "",
        "tags":         tags,
        "bench_mmlu":   None,
        "bench_humaneval": None,
        "bench_gsm8k":  None,
        "bench_math":   None,
        "speed_class":  "fast" if params_b < 8 else "medium" if params_b < 30 else "slow",
        "source":       source,
        "added_at":     _fmt_ts(time.time()),
        "is_new":       True,
    }


# ---------------------------------------------------------------------------
# Ollama tag check — detect new sizes in known families
# ---------------------------------------------------------------------------

def _fetch_ollama_tag_check(existing_ids: set[str]) -> list[dict]:
    """
    For each known Ollama model family, list tags from the registry and
    return entries for tags not yet in the catalog.
    """
    results: list[dict] = []
    for family in _OLLAMA_FAMILIES:
        try:
            tags = _fetch_ollama_tags(family)
            for tag in tags:
                full_id = f"{family}:{tag}"
                if full_id in existing_ids:
                    continue
                # Only add tags that look like size variants
                if not re.search(r"\d+[bBmMkK]", tag):
                    continue
                params_b = _parse_params_b(tag) or _parse_params_b(full_id)
                if not params_b:
                    continue
                size_gb = round(params_b * 0.61, 1)
                entry: dict = {
                    "id":           full_id,
                    "name":         f"{family.title()} {tag.upper()}",
                    "family":       family,
                    "tier":         _classify_tier(params_b),
                    "parameters_b": params_b,
                    "size_gb":      size_gb,
                    "ram_gb":       round(size_gb * 1.25, 1),
                    "context_k":    32,
                    "ollama_tag":   full_id,
                    "hf_repo":      "",
                    "mlx_repo":     "",
                    "description":  f"New tag detected from Ollama registry: {family}:{tag}",
                    "highlight":    "",
                    "tags":         _infer_tags(family),
                    "bench_mmlu":   None,
                    "bench_humaneval": None,
                    "bench_gsm8k":  None,
                    "bench_math":   None,
                    "speed_class":  "fast" if params_b < 8 else "medium" if params_b < 30 else "slow",
                    "source":       "ollama",
                    "added_at":     _fmt_ts(time.time()),
                    "is_new":       True,
                }
                results.append(entry)
        except Exception:
            continue
    return results


def _fetch_ollama_tags(family: str) -> list[str]:
    """Return available tags for an Ollama model family."""
    url = f"{_OLLAMA_REG}/{family}/tags/list"
    data = _get_json(url)
    if not data or "tags" not in data:
        return []
    return data["tags"]


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_json(url: str) -> Optional[dict | list]:
    """GET *url* and return parsed JSON, or None on any error."""
    try:
        req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        with urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_PARAM_PATTERNS = [
    r"[:\-_/](\d+\.?\d*)[bB](?:[:\-_\s.]|$)",
    r"^(\d+\.?\d*)[bB](?:[:\-_\s.]|$)",
    r"-(\d+\.?\d*)[bB](?:-|$)",
    r"(\d{1,3}\.?\d*)[bB]\b",
]

def _parse_params_b(name: str) -> Optional[float]:
    lower = name.lower()
    for pat in _PARAM_PATTERNS:
        m = re.search(pat, lower)
        if m:
            v = float(m.group(1))
            if 0 < v < 1000:
                return v
    # Check for million-param models (e.g. "135m")
    m = re.search(r"(\d+)[mM]\b", lower)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def _classify_tier(params_b: float) -> str:
    if params_b < 3:    return "tiny"
    if params_b < 8:    return "small"
    if params_b < 15:   return "medium"
    if params_b < 35:   return "large"
    if params_b < 80:   return "xl"
    return "flagship"


def _infer_family(model_id: str) -> str:
    lower = model_id.lower()
    for fam in ["qwen3", "qwen2.5", "qwen2", "qwen", "llama4", "llama3",
                "llama", "phi4", "phi3", "phi", "gemma4", "gemma3", "gemma2", "gemma",
                "mistral", "deepseek", "smollm", "falcon", "yi", "stablelm"]:
        if fam in lower:
            return fam
    return model_id.split("/")[-1].split(":")[0].lower()


def _infer_tags(model_id: str) -> list[str]:
    lower = model_id.lower()
    tags: list[str] = ["general"]
    if any(w in lower for w in ["coder", "code", "coding"]):
        tags.append("coding")
    if any(w in lower for w in ["math", "qwq", "r1", "reasoning"]):
        tags.append("reasoning")
    if any(w in lower for w in ["multi", "ml-community", "qwen", "mistral"]):
        tags.append("multilingual")
    if any(w in lower for w in ["moe", "mixture", "a3b", "a22b"]):
        tags.append("moe")
    return list(dict.fromkeys(tags))  # dedup, preserve order


def _humanize_name(model_id: str) -> str:
    base = model_id.split("/")[-1].split(":")[0]
    return base.replace("-", " ").replace("_", " ").title()
