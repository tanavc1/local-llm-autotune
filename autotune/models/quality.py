"""
Model quality reference data.

Scores are from publicly published benchmarks:
  MMLU  — Massive Multitask Language Understanding (5-shot, % correct)
           Tests breadth of world knowledge and reasoning across 57 subjects.
  HumanEval — OpenAI's Python coding benchmark (pass@1 %)
  Reason    — Composite reasoning score (MATH, GSM8K, ARC-Challenge averages)

All numbers are approximate — they vary by exact quantization, eval harness,
and prompting style.  Treat them as relative comparisons, not absolute truth.

Tiers:
  S  — Flagship (≥ 80 MMLU) — best-in-class for their size or overall
  A  — Strong   (70–79 MMLU) — excellent for everyday work
  B  — Good     (60–69 MMLU) — solid; handles most tasks well
  C  — Capable  (50–59 MMLU) — adequate for basic tasks; may struggle with complex reasoning
  D  — Limited  (< 50 MMLU)  — small/specialized; best for narrow use cases
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QualityInfo:
    tier: str            # "S" | "A" | "B" | "C" | "D"
    mmlu: Optional[float]       # MMLU % (None = not publicly benchmarked)
    humaneval: Optional[float]  # HumanEval % (None = unknown)
    reason: Optional[float]     # Composite reasoning %
    note: str                   # one-line summary


# Keys are base model names (no tag, lowercase).  Lookup is done via
# _normalize() which strips tags and lowercases.  More specific keys
# (e.g. "qwen2.5-coder:7b") take priority over broader ones ("qwen2.5").

_QUALITY_DB: dict[str, QualityInfo] = {

    # ── SmolLM2 ─────────────────────────────────────────────────────────────
    "smollm2:135m":  QualityInfo("D", 28.1, None,  None,  "Ultra-tiny; simple classification/completion only"),
    "smollm2:360m":  QualityInfo("D", 33.9, None,  None,  "Tiny; useful for on-device keyword tasks"),
    "smollm2:1.7b":  QualityInfo("D", 48.2, None,  None,  "Small but surprisingly coherent for its size"),

    # ── Llama 3.x ────────────────────────────────────────────────────────────
    "llama3.2:1b":   QualityInfo("D", 47.3, 28.5,  None,  "Ultra-fast, limited reasoning depth"),
    "llama3.2:3b":   QualityInfo("C", 58.3, 34.4,  None,  "Good balance of speed and quality for a 3B"),
    "llama3.1:8b":   QualityInfo("A", 73.0, 62.9,  52.2,  "Meta's well-rounded 8B; excellent instruction following"),

    # ── Qwen 2.5 base ────────────────────────────────────────────────────────
    "qwen2.5:0.5b":  QualityInfo("D", 46.8, None,  None,  "Minimal size; limited complex reasoning"),
    "qwen2.5:1.5b":  QualityInfo("C", 60.4, None,  None,  "Strong for its size vs other 1-2B models"),
    "qwen2.5:3b":    QualityInfo("B", 65.5, 40.2,  None,  "Punches above its weight; great multilingual"),
    "qwen2.5:7b":    QualityInfo("A", 74.9, 60.4,  62.1,  "Excellent all-rounder; best 7B for general tasks"),
    "qwen2.5:14b":   QualityInfo("S", 79.4, 71.7,  70.8,  "Near-flagship quality; strong across all domains"),
    "qwen2.5:32b":   QualityInfo("S", 83.3, 79.9,  76.4,  "Premium quality; competes with much larger models"),
    "qwen2.5:72b":   QualityInfo("S", 86.1, 84.8,  80.2,  "Near-frontier open-weight model"),

    # ── Qwen 2.5 Coder ───────────────────────────────────────────────────────
    "qwen2.5-coder:1.5b": QualityInfo("C", 55.0, 56.3, None,  "Best small coding model; HumanEval near larger rivals"),
    "qwen2.5-coder:7b":   QualityInfo("B", 69.8, 79.3, None,  "Best-in-class 7B coder; beats larger general models on code"),
    "qwen2.5-coder:14b":  QualityInfo("A", 77.1, 85.2, None,  "State-of-the-art coding under 20 GB"),
    "qwen2.5-coder:32b":  QualityInfo("S", 82.5, 90.2, None,  "Top open-source coding model available"),

    # ── Phi family ───────────────────────────────────────────────────────────
    "phi4-mini":     QualityInfo("A", 72.1, 63.4,  68.5,  "Microsoft's efficient 3.8B; strong reasoning for size"),
    "phi4":          QualityInfo("S", 84.0, 82.4,  79.3,  "Microsoft's flagship 14B; near GPT-4o quality on reasoning"),
    "phi3:mini":     QualityInfo("B", 67.8, 57.5,  None,  "Phi-3 mini; good reasoning, smaller footprint"),
    "phi3:medium":   QualityInfo("A", 77.9, 70.6,  None,  "Phi-3 medium; strong on instruction tasks"),

    # ── Gemma 2 / 3 ───────────────────────────────────────────────────────────
    "gemma2:2b":     QualityInfo("C", 52.2, 26.3,  None,  "Google Gemma 2 2B; solid small model"),
    "gemma2:9b":     QualityInfo("A", 70.7, 58.7,  None,  "Google Gemma 2 9B; excellent instruction following"),
    "gemma2:27b":    QualityInfo("S", 78.0, 72.5,  None,  "Gemma 2 27B; competitive with 70B models on some tasks"),
    "gemma3:1b":     QualityInfo("D", 42.1, None,  None,  "Gemma 3 1B; vision capable but limited language reasoning"),
    "gemma3:4b":     QualityInfo("C", 59.2, 36.0,  None,  "Gemma 3 4B; multimodal, reasonable language quality"),
    "gemma3:12b":    QualityInfo("A", 72.6, 65.5,  None,  "Gemma 3 12B; strong across English and multilingual tasks"),
    "gemma3:27b":    QualityInfo("S", 79.8, 75.4,  None,  "Gemma 3 27B; Google's previous flagship"),
    # ── Gemma 4 (April 2025) ──────────────────────────────────────────────────
    # Benchmark scores from Google's official Gemma 4 release benchmarks.
    # MMLU = 5-shot; HumanEval = pass@1; note: these are *large* scores for
    # the parameter count because of the MoE architecture and PLE technique.
    "gemma4:e2b":    QualityInfo("B", 69.0, 62.0,  None,  "Gemma 4 2B; compact, multimodal, 128k context"),
    "gemma4:e4b":    QualityInfo("A", 74.0, 68.0,  None,  "Gemma 4 4B; best small model for 16GB laptops"),
    "gemma4:26b":    QualityInfo("S", 85.2, 89.0,  None,  "Gemma 4 26B MoE (4B active); near-frontier, 256k context"),
    "gemma4:31b":    QualityInfo("S", 85.5, 90.0,  None,  "Gemma 4 31B dense; Google's flagship open model"),
    "gemma4":        QualityInfo("A", 74.0, 68.0,  None,  "Gemma 4 (default tag); multimodal, 128k context"),

    # ── Mistral ──────────────────────────────────────────────────────────────
    "mistral:7b":    QualityInfo("B", 62.5, 50.0,  48.0,  "Mistral 7B v0.3; fast and reliable workhorse"),
    "mistral-nemo":  QualityInfo("B", 68.0, 62.0,  None,  "Mistral NeMo 12B; good balance of size and quality"),
    "mixtral:8x7b":  QualityInfo("A", 70.6, 63.9,  57.5,  "MoE architecture; quality of 40B at 7B inference cost"),

    # ── DeepSeek R1 distills ─────────────────────────────────────────────────
    "deepseek-r1:1.5b": QualityInfo("C", 55.0, 42.0,  62.0,  "Chain-of-thought distill; reasoning > raw MMLU"),
    "deepseek-r1:7b":   QualityInfo("B", 67.1, 63.2,  78.5,  "R1 chain-of-thought; strong math/logic"),
    "deepseek-r1:8b":   QualityInfo("B", 68.5, 68.7,  79.2,  "Llama 3.1 8B base distill; excellent reasoning"),
    "deepseek-r1:14b":  QualityInfo("A", 76.8, 79.8,  85.4,  "Qwen 2.5 14B base distill; top reasoning under 15 GB"),
    "deepseek-r1:32b":  QualityInfo("S", 82.6, 86.7,  89.6,  "Best open reasoning model under 40 GB"),
    "deepseek-r1:70b":  QualityInfo("S", 85.9, 89.2,  91.7,  "Near-frontier reasoning; beats many proprietary models"),
    # MLX-style distill names include "distill-qwen" or "distill-llama" in the middle
    "deepseek-r1-distill-qwen-1.5b":  QualityInfo("C", 55.0, 42.0, 62.0,  "R1 Qwen 1.5B distill; strong reasoning for tiny size"),
    "deepseek-r1-distill-qwen-7b":    QualityInfo("B", 67.1, 63.2, 78.5,  "R1 Qwen 7B distill; strong chain-of-thought"),
    "deepseek-r1-distill-qwen-14b":   QualityInfo("A", 76.8, 79.8, 85.4,  "R1 Qwen 14B distill; top reasoning under 15 GB"),
    "deepseek-r1-distill-qwen-32b":   QualityInfo("S", 82.6, 86.7, 89.6,  "Best open reasoning model under 40 GB"),
    "deepseek-r1-distill-llama-8b":   QualityInfo("B", 68.5, 68.7, 79.2,  "R1 Llama 8B distill; excellent reasoning"),
    "deepseek-r1-distill-llama-70b":  QualityInfo("S", 85.9, 89.2, 91.7,  "Near-frontier reasoning"),

    # ── QwQ ──────────────────────────────────────────────────────────────────
    "qwq:32b":       QualityInfo("S", 84.3, 87.5,  91.2,  "Qwen reasoning model; exceptional math/science"),
    "qwq":           QualityInfo("S", 84.3, 87.5,  91.2,  "Qwen reasoning model; exceptional math/science"),

    # ── Llama 3.3 / 3 ────────────────────────────────────────────────────────
    "llama3.3:70b":  QualityInfo("S", 85.7, 81.2,  81.8,  "Meta's current flagship open model; near GPT-4 quality"),
    "llama3:8b":     QualityInfo("A", 73.0, 62.9,  52.2,  "Original Llama 3 8B"),
    "llama3:70b":    QualityInfo("S", 82.0, 77.0,  76.5,  "Original Llama 3 70B; very capable"),

    # ── Command R ─────────────────────────────────────────────────────────────
    "command-r":     QualityInfo("A", 73.5, None,  None,  "Cohere Command R; optimized for RAG and tool use"),
    "command-r-plus": QualityInfo("S", 80.3, None, None,  "Cohere's flagship; enterprise-grade instruction following"),

    # ── Codestral / other coding ─────────────────────────────────────────────
    "codestral":     QualityInfo("A", 72.0, 81.1,  None,  "Mistral's coding-specialist 22B"),
    "codegemma:7b":  QualityInfo("B", 64.0, 56.2,  None,  "Google CodeGemma 7B; good code but weaker general"),

    # ── Qwen 3 ────────────────────────────────────────────────────────────────
    "qwen3:0.6b":    QualityInfo("D", 45.0, None,  None,  "Qwen 3 0.6B; tiny, for on-device edge tasks"),
    "qwen3:1.7b":    QualityInfo("C", 59.0, None,  None,  "Qwen 3 1.7B; strong reasoning for its size"),
    "qwen3:4b":      QualityInfo("B", 66.0, 52.0,  None,  "Qwen 3 4B; hybrid thinking model, good all-rounder"),
    "qwen3:8b":      QualityInfo("A", 74.0, 68.0,  None,  "Qwen 3 8B; near-frontier quality for its class"),
    "qwen3:14b":     QualityInfo("A", 79.0, 78.0,  None,  "Qwen 3 14B; excellent reasoning and instruction following"),
    "qwen3:30b":     QualityInfo("S", 83.0, 84.0,  None,  "Qwen 3 30B MoE; flagship-level at lower inference cost"),
    "qwen3:32b":     QualityInfo("S", 84.0, 85.0,  None,  "Qwen 3 32B dense; top open model in its size class"),

    # ── Qwen 3 VL (vision-language) ──────────────────────────────────────────
    "qwen3-vl:3b":   QualityInfo("B", 62.0, None,  None,  "Qwen 3 VL 3B; vision + language, capable for its size"),
    "qwen3-vl:8b":   QualityInfo("A", 71.0, None,  None,  "Qwen 3 VL 8B; strong vision-language model"),
    "qwen3-vl:32b":  QualityInfo("S", 81.0, None,  None,  "Qwen 3 VL 32B; top open vision-language model"),

    # ── Qwen 2.5 VL ──────────────────────────────────────────────────────────
    "qwen2.5-vl:3b": QualityInfo("B", 64.0, None,  None,  "Qwen 2.5 VL 3B; vision-language, reasonable quality"),
    "qwen2.5-vl:7b": QualityInfo("A", 72.0, None,  None,  "Qwen 2.5 VL 7B; strong multimodal capabilities"),
    "qwen2.5-vl:72b":QualityInfo("S", 85.0, None,  None,  "Qwen 2.5 VL 72B; near-frontier vision-language model"),

    # ── Llama 4 ──────────────────────────────────────────────────────────────
    "llama4:scout":  QualityInfo("A", 76.0, 68.0,  None,  "Meta Llama 4 Scout 17B MoE; fast and capable"),
    "llama4:maverick": QualityInfo("S", 85.0, 77.0, None, "Meta Llama 4 Maverick 400B MoE; frontier-tier quality"),

    # ── TinyLlama ────────────────────────────────────────────────────────────
    "tinyllama":     QualityInfo("D", 38.3, 15.8,  None,  "1.1B; ultra-fast but very limited capability"),

    # ── Vicuna / older models ─────────────────────────────────────────────────
    "vicuna":        QualityInfo("C", 56.0, None,  None,  "RLHF-tuned Llama; decent instruction following"),
}

# ── Tier display helpers ──────────────────────────────────────────────────────

_TIER_LABEL: dict[str, str] = {
    "S": "[bold green]S — Flagship[/bold green]",
    "A": "[green]A — Strong[/green]",
    "B": "[yellow]B — Good[/yellow]",
    "C": "[yellow]C — Capable[/yellow]",
    "D": "[dim]D — Limited[/dim]",
}

# Short single-character badge for table cells where space is tight
_TIER_BADGE: dict[str, str] = {
    "S": "[bold green]S[/bold green]",
    "A": "[green]A[/green]",
    "B": "[yellow]B[/yellow]",
    "C": "[yellow]C[/yellow]",
    "D": "[dim]D[/dim]",
}

_TIER_COLOR: dict[str, str] = {
    "S": "bold green",
    "A": "green",
    "B": "yellow",
    "C": "yellow",
    "D": "dim",
}


_HUB_PREFIXES = ("mlx-community/", "unsloth/", "bartowski/", "ggml-org/", "lmstudio-community/")
_NOISE_SUFFIXES = ("-instruct", "-chat", "-it", "-base", "-preview",
                   "-4bit", "-8bit", "-6bit", "-3bit", "-2bit", "-f16", "-bf16",
                   "-q4_k_m", "-q4_k_s", "-q5_k_m", "-q8_0")


def _normalize(model_id: str) -> str:
    """Lower-case and strip hub prefixes + :latest tag."""
    s = model_id.lower().strip()
    for prefix in _HUB_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if s.endswith(":latest"):
        s = s[:-7]
    return s


def _slug(s: str) -> str:
    """Collapse dashes, dots, underscores, colons for fuzzy comparison."""
    import re
    return re.sub(r"[-._:]", "", s)


def _strip_noise(s: str) -> str:
    """Strip common noise suffixes iteratively."""
    changed = True
    while changed:
        changed = False
        for suf in _NOISE_SUFFIXES:
            if s.endswith(suf):
                s = s[:-len(suf)]
                changed = True
    return s


def get_quality(model_id: str) -> Optional[QualityInfo]:
    """
    Return quality info using a multi-pass fuzzy lookup.

    Pass 1 — exact match on normalized id (e.g. "qwen2.5:7b")
    Pass 2 — longest DB key that is a prefix of the normalized id
    Pass 3 — same after stripping noise suffixes (handles MLX names like
              "phi-4-mini-instruct-4bit" → "phi4-mini")
    Pass 4 — slug comparison (dashes/dots removed) for hyphen variants
    """
    norm = _normalize(model_id)

    # Pass 1: exact
    if norm in _QUALITY_DB:
        return _QUALITY_DB[norm]

    # Pass 2: longest DB key that is a prefix of norm
    best_key = max(
        (k for k in _QUALITY_DB if norm.startswith(k)),
        key=len, default="",
    )
    if best_key:
        return _QUALITY_DB[best_key]

    # Pass 3: strip noise from norm, retry passes 1 & 2
    clean = _strip_noise(norm)
    if clean != norm:
        if clean in _QUALITY_DB:
            return _QUALITY_DB[clean]
        best_key = max(
            (k for k in _QUALITY_DB if clean.startswith(k)),
            key=len, default="",
        )
        if best_key:
            return _QUALITY_DB[best_key]

    # Pass 4: slug comparison — strip all separators and compare
    norm_slug = _slug(clean or norm)
    for key, info in _QUALITY_DB.items():
        if _slug(key) == norm_slug:
            return info
    # Partial slug prefix match (e.g. "phi4mini" startswith "phi4mini")
    for key, info in _QUALITY_DB.items():
        key_slug = _slug(key)
        if norm_slug.startswith(key_slug) and len(key_slug) >= 4:
            return info

    return None


def tier_markup(tier: str) -> str:
    return _TIER_LABEL.get(tier, tier)


def tier_color(tier: str) -> str:
    return _TIER_COLOR.get(tier, "")


def tier_badge(tier: str) -> str:
    """Single-character Rich markup badge for compact table cells."""
    return _TIER_BADGE.get(tier, tier)
