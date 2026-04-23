"""
Pre-flight model selector — airtight memory analysis before loading.

Answers three questions before a model is started:
  1. Will this model + KV cache fit in RAM without swap?
  2. What is the maximum context length that stays safely in RAM?
  3. Is there a smaller quantization that would fit with more headroom?

Memory model
------------
  total_required = weights + kv_cache(context_len, precision) + runtime_overhead
  safe threshold = available_ram * 0.85   → comfortable
  swap threshold = available_ram * 0.92   → macOS starts compressing/swapping
  oom  threshold = available_ram * 1.00   → OOM / kernel kill

KV cache formula (exact when architecture is known):
  kv_bytes = 2 * n_layers * n_kv_heads * head_dim * context_len * element_bytes
  element_bytes = 2 for F16, 1 for Q8

Architecture is extracted from Ollama's /api/show modelinfo dict.
Falls back to parameter-count-based estimates when unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Tuneable thresholds
# ---------------------------------------------------------------------------

RUNTIME_OVERHEAD_GB:  float = 0.40  # llama.cpp process + Ollama wrapper
SAFE_RAM_FRACTION:    float = 0.85  # above this → marginal
SWAP_RISK_FRACTION:   float = 0.92  # above this → swap risk (macOS compressor)

_F16_BYTES = 2
_Q8_BYTES  = 1

# Architecture prefixes used by Ollama's modelinfo keys.
# Checked in order — first match wins.  More specific prefixes (gemma4, gemma3)
# must come before shorter ones (gemma) to avoid false early matches.
_ARCH_PREFIXES = [
    "llama", "qwen3", "qwen2", "phi4", "phi3",
    "gemma4", "gemma3", "gemma2", "gemma",
    "mistral", "falcon", "mpt", "stablelm", "deepseek2",
    "internlm2", "cohere", "command_r", "starcoder",
]

# Bytes-per-parameter ratio for each GGUF quantization.
# Used only for quant-downgrade size estimation when we lack exact GGUF sizes.
_QUANT_BPP: dict[str, float] = {
    "Q2_K":   0.329,
    "Q3_K_M": 0.411,
    "Q4_K_S": 0.546,
    "Q4_K_M": 0.606,
    "Q5_K_M": 0.710,
    "Q6_K":   0.821,
    "Q8_0":   1.063,
    "F16":    2.000,
}

# Ordered lightest → heaviest (preference order for downgrade suggestions)
_QUANT_ORDER = ["Q2_K", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]

# Minimum quality we'll suggest (don't go below Q4_K_S)
_MIN_ACCEPTABLE_QUANT = "Q4_K_S"


# ---------------------------------------------------------------------------
# Architecture extraction
# ---------------------------------------------------------------------------

@dataclass
class ArchInfo:
    """Model architecture parameters — derived from Ollama modelinfo or estimated."""
    n_layers:       int
    n_kv_heads:     int    # GQA KV heads (can be << n_heads)
    n_heads:        int    # total attention heads
    embedding_size: int    # hidden/residual stream dimension
    source:         str    # "modelinfo" | "estimate"

    @property
    def head_dim(self) -> int:
        return max(1, self.embedding_size // max(1, self.n_heads))

    def kv_bytes_per_token(self, kv_precision: str = "F16") -> int:
        """KV cache bytes consumed per token, for one full sequence."""
        el = _F16_BYTES if kv_precision == "F16" else _Q8_BYTES
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * el

    def kv_cache_gb(self, context_len: int, kv_precision: str = "F16") -> float:
        return self.kv_bytes_per_token(kv_precision) * context_len / 1024**3

    def kv_mb_per_1k_tokens(self, kv_precision: str = "F16") -> float:
        return self.kv_bytes_per_token(kv_precision) * 1000 / 1024**2


def extract_arch_from_modelinfo(modelinfo: dict) -> Optional[ArchInfo]:
    """
    Extract architecture parameters from Ollama's /api/show modelinfo dict.

    Handles all common model families: llama, qwen, phi, gemma, mistral, etc.
    Returns None if essential fields are missing.
    """
    if not modelinfo:
        return None

    # Detect which architecture prefix these keys use
    arch_prefix = None
    for key in modelinfo:
        for prefix in _ARCH_PREFIXES:
            if key.startswith(prefix + "."):
                arch_prefix = prefix
                break
        if arch_prefix:
            break

    if not arch_prefix:
        return None

    def _get(*suffixes: str) -> Optional[int]:
        """Try multiple key suffixes, return first non-None int value found."""
        for s in suffixes:
            v = modelinfo.get(f"{arch_prefix}.{s}")
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return None

    n_layers = _get("block_count", "num_hidden_layers", "n_layer", "num_layers")
    n_heads  = _get(
        "attention.head_count", "num_attention_heads", "n_head",
        "attention.num_heads",
    )
    n_kv     = _get(
        "attention.head_count_kv", "num_key_value_heads", "n_kv_head",
        "attention.num_kv_heads",
    )
    embed    = _get(
        "embedding_length", "hidden_size", "d_model", "n_embd",
        "model_dim",
    )

    if not all([n_layers, n_heads, embed]):
        return None

    # MHA when no separate KV head count (n_kv == n_heads)
    if n_kv is None:
        n_kv = n_heads

    return ArchInfo(
        n_layers=n_layers,
        n_kv_heads=n_kv,
        n_heads=n_heads,
        embedding_size=embed,
        source="modelinfo",
    )


def estimate_arch_from_params(params_b: float) -> ArchInfo:
    """
    Fallback architecture estimate based on parameter count.

    Uses typical GQA configurations for mainstream model sizes.
    Accuracy is within ~15% for KV cache estimates.
    """
    # (max_params_b, n_layers, n_heads, n_kv_heads, embed_dim)
    # Based on published configs for llama/qwen/mistral families
    configs = [
        (0.6,   22, 16,  8,  1024),
        (1.7,   24, 16,  8,  2048),
        (4.0,   28, 24,  8,  3072),
        (7.5,   32, 32,  8,  4096),
        (10.0,  36, 32,  8,  4096),
        (15.0,  48, 40,  8,  5120),
        (25.0,  52, 48,  8,  6144),
        (35.0,  64, 64,  8,  8192),
        (75.0,  80, 64,  8,  8192),
        (float("inf"), 96, 80, 8, 8192),
    ]
    for max_p, layers, n_heads, n_kv, embed in configs:
        if params_b <= max_p:
            return ArchInfo(
                n_layers=layers, n_kv_heads=n_kv, n_heads=n_heads,
                embedding_size=embed, source="estimate",
            )
    return ArchInfo(n_layers=96, n_kv_heads=8, n_heads=80, embedding_size=8192, source="estimate")


# ---------------------------------------------------------------------------
# Fit classification
# ---------------------------------------------------------------------------

class FitClass(Enum):
    SAFE      = "safe"       # <85% available RAM — comfortable headroom
    MARGINAL  = "marginal"   # 85–92% available — will work, use Q8 KV
    SWAP_RISK = "swap_risk"  # 92–100% available — autotune compresses context to help
    TIGHT     = "tight"      # >available but fits in total RAM — close other apps
    OOM       = "oom"        # >95% total RAM — physically won't fit


@dataclass
class ContextBudget:
    """Context token limit for a given KV precision without triggering swap."""
    kv_precision:        str    # "F16" or "Q8_0"
    max_safe_tokens:     int    # stays under SAFE_RAM_FRACTION
    max_marginal_tokens: int    # stays under SWAP_RISK_FRACTION
    kv_mb_per_1k:        float  # MB per 1000 context tokens


@dataclass
class FitReport:
    """Complete pre-flight analysis result."""
    model_name:   str
    params_b:     Optional[float]
    quant:        str
    size_gb:      float
    available_gb: float
    total_ram_gb: float

    arch:         Optional[ArchInfo]

    # Memory breakdown at reference context (8192 tokens)
    weights_gb:   float
    kv_f16_gb:    float    # KV at 8k context, F16
    kv_q8_gb:     float    # KV at 8k context, Q8
    overhead_gb:  float
    total_est_gb: float    # weights + kv_q8 + overhead (conservative)
    ram_util_pct: float    # total_est / available * 100

    fit_class:              FitClass
    recommended_profile:    str
    recommended_kv:         str    # "F16" or "Q8_0"
    safe_max_context:       int    # the context cap to actually use

    budget_f16:   ContextBudget
    budget_q8:    ContextBudget

    # Quant downgrade
    quant_too_heavy:       bool
    suggested_quant:       Optional[str]
    suggested_quant_gb:    Optional[float]
    suggested_headroom_gb: Optional[float]

    warning: Optional[str] = None
    fatal:   bool = False


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class ModelSelector:
    """
    Assesses model fit and selects safe context length + KV precision.

    Usage:
        hw = profile_hardware()
        sel = ModelSelector(hw.effective_memory_gb, hw.memory.total_gb)
        report = sel.assess("qwen3:8b", size_gb=5.2, params_b=8.0,
                            quant="Q4_K_M", modelinfo={...})
        if report.fatal:
            print(report.warning)
        else:
            safe_ctx = report.safe_max_context
    """

    def __init__(self, available_gb: float, total_ram_gb: float) -> None:
        self.available_gb = max(available_gb, 0.5)
        self.total_ram_gb = total_ram_gb

    def assess(
        self,
        model_name: str,
        size_gb: float,
        params_b: Optional[float],
        quant: str,
        modelinfo: Optional[dict] = None,
    ) -> FitReport:
        """
        Full pre-flight analysis for a model on this hardware.

        Parameters
        ----------
        model_name : Ollama tag (e.g. "qwen2.5-coder:14b")
        size_gb    : actual GGUF file size from Ollama /api/tags
        params_b   : total parameter count in billions (None = unknown)
        quant      : quantization string from Ollama details
        modelinfo  : raw modelinfo dict from /api/show (optional but improves accuracy)
        """
        # ── 1. Architecture ─────────────────────────────────────────────
        arch: Optional[ArchInfo] = None
        if modelinfo:
            arch = extract_arch_from_modelinfo(modelinfo)
        if arch is None and params_b:
            arch = estimate_arch_from_params(params_b)

        # ── 2. Memory breakdown ──────────────────────────────────────────
        weights_gb  = size_gb
        overhead_gb = RUNTIME_OVERHEAD_GB
        base_used   = weights_gb + overhead_gb

        # Reference KV at 8k context — used for fit classification
        ref_ctx = 8192
        kv_f16  = arch.kv_cache_gb(ref_ctx, "F16")  if arch else size_gb * 0.20
        kv_q8   = arch.kv_cache_gb(ref_ctx, "Q8_0") if arch else size_gb * 0.10

        # Conservative total: weights + Q8 KV + overhead
        total_est = base_used + kv_q8
        util      = total_est / self.available_gb

        # ── 3. Fit classification ────────────────────────────────────────
        # OOM = won't fit even in total installed RAM (hard limit).
        # TIGHT = exceeds currently free RAM but fits in total RAM; the user
        #   can run it by closing other apps, or macOS memory compression will
        #   handle it — this is NOT the same as being unable to run at all.
        if total_est > self.total_ram_gb * 0.95:
            fit_class = FitClass.OOM
        elif total_est > self.available_gb:
            fit_class = FitClass.TIGHT
        elif util > SWAP_RISK_FRACTION:
            fit_class = FitClass.SWAP_RISK
        elif util > SAFE_RAM_FRACTION:
            fit_class = FitClass.MARGINAL
        else:
            fit_class = FitClass.SAFE

        # ── 4. Context budgets ───────────────────────────────────────────
        budget_f16 = self._context_budget(arch, base_used, "F16")
        budget_q8  = self._context_budget(arch, base_used, "Q8_0")

        # ── 5. KV precision recommendation ──────────────────────────────
        if fit_class in (FitClass.MARGINAL, FitClass.SWAP_RISK):
            rec_kv = "Q8_0"   # halves KV memory
        else:
            rec_kv = "F16"

        # ── 6. Profile + safe context ────────────────────────────────────
        # Use the budget for recommended KV precision
        active_budget = budget_q8 if rec_kv == "Q8_0" else budget_f16
        safe_ctx      = active_budget.max_safe_tokens

        if fit_class == FitClass.OOM:
            rec_profile = "—"
            safe_ctx    = 0
        elif fit_class == FitClass.TIGHT:
            # Budget against total_ram with conservative headroom for OS + apps.
            # Using 75% of total_ram leaves room for the OS and a few background
            # processes after the user closes heavy apps.
            tight_budget_gb = self.total_ram_gb * 0.75 - base_used
            if arch and tight_budget_gb > 0:
                bpt = arch.kv_bytes_per_token("Q8_0")
                tight_tokens = int(tight_budget_gb * 1024**3 / max(bpt, 1))
                safe_ctx = _round_context(min(tight_tokens, 8192))
            else:
                safe_ctx = 2048
            safe_ctx    = max(safe_ctx, 512)
            rec_profile = "fast"
        elif safe_ctx < 2048:
            rec_profile = "fast"
            safe_ctx    = max(safe_ctx, 512)
        elif safe_ctx < 8192:
            rec_profile = "fast"    # fast uses 2048
        elif safe_ctx < 32768:
            rec_profile = "balanced"
        else:
            rec_profile = "quality"

        # ── 7. Quantization downgrade suggestion ─────────────────────────
        norm_quant = _normalize_quant(quant)
        too_heavy  = fit_class in (FitClass.TIGHT, FitClass.SWAP_RISK, FitClass.OOM)
        sug_quant = sug_gb = sug_headroom = None

        if too_heavy and params_b:
            f16_size = _back_to_f16(size_gb, norm_quant)
            cur_idx  = _quant_idx(norm_quant)
            min_idx  = _quant_idx(_MIN_ACCEPTABLE_QUANT)

            # Iterate heaviest → lightest so we suggest the best quality quant
            # that still fits safely in RAM (not the smallest possible).
            for q in reversed(_QUANT_ORDER[min_idx:cur_idx]):
                est_w    = f16_size * (_QUANT_BPP[q] / _QUANT_BPP["F16"])
                est_base = est_w + overhead_gb
                est_kv   = arch.kv_cache_gb(ref_ctx, "Q8_0") if arch else est_w * 0.10
                est_tot  = est_base + est_kv
                if est_tot / self.available_gb < SAFE_RAM_FRACTION:
                    sug_quant    = q
                    sug_gb       = round(est_w, 2)
                    sug_headroom = round(self.available_gb - est_tot, 2)
                    break

        # ── 8. Warning ───────────────────────────────────────────────────
        warning = None
        fatal   = False

        if fit_class == FitClass.OOM:
            fatal = True
            warning = (
                f"Model needs ~{total_est:.1f} GB "
                f"(weights {size_gb:.1f} GB + KV {kv_q8:.2f} GB + overhead) "
                f"but your machine only has {self.total_ram_gb:.0f} GB total. "
                f"This model is too large to run on this hardware."
            )
            if sug_quant:
                warning += (
                    f" Try a smaller quantization: "
                    f"autotune pull {model_name.split(':')[0]}:{sug_quant.lower()} "
                    f"(~{sug_gb:.1f} GB)"
                )

        elif fit_class == FitClass.TIGHT:
            warning = (
                f"Model needs ~{total_est:.1f} GB but only {self.available_gb:.1f} GB "
                f"is free right now (you have {self.total_ram_gb:.0f} GB total). "
                f"Close other apps to free memory, then this will run fine."
            )
            if sug_quant:
                warning += (
                    f" Or pull a lighter quant for comfortable headroom: "
                    f"autotune pull {model_name.split(':')[0]}:{sug_quant.lower()} "
                    f"(~{sug_gb:.1f} GB)"
                )

        elif fit_class == FitClass.SWAP_RISK:
            warning = (
                f"Model uses {util*100:.0f}% of free RAM at 8k context — autotune "
                f"will cap context to {safe_ctx:,} tokens and use Q8 KV to stay "
                f"below the swap threshold."
            )
            if sug_quant:
                warning += f" For more breathing room: pull {sug_quant} (~{sug_gb:.1f} GB)."

        elif fit_class == FitClass.MARGINAL:
            warning = (
                f"Model is a snug fit ({util*100:.0f}% of free RAM at 8k context). "
                f"autotune will use Q8 KV cache and keep context under {safe_ctx:,} tokens."
            )

        return FitReport(
            model_name=model_name,
            params_b=params_b,
            quant=quant,
            size_gb=size_gb,
            available_gb=self.available_gb,
            total_ram_gb=self.total_ram_gb,
            arch=arch,
            weights_gb=weights_gb,
            kv_f16_gb=round(kv_f16, 3),
            kv_q8_gb=round(kv_q8, 3),
            overhead_gb=overhead_gb,
            total_est_gb=round(total_est, 3),
            ram_util_pct=round(util * 100, 1),
            fit_class=fit_class,
            recommended_profile=rec_profile,
            recommended_kv=rec_kv,
            safe_max_context=safe_ctx,
            budget_f16=budget_f16,
            budget_q8=budget_q8,
            quant_too_heavy=too_heavy,
            suggested_quant=sug_quant,
            suggested_quant_gb=sug_gb,
            suggested_headroom_gb=sug_headroom,
            warning=warning,
            fatal=fatal,
        )

    def _context_budget(
        self,
        arch: Optional[ArchInfo],
        base_used_gb: float,
        kv_precision: str,
    ) -> ContextBudget:
        """
        Compute safe/marginal context limits for a given KV precision.
        """
        safe_budget_gb     = self.available_gb * SAFE_RAM_FRACTION - base_used_gb
        marginal_budget_gb = self.available_gb * SWAP_RISK_FRACTION - base_used_gb

        if arch:
            bpt = arch.kv_bytes_per_token(kv_precision)
            mb_per_1k = arch.kv_mb_per_1k_tokens(kv_precision)
        else:
            # Fallback: ~128 KB/token for F16 at 7B scale, halved for Q8
            bpt = 131_072 if kv_precision == "F16" else 65_536
            mb_per_1k = bpt * 1000 / 1024**2

        def _tokens(budget_gb: float) -> int:
            if budget_gb <= 0:
                return 0
            return _round_context(int(budget_gb * 1024**3 / bpt))

        return ContextBudget(
            kv_precision=kv_precision,
            max_safe_tokens=_tokens(safe_budget_gb),
            max_marginal_tokens=_tokens(marginal_budget_gb),
            kv_mb_per_1k=round(mb_per_1k, 1),
        )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _round_context(tokens: int) -> int:
    """Round down to the nearest clean context step (powers of 2, 4k steps)."""
    if tokens <= 0:
        return 0
    steps = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    for s in reversed(steps):
        if tokens >= s:
            return s
    return max(512, (tokens // 512) * 512)


def _normalize_quant(quant_str: str) -> str:
    """Map Ollama quant strings to our canonical keys."""
    q = (quant_str or "").upper().strip().replace(" ", "_")
    aliases = {
        "Q4_K":    "Q4_K_M",
        "Q5_K":    "Q5_K_M",
        "Q3_K":    "Q3_K_M",
        "Q4KM":    "Q4_K_M",
        "Q5KM":    "Q5_K_M",
        "IQ4_NL":  "Q4_K_S",
        "IQ4_XS":  "Q4_K_S",
        "IQ3_XS":  "Q3_K_M",
        "IQ2_XXS": "Q2_K",
        "Q8":      "Q8_0",
        "UNKNOWN": "Q4_K_M",   # assume a reasonable default
    }
    return aliases.get(q, q if q in _QUANT_BPP else "Q4_K_M")


def _quant_idx(quant: str) -> int:
    try:
        return _QUANT_ORDER.index(quant)
    except ValueError:
        return len(_QUANT_ORDER) // 2


def _back_to_f16(size_gb: float, quant: str) -> float:
    """Estimate F16 model size from a known quantized size."""
    bpp    = _QUANT_BPP.get(quant, 0.606)
    f16bpp = _QUANT_BPP["F16"]
    return size_gb * (f16bpp / max(bpp, 0.1))
