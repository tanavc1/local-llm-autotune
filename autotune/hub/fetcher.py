"""
HuggingFace Hub fetcher.

Pulls config.json + model card metadata for any public HF model and maps
the raw fields to the hardware-relevant schema used by the local DB.

No HF token needed for public models.  Set HF_TOKEN env var for gated ones.
"""

from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.request
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

HF_API = "https://huggingface.co"
HF_API_ENDPOINT = "https://huggingface.co/api"

HF_FETCH_ERROR = "HF_FETCH_ERROR"

# Bytes-per-parameter for each common GGUF quant format
_BYTES_PER_PARAM: dict[str, float] = {
    "F16":    2.000,
    "Q8_0":   1.063,
    "Q6_K":   0.821,
    "Q5_K_M": 0.710,
    "Q4_K_M": 0.606,
    "Q4_K_S": 0.546,
    "Q3_K_M": 0.419,
    "Q2_K":   0.329,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HFModelSpec:
    # Identity
    id: str
    name: str
    organization: str = ""
    family: str = ""
    license: str = ""
    description: str = ""
    hf_url: str = ""
    gguf_url: str = ""
    paper_url: str = ""
    release_date: str = ""

    # Parameters
    total_params_b: Optional[float] = None
    active_params_b: Optional[float] = None
    is_moe: bool = False
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None

    # Architecture
    arch_type: str = "decoder-only"
    n_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    vocab_size: Optional[int] = None
    max_context_window: Optional[int] = None
    rope_theta: Optional[float] = None
    positional_encoding: str = "rope"
    activation: str = ""
    normalization: str = "rms_norm"
    attention_type: str = ""          # mha / gqa / mqa / mla
    sliding_window_size: Optional[int] = None
    sliding_window_pattern: Optional[str] = None
    kv_latent_dim: Optional[int] = None
    logit_softcapping: Optional[float] = None
    attn_logit_softcapping: Optional[float] = None
    tie_word_embeddings: bool = False
    num_shared_experts: Optional[int] = None

    # Memory estimates (GB, weights only)
    mem_f16_gb: Optional[float] = None
    mem_q8_0_gb: Optional[float] = None
    mem_q6_k_gb: Optional[float] = None
    mem_q5_k_m_gb: Optional[float] = None
    mem_q4_k_m_gb: Optional[float] = None
    mem_q4_k_s_gb: Optional[float] = None
    mem_q3_k_m_gb: Optional[float] = None
    mem_q2_k_gb: Optional[float] = None

    # Quantization
    available_quants: list[str] = field(default_factory=list)
    recommended_quant: str = ""
    supports_awq: bool = False
    supports_gptq: bool = False
    supports_exl2: bool = False
    quant_notes: str = ""

    # Benchmarks
    bench_mmlu: Optional[float] = None
    bench_humaneval: Optional[float] = None
    bench_gsm8k: Optional[float] = None
    bench_hellaswag: Optional[float] = None
    bench_mt_bench: Optional[float] = None
    bench_source: str = ""

    use_cases: list[str] = field(default_factory=list)

    fetched_at: float = field(default_factory=time.time)
    raw_config: dict[str, Any] = field(default_factory=dict)

    def to_db_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    h = {"User-Agent": "autotune/0.1 (local LLM optimizer)"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_json(url: str, timeout: int = 15) -> Optional[dict[str, Any]]:
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def _get_text(url: str, timeout: int = 15) -> Optional[str]:
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode(errors="replace")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Config.json parser
# ---------------------------------------------------------------------------

def _parse_config(cfg: dict[str, Any], spec: HFModelSpec) -> None:
    """Extract all hardware-relevant fields from a raw config.json."""

    # Architecture type
    archs = cfg.get("architectures", [])
    if archs:
        a = archs[0].lower()
        if "forcausallm" in a or "decoder" in a:
            spec.arch_type = "decoder-only"
        elif "forconditionalgeneration" in a:
            spec.arch_type = "encoder-decoder"

    # Core dimensions
    spec.n_layers = (
        cfg.get("num_hidden_layers")
        or cfg.get("n_layer")
        or cfg.get("num_layers")
    )
    spec.hidden_size = (
        cfg.get("hidden_size")
        or cfg.get("d_model")
        or cfg.get("n_embd")
    )
    spec.n_heads = (
        cfg.get("num_attention_heads")
        or cfg.get("n_head")
        or cfg.get("num_heads")
    )
    spec.n_kv_heads = cfg.get("num_key_value_heads") or spec.n_heads
    spec.intermediate_size = (
        cfg.get("intermediate_size")
        or cfg.get("ffn_dim")
        or cfg.get("n_inner")
    )
    spec.vocab_size = cfg.get("vocab_size")

    # Head dim: explicit first, then derive
    spec.head_dim = cfg.get("head_dim")
    if spec.head_dim is None and spec.hidden_size and spec.n_heads:
        spec.head_dim = spec.hidden_size // spec.n_heads

    # Context window
    spec.max_context_window = (
        cfg.get("max_position_embeddings")
        or cfg.get("max_seq_len")
        or cfg.get("sliding_window")  # some models set this as max
        or cfg.get("seq_length")
    )

    # RoPE
    rope_cfg = cfg.get("rope_scaling") or {}
    spec.rope_theta = cfg.get("rope_theta") or cfg.get("rotary_emb_base")

    # Activation
    spec.activation = (
        cfg.get("hidden_act")
        or cfg.get("activation_function")
        or cfg.get("hidden_activation")
        or ""
    )

    # Normalization
    if cfg.get("rms_norm_eps") or cfg.get("norm_type", "").lower() == "rms_norm":
        spec.normalization = "rms_norm"
    elif cfg.get("layer_norm_eps"):
        spec.normalization = "layer_norm"

    # Attention type
    if spec.n_kv_heads and spec.n_heads:
        if spec.n_kv_heads == 1:
            spec.attention_type = "mqa"
        elif spec.n_kv_heads < spec.n_heads:
            spec.attention_type = "gqa"
        else:
            spec.attention_type = "mha"

    # MLA (DeepSeek)
    if cfg.get("kv_lora_rank"):
        spec.attention_type = "mla"
        spec.kv_latent_dim = cfg.get("kv_lora_rank")

    # Sliding window (Mistral-style)
    sw = cfg.get("sliding_window")
    if isinstance(sw, int) and sw > 0:
        spec.sliding_window_size = sw

    # Gemma-2 specifics
    if cfg.get("attn_logit_softcapping"):
        spec.attn_logit_softcapping = cfg["attn_logit_softcapping"]
    if cfg.get("final_logit_softcapping"):
        spec.logit_softcapping = cfg["final_logit_softcapping"]
    if cfg.get("query_pre_attn_scalar"):
        spec.sliding_window_pattern = cfg.get("sliding_window_pattern", "every_other")

    # MoE
    num_experts = (
        cfg.get("num_local_experts")
        or cfg.get("num_experts")
        or cfg.get("n_routed_experts")
    )
    experts_per_tok = (
        cfg.get("num_experts_per_tok")
        or cfg.get("top_k")
        or cfg.get("experts_per_token")
    )
    if num_experts:
        spec.is_moe = True
        spec.num_experts = num_experts
        spec.experts_per_token = experts_per_tok
        spec.num_shared_experts = cfg.get("n_shared_experts")

    # Tie embeddings
    spec.tie_word_embeddings = bool(cfg.get("tie_word_embeddings", False))


# ---------------------------------------------------------------------------
# Parameter count estimator (fallback when not on model card)
# ---------------------------------------------------------------------------

def _estimate_params(spec: HFModelSpec) -> Optional[float]:
    """Rough parameter count estimate from architecture dimensions (billions)."""
    if not all([spec.n_layers, spec.hidden_size, spec.n_heads, spec.intermediate_size, spec.vocab_size]):
        return None
    h = spec.hidden_size
    n = spec.n_layers
    ffn = spec.intermediate_size
    v = spec.vocab_size
    kv_h = spec.n_kv_heads or spec.n_heads
    hd = spec.head_dim or (h // spec.n_heads)

    # Attention: Q + K + V + O projections
    attn = n * (h * spec.n_heads * hd + 2 * h * kv_h * hd + spec.n_heads * hd * h)
    # FFN: gate + up + down (SwiGLU)
    ffn_p = n * (h * ffn * 3)
    # Embeddings + norms
    embed = v * h * 2 + n * h * 2 + h

    if spec.is_moe and spec.num_experts and spec.experts_per_token:
        # scale FFN by expert ratio
        ffn_p = ffn_p * spec.num_experts / spec.experts_per_token

    total = (attn + ffn_p + embed) / 1e9
    return round(total, 2)


# ---------------------------------------------------------------------------
# Memory estimates
# ---------------------------------------------------------------------------

def _compute_memory(spec: HFModelSpec) -> None:
    params = spec.active_params_b or spec.total_params_b
    if params is None:
        return
    for attr, quant in [
        ("mem_f16_gb",    "F16"),
        ("mem_q8_0_gb",   "Q8_0"),
        ("mem_q6_k_gb",   "Q6_K"),
        ("mem_q5_k_m_gb", "Q5_K_M"),
        ("mem_q4_k_m_gb", "Q4_K_M"),
        ("mem_q4_k_s_gb", "Q4_K_S"),
        ("mem_q3_k_m_gb", "Q3_K_M"),
        ("mem_q2_k_gb",   "Q2_K"),
    ]:
        setattr(spec, attr, round(params * _BYTES_PER_PARAM[quant], 2))


# ---------------------------------------------------------------------------
# Quant support inference
# ---------------------------------------------------------------------------

def _quant_defaults(spec: HFModelSpec) -> None:
    params = spec.active_params_b or 0
    quants = ["Q2_K", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
    if params < 14:
        quants.append("F16")
    spec.available_quants = quants

    # Recommended quant based on size
    if params < 4:
        spec.recommended_quant = "Q8_0"
    elif params < 14:
        spec.recommended_quant = "Q5_K_M"
    else:
        spec.recommended_quant = "Q4_K_M"

    spec.supports_awq = params < 100
    spec.supports_gptq = True
    spec.supports_exl2 = True


# ---------------------------------------------------------------------------
# GGUF URL finder
# ---------------------------------------------------------------------------

def _find_gguf_url(model_id: str) -> str:
    """Best-effort GGUF repo URL."""
    org, _, name = model_id.partition("/")
    candidates = [
        f"{HF_API}/{model_id}-GGUF",
        f"{HF_API}/bartowski/{name}-GGUF",
        f"{HF_API}/unsloth/{name}-GGUF",
        f"{HF_API}/lmstudio-community/{name}-GGUF",
    ]
    for url in candidates:
        req = urllib.request.Request(url, headers=_headers(), method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return url
        except Exception:
            pass
    # Return bartowski as fallback (most likely to have it)
    return f"{HF_API}/bartowski/{name}-GGUF"


# ---------------------------------------------------------------------------
# Model card metadata
# ---------------------------------------------------------------------------

def _parse_model_card_api(meta: dict[str, Any], spec: HFModelSpec) -> None:
    """Fill spec from the /api/models/{id} response."""
    spec.license = meta.get("cardData", {}).get("license", "") or ""
    spec.description = (meta.get("cardData", {}).get("model-index") or [{}])[0].get("name", "") if meta.get("cardData") else ""

    # Try to get param count from model metadata safetensors info
    safetensors = meta.get("safetensors", {})
    total = safetensors.get("total")
    if total:
        spec.total_params_b = round(total / 1e9, 3)
        spec.active_params_b = spec.total_params_b

    # Tags → family / use cases
    tags: list[str] = meta.get("tags", [])
    use_cases = []
    for tag in tags:
        if tag in ("code", "coding"):
            use_cases.append("code")
        elif tag in ("math", "mathematics"):
            use_cases.append("math")
        elif tag in ("chat", "conversational"):
            use_cases.append("chat")
        elif tag == "reasoning":
            use_cases.append("reasoning")
        elif tag in ("multilingual",):
            use_cases.append("multilingual")
    spec.use_cases = list(dict.fromkeys(["general"] + use_cases))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_model(model_id: str, cache_days: int = 7) -> HFModelSpec:
    """
    Fetch and parse a HuggingFace model's hardware-relevant specs.

    Parameters
    ----------
    model_id   : "org/model-name" e.g. "meta-llama/Meta-Llama-3.1-8B"
    cache_days : not used here (caching is handled by the DB layer)
    """
    spec = HFModelSpec(
        id=model_id,
        name=model_id.split("/")[-1],
        organization=model_id.split("/")[0] if "/" in model_id else "",
        hf_url=f"{HF_API}/{model_id}",
        fetched_at=time.time(),
    )

    # 1. Model card API
    meta = _get_json(f"{HF_API_ENDPOINT}/models/{model_id}")
    if meta:
        _parse_model_card_api(meta, spec)

    # 2. config.json
    cfg_url = f"{HF_API}/{model_id}/resolve/main/config.json"
    cfg = _get_json(cfg_url)
    if cfg:
        spec.raw_config = cfg
        _parse_config(cfg, spec)

    # 3. Estimate params if not found
    if spec.total_params_b is None:
        est = _estimate_params(spec)
        if est:
            spec.total_params_b = est
            spec.active_params_b = est

    # For MoE: active params = total * (experts_per_token / num_experts)
    if spec.is_moe and spec.num_experts and spec.experts_per_token and spec.total_params_b:
        # FFN represents ~2/3 of params; only that fraction scales with expert ratio
        ffn_fraction = 0.65
        non_ffn = spec.total_params_b * (1 - ffn_fraction)
        ffn_active = spec.total_params_b * ffn_fraction * (spec.experts_per_token / spec.num_experts)
        if spec.num_shared_experts:
            ffn_active += spec.total_params_b * ffn_fraction * (spec.num_shared_experts / spec.num_experts)
        spec.active_params_b = round(non_ffn + ffn_active, 2)

    # 4. Memory estimates
    _compute_memory(spec)

    # 5. Quant defaults
    _quant_defaults(spec)

    # 6. GGUF URL (best-effort, non-blocking)
    try:
        spec.gguf_url = _find_gguf_url(model_id)
    except Exception:
        spec.gguf_url = f"{HF_API}/bartowski/{spec.name}-GGUF"

    # 7. Family detection
    name_lower = spec.name.lower()
    for family in ("llama", "mistral", "mixtral", "gemma", "qwen", "phi",
                   "deepseek", "falcon", "command", "yi", "internlm",
                   "olmo", "granite", "smollm", "qwq"):
        if family in name_lower or family in spec.organization.lower():
            spec.family = family
            break

    return spec
