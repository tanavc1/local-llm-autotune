"""Model profile registry with quantization specs for common OSS LLMs."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Quantization specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantizationSpec:
    name: str
    bits_per_weight: float   # effective bits after quantization
    bytes_per_param: float   # storage bytes per parameter
    quality_score: float     # 0–1 relative quality vs F16 baseline
    speed_multiplier: float  # relative throughput vs F16 (higher = faster)
    description: str


# Ordered from most-compressed → least-compressed
QUANTIZATIONS: dict[str, QuantizationSpec] = {
    "Q2_K": QuantizationSpec(
        name="Q2_K",
        bits_per_weight=2.63,
        bytes_per_param=0.329,
        quality_score=0.50,
        speed_multiplier=3.2,
        description="2-bit K-quant – very small, noticeable quality loss",
    ),
    "Q4_K_S": QuantizationSpec(
        name="Q4_K_S",
        bits_per_weight=4.37,
        bytes_per_param=0.546,
        quality_score=0.78,
        speed_multiplier=2.6,
        description="4-bit K-quant small – good balance for constrained RAM",
    ),
    "Q4_K_M": QuantizationSpec(
        name="Q4_K_M",
        bits_per_weight=4.85,
        bytes_per_param=0.606,
        quality_score=0.84,
        speed_multiplier=2.4,
        description="4-bit K-quant medium – recommended sweet spot",
    ),
    "Q5_K_M": QuantizationSpec(
        name="Q5_K_M",
        bits_per_weight=5.68,
        bytes_per_param=0.710,
        quality_score=0.91,
        speed_multiplier=2.0,
        description="5-bit K-quant medium – near-lossless with good speed",
    ),
    "Q6_K": QuantizationSpec(
        name="Q6_K",
        bits_per_weight=6.57,
        bytes_per_param=0.821,
        quality_score=0.95,
        speed_multiplier=1.7,
        description="6-bit K-quant – excellent quality, moderate size",
    ),
    "Q8_0": QuantizationSpec(
        name="Q8_0",
        bits_per_weight=8.50,
        bytes_per_param=1.063,
        quality_score=0.99,
        speed_multiplier=1.3,
        description="8-bit – nearly lossless, large footprint",
    ),
    "F16": QuantizationSpec(
        name="F16",
        bits_per_weight=16.0,
        bytes_per_param=2.000,
        quality_score=1.00,
        speed_multiplier=1.0,
        description="Full 16-bit half-precision – reference quality",
    ),
}


# ---------------------------------------------------------------------------
# Model profile
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    id: str
    name: str
    family: str
    parameters_b: float          # total parameter count in billions
    n_layers: int
    hidden_size: int             # embedding / residual stream dimension
    n_heads: int                 # attention heads
    n_kv_heads: int              # KV heads (< n_heads when GQA is used)
    vocab_size: int
    context_window: int          # maximum trained context length (tokens)
    quantization_options: list[str]
    description: str
    hf_repo: str = ""            # Hugging Face repo for reference

    # Real-world quality benchmarks (None = unknown, populated for registry models)
    # Sources: Open LLM Leaderboard, official model papers (5-shot MMLU)
    bench_mmlu: float | None = None        # 5-shot MMLU accuracy (0–1.0)
    bench_humaneval: float | None = None   # HumanEval pass@1 (0–1.0)
    bench_gsm8k: float | None = None       # GSM8K math reasoning (0–1.0)

    # ------------------------------------------------------------------ #
    # Derived helpers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    def weight_bytes(self, quant: str) -> float:
        """Estimated weight storage in bytes for a given quantization."""
        spec = QUANTIZATIONS[quant]
        params = self.parameters_b * 1e9
        return params * spec.bytes_per_param

    def weight_gb(self, quant: str) -> float:
        return self.weight_bytes(quant) / 1024**3

    def kv_cache_bytes(self, context_len: int, quant: str = "F16") -> float:
        """
        KV-cache bytes for *one* sequence of `context_len` tokens.

        Formula (standard attention):
          2 * n_layers * n_kv_heads * head_dim * context_len * element_bytes
        """
        # KV cache is typically stored in F16 regardless of weight quant,
        # unless the runtime uses quantised KV (rare). Default to F16.
        element_bytes = 2  # float16
        return (
            2
            * self.n_layers
            * self.n_kv_heads
            * self.head_dim
            * context_len
            * element_bytes
        )

    def kv_cache_gb(self, context_len: int) -> float:
        return self.kv_cache_bytes(context_len) / 1024**3


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelProfile] = {
    m.id: m
    for m in [
        # ── Tiny / edge (< 2 GB)  ──────────────────────────────────────
        ModelProfile(
            id="llama-3.2-1b",
            name="Llama 3.2 1B",
            family="llama",
            parameters_b=1.24,
            n_layers=16,
            hidden_size=2048,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=128256,
            context_window=131072,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
            description="Ultra-light Llama 3.2 model, ideal for CPU or low-VRAM devices",
            hf_repo="meta-llama/Llama-3.2-1B",
            bench_mmlu=0.328, bench_humaneval=0.289, bench_gsm8k=0.444,
        ),
        ModelProfile(
            id="qwen3-0.6b",
            name="Qwen3 0.6B",
            family="qwen3",
            parameters_b=0.60,
            n_layers=28,
            hidden_size=1024,
            n_heads=16,
            n_kv_heads=8,
            vocab_size=151936,
            context_window=32768,
            quantization_options=["Q4_K_M", "Q8_0", "F16"],
            description="Qwen3 ultra-tiny — hybrid thinking mode, on-device edge tasks",
            hf_repo="Qwen/Qwen3-0.6B",
            bench_mmlu=0.450,
        ),
        ModelProfile(
            id="qwen3-1.7b",
            name="Qwen3 1.7B",
            family="qwen3",
            parameters_b=1.70,
            n_layers=28,
            hidden_size=2048,
            n_heads=16,
            n_kv_heads=8,
            vocab_size=151936,
            context_window=32768,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
            description="Qwen3 1.7B — strong reasoning for its size, hybrid thinking mode",
            hf_repo="Qwen/Qwen3-1.7B",
            bench_mmlu=0.590,
        ),
        # ── Small (2–5 GB) ─────────────────────────────────────────────
        ModelProfile(
            id="llama-3.2-3b",
            name="Llama 3.2 3B",
            family="llama",
            parameters_b=3.21,
            n_layers=28,
            hidden_size=3072,
            n_heads=24,
            n_kv_heads=8,
            vocab_size=128256,
            context_window=131072,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
            description="Compact Llama 3.2 with solid performance on 4–8 GB systems",
            hf_repo="meta-llama/Llama-3.2-3B",
            bench_mmlu=0.586, bench_humaneval=0.524, bench_gsm8k=0.777,
        ),
        ModelProfile(
            id="qwen3-4b",
            name="Qwen3 4B",
            family="qwen3",
            parameters_b=4.00,
            n_layers=36,
            hidden_size=2560,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=151936,
            context_window=32768,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
            description="Qwen3 4B — hybrid thinking, punches above its weight",
            hf_repo="Qwen/Qwen3-4B",
            bench_mmlu=0.660, bench_humaneval=0.520,
        ),
        ModelProfile(
            id="gemma4-e2b",
            name="Gemma 4 E2B",
            family="gemma4",
            parameters_b=2.00,
            n_layers=26,
            hidden_size=2304,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=262144,
            context_window=131072,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q8_0"],
            description="Google Gemma 4 2B — compact, fast, multimodal-capable",
            hf_repo="google/gemma-4-2b-it",
            bench_mmlu=0.690,
        ),
        # ── Medium (5–12 GB) ───────────────────────────────────────────
        ModelProfile(
            id="qwen3-8b",
            name="Qwen3 8B",
            family="qwen3",
            parameters_b=8.00,
            n_layers=36,
            hidden_size=4096,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=151936,
            context_window=32768,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
            description="Qwen3 8B — near-frontier quality, hybrid thinking, best 8B available",
            hf_repo="Qwen/Qwen3-8B",
            bench_mmlu=0.740, bench_humaneval=0.680,
        ),
        ModelProfile(
            id="mistral-7b-v0.3",
            name="Mistral 7B v0.3",
            family="mistral",
            parameters_b=7.25,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=32768,
            context_window=32768,
            quantization_options=["Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
            description="Fast, capable 7B model; established community workhorse",
            hf_repo="mistralai/Mistral-7B-v0.3",
            bench_mmlu=0.642, bench_humaneval=0.270, bench_gsm8k=0.521,
        ),
        ModelProfile(
            id="llama-3.1-8b",
            name="Llama 3.1 8B",
            family="llama",
            parameters_b=8.03,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=128256,
            context_window=131072,
            quantization_options=["Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
            description="Meta Llama 3.1 8B with 128k context",
            hf_repo="meta-llama/Meta-Llama-3.1-8B",
            bench_mmlu=0.667, bench_humaneval=0.726, bench_gsm8k=0.845,
        ),
        ModelProfile(
            id="gemma4-e4b",
            name="Gemma 4 E4B",
            family="gemma4",
            parameters_b=4.00,
            n_layers=34,
            hidden_size=3072,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=262144,
            context_window=131072,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q8_0"],
            description="Google Gemma 4 4B — best compact model, multimodal, 128k context",
            hf_repo="google/gemma-4-4b-it",
            bench_mmlu=0.740, bench_humaneval=0.680,
        ),
        ModelProfile(
            id="gemma-2-9b",
            name="Gemma 2 9B",
            family="gemma",
            parameters_b=9.24,
            n_layers=42,
            hidden_size=3584,
            n_heads=16,
            n_kv_heads=8,
            vocab_size=256000,
            context_window=8192,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
            description="Google Gemma 2 9B – strong reasoning for its size",
            hf_repo="google/gemma-2-9b",
            bench_mmlu=0.723, bench_humaneval=0.402, bench_gsm8k=0.686,
        ),
        # ── Medium-large (12–20 GB) ─────────────────────────────────────
        ModelProfile(
            id="qwen3-14b",
            name="Qwen3 14B",
            family="qwen3",
            parameters_b=14.00,
            n_layers=40,
            hidden_size=5120,
            n_heads=40,
            n_kv_heads=8,
            vocab_size=151936,
            context_window=32768,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
            description="Qwen3 14B — excellent reasoning and instruction following, 16GB+ RAM",
            hf_repo="Qwen/Qwen3-14B",
            bench_mmlu=0.790, bench_humaneval=0.780,
        ),
        ModelProfile(
            id="qwen2.5-14b",
            name="Qwen 2.5 14B",
            family="qwen",
            parameters_b=14.77,
            n_layers=48,
            hidden_size=5120,
            n_heads=40,
            n_kv_heads=8,
            vocab_size=152064,
            context_window=131072,
            quantization_options=["Q2_K", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
            description="Alibaba Qwen 2.5 14B – excellent multilingual + code quality",
            hf_repo="Qwen/Qwen2.5-14B",
            bench_mmlu=0.798, bench_humaneval=0.769, bench_gsm8k=0.856,
        ),
        ModelProfile(
            id="gemma4-26b",
            name="Gemma 4 26B",
            family="gemma4",
            parameters_b=26.0,
            n_layers=62,
            hidden_size=4096,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=262144,
            context_window=262144,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q8_0"],
            description="Google Gemma 4 26B MoE (4B active) — 256k context, 32GB+ RAM",
            hf_repo="google/gemma-4-27b-it",
            bench_mmlu=0.852, bench_humaneval=0.890,
        ),
        ModelProfile(
            id="mistral-nemo-12b",
            name="Mistral NeMo 12B",
            family="mistral",
            parameters_b=12.25,
            n_layers=40,
            hidden_size=5120,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=131072,
            context_window=131072,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
            description="12B Mistral model with 128k context and large vocab",
            hf_repo="mistralai/Mistral-Nemo-Base-2407",
            bench_mmlu=0.680, bench_humaneval=0.325, bench_gsm8k=0.602,
        ),
        # ── Large (32–80 GB) ───────────────────────────────────────────
        ModelProfile(
            id="qwen3-32b",
            name="Qwen3 32B",
            family="qwen3",
            parameters_b=32.0,
            n_layers=64,
            hidden_size=5120,
            n_heads=64,
            n_kv_heads=8,
            vocab_size=151936,
            context_window=32768,
            quantization_options=["Q2_K", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K"],
            description="Qwen3 32B dense — top open model in its class, 32GB+ RAM",
            hf_repo="Qwen/Qwen3-32B",
            bench_mmlu=0.840, bench_humaneval=0.850,
        ),
        ModelProfile(
            id="llama-3.1-70b",
            name="Llama 3.1 70B",
            family="llama",
            parameters_b=70.55,
            n_layers=80,
            hidden_size=8192,
            n_heads=64,
            n_kv_heads=8,
            vocab_size=128256,
            context_window=131072,
            quantization_options=["Q2_K", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K"],
            description="Meta's flagship 70B – near-GPT-4 quality when quantised to Q4",
            hf_repo="meta-llama/Meta-Llama-3.1-70B",
            bench_mmlu=0.860, bench_humaneval=0.805, bench_gsm8k=0.951,
        ),
        ModelProfile(
            id="qwen2.5-72b",
            name="Qwen 2.5 72B",
            family="qwen",
            parameters_b=72.71,
            n_layers=80,
            hidden_size=8192,
            n_heads=64,
            n_kv_heads=8,
            vocab_size=152064,
            context_window=131072,
            quantization_options=["Q2_K", "Q4_K_S", "Q4_K_M", "Q5_K_M"],
            description="Alibaba's 72B model with top-tier multilingual and coding scores",
            hf_repo="Qwen/Qwen2.5-72B",
            bench_mmlu=0.865, bench_humaneval=0.861, bench_gsm8k=0.916,
        ),
        # ── MoE / Reasoning specialists ────────────────────────────────
        ModelProfile(
            id="qwen3-30b-a3b",
            name="Qwen3 30B-A3B",
            family="qwen3",
            parameters_b=30.5,
            n_layers=48,
            hidden_size=4096,
            n_heads=32,
            n_kv_heads=8,
            vocab_size=151936,
            context_window=131072,
            quantization_options=["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
            description="Qwen3 30B MoE (3.3B active) — flagship-level quality at 7B inference cost",
            hf_repo="Qwen/Qwen3-30B-A3B",
            bench_mmlu=0.830, bench_humaneval=0.840,
        ),
        ModelProfile(
            id="llama4-scout",
            name="Llama 4 Scout",
            family="llama4",
            parameters_b=109.0,
            n_layers=48,
            hidden_size=5120,
            n_heads=40,
            n_kv_heads=8,
            vocab_size=202048,
            context_window=10485760,  # 10M tokens
            quantization_options=["Q2_K", "Q4_K_M", "Q5_K_M"],
            description="Meta Llama 4 Scout 109B MoE (17B active) — multimodal, 10M context",
            hf_repo="meta-llama/Llama-4-Scout-17B-16E",
            bench_mmlu=0.760, bench_humaneval=0.680,
        ),
    ]
}


def get_model(model_id: str) -> ModelProfile:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_id!r}. Run `autotune models` for the list.")
    return MODEL_REGISTRY[model_id]


def list_models() -> list[ModelProfile]:
    return sorted(MODEL_REGISTRY.values(), key=lambda m: m.parameters_b)
