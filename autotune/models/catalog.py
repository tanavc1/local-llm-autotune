"""
Model Catalog — auto-refreshing directory of LLMs for local inference.

Stores richer data than the in-memory MODEL_REGISTRY:
  • Disk size (GB at Q4_K_M)
  • Runtime RAM estimate
  • Speed class (fast / medium / slow)
  • Tier (tiny / small / medium / large / xl / flagship)
  • Capability tags (coding, reasoning, multilingual, fast, moe, …)
  • Benchmark scores (MMLU, HumanEval, GSM8K, MATH)
  • Ollama tag, HF repo, MLX community repo
  • Human highlight ("Best 8B overall", "Top coding 32B")

The catalog lives at ~/.autotune/model_catalog.json and auto-refreshes
every update_interval_days (default 4). On first access it is seeded from
the bundled baseline below, which is compiled from Open LLM Leaderboard,
Ollama library downloads, and HuggingFace trending as of May 2026.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Storage path
# ---------------------------------------------------------------------------

CATALOG_PATH: Path = Path.home() / ".autotune" / "model_catalog.json"
_CATALOG_VERSION = "2"
DEFAULT_UPDATE_INTERVAL_DAYS = 4


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    id: str                         # primary identifier / ollama_tag
    name: str                       # display name
    family: str                     # model family
    tier: str                       # "tiny"|"small"|"medium"|"large"|"xl"|"flagship"
    parameters_b: float             # total params in billions
    size_gb: float                  # Q4_K_M download size in GB
    ram_gb: float                   # Q4_K_M runtime RAM in GB
    context_k: int                  # context window in thousands of tokens
    ollama_tag: str                 # exact `ollama pull` tag
    hf_repo: str                    # Hugging Face model repo ID
    mlx_repo: str                   # mlx-community repo (empty string if none)
    description: str                # one-line description
    highlight: str                  # what makes this model stand out
    tags: list[str]                 # capability tags
    bench_mmlu: Optional[float]     # 5-shot MMLU accuracy 0–1
    bench_humaneval: Optional[float]# HumanEval pass@1 0–1
    bench_gsm8k: Optional[float]    # GSM8K math accuracy 0–1
    bench_math: Optional[float]     # MATH competition accuracy 0–1
    speed_class: str                # "fast"|"medium"|"slow"
    source: str                     # "ollama"|"hf"|"mlx"
    added_at: str                   # ISO date
    is_new: bool = False            # flagged when first added via update

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CatalogEntry":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def best_bench(self) -> Optional[float]:
        """Highest available benchmark score as a proxy for general quality."""
        scores = [s for s in [self.bench_mmlu, self.bench_humaneval, self.bench_gsm8k] if s is not None]
        return max(scores) if scores else None


# ---------------------------------------------------------------------------
# Baseline catalog — compiled May 2026
# Sources: Open LLM Leaderboard, model papers, Ollama library, HF trending
# ---------------------------------------------------------------------------

_BASELINE_ENTRIES: list[dict] = [

    # ── Tiny (< 2 GB, < 3B params) ─────────────────────────────────────────

    dict(id="smollm2:135m", name="SmolLM2 135M", family="smollm", tier="tiny",
         parameters_b=0.135, size_gb=0.27, ram_gb=0.4, context_k=2,
         ollama_tag="smollm2:135m", hf_repo="HuggingFaceTB/SmolLM2-135M-Instruct",
         mlx_repo="mlx-community/SmolLM2-135M-Instruct-4bit",
         description="Ultra-tiny HuggingFace model, fits on any device, instant responses",
         highlight="Smallest usable model (90MB)",
         tags=["fast", "edge", "tiny"],
         bench_mmlu=0.295, bench_humaneval=None, bench_gsm8k=0.042,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="smollm2:1.7b", name="SmolLM2 1.7B", family="smollm", tier="tiny",
         parameters_b=1.71, size_gb=1.1, ram_gb=1.4, context_k=8,
         ollama_tag="smollm2:1.7b", hf_repo="HuggingFaceTB/SmolLM2-1.7B-Instruct",
         mlx_repo="mlx-community/SmolLM2-1.7B-Instruct-4bit",
         description="Best tiny open-source model from HuggingFace, 8k context, very fast",
         highlight="Best sub-2GB model",
         tags=["fast", "edge", "tiny"],
         bench_mmlu=0.510, bench_humaneval=0.155, bench_gsm8k=0.355,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="qwen3:0.6b", name="Qwen3 0.6B", family="qwen3", tier="tiny",
         parameters_b=0.60, size_gb=0.5, ram_gb=0.7, context_k=32,
         ollama_tag="qwen3:0.6b", hf_repo="Qwen/Qwen3-0.6B",
         mlx_repo="mlx-community/Qwen3-0.6B-4bit",
         description="Qwen3 ultra-tiny — hybrid thinking mode, on-device, 32k context",
         highlight="Best sub-1GB reasoning model",
         tags=["fast", "reasoning", "edge", "multilingual"],
         bench_mmlu=0.450, bench_humaneval=None, bench_gsm8k=0.310,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="qwen3:1.7b", name="Qwen3 1.7B", family="qwen3", tier="tiny",
         parameters_b=1.70, size_gb=1.1, ram_gb=1.4, context_k=32,
         ollama_tag="qwen3:1.7b", hf_repo="Qwen/Qwen3-1.7B",
         mlx_repo="mlx-community/Qwen3-1.7B-4bit",
         description="Qwen3 1.7B — strong reasoning for its size, hybrid thinking",
         highlight="Best 1.7B reasoning model",
         tags=["fast", "reasoning", "multilingual"],
         bench_mmlu=0.590, bench_humaneval=0.320, bench_gsm8k=0.520,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="llama3.2:1b", name="Llama 3.2 1B", family="llama", tier="tiny",
         parameters_b=1.24, size_gb=1.3, ram_gb=1.6, context_k=128,
         ollama_tag="llama3.2:1b", hf_repo="meta-llama/Llama-3.2-1B-Instruct",
         mlx_repo="mlx-community/Llama-3.2-1B-Instruct-4bit",
         description="Ultra-light Llama 3.2, 128k context, ideal for CPU/low-RAM devices",
         highlight="128k context in 1.6GB RAM",
         tags=["fast", "edge", "long-context"],
         bench_mmlu=0.328, bench_humaneval=0.289, bench_gsm8k=0.444,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="deepseek-r1:1.5b", name="DeepSeek R1 Distill 1.5B", family="deepseek", tier="tiny",
         parameters_b=1.50, size_gb=1.1, ram_gb=1.4, context_k=128,
         ollama_tag="deepseek-r1:1.5b", hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
         mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
         description="DeepSeek-R1 reasoning distilled into 1.5B — chain-of-thought at tiny size",
         highlight="Best reasoning under 2GB",
         tags=["reasoning", "fast", "chain-of-thought"],
         bench_mmlu=0.450, bench_humaneval=0.280, bench_gsm8k=0.655,
         bench_math=0.285, speed_class="fast", source="ollama", added_at="2026-05-25"),

    # ── Small (2–5 GB, 3–7B params) ────────────────────────────────────────

    dict(id="llama3.2:3b", name="Llama 3.2 3B", family="llama", tier="small",
         parameters_b=3.21, size_gb=2.0, ram_gb=2.5, context_k=128,
         ollama_tag="llama3.2:3b", hf_repo="meta-llama/Llama-3.2-3B-Instruct",
         mlx_repo="mlx-community/Llama-3.2-3B-Instruct-4bit",
         description="Compact Llama 3.2 — solid 3B with 128k context, great all-rounder",
         highlight="Best 3B for everyday tasks",
         tags=["fast", "long-context", "general"],
         bench_mmlu=0.586, bench_humaneval=0.524, bench_gsm8k=0.777,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="qwen3:4b", name="Qwen3 4B", family="qwen3", tier="small",
         parameters_b=4.00, size_gb=2.6, ram_gb=3.3, context_k=32,
         ollama_tag="qwen3:4b", hf_repo="Qwen/Qwen3-4B",
         mlx_repo="mlx-community/Qwen3-4B-4bit",
         description="Qwen3 4B — hybrid thinking, punches well above its weight in reasoning",
         highlight="Best reasoning under 3GB",
         tags=["reasoning", "coding", "multilingual", "chain-of-thought"],
         bench_mmlu=0.660, bench_humaneval=0.520, bench_gsm8k=0.810,
         bench_math=0.620, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="qwen2.5:3b", name="Qwen 2.5 3B", family="qwen", tier="small",
         parameters_b=3.09, size_gb=2.0, ram_gb=2.5, context_k=128,
         ollama_tag="qwen2.5:3b", hf_repo="Qwen/Qwen2.5-3B-Instruct",
         mlx_repo="mlx-community/Qwen2.5-3B-Instruct-4bit",
         description="Qwen 2.5 3B — fast, multilingual, 128k context, good for everyday tasks",
         highlight="Best multilingual 3B",
         tags=["fast", "multilingual", "long-context"],
         bench_mmlu=0.652, bench_humaneval=0.490, bench_gsm8k=0.791,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="phi3.5:3.8b", name="Phi-3.5 Mini 3.8B", family="phi", tier="small",
         parameters_b=3.82, size_gb=2.3, ram_gb=2.9, context_k=128,
         ollama_tag="phi3.5:3.8b", hf_repo="microsoft/Phi-3.5-mini-instruct",
         mlx_repo="mlx-community/Phi-3.5-mini-instruct-4bit",
         description="Microsoft Phi-3.5 Mini — 128k context, strong reasoning and coding",
         highlight="Best Microsoft model under 3GB",
         tags=["reasoning", "coding", "long-context"],
         bench_mmlu=0.690, bench_humaneval=0.590, bench_gsm8k=0.880,
         bench_math=0.750, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="gemma3:4b", name="Gemma 3 4B", family="gemma3", tier="small",
         parameters_b=4.30, size_gb=2.8, ram_gb=3.5, context_k=128,
         ollama_tag="gemma3:4b", hf_repo="google/gemma-3-4b-it",
         mlx_repo="mlx-community/gemma-3-4b-it-4bit",
         description="Google Gemma 3 4B — excellent quality/size, 128k context, multilingual",
         highlight="Best Google model under 4GB",
         tags=["general", "multilingual", "long-context"],
         bench_mmlu=0.700, bench_humaneval=0.580, bench_gsm8k=0.840,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="gemma4:4b", name="Gemma 4 4B", family="gemma4", tier="small",
         parameters_b=4.00, size_gb=2.5, ram_gb=3.1, context_k=128,
         ollama_tag="gemma4:4b", hf_repo="google/gemma-4-4b-it",
         mlx_repo="mlx-community/gemma-4-4b-it-4bit",
         description="Google Gemma 4 4B — multimodal-capable, strong reasoning, 128k context",
         highlight="Gemma4 compact — best sub-3GB general model",
         tags=["general", "multilingual", "multimodal", "long-context"],
         bench_mmlu=0.740, bench_humaneval=0.680, bench_gsm8k=0.870,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="deepseek-r1:7b", name="DeepSeek R1 Distill 7B", family="deepseek", tier="small",
         parameters_b=7.62, size_gb=4.7, ram_gb=5.9, context_k=128,
         ollama_tag="deepseek-r1:7b", hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
         mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
         description="DeepSeek-R1 reasoning distilled into Qwen2.5-7B — best math/code reasoning at 7B cost",
         highlight="Best reasoning model under 5GB",
         tags=["reasoning", "coding", "chain-of-thought", "math"],
         bench_mmlu=0.718, bench_humaneval=0.550, bench_gsm8k=0.880,
         bench_math=0.650, speed_class="fast", source="ollama", added_at="2026-05-25"),

    # ── Medium (5–12 GB, 7–13B params) ─────────────────────────────────────

    dict(id="qwen3:8b", name="Qwen3 8B", family="qwen3", tier="medium",
         parameters_b=8.00, size_gb=5.2, ram_gb=6.5, context_k=32,
         ollama_tag="qwen3:8b", hf_repo="Qwen/Qwen3-8B",
         mlx_repo="mlx-community/Qwen3-8B-4bit",
         description="Qwen3 8B — near-frontier quality, hybrid thinking/non-thinking, best 8B available",
         highlight="⭐ Best overall 8B model",
         tags=["reasoning", "coding", "multilingual", "chain-of-thought", "general"],
         bench_mmlu=0.740, bench_humaneval=0.680, bench_gsm8k=0.900,
         bench_math=0.780, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="qwen2.5:7b", name="Qwen 2.5 7B", family="qwen", tier="medium",
         parameters_b=7.62, size_gb=4.7, ram_gb=5.9, context_k=128,
         ollama_tag="qwen2.5:7b", hf_repo="Qwen/Qwen2.5-7B-Instruct",
         mlx_repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
         description="Qwen 2.5 7B — multilingual, strong math and coding, 128k context",
         highlight="Best 128k-context 7B",
         tags=["general", "multilingual", "long-context", "coding"],
         bench_mmlu=0.740, bench_humaneval=0.720, bench_gsm8k=0.888,
         bench_math=None, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="qwen2.5-coder:7b", name="Qwen 2.5 Coder 7B", family="qwen", tier="medium",
         parameters_b=7.62, size_gb=4.7, ram_gb=5.9, context_k=128,
         ollama_tag="qwen2.5-coder:7b", hf_repo="Qwen/Qwen2.5-Coder-7B-Instruct",
         mlx_repo="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
         description="Best coding model at 7B — near GPT-4o level on HumanEval",
         highlight="⭐ Best coding model under 6GB",
         tags=["coding", "general"],
         bench_mmlu=0.720, bench_humaneval=0.880, bench_gsm8k=0.830,
         bench_math=None, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="deepseek-r1:8b", name="DeepSeek R1 8B (Llama)", family="deepseek", tier="medium",
         parameters_b=8.03, size_gb=4.7, ram_gb=5.9, context_k=128,
         ollama_tag="deepseek-r1:8b", hf_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
         mlx_repo="mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
         description="DeepSeek-R1 reasoning in Llama 3.1 8B — excellent math and code with Llama compat",
         highlight="Best Llama-based reasoning at 8B",
         tags=["reasoning", "coding", "math", "chain-of-thought"],
         bench_mmlu=0.726, bench_humaneval=0.560, bench_gsm8k=0.892,
         bench_math=0.682, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="llama3.1:8b", name="Llama 3.1 8B", family="llama", tier="medium",
         parameters_b=8.03, size_gb=4.7, ram_gb=5.9, context_k=128,
         ollama_tag="llama3.1:8b", hf_repo="meta-llama/Meta-Llama-3.1-8B-Instruct",
         mlx_repo="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
         description="Meta Llama 3.1 8B — 128k context, solid all-rounder with broad tool-use support",
         highlight="Most compatible 8B (broad ecosystem)",
         tags=["general", "long-context", "tool-use"],
         bench_mmlu=0.667, bench_humaneval=0.726, bench_gsm8k=0.845,
         bench_math=None, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="mistral:7b", name="Mistral 7B v0.3", family="mistral", tier="medium",
         parameters_b=7.25, size_gb=4.1, ram_gb=5.1, context_k=32,
         ollama_tag="mistral:7b", hf_repo="mistralai/Mistral-7B-v0.3",
         mlx_repo="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
         description="Fast, reliable community workhorse — extremely well-tested at 7B",
         highlight="Most battle-tested 7B",
         tags=["fast", "general"],
         bench_mmlu=0.642, bench_humaneval=0.270, bench_gsm8k=0.521,
         bench_math=None, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="phi4-mini:3.8b", name="Phi-4 Mini 3.8B", family="phi", tier="small",
         parameters_b=3.82, size_gb=2.5, ram_gb=3.1, context_k=16,
         ollama_tag="phi4-mini", hf_repo="microsoft/phi-4-mini-instruct",
         mlx_repo="mlx-community/phi-4-mini-instruct-4bit",
         description="Microsoft Phi-4 Mini — exceptional reasoning for 3.8B, 16k context",
         highlight="Best Microsoft mini for reasoning",
         tags=["reasoning", "math", "coding"],
         bench_mmlu=0.700, bench_humaneval=0.620, bench_gsm8k=0.890,
         bench_math=0.750, speed_class="fast", source="ollama", added_at="2026-05-25"),

    dict(id="gemma3:12b", name="Gemma 3 12B", family="gemma3", tier="medium",
         parameters_b=12.2, size_gb=8.1, ram_gb=10.1, context_k=128,
         ollama_tag="gemma3:12b", hf_repo="google/gemma-3-12b-it",
         mlx_repo="mlx-community/gemma-3-12b-it-4bit",
         description="Google Gemma 3 12B — excellent quality/size ratio, strong reasoning and code",
         highlight="⭐ Best quality/size ratio at 12B",
         tags=["general", "reasoning", "coding", "multilingual", "long-context"],
         bench_mmlu=0.778, bench_humaneval=0.710, bench_gsm8k=0.890,
         bench_math=None, speed_class="medium", source="ollama", added_at="2026-05-25"),

    # ── Large (12–20 GB, 13–24B params) ────────────────────────────────────

    dict(id="qwen3:14b", name="Qwen3 14B", family="qwen3", tier="large",
         parameters_b=14.00, size_gb=9.0, ram_gb=11.3, context_k=32,
         ollama_tag="qwen3:14b", hf_repo="Qwen/Qwen3-14B",
         mlx_repo="mlx-community/Qwen3-14B-4bit",
         description="Qwen3 14B — excellent reasoning and instruction following, 16GB+ RAM",
         highlight="⭐ Best 14B reasoning model",
         tags=["reasoning", "coding", "multilingual", "chain-of-thought", "general"],
         bench_mmlu=0.790, bench_humaneval=0.780, bench_gsm8k=0.930,
         bench_math=0.820, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="phi4:14b", name="Phi-4 14B", family="phi", tier="large",
         parameters_b=14.7, size_gb=9.1, ram_gb=11.4, context_k=16,
         ollama_tag="phi4:latest", hf_repo="microsoft/phi-4",
         mlx_repo="mlx-community/phi-4-4bit",
         description="Microsoft Phi-4 — punches well above its weight on reasoning and math",
         highlight="⭐ Best reasoning per GB at 14B",
         tags=["reasoning", "math", "coding"],
         bench_mmlu=0.840, bench_humaneval=0.820, bench_gsm8k=0.912,
         bench_math=0.848, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="deepseek-r1:14b", name="DeepSeek R1 Distill 14B", family="deepseek", tier="large",
         parameters_b=14.77, size_gb=9.0, ram_gb=11.3, context_k=128,
         ollama_tag="deepseek-r1:14b", hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
         mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
         description="DeepSeek-R1 reasoning in Qwen2.5-14B — best reasoning per GB at 14B",
         highlight="Best chain-of-thought 14B",
         tags=["reasoning", "coding", "math", "chain-of-thought"],
         bench_mmlu=0.759, bench_humaneval=0.690, bench_gsm8k=0.930,
         bench_math=0.734, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="qwen2.5:14b", name="Qwen 2.5 14B", family="qwen", tier="large",
         parameters_b=14.77, size_gb=9.0, ram_gb=11.3, context_k=128,
         ollama_tag="qwen2.5:14b", hf_repo="Qwen/Qwen2.5-14B-Instruct",
         mlx_repo="mlx-community/Qwen2.5-14B-Instruct-4bit",
         description="Alibaba Qwen 2.5 14B — excellent multilingual + code quality, 128k context",
         highlight="Best multilingual 14B",
         tags=["general", "multilingual", "long-context", "coding"],
         bench_mmlu=0.798, bench_humaneval=0.769, bench_gsm8k=0.856,
         bench_math=None, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="qwen2.5-coder:14b", name="Qwen 2.5 Coder 14B", family="qwen", tier="large",
         parameters_b=14.77, size_gb=9.0, ram_gb=11.3, context_k=128,
         ollama_tag="qwen2.5-coder:14b", hf_repo="Qwen/Qwen2.5-Coder-14B-Instruct",
         mlx_repo="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
         description="Qwen 2.5 Coder 14B — near GPT-4o on HumanEval, best open coding model at 14B",
         highlight="⭐ Best coding model under 12GB",
         tags=["coding", "general"],
         bench_mmlu=0.790, bench_humaneval=0.891, bench_gsm8k=0.856,
         bench_math=None, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="mistral-small:24b", name="Mistral Small 3.1 24B", family="mistral", tier="large",
         parameters_b=24.15, size_gb=14.5, ram_gb=18.1, context_k=128,
         ollama_tag="mistral-small:24b", hf_repo="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
         mlx_repo="mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
         description="Mistral Small 3.1 24B — very competitive quality, 128k context, 24GB+ RAM",
         highlight="Best Mistral model at 24B",
         tags=["general", "multilingual", "long-context"],
         bench_mmlu=0.810, bench_humaneval=0.720, bench_gsm8k=0.870,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="gemma4:27b", name="Gemma 4 27B", family="gemma4", tier="large",
         parameters_b=27.0, size_gb=17.0, ram_gb=21.3, context_k=256,
         ollama_tag="gemma4:27b", hf_repo="google/gemma-4-27b-it",
         mlx_repo="mlx-community/gemma-4-27b-it-4bit",
         description="Google Gemma 4 27B — 256k context, multimodal, top Google open model",
         highlight="Longest context 27B (256k)",
         tags=["general", "multimodal", "long-context", "reasoning"],
         bench_mmlu=0.852, bench_humaneval=0.890, bench_gsm8k=0.930,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="gemma3:27b", name="Gemma 3 27B", family="gemma3", tier="large",
         parameters_b=27.2, size_gb=17.0, ram_gb=21.3, context_k=128,
         ollama_tag="gemma3:27b", hf_repo="google/gemma-3-27b-it",
         mlx_repo="mlx-community/gemma-3-27b-it-4bit",
         description="Google Gemma 3 27B — high quality reasoning and coding, multimodal, 128k",
         highlight="Best Google 27B (reasoning + code)",
         tags=["reasoning", "coding", "multimodal", "long-context"],
         bench_mmlu=0.842, bench_humaneval=0.780, bench_gsm8k=0.900,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),

    # ── XL (20–48 GB, 25–75B params) ───────────────────────────────────────

    dict(id="qwen3:30b-a3b", name="Qwen3 30B-A3B (MoE)", family="qwen3", tier="xl",
         parameters_b=30.5, size_gb=17.0, ram_gb=21.3, context_k=128,
         ollama_tag="qwen3:30b-a3b", hf_repo="Qwen/Qwen3-30B-A3B",
         mlx_repo="mlx-community/Qwen3-30B-A3B-4bit",
         description="Qwen3 30B MoE with only 3.3B active params — flagship quality at 7B inference cost",
         highlight="⭐ MoE: 30B quality, 7B speed",
         tags=["reasoning", "coding", "multilingual", "moe", "chain-of-thought", "general"],
         bench_mmlu=0.830, bench_humaneval=0.840, bench_gsm8k=0.940,
         bench_math=0.860, speed_class="medium", source="ollama", added_at="2026-05-25"),

    dict(id="qwen3:32b", name="Qwen3 32B", family="qwen3", tier="xl",
         parameters_b=32.0, size_gb=20.0, ram_gb=25.0, context_k=32,
         ollama_tag="qwen3:32b", hf_repo="Qwen/Qwen3-32B",
         mlx_repo="mlx-community/Qwen3-32B-4bit",
         description="Qwen3 32B dense — top open model in its class, hybrid thinking, 32GB+ RAM",
         highlight="⭐ Best dense 32B model",
         tags=["reasoning", "coding", "multilingual", "chain-of-thought", "general"],
         bench_mmlu=0.840, bench_humaneval=0.850, bench_gsm8k=0.950,
         bench_math=0.880, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="qwq:32b", name="QwQ 32B", family="qwen", tier="xl",
         parameters_b=32.5, size_gb=20.0, ram_gb=25.0, context_k=128,
         ollama_tag="qwq:32b", hf_repo="Qwen/QwQ-32B",
         mlx_repo="mlx-community/QwQ-32B-4bit",
         description="QwQ 32B — Qwen's specialized reasoning model, exceptional math and chain-of-thought",
         highlight="Best specialized reasoning (32B)",
         tags=["reasoning", "math", "chain-of-thought", "long-context"],
         bench_mmlu=0.856, bench_humaneval=0.650, bench_gsm8k=0.960,
         bench_math=0.900, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="deepseek-r1:32b", name="DeepSeek R1 Distill 32B", family="deepseek", tier="xl",
         parameters_b=32.5, size_gb=20.0, ram_gb=25.0, context_k=128,
         ollama_tag="deepseek-r1:32b", hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
         mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
         description="DeepSeek-R1 distilled into Qwen2.5-32B — near-frontier reasoning, 24GB+ RAM",
         highlight="Best R1 distill at 32B",
         tags=["reasoning", "coding", "math", "chain-of-thought"],
         bench_mmlu=0.820, bench_humaneval=0.720, bench_gsm8k=0.950,
         bench_math=0.872, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="qwen2.5-coder:32b", name="Qwen 2.5 Coder 32B", family="qwen", tier="xl",
         parameters_b=32.5, size_gb=20.0, ram_gb=25.0, context_k=128,
         ollama_tag="qwen2.5-coder:32b", hf_repo="Qwen/Qwen2.5-Coder-32B-Instruct",
         mlx_repo="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
         description="Qwen 2.5 Coder 32B — top open-source coding model, near GPT-4o on all code benchmarks",
         highlight="⭐ Best open-source coding model (32B)",
         tags=["coding", "general"],
         bench_mmlu=0.830, bench_humaneval=0.920, bench_gsm8k=0.900,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="qwen2.5:32b", name="Qwen 2.5 32B", family="qwen", tier="xl",
         parameters_b=32.5, size_gb=20.0, ram_gb=25.0, context_k=128,
         ollama_tag="qwen2.5:32b", hf_repo="Qwen/Qwen2.5-32B-Instruct",
         mlx_repo="mlx-community/Qwen2.5-32B-Instruct-4bit",
         description="Qwen 2.5 32B — excellent general-purpose model, strong multilingual, 128k context",
         highlight="Best balanced 32B general model",
         tags=["general", "multilingual", "long-context", "coding"],
         bench_mmlu=0.835, bench_humaneval=0.820, bench_gsm8k=0.920,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),

    # ── Flagship (48 GB+) ──────────────────────────────────────────────────

    dict(id="llama3.3:70b", name="Llama 3.3 70B", family="llama", tier="flagship",
         parameters_b=70.55, size_gb=40.0, ram_gb=50.0, context_k=128,
         ollama_tag="llama3.3:70b", hf_repo="meta-llama/Llama-3.3-70B-Instruct",
         mlx_repo="mlx-community/Llama-3.3-70B-Instruct-4bit",
         description="Meta Llama 3.3 70B — best open 70B, improved over 3.1, 128k context, 64GB+ RAM",
         highlight="⭐ Best open 70B model",
         tags=["general", "reasoning", "coding", "long-context"],
         bench_mmlu=0.870, bench_humaneval=0.880, bench_gsm8k=0.958,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="llama3.1:70b", name="Llama 3.1 70B", family="llama", tier="flagship",
         parameters_b=70.55, size_gb=40.0, ram_gb=50.0, context_k=128,
         ollama_tag="llama3.1:70b", hf_repo="meta-llama/Meta-Llama-3.1-70B-Instruct",
         mlx_repo="mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
         description="Meta Llama 3.1 70B — near-GPT-4 quality, 128k context, 64GB+ RAM",
         highlight="Original frontier-class Llama",
         tags=["general", "reasoning", "long-context"],
         bench_mmlu=0.860, bench_humaneval=0.805, bench_gsm8k=0.951,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="deepseek-r1:70b", name="DeepSeek R1 Distill 70B", family="deepseek", tier="flagship",
         parameters_b=70.55, size_gb=40.0, ram_gb=50.0, context_k=128,
         ollama_tag="deepseek-r1:70b", hf_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
         mlx_repo="mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit",
         description="DeepSeek-R1 reasoning distilled into Llama 3.3 70B — frontier-class reasoning at open-source cost",
         highlight="Best reasoning at 70B",
         tags=["reasoning", "coding", "math", "chain-of-thought"],
         bench_mmlu=0.875, bench_humaneval=0.850, bench_gsm8k=0.963,
         bench_math=0.942, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="qwen2.5:72b", name="Qwen 2.5 72B", family="qwen", tier="flagship",
         parameters_b=72.71, size_gb=43.0, ram_gb=53.8, context_k=128,
         ollama_tag="qwen2.5:72b", hf_repo="Qwen/Qwen2.5-72B-Instruct",
         mlx_repo="mlx-community/Qwen2.5-72B-Instruct-4bit",
         description="Alibaba Qwen 2.5 72B — top multilingual and coding at 72B, 48GB+ RAM",
         highlight="Best multilingual + coding 72B",
         tags=["general", "multilingual", "coding", "long-context"],
         bench_mmlu=0.865, bench_humaneval=0.861, bench_gsm8k=0.916,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="qwen3:72b", name="Qwen3 72B", family="qwen3", tier="flagship",
         parameters_b=72.0, size_gb=43.0, ram_gb=53.8, context_k=128,
         ollama_tag="qwen3:72b", hf_repo="Qwen/Qwen3-72B",
         mlx_repo="mlx-community/Qwen3-72B-4bit",
         description="Qwen3 72B — hybrid thinking, top-tier quality, 64GB+ RAM",
         highlight="⭐ Best balanced 70B+ model",
         tags=["reasoning", "coding", "multilingual", "chain-of-thought", "general"],
         bench_mmlu=0.875, bench_humaneval=0.890, bench_gsm8k=0.965,
         bench_math=0.910, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="qwen3:235b-a22b", name="Qwen3 235B-A22B (MoE)", family="qwen3", tier="flagship",
         parameters_b=235.0, size_gb=142.0, ram_gb=177.5, context_k=128,
         ollama_tag="qwen3:235b-a22b", hf_repo="Qwen/Qwen3-235B-A22B",
         mlx_repo="",
         description="Qwen3 flagship MoE (22B active) — frontier-level quality, needs 128GB+ RAM at Q4",
         highlight="Largest open MoE — frontier quality",
         tags=["reasoning", "coding", "multilingual", "moe", "flagship"],
         bench_mmlu=0.875, bench_humaneval=0.870, bench_gsm8k=0.970,
         bench_math=0.940, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="llama4:maverick", name="Llama 4 Maverick", family="llama4", tier="flagship",
         parameters_b=400.0, size_gb=140.0, ram_gb=175.0, context_k=1024,
         ollama_tag="llama4:maverick", hf_repo="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
         mlx_repo="",
         description="Meta Llama 4 Maverick 400B MoE (17B active) — multimodal, 1M context, 128GB+ RAM",
         highlight="1M context, multimodal MoE",
         tags=["multimodal", "long-context", "moe", "flagship"],
         bench_mmlu=0.852, bench_humaneval=0.770, bench_gsm8k=0.930,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),

    dict(id="llama4:scout", name="Llama 4 Scout", family="llama4", tier="flagship",
         parameters_b=109.0, size_gb=67.0, ram_gb=83.8, context_k=10240,
         ollama_tag="llama4:scout", hf_repo="meta-llama/Llama-4-Scout-17B-16E",
         mlx_repo="",
         description="Meta Llama 4 Scout 109B MoE (17B active) — 10M context, multimodal",
         highlight="10M context window",
         tags=["multimodal", "long-context", "moe"],
         bench_mmlu=0.760, bench_humaneval=0.680, bench_gsm8k=0.880,
         bench_math=None, speed_class="slow", source="ollama", added_at="2026-05-25"),
]


# ---------------------------------------------------------------------------
# Catalog structure helpers
# ---------------------------------------------------------------------------

def _make_default_catalog() -> dict:
    """Return the default catalog dict seeded with the baseline entries."""
    now = time.time()
    return {
        "version": _CATALOG_VERSION,
        "last_updated": _fmt_ts(now),
        "next_update":  _fmt_ts(now + DEFAULT_UPDATE_INTERVAL_DAYS * 86400),
        "update_interval_days": DEFAULT_UPDATE_INTERVAL_DAYS,
        "models": [dict(e, is_new=False) for e in _BASELINE_ENTRIES],
    }


def _fmt_ts(ts: float) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(s: str) -> float:
    import datetime
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_catalog() -> dict:
    """
    Load the catalog from disk.  If it doesn't exist, seed it from the
    bundled baseline and write it to CATALOG_PATH.
    """
    if CATALOG_PATH.exists():
        try:
            data = json.loads(CATALOG_PATH.read_text("utf-8"))
            if data.get("version") == _CATALOG_VERSION:
                return data
        except Exception:
            pass
    # Seed from baseline
    catalog = _make_default_catalog()
    save_catalog(catalog)
    return catalog


def save_catalog(catalog: dict) -> None:
    """Write the catalog to disk, creating ~/.autotune if needed."""
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), "utf-8")


def get_entries(catalog: dict) -> list[CatalogEntry]:
    """Return all CatalogEntry objects from the catalog."""
    entries = []
    for m in catalog.get("models", []):
        try:
            entries.append(CatalogEntry.from_dict(m))
        except Exception:
            pass
    return entries


def is_stale(catalog: dict) -> bool:
    """True if the catalog is due for a refresh."""
    try:
        return time.time() > _parse_ts(catalog.get("next_update", "2000-01-01T00:00:00Z"))
    except Exception:
        return True


def set_update_interval(catalog: dict, days: int) -> None:
    """Change the refresh interval and recompute next_update."""
    catalog["update_interval_days"] = days
    try:
        last = _parse_ts(catalog.get("last_updated", _fmt_ts(time.time())))
    except Exception:
        last = time.time()
    catalog["next_update"] = _fmt_ts(last + days * 86400)


def get_best_for_ram(
    catalog: dict,
    ram_gb: float,
    mode: str = "balanced",
) -> list[CatalogEntry]:
    """
    Return models that fit in *ram_gb* RAM, sorted by suitability.

    mode="fastest"     — prioritise small + fast
    mode="balanced"    — best MMLU score that fits comfortably
    mode="best_quality"— highest quality regardless of comfort
    """
    entries = get_entries(catalog)
    # Leave 20% headroom for OS + apps
    budget = ram_gb * 0.80
    fits = [e for e in entries if e.ram_gb <= budget]

    if mode == "fastest":
        # For fastest: prefer smallest RAM footprint, break ties by quality
        fits.sort(key=lambda e: (e.ram_gb, -(e.bench_mmlu or 0)))
    elif mode == "best_quality":
        fits.sort(key=lambda e: -(e.bench_mmlu or 0))
    else:  # balanced — quality first, small headroom bonus, slight speed bonus
        fits.sort(key=lambda e: -(
            0.70 * (e.bench_mmlu or 0)
            + 0.20 * max(0.0, 1.0 - e.ram_gb / ram_gb)
            + 0.10 * (1.0 if e.speed_class == "fast" else 0.5 if e.speed_class == "medium" else 0.0)
        ))
    return fits


def get_by_tier(catalog: dict, tier: str) -> list[CatalogEntry]:
    """All entries for a specific tier, sorted by MMLU score descending."""
    entries = [e for e in get_entries(catalog) if e.tier == tier]
    entries.sort(key=lambda e: -(e.bench_mmlu or 0))
    return entries


def get_by_tag(catalog: dict, tag: str) -> list[CatalogEntry]:
    """All entries with the given tag, sorted by MMLU score descending."""
    entries = [e for e in get_entries(catalog) if tag in e.tags]
    entries.sort(key=lambda e: -(e.bench_mmlu or 0))
    return entries


def format_for_api(catalog: dict) -> dict:
    """Serialise for the dashboard REST endpoint."""
    return {
        "last_updated": catalog.get("last_updated"),
        "next_update":  catalog.get("next_update"),
        "update_interval_days": catalog.get("update_interval_days", DEFAULT_UPDATE_INTERVAL_DAYS),
        "is_stale": is_stale(catalog),
        "count": len(catalog.get("models", [])),
        "models": catalog.get("models", []),
    }
