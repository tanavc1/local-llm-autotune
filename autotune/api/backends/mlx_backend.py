"""
MLX-LM backend for Apple Silicon (M-series) Macs.

MLX is Apple's machine learning framework built for unified memory.
Runs entirely on-chip: GPU kernels (Metal), CPU, and Neural Engine all
share the same physical RAM — no PCIe copies, no VRAM limit.

Why this is faster than Ollama on Apple Silicon:
  - Native Metal GPU kernels tuned for Apple's matrix units (AMX/ANE)
  - Zero CPU↔GPU copy overhead (unified memory)
  - bfloat16 natively on M4+; fp16 on M1/M2/M3
  - Lazy evaluation graph — adapts to input length without recompilation
  - Typically 10–40% higher tok/s than llama.cpp/Ollama on the same model

Model loading:
  Models are loaded from HuggingFace Hub (mlx-community org hosts
  pre-quantized MLX versions of most popular models).  The model stays
  resident in unified memory between turns — no reload per request.

Only activates on Darwin/arm64.  Silently unavailable on other platforms.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional

from .base import Backend, ChatChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform gate
# ---------------------------------------------------------------------------

IS_APPLE_SILICON: bool = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)

_mlx_available: Optional[bool] = None


def mlx_available() -> bool:
    """Return True if mlx-lm is installed, we are on Apple Silicon, and MLX
    routing has not been disabled via AUTOTUNE_DISABLE_MLX=1.

    Setting AUTOTUNE_DISABLE_MLX=1 forces all requests through Ollama.  This
    prevents mlx_lm from loading the HuggingFace transformers tokenizer, which
    transitively imports torch (~250-300 MB RSS) on first use.  Use this when
    you want the smallest possible server memory footprint.
    """
    if os.environ.get("AUTOTUNE_DISABLE_MLX", "").strip() in ("1", "true", "yes"):
        return False
    global _mlx_available
    if _mlx_available is None:
        if not IS_APPLE_SILICON:
            _mlx_available = False
        else:
            try:
                import mlx_lm  # noqa: F401
                _mlx_available = True
            except ImportError:
                _mlx_available = False
    return _mlx_available


# ---------------------------------------------------------------------------
# Ollama → MLX model ID mapping
# ---------------------------------------------------------------------------

# Known mlx-community models for popular Ollama tags.
# Format: {ollama_base_name: mlx_community_id}
# Ollama base name = tag without ":version" suffix, lowercased.
_KNOWN_MLX_MAP: dict[str, str] = {
    # Phi
    "phi4-mini":          "mlx-community/Phi-4-mini-instruct-4bit",
    "phi4":               "mlx-community/phi-4-4bit",
    "phi3.5":             "mlx-community/Phi-3.5-mini-instruct-4bit",
    "phi3":               "mlx-community/Phi-3-mini-4k-instruct-4bit",
    # Qwen 2.5
    "qwen2.5:0.5b":       "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "qwen2.5:1.5b":       "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "qwen2.5:3b":         "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "qwen2.5:7b":         "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen2.5:14b":        "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "qwen2.5:32b":        "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "qwen2.5":            "mlx-community/Qwen2.5-7B-Instruct-4bit",
    # Qwen 2.5 Coder
    "qwen2.5-coder:1.5b": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "qwen2.5-coder:7b":   "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "qwen2.5-coder:14b":  "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "qwen2.5-coder:32b":  "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    "qwen2.5-coder":      "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    # Qwen 3
    "qwen3:0.6b":         "mlx-community/Qwen3-0.6B-4bit",
    "qwen3:1.7b":         "mlx-community/Qwen3-1.7B-4bit",
    "qwen3:4b":           "mlx-community/Qwen3-4B-4bit",
    "qwen3:8b":           "mlx-community/Qwen3-8B-4bit",
    "qwen3:14b":          "mlx-community/Qwen3-14B-4bit",
    "qwen3:32b":          "mlx-community/Qwen3-32B-4bit",
    "qwen3":              "mlx-community/Qwen3-8B-4bit",
    # Llama 3.2 / 3.1
    "llama3.2:1b":        "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama3.2:3b":        "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "llama3.2":           "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "llama3.1:8b":        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "llama3.1:70b":       "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
    "llama3.1":           "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "llama3:8b":          "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    "llama3":             "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    # Mistral
    "mistral:7b":         "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mistral":            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mistral-nemo":       "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    # Gemma
    "gemma2:2b":          "mlx-community/gemma-2-2b-it-4bit",
    "gemma2:9b":          "mlx-community/gemma-2-9b-it-4bit",
    "gemma2:27b":         "mlx-community/gemma-2-27b-it-4bit",
    "gemma2":             "mlx-community/gemma-2-9b-it-4bit",
    "gemma3:1b":          "mlx-community/gemma-3-1b-it-4bit",
    "gemma3:4b":          "mlx-community/gemma-3-4b-it-4bit",
    "gemma3:12b":         "mlx-community/gemma-3-12b-it-4bit",
    "gemma3":             "mlx-community/gemma-3-4b-it-4bit",
    # Gemma 4 (April 2025) — multimodal, 128k context
    "gemma4:e2b":         "mlx-community/gemma-4-2b-it-4bit",
    "gemma4:e4b":         "mlx-community/gemma-4-4b-it-4bit",
    "gemma4:26b":         "mlx-community/gemma-4-27b-it-4bit",
    "gemma4":             "mlx-community/gemma-4-4b-it-4bit",
    # DeepSeek
    "deepseek-r1:7b":     "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
    "deepseek-r1:8b":     "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
    "deepseek-r1:14b":    "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
    "deepseek-r1:32b":    "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
    "deepseek-r1":        "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
    # SmolLM
    "smollm2:135m":       "mlx-community/SmolLM2-135M-Instruct-4bit",
    "smollm2:360m":       "mlx-community/SmolLM2-360M-Instruct-4bit",
    "smollm2:1.7b":       "mlx-community/SmolLM2-1.7B-Instruct-4bit",
    "smollm2":            "mlx-community/SmolLM2-1.7B-Instruct-4bit",
    # Qwen3 VL / Qwen2.5 VL
    "qwen3-vl:8b":        "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "qwen2.5-vl:7b":      "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    # Llama 4
    "llama4:scout":       "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
}

# HuggingFace cache root
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def resolve_mlx_model_id(ollama_or_hf_id: str) -> Optional[str]:
    """
    Map an Ollama model tag or HF model ID to an MLX-compatible HF model ID.

    Resolution order:
      1. If already an HF-style ID (contains "/"), return as-is.
      2. Check the known map using the full tag (e.g. "qwen2.5-coder:14b").
      3. Check the known map using only the base name (no tag, e.g. "qwen2.5-coder").
      4. Scan local HF cache for any mlx-community model whose name contains
         the base model name.
      5. Return None (caller should fall back to Ollama).
    """
    raw = ollama_or_hf_id.strip()

    # 1. Already an HF ID
    if "/" in raw:
        return raw

    lower = raw.lower()

    # 2. Full tag match (e.g. "phi4-mini:latest" → "phi4-mini")
    # Strip ":latest" suffix but keep version tags like ":14b"
    if lower.endswith(":latest"):
        lower = lower[: -len(":latest")]

    if lower in _KNOWN_MLX_MAP:
        return _KNOWN_MLX_MAP[lower]

    # 3. Base name only (strip version suffix)
    base = lower.split(":")[0]
    if base in _KNOWN_MLX_MAP:
        return _KNOWN_MLX_MAP[base]

    # 4. Scan local HF cache for a matching mlx-community model
    cached = _find_mlx_in_cache(base)
    if cached:
        return cached

    return None


def _find_mlx_in_cache(model_base: str) -> Optional[str]:
    """Scan ~/.cache/huggingface/hub for any mlx-community model matching `model_base`."""
    if not _HF_CACHE.exists():
        return None
    needle = model_base.replace("-", "").replace("_", "").lower()
    candidates: list[str] = []
    for item in _HF_CACHE.iterdir():
        if not (item.is_dir() and item.name.startswith("models--mlx-community--")):
            continue
        # models--mlx-community--Phi-4-mini-instruct-4bit
        model_name = item.name[len("models--mlx-community--"):]
        # Check if snapshots dir is non-empty (i.e., actually downloaded)
        snap = item / "snapshots"
        if not snap.exists() or not any(snap.iterdir()):
            continue
        norm = model_name.replace("-", "").replace("_", "").lower()
        if needle in norm:
            hf_id = f"mlx-community/{model_name}"
            candidates.append(hf_id)

    if not candidates:
        return None
    # Prefer 4bit quantization if multiple matches
    for c in candidates:
        if "4bit" in c.lower():
            return c
    return candidates[0]


def list_cached_mlx_models() -> list[dict]:
    """Return all MLX models in the local HuggingFace cache."""
    if not _HF_CACHE.exists():
        return []
    models = []
    for item in sorted(_HF_CACHE.iterdir()):
        if not (item.is_dir() and item.name.startswith("models--mlx-community--")):
            continue
        snap = item / "snapshots"
        if not snap.exists() or not any(snap.iterdir()):
            continue
        model_name = item.name[len("models--mlx-community--"):]
        hf_id = f"mlx-community/{model_name}"
        # Estimate size from largest snapshot
        size_gb = 0.0
        for snap_dir in snap.iterdir():
            if snap_dir.is_dir():
                total = sum(f.stat().st_size for f in snap_dir.rglob("*") if f.is_file())
                size_gb = max(size_gb, total / 1024**3)
        models.append({"id": hf_id, "name": model_name, "size_gb": round(size_gb, 2)})
    return models


# ---------------------------------------------------------------------------
# Loaded model cache — keeps last model in unified memory between requests
# ---------------------------------------------------------------------------

@dataclass
class _LoadedModel:
    model_id: str
    model: object
    tokenizer: object
    loaded_at: float


_model_cache: Optional[_LoadedModel] = None

# State file: written on load, deleted on unload/exit.
# Lets `autotune ps` (a separate process) see the loaded model.
_STATE_FILE = Path.home() / ".autotune" / "mlx_running.json"


def _write_mlx_state(model_id: str, loaded_at: float) -> None:
    """Persist MLX model state to disk so other processes can read it."""
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps({
            "model_id": model_id,
            "loaded_at": loaded_at,
            "pid": os.getpid(),
        }))
    except Exception:
        pass


def _clear_mlx_state() -> None:
    """Remove the MLX state file (model unloaded or process exiting)."""
    try:
        _STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _load_model_sync(model_id: str) -> _LoadedModel:
    """Load (or return cached) MLX model. Called from a thread pool."""
    global _model_cache

    if _model_cache and _model_cache.model_id == model_id:
        logger.debug("MLX: cache hit for %s", model_id)
        return _model_cache

    logger.info("MLX: loading %s into unified memory…", model_id)
    t0 = time.perf_counter()

    from mlx_lm import load
    model, tokenizer = load(model_id)

    elapsed = time.perf_counter() - t0
    logger.info("MLX: loaded %s in %.1fs", model_id, elapsed)

    loaded_at = time.time()
    _model_cache = _LoadedModel(
        model_id=model_id, model=model, tokenizer=tokenizer, loaded_at=loaded_at,
    )
    _write_mlx_state(model_id, loaded_at)
    return _model_cache


def _format_messages(tokenizer, messages: list[dict]) -> str:
    """
    Apply the model's chat template to produce a single prompt string.

    Falls back to a simple ChatML-style format when no template is defined.
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning("apply_chat_template failed (%s), using fallback", e)

    # Fallback: simple ChatML format
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MLX Backend
# ---------------------------------------------------------------------------

class MLXBackend(Backend):
    """
    MLX-LM inference backend for Apple Silicon.

    Loads models directly from HuggingFace (mlx-community quantized versions)
    into unified memory and runs generation via Apple's Metal GPU kernels.

    The loaded model is cached in-process between requests.  Switching to a
    different model evicts the previous one from unified memory.
    """

    name = "mlx"

    async def is_available(self) -> bool:
        return mlx_available()

    async def has_model(self, model_id: str) -> bool:
        if not mlx_available():
            return False
        return resolve_mlx_model_id(model_id) is not None

    async def list_models(self) -> list[str]:
        return [m["id"] for m in list_cached_mlx_models()]

    async def stream(
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        timeout: float = 300.0,
        **_kwargs,   # absorbs num_ctx, ollama_options, etc.
    ) -> AsyncGenerator[ChatChunk, None]:
        if not mlx_available():
            raise RuntimeError("MLX is not available on this platform")

        mlx_id = resolve_mlx_model_id(model_id)
        if mlx_id is None:
            raise RuntimeError(
                f"No MLX model found for {model_id!r}. "
                f"Pull one with: autotune mlx pull {model_id}"
            )

        loop = asyncio.get_running_loop()

        # Load model in thread pool (blocks, but must not block event loop)
        loaded = await loop.run_in_executor(None, _load_model_sync, mlx_id)
        prompt  = _format_messages(loaded.tokenizer, messages)

        # Bridge: synchronous generator → asyncio.Queue → async generator
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=256)

        def _run_generation() -> None:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
            try:
                # Build sampler — mlx-lm ≥ 0.21 uses sampler/logits_processors
                # instead of bare temp/top_p kwargs.
                sampler = make_sampler(temp=temperature, top_p=top_p if top_p < 1.0 else 0.0)
                logits_processors = None
                if repetition_penalty != 1.0:
                    logits_processors = make_logits_processors(
                        repetition_penalty=repetition_penalty
                    )

                kwargs: dict = {"sampler": sampler}
                if logits_processors:
                    kwargs["logits_processors"] = logits_processors

                for response in stream_generate(
                    loaded.model,
                    loaded.tokenizer,
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    **kwargs,
                ):
                    # response.text is the newly generated text for this step
                    if response.text:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(response.text), loop
                        ).result(timeout=5.0)
            except Exception as exc:
                logger.error("MLX generation error: %s", exc)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=5.0)

        # Start generation in background thread
        gen_future = loop.run_in_executor(None, _run_generation)

        t_start = time.perf_counter()
        first_token = True

        try:
            while True:
                try:
                    text = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning("MLX: generation timed out after %.0fs", timeout)
                    break

                if text is None:
                    break   # sentinel — generation complete

                if first_token:
                    ttft = (time.perf_counter() - t_start) * 1000
                    logger.debug("MLX TTFT: %.0fms for %s", ttft, mlx_id)
                    first_token = False

                yield ChatChunk(content=text, backend="mlx", model=mlx_id)
        finally:
            # Ensure the background thread completes
            try:
                await asyncio.wait_for(asyncio.wrap_future(gen_future), timeout=10.0)
            except (asyncio.TimeoutError, Exception):
                pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_backend: Optional[MLXBackend] = None


def get_mlx_backend() -> MLXBackend:
    global _backend
    if _backend is None:
        _backend = MLXBackend()
    return _backend


def is_mlx_model_loaded(mlx_id: str) -> bool:
    """Return True if mlx_id is already resident in unified memory."""
    return _model_cache is not None and _model_cache.model_id == mlx_id


def unload_mlx_model() -> bool:
    """Release the MLX model from unified memory.

    Clears the module-level cache and runs the garbage collector so the
    Metal allocator can reclaim the weights immediately.  Returns True if
    a model was actually unloaded.
    """
    global _model_cache
    import gc
    if _model_cache is None:
        return False
    _model_cache = None
    _clear_mlx_state()
    gc.collect()
    # Best-effort: clear Metal cache if mlx exposes it
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
    return True


# Ensure the state file is cleaned up if the process exits unexpectedly.
atexit.register(_clear_mlx_state)
