"""
Generic OpenAI-compatible backend.

Works with any server that speaks the /v1/chat/completions SSE protocol:
  - HuggingFace Inference API
  - Ollama (11434)
  - LM Studio (1234)
  - vLLM
  - Any OpenAI-compatible server

KV-cache hint: send the system prompt as the very first message in every
request so the server (Ollama, vLLM) can cache it.  We never strip it
between turns to maximize cache hit rate.
"""

from __future__ import annotations

import json
import time
from typing import AsyncGenerator, Optional

import httpx

from .base import Backend, ChatChunk


class OpenAICompatBackend(Backend):
    """
    Backend that speaks /v1/chat/completions.

    base_url examples:
      HF:       "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct"
      Ollama:   "http://localhost:11434"
      LMStudio: "http://localhost:1234"
    """

    def __init__(self, base_url: str, api_key: str = "", backend_name: str = "openai_compat") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = backend_name

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base_url}/v1/models", headers=self._headers())
                return r.status_code < 500
        except Exception:
            return False

    async def has_model(self, model_id: str) -> bool:
        models = await self.list_models()
        return any(m == model_id or m.endswith(model_id.split("/")[-1]) for m in models)

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/v1/models", headers=self._headers())
                data = r.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    async def stream(  # type: ignore[override]
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        timeout: float = 120.0,
        extra_body: Optional[dict] = None,
    ) -> AsyncGenerator[ChatChunk, None]:
        payload: dict = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty
        if extra_body:
            payload.update(extra_body)

        url = f"{self.base_url}/v1/chat/completions"

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as resp:
                if resp.status_code == 404:
                    raise ModelNotAvailableError(
                        f"Model {model_id!r} returned 404 at {self.name}. "
                        f"Make sure the model is available at {url}."
                    )
                if resp.status_code == 401:
                    raise AuthError(
                        "Authentication failed. Set HF_TOKEN env var for HuggingFace models."
                    )
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise BackendError(
                        f"{self.name} returned {resp.status_code}: {body.decode()[:200]}"
                    )

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue
                    try:
                        chunk = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    finish = choices[0].get("finish_reason")

                    if content or finish:
                        yield ChatChunk(
                            content=content,
                            finish_reason=finish,
                            model=chunk.get("model", model_id),
                            backend=self.name,
                        )


class ModelNotAvailableError(Exception):
    pass


class AuthError(Exception):
    pass


class BackendError(Exception):
    pass
