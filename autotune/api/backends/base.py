"""Abstract backend + shared response types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional


@dataclass
class ChatChunk:
    """One streamed token chunk."""
    content: str
    finish_reason: Optional[str] = None   # "stop" | "length" | None
    model: str = ""
    backend: str = ""


@dataclass
class CompletionStats:
    backend: str
    model: str
    ttft_ms: float
    tokens_per_sec: float
    gen_tokens_per_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str


class Backend(ABC):
    name: str = "abstract"

    @abstractmethod
    async def is_available(self) -> bool:
        ...

    @abstractmethod
    async def has_model(self, model_id: str) -> bool:
        ...

    @abstractmethod
    def stream(
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        timeout: float,
    ) -> AsyncGenerator[ChatChunk, None]:
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        ...
