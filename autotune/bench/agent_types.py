"""
Agent benchmark dataclasses.

Defines the three-tier hierarchy used by the agent harness:
  AgentTask        — one benchmark scenario (goal + tools + success criterion)
  AgentTurnResult  — per-turn measurements from one inference + tool call
  AgentRunResult   — full task run (all turns, final success/failure, aggregates)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]   # param_name → short description

    def system_block(self) -> str:
        param_str = ", ".join(f'"{k}": "<{v}>"' for k, v in self.params.items())
        return f"- {self.name}: {self.description}\n  Input: {{{param_str}}}"


# Standard tool set available to agents
TOOL_READ_FILE = Tool(
    name="read_file",
    description="Read a file's contents from the workspace.",
    params={"path": "relative file path"},
)
TOOL_WRITE_FILE = Tool(
    name="write_file",
    description="Write or overwrite a file in the workspace.",
    params={"path": "relative file path", "content": "full file contents"},
)
TOOL_LIST_FILES = Tool(
    name="list_files",
    description="List files in a workspace directory.",
    params={"directory": "directory path, or empty string for root"},
)
TOOL_RUN_PYTHON = Tool(
    name="run_python",
    description="Execute Python code and return stdout + stderr.",
    params={"code": "valid Python code to execute"},
)
TOOL_SEARCH_DOCS = Tool(
    name="search_docs",
    description="Search the document corpus for relevant passages.",
    params={"query": "natural language search query"},
)


# ---------------------------------------------------------------------------
# Agent task
# ---------------------------------------------------------------------------

@dataclass
class AgentTask:
    task_id: str
    label: str
    difficulty: str              # "shallow" | "deep" | "adversarial"
    goal: str                    # the initial user message
    system_prompt: str           # system prompt for this task
    tools: list[Tool]            # tools available to the agent
    max_turns: int               # hard cap
    success_fn: Callable[        # given (final_text, files_written, messages) → bool
        [str, dict[str, str], list[dict]], bool
    ]
    synthetic_files: dict[str, str] = field(default_factory=dict)
    doc_corpus: dict[str, str]   = field(default_factory=dict)
    stress_context: str          = ""   # extra text injected to inflate context (adversarial)
    expected_kpi_wins: str       = ""   # human description of where autotune should win

    def tool_block(self) -> str:
        return "\n".join(t.system_block() for t in self.tools)


# ---------------------------------------------------------------------------
# Per-turn measurements
# ---------------------------------------------------------------------------

@dataclass
class AgentTurnResult:
    turn_idx: int
    role: str                    # "reasoning" | "tool_call" | "final" | "error"

    # Ollama-native timings (nanosecond precision from Ollama Go runtime)
    prefill_ms: float            # prompt_eval_duration → KV fill cost
    ttft_ms: float               # load_ms + prefill_ms
    eval_tps: float              # generation tok/s
    total_ms: float              # end-to-end turn wall time

    # Memory
    ollama_ram_gb: float         # Ollama runner peak RSS this turn
    kv_cache_mb: float           # estimated KV cache for this num_ctx
    swap_delta_gb: float         # swap pressure this turn

    # Context state at this turn
    tokens_in_context: int       # estimated tokens in the message list
    num_ctx: int                 # context window allocated

    # Tool info (None when role is "reasoning" or "final")
    tool_name: Optional[str] = None
    tool_success: Optional[bool] = None
    tool_latency_ms: Optional[float] = None
    tool_result_preview: Optional[str] = None

    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.eval_tps > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_idx":           self.turn_idx,
            "role":               self.role,
            "prefill_ms":         round(self.prefill_ms, 1),
            "ttft_ms":            round(self.ttft_ms, 1),
            "eval_tps":           round(self.eval_tps, 2),
            "total_ms":           round(self.total_ms, 1),
            "ollama_ram_gb":      round(self.ollama_ram_gb, 3),
            "kv_cache_mb":        round(self.kv_cache_mb, 1),
            "swap_delta_gb":      round(self.swap_delta_gb, 4),
            "tokens_in_context":  self.tokens_in_context,
            "num_ctx":            self.num_ctx,
            "tool_name":          self.tool_name,
            "tool_success":       self.tool_success,
            "tool_latency_ms":    round(self.tool_latency_ms, 1) if self.tool_latency_ms else None,
            "error":              self.error,
        }


# ---------------------------------------------------------------------------
# Full task run (one trial)
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    task_id: str
    condition: str               # "raw" | "autotune"
    model_id: str
    trial_idx: int

    # Turn trace
    turns: list[AgentTurnResult]

    # Outcome
    task_success: bool
    exit_reason: str             # "success" | "max_turns" | "oom" | "error"

    # Aggregates
    total_wall_sec: float
    total_tool_calls: int
    tool_error_count: int
    backtrack_count: int
    reload_count: int

    # Memory aggregates
    peak_ram_gb: float
    swap_occurred: bool
    free_floor_gb: float

    # Context
    final_context_tokens: int

    # Per-turn series (for TTFT growth curves)
    @property
    def ttft_series(self) -> list[float]:
        return [t.ttft_ms for t in self.turns if t.ok]

    @property
    def prefill_series(self) -> list[float]:
        return [t.prefill_ms for t in self.turns if t.ok]

    @property
    def ram_series(self) -> list[float]:
        return [t.ollama_ram_gb for t in self.turns if t.ok]

    @property
    def context_series(self) -> list[int]:
        return [t.tokens_in_context for t in self.turns if t.ok]

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def avg_ttft_ms(self) -> float:
        s = self.ttft_series
        return sum(s) / len(s) if s else 0.0

    @property
    def avg_tps(self) -> float:
        vals = [t.eval_tps for t in self.turns if t.ok]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def total_tool_errors_pct(self) -> float:
        if self.total_tool_calls == 0:
            return 0.0
        return 100.0 * self.tool_error_count / self.total_tool_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id":               self.task_id,
            "condition":             self.condition,
            "model_id":              self.model_id,
            "trial_idx":             self.trial_idx,
            "task_success":          self.task_success,
            "exit_reason":           self.exit_reason,
            "total_wall_sec":        round(self.total_wall_sec, 2),
            "total_tool_calls":      self.total_tool_calls,
            "tool_error_count":      self.tool_error_count,
            "backtrack_count":       self.backtrack_count,
            "reload_count":          self.reload_count,
            "n_turns":               self.n_turns,
            "peak_ram_gb":           round(self.peak_ram_gb, 3),
            "swap_occurred":         self.swap_occurred,
            "free_floor_gb":         round(self.free_floor_gb, 3),
            "final_context_tokens":  self.final_context_tokens,
            "avg_ttft_ms":           round(self.avg_ttft_ms, 1),
            "avg_tps":               round(self.avg_tps, 2),
            "ttft_series":           [round(v, 1) for v in self.ttft_series],
            "context_series":        self.context_series,
            "turns":                 [t.to_dict() for t in self.turns],
        }


# ---------------------------------------------------------------------------
# Cross-trial aggregates (one task × one condition × N trials)
# ---------------------------------------------------------------------------

@dataclass
class TaskConditionSummary:
    task_id: str
    condition: str
    model_id: str
    n_trials: int
    success_rate: float            # fraction 0..1
    avg_turns: float
    avg_wall_sec: float
    avg_ttft_last_turn_ms: float   # TTFT at the final turn (shows degradation worst case)
    avg_ttft_all_ms: float         # mean TTFT across all turns of all trials
    avg_tps: float
    avg_peak_ram_gb: float
    avg_tool_errors: float
    avg_reload_count: float
    swap_trial_count: int          # trials where any swap occurred
    avg_final_ctx_tokens: float

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 3) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}
