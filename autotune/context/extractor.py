"""
Conversation fact extractor — builds a structured summary block from
older turns that will be compressed out of the active context window.

The summary is injected as a system message so the model understands:
  - What has been accomplished so far
  - What decisions / constraints are in force
  - What key facts have been established
  - What topics have been discussed

This is deterministic (no LLM call required) so it adds zero latency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConversationFacts:
    """Structured knowledge extracted from older conversation turns."""
    accomplishments: list[str] = field(default_factory=list)
    decisions:       list[str] = field(default_factory=list)
    facts:           list[str] = field(default_factory=list)
    errors_seen:     list[str] = field(default_factory=list)
    topics:          list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any([
            self.accomplishments, self.decisions,
            self.facts, self.errors_seen, self.topics,
        ])


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Phrases that signal completion / accomplishment
_DONE_RE = re.compile(
    r"\b(done|completed|fixed|added|implemented|created|updated|changed|"
    r"removed|resolved|solved|deployed|installed|now works?|working|"
    r"migrated|refactored|passed|success)\b",
    re.IGNORECASE,
)

# Phrases that signal a decision or constraint
_DECISION_RE = re.compile(
    r"\b(decided|chose|using|we('re| are| will)|going with|"
    r"the approach is|the solution is|let'?s use|I will|we need to|"
    r"we should|we must|the plan is|agreed to|switching to)\b",
    re.IGNORECASE,
)

# Phrases that signal a factual statement worth keeping
_FACT_RE = re.compile(
    r"\b(is|are|was|were|has|have|equals?|means?|defined as|"
    r"configured as|set to|returns?|provides?|requires?|"
    r"takes?|accepts?|produces?|stores?)\b",
    re.IGNORECASE,
)

# Error / exception signals
_ERROR_RE = re.compile(
    r"\b(Error|Exception|Traceback|failed|failure|crash|OOM|"
    r"MemoryError|TimeoutError|404|500|503|SIGKILL)\b",
    re.IGNORECASE,
)

# Specific identifiers that make a line more "factual"
_IDENTIFIER_RE = re.compile(r"[A-Z][a-zA-Z0-9_]{3,}|[a-z_][a-z0-9_]{3,}=[^\s,;]+")

# Topic vocabulary
_TOPIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(Python|JavaScript|TypeScript|Go|Rust|Java|C\+\+|C#|Swift|Kotlin)\b"), "{}"),
    (re.compile(r"\b(Docker|Kubernetes|k8s|Helm|Terraform|Ansible)\b"), "{}"),
    (re.compile(r"\b(AWS|GCP|Azure|Vercel|Fly\.io|Railway)\b"), "{}"),
    (re.compile(r"\b(PostgreSQL|MySQL|SQLite|MongoDB|Redis|Elasticsearch)\b"), "{}"),
    (re.compile(r"\b(FastAPI|Flask|Django|Express|Next\.?[Jj]s|React|Vue|Angular)\b"), "{}"),
    (re.compile(r"\b(REST|GraphQL|gRPC|WebSocket|HTTP|HTTPS)\b"), "{}"),
    (re.compile(r"\b(LLM|transformer|embedding|fine-?tun|RAG|vector)\b", re.I), "{}"),
    (re.compile(r"\b(performance|latency|throughput|memory|optimization)\b", re.I), "performance"),
    (re.compile(r"\b(security|auth|JWT|OAuth|CSRF|XSS|injection)\b", re.I), "security"),
    (re.compile(r"\b(test|unittest|pytest|mock|coverage|CI/CD)\b", re.I), "testing"),
]

# Lines to skip entirely
_SKIP_LINE_RE = re.compile(
    r"^(ok|okay|sure|thanks|noted|done|yes|no|I see|understood|"
    r"great|perfect|sounds good|will do)\W*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_facts(messages: list[dict]) -> ConversationFacts:
    """
    Scan a list of messages and return structured facts.

    Only looks at lines that are likely to carry permanent information:
    specific values, identifiers, measured quantities, decisions.
    """
    facts = ConversationFacts()
    seen_keys: set[str] = set()

    def _add(bucket: list[str], line: str) -> None:
        """Deduplicated append."""
        key = line[:60].lower().strip()
        if key and key not in seen_keys and len(bucket) < 8:
            seen_keys.add(key)
            bucket.append(line.strip())

    for msg in messages:
        content: str = msg.get("content", "") or ""
        role: str = msg.get("role", "")

        for raw_line in content.splitlines():
            line = raw_line.strip()

            # Skip empty, very short, or pure chatter lines
            if len(line) < 12 or len(line) > 220:
                continue
            if _SKIP_LINE_RE.match(line):
                continue
            if len(line.split()) < 4:
                continue

            # Categorise
            if _ERROR_RE.search(line):
                _add(facts.errors_seen, line)
            elif _DONE_RE.search(line):
                _add(facts.accomplishments, line)
            elif _DECISION_RE.search(line):
                _add(facts.decisions, line)
            elif _FACT_RE.search(line) and _IDENTIFIER_RE.search(line):
                _add(facts.facts, line)

        # Topic extraction (whole message)
        for pattern, fmt in _TOPIC_PATTERNS:
            for m in pattern.finditer(content):
                topic = fmt.format(m.group(0)) if "{}" in fmt else fmt
                if topic and topic not in facts.topics and len(facts.topics) < 8:
                    facts.topics.append(topic)

    return facts


# ---------------------------------------------------------------------------
# Summary block builder
# ---------------------------------------------------------------------------

def build_summary_block(
    messages: list[dict],
    facts: Optional[ConversationFacts] = None,
    compact: bool = False,
) -> str:
    """
    Build a compact summary block to inject in place of older messages.

    The block is injected as a system message so the model treats it as
    authoritative context rather than something a user said.

    Parameters
    ----------
    messages : the older turns being replaced
    facts    : pre-extracted facts (computed if not provided)
    compact  : True for EMERGENCY tier — ultra-short one-liners
    """
    if facts is None:
        facts = extract_facts(messages)

    # Count total turns for the header
    turn_count = len([m for m in messages if m.get("role") in ("user", "assistant")])

    if compact:
        # Emergency format: everything on one line per category
        parts = [f"[CONTEXT: {turn_count} earlier turns summarized]"]
        if facts.accomplishments:
            parts.append("Done: " + " | ".join(facts.accomplishments[:3]))
        if facts.decisions:
            parts.append("Using: " + " | ".join(facts.decisions[:3]))
        if facts.facts:
            parts.append("Facts: " + " | ".join(facts.facts[:3]))
        if facts.errors_seen:
            parts.append("Errors seen: " + " | ".join(facts.errors_seen[:2]))
        if facts.topics:
            parts.append("Topics: " + ", ".join(facts.topics))
        parts.append("[/CONTEXT]")
        return "\n".join(parts)

    # Standard format
    lines = [
        f"[CONVERSATION HISTORY SUMMARY — {turn_count} earlier turns]",
        "The following summarizes the earlier part of this conversation.",
        "",
    ]

    if facts.accomplishments:
        lines.append("✓ Accomplished so far:")
        for item in facts.accomplishments:
            lines.append(f"  • {item}")
        lines.append("")

    if facts.decisions:
        lines.append("⚙ Active decisions / constraints:")
        for item in facts.decisions:
            lines.append(f"  • {item}")
        lines.append("")

    if facts.facts:
        lines.append("📌 Key facts established:")
        for item in facts.facts:
            lines.append(f"  • {item}")
        lines.append("")

    if facts.errors_seen:
        lines.append("⚠ Errors / issues encountered:")
        for item in facts.errors_seen:
            lines.append(f"  • {item}")
        lines.append("")

    if facts.topics:
        lines.append(f"💬 Topics covered: {', '.join(facts.topics)}")
        lines.append("")

    lines.append("[END OF HISTORY SUMMARY — current conversation continues below]")
    return "\n".join(lines)
