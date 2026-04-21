"""
Tests for autotune.context.extractor — conversation fact extraction.

Covers:
- extract_facts: accomplishments, decisions, facts, errors_seen, topics
- ConversationFacts.is_empty property
- Deduplication
- Line-length filtering (12–220 chars)
- Word-count filtering (< 4 words skipped)
- build_summary_block: standard and compact modes
- Empty/minimal conversation handling
- Topic detection across all supported tech vocabularies
"""

from __future__ import annotations

import pytest

from autotune.context.extractor import (
    ConversationFacts,
    build_summary_block,
    extract_facts,
)


# ---------------------------------------------------------------------------
# ConversationFacts
# ---------------------------------------------------------------------------

class TestConversationFacts:
    def test_is_empty_when_all_lists_empty(self):
        f = ConversationFacts()
        assert f.is_empty

    def test_not_empty_with_one_accomplishment(self):
        f = ConversationFacts(accomplishments=["completed the migration"])
        assert not f.is_empty

    def test_not_empty_with_only_topics(self):
        f = ConversationFacts(topics=["Python"])
        assert not f.is_empty

    def test_not_empty_with_errors(self):
        f = ConversationFacts(errors_seen=["sqlite3.OperationalError"])
        assert not f.is_empty


# ---------------------------------------------------------------------------
# extract_facts — accomplishments
# ---------------------------------------------------------------------------

class TestExtractAccomplishments:
    def test_done_keyword_captured(self):
        msgs = [{"role": "assistant", "content": "We are done with the database migration now."}]
        facts = extract_facts(msgs)
        assert len(facts.accomplishments) >= 1

    def test_fixed_keyword_captured(self):
        msgs = [{"role": "assistant", "content": "The authentication bug has been fixed and tests pass."}]
        facts = extract_facts(msgs)
        assert len(facts.accomplishments) >= 1

    def test_implemented_keyword_captured(self):
        msgs = [{"role": "user", "content": "I implemented the new caching layer and it works correctly."}]
        facts = extract_facts(msgs)
        assert len(facts.accomplishments) >= 1

    def test_short_line_skipped(self):
        msgs = [{"role": "assistant", "content": "done\nFixed."}]
        facts = extract_facts(msgs)
        # "done" and "Fixed." are too short (< 12 chars or < 4 words) to be captured
        assert len(facts.accomplishments) == 0

    def test_max_8_accomplishments(self):
        lines = [f"completed the task number {i} successfully without issues" for i in range(20)]
        msgs = [{"role": "assistant", "content": "\n".join(lines)}]
        facts = extract_facts(msgs)
        assert len(facts.accomplishments) <= 8


# ---------------------------------------------------------------------------
# extract_facts — decisions
# ---------------------------------------------------------------------------

class TestExtractDecisions:
    def test_using_keyword_captured(self):
        msgs = [{"role": "user", "content": "We are using PostgreSQL for the production database setup."}]
        facts = extract_facts(msgs)
        assert len(facts.decisions) >= 1

    def test_decided_keyword_captured(self):
        msgs = [{"role": "assistant", "content": "We decided to use FastAPI instead of Flask for the backend."}]
        facts = extract_facts(msgs)
        assert len(facts.decisions) >= 1

    def test_going_with_captured(self):
        msgs = [{"role": "user", "content": "Going with the SQLite approach for now since it is simpler."}]
        facts = extract_facts(msgs)
        assert len(facts.decisions) >= 1


# ---------------------------------------------------------------------------
# extract_facts — errors
# ---------------------------------------------------------------------------

class TestExtractErrors:
    def test_error_keyword_captured(self):
        msgs = [{"role": "user", "content": "Error: connection refused when connecting to database."}]
        facts = extract_facts(msgs)
        assert len(facts.errors_seen) >= 1

    def test_exception_keyword_captured(self):
        msgs = [{"role": "user", "content": "Exception raised: MemoryError during model loading process."}]
        facts = extract_facts(msgs)
        assert len(facts.errors_seen) >= 1

    def test_traceback_keyword_captured(self):
        msgs = [{"role": "user", "content": "Traceback showed ValueError on line 42 of main.py file."}]
        facts = extract_facts(msgs)
        assert len(facts.errors_seen) >= 1

    def test_http_500_captured(self):
        msgs = [{"role": "assistant", "content": "The server returns 500 when the database is unavailable."}]
        facts = extract_facts(msgs)
        assert len(facts.errors_seen) >= 1


# ---------------------------------------------------------------------------
# extract_facts — topics
# ---------------------------------------------------------------------------

class TestExtractTopics:
    @pytest.mark.parametrize("lang,text", [
        ("Python",     "We are writing this in Python and using pandas."),
        ("JavaScript", "The frontend uses JavaScript with React hooks."),
        ("TypeScript", "All code is TypeScript with strict mode enabled."),
        ("Go",         "The backend service is written in Go for performance."),
        ("Rust",       "Memory safety is guaranteed because we use Rust here."),
    ])
    def test_language_topics_detected(self, lang: str, text: str):
        msgs = [{"role": "user", "content": text}]
        facts = extract_facts(msgs)
        assert lang in facts.topics, f"Expected {lang} in topics"

    @pytest.mark.parametrize("tech,text", [
        ("Docker",      "We containerize everything with Docker and Compose."),
        ("PostgreSQL",  "Our database is PostgreSQL running on version 15."),
        ("FastAPI",     "The API server is built with FastAPI and uvicorn."),
        ("Redis",       "Session state is stored in Redis for performance."),
        ("AWS",         "We deploy to AWS using EC2 and RDS instances."),
    ])
    def test_infrastructure_topics_detected(self, tech: str, text: str):
        msgs = [{"role": "user", "content": text}]
        facts = extract_facts(msgs)
        assert tech in facts.topics, f"Expected {tech} in topics"

    def test_performance_topic_detected(self):
        msgs = [{"role": "user", "content": "We need to optimize the latency and throughput of our service."}]
        facts = extract_facts(msgs)
        assert "performance" in facts.topics

    def test_security_topic_detected(self):
        msgs = [{"role": "user", "content": "We need to add JWT auth and prevent XSS injection attacks."}]
        facts = extract_facts(msgs)
        assert "security" in facts.topics

    def test_testing_topic_detected(self):
        msgs = [{"role": "user", "content": "We need to add pytest coverage and mock the database calls."}]
        facts = extract_facts(msgs)
        assert "testing" in facts.topics

    def test_max_8_topics(self):
        text = (
            "Python JavaScript TypeScript Go Rust Docker Kubernetes AWS "
            "GCP Azure PostgreSQL Redis MongoDB FastAPI Flask Django"
        )
        msgs = [{"role": "user", "content": text}]
        facts = extract_facts(msgs)
        assert len(facts.topics) <= 8


# ---------------------------------------------------------------------------
# extract_facts — deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_identical_lines_not_duplicated(self):
        line = "We completed the database migration successfully today."
        msgs = [{"role": "assistant", "content": "\n".join([line] * 10)}]
        facts = extract_facts(msgs)
        assert len(facts.accomplishments) == 1

    def test_near_duplicate_lines_deduplicated(self):
        # First 60 chars are the same → dedup key matches
        line1 = "We completed the database migration successfully this morning at 9am."
        line2 = "We completed the database migration successfully this morning at 10am."
        msgs = [{"role": "assistant", "content": f"{line1}\n{line2}"}]
        facts = extract_facts(msgs)
        # Both lines start with same 60 chars → only first kept
        assert len(facts.accomplishments) <= 1


# ---------------------------------------------------------------------------
# extract_facts — line filtering
# ---------------------------------------------------------------------------

class TestLineFiltering:
    def test_line_under_12_chars_skipped(self):
        msgs = [{"role": "assistant", "content": "done.\nerror."}]
        facts = extract_facts(msgs)
        assert facts.is_empty

    def test_line_over_220_chars_skipped(self):
        long_line = "Error: " + "x" * 220  # 227 chars
        msgs = [{"role": "user", "content": long_line}]
        facts = extract_facts(msgs)
        assert len(facts.errors_seen) == 0

    def test_line_under_4_words_skipped(self):
        msgs = [{"role": "user", "content": "Error occurred."}]  # 2 words
        facts = extract_facts(msgs)
        assert len(facts.errors_seen) == 0

    def test_skip_line_patterns_excluded(self):
        msgs = [{"role": "user", "content": "ok\nunderstood\nwill do\nyes"}]
        facts = extract_facts(msgs)
        assert facts.is_empty

    def test_empty_messages_ignored(self):
        msgs = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": None},
        ]
        facts = extract_facts(msgs)
        assert facts.is_empty


# ---------------------------------------------------------------------------
# build_summary_block — standard mode
# ---------------------------------------------------------------------------

class TestBuildSummaryBlockStandard:
    def test_returns_string(self):
        msgs = [{"role": "user", "content": "We are using Python for this project setup."}]
        result = build_summary_block(msgs)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_turn_count(self):
        msgs = [
            {"role": "user", "content": "First question about Python setup here."},
            {"role": "assistant", "content": "Answer: Python is configured and ready."},
        ]
        result = build_summary_block(msgs)
        assert "2" in result  # 2 turns

    def test_accomplishments_section_when_present(self):
        msgs = [{"role": "assistant", "content": "The migration has been completed and deployed successfully."}]
        result = build_summary_block(msgs)
        # Check block header markers are present
        assert "[CONVERSATION HISTORY SUMMARY" in result
        assert "[END OF HISTORY SUMMARY" in result

    def test_empty_conversation_valid_output(self):
        result = build_summary_block([])
        assert isinstance(result, str)

    def test_uses_provided_facts(self):
        facts = ConversationFacts(topics=["Python", "Docker"])
        result = build_summary_block([], facts=facts)
        assert "Python" in result
        assert "Docker" in result

    def test_errors_section_when_present(self):
        facts = ConversationFacts(errors_seen=["sqlite3.OperationalError on startup"])
        result = build_summary_block([], facts=facts)
        assert "sqlite3.OperationalError" in result

    def test_decisions_section_when_present(self):
        facts = ConversationFacts(decisions=["Using PostgreSQL for the production setup"])
        result = build_summary_block([], facts=facts)
        assert "PostgreSQL" in result


# ---------------------------------------------------------------------------
# build_summary_block — compact mode
# ---------------------------------------------------------------------------

class TestBuildSummaryBlockCompact:
    def test_compact_shorter_than_standard(self):
        msgs = [
            {"role": "user", "content": "We are using Python with FastAPI for this backend project."},
            {"role": "assistant", "content": "The setup is complete and migrations have been done."},
        ]
        standard = build_summary_block(msgs, compact=False)
        compact  = build_summary_block(msgs, compact=True)
        assert len(compact) <= len(standard)

    def test_compact_starts_with_context_tag(self):
        result = build_summary_block([], compact=True)
        assert result.startswith("[CONTEXT:")

    def test_compact_ends_with_closing_tag(self):
        result = build_summary_block([], compact=True)
        assert "[/CONTEXT]" in result

    def test_compact_shows_turn_count(self):
        msgs = [
            {"role": "user", "content": "Hello, how do I set up Python virtualenv?"},
            {"role": "assistant", "content": "Run python -m venv .venv to create it."},
        ]
        result = build_summary_block(msgs, compact=True)
        assert "2" in result

    def test_compact_topics_on_one_line(self):
        facts = ConversationFacts(topics=["Python", "Docker", "AWS"])
        result = build_summary_block([], facts=facts, compact=True)
        line_with_topics = [l for l in result.split("\n") if "Topics:" in l]
        assert len(line_with_topics) == 1
        assert "Python" in line_with_topics[0]
