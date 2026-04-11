"""Tests for autotune.recall.store — RecallStore CRUD and search."""

import time
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autotune.recall.store import RecallStore, MemoryChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Fresh RecallStore backed by a temp file."""
    return RecallStore(tmp_path / "recall.db")


def _vec(values: list[float]) -> bytes:
    """Pack a list of floats into float32 bytes."""
    return np.array(values, dtype=np.float32).tobytes()


def _unit_vec(size: int, hot_index: int) -> bytes:
    """One-hot float32 vector of given size."""
    v = [0.0] * size
    v[hot_index] = 1.0
    return _vec(v)


# ---------------------------------------------------------------------------
# save / is_saved / mark_saved
# ---------------------------------------------------------------------------

class TestSaveAndMark:
    def test_save_returns_count(self, store):
        chunks = [
            {"text": "User: hello\nAssistant: world answer here", "created_at": time.time(), "turn_start": 0, "turn_end": 1},
        ]
        n = store.save("conv1", "qwen3:8b", chunks)
        assert n == 1

    def test_save_multiple_chunks(self, store):
        chunks = [
            {"text": f"User: q{i}\nAssistant: a{i} answer", "created_at": time.time(), "turn_start": i*2, "turn_end": i*2+1}
            for i in range(3)
        ]
        n = store.save("conv2", "llama3", chunks)
        assert n == 3

    def test_is_saved_false_initially(self, store):
        assert store.is_saved("conv_new") is False

    def test_mark_saved_sets_flag(self, store):
        store.mark_saved("conv3")
        assert store.is_saved("conv3") is True

    def test_mark_saved_idempotent(self, store):
        store.mark_saved("conv4")
        store.mark_saved("conv4")
        assert store.is_saved("conv4") is True


# ---------------------------------------------------------------------------
# get_recent
# ---------------------------------------------------------------------------

class TestGetRecent:
    def test_returns_empty_for_fresh_store(self, store):
        assert store.get_recent() == []

    def test_returns_chunks_newest_first(self, store):
        now = time.time()
        store.save("conv1", "model_a", [
            {"text": "User: old\nAssistant: old answer here", "created_at": now - 3600, "turn_start": 0, "turn_end": 1},
        ])
        store.save("conv2", "model_b", [
            {"text": "User: new\nAssistant: new answer here", "created_at": now, "turn_start": 0, "turn_end": 1},
        ])
        chunks = store.get_recent(limit=10)
        assert len(chunks) == 2
        assert chunks[0].created_at >= chunks[1].created_at

    def test_respects_limit(self, store):
        now = time.time()
        for i in range(10):
            store.save(f"conv{i}", "model", [
                {"text": f"User: q{i}\nAssistant: a{i} answer", "created_at": now + i, "turn_start": 0, "turn_end": 1},
            ])
        chunks = store.get_recent(limit=3)
        assert len(chunks) == 3

    def test_filter_by_model(self, store):
        now = time.time()
        store.save("c1", "model_a", [{"text": "User: x\nAssistant: y answer here", "created_at": now, "turn_start": 0, "turn_end": 1}])
        store.save("c2", "model_b", [{"text": "User: x\nAssistant: y answer here", "created_at": now, "turn_start": 0, "turn_end": 1}])
        chunks = store.get_recent(model_id="model_a")
        assert all(c.model_id == "model_a" for c in chunks)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# delete operations
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_single_chunk(self, store):
        n = store.save("c1", "m", [
            {"text": "User: del\nAssistant: del answer here", "created_at": time.time(), "turn_start": 0, "turn_end": 1}
        ])
        chunks = store.get_recent()
        assert len(chunks) == 1
        deleted = store.delete(chunks[0].id)
        assert deleted is True
        assert store.get_recent() == []

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete(99999) is False

    def test_delete_conversation(self, store):
        store.save("c1", "m", [
            {"text": "User: q1\nAssistant: a1 answer here", "created_at": time.time(), "turn_start": 0, "turn_end": 1},
            {"text": "User: q2\nAssistant: a2 answer here", "created_at": time.time(), "turn_start": 2, "turn_end": 3},
        ])
        store.save("c2", "m", [
            {"text": "User: q3\nAssistant: a3 answer here", "created_at": time.time(), "turn_start": 0, "turn_end": 1},
        ])
        n = store.delete_conversation("c1")
        assert n == 2
        remaining = store.get_recent()
        assert all(c.conv_id != "c1" for c in remaining)

    def test_delete_all(self, store):
        store.save("c1", "m", [{"text": "User: x\nAssistant: y answer", "created_at": time.time(), "turn_start": 0, "turn_end": 1}])
        store.save("c2", "m", [{"text": "User: x\nAssistant: y answer", "created_at": time.time(), "turn_start": 0, "turn_end": 1}])
        store.delete_all()
        assert store.get_recent() == []


# ---------------------------------------------------------------------------
# search_fts
# ---------------------------------------------------------------------------

class TestSearchFts:
    def test_finds_matching_chunk(self, store):
        store.save("c1", "m", [
            {"text": "User: What is recursion?\nAssistant: Recursion is a function calling itself repeatedly.", "created_at": time.time(), "turn_start": 0, "turn_end": 1},
        ])
        results = store.search_fts("recursion")
        assert len(results) >= 1
        assert "recursion" in results[0].chunk_text.lower()

    def test_returns_empty_for_no_match(self, store):
        store.save("c1", "m", [
            {"text": "User: Hello\nAssistant: Hi there how are you doing today!", "created_at": time.time(), "turn_start": 0, "turn_end": 1},
        ])
        results = store.search_fts("xyzzy_nonexistent_word")
        assert results == []

    def test_respects_top_k(self, store):
        for i in range(5):
            store.save(f"c{i}", "m", [
                {"text": f"User: python question {i}\nAssistant: python answer {i} here in detail for testing.", "created_at": time.time() + i, "turn_start": 0, "turn_end": 1},
            ])
        results = store.search_fts("python", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# search_vector
# ---------------------------------------------------------------------------

class TestSearchVector:
    def test_finds_similar_vector(self, store):
        vec = _unit_vec(8, 0)
        store.save("c1", "m", [
            {
                "text": "User: topic A\nAssistant: detailed answer about topic A here.",
                "created_at": time.time(),
                "turn_start": 0, "turn_end": 1,
                "embedding": vec,
                "embedding_model": "test",
            }
        ])
        results = store.search_vector(vec, "test", top_k=5, min_score=0.9)
        assert len(results) == 1
        assert abs(results[0].score - 1.0) < 0.001

    def test_orthogonal_vectors_not_returned(self, store):
        vec_a = _unit_vec(8, 0)
        vec_b = _unit_vec(8, 1)
        store.save("c1", "m", [
            {
                "text": "User: topic A\nAssistant: detailed answer about A.",
                "created_at": time.time(),
                "turn_start": 0, "turn_end": 1,
                "embedding": vec_a,
                "embedding_model": "test",
            }
        ])
        results = store.search_vector(vec_b, "test", top_k=5, min_score=0.3)
        assert results == []

    def test_respects_min_score(self, store):
        vec_a = _unit_vec(8, 0)
        vec_b = _unit_vec(8, 1)
        # Diagonal vector has 0.707 cosine similarity to both unit vectors
        diag = _vec([1.0, 1.0, 0, 0, 0, 0, 0, 0])
        store.save("c1", "m", [
            {
                "text": "User: diagonal\nAssistant: diagonal answer in detail here.",
                "created_at": time.time(),
                "turn_start": 0, "turn_end": 1,
                "embedding": diag,
                "embedding_model": "test",
            }
        ])
        high_thresh = store.search_vector(vec_a, "test", top_k=5, min_score=0.8)
        low_thresh = store.search_vector(vec_a, "test", top_k=5, min_score=0.3)
        assert len(high_thresh) == 0
        assert len(low_thresh) == 1

    def test_empty_store_returns_empty(self, store):
        vec = _unit_vec(8, 0)
        results = store.search_vector(vec, "test", top_k=5, min_score=0.1)
        assert results == []

    def test_respects_embedding_model_filter(self, store):
        vec = _unit_vec(8, 0)
        store.save("c1", "m", [
            {
                "text": "User: x\nAssistant: y detailed answer about x for testing.",
                "created_at": time.time(),
                "turn_start": 0, "turn_end": 1,
                "embedding": vec,
                "embedding_model": "model_a",
            }
        ])
        # Query with different embedding model → no results
        results = store.search_vector(vec, "model_b", top_k=5, min_score=0.1)
        assert results == []


# ---------------------------------------------------------------------------
# list_conversations
# ---------------------------------------------------------------------------

class TestListConversations:
    def test_returns_empty_for_fresh_store(self, store):
        assert store.list_conversations() == []

    def test_groups_by_conv_id(self, store):
        now = time.time()
        store.save("conv1", "m", [
            {"text": "User: q1\nAssistant: a1 answer here", "created_at": now, "turn_start": 0, "turn_end": 1},
            {"text": "User: q2\nAssistant: a2 answer here", "created_at": now + 1, "turn_start": 2, "turn_end": 3},
        ])
        store.save("conv2", "m", [
            {"text": "User: q3\nAssistant: a3 answer here", "created_at": now + 2, "turn_start": 0, "turn_end": 1},
        ])
        rows = store.list_conversations()
        assert len(rows) == 2

    def test_sorted_by_most_recent(self, store):
        now = time.time()
        store.save("old_conv", "m", [
            {"text": "User: old\nAssistant: old answer here", "created_at": now - 3600, "turn_start": 0, "turn_end": 1},
        ])
        store.save("new_conv", "m", [
            {"text": "User: new\nAssistant: new answer here", "created_at": now, "turn_start": 0, "turn_end": 1},
        ])
        rows = store.list_conversations()
        assert rows[0]["conv_id"] == "new_conv"

    def test_row_has_required_keys(self, store):
        now = time.time()
        store.save("c1", "qwen3:8b", [
            {"text": "User: hello\nAssistant: hello back to you!", "created_at": now, "turn_start": 0, "turn_end": 1},
        ])
        rows = store.list_conversations()
        row = rows[0]
        for key in ("conv_id", "model_id", "first_at", "last_at", "chunk_count", "sample_text"):
            assert key in row

    def test_chunk_count_correct(self, store):
        now = time.time()
        store.save("c1", "m", [
            {"text": "User: q1\nAssistant: a1 answer", "created_at": now, "turn_start": 0, "turn_end": 1},
            {"text": "User: q2\nAssistant: a2 answer", "created_at": now + 1, "turn_start": 2, "turn_end": 3},
            {"text": "User: q3\nAssistant: a3 answer", "created_at": now + 2, "turn_start": 4, "turn_end": 5},
        ])
        rows = store.list_conversations()
        assert rows[0]["chunk_count"] == 3

    def test_respects_limit(self, store):
        now = time.time()
        for i in range(5):
            store.save(f"c{i}", "m", [
                {"text": f"User: q{i}\nAssistant: a{i} answer here", "created_at": now + i, "turn_start": 0, "turn_end": 1},
            ])
        rows = store.list_conversations(limit=3)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_store_stats(self, store):
        s = store.stats()
        assert s["total_chunks"] == 0
        assert s["with_embeddings"] == 0

    def test_counts_chunks(self, store):
        now = time.time()
        store.save("c1", "m", [
            {"text": "User: q\nAssistant: a answer", "created_at": now, "turn_start": 0, "turn_end": 1},
            {"text": "User: q2\nAssistant: a2 answer", "created_at": now + 1, "turn_start": 2, "turn_end": 3},
        ])
        s = store.stats()
        assert s["total_chunks"] == 2


# ---------------------------------------------------------------------------
# MemoryChunk.age_str
# ---------------------------------------------------------------------------

class TestMemoryChunkAgeStr:
    def _make_chunk(self, age_sec: float) -> MemoryChunk:
        return MemoryChunk(
            id=1, conv_id="c", created_at=time.time() - age_sec,
            model_id=None, chunk_text="x", turn_start=0, turn_end=0,
        )

    def test_just_now(self):
        assert self._make_chunk(10).age_str == "just now"

    def test_minutes(self):
        age = self._make_chunk(120).age_str
        assert "min" in age

    def test_hours(self):
        age = self._make_chunk(7200).age_str
        assert "hr" in age

    def test_days(self):
        age = self._make_chunk(3 * 86400).age_str
        assert "days" in age

    def test_weeks(self):
        age = self._make_chunk(14 * 86400).age_str
        assert "week" in age

    def test_months(self):
        age = self._make_chunk(60 * 86400).age_str
        assert "month" in age
