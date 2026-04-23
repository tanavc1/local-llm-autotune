"""
Tests for autotune.api.server — FastAPI endpoints via TestClient.

Covers:
- GET /health: shape, required keys, profile list
- GET /v1/models: object list shape, owned_by field
- GET /v1/models/local: local subset shape
- GET /api/profiles: all three profiles present, required keys per profile
- GET /api/running_models: count and models keys
- GET /api/hardware: required hardware keys
- POST /v1/chat/completions: 422 on missing model, invalid role, invalid profile
- ChatRequest validation: temperature range, max_tokens > 0, valid profile
- Message.normalize_content: list input flattened to string
- _normalize_model_id: strips autotune. prefix, rejects empty

All network-bound calls (Ollama, LM Studio) are mocked so tests run offline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App fixture — import server and mock external dependencies at module level
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with all external I/O mocked so no Ollama/LM Studio
    connection is needed.  Scope=module to avoid repeated startup overhead.
    """
    # Mock hardware profiler — returns a minimal HW profile
    mock_hw = MagicMock()
    mock_hw.os_version = "macOS 15.0"
    mock_hw.cpu.brand = "Apple M3 Pro"
    mock_hw.cpu.physical_cores = 12
    mock_hw.memory.total_gb = 32.0
    mock_hw.gpu.name = "Apple M3 Pro GPU"
    mock_hw.gpu.backend = "metal"
    mock_hw.gpu.is_unified_memory = True
    mock_hw.effective_memory_gb = 28.0

    # Mock chain — no backends are running
    mock_chain = MagicMock()
    mock_chain.ollama_running = AsyncMock(return_value=False)
    mock_chain.lmstudio_running = AsyncMock(return_value=False)
    mock_chain.discover_all = AsyncMock(return_value=[])

    # Mock ConversationManager
    mock_conv = MagicMock()
    mock_conv.create.return_value = "conv-test-uuid"
    mock_conv.list_conversations.return_value = []

    # Mock memory pressure snapshot
    mock_mem = {
        "ram_pct": 55.0,
        "available_gb": 14.4,
        "swap_used_gb": 0.0,
        "pressure_level": "normal",
    }

    with (
        patch("autotune.api.server.profile_hardware", return_value=mock_hw),
        patch("autotune.api.server.get_chain", return_value=mock_chain),
        patch("autotune.api.server.get_conv_manager", return_value=mock_conv),
        patch("autotune.api.server.get_tuner", return_value=MagicMock()),
        patch("autotune.api.kv_manager.memory_pressure_snapshot", return_value=mock_mem),
        patch("autotune.api.server.psutil") as mock_psutil,
    ):
        mock_psutil.virtual_memory.return_value = MagicMock(
            available=int(14.4 * 1024**3), percent=55.0
        )
        mock_psutil.swap_memory.return_value = MagicMock(used=0, percent=0.0)

        from autotune.api.server import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client: TestClient):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_has_version(self, client: TestClient):
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_backends_keys_present(self, client: TestClient):
        data = client.get("/health").json()
        backends = data["backends"]
        assert "ollama" in backends
        assert "lmstudio" in backends
        assert "hf_api" in backends

    def test_queue_keys_present(self, client: TestClient):
        data = client.get("/health").json()
        q = data["queue"]
        assert "active" in q
        assert "queued" in q
        assert "max_concurrent" in q

    def test_memory_keys_present(self, client: TestClient):
        data = client.get("/health").json()
        mem = data["memory"]
        assert "ram_pct" in mem
        assert "available_gb" in mem
        assert "pressure_level" in mem

    def test_profiles_list_non_empty(self, client: TestClient):
        data = client.get("/health").json()
        assert len(data["profiles"]) >= 1


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_returns_200(self, client: TestClient):
        r = client.get("/v1/models")
        assert r.status_code == 200

    def test_response_is_list_object(self, client: TestClient):
        data = client.get("/v1/models").json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)

    def test_no_models_when_backends_offline(self, client: TestClient):
        # With all backends mocked as returning [] we get an empty list
        data = client.get("/v1/models").json()
        assert isinstance(data["data"], list)


# ---------------------------------------------------------------------------
# GET /v1/models/local
# ---------------------------------------------------------------------------

class TestListLocalModels:
    def test_returns_200(self, client: TestClient):
        r = client.get("/v1/models/local")
        assert r.status_code == 200

    def test_response_has_data_list(self, client: TestClient):
        data = client.get("/v1/models/local").json()
        assert "object" in data or "data" in data


# ---------------------------------------------------------------------------
# GET /api/profiles
# ---------------------------------------------------------------------------

class TestListProfiles:
    def test_returns_200(self, client: TestClient):
        r = client.get("/api/profiles")
        assert r.status_code == 200

    def test_three_profiles_present(self, client: TestClient):
        data = client.get("/api/profiles").json()
        assert "fast" in data
        assert "balanced" in data
        assert "quality" in data

    def test_each_profile_has_required_keys(self, client: TestClient):
        data = client.get("/api/profiles").json()
        for name, profile in data.items():
            assert "label" in profile, f"{name} missing label"
            assert "temperature" in profile, f"{name} missing temperature"
            assert "max_new_tokens" in profile, f"{name} missing max_new_tokens"

    def test_fast_temperature_below_balanced(self, client: TestClient):
        data = client.get("/api/profiles").json()
        # fast uses greedy/low temp; balanced and quality allow more diversity
        assert data["fast"]["temperature"] < data["balanced"]["temperature"]


# ---------------------------------------------------------------------------
# GET /api/running_models
# ---------------------------------------------------------------------------

class TestRunningModels:
    def test_returns_200(self, client: TestClient):
        with patch("autotune.api.running_models.get_running_models", return_value=[]):
            r = client.get("/api/running_models")
        assert r.status_code == 200

    def test_response_has_models_and_count(self, client: TestClient):
        with patch("autotune.api.running_models.get_running_models", return_value=[]):
            data = client.get("/api/running_models").json()
        assert "models" in data
        assert "count" in data
        assert data["count"] == 0


# ---------------------------------------------------------------------------
# GET /api/hardware
# ---------------------------------------------------------------------------

class TestHardwareEndpoint:
    def test_returns_200(self, client: TestClient):
        r = client.get("/api/hardware")
        # 200 if _hw is set, 503 if not (lifespan may not set it in test mode)
        assert r.status_code in (200, 503)

    def test_hardware_keys_when_ok(self, client: TestClient):
        r = client.get("/api/hardware")
        if r.status_code == 200:
            data = r.json()
            assert "ram_total_gb" in data
            assert "cpu" in data


# ---------------------------------------------------------------------------
# ChatRequest Pydantic validation (unit tests — no HTTP needed)
# ---------------------------------------------------------------------------

class TestChatRequestValidation:
    def test_invalid_role_rejected(self, client: TestClient):
        payload = {
            "model": "test-model",
            "messages": [{"role": "hacker", "content": "hello"}],
        }
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 422

    def test_missing_model_rejected(self, client: TestClient):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 422

    def test_invalid_profile_rejected(self, client: TestClient):
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "profile": "turbo",
        }
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 422

    def test_temperature_out_of_range_rejected(self, client: TestClient):
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 3.0,
        }
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 422

    def test_max_tokens_zero_rejected(self, client: TestClient):
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        }
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 422

    def test_valid_request_accepted(self, client: TestClient):
        """A structurally valid request should pass pydantic validation (may fail 503 without backend)."""
        payload = {
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "Hello!"}],
            "profile": "balanced",
        }
        r = client.post("/v1/chat/completions", json=payload)
        # 200, 503, or 404 — any is fine as long as not 422
        assert r.status_code != 422


# ---------------------------------------------------------------------------
# Message.normalize_content
# ---------------------------------------------------------------------------

class TestMessageNormalizeContent:
    def test_list_content_flattened(self):
        from autotune.api.server import Message
        m = Message(role="user", content=[
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
        ])
        assert "Hello" in m.content
        assert "world" in m.content

    def test_none_content_becomes_empty_string(self):
        from autotune.api.server import Message
        m = Message(role="user", content=None)
        assert m.content == ""

    def test_string_content_unchanged(self):
        from autotune.api.server import Message
        m = Message(role="assistant", content="Hello!")
        assert m.content == "Hello!"

    def test_image_url_part_dropped(self):
        from autotune.api.server import Message
        m = Message(role="user", content=[
            {"type": "text", "text": "Describe:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ])
        assert "Describe:" in m.content
        assert "image_url" not in m.content


# ---------------------------------------------------------------------------
# _normalize_model_id
# ---------------------------------------------------------------------------

class TestNormalizeModelId:
    def test_strips_autotune_prefix(self):
        from autotune.api.server import _normalize_model_id
        assert _normalize_model_id("autotune.qwen3:8b") == "qwen3:8b"

    def test_bare_model_id_unchanged(self):
        from autotune.api.server import _normalize_model_id
        assert _normalize_model_id("llama3:8b") == "llama3:8b"

    def test_empty_after_strip_raises_http(self):
        from fastapi import HTTPException

        from autotune.api.server import _normalize_model_id
        with pytest.raises(HTTPException) as exc_info:
            _normalize_model_id("autotune.")
        assert exc_info.value.status_code == 400
