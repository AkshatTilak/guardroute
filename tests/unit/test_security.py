"""Unit tests for Gateway security features.

These tests cover:
1. Filename sanitization (path traversal prevention)
2. API key auth toggle (AUTH_ENABLED flag)
3. Request body size limiting (413 enforcement)
4. Rate limiting middleware presence
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from fastapi.testclient import TestClient

from projects.syntraflow.api import sanitize_filename
from common.config.settings import settings


# ---------------------------------------------------------------------------
# 1. Filename sanitization (pure unit test, no I/O needed)
# ---------------------------------------------------------------------------

def test_filename_sanitization_normal():
    """Standard filenames are preserved."""
    assert sanitize_filename("document.pdf") == "document.pdf"
    assert sanitize_filename("image_123.PNG") == "image_123.PNG"
    assert sanitize_filename("my-file.mp4") == "my-file.mp4"


def test_filename_sanitization_path_traversal():
    """Path traversal sequences are stripped."""
    assert ".." not in sanitize_filename("../../../etc/passwd")
    assert "/" not in sanitize_filename("subfolder/file.txt")
    assert "\\" not in sanitize_filename("C:\\Windows\\win.ini")


def test_filename_sanitization_special_chars():
    """Special shell-sensitive characters are replaced with underscores."""
    sanitized = sanitize_filename("my file!@#$.pdf")
    assert "!" not in sanitized
    assert "@" not in sanitized
    assert "$" not in sanitized
    # Extension should still be there
    assert ".pdf" in sanitized


# ---------------------------------------------------------------------------
# 2. verify_api_key dependency unit test (mocked DB)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_api_key_auth_disabled():
    """When AUTH_ENABLED=False the dependency returns without checking the DB."""
    from gateway.api import verify_api_key

    original = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    try:
        # Should return None without touching DB
        result = await verify_api_key(x_api_key=None)
        assert result is None
    finally:
        settings.AUTH_ENABLED = original


@pytest.mark.asyncio
async def test_verify_api_key_missing_header():
    """When AUTH_ENABLED=True and header is absent, 401 is raised."""
    from fastapi import HTTPException
    from gateway.api import verify_api_key

    original = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = True
    try:
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(x_api_key=None)
        assert exc_info.value.status_code == 401
        assert "Missing X-API-Key" in exc_info.value.detail
    finally:
        settings.AUTH_ENABLED = original


@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    """When AUTH_ENABLED=True and key is not in DB, 401 is raised."""
    from fastapi import HTTPException
    from gateway.api import verify_api_key

    original = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = True

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_factory = MagicMock(return_value=mock_session)

    try:
        with patch("gateway.api.get_sessionmaker", return_value=mock_factory):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(x_api_key="bad-key")
            assert exc_info.value.status_code == 401
            assert "Invalid X-API-Key" in exc_info.value.detail
    finally:
        settings.AUTH_ENABLED = original


@pytest.mark.asyncio
async def test_verify_api_key_valid():
    """When AUTH_ENABLED=True and key exists in DB, dependency passes."""
    from gateway.api import verify_api_key
    from common.models.database import APIKeyModel

    original = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = True

    mock_key = MagicMock(spec=APIKeyModel)
    mock_key.key = "sk_live_default_key"
    mock_key.is_active = True

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_key
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_factory = MagicMock(return_value=mock_session)

    try:
        with patch("gateway.api.get_sessionmaker", return_value=mock_factory):
            # Should not raise
            result = await verify_api_key(x_api_key="sk_live_default_key")
            assert result is None
    finally:
        settings.AUTH_ENABLED = original


# ---------------------------------------------------------------------------
# 3. RequestSizeLimitMiddleware unit test (isolated ASGI app)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_size_limit_blocks_large_json():
    """Requests exceeding MAX_JSON_SIZE are rejected with 413."""
    from gateway.main import RequestSizeLimitMiddleware
    from fastapi import FastAPI

    small_app = FastAPI()

    @small_app.post("/echo")
    async def echo(body: dict):
        return {"ok": True}

    # Wrap with 100-byte limit
    protected = RequestSizeLimitMiddleware(small_app, max_upload_size=10_000, max_json_size=100)

    async with AsyncClient(transport=ASGITransport(app=protected), base_url="http://test") as client:
        # Small body -> should pass
        response = await client.post("/echo", json={"prompt": "hi"})
        assert response.status_code == 200

        # Large body -> 413
        large_body = {"prompt": "A" * 200}
        response = await client.post("/echo", json=large_body)
        assert response.status_code == 413
        assert "Request body too large" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 4. Rate limiting sanity check
# ---------------------------------------------------------------------------

def test_limiter_is_configured():
    """The shared limiter object is initialized from common.observability.limiter."""
    from common.observability.limiter import limiter
    assert limiter is not None
    assert hasattr(limiter, "_storage")


# ---------------------------------------------------------------------------
# 5. Log Redactor tests
# ---------------------------------------------------------------------------

def test_log_redactor_scrubbing():
    """Verify that scrub_sensitive_data redacts DB credentials, PII and generic secrets."""
    from common.observability.logger import scrub_sensitive_data

    # Test DB credentials
    db_log = "Connection established to postgresql+asyncpg://contained:mysecretpassword@localhost:5432/contained_platform"
    assert "mysecretpassword" not in scrub_sensitive_data(db_log)
    assert "[REDACTED_PASSWORD]" in scrub_sensitive_data(db_log)

    neo4j_log = "Connected bolt://neo4j:neo4jpass@localhost:7687"
    assert "neo4jpass" not in scrub_sensitive_data(neo4j_log)
    assert "[REDACTED_PASSWORD]" in scrub_sensitive_data(neo4j_log)

    # Test PII
    pii_log = "User test@example.com with phone 123-456-7890 and SSN 000-12-3456 requested help"
    scrubbed = scrub_sensitive_data(pii_log)
    assert "test@example.com" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "123-456-7890" not in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "000-12-3456" not in scrubbed
    assert "[REDACTED_SSN]" in scrubbed

    # Test generic secrets
    secret_log = "Auth parameters: password=supersecret, key: sk_live_abc123"
    scrubbed_secret = scrub_sensitive_data(secret_log)
    assert "supersecret" not in scrubbed_secret
    assert "password=[REDACTED]" in scrubbed_secret
    assert "sk_live_abc123" not in scrubbed_secret
    assert "[REDACTED_API_KEY]" in scrubbed_secret


# ---------------------------------------------------------------------------
# 6. Toxicity check test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_toxicity_filter_rejects_toxic_response(mocker):
    """Orchestrator rejects synthesized response if it is toxic."""
    from projects.guardroute.src.orchestrator import execute_orchestrator
    
    # Mock completion_with_fallback to return a toxic statement
    mock_choice = MagicMock()
    mock_choice.delta.content = "Shut up, you idiot! This is crap."
    mock_chunk = MagicMock()
    mock_chunk.choices = [mock_choice]
    mock_chunk.usage = None
    
    async def mock_completion(*args, **kwargs):
        async def mock_generator():
            yield mock_chunk
        return mock_generator()

    mocker.patch("projects.guardroute.src.orchestrator.completion_with_fallback", mock_completion)
    mocker.patch("projects.guardroute.src.orchestrator.get_redis_connection", return_value=None)
    mocker.patch("projects.guardroute.src.orchestrator.persist_session_and_usage", AsyncMock())
    mocker.patch("projects.guardroute.src.orchestrator.publish_trace_to_kafka", MagicMock())
    mocker.patch("projects.guardroute.src.orchestrator.classify_prompt", AsyncMock(return_value={"complexity": "simple", "required_agents": []}))

    with pytest.raises(ValueError, match="toxic content detected"):
        await execute_orchestrator("Hello")


# ---------------------------------------------------------------------------
# 7. Hallucination grounding test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hallucination_grounding_rejects_ungrounded_response(mocker):
    """Orchestrator rejects response if the grounding check fails."""
    from projects.guardroute.src.orchestrator import execute_orchestrator
    from common.schemas.agent_types import SubAgentResult, SubAgentStatus

    # Mock completion_with_fallback to stream synthesized answer
    mock_choice = MagicMock()
    mock_choice.delta.content = "Visual summary shows a blue truck."
    mock_chunk = MagicMock()
    mock_chunk.choices = [mock_choice]
    mock_chunk.usage = None

    async def mock_completion(*args, **kwargs):
        async def mock_generator():
            yield mock_chunk
        return mock_generator()

    mocker.patch("projects.guardroute.src.orchestrator.completion_with_fallback", mock_completion)
    mocker.patch("projects.guardroute.src.orchestrator.get_redis_connection", return_value=None)
    mocker.patch("projects.guardroute.src.orchestrator.persist_session_and_usage", AsyncMock())
    mocker.patch("projects.guardroute.src.orchestrator.publish_trace_to_kafka", MagicMock())

    # Mock classifer to return retrieval requirement
    mocker.patch("projects.guardroute.src.orchestrator.classify_prompt", AsyncMock(return_value={"complexity": "medium", "required_agents": ["retrieval"]}))
    
    # Mock retrieval subagent node execution to succeed and return content
    retrieval_res = SubAgentResult(
        source="retrieval",
        status=SubAgentStatus.SUCCESS,
        content="Document content details a red car in the yard."
    )
    mocker.patch("projects.guardroute.src.orchestrator.run_subagent_node", AsyncMock(return_value=retrieval_res))

    # Mock grounding check to fail (return ungrounded)
    mocker.patch("projects.guardroute.src.orchestrator.check_hallucination_grounding", AsyncMock(return_value=(False, "Response rejected: failed hallucination grounding check. Reason: response mentions blue truck, context specifies red car.")))

    with pytest.raises(ValueError, match="failed hallucination grounding check"):
        await execute_orchestrator("Find information about the truck")


# ---------------------------------------------------------------------------
# 8. MCP query_graph parameters test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_query_graph_parameters(mocker):
    """Verify that query_graph MCP tool accepts and forwards query parameters."""
    from projects.syntraflow.src.mcp_server import query_graph
    
    mock_execute = AsyncMock(return_value=[{"result": 1}])
    mocker.patch("common.clients.neo4j.execute_read_query", mock_execute)

    cypher = "MATCH (n:SYNTRAFLOW_Node) WHERE n.name = $name RETURN n"
    params = {"name": "TestNode"}
    
    res = await query_graph(cypher, parameters=params)
    
    mock_execute.assert_called_once_with(cypher, params)
    assert "TestNode" not in res  # result matches mock execute_read_query
    assert "result" in res

