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
