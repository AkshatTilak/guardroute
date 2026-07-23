"""Unit tests for GuardRoute webhook_executor and SSRF protection."""

import pytest
from projects.guardroute.src.nodes.ssrf_protection import validate_url_for_ssrf, SSRFValidationError
from projects.guardroute.src.nodes.webhook_executor import execute_webhook, _interpolate_string


def test_ssrf_blocks_private_ips():
    with pytest.raises(SSRFValidationError):
        validate_url_for_ssrf("http://127.0.0.1/admin")

    with pytest.raises(SSRFValidationError):
        validate_url_for_ssrf("http://10.0.0.5/api")

    with pytest.raises(SSRFValidationError):
        validate_url_for_ssrf("http://192.168.1.1/router")

    with pytest.raises(SSRFValidationError):
        validate_url_for_ssrf("http://localhost:8000")


def test_ssrf_allows_public_urls():
    assert validate_url_for_ssrf("https://httpbin.org/post") is True


def test_template_interpolation():
    state = {"prompt": "Hello World", "user": {"name": "Alice"}}
    res = _interpolate_string("User {{user.name}} asked: {{prompt}}", state)
    assert res == "User Alice asked: Hello World"


@pytest.mark.asyncio
async def test_execute_webhook_ssrf_blocked():
    state = {"prompt": "test"}
    config = {
        "url": "http://127.0.0.1:9000/webhook",
        "method": "POST"
    }
    result = await execute_webhook(config, state)
    assert result["success"] is False
    assert result["status_code"] == 403
    assert "SSRF Blocked" in result["error"]
