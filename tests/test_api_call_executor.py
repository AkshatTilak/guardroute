"""Unit tests for GuardRoute api_call_executor."""

import pytest
from projects.guardroute.src.nodes.api_call_executor import execute_api_call, _extract_jsonpath_field


def test_extract_jsonpath_field():
    data = {
        "status": "success",
        "data": {
            "items": [{"id": 10, "name": "Alpha"}, {"id": 20, "name": "Beta"}]
        }
    }
    assert _extract_jsonpath_field(data, "$.status") == "success"
    assert _extract_jsonpath_field(data, "$.data.items.0.name") == "Alpha"
    assert _extract_jsonpath_field(data, "$.data.items.1.id") == 20
    assert _extract_jsonpath_field(data, "$.nonexistent") is None


@pytest.mark.asyncio
async def test_execute_api_call_ssrf_blocked():
    state = {}
    config = {
        "url": "http://127.0.0.1:8000/internal-api",
        "method": "GET"
    }
    res = await execute_api_call(config, state)
    assert res["success"] is False
    assert res["status_code"] == 403
    assert "SSRF Blocked" in res["error"]
