"""Unit tests for GuardRoute mcp_tool_executor."""

import pytest
from projects.guardroute.src.nodes.mcp_tool_executor import execute_mcp_tool


@pytest.mark.asyncio
async def test_execute_mcp_tool():
    state = {"prompt": "SQL Query for users"}
    config = {
        "server_id": "postgres-mcp",
        "tool_name": "execute_query",
        "input_mapping": {"query": "{{prompt}}"}
    }
    res = await execute_mcp_tool(config, state)
    assert res["success"] is True
    assert res["result"]["server_id"] == "postgres-mcp"
    assert res["result"]["tool"] == "execute_query"
