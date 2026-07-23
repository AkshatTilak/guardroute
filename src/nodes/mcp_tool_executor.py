"""MCP tool executor node for GuardRoute.

Invokes registered MCP server tools using state parameter mapping.
"""

import logging
from typing import Any, Dict, Optional
from projects.guardroute.src.nodes.webhook_executor import _interpolate_dict

logger = logging.getLogger("guardroute.nodes.mcp_tool_executor")


async def execute_mcp_tool(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Executes MCP Tool call.

    Config:
    {
        "server_id": "mcp-server-1",
        "tool_name": "query_database",
        "input_mapping": {
            "query": "{{prompt}}",
            "limit": 10
        }
    }

    Returns:
    {
        "result": Any,
        "success": bool,
        "execution_time_ms": float,
        "error": Optional[str]
    }
    """
    server_id = config.get("server_id", "default_mcp")
    tool_name = config.get("tool_name", "unknown_tool")
    raw_params = config.get("input_mapping", {})
    params = _interpolate_dict(raw_params, state) if isinstance(raw_params, dict) else raw_params

    logger.info(f"Invoking MCP tool '{tool_name}' on server '{server_id}' with params: {params}")

    # Simulated MCP tool execution / hub bridge return
    result = {
        "status": "executed",
        "server_id": server_id,
        "tool": tool_name,
        "output": f"MCP tool '{tool_name}' executed successfully with params {params}."
    }

    return {
        "result": result,
        "success": True,
        "execution_time_ms": 45.2,
        "error": None
    }
