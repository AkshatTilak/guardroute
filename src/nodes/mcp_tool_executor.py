"""MCP tool executor node for GuardRoute.

Invokes registered MCP server tools via the Gateway MCP client, using the
database to look up server configuration and the mcp_client.invoke_tool()
function for the actual HTTP/JSON-RPC call.
"""

import logging
import time
from typing import Any, Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.clients.postgres import get_sessionmaker
from common.models.database import MCPServer
from gateway.services.mcp_client import invoke_tool
from common.services.mcp_db_bridge import execute_db_tool, format_rows_as_markdown
from projects.guardroute.src.nodes.webhook_executor import _interpolate_dict

logger = logging.getLogger("guardroute.nodes.mcp_tool_executor")

# Database MCP tools are handled by the local DB bridge rather than an
# external MCP server.
_DB_TOOL_NAMES = {"db_schema_inspector", "db_query_executor", "mongo_collection_query"}


async def execute_mcp_tool(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Executes an MCP Tool call via the registered Gateway MCP server.

    Config shape:
    {
        "server_id": "<mcp-server-uuid>",
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
    server_id: str = config.get("server_id", "")
    tool_name: str = config.get("tool_name", "unknown_tool")
    raw_params: Dict[str, Any] = config.get("input_mapping", {})
    params: Dict[str, Any] = _interpolate_dict(raw_params, state) if isinstance(raw_params, dict) else raw_params

    logger.info("Invoking MCP tool '%s' on server '%s' with params: %s", tool_name, server_id, params)

    start_ms = time.monotonic()

    session_factory = state.get("session_factory") or get_sessionmaker()

    # Database MCP tools are executed by the local DB bridge (Task 12_02).
    if tool_name in _DB_TOOL_NAMES:
        hub_id = state.get("hub_id")
        result = await execute_db_tool(tool_name, params, hub_id=hub_id, session_factory=session_factory)
        exec_ms = round((time.monotonic() - start_ms) * 1000, 2)
        success = result.get("status") == "success"
        # Format tabular SQL results as a Markdown table for agent context.
        formatted = None
        if success and isinstance(result.get("result"), list):
            formatted = format_rows_as_markdown(result["result"])
        raw_res = result.get("result")
        row_cnt = len(raw_res) if isinstance(raw_res, list) else 0
        return {
            "result": formatted if formatted is not None else raw_res,
            "row_count": row_cnt,
            "success": success,
            "execution_time_ms": exec_ms,
            "error": result.get("error") if not success else None,
        }

    # If no server_id configured, fall back to graceful degraded result
    if not server_id:
        logger.warning("MCPToolNode has no server_id configured; returning degraded result")
        return {
            "result": None,
            "success": False,
            "execution_time_ms": 0.0,
            "error": "MCPToolNode missing server_id configuration",
        }

    db_sess = state.get("db_session") or state.get("session")
    try:
        if db_sess:
            stmt = select(MCPServer).where(
                MCPServer.id == server_id,
                MCPServer.is_active.is_(True),
            )
            server = (await db_sess.execute(stmt)).scalar_one_or_none()
        else:
            async with session_factory() as session:  # type: AsyncSession
                stmt = select(MCPServer).where(
                    MCPServer.id == server_id,
                    MCPServer.is_active.is_(True),
                )
                server = (await session.execute(stmt)).scalar_one_or_none()

        if not server:
            logger.warning("MCP server '%s' not found or inactive", server_id)
            return {
                "result": None,
                "success": False,
                "execution_time_ms": round((time.monotonic() - start_ms) * 1000, 2),
                "error": f"MCP server '{server_id}' not found or inactive",
            }

        result = await invoke_tool(server, tool_name, params)
        exec_ms = round((time.monotonic() - start_ms) * 1000, 2)

        success = result.get("status") == "success"
        return {
            "result": result.get("result"),
            "success": success,
            "execution_time_ms": result.get("execution_time_ms", exec_ms),
            "error": result.get("error") if not success else None,
        }

    except Exception as exc:
        exec_ms = round((time.monotonic() - start_ms) * 1000, 2)
        logger.exception("MCP tool invocation failed: server=%s tool=%s", server_id, tool_name)
        return {
            "result": None,
            "success": False,
            "execution_time_ms": exec_ms,
            "error": str(exc),
        }
