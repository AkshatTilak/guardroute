"""Agent tool executor for GuardRoute.

Vector retrieval, MCP tools, and external database access are no longer
standalone workflow nodes — they are capabilities (tools) bound to an agent
node. This module dispatches a list of `ToolBinding` objects to the underlying
executors and returns a consolidated result map the agent can consume.

Supported tool types:
  - `retrieval`  : vector search over an Ingestion Hub collection
  - `mcp`        : a registered MCP server tool
  - `db`         : a read-only query over an ExternalCredential
  - `web_search` : web search capability
  - `api_call`   : an external HTTP API call
"""

import logging
from typing import Any, Dict, List, Optional

from projects.guardroute.src.nodes.mcp_tool_executor import execute_mcp_tool
from projects.guardroute.src.nodes.db_query_executor import execute_database_query_node
from projects.guardroute.src.nodes.api_call_executor import execute_api_call

logger = logging.getLogger("guardroute.nodes.tool_executor")


async def _run_retrieval_tool(tool: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Vector retrieval over a specific Ingestion Hub collection."""
    hub_id = tool.get("hub_id")
    collection_id = tool.get("collection_id")
    query = state.get("prompt") or state.get("input") or ""
    limit = int(tool.get("limit", 5))
    try:
        from projects.syntraflow.src.retrieval.engine import RetrievalEngine
        from projects.syntraflow.src.database.models import SyntraFlowCollection
        from common.clients.postgres import get_sessionmaker
        SessionLocal = get_sessionmaker()
        async with SessionLocal() as db:
            if not hub_id and collection_id:
                col_row = await db.get(SyntraFlowCollection, collection_id)
                if col_row:
                    hub_id = col_row.hub_id

            if not hub_id:
                return {"success": False, "rows": [], "row_count": 0, "error": "Missing hub_id for retrieval", "tool_type": "retrieval"}

            engine = RetrievalEngine(db, hub_id)
            results = await engine.search(
                query=query,
                collection_ids=[collection_id] if collection_id else None,
                limit=limit,
            )
            return {"success": True, "rows": results, "row_count": len(results), "tool_type": "retrieval"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retrieval tool failed for collection %s: %s", collection_id, exc)
        return {"success": False, "rows": [], "row_count": 0, "error": str(exc), "tool_type": "retrieval"}


async def _run_mcp_tool(tool: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke a registered MCP server tool."""
    config = {
        "server_id": tool.get("server_id", ""),
        "tool_name": tool.get("tool_name", ""),
        "input_mapping": tool.get("input_mapping", {}),
    }
    res = await execute_mcp_tool(config, state)
    res["tool_type"] = "mcp"
    return res


async def _run_db_tool(tool: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Run a read-only query against an external database credential."""
    config = {
        "credential_id": tool.get("credential_id", ""),
        "query_template": tool.get("query_template", ""),
        "params_mapping": tool.get("params_mapping", {}),
        "timeout_s": tool.get("timeout_s", 30),
        "max_rows": tool.get("max_rows", 500),
    }
    res = await execute_database_query_node(config, state, hub_id=state.get("hub_id"))
    res["tool_type"] = "db"
    return res


async def _run_web_search_tool(tool: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Web search capability."""
    query = state.get("prompt") or state.get("input") or ""
    try:
        from projects.guardroute.src.agents.search import run_web_search
        res = await run_web_search(query)
        return {"success": True, "result": res, "tool_type": "web_search"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search tool failed: %s", exc)
        return {"success": False, "error": str(exc), "tool_type": "web_search"}


async def _run_api_call_tool(tool: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """External HTTP API call."""
    config = {
        "url": tool.get("url", ""),
        "method": tool.get("method", "GET"),
        "headers": tool.get("headers", {}),
        "body": tool.get("body", {}),
        "params_mapping": tool.get("params_mapping", {}),
    }
    res = await execute_api_call(config, state)
    res["tool_type"] = "api_call"
    return res


async def execute_agent_tools(
    tools: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute all enabled tool bindings for an agent node.

    Returns a map keyed by tool index (or label) -> result dict, plus a
    convenience `tool_results` list. Failures are captured per-tool so one
    failing tool does not abort the whole agent turn.
    """
    results: Dict[str, Any] = {}
    for idx, tool in enumerate(tools or []):
        if not isinstance(tool, dict):
            continue
        if tool.get("enabled", True) is False:
            continue
        ttype = tool.get("type")
        key = tool.get("label") or f"tool_{idx}"
        try:
            if ttype == "retrieval":
                results[key] = await _run_retrieval_tool(tool, state)
            elif ttype == "mcp":
                results[key] = await _run_mcp_tool(tool, state)
            elif ttype == "db":
                results[key] = await _run_db_tool(tool, state)
            elif ttype == "web_search":
                results[key] = await _run_web_search_tool(tool, state)
            elif ttype == "api_call":
                results[key] = await _run_api_call_tool(tool, state)
            else:
                results[key] = {"success": False, "error": f"Unknown tool type '{ttype}'", "tool_type": ttype}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool '%s' (%s) raised %s: %s", key, ttype, type(exc).__name__, exc)
            results[key] = {"success": False, "error": str(exc), "tool_type": ttype}

    return {
        "tool_results": list(results.values()),
        "tool_results_map": results,
        "tool_count": len(results),
    }
