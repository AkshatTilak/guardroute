"""DatabaseQueryNode executor for GuardRoute (Task 12).

Executes a parametrized read-only query against an external database
credential (hub-scoped `ExternalCredential`) and routes the result rows
to the `out` handle, or transitions to the `error` handle on failure.

Config shape (node `data`):
{
    "credential_id": "<external_credential_uuid>",
    "query_template": "SELECT * FROM users WHERE user_id = :user_id",
    "params_mapping": {"user_id": "{{input.user_id}}"},   # optional
    "timeout_s": 30,
    "max_rows": 500
}
"""

import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.clients.postgres import get_sessionmaker
from common.models.database import ExternalCredential
from common.clients.db_connectors.pool_manager import get_connector
from projects.guardroute.src.nodes.webhook_executor import _interpolate_dict

logger = logging.getLogger("guardroute.nodes.db_query_executor")

# Matches :param_name placeholders in a SQL template (excluding :: casts).
_PARAM_PATTERN = re.compile(r"(?<!:):([a-zA-Z_][a-zA-Z0-9_]*)")


def _extract_params(query_template: str) -> List[str]:
    """Return the ordered list of named parameter placeholders in a SQL template."""
    return _PARAM_PATTERN.findall(query_template)


def _resolve_param_value(param: str, state: Dict[str, Any], params_mapping: Dict[str, Any]) -> Any:
    """Resolve a single parameter value from state or the params_mapping."""
    # 1. Explicit mapping (e.g. {"user_id": "{{input.user_id}}"})
    if param in params_mapping:
        mapped = params_mapping[param]
        if isinstance(mapped, str):
            # Interpolate {{...}} templates against state
            interpolated = _interpolate_dict({"v": mapped}, state)["v"]
            return interpolated
        return mapped
    # 2. Fall back to a top-level state key of the same name
    return state.get(param)


async def _load_credential(session: AsyncSession, hub_id: str, credential_id: str) -> Optional[ExternalCredential]:
    """Load a hub-scoped ExternalCredential row."""
    stmt = select(ExternalCredential).where(
        ExternalCredential.hub_id == hub_id,
        ExternalCredential.id == credential_id,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def execute_database_query_node(
    config: Dict[str, Any],
    state: Dict[str, Any],
    *,
    hub_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a parametrized read-only query against an external database.

    Returns:
    {
        "rows": List[dict],
        "row_count": int,
        "success": bool,
        "error": Optional[str]
    }
    """
    credential_id: str = config.get("credential_id", "")
    query_template: str = config.get("query_template", "")
    timeout_s: int = int(config.get("timeout_s", 30))
    max_rows: int = int(config.get("max_rows", 500))
    params_mapping: Dict[str, Any] = config.get("params_mapping", {}) or {}

    if not credential_id:
        return {"rows": [], "row_count": 0, "success": False, "error": "DatabaseQueryNode missing credential_id"}
    if not query_template:
        return {"rows": [], "row_count": 0, "success": False, "error": "DatabaseQueryNode missing query_template"}

    # Resolve hub_id from state if not passed explicitly
    hub_id = hub_id or state.get("hub_id")

    try:
        session_factory = get_sessionmaker()
        async with session_factory() as session:  # type: AsyncSession
            cred = await _load_credential(session, hub_id, credential_id)
            if not cred:
                return {
                    "rows": [],
                    "row_count": 0,
                    "success": False,
                    "error": f"ExternalCredential '{credential_id}' not found in hub '{hub_id}'",
                }

            # Build the parametrized query: substitute :param placeholders with
            # bound values resolved from state / params_mapping.
            params: Dict[str, Any] = {}
            for param in _extract_params(query_template):
                params[param] = _resolve_param_value(param, state, params_mapping)

            connector = await get_connector(cred)
            rows = await connector.execute_query(
                query_template,
                params=params or None,
                timeout_s=timeout_s,
                max_rows=max_rows,
            )
            return {
                "rows": rows,
                "row_count": len(rows),
                "success": True,
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001 - surface any DB failure to the error handle
        logger.warning("DatabaseQueryNode failed: %s", exc)
        return {
            "rows": [],
            "row_count": 0,
            "success": False,
            "error": str(exc),
        }
