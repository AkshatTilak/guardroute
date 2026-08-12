"""DBStoreNode executor for GuardRoute (Task 12).

Persists a data record into an external database as a first-class workflow
step. Supports `insert`, `upsert`, and `append` operations against a target
table (SQL) or collection (MongoDB).

Config shape (node `data`):
{
    "credential_id": "<external_credential_uuid>",
    "target_table": "users",
    "operation": "insert" | "upsert" | "append",
    "record_mapping": {"name": "{{input.name}}", "role": "{{input.role}}"}
}
"""

import logging
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.clients.postgres import get_sessionmaker
from common.models.database import ExternalCredential
from common.clients.db_connectors.pool_manager import get_connector
from projects.guardroute.src.nodes.webhook_executor import _interpolate_dict

logger = logging.getLogger("guardroute.nodes.db_store_executor")


async def _load_credential(session: AsyncSession, hub_id: str, credential_id: str) -> Optional[ExternalCredential]:
    """Load a hub-scoped ExternalCredential row."""
    stmt = select(ExternalCredential).where(
        ExternalCredential.hub_id == hub_id,
        ExternalCredential.id == credential_id,
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def execute_db_store_node(
    config: Dict[str, Any],
    state: Dict[str, Any],
    *,
    hub_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a record into an external database.

    Returns:
    {
        "affected": int,
        "primary_key": Optional[Any],
        "success": bool,
        "error": Optional[str]
    }
    """
    credential_id: str = config.get("credential_id", "")
    target_table: str = config.get("target_table", "")
    operation: str = config.get("operation", "insert")
    record_mapping: Dict[str, Any] = config.get("record_mapping", {}) or {}

    if not credential_id:
        return {"affected": 0, "primary_key": None, "success": False, "error": "DBStoreNode missing credential_id"}
    if not target_table:
        return {"affected": 0, "primary_key": None, "success": False, "error": "DBStoreNode missing target_table"}

    hub_id = hub_id or state.get("hub_id")

    try:
        session_factory = get_sessionmaker()
        async with session_factory() as session:  # type: AsyncSession
            cred = await _load_credential(session, hub_id, credential_id)
            if not cred:
                return {
                    "affected": 0,
                    "primary_key": None,
                    "success": False,
                    "error": f"ExternalCredential '{credential_id}' not found in hub '{hub_id}'",
                }

            # Resolve the record to persist from state / record_mapping.
            record: Dict[str, Any] = {}
            if record_mapping:
                record = _interpolate_dict(record_mapping, state)
            else:
                # Fall back to the incoming payload on state["input"].
                incoming = state.get("input") or state.get("payload") or {}
                if isinstance(incoming, dict):
                    record = incoming

            connector = await get_connector(cred)
            result = await connector.store_record(
                target_table=target_table,
                record=record,
                operation=operation,
            )
            return {
                "affected": result.get("affected", 0),
                "primary_key": result.get("primary_key"),
                "success": True,
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001 - surface any DB failure to the error handle
        logger.warning("DBStoreNode failed: %s", exc)
        return {
            "affected": 0,
            "primary_key": None,
            "success": False,
            "error": str(exc),
        }
