"""Agent ↔ Collection Binding via Hub Links (S6-05b).

Validates and resolves qualified collection bindings ({hub_id, collection_id}).
Enforces cross-hub link grants (HUB_LINK_REQUIRED at save, HUB_LINK_REVOKED at invoke).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.models.database import Hub, HubLink
from common.schemas.agent_types import CollectionBinding, CollectionBindingResponse
from common.services.hub_repository import get_link
from projects.syntraflow.src.collections.manager import CollectionManager
from projects.syntraflow.src.database.models import SyntraFlowCollection
from projects.syntraflow.src.datastores.resolver import resolve_vector_client

logger = logging.getLogger("guardroute.agents.collection_binding")


@dataclass
class ResolvedBinding:
    """Resolved collection binding with target store client and physical name."""

    binding: CollectionBinding
    collection: SyntraFlowCollection
    owning_hub: Hub
    physical_name: str


async def validate_bindings(
    session: AsyncSession,
    *,
    source_hub_id: str,
    bindings: List[CollectionBinding],
) -> None:
    """Validate collection bindings at create/update time.

    Raises:
        HTTPException(403, HUB_LINK_REQUIRED) if hub link is missing.
        HTTPException(422, CROSS_HUB_REFERENCE_MISMATCH) if collection is not in specified hub.
        HTTPException(404, BINDING_TARGET_MISSING) if collection doesn't exist.
    """
    for b in bindings:
        # Check target hub exists
        target_hub = await session.get(Hub, b.hub_id)
        if not target_hub or target_hub.hub_type != "ingestion":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target ingestion hub '{b.hub_id}' not found.",
                headers={"X-Error-Code": "TARGET_HUB_NOT_FOUND"},
            )

        # Check collection exists and belongs to b.hub_id
        col = await session.get(SyntraFlowCollection, b.collection_id)
        if not col:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bound collection '{b.collection_id}' no longer exists.",
                headers={"X-Error-Code": "BINDING_TARGET_MISSING"},
            )

        if col.hub_id != b.hub_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Collection '{b.collection_id}' does not belong to hub '{b.hub_id}'.",
                headers={"X-Error-Code": "CROSS_HUB_REFERENCE_MISMATCH"},
            )

        # Check hub link exists
        link = await get_link(session, source_hub_id=source_hub_id, target_hub_id=b.hub_id)
        if not link or link.access_level not in ("read", "use"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Agent hub '{source_hub_id}' is not linked to ingestion hub '{b.hub_id}'.",
                headers={"X-Error-Code": "HUB_LINK_REQUIRED"},
            )


async def resolve_bindings(
    session: AsyncSession,
    *,
    source_hub_id: str,
    bindings: List[CollectionBinding],
) -> List[ResolvedBinding]:
    """Resolve collection bindings dynamically at execution time.

    Raises:
        HTTPException(403, HUB_LINK_REVOKED) if link was revoked since binding creation.
        HTTPException(404, BINDING_TARGET_MISSING) if collection was deleted.
    """
    resolved = []
    for b in bindings:
        target_hub = await session.get(Hub, b.hub_id)
        if not target_hub or target_hub.is_archived:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Hub link to '{b.hub_id}' was revoked; target hub is unavailable.",
                headers={"X-Error-Code": "HUB_LINK_REVOKED"},
            )

        link = await get_link(session, source_hub_id=source_hub_id, target_hub_id=b.hub_id)
        if not link or link.access_level not in ("read", "use"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Hub link to '{b.hub_id}' was revoked; retrieval aborted.",
                headers={"X-Error-Code": "HUB_LINK_REVOKED"},
            )

        col = await session.get(SyntraFlowCollection, b.collection_id)
        if not col or col.hub_id != b.hub_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bound collection '{b.collection_id}' no longer exists.",
                headers={"X-Error-Code": "BINDING_TARGET_MISSING"},
            )

        resolved.append(
            ResolvedBinding(
                binding=b,
                collection=col,
                owning_hub=target_hub,
                physical_name=col.physical_name,
            )
        )
    return resolved


async def inspect_binding_statuses(
    session: AsyncSession,
    *,
    source_hub_id: str,
    bindings_raw: List[Dict[str, Any]],
) -> List[CollectionBindingResponse]:
    """Inspect binding statuses for AgentResponse presentation."""
    res = []
    for raw in bindings_raw:
        hub_id = raw.get("hub_id", "")
        col_id = raw.get("collection_id", "")
        alias = raw.get("alias")
        top_k = raw.get("top_k", 5)

        col = await session.get(SyntraFlowCollection, col_id)
        if not col or col.hub_id != hub_id:
            status_val = "missing"
        else:
            link = await get_link(session, source_hub_id=source_hub_id, target_hub_id=hub_id)
            if not link or link.access_level not in ("read", "use"):
                status_val = "link_revoked"
            else:
                status_val = "ok"

        res.append(
            CollectionBindingResponse(
                hub_id=hub_id,
                collection_id=col_id,
                alias=alias,
                top_k=top_k,
                status=status_val,
            )
        )
    return res
