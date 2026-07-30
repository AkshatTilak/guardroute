"""Hub-Scoped Agent Repository (S6-05a).

Provides CRUD data accessors and hub-local slug generation for AgentDefinition rows.
Every query strictly includes `hub_id` in its WHERE clause (hubs.md §5.3).
"""

import logging
import re
import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.models.database import AgentDefinition, Hub
from common.schemas.agent_types import AgentCreate, AgentUpdate

logger = logging.getLogger("guardroute.agents.repository")

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9_-]+")


def slugify(text: str) -> str:
    """Normalize text into an URL-friendly slug."""
    s = text.lower().strip()
    s = re.sub(r"[\s_]+", "-", s)
    s = _SLUG_CLEAN_RE.sub("", s)
    return s.strip("-") or "agent"


async def generate_unique_slug(
    session: AsyncSession,
    *,
    hub_id: str,
    base_name: str,
    exclude_agent_id: Optional[str] = None,
) -> str:
    """Generate a unique endpoint_slug within a specific hub."""
    base_slug = slugify(base_name)
    candidate = base_slug
    counter = 1

    while True:
        stmt = select(AgentDefinition).where(
            AgentDefinition.hub_id == hub_id,
            AgentDefinition.endpoint_slug == candidate,
        )
        if exclude_agent_id:
            stmt = stmt.where(AgentDefinition.id != exclude_agent_id)

        res = await session.execute(stmt)
        existing = res.scalar_one_or_none()
        if not existing:
            return candidate

        counter += 1
        candidate = f"{base_slug}-{counter}"


async def _verify_agent_hub(session: AsyncSession, hub_id: str) -> Hub:
    """Ensure hub exists and is an agent hub."""
    hub = await session.get(Hub, hub_id)
    if not hub or hub.hub_type != "agent":
        raise ValueError(f"Hub '{hub_id}' not found or is not an agent hub.")
    return hub


async def list_agents(
    session: AsyncSession,
    *,
    hub_id: str,
    is_active: Optional[bool] = None,
    q: Optional[str] = None,
) -> List[AgentDefinition]:
    """List agent definitions owned by a hub."""
    stmt = select(AgentDefinition).where(AgentDefinition.hub_id == hub_id)
    if is_active is not None:
        stmt = stmt.where(AgentDefinition.is_active == is_active)
    if q:
        stmt = stmt.where(AgentDefinition.name.ilike(f"%{q}%"))

    stmt = stmt.order_by(AgentDefinition.name.asc())
    res = await session.execute(stmt)
    return list(res.scalars().all())


async def get_agent(
    session: AsyncSession,
    *,
    hub_id: str,
    agent_id: str,
) -> Optional[AgentDefinition]:
    """Fetch an agent definition strictly scoped to hub_id."""
    stmt = select(AgentDefinition).where(
        AgentDefinition.id == agent_id,
        AgentDefinition.hub_id == hub_id,
    )
    res = await session.execute(stmt)
    return res.scalar_one_or_none()


async def get_agent_by_slug(
    session: AsyncSession,
    *,
    hub_id: str,
    slug: str,
) -> Optional[AgentDefinition]:
    """Fetch an agent by endpoint_slug within a hub."""
    stmt = select(AgentDefinition).where(
        AgentDefinition.hub_id == hub_id,
        AgentDefinition.endpoint_slug == slug,
    )
    res = await session.execute(stmt)
    return res.scalar_one_or_none()


async def create_agent(
    session: AsyncSession,
    *,
    hub_id: str,
    payload: AgentCreate,
) -> AgentDefinition:
    """Create a new agent definition in an agent hub."""
    await _verify_agent_hub(session, hub_id)

    slug = payload.endpoint_slug
    if not slug:
        slug = await generate_unique_slug(session, hub_id=hub_id, base_name=payload.name)
    else:
        slug = slugify(slug)
        existing = await get_agent_by_slug(session, hub_id=hub_id, slug=slug)
        if existing:
            raise ValueError(f"Endpoint slug '{slug}' is already taken in this hub.")

    bindings_dict = [b.model_dump() for b in payload.collection_bindings]

    agent = AgentDefinition(
        id=str(uuid.uuid4()),
        hub_id=hub_id,
        name=payload.name,
        role=payload.role,
        system_prompt=payload.system_prompt,
        model_id=payload.model_id,
        tools=payload.tools,
        collection_bindings_json=bindings_dict,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        is_active=payload.is_active,
        endpoint_slug=slug,
    )
    session.add(agent)
    return agent


async def update_agent(
    session: AsyncSession,
    *,
    hub_id: str,
    agent_id: str,
    payload: AgentUpdate,
) -> AgentDefinition:
    """Update an existing agent definition within a hub."""
    agent = await get_agent(session, hub_id=hub_id, agent_id=agent_id)
    if not agent:
        raise ValueError(f"Agent '{agent_id}' not found in hub '{hub_id}'.")

    data = payload.model_dump(exclude_unset=True)

    if "endpoint_slug" in data and data["endpoint_slug"]:
        slug = slugify(data["endpoint_slug"])
        existing = await get_agent_by_slug(session, hub_id=hub_id, slug=slug)
        if existing and existing.id != agent_id:
            raise ValueError(f"Endpoint slug '{slug}' is already taken in this hub.")
        agent.endpoint_slug = slug
    elif "name" in data and data["name"] != agent.name and not agent.endpoint_slug:
        agent.endpoint_slug = await generate_unique_slug(
            session, hub_id=hub_id, base_name=data["name"], exclude_agent_id=agent_id
        )

    if "collection_bindings" in data and data["collection_bindings"] is not None:
        agent.collection_bindings_json = [b for b in data["collection_bindings"]]

    for field in ("name", "role", "system_prompt", "model_id", "tools", "temperature", "max_tokens", "is_active"):
        if field in data and data[field] is not None:
            setattr(agent, field, data[field])

    return agent


async def delete_agent(
    session: AsyncSession,
    *,
    hub_id: str,
    agent_id: str,
) -> bool:
    """Delete an agent definition from a hub."""
    agent = await get_agent(session, hub_id=hub_id, agent_id=agent_id)
    if not agent:
        return False
    await session.delete(agent)
    return True
