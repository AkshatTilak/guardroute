"""Version Lifecycle Service for Workflow Hub (S6-06b).

Implements the draft/publish/restore state machine, optimistic concurrency control via ETags,
graph diffing, version duplication, and archived hub checks.
"""

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from common.models.database import Hub, WorkflowDefinition, WorkflowVersion, User
from projects.guardroute.src.core.graph_parser import GraphValidationError, validate_workflow_graph


class WorkflowNotFoundError(Exception):
    """Raised when a workflow is not found within the specified hub."""
    pass


class ETagRequiredError(Exception):
    """Raised when an update is attempted without expected_etag on a published workflow."""
    pass


class DraftConflict(Exception):
    """Raised when an update carries a stale ETag."""

    def __init__(
        self,
        message: str,
        server_etag: str,
        server_version_number: int,
        server_graph: Dict[str, Any],
        updated_by: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.server_etag = server_etag
        self.server_version_number = server_version_number
        self.server_graph = server_graph
        self.updated_by = updated_by
        self.updated_at = updated_at


class HubArchivedError(Exception):
    """Raised when attempting a mutating operation on an archived hub."""
    pass


class VersionNotFoundError(Exception):
    """Raised when a specific version number is not found for a workflow."""
    pass


@dataclass
class NodeChange:
    node_id: str
    changed_fields: List[str]


@dataclass
class GraphDiff:
    nodes_added: List[str]
    nodes_removed: List[str]
    nodes_changed: List[NodeChange]
    edges_added: List[str]
    edges_removed: List[str]
    edges_changed: List[str]


@dataclass
class DraftUpdateResult:
    version: WorkflowVersion
    etag: str
    is_valid: bool
    validation_json: Optional[Dict[str, Any]]


def compute_etag(version: WorkflowVersion) -> str:
    """Compute a stable ETag for a WorkflowVersion.

    Payload consists of version id, version_number, and canonical sorted json of graph_json.
    """
    graph = version.graph_json or {}
    payload = json.dumps(graph, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{version.id}:{version.version_number}:{payload}".encode()).hexdigest()
    return f'W/"{digest[:32]}"'


def _normalize_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized node copy stripped of layout-only fields (position, viewport, selected, etc.)."""
    return {
        k: v for k, v in node.items()
        if k not in ("position", "positionAbsolute", "viewport", "selected", "dragging", "width", "height")
    }


def diff_versions(base_graph: Dict[str, Any], head_graph: Dict[str, Any]) -> GraphDiff:
    """Compute structural diff between base and head graphs ignoring position & viewport layout changes."""
    base_nodes = {n["id"]: n for n in base_graph.get("nodes", []) if "id" in n}
    head_nodes = {n["id"]: n for n in head_graph.get("nodes", []) if "id" in n}

    nodes_added = sorted([nid for nid in head_nodes if nid not in base_nodes])
    nodes_removed = sorted([nid for nid in base_nodes if nid not in head_nodes])

    nodes_changed: List[NodeChange] = []
    common_nodes = set(base_nodes.keys()) & set(head_nodes.keys())
    for nid in sorted(common_nodes):
        b_norm = _normalize_node(base_nodes[nid])
        h_norm = _normalize_node(head_nodes[nid])
        changed_fields = []
        all_keys = set(b_norm.keys()) | set(h_norm.keys())
        for k in sorted(all_keys):
            if b_norm.get(k) != h_norm.get(k):
                changed_fields.append(k)
        if changed_fields:
            nodes_changed.append(NodeChange(node_id=nid, changed_fields=changed_fields))

    base_edges = {
        f"{e.get('source')}->{e.get('target')}": e
        for e in base_graph.get("edges", [])
        if e.get("source") and e.get("target")
    }
    head_edges = {
        f"{e.get('source')}->{e.get('target')}": e
        for e in head_graph.get("edges", [])
        if e.get("source") and e.get("target")
    }

    edges_added = sorted([eid for eid in head_edges if eid not in base_edges])
    edges_removed = sorted([eid for eid in base_edges if eid not in head_edges])
    edges_changed: List[str] = []

    common_edges = set(base_edges.keys()) & set(head_edges.keys())
    for eid in sorted(common_edges):
        b_e = base_edges[eid]
        h_e = head_edges[eid]
        if b_e.get("label") != h_e.get("label") or b_e.get("data") != h_e.get("data"):
            edges_changed.append(eid)

    return GraphDiff(
        nodes_added=nodes_added,
        nodes_removed=nodes_removed,
        nodes_changed=nodes_changed,
        edges_added=edges_added,
        edges_removed=edges_removed,
        edges_changed=edges_changed,
    )


async def _get_hub(session: AsyncSession, hub_id: str) -> Hub:
    """Fetch hub by ID; fail if not found."""
    stmt = select(Hub).where(Hub.id == hub_id)
    hub = (await session.execute(stmt)).scalar_one_or_none()
    if not hub:
        raise ValueError(f"Hub '{hub_id}' not found.")
    return hub


async def _get_workflow_with_lock(
    session: AsyncSession, hub_id: str, workflow_id: str, for_update: bool = True
) -> tuple[Hub, WorkflowDefinition]:
    """Fetch Hub and WorkflowDefinition ensuring hub ownership. Lock workflow row if for_update is True."""
    hub = await _get_hub(session, hub_id)
    stmt = select(WorkflowDefinition).where(
        WorkflowDefinition.hub_id == hub_id,
        WorkflowDefinition.id == workflow_id,
    )
    if for_update:
        stmt = stmt.with_for_update()
    wf = (await session.execute(stmt)).scalar_one_or_none()
    if not wf:
        raise WorkflowNotFoundError(f"Workflow '{workflow_id}' not found in hub '{hub_id}'.")
    return hub, wf


async def _validate_graph_payload(
    session: Optional[AsyncSession],
    source_hub_id: str,
    graph: Dict[str, Any],
    strict: bool = False,
) -> tuple[bool, Dict[str, Any]]:
    """Run validate_workflow_graph on graph payload and return (is_valid, validation_json dict)."""
    try:
        val_res = await validate_workflow_graph(
            session=session,  # type: ignore
            graph_json=graph,
            source_hub_id=source_hub_id,
            strict=strict,
        )
        return val_res.is_valid, val_res.model_dump()
    except GraphValidationError as e:
        return False, {"is_valid": False, "errors": [{"message": str(e)}], "warnings": []}
    except Exception as e:
        return False, {"is_valid": False, "errors": [{"message": str(e)}], "warnings": []}


def _slugify(name: str) -> str:
    text = name.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text.strip('-') or 'workflow'


async def get_draft(session: AsyncSession, *, hub_id: str, workflow_id: str) -> WorkflowVersion:
    """Get active draft version for workflow. Auto-create draft if absent."""
    hub, wf = await _get_workflow_with_lock(session, hub_id, workflow_id, for_update=True)

    if wf.draft_version_id:
        stmt_draft = select(WorkflowVersion).where(WorkflowVersion.id == wf.draft_version_id)
        draft = (await session.execute(stmt_draft)).scalar_one_or_none()
        if draft:
            return draft

    # Auto-create draft version
    stmt_max = select(func.coalesce(func.max(WorkflowVersion.version_number), 0)).where(
        WorkflowVersion.workflow_id == workflow_id
    )
    max_ver = (await session.execute(stmt_max)).scalar() or 0
    next_ver = max_ver + 1

    source_graph: Dict[str, Any] = {"nodes": [], "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}
    if wf.published_version_id:
        stmt_pub = select(WorkflowVersion).where(WorkflowVersion.id == wf.published_version_id)
        pub_ver = (await session.execute(stmt_pub)).scalar_one_or_none()
        if pub_ver and pub_ver.graph_json:
            source_graph = pub_ver.graph_json

    is_valid, val_json = await _validate_graph_payload(session, hub_id, source_graph, strict=False)

    new_draft = WorkflowVersion(
        id=str(uuid.uuid4()),
        workflow_id=workflow_id,
        version_number=next_ver,
        graph_json=source_graph,
        change_note="Auto-created draft",
        is_valid=is_valid,
        validation_json=val_json,
        created_at=datetime.utcnow(),
    )
    session.add(new_draft)
    await session.flush()

    wf.draft_version_id = new_draft.id
    wf.updated_at = datetime.utcnow()
    await session.commit()
    return new_draft


async def update_draft(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
    graph: Dict[str, Any],
    expected_etag: Optional[str],
    actor_id: str,
    change_note: Optional[str] = None,
) -> DraftUpdateResult:
    """Update workflow draft in place with optimistic locking via ETag."""
    hub, wf = await _get_workflow_with_lock(session, hub_id, workflow_id, for_update=True)
    if hub.is_archived:
        raise HubArchivedError(f"Cannot update draft: Hub '{hub_id}' is archived.")

    draft = await get_draft(session, hub_id=hub_id, workflow_id=workflow_id)
    current_etag = compute_etag(draft)

    # Require ETag if workflow has ever been published
    if expected_etag is None and wf.published_version_id is not None:
        raise ETagRequiredError("If-Match header with version_etag is required to update draft.")

    if expected_etag is not None and expected_etag != current_etag:
        raise DraftConflict(
            message="Draft was modified by another editor.",
            server_etag=current_etag,
            server_version_number=draft.version_number,
            server_graph=draft.graph_json or {},
            updated_by=draft.created_by,
            updated_at=draft.created_at.isoformat() if draft.created_at else None,
        )

    is_valid, val_json = await _validate_graph_payload(session, hub_id, graph, strict=False)

    draft.graph_json = graph
    draft.is_valid = is_valid
    draft.validation_json = val_json
    if change_note:
        draft.change_note = change_note
    draft.created_by = actor_id

    wf.updated_at = datetime.utcnow()
    await session.commit()

    new_etag = compute_etag(draft)
    return DraftUpdateResult(
        version=draft,
        etag=new_etag,
        is_valid=is_valid,
        validation_json=val_json,
    )


async def publish(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
    actor_id: str,
    change_note: Optional[str] = None,
) -> WorkflowVersion:
    """Freeze draft version, update published_version_id, and reset draft_version_id."""
    hub, wf = await _get_workflow_with_lock(session, hub_id, workflow_id, for_update=True)
    if hub.is_archived:
        raise HubArchivedError(f"Cannot publish workflow: Hub '{hub_id}' is archived.")

    draft = await get_draft(session, hub_id=hub_id, workflow_id=workflow_id)

    # Validate graph topology & references before publishing
    is_valid, val_json = await _validate_graph_payload(session, hub_id, draft.graph_json, strict=True)
    if not is_valid:
        errors_desc = "; ".join([e.get("message", "Validation error") for e in val_json.get("errors", [])])
        raise GraphValidationError(f"Cannot publish invalid workflow graph: {errors_desc}")

    # Freeze draft into published version
    if change_note:
        draft.change_note = change_note
    draft.is_valid = True
    draft.validation_json = val_json
    draft.created_by = actor_id

    wf.published_version_id = draft.id
    wf.draft_version_id = None
    wf.status = "published"
    wf.updated_at = datetime.utcnow()

    await session.commit()
    return draft


async def restore(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
    version_number: int,
    actor_id: str,
) -> WorkflowVersion:
    """Restore historical version as a brand-new draft version without altering history."""
    hub, wf = await _get_workflow_with_lock(session, hub_id, workflow_id, for_update=True)
    if hub.is_archived:
        raise HubArchivedError(f"Cannot restore version: Hub '{hub_id}' is archived.")

    stmt_target = select(WorkflowVersion).where(
        WorkflowVersion.workflow_id == workflow_id,
        WorkflowVersion.version_number == version_number,
    )
    target_ver = (await session.execute(stmt_target)).scalar_one_or_none()
    if not target_ver:
        raise ValueError(f"Version {version_number} not found for workflow '{workflow_id}'.")

    stmt_max = select(func.coalesce(func.max(WorkflowVersion.version_number), 0)).where(
        WorkflowVersion.workflow_id == workflow_id
    )
    max_ver = (await session.execute(stmt_max)).scalar() or 0
    next_ver = max_ver + 1

    is_valid, val_json = await _validate_graph_payload(session, hub_id, target_ver.graph_json, strict=False)

    new_draft = WorkflowVersion(
        id=str(uuid.uuid4()),
        workflow_id=workflow_id,
        version_number=next_ver,
        graph_json=target_ver.graph_json,
        change_note=f"Restored from v{version_number}",
        is_valid=is_valid,
        validation_json=val_json,
        created_by=actor_id,
        created_at=datetime.utcnow(),
    )
    session.add(new_draft)
    await session.flush()

    wf.draft_version_id = new_draft.id
    wf.updated_at = datetime.utcnow()
    await session.commit()
    return new_draft


async def duplicate(
    session: AsyncSession,
    *,
    hub_id: Optional[str] = None,
    source_hub_id: Optional[str] = None,
    workflow_id: str,
    new_name: Optional[str] = None,
    name_override: Optional[str] = None,
    actor_id: str,
    target_hub_id: Optional[str] = None,
) -> WorkflowDefinition:
    """Duplicate a workflow and its published/draft graph into a target hub."""
    actual_source_hub_id = source_hub_id or hub_id
    if not actual_source_hub_id:
        raise ValueError("source_hub_id or hub_id is required")
    src_hub, src_wf = await _get_workflow_with_lock(session, actual_source_hub_id, workflow_id, for_update=False)

    actual_name = name_override or new_name or f"{src_wf.name} (Copy)"
    dest_hub_id = target_hub_id or actual_source_hub_id
    dest_hub = await _get_hub(session, dest_hub_id)
    if dest_hub.is_archived:
        raise HubArchivedError(f"Cannot duplicate workflow: Target Hub '{dest_hub_id}' is archived.")

    # Determine graph to copy (prefer published, fallback to draft)
    graph_to_copy = {"nodes": [], "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}
    ref_ver_id = src_wf.published_version_id or src_wf.draft_version_id
    if ref_ver_id:
        stmt_v = select(WorkflowVersion).where(WorkflowVersion.id == ref_ver_id)
        ver_obj = (await session.execute(stmt_v)).scalar_one_or_none()
        if ver_obj and ver_obj.graph_json:
            graph_to_copy = ver_obj.graph_json

    # Generate unique slug in target hub
    base_slug = _slugify(actual_name)
    candidate_slug = base_slug
    suffix = 1
    while True:
        stmt_exist = select(WorkflowDefinition).where(
            WorkflowDefinition.hub_id == dest_hub_id,
            WorkflowDefinition.slug == candidate_slug,
        )
        existing = (await session.execute(stmt_exist)).scalar_one_or_none()
        if not existing:
            break
        candidate_slug = f"{base_slug}-{suffix}"
        suffix += 1

    new_wf_id = str(uuid.uuid4())
    new_wf = WorkflowDefinition(
        id=new_wf_id,
        hub_id=dest_hub_id,
        name=actual_name,
        slug=candidate_slug,
        description=f"Duplicated from {src_wf.name}",
        tags_json=src_wf.tags_json or [],
        status="draft",
        created_by=actor_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    session.add(new_wf)
    await session.flush()

    is_valid, val_json = await _validate_graph_payload(session, dest_hub_id, graph_to_copy, strict=False)
    new_v1 = WorkflowVersion(
        id=str(uuid.uuid4()),
        workflow_id=new_wf_id,
        version_number=1,
        graph_json=graph_to_copy,
        change_note=f"Duplicated from {src_wf.name} (id: {src_wf.id})",
        is_valid=is_valid,
        validation_json=val_json,
        created_by=actor_id,
        created_at=datetime.utcnow(),
    )
    session.add(new_v1)
    await session.flush()

    new_wf.draft_version_id = new_v1.id
    await session.commit()
    return new_wf


async def list_versions(session: AsyncSession, *, hub_id: str, workflow_id: str) -> List[WorkflowVersion]:
    """List all version history rows for a workflow ordered by version_number ASC."""
    hub, wf = await _get_workflow_with_lock(session, hub_id, workflow_id, for_update=False)
    stmt = (
        select(WorkflowVersion)
        .where(WorkflowVersion.workflow_id == workflow_id)
        .order_by(WorkflowVersion.version_number.asc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def list_workflows(
    session: AsyncSession,
    *,
    hub_id: str,
    q: Optional[str] = None,
    tag: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[WorkflowDefinition]:
    """List workflows within the hub with filtering and pagination."""
    await _get_hub(session, hub_id)
    stmt = select(WorkflowDefinition).where(WorkflowDefinition.hub_id == hub_id)
    if q:
        stmt = stmt.where(WorkflowDefinition.name.ilike(f"%{q}%"))
    if status:
        stmt = stmt.where(WorkflowDefinition.status == status)
    stmt = stmt.order_by(WorkflowDefinition.updated_at.desc()).limit(limit).offset(offset)
    wfs = (await session.execute(stmt)).scalars().all()
    results = []
    for wf in wfs:
        if tag and tag not in (wf.tags_json or []):
            continue
        results.append(wf)
    return results


async def create_workflow(
    session: AsyncSession,
    *,
    hub_id: str,
    payload: Any,
    actor_id: str,
) -> WorkflowDefinition:
    """Create a new WorkflowDefinition with draft v1."""
    hub = await _get_hub(session, hub_id)
    if hub.is_archived:
        raise HubArchivedError(f"Cannot create workflow: Hub '{hub_id}' is archived.")

    name = payload.name
    slug = payload.slug or _slugify(name)
    stmt_chk = select(WorkflowDefinition).where(
        WorkflowDefinition.hub_id == hub_id,
        WorkflowDefinition.slug == slug,
    )
    if (await session.execute(stmt_chk)).scalar_one_or_none():
        raise ValueError(f"WORKFLOW_SLUG_TAKEN: Workflow with slug '{slug}' already exists in hub.")

    wf_id = str(uuid.uuid4())
    tags = getattr(payload, "tags_json", None) if getattr(payload, "tags_json", None) is not None else getattr(payload, "tags", []) or []

    valid_actor = actor_id
    if valid_actor:
        u_res = await session.execute(select(User.id).where(User.id == valid_actor))
        if not u_res.scalar_one_or_none():
            admin_u = (await session.execute(select(User.id).where(User.platform_role == "admin", User.is_deleted.is_(False)))).scalars().first()
            valid_actor = admin_u if admin_u else None

    wf = WorkflowDefinition(
        id=wf_id,
        hub_id=hub_id,
        name=name,
        slug=slug,
        description=getattr(payload, "description", None),
        tags_json=tags,
        status="draft",
        created_by=valid_actor,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    session.add(wf)
    await session.flush()

    ver_id = str(uuid.uuid4())
    raw_graph = getattr(payload, "graph", None)
    if raw_graph is not None:
        initial_graph = raw_graph.model_dump() if hasattr(raw_graph, "model_dump") else raw_graph
    else:
        initial_graph = {"nodes": [], "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}

    is_valid, val_json = await _validate_graph_payload(session, hub_id, initial_graph, strict=False)
    ver = WorkflowVersion(
        id=ver_id,
        workflow_id=wf_id,
        version_number=1,
        graph_json=initial_graph,
        change_note="Initial draft",
        is_valid=is_valid,
        validation_json=val_json,
        created_by=valid_actor,
        created_at=datetime.utcnow(),
    )
    session.add(ver)
    await session.flush()
    wf.draft_version_id = ver_id
    await session.commit()
    return wf


async def get_workflow(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
) -> Any:
    """Fetch workflow metadata & version details."""
    _, wf = await _get_workflow_with_lock(session, hub_id, workflow_id, for_update=False)
    return wf


async def update_workflow(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
    payload: Any,
) -> WorkflowDefinition:
    """Update workflow metadata (name, description, tags, slug)."""
    hub, wf = await _get_workflow_with_lock(session, hub_id, workflow_id, for_update=True)
    if hub.is_archived:
        raise HubArchivedError(f"Cannot update workflow: Hub '{hub_id}' is archived.")

    if getattr(payload, "slug", None) and payload.slug != wf.slug:
        stmt_chk = select(WorkflowDefinition).where(
            WorkflowDefinition.hub_id == hub_id,
            WorkflowDefinition.slug == payload.slug,
        )
        if (await session.execute(stmt_chk)).scalar_one_or_none():
            raise ValueError(f"WORKFLOW_SLUG_TAKEN: Workflow with slug '{payload.slug}' already exists in hub.")
        wf.slug = payload.slug

    if getattr(payload, "name", None) is not None:
        wf.name = payload.name
    if getattr(payload, "description", None) is not None:
        wf.description = payload.description
    
    tags = getattr(payload, "tags_json", None) if getattr(payload, "tags_json", None) is not None else getattr(payload, "tags", None)
    if tags is not None:
        wf.tags_json = tags

    wf.updated_at = datetime.utcnow()
    await session.commit()
    return wf


async def delete_workflow(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
) -> None:
    """Delete a workflow definition and all associated versions & runs."""
    hub, wf = await _get_workflow_with_lock(session, hub_id, workflow_id, for_update=True)
    if hub.is_archived:
        raise HubArchivedError(f"Cannot delete workflow: Hub '{hub_id}' is archived.")

    await session.delete(wf)
    await session.commit()


async def get_version(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
    version_number: int,
) -> WorkflowVersion:
    """Fetch specific WorkflowVersion by version_number."""
    await _get_workflow_with_lock(session, hub_id, workflow_id, for_update=False)
    stmt = select(WorkflowVersion).where(
        WorkflowVersion.workflow_id == workflow_id,
        WorkflowVersion.version_number == version_number,
    )
    ver = (await session.execute(stmt)).scalar_one_or_none()
    if not ver:
        raise VersionNotFoundError(f"Version '{version_number}' not found for workflow '{workflow_id}'.")
    return ver
