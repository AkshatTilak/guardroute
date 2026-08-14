"""Workflow Portability Service (Export, Import, Duplicate & Templates) (S6-06f)."""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.models.database import (
    WorkflowDefinition,
    WorkflowVersion,
)
from common.services.hub_repository import get_hub
from common.services import hub_resolver
from projects.guardroute.src.core.graph_parser import (
    GraphParser,
    collect_references,
)

logger = logging.getLogger("guardroute.workflows.portability")

EXPORT_FORMAT_VERSION = "contained.workflow/v6"
SECRET_KEY_PATTERN = re.compile(r"(?i)(authorization|api[_-]?key|token|password|secret)")
TEMPLATES_DIR = Path(__file__).parent / "templates"


def _sanitize_secrets(obj: Any, path: str = "", warnings: Optional[List[str]] = None) -> Any:
    """Recursively strip credentials, tokens and headers from node data, replacing with None and adding warning."""
    if warnings is None:
        warnings = []

    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if SECRET_KEY_PATTERN.search(str(k)):
                new_dict[k] = None
                warnings.append(f"Sanitized secret field '{k}' at {path}")
            else:
                new_dict[k] = _sanitize_secrets(v, f"{path}.{k}" if path else k, warnings)
        return new_dict
    elif isinstance(obj, list):
        return [_sanitize_secrets(item, f"{path}[{idx}]", warnings) for idx, item in enumerate(obj)]
    return obj


async def export_workflow(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
    version_number: Optional[int] = None,
) -> Dict[str, Any]:
    """Export a workflow to a self-describing JSON document."""
    hub = await get_hub(session, hub_id)
    if not hub:
        raise ValueError(f"Hub '{hub_id}' not found.")

    stmt_wf = select(WorkflowDefinition).where(
        WorkflowDefinition.hub_id == hub_id,
        WorkflowDefinition.id == workflow_id,
    )
    wf = (await session.execute(stmt_wf)).scalar_one_or_none()
    if not wf:
        raise ValueError(f"Workflow '{workflow_id}' not found in hub '{hub_id}'.")

    if version_number is not None:
        stmt_ver = select(WorkflowVersion).where(
            WorkflowVersion.workflow_id == workflow_id,
            WorkflowVersion.version_number == version_number,
        )
    elif wf.published_version_id:
        stmt_ver = select(WorkflowVersion).where(WorkflowVersion.id == wf.published_version_id)
    else:
        stmt_ver = select(WorkflowVersion).where(WorkflowVersion.id == wf.draft_version_id)

    ver = (await session.execute(stmt_ver)).scalar_one_or_none()
    if not ver:
        raise ValueError(f"Workflow version for '{workflow_id}' not found.")

    graph_json = dict(ver.graph_json or {})
    warnings: List[str] = []
    sanitized_graph = _sanitize_secrets(graph_json, warnings=warnings)

    raw_refs = collect_references(sanitized_graph)
    dependencies: List[Dict[str, Any]] = []
    seen_keys = set()

    for node_id, ref in raw_refs:
        ref_key = f"{ref.type}:{ref.resource_id}"
        if ref_key not in seen_keys:
            seen_keys.add(ref_key)
            dependencies.append({
                "ref_key": ref_key,
                "type": ref.type,
                "source_hub_id": ref.hub_id,
                "resource_id": ref.resource_id,
                "node_ids": [n_id for n_id, r in raw_refs if r.resource_id == ref.resource_id],
            })

    return {
        "format": EXPORT_FORMAT_VERSION,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "source": {
            "hub_id": hub.id,
            "hub_slug": hub.slug,
            "hub_type": hub.hub_type,
        },
        "workflow": {
            "name": wf.name,
            "slug": wf.slug,
            "description": wf.description,
            "tags": wf.tags_json or [],
        },
        "version": {
            "version_number": ver.version_number,
            "change_note": ver.change_note,
            "graph": sanitized_graph,
        },
        "dependencies": dependencies,
        "warnings": warnings,
    }


async def plan_import(
    session: AsyncSession,
    *,
    target_hub_id: str,
    document: Dict[str, Any],
) -> Dict[str, Any]:
    """Dry run plan evaluating reference resolution for an import document without mutating state."""
    hub = await get_hub(session, target_hub_id)
    if not hub:
        raise ValueError(f"Target hub '{target_hub_id}' not found.")

    if document.get("format") != EXPORT_FORMAT_VERSION:
        raise ValueError(f"UNSUPPORTED_EXPORT_FORMAT: Format '{document.get('format')}' is not supported.")

    version_doc = document.get("version", {})
    graph = version_doc.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if len(nodes) > 200 or len(edges) > 400:
        raise ValueError("GRAPH_TOO_LARGE: Graph exceeds max limit of 200 nodes or 400 edges.")

    unresolved: List[Dict[str, Any]] = []
    deps_report: List[Dict[str, Any]] = []

    for dep in document.get("dependencies", []):
        ref_key = dep.get("ref_key")
        rtype = dep.get("type")
        res_id = dep.get("resource_id")
        src_hub = dep.get("source_hub_id", target_hub_id)

        try:
            res = await hub_resolver.resolve_linked(
                session,
                source_hub_id=target_hub_id,
                target_resource_type=rtype,
                target_resource_id=res_id,
            )
            deps_report.append({
                "ref_key": ref_key,
                "status": "resolved",
                "target_hub_id": res.hub_id,
                "resource_id": res.id,
            })
        except Exception as err:
            unresolved.append({
                "ref_key": ref_key,
                "node_ids": dep.get("node_ids", []),
                "reason": "HUB_LINK_REQUIRED",
                "message": str(err),
            })

    return {
        "valid": len(unresolved) == 0,
        "unresolved": unresolved,
        "dependencies": deps_report,
    }


async def import_workflow(
    session: AsyncSession,
    *,
    target_hub_id: str,
    document: Dict[str, Any],
    actor_id: str,
    mapping: Optional[Dict[str, str]] = None,
    name_override: Optional[str] = None,
) -> WorkflowDefinition:
    """Import a workflow JSON document into target hub as a draft workflow."""
    hub = await get_hub(session, target_hub_id)
    if not hub:
        raise ValueError(f"Target hub '{target_hub_id}' not found.")
    if hub.is_archived:
        raise ValueError(f"Cannot import: Target hub '{target_hub_id}' is archived.")

    if document.get("format") != EXPORT_FORMAT_VERSION:
        raise ValueError(f"UNSUPPORTED_EXPORT_FORMAT: Unsupported format '{document.get('format')}'.")

    wf_meta = document.get("workflow", {})
    name = name_override or wf_meta.get("name", "Imported Workflow")
    base_slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or "imported-workflow"

    # Derive unique slug in target hub
    slug = base_slug
    counter = 1
    while True:
        stmt_chk = select(WorkflowDefinition).where(
            WorkflowDefinition.hub_id == target_hub_id,
            WorkflowDefinition.slug == slug,
        )
        existing = (await session.execute(stmt_chk)).scalar_one_or_none()
        if not existing:
            break
        counter += 1
        slug = f"{base_slug}-{counter}"

    graph = dict(document.get("version", {}).get("graph", {}))

    # Apply reference mapping if provided
    if mapping:
        for node in graph.get("nodes", []):
            ref = node.get("data", {}).get("reference")
            if ref and isinstance(ref, dict):
                ref_key = f"{ref.get('type')}:{ref.get('resource_id')}"
                if ref_key in mapping:
                    ref["resource_id"] = mapping[ref_key]

    # Validate topology & references
    parser = GraphParser(graph)
    ref_issues = await parser.validate_references(graph, session=session, source_hub_id=target_hub_id)
    forbidden = [i for i in ref_issues if i.code in ("HUB_LINK_REQUIRED", "HUB_LINK_REVOKED")]
    if forbidden and not mapping:
        raise ValueError(f"IMPORT_UNRESOLVED_REFERENCES: {len(forbidden)} references could not be resolved.")

    wf_id = str(uuid.uuid4())
    wf = WorkflowDefinition(
        id=wf_id,
        hub_id=target_hub_id,
        name=name,
        slug=slug,
        description=wf_meta.get("description"),
        tags_json=wf_meta.get("tags", []),
        status="draft",
        created_by=actor_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    session.add(wf)
    await session.flush()

    ver_id = str(uuid.uuid4())
    ver = WorkflowVersion(
        id=ver_id,
        workflow_id=wf_id,
        version_number=1,
        graph_json=graph,
        change_note=f"Imported from {document.get('source', {}).get('hub_slug', 'export')}",
        is_valid=True,
        created_by=actor_id,
        created_at=datetime.utcnow(),
    )
    session.add(ver)
    await session.flush()
    wf.draft_version_id = ver_id
    await session.commit()
    return wf


async def list_templates() -> List[Dict[str, Any]]:
    """Return available seed workflow templates."""
    templates = []
    if not TEMPLATES_DIR.exists():
        return templates

    for f in TEMPLATES_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text("utf-8"))
            templates.append({
                "key": f.stem,
                "name": data.get("workflow", {}).get("name", f.stem),
                "description": data.get("workflow", {}).get("description"),
                "tags": data.get("workflow", {}).get("tags", []),
            })
        except Exception as e:
            logger.error(f"Failed to read template {f}: {e}")
    return templates


async def instantiate_template(
    session: AsyncSession,
    *,
    target_hub_id: str,
    template_key: str,
    actor_id: str,
    mapping: Optional[Dict[str, str]] = None,
) -> WorkflowDefinition:
    """Instantiate a workflow from a seed template."""
    file_path = TEMPLATES_DIR / f"{template_key}.json"
    if not file_path.exists():
        raise ValueError(f"Template '{template_key}' not found.")

    doc = json.loads(file_path.read_text("utf-8"))
    return await import_workflow(
        session,
        target_hub_id=target_hub_id,
        document=doc,
        actor_id=actor_id,
        mapping=mapping,
    )
