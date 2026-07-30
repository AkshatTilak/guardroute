"""Run Orchestration, Persistence & SSE Streaming for Workflow Hub (S6-06d)."""

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Literal

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from common.models.database import (
    Hub,
    WorkflowDefinition,
    WorkflowVersion,
    WorkflowRun,
    EvalFlowTrace,
)
from common.schemas.workflows import (
    WorkflowRunSummary,
    WorkflowRunDetail,
)
from common.services.hub_repository import get_hub
from common.clients.postgres import get_sessionmaker
from projects.evalops.src.runner.trace_collector import TraceCollector
from projects.guardroute.src.core.graph_parser import (
    GraphParser,
    GraphValidationError,
    validate_workflow_graph,
)

logger = logging.getLogger("guardroute.workflows.run_service")


class WorkflowNotFoundError(Exception):
    """Raised when a workflow is not found within the specified hub."""
    pass


class WorkflowNotPublishedError(Exception):
    """Raised when starting a published run for an un-published workflow."""
    pass


class RunNotFoundError(Exception):
    """Raised when a workflow run is not found within the specified hub."""
    pass


class RunNotCancellableError(Exception):
    """Raised when attempting to cancel a run already in a terminal state."""
    pass


class HubArchivedError(Exception):
    """Raised when mutating an archived hub."""
    pass


SECRET_KEY_PATTERN = re.compile(r"(?i)(authorization|api[_-]?key|token|password|secret)")


def redact_secrets(obj: Any) -> Any:
    """Recursively strip secret values from dictionaries and lists, replacing values with '***'."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if SECRET_KEY_PATTERN.search(str(k)):
                new_dict[k] = "***"
            else:
                new_dict[k] = redact_secrets(v)
        return new_dict
    elif isinstance(obj, list):
        return [redact_secrets(item) for item in obj]
    return obj


# Global trace collector instance
global_trace_collector = TraceCollector(db_session_factory=get_sessionmaker())

# Event bus structures for SSE streaming
_RUN_EVENT_BUFFERS: Dict[str, List[Dict[str, Any]]] = {}
_RUN_LISTENERS: Dict[str, Set[asyncio.Queue]] = {}
_CANCELLED_RUNS: Set[str] = set()


def _publish_event(run_id: str, event_name: str, payload: Dict[str, Any]) -> None:
    """Buffer and broadcast an SSE event payload to all active run listeners."""
    frame = {"event": event_name, "data": payload}
    if run_id not in _RUN_EVENT_BUFFERS:
        _RUN_EVENT_BUFFERS[run_id] = []
    _RUN_EVENT_BUFFERS[run_id].append(frame)

    listeners = _RUN_LISTENERS.get(run_id, set())
    for q in list(listeners):
        try:
            q.put_nowait(frame)
        except Exception:
            pass


async def stream_run(run_id: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Async generator streaming SSE events for run_id."""
    q: asyncio.Queue = asyncio.Queue()
    if run_id not in _RUN_LISTENERS:
        _RUN_LISTENERS[run_id] = set()
    _RUN_LISTENERS[run_id].add(q)

    try:
        buffered = list(_RUN_EVENT_BUFFERS.get(run_id, []))
        for evt in buffered:
            yield evt
            if evt.get("event") == "run_end":
                return

        while True:
            evt = await q.get()
            yield evt
            q.task_done()
            if evt.get("event") == "run_end":
                break
    finally:
        if run_id in _RUN_LISTENERS:
            _RUN_LISTENERS[run_id].discard(q)
            if not _RUN_LISTENERS[run_id]:
                del _RUN_LISTENERS[run_id]


async def start_run(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
    input_json: Dict[str, Any],
    trigger: Literal["manual", "api", "eval", "schedule"] = "manual",
    started_by: Optional[str] = None,
    use_draft: bool = False,
    timeout_s: int = 300,
    session_factory: Optional[Any] = None,
) -> WorkflowRun:
    """Start workflow execution run, persist queued state, and launch async background runner."""
    hub = await get_hub(session, hub_id)
    if not hub:
        raise ValueError(f"Hub '{hub_id}' not found.")
    if hub.is_archived:
        raise HubArchivedError(f"Cannot start run: Hub '{hub_id}' is archived.")

    stmt_wf = select(WorkflowDefinition).where(
        WorkflowDefinition.hub_id == hub_id,
        WorkflowDefinition.id == workflow_id,
    )
    wf = (await session.execute(stmt_wf)).scalar_one_or_none()
    if not wf:
        raise WorkflowNotFoundError(f"Workflow '{workflow_id}' not found in hub '{hub_id}'.")

    input_payload = dict(input_json) if isinstance(input_json, dict) else {}

    if use_draft:
        version_id = wf.draft_version_id
        if not version_id:
            from projects.guardroute.src.workflows.version_service import get_draft
            draft_ver = await get_draft(session, hub_id=hub_id, workflow_id=workflow_id)
            version_id = draft_ver.id
        trigger = "manual"
        input_payload["_dry_run"] = True
    else:
        version_id = wf.published_version_id
        if not version_id:
            raise WorkflowNotPublishedError(f"Workflow '{workflow_id}' is not published.")

    stmt_ver = select(WorkflowVersion).where(WorkflowVersion.id == version_id)
    ver = (await session.execute(stmt_ver)).scalar_one_or_none()
    if not ver:
        raise ValueError(f"Workflow version '{version_id}' not found.")

    nodes = ver.graph_json.get("nodes", []) if isinstance(ver.graph_json, dict) else []
    node_count = len(nodes)

    run_id = str(uuid.uuid4())
    run = WorkflowRun(
        id=run_id,
        hub_id=hub_id,
        workflow_id=workflow_id,
        version_id=version_id,
        trigger=trigger,
        input_json=input_payload,
        status="queued",
        node_count=node_count,
        started_by=started_by,
        started_at=datetime.utcnow(),
    )
    session.add(run)
    await session.commit()

    if not global_trace_collector._is_running:
        await global_trace_collector.start()

    sf = session_factory or get_sessionmaker()
    if session_factory is not None:
        global_trace_collector.db_session_factory = session_factory
    asyncio.create_task(
        _execute_run_task(
            run_id=run_id,
            hub_id=hub_id,
            workflow_id=workflow_id,
            version_id=version_id,
            version_number=ver.version_number,
            graph_json=ver.graph_json or {},
            input_json=input_payload,
            timeout_s=timeout_s,
            session_factory=sf,
        )
    )

    return run


async def _execute_run_task(
    *,
    run_id: str,
    hub_id: str,
    workflow_id: str,
    version_id: str,
    version_number: int,
    graph_json: Dict[str, Any],
    input_json: Dict[str, Any],
    timeout_s: int,
    session_factory: Any,
) -> None:
    """Background task executing the graph step-by-step with state persistence, trace emission, & SSE."""
    start_time = datetime.utcnow()
    nodes = graph_json.get("nodes", []) if isinstance(graph_json, dict) else []
    edges = graph_json.get("edges", []) if isinstance(graph_json, dict) else []
    node_map = {n["id"]: n for n in nodes if "id" in n}

    _publish_event(
        run_id,
        "run_start",
        {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "version_id": version_id,
            "version_number": version_number,
            "node_count": len(nodes),
        },
    )

    async with session_factory() as session:
        stmt = update(WorkflowRun).where(WorkflowRun.id == run_id).values(status="running")
        await session.execute(stmt)
        await session.commit()

    run_status = "succeeded"
    error_info: Optional[Dict[str, Any]] = None
    output_data: Dict[str, Any] = {}
    seq = 0

    try:
        async def _run_inner():
            nonlocal seq, run_status, error_info, output_data
            current_state = dict(input_json)

            adj: Dict[str, List[str]] = {nid: [] for nid in node_map}
            in_degree: Dict[str, int] = {nid: 0 for nid in node_map}
            for e in edges:
                src, tgt = e.get("source"), e.get("target")
                if src in node_map and tgt in node_map:
                    adj[src].append(tgt)
                    in_degree[tgt] += 1

            queue = [nid for nid in node_map if in_degree[nid] == 0]
            if not queue and node_map:
                queue = list(node_map.keys())

            visited = set()
            while queue:
                if run_id in _CANCELLED_RUNS:
                    run_status = "cancelled"
                    return

                curr_id = queue.pop(0)
                if curr_id in visited:
                    continue
                visited.add(curr_id)

                curr_node = node_map[curr_id]
                ntype = curr_node.get("type", "AgentNode")

                node_start_time = datetime.utcnow()
                _publish_event(
                    run_id,
                    "node_start",
                    {
                        "run_id": run_id,
                        "node_id": curr_id,
                        "node_type": ntype,
                        "sequence": seq,
                        "started_at": node_start_time.isoformat(),
                    },
                )

                # Re-validate qualified reference at node execution time
                async with session_factory() as session:
                    parser = GraphParser(graph_json)
                    single_node_graph = {"nodes": [curr_node], "edges": []}
                    ref_issues = await parser.validate_references(single_node_graph, session=session, source_hub_id=hub_id)
                    if any(i.code in ("HUB_LINK_REQUIRED", "HUB_LINK_REVOKED") for i in ref_issues):
                        ref_obj = curr_node.get("data", {}).get("reference", {})
                        target_hub = ref_obj.get("hub_id", "unknown")
                        res_id = ref_obj.get("resource_id", "unknown")
                        err_msg = f"HUB_LINK_REVOKED: node {curr_id} references {target_hub}/{res_id}"
                        error_info = {"code": "HUB_LINK_REVOKED", "message": err_msg, "node_id": curr_id}
                        run_status = "failed"
                        return

                node_latency = (datetime.utcnow() - node_start_time).total_seconds() * 1000.0
                node_output = {"result": f"Executed {ntype} {curr_id}", "status": "completed"}
                current_state[f"{curr_id}_output"] = node_output
                output_data = current_state

                redacted_in = redact_secrets(current_state)
                redacted_out = redact_secrets(node_output)

                global_trace_collector.emit_event(
                    {
                        "hub_id": hub_id,
                        "run_id": run_id,
                        "workflow_id": workflow_id,
                        "node_id": curr_id,
                        "node_type": ntype,
                        "sequence": seq,
                        "input_state": redacted_in,
                        "output_state": redacted_out,
                        "latency_ms": node_latency,
                        "timestamp": datetime.utcnow(),
                    }
                )

                out_str = json.dumps(redacted_out)
                out_preview = out_str[:2000] if len(out_str) > 2000 else out_str
                _publish_event(
                    run_id,
                    "node_end",
                    {
                        "run_id": run_id,
                        "node_id": curr_id,
                        "status": "succeeded",
                        "latency_ms": node_latency,
                        "output_preview": out_preview,
                        "sequence": seq,
                    },
                )
                seq += 1

                for nxt in adj[curr_id]:
                    in_degree[nxt] -= 1
                    if in_degree[nxt] == 0 and nxt not in visited:
                        queue.append(nxt)

        await asyncio.wait_for(_run_inner(), timeout=float(timeout_s))

    except asyncio.TimeoutError:
        run_status = "failed"
        error_info = {"code": "RUN_TIMEOUT", "message": f"RUN_TIMEOUT after {timeout_s}s", "node_id": None}
    except Exception as exc:
        run_status = "failed"
        error_info = {"code": "EXECUTION_ERROR", "message": str(exc), "node_id": None}

    finish_time = datetime.utcnow()
    duration_ms = int((finish_time - start_time).total_seconds() * 1000.0)

    async with session_factory() as session:
        stmt_update = (
            update(WorkflowRun)
            .where(WorkflowRun.id == run_id)
            .values(
                status=run_status,
                error_message=error_info["message"] if error_info else None,
                output_json=redact_secrets(output_data),
                duration_ms=duration_ms,
                finished_at=finish_time,
            )
        )
        await session.execute(stmt_update)
        await session.commit()

    _publish_event(
        run_id,
        "run_end",
        {
            "run_id": run_id,
            "status": run_status,
            "duration_ms": duration_ms,
            "output": redact_secrets(output_data),
            "error": error_info,
        },
    )


async def cancel_run(
    session: AsyncSession,
    *,
    hub_id: str,
    run_id: str,
    actor_id: str,
) -> WorkflowRun:
    """Cancel a running workflow execution."""
    stmt = select(WorkflowRun).where(
        WorkflowRun.hub_id == hub_id,
        WorkflowRun.id == run_id,
    )
    run = (await session.execute(stmt)).scalar_one_or_none()
    if not run:
        raise RunNotFoundError(f"Workflow run '{run_id}' not found in hub '{hub_id}'.")

    if run.status in {"succeeded", "failed", "cancelled"}:
        raise RunNotCancellableError(f"Run '{run_id}' is already in terminal state '{run.status}'.")

    _CANCELLED_RUNS.add(run_id)
    finish_time = datetime.utcnow()
    duration_ms = int((finish_time - run.started_at).total_seconds() * 1000.0) if run.started_at else 0

    run.status = "cancelled"
    run.finished_at = finish_time
    run.duration_ms = duration_ms
    await session.commit()

    _publish_event(
        run_id,
        "run_end",
        {
            "run_id": run_id,
            "status": "cancelled",
            "duration_ms": duration_ms,
            "output": None,
            "error": None,
        },
    )
    return run


async def get_run(session: AsyncSession, *, hub_id: str, run_id: str) -> WorkflowRunDetail:
    """Fetch WorkflowRunDetail enforcing hub scoping."""
    stmt = select(WorkflowRun).where(
        WorkflowRun.hub_id == hub_id,
        WorkflowRun.id == run_id,
    )
    run = (await session.execute(stmt)).scalar_one_or_none()
    if not run:
        raise RunNotFoundError(f"Workflow run '{run_id}' not found in hub '{hub_id}'.")
    return WorkflowRunDetail.model_validate(run)


async def list_runs(
    session: AsyncSession,
    *,
    hub_id: str,
    workflow_id: str,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[WorkflowRunSummary]:
    """List WorkflowRunSummary rows for a workflow ordered by started_at DESC."""
    stmt = (
        select(WorkflowRun)
        .where(
            WorkflowRun.hub_id == hub_id,
            WorkflowRun.workflow_id == workflow_id,
        )
    )
    if status:
        stmt = stmt.where(WorkflowRun.status == status)
    stmt = stmt.order_by(WorkflowRun.started_at.desc()).limit(limit).offset(offset)
    runs = (await session.execute(stmt)).scalars().all()
    return [WorkflowRunSummary.model_validate(r) for r in runs]


async def reconcile_orphaned_runs(session: AsyncSession) -> int:
    """Reconcile runs left in queued/running from previous process crash."""
    stmt = (
        update(WorkflowRun)
        .where(WorkflowRun.status.in_(["queued", "running"]))
        .values(
            status="failed",
            error_message="ORPHANED_RUN",
            finished_at=datetime.utcnow(),
        )
    )
    res = await session.execute(stmt)
    await session.commit()
    return res.rowcount
