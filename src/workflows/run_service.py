"""Run Orchestration, Persistence & SSE Streaming for Workflow Hub (S6-06d).

Execution flow:
1. start_run()   — creates WorkflowRun row, launches _execute_run_task() in background.
2. _execute_run_task() — compiles graph via GraphParser.build_langgraph(), seeds
   GraphState from input_json, calls compiled_graph.ainvoke(), then persists one
   WorkflowRunStep row per node with input/output state, latency, and status.
3. stream_run()  — async generator yielding SSE events to callers.
"""

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Literal

from sqlalchemy import select, update, insert
from sqlalchemy.ext.asyncio import AsyncSession

from common.models.database import (
    WorkflowDefinition,
    WorkflowVersion,
    WorkflowRun,
    WorkflowRunStep,
)
from common.schemas.workflows import (
    WorkflowRunSummary,
    WorkflowRunDetail,
    WorkflowRunStepSummary,
)
from common.services.hub_repository import get_hub
from common.clients.postgres import get_sessionmaker
from projects.evalops.src.runner.trace_collector import TraceCollector
from projects.guardroute.src.core.graph_parser import (
    GraphParser,
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


def _json_sanitize(obj: Any) -> Any:
    """Recursively sanitize objects into JSON-serializable Python primitives."""
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(item) for item in obj]
    if hasattr(obj, "content"):
        # LangChain BaseMessage / AIMessage / HumanMessage
        return {"role": getattr(obj, "type", "message"), "content": getattr(obj, "content", str(obj))}
    if hasattr(obj, "model_dump"):
        return _json_sanitize(obj.model_dump())
    if hasattr(obj, "dict"):
        return _json_sanitize(obj.dict())
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return str(obj)


def redact_secrets(obj: Any) -> Any:
    """Recursively strip secret values from dictionaries and lists, replacing values with '***'."""
    obj = _json_sanitize(obj)
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
_RUN_TASKS: Dict[str, asyncio.Task] = {}


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
    # 1. Drain any already-buffered events
    for evt in list(_RUN_EVENT_BUFFERS.get(run_id, [])):
        yield evt
        if evt.get("event") == "run_end":
            return

    # 2. Subscribe to live events
    q: asyncio.Queue = asyncio.Queue()
    if run_id not in _RUN_LISTENERS:
        _RUN_LISTENERS[run_id] = set()
    _RUN_LISTENERS[run_id].add(q)

    try:
        while True:
            try:
                evt = await asyncio.wait_for(q.get(), timeout=5.0)
            except asyncio.TimeoutError:
                for b_evt in list(_RUN_EVENT_BUFFERS.get(run_id, [])):
                    if b_evt.get("event") == "run_end":
                        yield b_evt
                        return
                break
            q.task_done()
            yield evt
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

    if session_factory is not None:
        sf = session_factory
    elif session is not None and getattr(session, "bind", None) is not None:
        from sqlalchemy.ext.asyncio import async_sessionmaker
        sf = async_sessionmaker(bind=session.bind, class_=AsyncSession, expire_on_commit=False)
    else:
        sf = get_sessionmaker()
    if sf is not None:
        global_trace_collector.db_session_factory = sf
    task = asyncio.create_task(
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
    _RUN_TASKS[run_id] = task

    return run


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seed_graph_state(input_json: Dict[str, Any]) -> Dict[str, Any]:
    """Map input_json into the GraphState seed dict.

    The top-level 'input' key (or 'prompt') is mapped to 'prompt';
    everything else is passed through as additional state keys.
    """
    state: Dict[str, Any] = {}
    state["prompt"] = (
        input_json.get("input")
        or input_json.get("prompt")
        or ""
    )
    state["subagent_results"] = []
    state["final_response"] = None
    state["webhook_results"] = {}
    state["api_call_results"] = {}
    state["eval_results"] = {}
    state["transform_outputs"] = {}
    state["mcp_tool_results"] = {}
    state["conditional_flags"] = {}
    state["errors"] = {}
    state["db_query_results"] = {}
    state["db_store_results"] = {}
    state["tool_results"] = {}
    state["session_id"] = input_json.get("session_id") or str(uuid.uuid4())
    state["complexity"] = input_json.get("complexity", "LOW")
    state["required_agents"] = input_json.get("required_agents", [])
    state["token_usage"] = input_json.get("token_usage", {"input": 0, "output": 0})
    # Pass through any extra user-supplied keys
    for k, v in input_json.items():
        if k not in ("input", "prompt", "session_id", "complexity", "required_agents", "token_usage"):
            state[k] = v
    return state


def _build_topo_order(nodes: List[Dict], edges: List[Dict]) -> List[str]:
    """Return a Kahn's-algorithm topological order of node IDs.

    Falls back to the node insertion order if the graph has cycles
    (validation should have already caught that, but we defend here).
    """
    node_ids = [n["id"] for n in nodes if "id" in n]
    adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src in adj and tgt in adj:
            adj[src].append(tgt)
            in_degree[tgt] += 1

    queue = [nid for nid in node_ids if in_degree[nid] == 0]
    order: List[str] = []
    in_degree_copy = dict(in_degree)
    while queue:
        curr = queue.pop(0)
        order.append(curr)
        for nxt in adj[curr]:
            in_degree_copy[nxt] -= 1
            if in_degree_copy[nxt] == 0:
                queue.append(nxt)
    # If cycle prevented full traversal fall back to full node list
    if len(order) < len(node_ids):
        remaining = [nid for nid in node_ids if nid not in set(order)]
        order.extend(remaining)
    return order


def _has_error_edge(node_id: str, edges: List[Dict]) -> bool:
    """Return True if node_id has an outgoing edge from the 'error' handle."""
    for e in edges:
        if e.get("source") == node_id and e.get("sourceHandle") == "error":
            return True
    return False


async def _persist_run_step(
    session_factory: Any,
    *,
    hub_id: str,
    run_id: str,
    workflow_id: str,
    node_id: str,
    node_type: str,
    sequence: int,
    status: str,
    input_state: Dict[str, Any],
    output_state: Dict[str, Any],
    error_json: Optional[Dict[str, Any]],
    started_at: datetime,
    finished_at: datetime,
    latency_ms: float,
) -> None:
    """Persist a single WorkflowRunStep row."""
    try:
        async with session_factory() as session:
            step = WorkflowRunStep(
                id=str(uuid.uuid4()),
                hub_id=hub_id,
                run_id=run_id,
                workflow_id=workflow_id,
                node_id=node_id,
                node_type=node_type,
                sequence=sequence,
                status=status,
                input_state=input_state,
                output_state=output_state,
                error_json=error_json,
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
            )
            session.add(step)
            await session.flush()
            try:
                await session.commit()
            except Exception:
                pass
    except Exception as exc:
        logger.warning(
            "Failed to persist WorkflowRunStep node=%s run=%s: %s",
            node_id,
            run_id,
            exc,
        )


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
    """Background task: compile and execute the real LangGraph, persist step telemetry, emit SSE."""
    start_time = datetime.utcnow()
    nodes: List[Dict] = graph_json.get("nodes", []) if isinstance(graph_json, dict) else []
    edges: List[Dict] = graph_json.get("edges", []) if isinstance(graph_json, dict) else []
    node_map: Dict[str, Dict] = {n["id"]: n for n in nodes if "id" in n}
    topo_order: List[str] = _build_topo_order(nodes, edges)

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

    try:
        async with session_factory() as session:
            stmt = update(WorkflowRun).where(WorkflowRun.id == run_id).values(status="running")
            await session.execute(stmt)
            await session.flush()
            try:
                await session.commit()
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Failed to update status to running for run %s: %s", run_id, exc)

    run_status = "succeeded"
    error_info: Optional[Dict[str, Any]] = None
    output_data: Dict[str, Any] = {}
    final_state: Dict[str, Any] = {}

    try:
        # ------------------------------------------------------------------
        # Build the compiled LangGraph from the visual graph JSON
        # ------------------------------------------------------------------
        parser = GraphParser(graph_json, session_factory=session_factory)
        compiled_graph = parser.build_langgraph()

        initial_state = _seed_graph_state(input_json)
        initial_state["hub_id"] = hub_id
        initial_state["session_factory"] = session_factory

        async def _run_inner():
            nonlocal run_status, error_info, output_data, final_state

            # Validate cross-hub references at run start
            async with session_factory() as session:
                ref_issues = await parser.validate_references(
                    graph_json, session=session, source_hub_id=hub_id
                )
            blocking = [
                i for i in ref_issues
                if i.code in ("HUB_LINK_REQUIRED", "HUB_LINK_REVOKED", "REFERENCE_TARGET_MISSING")
            ]
            if blocking:
                issue = blocking[0]
                error_info = {"code": issue.code, "message": issue.message, "node_id": issue.node_id}
                run_status = "failed"
                return

            # Check cancellation before long invocation
            if run_id in _CANCELLED_RUNS:
                run_status = "cancelled"
                return

            # ------------------------------------------------------------------
            # Emit node_start events in topo order, then invoke the full graph.
            # Per-node SSE events reflect the logical execution order rather
            # than actual async scheduling (which LangGraph handles internally).
            # ------------------------------------------------------------------
            seq = 0
            node_start_times: Dict[str, datetime] = {}

            for nid in topo_order:
                if run_id in _CANCELLED_RUNS:
                    run_status = "cancelled"
                    return
                node_start_times[nid] = datetime.utcnow()
                ntype = node_map[nid].get("type", "AgentNode") if nid in node_map else "unknown"
                _publish_event(
                    run_id,
                    "node_start",
                    {
                        "run_id": run_id,
                        "node_id": nid,
                        "node_type": ntype,
                        "sequence": seq,
                        "started_at": node_start_times[nid].isoformat(),
                    },
                )
                seq += 1

            # ------------------------------------------------------------------
            # Execute the compiled LangGraph
            # ------------------------------------------------------------------
            try:
                invocation_start = datetime.utcnow()
                final_state = await compiled_graph.ainvoke(initial_state)
                invocation_end = datetime.utcnow()
                run_status = "succeeded"
            except Exception as graph_exc:
                run_status = "failed"
                error_info = {
                    "code": "GRAPH_EXECUTION_ERROR",
                    "message": str(graph_exc),
                    "node_id": None,
                }
                logger.exception("LangGraph invocation failed for run %s", run_id)
                return

            output_data = dict(final_state)

            # ------------------------------------------------------------------
            # Emit node_end events + persist WorkflowRunStep per node
            # ------------------------------------------------------------------
            total_duration_s = (invocation_end - invocation_start).total_seconds()
            per_node_ms = (
                (total_duration_s * 1000.0 / len(topo_order)) if topo_order else 0.0
            )

            for i, nid in enumerate(topo_order):
                if nid not in node_map:
                    continue
                ntype = node_map[nid].get("type", "AgentNode")
                node_start = node_start_times.get(nid, invocation_start)
                node_latency = per_node_ms
                node_finish = datetime.utcnow()

                # Extract per-node output from final_state where possible
                node_output: Dict[str, Any] = {}
                for key in (
                    "subagent_results",
                    "transform_outputs",
                    "webhook_results",
                    "api_call_results",
                    "eval_results",
                    "mcp_tool_results",
                ):
                    val = final_state.get(key)
                    if isinstance(val, dict) and nid in val:
                        node_output[key] = val[nid]

                # Include final_response on last node
                if i == len(topo_order) - 1 and final_state.get("final_response"):
                    node_output["final_response"] = final_state["final_response"]

                node_error = final_state.get("errors", {}).get(nid)
                node_status = "failed" if node_error else "succeeded"

                # Expose per-node output under {node_id}_output for downstream
                # consumers (matches the pre-LangGraph run contract).
                output_data[f"{nid}_output"] = node_output

                redacted_in = redact_secrets(dict(initial_state))
                redacted_out = redact_secrets(node_output)

                out_str = json.dumps(redacted_out)
                out_preview = out_str[:2000] if len(out_str) > 2000 else out_str

                _publish_event(
                    run_id,
                    "node_end",
                    {
                        "run_id": run_id,
                        "node_id": nid,
                        "node_type": ntype,
                        "status": node_status,
                        "latency_ms": node_latency,
                        "output_preview": out_preview,
                        "sequence": i,
                    },
                )

                # EvalFlowTrace emission (existing infrastructure)
                global_trace_collector.emit_event(
                    {
                        "hub_id": hub_id,
                        "run_id": run_id,
                        "workflow_id": workflow_id,
                        "node_id": nid,
                        "node_type": ntype,
                        "sequence": i,
                        "input_state": redacted_in,
                        "output_state": redacted_out,
                        "latency_ms": node_latency,
                        "timestamp": node_finish,
                    }
                )

                # Persist WorkflowRunStep
                await _persist_run_step(
                    session_factory,
                    hub_id=hub_id,
                    run_id=run_id,
                    workflow_id=workflow_id,
                    node_id=nid,
                    node_type=ntype,
                    sequence=i,
                    status=node_status,
                    input_state=redacted_in,
                    output_state=redacted_out,
                    error_json={"error": str(node_error)} if node_error else None,
                    started_at=node_start,
                    finished_at=node_finish,
                    latency_ms=node_latency,
                )

        await asyncio.wait_for(_run_inner(), timeout=float(timeout_s))

    except asyncio.TimeoutError:
        run_status = "failed"
        error_info = {"code": "RUN_TIMEOUT", "message": f"RUN_TIMEOUT after {timeout_s}s", "node_id": None}
    except Exception as exc:
        run_status = "failed"
        error_info = {"code": "EXECUTION_ERROR", "message": str(exc), "node_id": None}

    finish_time = datetime.utcnow()
    duration_ms = int((finish_time - start_time).total_seconds() * 1000.0)

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

    try:
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
            await session.flush()
            try:
                await session.commit()
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Failed to persist final WorkflowRun update for run %s: %s", run_id, exc)


async def cancel_run(
    session: AsyncSession,
    *,
    hub_id: str,
    run_id: str,
    actor_id: str,
) -> WorkflowRunDetail:
    """Cancel a running workflow execution."""
    stmt = (
        select(WorkflowRun)
        .where(
            WorkflowRun.hub_id == hub_id,
            WorkflowRun.id == run_id,
        )
        .execution_options(populate_existing=True)
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
    await session.flush()
    try:
        await session.commit()
    except Exception:
        pass

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
    return await get_run(session, hub_id=hub_id, run_id=run_id)


async def get_run(session: AsyncSession, *, hub_id: str, run_id: str) -> WorkflowRunDetail:
    """Fetch WorkflowRunDetail enforcing hub scoping."""
    session.expire_all()
    stmt = select(WorkflowRun).where(
        WorkflowRun.hub_id == hub_id,
        WorkflowRun.id == run_id,
    ).execution_options(populate_existing=True)
    run = (await session.execute(stmt)).scalar_one_or_none()
    if not run:
        raise RunNotFoundError(f"Workflow run '{run_id}' not found in hub '{hub_id}'.")

    from sqlalchemy import select as sa_select
    from common.models.database import WorkflowRunStep as WRStep
    steps_stmt = (
        sa_select(WRStep)
        .where(WRStep.hub_id == hub_id, WRStep.run_id == run_id)
        .order_by(WRStep.sequence)
    )
    steps = (await session.execute(steps_stmt)).scalars().all()

    detail = WorkflowRunDetail.model_validate(run)
    detail.steps = [WorkflowRunStepSummary.model_validate(s) for s in steps]
    return detail


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
