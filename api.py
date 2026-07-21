"""GuardRoute API routes.

Mounted at /api/guardroute/* by the gateway's dynamic route loader.
Provides task complexity classification and Scatter-Gather agent chat orchestration.
Supports token-by-token response streaming via Server-Sent Events (SSE).
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from projects.guardroute.src.orchestrator import (
    execute_orchestrator,
    execute_orchestrator_stream,
    run_classification
)
from common.clients.inference import InferenceServerError

router = APIRouter(tags=["guardroute"])
logger = logging.getLogger("guardroute.api")


class ChatRequest(BaseModel):
    """Payload containing user prompt and optional session ID."""

    prompt: str
    session_id: Optional[str] = None


@router.get("/status")
async def guardroute_status(request: Request) -> dict:
    """GuardRoute service status."""
    inference = getattr(request.app.state, "guardroute_inference", None)
    return {
        "project": "guardroute",
        "status": "active",
        "inference_connected": inference is not None,
    }


@router.post("/chat")
async def chat(request: Request, req: ChatRequest):
    """Main orchestration endpoint — supports streaming (SSE) and standard JSON response.
    
    If the 'Accept' header contains 'text/event-stream', this endpoint streams
    the response token-by-token using sse-starlette.
    """
    logger.info("Received chat orchestration request: '%s'", req.prompt[:40])
    
    accept_header = request.headers.get("Accept", "")
    if "text/event-stream" in accept_header:
        logger.info("Streaming response via SSE event source...")
        generator = execute_orchestrator_stream(req.prompt, req.session_id)
        return EventSourceResponse(generator)
        
    try:
        trace_result = await execute_orchestrator(req.prompt, req.session_id)
        if "error" in trace_result.get("complexity", "") or "Error" in trace_result.get("final_response", ""):
            # If internal execution indicates failure
            raise HTTPException(status_code=500, detail=trace_result.get("final_response"))

        return {
            "status": "success",
            "session_id": trace_result.get("session_id"),
            "complexity": trace_result.get("complexity"),
            "subagents_ran": trace_result.get("subagents_ran"),
            "response": trace_result.get("final_response"),
            "latency_sec": trace_result.get("duration_sec"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat orchestration pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Orchestration pipeline failed: {str(e)}")


@router.post("/classify")
async def classify_task(request: Request, req: ChatRequest) -> dict:
    """Classify a prompt's complexity and required subagents by forwarding to the inference server."""
    logger.info("Received classification query: '%s'", req.prompt[:40])
    inference_client = request.app.state.guardroute_inference
    if not inference_client:
        raise HTTPException(status_code=503, detail="Inference client is unavailable.")
        
    try:
        result = await run_classification(req.prompt, inference_client)
        return {
            "status": "success",
            "prompt": req.prompt,
            "complexity": result.get("complexity"),
            "required_agents": result.get("required_agents"),
            "confidence": result.get("confidence", 1.0)
        }
    except InferenceServerError as e:
        logger.error("Inference server error during classification: %s", e)
        raise HTTPException(status_code=503, detail=f"Inference server is currently offline: {str(e)}")
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Classification query failed: {str(e)}")


# --- V2 Visual Workflow CRUD REST APIs ---

import uuid
from datetime import datetime
from typing import List
from fastapi import Depends
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from common.clients.postgres import get_async_db
from common.models.database import WorkflowDefinition
from projects.guardroute.src.core.graph_parser import GraphParser, GraphValidationError


class WorkflowCreatePayload(BaseModel):
    name: str
    graph_json: Dict[str, Any]
    is_active: Optional[bool] = False


class WorkflowResponse(BaseModel):
    id: str
    name: str
    graph_json: Dict[str, Any]
    is_active: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def _to_iso(dt: Any) -> Optional[str]:
    if not dt:
        return None
    if isinstance(dt, str):
        return dt
    if hasattr(dt, "isoformat"):
        val = dt.isoformat()
        if isinstance(val, str):
            return val
    return str(dt)


@router.get("/workflows", response_model=List[WorkflowResponse])
async def list_workflows(db: AsyncSession = Depends(get_async_db)) -> List[WorkflowResponse]:
    """Retrieve all saved visual workflow graph configurations."""
    stmt = select(WorkflowDefinition).order_by(WorkflowDefinition.created_at.desc())
    res = await db.execute(stmt)
    workflows = res.scalars().all()
    return [
        WorkflowResponse(
            id=w.id,
            name=w.name,
            graph_json=w.graph_json,
            is_active=w.is_active,
            created_at=_to_iso(w.created_at),
            updated_at=_to_iso(w.updated_at)
        )
        for w in workflows
    ]


@router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    payload: WorkflowCreatePayload,
    db: AsyncSession = Depends(get_async_db)
) -> WorkflowResponse:
    """Save/create a visual workflow graph definition after validating topology."""
    parser = GraphParser()
    try:
        parser.validate_graph(payload.graph_json)
    except GraphValidationError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid graph topology: {str(ve)}")

    if payload.is_active:
        # Deactivate all existing workflows if this one is set to active
        await db.execute(update(WorkflowDefinition).values(is_active=False))

    workflow = WorkflowDefinition(
        id=str(uuid.uuid4()),
        name=payload.name,
        graph_json=payload.graph_json,
        is_active=payload.is_active or False
    )
    db.add(workflow)
    await db.commit()
    await db.refresh(workflow)

    return WorkflowResponse(
        id=workflow.id,
        name=workflow.name,
        graph_json=workflow.graph_json,
        is_active=workflow.is_active,
        created_at=_to_iso(workflow.created_at),
        updated_at=_to_iso(workflow.updated_at)
    )


@router.put("/workflows/{workflow_id}/activate", response_model=WorkflowResponse)
async def activate_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_async_db)
) -> WorkflowResponse:
    """Activates a workflow and deactivates all others."""
    stmt = select(WorkflowDefinition).filter(WorkflowDefinition.id == workflow_id)
    res = await db.execute(stmt)
    workflow = res.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow with ID '{workflow_id}' not found.")

    # Deactivate all workflows
    await db.execute(update(WorkflowDefinition).values(is_active=False))
    workflow.is_active = True
    workflow.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(workflow)

    return WorkflowResponse(
        id=workflow.id,
        name=workflow.name,
        graph_json=workflow.graph_json,
        is_active=workflow.is_active,
        created_at=_to_iso(workflow.created_at),
        updated_at=_to_iso(workflow.updated_at)
    )


@router.get("/workflows/active", response_model=Optional[WorkflowResponse])
async def get_active_workflow(db: AsyncSession = Depends(get_async_db)) -> Optional[WorkflowResponse]:
    """Retrieve the currently active visual workflow configuration."""
    stmt = select(WorkflowDefinition).filter(WorkflowDefinition.is_active == True).limit(1)
    res = await db.execute(stmt)
    workflow = res.scalar_one_or_none()

    if not workflow:
        return None

    return WorkflowResponse(
        id=workflow.id,
        name=workflow.name,
        graph_json=workflow.graph_json,
        is_active=workflow.is_active,
        created_at=_to_iso(workflow.created_at),
        updated_at=_to_iso(workflow.updated_at)
    )

