"""GuardRoute API routes.

Mounted at /api/guardroute/* by the gateway's dynamic route loader.
Provides task complexity classification and Scatter-Gather agent chat orchestration.
Supports token-by-token response streaming via Server-Sent Events (SSE).
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.event_source import EventSourceResponse

from projects.guardroute.src.orchestrator import (
    execute_orchestrator,
    execute_orchestrator_stream,
    run_classification
)

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
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Classification query failed: {str(e)}")
