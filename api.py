"""GuardRoute API routes.

Mounted at /api/guardroute/* by the gateway's dynamic route loader.
Provides task complexity classification and Scatter-Gather agent chat orchestration.
"""

import logging
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from projects.guardroute.src.orchestrator import execute_orchestrator, run_classification

router = APIRouter(tags=["guardroute"])
logger = logging.getLogger("guardroute.api")


class ChatRequest(BaseModel):
    """Payload containing user prompt."""

    prompt: str


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
async def chat(request: Request, req: ChatRequest) -> dict:
    """Main orchestration endpoint — classify complexity, scatter tasks, gather outputs, synthesize response."""
    logger.info("Received chat orchestration request: '%s'", req.prompt[:40])
    try:
        trace_result = await execute_orchestrator(req.prompt)
        return {
            "status": "success",
            "session_id": trace_result.get("session_id"),
            "complexity": trace_result.get("complexity"),
            "subagents_ran": trace_result.get("subagents_ran"),
            "response": trace_result.get("final_response"),
            "latency_sec": trace_result.get("duration_sec"),
        }
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
