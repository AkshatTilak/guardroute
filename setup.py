"""GuardRoute project setup hook.

Called by the gateway's lifespan factory during startup/shutdown.
Initializes the inference client (for classifier) and LangGraph orchestrator.
"""

from fastapi import FastAPI

from common.clients.inference import InferenceClient
from common.observability.logger import get_logger

logger = get_logger("guardroute")


async def init_app_state(app: FastAPI, settings) -> None:
    """Initialize GuardRoute state on gateway startup.

    Sets up:
    - Inference client (for task classifier via Arch-Router)
    - LangGraph orchestrator (future: scatter-gather graph)
    """
    app.state.guardroute_inference = InferenceClient(
        base_url=settings.INFERENCE_SERVER_URL,
    )
    logger.info("GuardRoute initialized — inference client connected to %s", settings.INFERENCE_SERVER_URL)


async def shutdown_app_state(app: FastAPI, settings) -> None:
    """Clean up GuardRoute state on gateway shutdown."""
    if hasattr(app.state, "guardroute_inference"):
        await app.state.guardroute_inference.close()
    logger.info("GuardRoute shut down")
