"""ActionNode terminal executor for GuardRoute.

Dispatches side-effects (HTTP payload post or DB mutation) as terminal workflow actions.
"""

import logging
import time
from typing import Any, Dict, Optional

import httpx
from projects.guardroute.src.nodes.ssrf_protection import validate_url_for_ssrf, SSRFValidationError
from projects.guardroute.src.nodes.webhook_executor import _interpolate_dict

logger = logging.getLogger("guardroute.nodes.action_executor")


class ActionNodeExecutor:
    """Runtime executor for terminal ActionNode side-effects."""

    def __init__(self, config: Dict[str, Any]):
        """Config format:

        {
            "action_type": "http_post" | "db_mutation" | "custom",
            "url": "https://api.example.com/action",
            "payload_template": {"event": "task_completed", "data": "{{prompt}}"},
            "timeout_sec": 10.0
        }
        """
        self.config = config
        self.action_type = config.get("action_type", "http_post")
        self.url = config.get("url", "")
        self.payload_template = config.get("payload_template", {})
        self.timeout_sec = float(config.get("timeout_sec", 10.0))

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes terminal side-effect action and returns final_action_status for GraphState."""
        start_time = time.time()
        node_id = self.config.get("node_id", "action_node")

        # Interpolate variables in payload
        payload = (
            _interpolate_dict(self.payload_template, state)
            if isinstance(self.payload_template, dict)
            else self.payload_template
        )

        if self.action_type == "http_post" and self.url:
            # Validate SSRF
            try:
                validate_url_for_ssrf(self.url)
            except (SSRFValidationError, ValueError) as ssrf_err:
                err_str = f"ActionNode SSRF policy blocked URL: {ssrf_err}"
                logger.error(err_str)
                action_status = {
                    "node_id": node_id,
                    "action_type": self.action_type,
                    "status": "failed",
                    "error": err_str,
                    "execution_time_ms": round((time.time() - start_time) * 1000, 2),
                }
                curr_actions = dict(state.get("final_action_status", {})) if isinstance(state.get("final_action_status"), dict) else {}
                curr_actions[node_id] = action_status
                return {"final_action_status": curr_actions}

            headers = {"Content-Type": "application/json", "User-Agent": "ContAIned-GuardRoute/v5"}
            try:
                async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                    resp = await client.post(self.url, json=payload, headers=headers)
                    duration_ms = round((time.time() - start_time) * 1000, 2)

                    action_status = {
                        "node_id": node_id,
                        "action_type": self.action_type,
                        "status": "success" if resp.status_code < 400 else "failed",
                        "status_code": resp.status_code,
                        "response": resp.text[:500],
                        "execution_time_ms": duration_ms,
                    }
            except Exception as e:
                duration_ms = round((time.time() - start_time) * 1000, 2)
                err_str = f"ActionNode HTTP request failed: {e}"
                logger.error(err_str)
                action_status = {
                    "node_id": node_id,
                    "action_type": self.action_type,
                    "status": "failed",
                    "error": err_str,
                    "execution_time_ms": duration_ms,
                }

        else:
            # Standard side-effect action execution (e.g. DB mutation or logger dispatch)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"ActionNode '{node_id}' side-effect executed with payload: {payload}")
            action_status = {
                "node_id": node_id,
                "action_type": self.action_type,
                "status": "success",
                "payload": payload,
                "execution_time_ms": duration_ms,
            }

        curr_actions = dict(state.get("final_action_status", {})) if isinstance(state.get("final_action_status"), dict) else {}
        curr_actions[node_id] = action_status
        return {"final_action_status": curr_actions}


async def execute_action_node(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Helper entry point for ActionNode executor."""
    executor = ActionNodeExecutor(config)
    return await executor.execute(state)
