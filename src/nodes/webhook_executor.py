"""Webhook executor node for GuardRoute.

Sends HTTP POST/PUT/PATCH webhooks asynchronously with template interpolation, SSRF protection,
and exponential backoff retry handling.
"""

import asyncio
import logging
import re
from typing import Any, Dict, Optional
import httpx

from projects.guardroute.src.nodes.ssrf_protection import validate_url_for_ssrf, SSRFValidationError

logger = logging.getLogger("guardroute.nodes.webhook_executor")


def _interpolate_string(template: str, state: Dict[str, Any]) -> str:
    """Replaces {{variable}} placeholders with values from state."""
    if not template:
        return template

    def replace_match(match):
        var_name = match.group(1).strip()
        parts = var_name.split(".")
        curr = state
        for p in parts:
            if isinstance(curr, dict):
                curr = curr.get(p)
            elif hasattr(curr, p):
                curr = getattr(curr, p)
            else:
                curr = None
                break
        return str(curr) if curr is not None else ""

    return re.sub(r"\{\{\s*([a-zA-Z0-9_\.]+)\s*\}\}", replace_match, template)


def _interpolate_dict(data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively interpolates strings inside dictionaries or lists."""
    res = {}
    for k, v in data.items():
        if isinstance(v, str):
            res[k] = _interpolate_string(v, state)
        elif isinstance(v, dict):
            res[k] = _interpolate_dict(v, state)
        elif isinstance(v, list):
            res[k] = [
                _interpolate_string(item, state) if isinstance(item, str) else item
                for item in v
            ]
        else:
            res[k] = v
    return res


async def execute_webhook(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Executes a webhook call according to config and state.

    Config:
    {
        "url": "https://example.com/api/webhook",
        "method": "POST" | "PUT" | "PATCH",
        "headers": {"Content-Type": "application/json"},
        "body_template": "{\"prompt\": \"{{prompt}}\", \"result\": \"{{final_response}}\"}",
        "timeout": 30,
        "retry_count": 2
    }

    Returns:
    {
        "status_code": int,
        "response_body": Any,
        "success": bool,
        "error": Optional[str]
    }
    """
    raw_url = config.get("url", "")
    url = _interpolate_string(raw_url, state)

    try:
        validate_url_for_ssrf(url)
    except SSRFValidationError as e:
        logger.error(f"Webhook blocked by SSRF protection: {e}")
        return {
            "status_code": 403,
            "response_body": None,
            "success": False,
            "error": f"SSRF Blocked: {str(e)}"
        }

    method = config.get("method", "POST").upper()
    headers = _interpolate_dict(config.get("headers", {}), state)
    raw_body = config.get("body_template", "")
    interpolated_body = _interpolate_string(raw_body, state) if isinstance(raw_body, str) else raw_body

    timeout = float(config.get("timeout", 30))
    retries = min(int(config.get("retry_count", 0)), 3)

    attempt = 0
    last_error = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        while attempt <= retries:
            try:
                attempt += 1
                logger.info(f"Sending webhook {method} to {url} (Attempt {attempt}/{retries + 1})")
                
                resp = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=interpolated_body if isinstance(interpolated_body, str) else None,
                    json=interpolated_body if isinstance(interpolated_body, dict) else None
                )

                try:
                    resp_data = resp.json()
                except Exception:
                    resp_data = resp.text

                success = resp.is_success
                return {
                    "status_code": resp.status_code,
                    "response_body": resp_data,
                    "success": success,
                    "error": None if success else f"HTTP {resp.status_code}"
                }

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Webhook attempt {attempt} failed: {e}")
                if attempt <= retries:
                    await asyncio.sleep(2 ** attempt)

    return {
        "status_code": 500,
        "response_body": None,
        "success": False,
        "error": f"Failed after {retries + 1} attempts: {last_error}"
    }
