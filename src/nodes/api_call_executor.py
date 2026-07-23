"""APICall executor node for GuardRoute.

Supports REST API calls (GET, POST, PUT, PATCH, DELETE) with auth options (Bearer/API Key),
SSRF protection, and JSONPath output mapping.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
import httpx

from projects.guardroute.src.nodes.ssrf_protection import validate_url_for_ssrf, SSRFValidationError
from projects.guardroute.src.nodes.webhook_executor import _interpolate_string, _interpolate_dict

logger = logging.getLogger("guardroute.nodes.api_call_executor")


def _extract_jsonpath_field(data: Any, path: str) -> Any:
    """Simple JSONPath / dot-notation field extractor."""
    if not path or not data:
        return data

    clean_path = path.lstrip("$.")
    parts = clean_path.split(".")
    curr = data
    for part in parts:
        if isinstance(curr, dict):
            curr = curr.get(part)
        elif isinstance(curr, list) and part.isdigit():
            idx = int(part)
            curr = curr[idx] if 0 <= idx < len(curr) else None
        else:
            return None
    return curr


async def execute_api_call(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Executes REST API call with auth and response field mapping into graph state.

    Config:
    {
        "url": "https://api.external.com/v1/resource",
        "method": "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
        "auth_type": "none" | "bearer" | "api_key",
        "auth_value": "secret_token_123",
        "headers": {"Content-Type": "application/json"},
        "body_template": "{...}",
        "response_mapping": {
            "mapped_result": "$.data.result",
            "score": "$.metrics.score"
        },
        "timeout": 30
    }

    Returns:
    {
        "status_code": int,
        "mapped_outputs": Dict[str, Any],
        "raw_response": Any,
        "success": bool,
        "error": Optional[str]
    }
    """
    raw_url = config.get("url", "")
    url = _interpolate_string(raw_url, state)

    try:
        validate_url_for_ssrf(url)
    except SSRFValidationError as e:
        logger.error(f"API call blocked by SSRF protection: {e}")
        return {
            "status_code": 403,
            "mapped_outputs": {},
            "raw_response": None,
            "success": False,
            "error": f"SSRF Blocked: {str(e)}"
        }

    method = config.get("method", "GET").upper()
    headers = _interpolate_dict(config.get("headers", {}), state)
    
    # Auth handling
    auth_type = config.get("auth_type", "none").lower()
    auth_val = _interpolate_string(config.get("auth_value", ""), state)
    if auth_type == "bearer" and auth_val:
        headers["Authorization"] = f"Bearer {auth_val}"
    elif auth_type == "api_key" and auth_val:
        header_key = config.get("api_key_header", "X-API-Key")
        headers[header_key] = auth_val

    raw_body = config.get("body_template", "")
    interpolated_body = _interpolate_string(raw_body, state) if isinstance(raw_body, str) else raw_body

    timeout = float(config.get("timeout", 30))

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            logger.info(f"Executing REST API {method} {url}")
            resp = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=interpolated_body if method in ("POST", "PUT", "PATCH") and isinstance(interpolated_body, str) else None,
                json=interpolated_body if method in ("POST", "PUT", "PATCH") and isinstance(interpolated_body, dict) else None
            )

            try:
                resp_json = resp.json()
            except Exception:
                resp_json = {"raw": resp.text}

            success = resp.is_success
            mapped_outputs = {}

            # Perform response field mapping if configured
            mapping_rules = config.get("response_mapping", {})
            if isinstance(mapping_rules, dict):
                for target_key, json_path in mapping_rules.items():
                    extracted = _extract_jsonpath_field(resp_json, json_path)
                    mapped_outputs[target_key] = extracted

            return {
                "status_code": resp.status_code,
                "mapped_outputs": mapped_outputs,
                "raw_response": resp_json,
                "success": success,
                "error": None if success else f"HTTP {resp.status_code}"
            }

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {
                "status_code": 500,
                "mapped_outputs": {},
                "raw_response": None,
                "success": False,
                "error": str(e)
            }
