"""Transform node executor for GuardRoute.

Handles data transformations:
- Jinja2 sandboxed template rendering
- Field extraction via dot notation / JSONPath
- Merging multiple state dictionaries
- Formatting and restructuring JSON data
"""

import json
import logging
from typing import Any, Dict, Optional
import jinja2
from jinja2.sandbox import SandboxedEnvironment

from projects.guardroute.src.nodes.conditional_evaluator import _get_nested_val

logger = logging.getLogger("guardroute.nodes.transform_executor")

# Safe Jinja2 sandboxed environment
_sandboxed_env = SandboxedEnvironment(
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True
)


def execute_transform(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Executes data transformation on state according to config.

    Config:
    {
        "mode": "template" | "extract_field" | "merge" | "format_json",
        "template": "Hello {{prompt}}, your complexity is {{complexity}}.",
        "field_path": "token_usage.total",
        "merge_fields": ["subagent_results", "webhook_results"],
        "field_mappings": {
            "output_text": "final_response",
            "session": "session_id"
        }
    }

    Returns:
    {
        "output": Any,
        "output_type": str,
        "success": bool,
        "error": Optional[str]
    }
    """
    mode = config.get("mode", "template").lower()

    try:
        if mode == "template":
            tmpl_str = config.get("template", "")
            template = _sandboxed_env.from_string(tmpl_str)
            rendered = template.render(**state)
            return {
                "output": rendered,
                "output_type": "string",
                "success": True,
                "error": None
            }

        elif mode == "extract_field":
            path = config.get("field_path", "")
            extracted = _get_nested_val(state, path)
            return {
                "output": extracted,
                "output_type": type(extracted).__name__,
                "success": True,
                "error": None
            }

        elif mode == "merge":
            fields = config.get("merge_fields", [])
            merged = {}
            for field in fields:
                val = _get_nested_val(state, field)
                if isinstance(val, dict):
                    merged.update(val)
                elif val is not None:
                    merged[field] = val
            return {
                "output": merged,
                "output_type": "dict",
                "success": True,
                "error": None
            }

        elif mode == "format_json":
            mappings = config.get("field_mappings", {})
            result = {}
            for new_key, src_path in mappings.items():
                result[new_key] = _get_nested_val(state, src_path)
            return {
                "output": result,
                "output_type": "dict",
                "success": True,
                "error": None
            }

        else:
            return {
                "output": None,
                "output_type": "unknown",
                "success": False,
                "error": f"Unsupported transform mode '{mode}'"
            }

    except Exception as e:
        logger.error(f"Transform execution failed (mode={mode}): {e}")
        return {
            "output": None,
            "output_type": "error",
            "success": False,
            "error": str(e)
        }
