"""Unit tests for GuardRoute transform_executor."""

import pytest
from projects.guardroute.src.nodes.transform_executor import execute_transform


def test_transform_template():
    state = {"prompt": "AI Workflow", "complexity": "HIGH"}
    config = {
        "mode": "template",
        "template": "Task {{prompt}} running with {{complexity}} complexity."
    }
    res = execute_transform(config, state)
    assert res["success"] is True
    assert res["output"] == "Task AI Workflow running with HIGH complexity."


def test_transform_extract_field():
    state = {"meta": {"token_count": 142}}
    config = {
        "mode": "extract_field",
        "field_path": "meta.token_count"
    }
    res = execute_transform(config, state)
    assert res["success"] is True
    assert res["output"] == 142


def test_transform_format_json():
    state = {"session_id": "abc-123", "final_response": "Done."}
    config = {
        "mode": "format_json",
        "field_mappings": {
            "session": "session_id",
            "reply": "final_response"
        }
    }
    res = execute_transform(config, state)
    assert res["success"] is True
    assert res["output"] == {"session": "abc-123", "reply": "Done."}
