"""Unit tests for GuardRoute conditional_evaluator."""

import pytest
from projects.guardroute.src.nodes.conditional_evaluator import (
    evaluate_condition,
    _safe_eval_ast,
)


def test_complexity_equals():
    state = {"complexity": "HIGH"}
    assert evaluate_condition({"type": "complexity_equals", "value": "HIGH", "operator": "=="}, state) is True
    assert evaluate_condition({"type": "complexity_equals", "value": "LOW", "operator": "=="}, state) is False
    assert evaluate_condition({"type": "complexity_equals", "value": "LOW", "operator": "!="}, state) is True


def test_output_contains():
    state = {"final_response": "The system initialized successfully.", "subagent_results": ["Data fetched"]}
    assert evaluate_condition({"type": "output_contains", "value": "initialized"}, state) is True
    assert evaluate_condition({"type": "output_contains", "value": "error"}, state) is False


def test_metadata_field():
    state = {"token_usage": {"total": 500}, "status": "active"}
    assert evaluate_condition({"type": "metadata_field", "field": "token_usage.total", "operator": ">", "value": 100}, state) is True
    assert evaluate_condition({"type": "metadata_field", "field": "token_usage.total", "operator": "<", "value": 100}, state) is False
    assert evaluate_condition({"type": "metadata_field", "field": "status", "operator": "==", "value": "active"}, state) is True


def test_regex_match():
    state = {"final_response": "Error code: ERR_404_NOT_FOUND"}
    assert evaluate_condition({"type": "regex_match", "field": "final_response", "value": r"ERR_\d+"}, state) is True
    assert evaluate_condition({"type": "regex_match", "field": "final_response", "value": r"SUCCESS"}, state) is False


def test_custom_expression():
    state = {"complexity": "HIGH", "token_count": 250, "is_valid": True}
    assert evaluate_condition({
        "type": "custom_expression",
        "expression": "complexity == 'HIGH' and token_count > 100"
    }, state) is True

    assert evaluate_condition({
        "type": "custom_expression",
        "expression": "complexity == 'LOW' or not is_valid"
    }, state) is False


def test_ast_sandbox_safety():
    state = {"val": 10}
    # Attempt code injection - should fail safely
    result = _safe_eval_ast("__import__('os').system('dir')", state)
    assert result is False
