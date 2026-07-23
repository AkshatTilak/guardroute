"""Unit tests for GuardRoute router_executor."""

import pytest
from projects.guardroute.src.nodes.router_executor import evaluate_routes


def test_evaluate_routes_matching():
    state = {"complexity": "HIGH", "session_id": "s123"}
    routes = [
        {"label": "fast_path", "condition": {"type": "complexity_equals", "value": "LOW"}},
        {"label": "deep_path", "condition": {"type": "complexity_equals", "value": "HIGH"}}
    ]
    res = evaluate_routes(routes, "default_path", state)
    assert res == "deep_path"


def test_evaluate_routes_default_fallback():
    state = {"complexity": "MEDIUM"}
    routes = [
        {"label": "fast_path", "condition": {"type": "complexity_equals", "value": "LOW"}},
        {"label": "deep_path", "condition": {"type": "complexity_equals", "value": "HIGH"}}
    ]
    res = evaluate_routes(routes, "default_path", state)
    assert res == "default_path"
