"""Multi-branch router node executor for GuardRoute.

Evaluates an array of route rules against state and selects the first matching branch target.
"""

import logging
from typing import Any, Dict, List, Optional
from projects.guardroute.src.nodes.conditional_evaluator import evaluate_condition

logger = logging.getLogger("guardroute.nodes.router_executor")


def evaluate_routes(routes: List[Dict[str, Any]], default_route: str, state: Dict[str, Any]) -> str:
    """Evaluates routes sequentially and returns target label of first matching route.

    Routes structure:
    [
        {"label": "high_priority", "condition": {"type": "complexity_equals", "value": "HIGH"}},
        {"label": "rag_route", "condition": {"type": "output_contains", "value": "retrieval"}}
    ]
    """
    if not routes:
        return default_route

    for route in routes:
        label = route.get("label", "")
        cond = route.get("condition", {})
        if evaluate_condition(cond, state):
            logger.info(f"RouterNode matched branch '{label}'")
            return label

    logger.info(f"RouterNode no branch matched; falling back to default route '{default_route}'")
    return default_route
