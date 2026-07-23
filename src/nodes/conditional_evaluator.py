"""Conditional evaluator for GuardRoute IfElse and Router nodes.

Evaluates branching conditions safely against GraphState without executing arbitrary code.
"""

import ast
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("guardroute.nodes.conditional_evaluator")


def _get_nested_val(data: Dict[str, Any], path: str) -> Any:
    """Safely fetch a value from nested dicts using dot notation."""
    parts = path.split(".")
    curr = data
    for part in parts:
        if isinstance(curr, dict):
            curr = curr.get(part)
        elif hasattr(curr, part):
            curr = getattr(curr, part)
        else:
            return None
    return curr


def _safe_eval_ast(expr: str, state: Dict[str, Any]) -> bool:
    """Sandboxed expression evaluator using Python AST.
    
    Supports:
    - Identifiers referencing state fields (e.g. `complexity`, `token_count`)
    - Literals (strings, numbers, booleans, None)
    - Comparison operators: ==, !=, <, <=, >, >=, in, not in
    - Boolean logic: and, or, not
    """
    try:
        parsed = ast.parse(expr, mode='eval')
    except Exception as e:
        logger.error(f"Failed to parse condition expression '{expr}': {e}")
        return False

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant):  # Numbers, strings, Bools, None
            return node.value
        elif isinstance(node, ast.Name):
            return _get_nested_val(state, node.id)
        elif isinstance(node, ast.Attribute):
            # E.g., state_field.subfield
            def _get_attr_path(n: ast.AST) -> str:
                if isinstance(n, ast.Name):
                    return n.id
                elif isinstance(n, ast.Attribute):
                    return f"{_get_attr_path(n.value)}.{n.attr}"
                return ""
            path = _get_attr_path(node)
            return _get_nested_val(state, path)
        elif isinstance(node, ast.UnaryOp):
            val = _eval_node(node.operand)
            if isinstance(node.op, ast.Not):
                return not val
            elif isinstance(node.op, ast.USub):
                return -val
            elif isinstance(node.op, ast.UAdd):
                return +val
        elif isinstance(node, ast.BoolOp):
            values = [_eval_node(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
        elif isinstance(node, ast.Compare):
            left = _eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = _eval_node(comparator)
                if isinstance(op, ast.Eq) and not (left == right):
                    return False
                elif isinstance(op, ast.NotEq) and not (left != right):
                    return False
                elif isinstance(op, ast.Lt) and not (left < right):
                    return False
                elif isinstance(op, ast.LtE) and not (left <= right):
                    return False
                elif isinstance(op, ast.Gt) and not (left > right):
                    return False
                elif isinstance(op, ast.GtE) and not (left >= right):
                    return False
                elif isinstance(op, ast.In) and not (left in right if right is not None else False):
                    return False
                elif isinstance(op, ast.NotIn) and not (left not in right if right is not None else True):
                    return False
                left = right
            return True

        raise ValueError(f"Unsupported AST node type in condition expression: {type(node).__name__}")

    try:
        result = _eval_node(parsed)
        return bool(result)
    except Exception as e:
        logger.error(f"Error evaluating condition AST '{expr}': {e}")
        return False


def _compare_values(left: Any, operator: str, right: Any) -> bool:
    """Helper to compare two values with standard operators."""
    try:
        if operator == "==":
            return str(left) == str(right) if isinstance(right, str) and not isinstance(left, str) else left == right
        elif operator == "!=":
            return left != right
        elif operator == ">":
            return float(left) > float(right)
        elif operator == "<":
            return float(left) < float(right)
        elif operator == ">=":
            return float(left) >= float(right)
        elif operator == "<=":
            return float(left) <= float(right)
        elif operator == "contains":
            if left is None:
                return False
            return str(right).lower() in str(left).lower()
        elif operator == "matches":
            if left is None:
                return False
            return bool(re.search(str(right), str(left)))
        return False
    except Exception as e:
        logger.warning(f"Comparison failed for operator '{operator}': {e}")
        return False


def evaluate_condition(condition_config: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """Evaluates condition_config against graph state.

    Config structure:
    {
        "type": "complexity_equals" | "output_contains" | "metadata_field" | "regex_match" | "custom_expression",
        "field": "field_name",
        "operator": "==" | "!=" | ">" | "<" | ">=" | "<=" | "contains" | "matches",
        "value": "expected_value",
        "expression": "complexity == 'HIGH' and token_usage.total > 100"
    }
    """
    if not condition_config:
        return True

    cond_type = condition_config.get("type", "complexity_equals")
    target_value = condition_config.get("value")
    operator = condition_config.get("operator", "==")

    if cond_type == "complexity_equals":
        actual = state.get("complexity", "LOW")
        return _compare_values(actual, operator, target_value)

    elif cond_type == "output_contains":
        final_resp = state.get("final_response", "")
        sub_results = state.get("subagent_results", [])
        combined_text = final_resp + " " + " ".join([str(r) for r in sub_results])
        return _compare_values(combined_text, "contains", target_value)

    elif cond_type == "metadata_field":
        field_name = condition_config.get("field", "")
        actual = _get_nested_val(state, field_name)
        return _compare_values(actual, operator, target_value)

    elif cond_type == "regex_match":
        field_name = condition_config.get("field", "final_response")
        actual = _get_nested_val(state, field_name)
        if actual is None:
            actual = state.get("final_response", "")
        return _compare_values(actual, "matches", target_value)

    elif cond_type == "custom_expression":
        expr = condition_config.get("expression", "")
        if not expr:
            return True
        return _safe_eval_ast(expr, state)

    return False
