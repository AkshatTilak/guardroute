"""Unit tests for GuardRoute eval_executor."""

import pytest
from projects.guardroute.src.nodes.eval_executor import execute_eval_node


@pytest.mark.asyncio
async def test_execute_eval_node_pass():
    state = {
        "prompt": "Explain Quantum Computing",
        "final_response": "Quantum Computing utilizes qubits and superposition to process complex calculations exponentially faster than classical bits."
    }
    config = {
        "suite_name": "QA Benchmark",
        "framework": "heuristic",
        "threshold": 0.5
    }
    res = await execute_eval_node(config, state)
    assert res["passed"] is True
    assert res["score"] >= 0.5


@pytest.mark.asyncio
async def test_execute_eval_node_fail_empty():
    state = {"prompt": "Test prompt", "final_response": ""}
    config = {
        "suite_name": "Strict Suite",
        "threshold": 0.8
    }
    res = await execute_eval_node(config, state)
    assert res["passed"] is False
    assert res["score"] == 0.0
