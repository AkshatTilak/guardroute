"""Unit tests for MultiAgentNode runtime executor."""

from unittest.mock import patch
import pytest

from common.schemas.agent_types import SubAgentStatus
from projects.guardroute.src.nodes.multi_agent_executor import execute_multi_agent


@pytest.mark.asyncio
async def test_execute_multi_agent_success():
    config = {
        "agent_id": "coding_agent",
        "agent_name": "Coding Agent",
        "system_prompt": "You are a python coding sub-agent.",
        "model_id": "gemini/gemma-3-27b-it",
    }
    state = {
        "prompt": "Write a python function to add two numbers.",
        "subagent_results": []
    }

    mock_response = {
        "choices": [{"message": {"content": "def add(a, b): return a + b"}}],
        "usage": {"total_tokens": 42}
    }

    with patch("projects.guardroute.src.nodes.multi_agent_executor.completion_with_fallback", return_value=mock_response):
        res = await execute_multi_agent(config, state)

    assert "subagent_results" in res
    results = res["subagent_results"]
    assert len(results) == 1
    assert results[0].source == "Coding Agent"
    assert results[0].status == SubAgentStatus.SUCCESS
    assert "def add(a, b)" in results[0].content


@pytest.mark.asyncio
async def test_execute_multi_agent_error_handling():
    config = {
        "agent_id": "failing_agent",
        "agent_name": "Failing Agent",
    }
    state = {"prompt": "Test prompt"}

    with patch("projects.guardroute.src.nodes.multi_agent_executor.completion_with_fallback", side_effect=RuntimeError("LLM Provider connection failed")):
        res = await execute_multi_agent(config, state)

    assert "subagent_results" in res
    results = res["subagent_results"]
    assert len(results) == 1
    assert results[0].status == SubAgentStatus.ERROR
    assert "LLM Provider connection failed" in results[0].error_message
