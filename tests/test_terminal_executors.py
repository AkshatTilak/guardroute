"""Unit tests for terminal node executors (ActionNode & FinalMessageNode)."""

from unittest.mock import patch
import pytest

from common.schemas.agent_types import SubAgentResult, SubAgentStatus
from projects.guardroute.src.nodes.action_executor import execute_action_node
from projects.guardroute.src.nodes.final_message_executor import execute_final_message_node


@pytest.mark.asyncio
async def test_execute_action_node_custom_side_effect():
    config = {
        "node_id": "action_db_write",
        "action_type": "db_mutation",
        "payload_template": {"status": "complete", "user_prompt": "{{prompt}}"}
    }
    state = {"prompt": "Run database task"}

    res = await execute_action_node(config, state)
    assert "final_action_status" in res
    status_map = res["final_action_status"]
    assert "action_db_write" in status_map
    assert status_map["action_db_write"]["status"] == "success"
    assert status_map["action_db_write"]["payload"]["user_prompt"] == "Run database task"


@pytest.mark.asyncio
async def test_execute_action_node_ssrf_blocked():
    config = {
        "node_id": "action_ssrf",
        "action_type": "http_post",
        "url": "http://127.0.0.1/latest/meta-data"
    }
    state = {"prompt": "Malicious probe"}

    res = await execute_action_node(config, state)
    status_map = res["final_action_status"]
    assert status_map["action_ssrf"]["status"] == "failed"
    assert "SSRF policy blocked" in status_map["action_ssrf"]["error"]


@pytest.mark.asyncio
async def test_execute_final_message_node():
    config = {
        "model_id": "gemini/gemini-3.5-flash",
        "system_prompt": "Synthesize response"
    }
    subagent_res = SubAgentResult(
        source="CodingAgent",
        status=SubAgentStatus.SUCCESS,
        content_type="text",
        content="Code executed without errors."
    )
    state = {
        "prompt": "Summarize result",
        "subagent_results": [subagent_res]
    }

    mock_llm = {
        "choices": [{"message": {"content": "Here is the synthesized final response."}}]
    }

    with patch("projects.guardroute.src.nodes.final_message_executor.completion_with_fallback", return_value=mock_llm):
        res = await execute_final_message_node(config, state)

    assert "final_response" in res
    assert res["final_response"] == "Here is the synthesized final response."
