"""MultiAgentNode runtime executor for GuardRoute.

Executes a pre-configured sub-agent within a workflow graph,
passing context and appending structured SubAgentResult objects to GraphState.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from common.clients.litellm import completion_with_fallback
from common.schemas.agent_types import SubAgentResult, SubAgentStatus

logger = logging.getLogger("guardroute.nodes.multi_agent_executor")


class MultiAgentExecutor:
    """Runtime executor for MultiAgent nodes in GuardRoute workflow graph."""

    def __init__(self, config: Dict[str, Any]):
        """Config options:

        - agent_id: str (ID or slug of agent definition)
        - agent_name: str (Optional fallback display name)
        - system_prompt: str (System prompt override)
        - model_id: str (Model override)
        - temperature: float (Temperature override)
        - max_tokens: int (Max tokens override)
        - timeout_sec: float (Timeout in seconds, default 30.0)
        """
        self.config = config
        self.agent_id = config.get("agent_id", "sub_agent")
        self.agent_name = config.get("agent_name", self.agent_id)
        self.system_prompt = config.get("system_prompt")
        self.model_id = config.get("model_id", "gemini/gemini-3.5-flash")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1000)
        self.timeout_sec = float(config.get("timeout_sec", 30.0))

    async def _resolve_agent_definition(self) -> Dict[str, Any]:
        """Tries to query AgentDefinition from database, or falls back to node config defaults."""
        try:
            from common.clients.postgres import get_sessionmaker
            from common.models.database import AgentDefinition
            from sqlalchemy import select

            SessionLocal = get_sessionmaker()
            async with SessionLocal() as session:
                stmt = select(AgentDefinition).where(
                    (AgentDefinition.id == self.agent_id) | (AgentDefinition.endpoint_slug == self.agent_id)
                )
                res = await session.execute(stmt)
                agent = res.scalar_one_or_none()
                if agent and agent.is_active:
                    return {
                        "agent_id": agent.id,
                        "name": agent.name,
                        "system_prompt": self.system_prompt or agent.system_prompt,
                        "model_id": self.model_id if self.config.get("model_id") else agent.model_id,
                        "temperature": agent.temperature or self.temperature,
                        "max_tokens": agent.max_tokens or self.max_tokens,
                    }
        except Exception as e:
            logger.debug(f"Could not resolve DB AgentDefinition for '{self.agent_id}': {e}")

        return {
            "agent_id": self.agent_id,
            "name": self.agent_name,
            "system_prompt": self.system_prompt or "You are a specialized sub-agent assistant.",
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Executes sub-agent LLM completion and updates subagent_results in GraphState.

        Returns dict to merge into GraphState:
        {"subagent_results": [SubAgentResult]}
        """
        start_time = time.time()
        agent_def = await self._resolve_agent_definition()

        prompt = state.get("prompt", "")
        # Build prompt context including previous subagent_results if present
        prior_results = state.get("subagent_results", [])
        context_blocks = []
        for res in prior_results:
            if hasattr(res, "content") and res.content:
                context_blocks.append(f"[{res.source} Output]:\n{res.content}")
            elif isinstance(res, dict) and res.get("content"):
                context_blocks.append(f"[{res.get('source')} Output]:\n{res.get('content')}")

        full_user_prompt = prompt
        if context_blocks:
            full_user_prompt = (
                f"User Prompt:\n{prompt}\n\n"
                "Prior SubAgent Context:\n" + "\n\n".join(context_blocks)
            )

        messages = [
            {"role": "system", "content": agent_def["system_prompt"]},
            {"role": "user", "content": full_user_prompt},
        ]

        try:
            completion_coro = completion_with_fallback(
                model=agent_def["model_id"],
                messages=messages,
                temperature=agent_def["temperature"],
                max_tokens=agent_def["max_tokens"],
            )
            response = await asyncio.wait_for(completion_coro, timeout=self.timeout_sec)
            latency_ms = (time.time() - start_time) * 1000.0

            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = response.get("usage", {})
            token_count = usage.get("total_tokens", len(content) // 4)

            subagent_res = SubAgentResult(
                source=agent_def["name"],
                status=SubAgentStatus.SUCCESS,
                content_type="text",
                content=content,
                token_count=token_count,
                latency_ms=round(latency_ms, 2),
            )
            logger.info(f"MultiAgentNode '{agent_def['name']}' completed successfully in {latency_ms:.1f}ms.")
            return {"subagent_results": [subagent_res]}

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000.0
            err_msg = f"MultiAgentNode '{agent_def['name']}' timed out after {self.timeout_sec}s."
            logger.warning(err_msg)
            subagent_res = SubAgentResult(
                source=agent_def["name"],
                status=SubAgentStatus.TIMEOUT,
                content_type="error",
                content="",
                error_message=err_msg,
                latency_ms=round(latency_ms, 2),
            )
            return {"subagent_results": [subagent_res]}

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            err_msg = f"MultiAgentNode '{agent_def['name']}' failed: {str(e)}"
            logger.error(err_msg)
            subagent_res = SubAgentResult(
                source=agent_def["name"],
                status=SubAgentStatus.ERROR,
                content_type="error",
                content="",
                error_message=err_msg,
                latency_ms=round(latency_ms, 2),
            )
            return {"subagent_results": [subagent_res]}


async def execute_multi_agent(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Helper entry point for MultiAgent executor."""
    executor = MultiAgentExecutor(config)
    return await executor.execute(state)
