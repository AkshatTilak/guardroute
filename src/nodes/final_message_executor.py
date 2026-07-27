"""FinalMessageNode terminal executor for GuardRoute.

Aggregates state context, subagent results, and runs LLM completion to generate
the final synthesized response returned to the user.
"""

import logging
import time
from typing import Any, Dict

from common.clients.litellm import completion_with_fallback
from common.schemas.agent_types import SubAgentStatus

logger = logging.getLogger("guardroute.nodes.final_message_executor")


class FinalMessageNodeExecutor:
    """Runtime executor for terminal FinalMessageNode synthesis."""

    def __init__(self, config: Dict[str, Any]):
        """Config format:

        {
            "model_id": "gemini/gemini-3.5-flash",
            "system_prompt": "You are the final synthesis node...",
            "temperature": 0.7,
            "max_tokens": 2000
        }
        """
        self.config = config
        self.model_id = config.get("model_id", "gemini/gemini-3.5-flash")
        self.system_prompt = config.get(
            "system_prompt",
            "You are the Final Synthesis Agent. Consolidate all gathered subagent findings and formulate a clear, factual answer to the prompt."
        )
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 2000))

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregates state context and generates final synthesized response in GraphState."""
        prompt = state.get("prompt", "")
        subagent_results = state.get("subagent_results", [])

        # Gather context snippets
        contexts = []
        for r in subagent_results:
            if hasattr(r, "status"):
                if r.status == SubAgentStatus.SUCCESS:
                    contexts.append(f"--- Context Source: {r.source} ---\n{r.content}")
                else:
                    contexts.append(f"--- Context Source: {r.source} ({r.status.value}) ---\nError: {r.error_message}")
            elif isinstance(r, dict):
                src = r.get("source", "subagent")
                content = r.get("content", "")
                err = r.get("error_message")
                if content:
                    contexts.append(f"--- Context Source: {src} ---\n{content}")
                elif err:
                    contexts.append(f"--- Context Source: {src} ---\nError: {err}")

        compiled_context = "\n\n".join(contexts) if contexts else "No prior subagent outputs."

        synthesis_prompt = (
            f"{self.system_prompt}\n\n"
            f"User Prompt: {prompt}\n\n"
            f"Gathered Context:\n{compiled_context}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": synthesis_prompt},
        ]

        try:
            res = await completion_with_fallback(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            final_ans = res.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info("FinalMessageNode synthesis completion generated successfully.")
            return {"final_response": final_ans}
        except Exception as e:
            logger.error(f"FinalMessageNode synthesis failed: {e}")
            fallback_ans = f"Synthesized Response: Processed user prompt '{prompt}' with context from {len(subagent_results)} subagents."
            return {"final_response": fallback_ans}


async def execute_final_message_node(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Helper entry point for FinalMessageNode executor."""
    executor = FinalMessageNodeExecutor(config)
    return await executor.execute(state)
