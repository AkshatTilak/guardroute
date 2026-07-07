# GuardRoute Developer Agent Guidelines

This document details coding standards and requirements specific to the **GuardRoute** submodule within the monorepo architecture. For general platform standards, refer to the [Root Monorepo Guidelines](../../agent.md).

---

## 1. Submodule Boundary & Interfaces

GuardRoute acts as the gateway's cognitive decision router. It integrates with the platform via:
1. **API Router (`api.py`)**: Mounts FastAPI endpoints (e.g. `/api/guardroute/chat`, `/api/guardroute/classify`).
2. **Setup Hook (`setup.py`)**: Initializes the shared `InferenceClient` (to call the local routing classifier) and creates connections to active MCP subagent servers.

---

## 2. Model Routing & LiteLLM Integration

GuardRoute coordinates task delegation and execution:
- **Task Classification**: Calls `InferenceClient.classify()` to send prompts to the local `Arch-Router-1.5B` GGUF model running on the inference server.
- **Provider Fallbacks**: Completion calls must use `common.clients.litellm.completion_with_fallback()`.
- **Capability Isolation**: When falling back from Google Gemini to OpenRouter free-tier models (e.g. Llama-3-8B), prompt payloads must be automatically updated or truncated to respect the fallback model's token limits and feature sets.

---

## 3. Scatter-Gather Orchestration

- **Subagent Results**: Every parallel agent node in the LangGraph execution flow must output a structured `SubAgentResult` Pydantic model (defined in `common.schemas.agent_types`).
- **Partial Failure Handling**: If a subagent node times out or throws an error, the gather node must catch the exception, set the status to `SubAgentStatus.TIMEOUT` or `SubAgentStatus.ERROR`, and proceed with the remaining successful subagent outputs rather than blocking the entire graph.
- **Observability**: Ensure Otel tracing spans are auto-registered for every LangGraph node execution.
