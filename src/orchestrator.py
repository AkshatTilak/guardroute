"""GuardRoute Multi-Agent Orchestrator.

Implements a LangGraph StateGraph performing Scatter-Gather routing
across parallel subagents, managing session memory, fallback completions,
guardrails, database logging, and Kafka trace publishing.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated, AsyncGenerator
import operator

from common.config.settings import settings
from common.clients.inference import InferenceClient
from common.clients.litellm import completion_with_fallback
from common.schemas.agent_types import (
    SubAgentResult,
    SubAgentStatus,
    TaskComplexity,
)
from projects.guardroute.src.agents.coding import run_code_sandbox
from projects.guardroute.src.agents.search import run_web_search
from projects.guardroute.src.agents.guardrails import (
    check_prompt_injection,
    scrub_pii,
    check_toxicity,
    clean_html_tags,
    check_hallucination_grounding,
)
from projects.guardroute.src.agents.classifier import classify_prompt
from common.models.registry import get_active_model


logger = logging.getLogger("guardroute.orchestrator")


class GraphState(TypedDict):
    """The state of the GuardRoute orchestrator graph."""

    prompt: str
    session_id: str
    complexity: str
    required_agents: List[str]
    subagent_results: Annotated[List[SubAgentResult], operator.add]
    final_response: str
    token_usage: Dict[str, int]  # Keys: input, output


# --- Helper to retrieve Redis and DB connection ---

def get_redis_connection() -> Optional[Any]:
    try:
        from common.clients.redis import get_redis_client
        return get_redis_client()
    except Exception as e:
        logger.warning("Redis client could not be loaded: %s. Caching disabled.", e)
        return None


# --- Kafka Trace and Postgres Persistence ---

def publish_trace_to_kafka(trace_data: Dict[str, Any]) -> None:
    """Publish the execution trace data asynchronously to a Kafka topic."""
    try:
        from confluent_kafka import Producer
        conf = {"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS}
        producer = Producer(conf)
        
        producer.produce(
            "guardroute-traces",
            key=str(trace_data.get("session_id", time.time())),
            value=json.dumps(trace_data)
        )
        producer.flush()
        logger.info("Trace event successfully published to Kafka: guardroute-traces")
    except Exception as e:
        logger.warning("Kafka broker unavailable: %s. Trace logged locally.", e)


async def persist_session_and_usage(trace_data: Dict[str, Any], input_tokens: int, output_tokens: int) -> None:
    """Saves session execution logs and usage statistics to PostgreSQL."""
    try:
        from common.clients.postgres import get_sessionmaker
        from projects.guardroute.src.database.models import GuardRouteSession, GuardRouteUsage
        from sqlalchemy import select
        
        session_factory = get_sessionmaker()
        async with session_factory() as session:
            # 1. Persist completed session trace
            db_session = GuardRouteSession(
                session_id=trace_data["session_id"],
                prompt=trace_data["prompt"],
                complexity=trace_data["complexity"],
                subagents_ran=trace_data["subagents_ran"],
                final_response=trace_data["final_response"],
                duration_sec=trace_data["duration_sec"]
            )
            session.add(db_session)
            
            # 2. Persist aggregated token usage
            stmt = select(GuardRouteUsage).where(GuardRouteUsage.session_id == trace_data["session_id"])
            res = await session.execute(stmt)
            db_usage = res.scalars().first()
            if db_usage:
                db_usage.input_tokens += input_tokens
                db_usage.output_tokens += output_tokens
                db_usage.total_tokens += (input_tokens + output_tokens)
            else:
                db_usage = GuardRouteUsage(
                    session_id=trace_data["session_id"],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=(input_tokens + output_tokens)
                )
                session.add(db_usage)
                
            await session.commit()
            logger.info("Session trace audit logged to PostgreSQL for session: %s", trace_data["session_id"])
    except Exception as e:
        logger.error("Failed to save audit logs to PostgreSQL: %s", e)


# --- Subagent execution wrapper with node-level timeout ---

async def run_subagent_node(agent_name: str, node_coro: Any) -> SubAgentResult:
    """Wraps subagent execution coroutine with node-level timeout."""
    timeout_limit = getattr(settings, "SUBAGENT_TIMEOUT_SECONDS", 30.0)
    try:
        return await asyncio.wait_for(node_coro, timeout=timeout_limit)
    except asyncio.TimeoutError:
        logger.warning("Subagent '%s' execution timed out after %ds limit.", agent_name, timeout_limit)
        return SubAgentResult(
            source=agent_name,
            status=SubAgentStatus.TIMEOUT,
            error_message=f"Subagent '{agent_name}' timed out after {timeout_limit} seconds."
        )
    except Exception as e:
        logger.error("Subagent '%s' execution failed: %s", agent_name, e)
        return SubAgentResult(
            source=agent_name,
            status=SubAgentStatus.ERROR,
            error_message=str(e)
        )


# --- Worker subagents wrappers for LangGraph nodes ---

async def retrieval_agent(prompt: str) -> SubAgentResult:
    """Retrieves context from SyntraFlow Hybrid Retrieval Engine."""
    try:
        from projects.syntraflow.src.retrieval import RetrievalEngine
        from common.clients.qdrant import VectorClient
        
        vector = VectorClient()
        engine = RetrievalEngine(vector)
        
        # Resolve active embedding model dimension and embed prompt
        dim = 1024
        try:
            model_spec = await get_active_model("embedding")
            dim = model_spec.vector_dim or 1024
        except Exception:
            pass

        try:
            inference = InferenceClient(base_url=settings.INFERENCE_SERVER_URL)
            embeds = await inference.embed(texts=[prompt])
            query_vector = embeds[0]
            await inference.close()
        except Exception:
            logger.warning("Inference Server embed failed. Falling back to zero query vector.")
            query_vector = [0.0] * dim

        results = await engine.search_hybrid(prompt, query_vector, limit=3)
        content_str = json.dumps(results, indent=2)
        
        return SubAgentResult(
            source="retrieval",
            status=SubAgentStatus.SUCCESS,
            content_type="json",
            content=content_str,
            token_count=len(content_str) // 4
        )
    except Exception as e:
        logger.error("Retrieval subagent logic failed: %s", e)
        return SubAgentResult(
            source="retrieval",
            status=SubAgentStatus.ERROR,
            error_message=str(e)
        )


# --- LangGraph Nodes ---

async def classify_node(state: GraphState) -> Dict[str, Any]:
    inference = InferenceClient(base_url=settings.INFERENCE_SERVER_URL)
    try:
        result = await classify_prompt(state["prompt"], inference)
    finally:
        await inference.close()
    return {
        "complexity": result["complexity"],
        "required_agents": result["required_agents"]
    }


async def retrieval_node(state: GraphState) -> Dict[str, Any]:
    res = await run_subagent_node("retrieval", retrieval_agent(state["prompt"]))
    return {"subagent_results": [res]}


async def coding_node(state: GraphState) -> Dict[str, Any]:
    res = await run_subagent_node("coding", run_code_sandbox(state["prompt"]))
    return {"subagent_results": [res]}


async def web_search_node(state: GraphState) -> Dict[str, Any]:
    res = await run_subagent_node("web_search", run_web_search(state["prompt"]))
    return {"subagent_results": [res]}


async def gather_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Gather Node: consolidated %d subagent results.", len(state["subagent_results"]))
    return {}


# --- LangGraph Construction ---

from langgraph.graph import StateGraph, START, END

def create_orchestrator_graph() -> StateGraph:
    workflow = StateGraph(GraphState)
    
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("gather", gather_node)
    
    workflow.set_entry_point("classify")
    
    def scatter_router(state: GraphState):
        agents = state["required_agents"]
        if not agents:
            return ["gather"]  # Directly go to gather if simple
        return agents
        
    workflow.add_conditional_edges(
        "classify",
        scatter_router,
        {
            "retrieval": "retrieval",
            "coding": "coding",
            "web_search": "web_search",
            "gather": "gather"
        }
    )
    
    workflow.add_edge("retrieval", "gather")
    workflow.add_edge("coding", "gather")
    workflow.add_edge("web_search", "gather")
    workflow.add_edge("gather", END)
    
    return workflow.compile()


# --- Stream Execution Pipeline ---

async def execute_orchestrator_stream(prompt: str, session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
    """Runs GuardRoute orchestrator and streams responses, status, and metadata events."""
    start_time = time.time()
    if not session_id:
        session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # 1. Pre-flight Guardrails and sanitization
    clean_prompt = clean_html_tags(prompt)

    # Enforce max prompt length based on completion model context window
    context_window = 1048576  # Default fallback
    try:
        model_spec = await get_active_model("completion")
        context_window = model_spec.context_window or 1048576
    except Exception:
        pass

    # Estimate prompt tokens (approx 1 token = 4 characters)
    estimated_tokens = len(clean_prompt) // 4
    # Reserve 2000 tokens for system prompt and responses
    max_allowed = max(1000, context_window - 2000)
    if estimated_tokens > max_allowed:
        logger.warning("Pre-flight rejection: prompt length %d exceeds max allowed %d", estimated_tokens, max_allowed)
        yield {
            "event": "error",
            "data": json.dumps({"detail": f"Prompt rejected: exceeds maximum length of {max_allowed} tokens."})
        }
        return

    is_safe, err_msg = check_prompt_injection(clean_prompt)
    if not is_safe:
        logger.warning("Pre-flight rejection: prompt injection detected.")
        yield {
            "event": "error",
            "data": json.dumps({"detail": err_msg})
        }
        return

    # 2. Redis History Caching
    redis = get_redis_connection()
    history = []
    if redis:
        try:
            cached = await redis.get(f"guardroute:session:{session_id}")
            if cached:
                history = json.loads(cached)
        except Exception as e:
            logger.warning("Failed to fetch session history from Redis: %s", e)

    # Maintain history window limit (20 turns = 10 user/assistant turns)
    history = history[-20:]

    # 3. Graph Classification & Parallel Gather Run
    graph = create_orchestrator_graph()
    initial_state = {
        "prompt": clean_prompt,
        "session_id": session_id,
        "complexity": "",
        "required_agents": [],
        "subagent_results": [],
        "final_response": "",
        "token_usage": {"input": 0, "output": 0}
    }

    class_start = time.time()
    final_state = await graph.ainvoke(initial_state)
    class_latency = (time.time() - class_start) * 1000

    # Resolve active completion model
    model_name = "gemini/gemini-3.5-flash"
    try:
        model_spec = await get_active_model("completion")
        model_name = model_spec.model_id
    except Exception:
        pass

    # Yield metadata event to client
    yield {
        "event": "metadata",
        "data": json.dumps({
            "session_id": session_id,
            "complexity": final_state["complexity"],
            "subagents_ran": final_state["required_agents"],
            "model_used": model_name,
            "classification_latency_ms": class_latency
        })
    }

    # 4. Context Consolidation & Synthesis Prompt
    contexts = []
    for r in final_state["subagent_results"]:
        if r.status == SubAgentStatus.SUCCESS:
            contexts.append(f"--- Context Source: {r.source} ---\n{r.content}")
        else:
            contexts.append(f"--- Context Source: {r.source} ({r.status.value}) ---\nError: {r.error_message}")
            
    compiled_context = "\n\n".join(contexts)

    # 5. Build conversation turns history payload
    synthesis_prompt = (
        "You are the Synthesis Agent of the GuardRoute orchestrator.\n"
        "Formulate a complete, coordinated, and factual response to the user query "
        "using the gathered context below. If any subagents failed, proceed with the successful contexts.\n\n"
        f"User Prompt: {clean_prompt}\n\n"
        f"Gathered Context:\n{compiled_context}"
    )

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(history)
    messages.append({"role": "user", "content": synthesis_prompt})

    # 6. Fallback Completion with SSE streaming
    ans_accum = []
    prompt_tokens = 0
    completion_tokens = 0
    try:
        response = await completion_with_fallback(
            model=model_name,
            messages=messages,
            stream=True
        )
        
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if token:
                ans_accum.append(token)
                yield {
                    "event": "token",
                    "data": token
                }
                
            # Extract token counts if usage object is present in chunk
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "completion_tokens", completion_tokens)
                
    except Exception as e:
        err_str = f"Error during final response synthesis: {str(e)}"
        logger.error(err_str)
        yield {
            "event": "error",
            "data": json.dumps({"detail": err_str})
        }
        return

    # Post-flight guardrails (PII and toxicity)
    synthesized_ans = "".join(ans_accum)
    scrubbed_ans = scrub_pii(synthesized_ans)

    # 1. Toxicity check
    toxicity_threshold = getattr(settings, "TOXICITY_THRESHOLD", 0.1)
    toxicity_score = check_toxicity(scrubbed_ans)
    if toxicity_score > toxicity_threshold:
        logger.warning(
            "Post-flight rejection: toxicity score %.2f exceeds threshold %.2f",
            toxicity_score, toxicity_threshold
        )
        yield {
            "event": "error",
            "data": json.dumps({"detail": "Response rejected: security policy violation (toxic content detected)."})
        }
        return

    # 2. Hallucination grounding check (for successful retrieval results)
    retrieved_contexts = [
        r.content for r in final_state["subagent_results"]
        if r.source == "retrieval" and r.status == SubAgentStatus.SUCCESS
    ]
    if retrieved_contexts:
        context_str = "\n\n".join(retrieved_contexts)
        is_grounded, grounding_err = await check_hallucination_grounding(scrubbed_ans, context_str)
        if not is_grounded:
            logger.warning("Post-flight rejection: hallucination grounding check failed.")
            yield {
                "event": "error",
                "data": json.dumps({"detail": grounding_err})
            }
            return

    # 7. Update Redis Conversation Cache (and scrub prompt of PII first)
    scrubbed_prompt = scrub_pii(clean_prompt)
    history.append({"role": "user", "content": scrubbed_prompt})
    history.append({"role": "assistant", "content": scrubbed_ans})
    if redis:
        try:
            await redis.set(f"guardroute:session:{session_id}", json.dumps(history), ex=1800)
        except Exception as e:
            logger.warning("Failed to save history to Redis: %s", e)

    duration = time.time() - start_time
    
    # If token counts weren't returned by streaming chunks, estimate them
    if not prompt_tokens:
        prompt_tokens = len(synthesis_prompt) // 4
    if not completion_tokens:
        completion_tokens = len(scrubbed_ans) // 4

    # Scrub PII from the user prompt inside the Kafka / Postgres trace payload
    scrubbed_trace_prompt = scrub_pii(prompt)
    trace = {
        "session_id": session_id,
        "prompt": scrubbed_trace_prompt,
        "complexity": final_state["complexity"],
        "subagents_ran": final_state["required_agents"],
        "final_response": scrubbed_ans,
        "duration_sec": duration,
        "timestamp": start_time
    }

    # 8. Persist and Publish Traces
    await persist_session_and_usage(trace, prompt_tokens, completion_tokens)
    publish_trace_to_kafka(trace)

    yield {
        "event": "done",
        "data": json.dumps({
            "session_id": session_id,
            "response": scrubbed_ans,
            "complexity": final_state["complexity"],
            "subagents_ran": final_state["required_agents"],
            "duration_sec": duration
        })
    }


# --- Legacy / Wrapper Non-Streaming Orchestrator ---

async def execute_orchestrator(prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Execute orchestrator pipeline and return accumulated final trace result."""
    trace_res = {}
    async for event in execute_orchestrator_stream(prompt, session_id):
        if event["event"] == "metadata":
            trace_res.update(json.loads(event["data"]))
        elif event["event"] == "error":
            raise ValueError(json.loads(event["data"]).get("detail", "Request failed security policy."))
        elif event["event"] == "done":
            trace_res.update(json.loads(event["data"]))
            trace_res["final_response"] = trace_res.get("response")
    return trace_res

