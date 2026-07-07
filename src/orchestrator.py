"""GuardRoute Multi-Agent Orchestrator.

Implements a LangGraph StateGraph performing Scatter-Gather routing
across parallel subagents with LiteLLM Google free-tier fallback support.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
import operator

from common.config.settings import settings
from common.clients.inference import InferenceClient
from common.clients.litellm import completion_with_fallback
from common.schemas.agent_types import (
    SubAgentResult,
    SubAgentStatus,
    TaskComplexity,
)
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger("guardroute.orchestrator")


class GraphState(TypedDict):
    """The state of the GuardRoute orchestrator graph."""

    prompt: str
    complexity: str
    required_agents: List[str]
    subagent_results: Annotated[List[SubAgentResult], operator.add]
    final_response: str


# Helper to convert messages for Capability Isolation (Google to OpenRouter fallback constraints)
def clean_messages_for_model(messages: List[Dict[str, str]], model_name: str) -> List[Dict[str, str]]:
    """Modifies prompts and payload limits depending on resolved model capabilities."""
    cleaned = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        # Capability Isolation: Check for context window limits
        if "lite" in model_name.lower() or "free" in model_name.lower():
            # Truncate content to fit smaller context models (e.g. max 4000 characters)
            if len(content) > 4000:
                logger.info("Capability Isolation: Truncating context for free model: %s", model_name)
                content = content[:4000] + "\n[Context Truncated due to model payload limits]"
                
        cleaned.append({"role": role, "content": content})
    return cleaned


async def completion_with_free_gemini_fallback(
    messages: List[Dict[str, str]],
    primary_model: str = "gemini/gemini-3.5-flash",
    fallback_model: str = "gemini/gemini-2.5-flash-lite",
    **kwargs: Any
) -> Any:
    """Invokes LiteLLM completion, falling back between free Gemini models on failure."""
    models_to_try = [primary_model, fallback_model]
    last_error = None

    for model in models_to_try:
        try:
            logger.info("Attempting completion with model: %s", model)
            cleaned_msgs = clean_messages_for_model(messages, model)
            
            # Start timer to track transition latency
            start_time = time.time()
            
            # Call completion
            response = await completion_with_fallback(
                model=model,
                messages=cleaned_msgs,
                fallbacks=[],  # We manage fallbacks manually here for tracing and logging
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            logger.info("Completion successful using %s. Latency: %.2f ms", model, latency_ms)
            return response
            
        except Exception as e:
            logger.warning("Completion failed for model %s: %s", model, e)
            last_error = e
            
    # If all fail, raise exception
    raise RuntimeError(f"All free Gemini models failed in fallback chain: {str(last_error)}")


async def run_classification(prompt: str, inference_client: InferenceClient) -> Dict[str, Any]:
    """Helper to classify prompt complexity using local classifier."""
    try:
        logger.info("Routing prompt to classifier...")
        res = await inference_client.classify(prompt)
        return res
    except Exception as e:
        logger.error("Classifier endpoint failed: %s. Falling back to rule-based routing.", e)
        # Rule-based fallback
        prompt_lower = prompt.lower()
        if "code" in prompt_lower or "script" in prompt_lower or "python" in prompt_lower:
            return {"complexity": "complex", "required_agents": ["retrieval", "coding", "web_search"]}
        elif "search" in prompt_lower or "retrieve" in prompt_lower or "find" in prompt_lower:
            return {"complexity": "medium", "required_agents": ["retrieval"]}
        else:
            return {"complexity": "simple", "required_agents": []}


# --- subagents implementations ---

async def retrieval_agent(prompt: str) -> SubAgentResult:
    """Subagent executing document/video RAG search via SyntraFlow."""
    logger.info("Retrieval Agent: starting search context gathering...")
    try:
        # Connect directly to SyntraFlow retrieval engine if running in same monorepo
        from projects.syntraflow.src.retrieval import RetrievalEngine
        from projects.syntraflow.src.vectors.client import VectorClient
        
        vector = VectorClient()
        engine = RetrievalEngine(vector)
        
        # Check if we have embeddings. If not, mock them
        try:
            inference = InferenceClient(base_url=settings.INFERENCE_SERVER_URL)
            embeds = await inference.embed(texts=[prompt])
            query_vector = embeds[0]
            await inference.close()
        except Exception:
            query_vector = [0.0] * 768
            
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
        logger.error("Retrieval subagent error: %s", e)
        return SubAgentResult(
            source="retrieval",
            status=SubAgentStatus.ERROR,
            error_message=str(e)
        )


async def coding_agent(prompt: str) -> SubAgentResult:
    """Subagent performing mathematical calculation or code sandbox emulation."""
    logger.info("Coding Agent: running code sandbox emulation...")
    try:
        # Simulate standard coding sandbox output
        await asyncio.sleep(0.2)
        code_output = (
            "# Auto-generated coding agent analysis block\n"
            "data = [12.5, 14.8, 11.2, 16.5]\n"
            "mean_val = sum(data) / len(data)\n"
            "print(f'Mean value calculated: {mean_val}')"
        )
        return SubAgentResult(
            source="coding",
            status=SubAgentStatus.SUCCESS,
            content_type="code",
            content=code_output,
            token_count=len(code_output) // 4
        )
    except Exception as e:
        return SubAgentResult(
            source="coding",
            status=SubAgentStatus.ERROR,
            error_message=str(e)
        )


async def web_search_agent(prompt: str) -> SubAgentResult:
    """Subagent calling mock web search API to collect external data."""
    logger.info("Web Search Agent: querying external commodities databases...")
    try:
        await asyncio.sleep(0.1)
        search_res = (
            "Commodity Indices Update:\n"
            "- Crude Oil (WTI): $75.40 (+1.2%)\n"
            "- Gold (oz): $2340.50 (-0.4%)\n"
            "- Copper (lb): $4.12 (+0.8%)"
        )
        return SubAgentResult(
            source="web_search",
            status=SubAgentStatus.SUCCESS,
            content_type="text",
            content=search_res,
            token_count=len(search_res) // 4
        )
    except Exception as e:
        return SubAgentResult(
            source="web_search",
            status=SubAgentStatus.ERROR,
            error_message=str(e)
        )


# --- Kafka Logging Client ---

def publish_trace_to_kafka(trace_data: Dict[str, Any]) -> None:
    """Publish the execution trace data asynchronously to a Kafka topic."""
    try:
        from confluent_kafka import Producer
        conf = {"bootstrap.servers": "localhost:9092"}
        producer = Producer(conf)
        
        producer.produce(
            "guardroute-traces",
            key=str(trace_data.get("session_id", time.time())),
            value=json.dumps(trace_data)
        )
        producer.flush()
        logger.info("Trace event successfully published to Kafka topic: guardroute-traces")
    except Exception as e:
        # Fallback to local logs silently without blocking transaction thread
        logger.debug("Kafka broker unavailable: %s. Trace logged locally.", e)


# --- LangGraph Nodes ---

async def classify_node(state: GraphState) -> Dict[str, Any]:
    inference = InferenceClient(base_url=settings.INFERENCE_SERVER_URL)
    result = await run_classification(state["prompt"], inference)
    await inference.close()
    return {
        "complexity": result["complexity"],
        "required_agents": result["required_agents"]
    }


async def retrieval_node(state: GraphState) -> Dict[str, Any]:
    res = await retrieval_agent(state["prompt"])
    return {"subagent_results": [res]}


async def coding_node(state: GraphState) -> Dict[str, Any]:
    res = await coding_agent(state["prompt"])
    return {"subagent_results": [res]}


async def web_search_node(state: GraphState) -> Dict[str, Any]:
    res = await web_search_agent(state["prompt"])
    return {"subagent_results": [res]}


async def gather_node(state: GraphState) -> Dict[str, Any]:
    # Gather node merges subagent results (automatically handled by operator.add)
    logger.info("Gather Phase: Consolidated %d subagent result(s).", len(state["subagent_results"]))
    return {}


async def synthesis_node(state: GraphState) -> Dict[str, Any]:
    # 1. Compile gathered contexts
    contexts = []
    for r in state["subagent_results"]:
        if r.status == SubAgentStatus.SUCCESS:
            contexts.append(f"--- Context Source: {r.source} ---\n{r.content}")
        else:
            contexts.append(f"--- Context Source: {r.source} (FAILED) ---\nError: {r.error_message}")
            
    compiled_context = "\n\n".join(contexts)
    
    # 2. Build synthesis prompt
    prompt = (
        "You are the Synthesis Agent of the GuardRoute orchestrator. "
        "Formulate a complete, coordinated, and factual response to the user query "
        "using the gathered context below. If any subagents failed, proceed with the successful contexts.\n\n"
        f"User Prompt: {state['prompt']}\n\n"
        f"Gathered Context:\n{compiled_context}"
    )
    
    # 3. Request Completion from free models
    try:
        response = await completion_with_free_gemini_fallback(
            messages=[{"role": "user", "content": prompt}]
        )
        ans = response.choices[0].message.content
    except Exception as e:
        ans = f"Error during final response synthesis: {str(e)}"
        
    return {"final_response": ans}


# --- Build LangGraph Workflow ---

def create_orchestrator_graph() -> StateGraph:
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("gather", gather_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # Define routing decisions
    workflow.set_entry_point("classify")
    
    def scatter_router(state: GraphState):
        agents = state["required_agents"]
        if not agents:
            # Route straight to synthesis if simple
            return ["synthesis"]
        return agents
        
    # Map edges
    workflow.add_conditional_edges(
        "classify",
        scatter_router,
        {
            "retrieval": "retrieval",
            "coding": "coding",
            "web_search": "web_search",
            "synthesis": "synthesis"
        }
    )
    
    # Join edges to gather node
    workflow.add_edge("retrieval", "gather")
    workflow.add_edge("coding", "gather")
    workflow.add_edge("web_search", "gather")
    
    workflow.add_edge("gather", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()


async def execute_orchestrator(prompt: str) -> Dict[str, Any]:
    """Execute the complete GuardRoute orchestrator pipeline."""
    graph = create_orchestrator_graph()
    
    session_id = f"sess_{int(time.time())}"
    start_time = time.time()
    
    initial_state = {
        "prompt": prompt,
        "complexity": "",
        "required_agents": [],
        "subagent_results": [],
        "final_response": ""
    }
    
    # Run graph execution
    final_state = await graph.ainvoke(initial_state)
    
    duration = time.time() - start_time
    
    # Construct complete execution trace payload
    trace = {
        "session_id": session_id,
        "prompt": prompt,
        "complexity": final_state.get("complexity"),
        "subagents_ran": final_state.get("required_agents"),
        "final_response": final_state.get("final_response"),
        "duration_sec": duration,
        "timestamp": start_time
    }
    
    # Asynchronously publish execution logs to Kafka topic
    publish_trace_to_kafka(trace)
    
    return trace
