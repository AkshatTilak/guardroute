"""GuardRoute task complexity classifier.

Supports:
1. Local model routing via Inference Server (Arch-Router)
2. Semantic routing using active embedding model (zero VRAM)
3. Cloud routing via Gemini 3.5 Flash API with JSON mode
4. Rule-based regex fallback classifier
5. Circuit breaker pattern for remote endpoints
"""

import time
import json
import logging
import asyncio
from typing import List, Dict, Any, Tuple

from common.config.settings import settings
from common.clients.inference import InferenceClient
from common.clients.litellm import completion_with_fallback
from common.models.registry import get_active_model

logger = logging.getLogger("guardroute.agents.classifier")

# Predefined descriptions for the Semantic Router
ROUTE_DESCRIPTIONS = {
    "simple": "A simple greeting, general conversation, casual chit-chat, hello, hi, how are you, tell me a joke, general question.",
    "medium": "Retrieve documents, search database, find info in papers, query knowledge graph, document search, retrieve pdf information.",
    "complex": "Run python code, write code, calculate math equations, execute script, search the web for live commodities, live stock prices, web search."
}

# Mapping of route to required subagents
ROUTE_AGENTS = {
    "simple": [],
    "medium": ["retrieval"],
    "complex": ["retrieval", "coding", "web_search"]
}


class ClassifierCircuitBreaker:
    """Tracks remote classifier failure state and handles cooling down."""

    def __init__(self, max_failures: int = 5, cooldown_sec: float = 60.0):
        self.max_failures = max_failures
        self.cooldown_sec = cooldown_sec
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.is_degraded = False

    def check_active(self) -> bool:
        """Returns True if the classifier endpoint is healthy, False if bypassed."""
        if self.is_degraded:
            now = time.time()
            if now - self.last_failure_time > self.cooldown_sec:
                # Cooldown elapsed, attempt recovery
                logger.info("Circuit breaker: recovery attempt for classifier endpoint.")
                return True
            return False
        return True

    def record_success(self):
        self.consecutive_failures = 0
        self.is_degraded = False

    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= self.max_failures:
            self.is_degraded = True
            logger.error("Circuit breaker: classifier reached %d consecutive failures. Bypassing.", self.consecutive_failures)


# Singleton circuit breaker state
circuit_breaker = ClassifierCircuitBreaker()

# Cache for semantic route embeddings
cached_route_embeddings: Dict[str, List[float]] = {}


def rule_based_classify(prompt: str) -> Dict[str, Any]:
    """Fallback classifier using keyword rules."""
    prompt_lower = prompt.lower()
    if "code" in prompt_lower or "script" in prompt_lower or "python" in prompt_lower or "calculate" in prompt_lower or "math" in prompt_lower:
        return {
            "complexity": "complex",
            "required_agents": ["retrieval", "coding", "web_search"],
            "confidence": 0.5
        }
    elif "search" in prompt_lower or "retrieve" in prompt_lower or "find" in prompt_lower or "document" in prompt_lower:
        return {
            "complexity": "medium",
            "required_agents": ["retrieval"],
            "confidence": 0.5
        }
    else:
        return {
            "complexity": "simple",
            "required_agents": [],
            "confidence": 0.5
        }


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(x * x for x in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


# Import math inside the function or file
import math

async def get_semantic_routing(prompt: str, inference_client: InferenceClient) -> Dict[str, Any]:
    """Classifies complexity using Semantic Router (cosine similarity of embeddings)."""
    global cached_route_embeddings
    
    # 1. Embed query
    query_embeds = await inference_client.embed(texts=[prompt])
    query_vector = query_embeds[0]

    # 2. Embed reference routes if not cached
    if not cached_route_embeddings:
        logger.info("Initializing semantic route description embeddings...")
        routes = list(ROUTE_DESCRIPTIONS.keys())
        descriptions = list(ROUTE_DESCRIPTIONS.values())
        desc_embeds = await inference_client.embed(texts=descriptions)
        for route, embed in zip(routes, desc_embeds):
            cached_route_embeddings[route] = embed

    # 3. Calculate similarities
    scores = {}
    for route, route_vector in cached_route_embeddings.items():
        scores[route] = _cosine_similarity(query_vector, route_vector)

    # 4. Find the best matching route
    best_route = max(scores, key=scores.get)
    confidence = scores[best_route]
    
    logger.info("Semantic Router classified prompt as: '%s' (confidence: %.2f)", best_route, confidence)
    
    return {
        "complexity": best_route,
        "required_agents": ROUTE_AGENTS[best_route],
        "confidence": confidence
    }


async def get_cloud_classification(prompt: str) -> Dict[str, Any]:
    """Classifies complexity using cloud Gemini Completion model with structured JSON."""
    system_prompt = (
        "You are the GuardRoute task classifier. Analyze the user prompt and classify its complexity "
        "and required subagents. Return raw JSON matching this schema:\n"
        "{\n"
        '  "complexity": "simple|medium|complex",\n'
        '  "required_agents": ["retrieval", "coding", "web_search"],\n'
        '  "confidence": 0.95\n'
        "}\n"
        "Rules:\n"
        "- simple: greetings, casual chit-chat, simple generic requests (required_agents: []).\n"
        "- medium: searching documents, reading database records (required_agents: ['retrieval']).\n"
        "- complex: coding requests, execution sandboxes, math calculations, searching live web data (required_agents: ['retrieval', 'coding', 'web_search']).\n"
        "Output raw JSON only. Do not wrap in markdown blocks."
    )

    response = await completion_with_fallback(
        model="gemini/gemini-3.5-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content.strip()
    # Strip markdown block wraps if present
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    data = json.loads(content)
    return {
        "complexity": data.get("complexity", "simple"),
        "required_agents": data.get("required_agents", []),
        "confidence": data.get("confidence", 0.9)
    }


async def classify_prompt(prompt: str, inference_client: InferenceClient) -> Dict[str, Any]:
    """Unified classification interface. Coordinates model registry and circuit breaker."""
    # 1. Check Circuit Breaker
    if not circuit_breaker.check_active():
        logger.warning("Classifier circuit breaker is active. Falling back directly to rule-based routing.")
        return rule_based_classify(prompt)

    try:
        # 2. Get active classifier model spec
        model_spec = await get_active_model("classifier")
        
        # 3. Route based on configured model
        if model_spec.mode == "cloud":
            res = await get_cloud_classification(prompt)
            circuit_breaker.record_success()
            return res
            
        elif model_spec.mode == "local":
            if model_spec.model_id == "semantic":
                res = await get_semantic_routing(prompt, inference_client)
                circuit_breaker.record_success()
                return res
            else:
                # Call local Inference Server
                res = await inference_client.classify(prompt)
                circuit_breaker.record_success()
                return res
        else:
            raise ValueError(f"Unknown classifier model mode: {model_spec.mode}")

    except Exception as e:
        logger.error("Classifier model execution failed: %s. Triggering fallback.", e)
        circuit_breaker.record_failure()
        return rule_based_classify(prompt)
