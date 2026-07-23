"""GuardRoute web search subagent.

Queries DuckDuckGo search engine to collect live context from the web.
"""

import asyncio
import logging
from typing import List, Dict, Any
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ModuleNotFoundError:
    HAS_DDGS = False
    DDGS = None

from common.schemas.agent_types import SubAgentResult, SubAgentStatus

logger = logging.getLogger("guardroute.agents.search")


def _execute_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Synchronous search function executed in a separate thread."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=limit))
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            return formatted
    except Exception as e:
        logger.error("DuckDuckGo search raw execution failed: %s", e)
        raise e


async def run_web_search(query: str, limit: int = 5, timeout: float = 2.0) -> SubAgentResult:
    """Invokes DuckDuckGo search with strict timeout constraint."""
    logger.info("Web Search Agent: querying DuckDuckGo for: '%s'", query[:40])
    try:
        # Run synchronous web request inside thread pool to prevent blocking event loop
        search_task = asyncio.to_thread(_execute_search, query, limit)
        results = await asyncio.wait_for(search_task, timeout=timeout)
        
        # Format the result payload
        import json
        content_str = json.dumps(results, indent=2)
        
        return SubAgentResult(
            source="web_search",
            status=SubAgentStatus.SUCCESS,
            content_type="json",
            content=content_str,
            token_count=len(content_str) // 4
        )
    except asyncio.TimeoutError:
        logger.warning("Web search timed out after %ss for query: '%s'", timeout, query[:40])
        return SubAgentResult(
            source="web_search",
            status=SubAgentStatus.TIMEOUT,
            error_message=f"Web search timed out after {timeout} seconds."
        )
    except Exception as e:
        logger.error("Web search failed for query '%s': %s", query[:40], e)
        return SubAgentResult(
            source="web_search",
            status=SubAgentStatus.ERROR,
            error_message=f"Web search failed: {str(e)}"
        )
