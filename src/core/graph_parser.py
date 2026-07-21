"""Visual workflow graph translation parser for GuardRoute.

Converts ReactFlow visual graph JSON configurations (nodes, edges, node parameters)
into compiled, executable LangGraph StateGraph dynamic workflows with topology validation.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from langgraph.graph import StateGraph, START, END
from projects.guardroute.src.orchestrator import (
    GraphState,
    classify_node,
    retrieval_node,
    coding_node,
    web_search_node,
    gather_node,
)

logger = logging.getLogger("guardroute.core.graph_parser")


class GraphValidationError(Exception):
    """Raised when a visual workflow graph topology violates safety constraints."""
    pass


class GraphParser:
    """Parses ReactFlow graph JSON and builds executable LangGraph StateGraph instances."""

    SUPPORTED_NODE_TYPES = {
        "ClassifierNode",
        "AgentNode",
        "RetrievalNode",
        "CodingNode",
        "WebSearchNode",
        "SynthesisNode",
        "GatherNode"
    }

    def __init__(self, graph_json: Optional[Dict[str, Any]] = None):
        self.graph_json = graph_json or {}

    def validate_graph(self, graph_json: Optional[Dict[str, Any]] = None) -> bool:
        """Validates ReactFlow JSON topology for safety constraints.

        Constraints enforced:
        - Must contain non-empty 'nodes' and 'edges' lists.
        - All edges must reference valid node IDs.
        - Must contain at least one terminal node (SynthesisNode/GatherNode).
        - Must not contain cycles (directed acyclic graph constraint for safety).
        """
        data = graph_json or self.graph_json
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not nodes:
            raise GraphValidationError("Workflow graph must contain at least one node.")

        node_ids = {node["id"] for node in nodes if "id" in node}
        node_types = {node.get("type", "AgentNode") for node in nodes}

        # Check edge validity
        adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}

        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src not in node_ids or tgt not in node_ids:
                raise GraphValidationError(f"Edge references invalid node ID: source={src}, target={tgt}")
            adj[src].append(tgt)
            in_degree[tgt] += 1

        # Cycle detection using Kahn's algorithm
        queue = [nid for nid in node_ids if in_degree[nid] == 0]
        visited_count = 0
        in_degree_copy = dict(in_degree)

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for neighbor in adj[curr]:
                in_degree_copy[neighbor] -= 1
                if in_degree_copy[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count < len(node_ids):
            raise GraphValidationError("Workflow graph contains an infinite cycle. Workflows must be acyclic.")

        # Check for terminal node (gather or synthesis)
        has_terminal = any(
            t in self.SUPPORTED_NODE_TYPES for t in node_types if t in {"SynthesisNode", "GatherNode"}
        ) or any(len(adj[nid]) == 0 for nid in node_ids)

        if not has_terminal:
            raise GraphValidationError("Workflow graph must have at least one terminal synthesis or output node.")

        return True

    def build_langgraph(self, graph_json: Optional[Dict[str, Any]] = None) -> Any:
        """Converts validated visual graph JSON into a compiled LangGraph StateGraph executable.

        Args:
            graph_json: ReactFlow graph dictionary containing 'nodes' and 'edges'.

        Returns:
            Compiled StateGraph executable object.
        """
        data = graph_json or self.graph_json
        if not data or not data.get("nodes"):
            logger.info("Empty graph JSON provided; returning standard orchestrator graph.")
            from projects.guardroute.src.orchestrator import create_orchestrator_graph
            return create_orchestrator_graph()

        # Validate topology
        self.validate_graph(data)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        workflow = StateGraph(GraphState)
        node_map = {}
        entry_node_id = None
        gather_node_ids = set()

        for node in nodes:
            nid = node["id"]
            ntype = node.get("type", "AgentNode")

            if ntype == "ClassifierNode":
                workflow.add_node(nid, classify_node)
                if not entry_node_id:
                    entry_node_id = nid
            elif ntype == "RetrievalNode":
                workflow.add_node(nid, retrieval_node)
            elif ntype == "CodingNode":
                workflow.add_node(nid, coding_node)
            elif ntype == "WebSearchNode":
                workflow.add_node(nid, web_search_node)
            elif ntype in {"SynthesisNode", "GatherNode"}:
                workflow.add_node(nid, gather_node)
                gather_node_ids.add(nid)
            else:
                # Custom AgentNode -> default to retrieval/agent execution
                workflow.add_node(nid, retrieval_node)

            node_map[nid] = ntype

        # Default entry point if no ClassifierNode was specified
        if not entry_node_id:
            entry_node_id = nodes[0]["id"]

        workflow.set_entry_point(entry_node_id)

        # Wire edges
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in node_map and tgt in node_map:
                workflow.add_edge(src, tgt)

        # Wire terminal nodes to END
        for g_id in gather_node_ids:
            workflow.add_edge(g_id, END)

        # If no explicit gather node edge to END, connect outer leaf nodes to END
        if not gather_node_ids:
            all_sources = {e.get("source") for e in edges}
            leaves = [nid for nid in node_map if nid not in all_sources]
            for leaf in leaves:
                workflow.add_edge(leaf, END)

        return workflow.compile()


def parse_graph_json_to_langgraph(graph_json: Dict[str, Any]) -> Any:
    """Convenience function to parse JSON graph and compile a LangGraph StateGraph."""
    parser = GraphParser(graph_json)
    return parser.build_langgraph()
