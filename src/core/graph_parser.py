"""Visual workflow graph translation parser for GuardRoute.

Converts ReactFlow visual graph JSON configurations (nodes, edges, node parameters)
into compiled, executable LangGraph StateGraph dynamic workflows with topology validation.
Supports V5 node types: Classifier, Agent, MultiAgent, Retrieval, Coding, WebSearch, Synthesis, Gather,
Action, FinalMessage, IfElse, Webhook, APICall, Eval, MCPTool, Router, Transform.
"""

import logging
from typing import Any, Dict, List, Optional, Set

try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except ModuleNotFoundError:
    class DummyStateGraph:
        def __init__(self, state_schema=None):
            self.nodes = {}
            self.edges = []
            self.entry_point = None

        def add_node(self, node_id, action):
            self.nodes[node_id] = action

        def add_edge(self, src, tgt):
            self.edges.append((src, tgt))

        def set_entry_point(self, entry_id):
            self.entry_point = entry_id

        def compile(self):
            return self

    StateGraph = DummyStateGraph
    START = "START"
    END = "END"
    HAS_LANGGRAPH = False

from projects.guardroute.src.orchestrator import (
    GraphState,
    classify_node,
    retrieval_node,
    coding_node,
    web_search_node,
    gather_node,
)
from projects.guardroute.src.nodes.conditional_evaluator import evaluate_condition
from projects.guardroute.src.nodes.webhook_executor import execute_webhook
from projects.guardroute.src.nodes.api_call_executor import execute_api_call
from projects.guardroute.src.nodes.eval_executor import execute_eval_node
from projects.guardroute.src.nodes.mcp_tool_executor import execute_mcp_tool
from projects.guardroute.src.nodes.router_executor import evaluate_routes
from projects.guardroute.src.nodes.transform_executor import execute_transform
from projects.guardroute.src.nodes.multi_agent_executor import execute_multi_agent
from projects.guardroute.src.nodes.action_executor import execute_action_node
from projects.guardroute.src.nodes.final_message_executor import execute_final_message_node

logger = logging.getLogger("guardroute.core.graph_parser")


class GraphValidationError(Exception):
    """Raised when a visual workflow graph topology violates safety constraints."""
    pass


class GraphParser:
    """Parses ReactFlow graph JSON and builds executable LangGraph StateGraph instances."""

    SUPPORTED_NODE_TYPES = {
        # Core V2 & Multi-Agent V5
        "ClassifierNode", "classifier",
        "AgentNode", "agent",
        "MultiAgentNode", "multi_agent",
        "RetrievalNode", "retrieval",
        "CodingNode", "coding",
        "WebSearchNode", "web_search",
        "SynthesisNode", "synthesis",
        "GatherNode", "gather",
        "ActionNode", "action",
        "FinalMessageNode", "final_message",
        # Logic V5
        "IfElseNode", "if_else",
        "RouterNode", "router",
        "TransformNode", "transform",
        # Integrations V5
        "WebhookNode", "webhook",
        "APICallNode", "api_call",
        # Evaluation V5
        "EvalNode", "eval",
        # Tools V5
        "MCPToolNode", "mcp_tool"
    }

    TERMINAL_NODE_TYPES = {
        "ActionNode", "action",
        "FinalMessageNode", "final_message",
        "SynthesisNode", "synthesis",
        "GatherNode", "gather",
    }

    def __init__(self, graph_json: Optional[Dict[str, Any]] = None):
        self.graph_json = graph_json or {}

    def validate_graph(self, graph_json: Optional[Dict[str, Any]] = None) -> bool:
        """Validates ReactFlow JSON topology for safety constraints.

        Constraints enforced:
        - Must contain non-empty 'nodes' list.
        - All edges must reference valid node IDs.
        - Must not contain cycles (directed acyclic graph constraint for safety).
        - **Rule 8 Strict Terminal Constraint:** All execution paths must end in an approved terminal node
          (ActionNode, FinalMessageNode, or SynthesisNode/GatherNode).
        """
        data = graph_json or self.graph_json
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not nodes:
            raise GraphValidationError("Workflow graph must contain at least one node.")

        node_map = {node["id"]: node for node in nodes if "id" in node}
        node_ids = set(node_map.keys())

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

        # --- Rule 8 Validation Check ---
        # Find all leaf nodes (nodes with 0 outgoing edges)
        all_sources = {e.get("source") for e in edges}
        leaf_ids = [nid for nid in node_ids if nid not in all_sources]

        for leaf_id in leaf_ids:
            leaf_node = node_map[leaf_id]
            ntype = leaf_node.get("type", "AgentNode")
            if ntype not in self.TERMINAL_NODE_TYPES:
                raise GraphValidationError(
                    f"Rule 8 Violation: Path terminates at non-terminal node '{leaf_id}' of type '{ntype}'. "
                    "Every workflow path MUST conclude in an ActionNode, FinalMessageNode, or SynthesisNode."
                )

        return True

    def build_langgraph(self, graph_json: Optional[Dict[str, Any]] = None) -> Any:
        """Converts validated visual graph JSON into a compiled LangGraph StateGraph executable."""
        data = graph_json or self.graph_json
        if not data or not data.get("nodes"):
            logger.info("Empty graph JSON provided; returning standard orchestrator graph.")
            from projects.guardroute.src.orchestrator import create_orchestrator_graph
            return create_orchestrator_graph()

        # Validate topology (enforces Rule 8 terminal constraint & cycle checks)
        self.validate_graph(data)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        workflow = StateGraph(GraphState)
        node_map = {}
        entry_node_id = None
        terminal_node_ids = set()

        for node in nodes:
            nid = node["id"]
            ntype = node.get("type", "AgentNode")
            data_cfg = node.get("data", {})
            data_cfg["node_id"] = nid

            if ntype in {"ClassifierNode", "classifier"}:
                workflow.add_node(nid, classify_node)
                if not entry_node_id:
                    entry_node_id = nid
            elif ntype in {"RetrievalNode", "retrieval"}:
                workflow.add_node(nid, retrieval_node)
            elif ntype in {"CodingNode", "coding"}:
                workflow.add_node(nid, coding_node)
            elif ntype in {"WebSearchNode", "web_search"}:
                workflow.add_node(nid, web_search_node)
            elif ntype in {"SynthesisNode", "GatherNode", "synthesis", "gather"}:
                workflow.add_node(nid, gather_node)
                terminal_node_ids.add(nid)

            # V5 Multi-Agent Node
            elif ntype in {"MultiAgentNode", "multi_agent"}:
                async def _multi_agent_fn(state: GraphState, cfg=data_cfg):
                    return await execute_multi_agent(cfg, state)
                workflow.add_node(nid, _multi_agent_fn)

            # V5 Terminal Nodes
            elif ntype in {"ActionNode", "action"}:
                async def _action_fn(state: GraphState, cfg=data_cfg):
                    return await execute_action_node(cfg, state)
                workflow.add_node(nid, _action_fn)
                terminal_node_ids.add(nid)

            elif ntype in {"FinalMessageNode", "final_message"}:
                async def _final_message_fn(state: GraphState, cfg=data_cfg):
                    return await execute_final_message_node(cfg, state)
                workflow.add_node(nid, _final_message_fn)
                terminal_node_ids.add(nid)

            # V5 Logic Nodes
            elif ntype in {"IfElseNode", "if_else"}:
                async def _if_else_fn(state: GraphState, cfg=data_cfg):
                    cond_cfg = cfg.get("condition", cfg)
                    result = evaluate_condition(cond_cfg, state)
                    flags = dict(state.get("conditional_flags", {}))
                    flags[nid] = result
                    return {"conditional_flags": flags}
                workflow.add_node(nid, _if_else_fn)

            elif ntype in {"RouterNode", "router"}:
                async def _router_fn(state: GraphState, cfg=data_cfg):
                    routes = cfg.get("routes", [])
                    default_r = cfg.get("default_route", "default")
                    selected = evaluate_routes(routes, default_r, state)
                    flags = dict(state.get("conditional_flags", {}))
                    flags[nid] = selected
                    return {"conditional_flags": flags}
                workflow.add_node(nid, _router_fn)

            elif ntype in {"TransformNode", "transform"}:
                async def _transform_fn(state: GraphState, cfg=data_cfg):
                    res = execute_transform(cfg, state)
                    outputs = dict(state.get("transform_outputs", {}))
                    outputs[nid] = res.get("output")
                    return {"transform_outputs": outputs}
                workflow.add_node(nid, _transform_fn)

            # V5 Integration Nodes
            elif ntype in {"WebhookNode", "webhook"}:
                async def _webhook_fn(state: GraphState, cfg=data_cfg):
                    res = await execute_webhook(cfg, state)
                    wh_results = dict(state.get("webhook_results", {}))
                    wh_results[nid] = res
                    return {"webhook_results": wh_results}
                workflow.add_node(nid, _webhook_fn)

            elif ntype in {"APICallNode", "api_call"}:
                async def _api_call_fn(state: GraphState, cfg=data_cfg):
                    res = await execute_api_call(cfg, state)
                    api_results = dict(state.get("api_call_results", {}))
                    api_results[nid] = res
                    return {"api_call_results": api_results}
                workflow.add_node(nid, _api_call_fn)

            # V5 Evaluation Node
            elif ntype in {"EvalNode", "eval"}:
                async def _eval_fn(state: GraphState, cfg=data_cfg):
                    res = await execute_eval_node(cfg, state)
                    eval_res = dict(state.get("eval_results", {}))
                    eval_res[nid] = res
                    return {"eval_results": eval_res}
                workflow.add_node(nid, _eval_fn)

            # V5 Tools Node
            elif ntype in {"MCPToolNode", "mcp_tool"}:
                async def _mcp_fn(state: GraphState, cfg=data_cfg):
                    res = await execute_mcp_tool(cfg, state)
                    mcp_res = dict(state.get("mcp_tool_results", {}))
                    mcp_res[nid] = res
                    return {"mcp_tool_results": mcp_res}
                workflow.add_node(nid, _mcp_fn)

            else:
                # Fallback AgentNode
                workflow.add_node(nid, retrieval_node)

            node_map[nid] = ntype

        # Default entry point
        if not entry_node_id:
            entry_node_id = nodes[0]["id"]

        workflow.set_entry_point(entry_node_id)

        # Wire edges
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in node_map and tgt in node_map:
                workflow.add_edge(src, tgt)

        # Connect terminal nodes to END
        for t_id in terminal_node_ids:
            workflow.add_edge(t_id, END)

        if not terminal_node_ids:
            all_sources = {e.get("source") for e in edges}
            leaves = [nid for nid in node_map if nid not in all_sources]
            for leaf in leaves:
                workflow.add_edge(leaf, END)

        return workflow.compile()


def parse_graph_json_to_langgraph(graph_json: Dict[str, Any]) -> Any:
    """Convenience function to parse JSON graph and compile a LangGraph StateGraph."""
    parser = GraphParser(graph_json)
    return parser.build_langgraph()
