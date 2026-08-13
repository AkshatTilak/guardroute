"""Visual workflow graph translation parser for GuardRoute.

Converts ReactFlow visual graph JSON configurations (nodes, edges, node parameters)
into compiled, executable LangGraph StateGraph dynamic workflows with topology validation.
Supports V5 node types: Classifier, Agent, MultiAgent, Retrieval, Coding, WebSearch, Synthesis, Gather,
Action, FinalMessage, IfElse, Webhook, APICall, Eval, MCPTool, Router, Transform.
"""

import logging
from typing import Any, Dict, List, Optional

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
from projects.guardroute.src.nodes.router_executor import evaluate_routes
from projects.guardroute.src.nodes.transform_executor import execute_transform
from projects.guardroute.src.nodes.multi_agent_executor import execute_multi_agent
from projects.guardroute.src.nodes.action_executor import execute_action_node
from projects.guardroute.src.nodes.final_message_executor import execute_final_message_node
from projects.guardroute.src.nodes.tool_executor import execute_agent_tools

from sqlalchemy.ext.asyncio import AsyncSession
from common.schemas.workflows import NodeReference, ValidationIssue, ValidationResult
from common.services.hub_resolver import resolve_linked, HubLinkError
from common.services.hub_repository import get_hub

logger = logging.getLogger("guardroute.core.graph_parser")

NODE_REFERENCE_REQUIREMENTS: Dict[str, str] = {
    "AgentNode": "agent",
    "agent": "agent",
    "MultiAgentNode": "agent",  # list-valued in data["references"]
}


def collect_tool_references(graph_json: Dict[str, Any]) -> List[tuple[str, NodeReference]]:
    """Extract (node_id, NodeReference) pairs from agent-node tool bindings.

    Vector retrieval, MCP, and DB capabilities are now tools bound to agent
    nodes. Each tool binding is resolved as a cross-hub reference so the
    underlying resource (collection / MCP tool / credential) can be validated
    and linked.
    """
    refs: List[tuple[str, NodeReference]] = []
    nodes = graph_json.get("nodes", []) if isinstance(graph_json, dict) else []
    for node in nodes:
        nid = node.get("id")
        if not nid:
            continue
        ntype = node.get("type")
        if ntype not in {"AgentNode", "agent", "MultiAgentNode", "multi_agent"}:
            continue
        data_cfg = node.get("data", {})
        tools = data_cfg.get("tools") or []
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            ttype = tool.get("type")
            try:
                if ttype == "retrieval":
                    refs.append((nid, NodeReference(
                        type="collection", hub_id=tool["hub_id"], resource_id=tool["collection_id"])))
                elif ttype == "mcp":
                    refs.append((nid, NodeReference(
                        type="mcp_tool", hub_id=tool.get("hub_id") or "", resource_id=tool["tool_name"])))
                elif ttype == "db":
                    refs.append((nid, NodeReference(
                        type="credential", hub_id=tool.get("hub_id") or "", resource_id=tool["credential_id"])))
            except (KeyError, TypeError):
                continue
    return refs


def collect_references(graph_json: Dict[str, Any]) -> List[tuple[str, NodeReference]]:
    """Extract (node_id, NodeReference) pairs from graph JSON."""
    refs: List[tuple[str, NodeReference]] = []
    nodes = graph_json.get("nodes", []) if isinstance(graph_json, dict) else []
    for node in nodes:
        nid = node.get("id")
        if not nid:
            continue
        data_cfg = node.get("data", {})
        if "references" in data_cfg and isinstance(data_cfg["references"], list):
            for ref_item in data_cfg["references"]:
                try:
                    nr = NodeReference.model_validate(ref_item)
                    refs.append((nid, nr))
                except Exception:
                    pass
        elif "reference" in data_cfg and data_cfg["reference"]:
            try:
                nr = NodeReference.model_validate(data_cfg["reference"])
                refs.append((nid, nr))
            except Exception:
                pass
        else:
            ntype = node.get("type")
            req_type = NODE_REFERENCE_REQUIREMENTS.get(ntype)
            if req_type:
                alias_val = (
                    data_cfg.get(f"{req_type}_id")
                    or data_cfg.get("agent_id")
                    or data_cfg.get("collection_id")
                    or data_cfg.get("mcp_tool_id")
                )
                hub_id = data_cfg.get("hub_id")
                if alias_val and hub_id:
                    try:
                        nr = NodeReference(type=req_type, hub_id=hub_id, resource_id=alias_val)
                        refs.append((nid, nr))
                    except Exception:
                        pass
    return refs


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
    }

    # Node types that were removed as standalone flow steps. Vector retrieval,
    # MCP tools, and external database access are now capabilities (tools) bound
    # to an agent node, not standalone nodes. Any graph still using them is
    # rejected with a clear, actionable validation error.
    REMOVED_NODE_TYPES = {
        "RetrievalNode", "retrieval",
        "MCPToolNode", "mcp_tool",
        "DatabaseQueryNode", "database_query",
        "DBStoreNode", "db_store",
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

        # Reject removed standalone tool node types with an actionable message.
        for node in nodes:
            ntype = node.get("type", "AgentNode")
            if ntype in self.REMOVED_NODE_TYPES:
                raise GraphValidationError(
                    f"Node '{node.get('id')}' uses removed node type '{ntype}'. "
                    "Vector retrieval, MCP tools, and external database access are no longer standalone "
                    "nodes — attach them as tools to an Agent node instead (agent node 'tools' bindings)."
                )

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

    def _guard_node(self, node_id: str, fn: Any) -> Any:
        """Wrap a node function so exceptions are captured into state['errors'][node_id].

        This enables error-handle fallback routing: instead of propagating the
        exception (which would abort the whole run), the error is recorded in
        state['errors'] and the graph can transition along an 'error' handle.
        """
        async def _wrapped(state: GraphState) -> Dict[str, Any]:
            try:
                result = await fn(state)
                if result is None:
                    result = {}
                return result
            except Exception as exc:  # noqa: BLE001 - capture any node failure
                logger.warning("Node '%s' raised %s: %s", node_id, type(exc).__name__, exc)
                errors = dict(state.get("errors", {}) or {})
                errors[node_id] = {"type": type(exc).__name__, "message": str(exc)}
                return {"errors": errors}
        return _wrapped

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
                workflow.add_node(nid, self._guard_node(nid, _multi_agent_fn))

            # V5 Terminal Nodes
            elif ntype in {"ActionNode", "action"}:
                async def _action_fn(state: GraphState, cfg=data_cfg):
                    return await execute_action_node(cfg, state)
                workflow.add_node(nid, self._guard_node(nid, _action_fn))
                terminal_node_ids.add(nid)

            elif ntype in {"FinalMessageNode", "final_message"}:
                async def _final_message_fn(state: GraphState, cfg=data_cfg):
                    return await execute_final_message_node(cfg, state)
                workflow.add_node(nid, self._guard_node(nid, _final_message_fn))
                terminal_node_ids.add(nid)

            # V5 Logic Nodes
            elif ntype in {"IfElseNode", "if_else"}:
                async def _if_else_fn(state: GraphState, cfg=data_cfg):
                    cond_cfg = cfg.get("condition", cfg)
                    result = evaluate_condition(cond_cfg, state)
                    flags = dict(state.get("conditional_flags", {}))
                    flags[nid] = result
                    return {"conditional_flags": flags}
                workflow.add_node(nid, self._guard_node(nid, _if_else_fn))

            elif ntype in {"RouterNode", "router"}:
                async def _router_fn(state: GraphState, cfg=data_cfg):
                    routes = cfg.get("routes", [])
                    default_r = cfg.get("default_route", "default")
                    selected = evaluate_routes(routes, default_r, state)
                    flags = dict(state.get("conditional_flags", {}))
                    flags[nid] = selected
                    return {"conditional_flags": flags}
                workflow.add_node(nid, self._guard_node(nid, _router_fn))

            elif ntype in {"TransformNode", "transform"}:
                async def _transform_fn(state: GraphState, cfg=data_cfg):
                    res = execute_transform(cfg, state)
                    outputs = dict(state.get("transform_outputs", {}))
                    outputs[nid] = res.get("output")
                    return {"transform_outputs": outputs}
                workflow.add_node(nid, self._guard_node(nid, _transform_fn))

            # V5 Integration Nodes
            elif ntype in {"WebhookNode", "webhook"}:
                async def _webhook_fn(state: GraphState, cfg=data_cfg):
                    res = await execute_webhook(cfg, state)
                    wh_results = dict(state.get("webhook_results", {}))
                    wh_results[nid] = res
                    return {"webhook_results": wh_results}
                workflow.add_node(nid, self._guard_node(nid, _webhook_fn))

            elif ntype in {"APICallNode", "api_call"}:
                async def _api_call_fn(state: GraphState, cfg=data_cfg):
                    res = await execute_api_call(cfg, state)
                    api_results = dict(state.get("api_call_results", {}))
                    api_results[nid] = res
                    return {"api_call_results": api_results}
                workflow.add_node(nid, self._guard_node(nid, _api_call_fn))

            # V5 Evaluation Node
            elif ntype in {"EvalNode", "eval"}:
                async def _eval_fn(state: GraphState, cfg=data_cfg):
                    res = await execute_eval_node(cfg, state)
                    eval_res = dict(state.get("eval_results", {}))
                    eval_res[nid] = res
                    return {"eval_results": eval_res}
                workflow.add_node(nid, self._guard_node(nid, _eval_fn))

            # Agent node: invokes its bound tools (retrieval / mcp / db /
            # web_search / api_call) during the agent turn.
            elif ntype in {"AgentNode", "agent"}:
                async def _agent_fn(state: GraphState, cfg=data_cfg):
                    tools = cfg.get("tools") or []
                    tool_out = await execute_agent_tools(tools, state)
                    tool_results = dict(state.get("tool_results", {}))
                    tool_results[nid] = tool_out
                    return {"tool_results": tool_results}
                workflow.add_node(nid, self._guard_node(nid, _agent_fn))

            else:
                # Fallback AgentNode
                workflow.add_node(nid, retrieval_node)

            node_map[nid] = ntype

        # Default entry point
        if not entry_node_id:
            entry_node_id = nodes[0]["id"]

        workflow.set_entry_point(entry_node_id)

        # ------------------------------------------------------------------
        # Wire edges with handle-level routing.
        #
        # Handles:
        #   - IfElseNode / RouterNode: sourceHandle "true"/"false" or "route_<name>"
        #     route conditionally based on state["conditional_flags"][node_id].
        #   - Any node: sourceHandle "error" routes to a fallback branch when
        #     state["errors"][node_id] is set (error-handle fallback).
        #   - Everything else: unconditional add_edge.
        # ------------------------------------------------------------------
        # Group outgoing edges by (source, sourceHandle).
        from collections import defaultdict
        out_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            src = edge.get("source")
            if src in node_map:
                out_edges[src].append(edge)

        def _is_conditional_source(ntype: str) -> bool:
            return ntype in {"IfElseNode", "if_else", "RouterNode", "router"}

        for src, src_edges in out_edges.items():
            src_type = node_map.get(src, "AgentNode")

            # --- Conditional branching (IfElse / Router) ---
            if _is_conditional_source(src_type):
                # Map handle -> target node id
                handle_targets: Dict[str, str] = {}
                for e in src_edges:
                    handle = e.get("sourceHandle") or "out"
                    tgt = e.get("target")
                    if tgt in node_map:
                        handle_targets[handle] = tgt

                if not handle_targets:
                    continue

                def _make_conditional_path(node_id: str, targets: Dict[str, str]):
                    async def _path(state: GraphState) -> str:
                        flags = state.get("conditional_flags", {}) or {}
                        selected = flags.get(node_id)
                        # RouterNode stores a route name string; IfElse stores a bool.
                        if isinstance(selected, bool):
                            return "true" if selected else "false"
                        # Router: match route_<name> handle, fall back to default/out.
                        route_key = f"route_{selected}" if selected else "out"
                        if route_key in targets:
                            return route_key
                        if "default" in targets:
                            return "default"
                        return "out"
                    return _path

                path_fn = _make_conditional_path(src, handle_targets)
                workflow.add_conditional_edges(src, path_fn, handle_targets)
                continue

            # --- Error-handle fallback ---
            error_target = None
            normal_targets: List[str] = []
            for e in src_edges:
                handle = e.get("sourceHandle") or "out"
                tgt = e.get("target")
                if tgt not in node_map:
                    continue
                if handle == "error":
                    error_target = tgt
                else:
                    normal_targets.append(tgt)

            if error_target is not None:
                def _make_error_path(node_id: str):
                    async def _path(state: GraphState) -> str:
                        errors = state.get("errors", {}) or {}
                        if errors.get(node_id):
                            return "error"
                        return "ok"
                    return _path

                path_map: Dict[str, str] = {"error": error_target}
                if normal_targets:
                    path_map["ok"] = normal_targets[0]
                workflow.add_conditional_edges(src, _make_error_path(src), path_map)
                continue

            # --- Unconditional edges ---
            for tgt in normal_targets:
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

    async def validate_references(
        self,
        graph_json: Optional[Dict[str, Any]] = None,
        *,
        session: AsyncSession,
        source_hub_id: str,
    ) -> List[ValidationIssue]:
        """Validate qualified node references against database session and source hub link policies."""
        data = graph_json or self.graph_json
        nodes = data.get("nodes", []) if isinstance(data, dict) else []
        issues: List[ValidationIssue] = []

        for node in nodes:
            nid = node.get("id")
            ntype = node.get("type", "")
            data_cfg = node.get("data", {})
            req_type = NODE_REFERENCE_REQUIREMENTS.get(ntype)
            if not req_type:
                continue

            if ntype in {"MultiAgentNode", "multi_agent"}:
                refs_list = data_cfg.get("references")
                if not refs_list or not isinstance(refs_list, list):
                    issues.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code="MISSING_REFERENCE",
                            level="error",
                            message=f"Node '{nid}' of type '{ntype}' is missing required 'data.references' list.",
                            field="data.references",
                        )
                    )
                    continue

                for idx, ref_item in enumerate(refs_list):
                    try:
                        nr = NodeReference.model_validate(ref_item)
                    except Exception as err:
                        issues.append(
                            ValidationIssue(
                                node_id=nid,
                                node_type=ntype,
                                code="MALFORMED_REFERENCE",
                                level="error",
                                message=f"Node '{nid}' reference at index {idx} is malformed: {str(err)}",
                                field=f"data.references[{idx}]",
                                reference=ref_item if isinstance(ref_item, dict) else None,
                            )
                        )
                        continue

                    if nr.type != req_type:
                        issues.append(
                            ValidationIssue(
                                node_id=nid,
                                node_type=ntype,
                                code="REFERENCE_TYPE_MISMATCH",
                                level="error",
                                message=f"Node '{nid}' requires reference type '{req_type}' but got '{nr.type}'.",
                                field=f"data.references[{idx}].type",
                                reference=nr.model_dump(),
                            )
                        )
                        continue

                    await self._check_single_reference(
                        session=session,
                        source_hub_id=source_hub_id,
                        nid=nid,
                        ntype=ntype,
                        field=f"data.references[{idx}]",
                        nr=nr,
                        issues=issues,
                    )
            else:
                raw_ref = data_cfg.get("reference")
                if not raw_ref:
                    alias_val = (
                        data_cfg.get(f"{req_type}_id")
                        or data_cfg.get("agent_id")
                        or data_cfg.get("collection_id")
                        or data_cfg.get("mcp_tool_id")
                    )
                    hub_id = data_cfg.get("hub_id")
                    if alias_val and hub_id:
                        raw_ref = {"type": req_type, "hub_id": hub_id, "resource_id": alias_val}
                    else:
                        issues.append(
                            ValidationIssue(
                                node_id=nid,
                                node_type=ntype,
                                code="MISSING_REFERENCE",
                                level="error",
                                message=f"Node '{nid}' of type '{ntype}' is missing required 'data.reference'.",
                                field="data.reference",
                            )
                        )
                        continue

                try:
                    nr = NodeReference.model_validate(raw_ref)
                except Exception as err:
                    issues.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code="MALFORMED_REFERENCE",
                            level="error",
                            message=f"Node '{nid}' reference is malformed: {str(err)}",
                            field="data.reference",
                            reference=raw_ref if isinstance(raw_ref, dict) else None,
                        )
                    )
                    continue

                if nr.type != req_type:
                    issues.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code="REFERENCE_TYPE_MISMATCH",
                            level="error",
                            message=f"Node '{nid}' requires reference type '{req_type}' but got '{nr.type}'.",
                            field="data.reference.type",
                            reference=nr.model_dump(),
                        )
                    )
                    continue

                await self._check_single_reference(
                    session=session,
                    source_hub_id=source_hub_id,
                    nid=nid,
                    ntype=ntype,
                    field="data.reference",
                    nr=nr,
                    issues=issues,
                )

        # Validate agent-node tool bindings as cross-hub references.
        for node in nodes:
            nid = node.get("id")
            ntype = node.get("type", "")
            if ntype not in {"AgentNode", "agent", "MultiAgentNode", "multi_agent"}:
                continue
            data_cfg = node.get("data", {})
            tools = data_cfg.get("tools") or []
            if not isinstance(tools, list):
                continue
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    continue
                ttype = tool.get("type")
                field = f"data.tools[{idx}]"
                try:
                    if ttype == "retrieval":
                        nr = NodeReference(type="collection", hub_id=tool["hub_id"], resource_id=tool["collection_id"])
                    elif ttype == "mcp":
                        nr = NodeReference(type="mcp_tool", hub_id=tool.get("hub_id") or "", resource_id=tool["tool_name"])
                    elif ttype == "db":
                        nr = NodeReference(type="credential", hub_id=tool.get("hub_id") or "", resource_id=tool["credential_id"])
                    else:
                        continue
                except (KeyError, TypeError):
                    continue
                await self._check_single_reference(
                    session=session,
                    source_hub_id=source_hub_id,
                    nid=nid,
                    ntype=ntype,
                    field=field,
                    nr=nr,
                    issues=issues,
                )

        return issues

    async def _check_single_reference(
        self,
        session: AsyncSession,
        source_hub_id: str,
        nid: str,
        ntype: str,
        field: str,
        nr: NodeReference,
        issues: List[ValidationIssue],
    ) -> None:
        target_hub = await get_hub(session, nr.hub_id)
        if not target_hub:
            issues.append(
                ValidationIssue(
                    node_id=nid,
                    node_type=ntype,
                    code="REFERENCE_TARGET_MISSING",
                    level="error",
                    message=f"Target hub '{nr.hub_id}' for node '{nid}' was not found.",
                    field=field,
                    reference=nr.model_dump(),
                )
            )
            return

        if target_hub.is_archived:
            issues.append(
                ValidationIssue(
                    node_id=nid,
                    node_type=ntype,
                    code="HUB_ARCHIVED",
                    level="error",
                    message=f"Target hub '{nr.hub_id}' for node '{nid}' is archived.",
                    field=field,
                    reference=nr.model_dump(),
                )
            )
            return

        if nr.type == "mcp_tool":
            if source_hub_id != nr.hub_id:
                try:
                    await resolve_linked(
                        session,
                        source_hub_id=source_hub_id,
                        target_resource_type="agent",  # check hub direction
                        target_resource_id=nr.resource_id,
                    )
                except HubLinkError as err:
                    code = "HUB_LINK_REQUIRED" if err.code == "HUB_LINK_REQUIRED" else "HUB_LINK_REVOKED"
                    issues.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code=code,
                            level="error",
                            message=f"Workflow hub '{source_hub_id}' is not linked to hub '{nr.hub_id}'.",
                            field=field,
                            reference=nr.model_dump(),
                        )
                    )
            return

        try:
            res_obj = await resolve_linked(
                session,
                source_hub_id=source_hub_id,
                target_resource_type=nr.type,
                target_resource_id=nr.resource_id,
            )
        except HubLinkError as err:
            code = "HUB_LINK_REQUIRED"
            if err.code == "HUB_LINK_REVOKED":
                code = "HUB_LINK_REVOKED"
            msg = f"Workflow hub '{source_hub_id}' is not linked to target hub '{nr.hub_id}'."
            issues.append(
                ValidationIssue(
                    node_id=nid,
                    node_type=ntype,
                    code=code,
                    level="error",
                    message=msg,
                    field=field,
                    reference=nr.model_dump(),
                )
            )
            return
        except Exception as err:
            issues.append(
                ValidationIssue(
                    node_id=nid,
                    node_type=ntype,
                    code="REFERENCE_TARGET_MISSING",
                    level="error",
                    message=f"Referenced resource '{nr.resource_id}' for node '{nid}' was not found: {str(err)}",
                    field=field,
                    reference=nr.model_dump(),
                )
            )
            return

        res_hub_id = getattr(res_obj, "hub_id", None)
        if res_hub_id and nr.hub_id and nr.hub_id != res_hub_id and nr.hub_id != source_hub_id:
            issues.append(
                ValidationIssue(
                    node_id=nid,
                    node_type=ntype,
                    code="CROSS_HUB_REFERENCE_MISMATCH",
                    level="error",
                    message=f"Reference hub_id '{nr.hub_id}' does not match resolved resource's hub_id '{res_hub_id}'.",
                    field=field,
                    reference=nr.model_dump(),
                )
            )
            return

        res_status = getattr(res_obj, "status", None)
        res_is_active = getattr(res_obj, "is_active", True)
        if res_status == "archived" or res_is_active is False:
            issues.append(
                ValidationIssue(
                    node_id=nid,
                    node_type=ntype,
                    code="REFERENCE_INACTIVE",
                    level="error",
                    message=f"Referenced {nr.type} '{nr.resource_id}' for node '{nid}' is inactive or archived.",
                    field=field,
                    reference=nr.model_dump(),
                )
            )


def parse_graph_json_to_langgraph(graph_json: Dict[str, Any]) -> Any:
    """Convenience function to parse JSON graph and compile a LangGraph StateGraph."""
    parser = GraphParser(graph_json)
    return parser.build_langgraph()


async def validate_workflow_graph(
    session: AsyncSession,
    *,
    graph_json: Dict[str, Any],
    source_hub_id: str,
    strict: bool = False,
) -> ValidationResult:
    """Single validation entry point for workflow graph topology and qualified references."""
    parser = GraphParser(graph_json)
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []

    nodes = graph_json.get("nodes", []) if isinstance(graph_json, dict) else []
    edges = graph_json.get("edges", []) if isinstance(graph_json, dict) else []

    if not nodes:
        errors.append(
            ValidationIssue(
                code="EMPTY_GRAPH",
                level="error",
                message="Workflow graph must contain at least one node.",
            )
        )
    else:
        node_map = {node["id"]: node for node in nodes if "id" in node}
        node_ids = set(node_map.keys())

        # Reject removed standalone tool node types with an actionable message.
        for node in nodes:
            ntype = node.get("type", "AgentNode")
            if ntype in GraphParser.REMOVED_NODE_TYPES:
                errors.append(
                    ValidationIssue(
                        node_id=node.get("id"),
                        node_type=ntype,
                        code="REMOVED_NODE_TYPE",
                        level="error",
                        message=(
                            f"Node '{node.get('id')}' uses removed node type '{ntype}'. "
                            "Vector retrieval, MCP tools, and external database access are no longer standalone "
                            "nodes — attach them as tools to an Agent node instead (agent node 'tools' bindings)."
                        ),
                    )
                )

        # Validate agent-node tool bindings.
        for node in nodes:
            ntype = node.get("type", "AgentNode")
            if ntype not in {"AgentNode", "agent", "MultiAgentNode", "multi_agent"}:
                continue
            nid = node.get("id")
            tools = (node.get("data") or {}).get("tools") or []
            if not isinstance(tools, list):
                continue
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    errors.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code="MALFORMED_TOOL",
                            level="error",
                            message=f"Agent node '{nid}' tool at index {idx} is not an object.",
                            field=f"data.tools[{idx}]",
                        )
                    )
                    continue
                ttype = tool.get("type")
                if ttype not in {"retrieval", "mcp", "db", "web_search", "api_call"}:
                    errors.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code="UNKNOWN_TOOL_TYPE",
                            level="error",
                            message=(
                                f"Agent node '{nid}' tool at index {idx} has unknown type '{ttype}'. "
                                "Allowed tool types: retrieval, mcp, db, web_search, api_call."
                            ),
                            field=f"data.tools[{idx}].type",
                        )
                    )
                    continue
                if ttype == "retrieval" and not (tool.get("hub_id") and tool.get("collection_id")):
                    errors.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code="TOOL_MISSING_REF",
                            level="error",
                            message=f"Agent node '{nid}' retrieval tool requires hub_id and collection_id.",
                            field=f"data.tools[{idx}]",
                        )
                    )
                elif ttype == "mcp" and not (tool.get("server_id") and tool.get("tool_name")):
                    errors.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code="TOOL_MISSING_REF",
                            level="error",
                            message=f"Agent node '{nid}' mcp tool requires server_id and tool_name.",
                            field=f"data.tools[{idx}]",
                        )
                    )
                elif ttype == "db" and not tool.get("credential_id"):
                    errors.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code="TOOL_MISSING_REF",
                            level="error",
                            message=f"Agent node '{nid}' db tool requires credential_id.",
                            field=f"data.tools[{idx}]",
                        )
                    )
                elif ttype == "api_call" and not tool.get("url"):
                    errors.append(
                        ValidationIssue(
                            node_id=nid,
                            node_type=ntype,
                            code="TOOL_MISSING_REF",
                            level="error",
                            message=f"Agent node '{nid}' api_call tool requires url.",
                            field=f"data.tools[{idx}]",
                        )
                    )

        adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}

        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src not in node_ids or tgt not in node_ids:
                errors.append(
                    ValidationIssue(
                        code="DANGLING_EDGE",
                        level="error",
                        message=f"Edge references invalid node ID: source={src}, target={tgt}",
                    )
                )
            else:
                adj[src].append(tgt)
                in_degree[tgt] += 1

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
            errors.append(
                ValidationIssue(
                    code="CYCLE_DETECTED",
                    level="error",
                    message="Workflow graph contains an infinite cycle. Workflows must be acyclic.",
                )
            )

        all_sources = {e.get("source") for e in edges}
        leaf_ids = [nid for nid in node_ids if nid not in all_sources]
        for leaf_id in leaf_ids:
            leaf_node = node_map[leaf_id]
            ntype = leaf_node.get("type", "AgentNode")
            if ntype not in GraphParser.TERMINAL_NODE_TYPES:
                errors.append(
                    ValidationIssue(
                        node_id=leaf_id,
                        node_type=ntype,
                        code="NON_TERMINAL_LEAF",
                        level="error",
                        message=(
                            f"Rule 8 Violation: Path terminates at non-terminal node '{leaf_id}' of type '{ntype}'. "
                            "Every workflow path MUST conclude in an ActionNode, FinalMessageNode, or SynthesisNode."
                        ),
                    )
                )

    if source_hub_id and session is not None:
        ref_issues = await parser.validate_references(graph_json, session=session, source_hub_id=source_hub_id)
        errors.extend(ref_issues)

    is_valid = len(errors) == 0
    res = ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    if strict and not is_valid:
        err_msgs = "; ".join([e.message for e in errors])
        raise GraphValidationError(f"Workflow graph validation failed: {err_msgs}")

    return res
