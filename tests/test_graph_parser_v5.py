"""Integration tests for GuardRoute GraphParser V5."""

import pytest
from projects.guardroute.src.core.graph_parser import GraphParser, GraphValidationError


def test_validate_graph_valid_dag():
    graph_json = {
        "nodes": [
            {"id": "node_1", "type": "ClassifierNode"},
            {"id": "node_2", "type": "IfElseNode"},
            {"id": "node_3", "type": "SynthesisNode"}
        ],
        "edges": [
            {"source": "node_1", "target": "node_2"},
            {"source": "node_2", "target": "node_3"}
        ]
    }
    parser = GraphParser(graph_json)
    assert parser.validate_graph() is True


def test_validate_graph_cycle_rejection():
    graph_json = {
        "nodes": [
            {"id": "a", "type": "AgentNode"},
            {"id": "b", "type": "TransformNode"}
        ],
        "edges": [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"}
        ]
    }
    parser = GraphParser(graph_json)
    with pytest.raises(GraphValidationError):
        parser.validate_graph()


def test_build_langgraph_v5_nodes():
    graph_json = {
        "nodes": [
            {"id": "c1", "type": "ClassifierNode"},
            {"id": "if1", "type": "IfElseNode", "data": {"type": "complexity_equals", "value": "HIGH"}},
            {"id": "wh1", "type": "WebhookNode", "data": {"url": "https://httpbin.org/post"}},
            {"id": "tr1", "type": "TransformNode", "data": {"mode": "template", "template": "{{prompt}}"}},
            {"id": "s1", "type": "SynthesisNode"}
        ],
        "edges": [
            {"source": "c1", "target": "if1"},
            {"source": "if1", "target": "wh1"},
            {"source": "wh1", "target": "tr1"},
            {"source": "tr1", "target": "s1"}
        ]
    }
    parser = GraphParser(graph_json)
    compiled = parser.build_langgraph()
    assert compiled is not None
