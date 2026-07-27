"""Unit tests for Rule 8 validation enforcement in GraphParser V5."""

import pytest
from projects.guardroute.src.core.graph_parser import GraphParser, GraphValidationError


def test_rule8_validation_valid_terminal_nodes():
    # Valid graph ending in ActionNode and FinalMessageNode
    graph_json = {
        "nodes": [
            {"id": "c1", "type": "ClassifierNode"},
            {"id": "ma1", "type": "MultiAgentNode", "data": {"agent_id": "coder"}},
            {"id": "act1", "type": "ActionNode", "data": {"action_type": "db_mutation"}},
            {"id": "msg1", "type": "FinalMessageNode", "data": {"model_id": "gemini-3.5-flash"}}
        ],
        "edges": [
            {"source": "c1", "target": "ma1"},
            {"source": "ma1", "target": "act1"},
            {"source": "ma1", "target": "msg1"}
        ]
    }
    parser = GraphParser(graph_json)
    assert parser.validate_graph() is True


def test_rule8_validation_dangling_non_terminal_rejection():
    # Graph ending in an AgentNode without a terminal node -> violates Rule 8
    graph_json = {
        "nodes": [
            {"id": "c1", "type": "ClassifierNode"},
            {"id": "a1", "type": "AgentNode", "data": {"model_id": "Arch-Router"}}
        ],
        "edges": [
            {"source": "c1", "target": "a1"}
        ]
    }
    parser = GraphParser(graph_json)
    with pytest.raises(GraphValidationError) as excinfo:
        parser.validate_graph()

    assert "Rule 8 Violation" in str(excinfo.value)
    assert "a1" in str(excinfo.value)


def test_rule8_validation_legacy_synthesis_allowed():
    # Legacy graph ending in SynthesisNode -> valid
    graph_json = {
        "nodes": [
            {"id": "c1", "type": "ClassifierNode"},
            {"id": "s1", "type": "SynthesisNode"}
        ],
        "edges": [
            {"source": "c1", "target": "s1"}
        ]
    }
    parser = GraphParser(graph_json)
    assert parser.validate_graph() is True
