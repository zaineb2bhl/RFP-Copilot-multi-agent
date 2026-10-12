"""Wire the nodes into a LangGraph StateGraph with a checkpointer (needed for interrupt)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rfp_copilot.graph import nodes
from rfp_copilot.graph.state import RFPState
from rfp_copilot.tracing import Tracer

Node = Callable[[RFPState], dict[str, Any]]


def _traced(tracer: Tracer, name: str, node: Node) -> Node:
    """Wrap a node in a tracing span (not used for human_review: interrupt() raises)."""

    def wrapper(state: RFPState) -> dict[str, Any]:
        with tracer.span(name):
            return node(state)

    wrapper.__name__ = name
    return wrapper


def build_graph(deps: nodes.PipelineDeps, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Build the pipeline graph.

    ingest -> extract_requirements -> retrieve -> write -> review -(deterministic)->
    write (revise) | human_review (interrupt) -> export_docx | END (rejected)
    """
    settings = deps.settings
    graph = StateGraph(RFPState)
    traced_nodes: dict[str, Node] = {
        "ingest": nodes.ingest,
        "extract_requirements": nodes.make_extract_node(deps),
        "retrieve": nodes.make_retrieve_node(deps),
        "write": nodes.make_write_node(deps),
        "review": nodes.make_review_node(deps),
    }
    for name, node in traced_nodes.items():
        graph.add_node(name, _traced(deps.tracer, name, node))
    graph.add_node("human_review", nodes.human_review)
    graph.add_node("export_docx", _traced(deps.tracer, "export_docx", nodes.make_export_node(deps)))

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "extract_requirements")
    graph.add_edge("extract_requirements", "retrieve")
    graph.add_edge("retrieve", "write")
    graph.add_edge("write", "review")
    graph.add_conditional_edges(
        "review",
        partial(
            nodes.route_after_review,
            threshold=settings.review_threshold,
            max_iterations=settings.max_iterations,
        ),
        ["write", "human_review"],
    )
    graph.add_conditional_edges("human_review", nodes.route_after_human, ["export_docx", END])
    graph.add_edge("export_docx", END)
    return graph.compile(checkpointer=checkpointer)
