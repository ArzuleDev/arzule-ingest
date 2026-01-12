"""Main entrypoint for LangGraph instrumentation."""

from __future__ import annotations

import sys
from typing import Literal, Optional

_instrumented = False
_handler_instance: Optional["ArzuleLangGraphHandler"] = None


def instrument_langgraph(
    mode: Literal["global", "minimal"] = "global",
    enable_graph_callbacks: bool = True,
    enable_node_callbacks: bool = True,
    enable_llm_callbacks: bool = True,
    enable_tool_callbacks: bool = True,
    enable_checkpoint_callbacks: bool = True,
) -> "ArzuleLangGraphHandler":
    """
    Instrument LangGraph for observability.

    Call this once at application startup, before creating any graphs.
    Returns the callback handler that can be passed to LangGraph components.

    Args:
        mode: Instrumentation mode:
            - "global": Full instrumentation (recommended)
            - "minimal": Only essential lifecycle events
        enable_graph_callbacks: Capture graph execution start/end events
        enable_node_callbacks: Capture node execution events
        enable_llm_callbacks: Capture LLM call events (from nested LangChain calls)
        enable_tool_callbacks: Capture tool invocation events
        enable_checkpoint_callbacks: Capture checkpoint/state persistence events

    Returns:
        ArzuleLangGraphHandler instance to pass to LangGraph components

    Example:
        from arzule_ingest.langgraph import instrument_langgraph
        from arzule_ingest import ArzuleRun
        from arzule_ingest.sinks import JsonlFileSink
        from langgraph.graph import StateGraph

        # Instrument LangGraph (call once at startup)
        handler = instrument_langgraph()

        # Use handler with LangGraph
        graph = StateGraph(State)
        graph.add_node("node_a", node_a_func)
        graph.add_edge("__start__", "node_a")
        compiled = graph.compile()

        # Run with ArzuleRun context
        sink = JsonlFileSink("out/trace.jsonl")
        with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
            result = compiled.invoke({"input": "..."}, config={"callbacks": [handler]})
    """
    global _instrumented, _handler_instance

    if _instrumented and _handler_instance is not None:
        return _handler_instance

    if mode == "minimal":
        enable_graph_callbacks = True
        enable_node_callbacks = True
        enable_llm_callbacks = True
        enable_tool_callbacks = False
        enable_checkpoint_callbacks = False

    # Create the callback handler
    from .callback_handler import ArzuleLangGraphHandler

    _handler_instance = ArzuleLangGraphHandler(
        enable_graph=enable_graph_callbacks,
        enable_node=enable_node_callbacks,
        enable_llm=enable_llm_callbacks,
        enable_tool=enable_tool_callbacks,
        enable_checkpoint=enable_checkpoint_callbacks,
    )

    _instrumented = True
    print("[arzule] LangGraph instrumentation installed", file=sys.stderr)

    return _handler_instance


def is_instrumented() -> bool:
    """Check if LangGraph has been instrumented."""
    return _instrumented


def get_handler() -> Optional["ArzuleLangGraphHandler"]:
    """Get the global handler instance, if instrumented."""
    return _handler_instance


def clear_handler_cache() -> None:
    """Clear the handler's cached run_id.
    
    Called by new_run() to prevent stale run_id from being used
    when callbacks arrive from background threads.
    """
    if _handler_instance is not None:
        _handler_instance._clear_cached_run_id()













