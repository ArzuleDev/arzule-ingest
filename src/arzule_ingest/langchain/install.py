"""Main entrypoint for LangChain instrumentation."""

from __future__ import annotations

import sys
from typing import Literal, Optional

_instrumented = False
_handler_instance: Optional["ArzuleLangChainHandler"] = None


def instrument_langchain(
    mode: Literal["global", "minimal"] = "global",
    enable_llm_callbacks: bool = True,
    enable_chain_callbacks: bool = True,
    enable_tool_callbacks: bool = True,
    enable_agent_callbacks: bool = True,
    enable_retriever_callbacks: bool = True,
) -> "ArzuleLangChainHandler":
    """
    Instrument LangChain for observability.

    Call this once at application startup, before creating any chains or agents.
    Returns the callback handler that can be passed to LangChain components.

    Args:
        mode: Instrumentation mode:
            - "global": Full instrumentation (recommended)
            - "minimal": Only essential lifecycle events
        enable_llm_callbacks: Capture LLM call start/end events
        enable_chain_callbacks: Capture chain execution events
        enable_tool_callbacks: Capture tool invocation events
        enable_agent_callbacks: Capture agent action/finish events
        enable_retriever_callbacks: Capture retriever events

    Returns:
        ArzuleLangChainHandler instance to pass to LangChain components

    Example:
        from arzule_ingest.langchain import instrument_langchain
        from arzule_ingest import ArzuleRun
        from arzule_ingest.sinks import JsonlFileSink

        # Instrument LangChain (call once at startup)
        handler = instrument_langchain()

        # Use handler with LangChain components
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(callbacks=[handler])

        # Or run with ArzuleRun context
        sink = JsonlFileSink("out/trace.jsonl")
        with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
            result = chain.invoke({"input": "..."}, config={"callbacks": [handler]})
    """
    global _instrumented, _handler_instance

    if _instrumented and _handler_instance is not None:
        return _handler_instance

    if mode == "minimal":
        enable_llm_callbacks = True
        enable_chain_callbacks = True
        enable_tool_callbacks = False
        enable_agent_callbacks = True
        enable_retriever_callbacks = False

    # Create the callback handler
    from .callback_handler import ArzuleLangChainHandler

    _handler_instance = ArzuleLangChainHandler(
        enable_llm=enable_llm_callbacks,
        enable_chain=enable_chain_callbacks,
        enable_tool=enable_tool_callbacks,
        enable_agent=enable_agent_callbacks,
        enable_retriever=enable_retriever_callbacks,
    )

    _instrumented = True
    print("[arzule] LangChain instrumentation installed", file=sys.stderr)

    return _handler_instance


def is_instrumented() -> bool:
    """Check if LangChain has been instrumented."""
    return _instrumented


def get_handler() -> Optional["ArzuleLangChainHandler"]:
    """Get the global handler instance, if instrumented."""
    return _handler_instance

