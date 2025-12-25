"""Main entrypoint for CrewAI instrumentation."""

from __future__ import annotations

from typing import Literal

_instrumented = False


def instrument_crewai(
    mode: Literal["global", "minimal"] = "global",
    enable_tool_hooks: bool = True,
    enable_llm_hooks: bool = True,
    enable_event_listener: bool = True,
) -> None:
    """
    Instrument CrewAI for observability.

    Call this once at application startup, before creating any crews.

    Args:
        mode: Instrumentation mode:
            - "global": Full instrumentation (recommended)
            - "minimal": Only essential lifecycle events
        enable_tool_hooks: Capture tool call inputs/outputs
        enable_llm_hooks: Capture LLM prompts/responses
        enable_event_listener: Listen to crew/agent/task lifecycle events

    Example:
        from arzule_ingest.crewai import instrument_crewai
        from arzule_ingest import ArzuleRun
        from arzule_ingest.sinks import JsonlFileSink

        # Instrument CrewAI (call once at startup)
        instrument_crewai()

        # Then run your crew with an ArzuleRun context
        sink = JsonlFileSink("out/trace.jsonl")
        with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
            result = crew.kickoff(inputs={...})
    """
    global _instrumented
    if _instrumented:
        return

    if mode == "minimal":
        enable_tool_hooks = False
        enable_llm_hooks = False

    # Install event listener (must stay in memory)
    if enable_event_listener:
        from .listener import get_listener

        listener = get_listener()
        listener.setup_listeners()

    # Install tool hooks for input/output capture + handoff detection
    if enable_tool_hooks:
        from .hooks_tool import install_tool_hooks

        install_tool_hooks()

    # Install LLM hooks for prompt/response capture
    if enable_llm_hooks:
        from .hooks_llm import install_llm_hooks

        install_llm_hooks()

    _instrumented = True


def is_instrumented() -> bool:
    """Check if CrewAI has been instrumented."""
    return _instrumented


