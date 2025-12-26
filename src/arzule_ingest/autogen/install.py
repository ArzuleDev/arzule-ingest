"""Main entrypoint for Microsoft AutoGen instrumentation."""

from __future__ import annotations

import sys
from typing import Literal

_instrumented = False


def instrument_autogen(
    mode: Literal["global", "minimal"] = "global",
    enable_message_hooks: bool = True,
    enable_llm_hooks: bool = True,
    enable_tool_hooks: bool = True,
    enable_code_execution_hooks: bool = True,
    enable_conversation_hooks: bool = True,
) -> None:
    """
    Instrument Microsoft AutoGen for observability.

    Call this once at application startup, before creating any agents.

    Args:
        mode: Instrumentation mode:
            - "global": Full instrumentation (recommended)
            - "minimal": Only essential lifecycle events
        enable_message_hooks: Capture agent message send/receive events
        enable_llm_hooks: Capture LLM calls (prompts/responses)
        enable_tool_hooks: Capture tool/function executions
        enable_code_execution_hooks: Capture code block executions
        enable_conversation_hooks: Capture conversation start/end events

    Example:
        from arzule_ingest.autogen import instrument_autogen
        from arzule_ingest import ArzuleRun
        from arzule_ingest.sinks import JsonlFileSink

        # Instrument AutoGen (call once at startup)
        instrument_autogen()

        # Then run your agents with an ArzuleRun context
        sink = JsonlFileSink("out/trace.jsonl")
        with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
            user_proxy.initiate_chat(assistant, message="Hello!")

    Note:
        AutoGen supports both sync and async methods. This instrumentation
        covers both variants (send/a_send, receive/a_receive, etc.).
    """
    global _instrumented
    if _instrumented:
        return

    if mode == "minimal":
        enable_llm_hooks = False
        enable_tool_hooks = False
        enable_code_execution_hooks = False

    # Check if AutoGen is available
    try:
        import autogen  # noqa: F401
    except ImportError:
        print(
            "[arzule] AutoGen not installed. Install with: pip install pyautogen",
            file=sys.stderr,
        )
        return

    # Install hooks with the configured options
    from .hooks import install_hooks

    install_hooks(
        enable_message_hooks=enable_message_hooks,
        enable_llm_hooks=enable_llm_hooks,
        enable_tool_hooks=enable_tool_hooks,
        enable_code_execution_hooks=enable_code_execution_hooks,
        enable_conversation_hooks=enable_conversation_hooks,
    )

    _instrumented = True
    print(f"[arzule] AutoGen instrumentation installed (mode={mode})", file=sys.stderr)


def is_instrumented() -> bool:
    """Check if AutoGen has been instrumented."""
    return _instrumented


def uninstrument_autogen() -> None:
    """
    Remove AutoGen instrumentation.

    This restores the original methods on ConversableAgent.
    Useful for testing or if you need to disable instrumentation.
    """
    global _instrumented
    if not _instrumented:
        return

    from .hooks import uninstall_hooks

    uninstall_hooks()
    _instrumented = False

