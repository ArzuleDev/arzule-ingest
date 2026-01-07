"""Main entrypoint for AutoGen v0.7+ instrumentation."""

from __future__ import annotations

import sys
from typing import Literal

_instrumented = False


def instrument_autogen_v2(
    mode: Literal["global", "minimal"] = "global",
    enable_message_hooks: bool = True,
    enable_llm_hooks: bool = True,
    enable_tool_hooks: bool = True,
    enable_agent_hooks: bool = True,
    enable_telemetry: bool = True,
) -> None:
    """
    Instrument AutoGen v0.7+ (autogen-core and autogen-agentchat) for observability.

    Call this once at application startup, before creating any agents.

    Args:
        mode: Instrumentation mode:
            - "global": Full instrumentation (recommended)
            - "minimal": Only essential lifecycle events
        enable_message_hooks: Capture agent message events
        enable_llm_hooks: Capture LLM calls (prompts/responses)
        enable_tool_hooks: Capture tool executions
        enable_agent_hooks: Capture agent lifecycle events
        enable_telemetry: Hook into autogen-core telemetry system

    Example:
        import arzule_ingest
        from arzule_ingest.autogen_v2 import instrument_autogen_v2

        # Initialize Arzule
        arzule_ingest.init()

        # Instrument AutoGen v0.7+
        instrument_autogen_v2()

        # Create and run agents
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        model_client = OpenAIChatCompletionClient(model="gpt-4")
        agent = AssistantAgent("assistant", model_client=model_client)
        await agent.run(task="Hello!")

    Note:
        This instrumentation is for AutoGen v0.7+ (autogen-core, autogen-agentchat).
        For legacy AutoGen v0.2 (pyautogen), use the `autogen` module instead.
    """
    global _instrumented
    if _instrumented:
        return

    if mode == "minimal":
        enable_llm_hooks = False
        enable_tool_hooks = False

    # Check if AutoGen v0.7+ is available
    try:
        import autogen_core  # noqa: F401
        import autogen_agentchat  # noqa: F401
    except ImportError:
        print(
            "[arzule] AutoGen v0.7+ not installed. Install with: pip install autogen-agentchat autogen-ext",
            file=sys.stderr,
        )
        return

    # Install hooks with the configured options
    from .hooks import install_hooks

    install_hooks(
        enable_message_hooks=enable_message_hooks,
        enable_llm_hooks=enable_llm_hooks,
        enable_tool_hooks=enable_tool_hooks,
        enable_agent_hooks=enable_agent_hooks,
        enable_telemetry=enable_telemetry,
    )

    _instrumented = True
    print(f"[arzule] AutoGen v0.7+ instrumentation installed (mode={mode})", file=sys.stderr)


def is_instrumented() -> bool:
    """Check if AutoGen v0.7+ has been instrumented."""
    return _instrumented


def uninstrument_autogen_v2() -> None:
    """
    Remove AutoGen v0.7+ instrumentation.

    This restores the original methods.
    Useful for testing or if you need to disable instrumentation.
    """
    global _instrumented
    if not _instrumented:
        return

    from .hooks import uninstall_hooks

    uninstall_hooks()
    _instrumented = False










