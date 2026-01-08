"""Integration for AutoGen v0.7+ (autogen-core and autogen-agentchat).

This module provides instrumentation for the new AutoGen architecture:
- autogen-core: Low-level agent runtime
- autogen-agentchat: High-level agents API (AssistantAgent, etc.)

For legacy AutoGen v0.2 (pyautogen), use the `autogen` module instead.
"""

from .install import instrument_autogen_v2, is_instrumented, uninstrument_autogen_v2

__all__ = [
    "instrument_autogen_v2",
    "is_instrumented",
    "uninstrument_autogen_v2",
]












