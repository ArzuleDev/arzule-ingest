"""Tool call hooks for CrewAI instrumentation (CrewAI 1.7.x).

Note: In CrewAI 1.7.x, tool events are emitted via the event bus.
The listener.py handles these events directly. This module provides
additional handoff detection functionality that wraps tool execution.
"""

from __future__ import annotations

import sys

_hooks_installed = False


def install_tool_hooks() -> None:
    """
    Install tool hooks for handoff detection.

    In CrewAI 1.7.x, tool events are already captured via the event bus
    in listener.py. This function is kept for backwards compatibility
    and for any additional tool-level instrumentation needed.
    """
    global _hooks_installed
    if _hooks_installed:
        return

    # In CrewAI 1.7.x, tool events are handled by the event bus
    # The listener.py registers for ToolUsageStartedEvent, ToolUsageFinishedEvent, etc.
    # No additional monkey-patching needed for basic tool tracing.

    # If we need handoff detection at the tool level, we could hook into
    # the delegation tool execution here, but the current approach
    # of detecting handoffs via task descriptions is sufficient.

    _hooks_installed = True
    print("[arzule] Tool hooks installed (event bus mode)", file=sys.stderr)
