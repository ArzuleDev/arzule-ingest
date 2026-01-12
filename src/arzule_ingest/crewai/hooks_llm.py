"""LLM call hooks for CrewAI instrumentation (CrewAI 1.7.x).

Note: In CrewAI 1.7.x, LLM events are emitted via the event bus.
The listener.py handles these events directly.
"""

from __future__ import annotations

import sys

_hooks_installed = False


def install_llm_hooks() -> None:
    """
    Install LLM hooks for CrewAI.

    In CrewAI 1.7.x, LLM events are already captured via the event bus
    in listener.py. This function is kept for backwards compatibility.
    """
    global _hooks_installed
    if _hooks_installed:
        return

    # In CrewAI 1.7.x, LLM events are handled by the event bus
    # The listener.py registers for LLMCallStartedEvent, LLMCallCompletedEvent, etc.
    # No additional monkey-patching needed for basic LLM tracing.

    _hooks_installed = True
    print("[arzule] LLM hooks installed (event bus mode)", file=sys.stderr)
