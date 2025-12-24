"""Tool call hooks for CrewAI instrumentation."""

from __future__ import annotations

from typing import Any, Optional

from ..run import current_run
from .handoff import maybe_emit_handoff_proposed, maybe_inject_handoff_key
from .normalize import evt_tool_end, evt_tool_start
from .spanctx import end_span, start_child_span

_hooks_installed = False


def install_tool_hooks() -> None:
    """
    Install before/after tool call hooks for CrewAI.

    These hooks capture tool inputs and outputs, and detect delegation calls
    for handoff tracking.
    """
    global _hooks_installed
    if _hooks_installed:
        return

    try:
        from crewai.utilities.events.tool_event_listener import ToolEventListener
    except ImportError:
        # CrewAI not installed or version mismatch
        import sys

        print("[arzule] CrewAI tool hooks not available", file=sys.stderr)
        return

    # Store original methods
    _original_on_tool_started = ToolEventListener.on_tool_started
    _original_on_tool_ended = ToolEventListener.on_tool_ended

    def _before_tool_call(self: Any, event: Any) -> None:
        """Hook called before tool execution."""
        run = current_run()
        if run:
            # Build a context-like object from the event
            context = _ToolContext(event)

            # 1) Detect delegation-like tool calls and inject correlation key
            maybe_inject_handoff_key(run, context)

            # 2) Start a tool span
            span_id = start_child_span(run, kind="tool", name=context.tool_name or "unknown")

            # Store span_id for correlation with after hook
            tool_key = f"tool:{id(event)}"
            run._inflight[tool_key] = span_id

            # 3) Emit start event
            run.emit(evt_tool_start(run, context, span_id))

        # Call original
        return _original_on_tool_started(self, event)

    def _after_tool_call(self: Any, event: Any) -> None:
        """Hook called after tool execution."""
        run = current_run()
        if run:
            context = _ToolContext(event)

            # Retrieve span_id
            tool_key = f"tool:{id(event)}"
            span_id = run._inflight.pop(tool_key, None)

            # Emit end event
            run.emit(evt_tool_end(run, context, span_id))

            # Emit handoff.proposed if this was a delegation call
            maybe_emit_handoff_proposed(run, context, span_id)

            # End span
            if span_id:
                end_span(run, span_id)

        # Call original
        return _original_on_tool_ended(self, event)

    # Monkey-patch the listener
    ToolEventListener.on_tool_started = _before_tool_call
    ToolEventListener.on_tool_ended = _after_tool_call

    _hooks_installed = True


class _ToolContext:
    """Adapter to normalize CrewAI tool events to a consistent interface."""

    def __init__(self, event: Any) -> None:
        self._event = event

    @property
    def tool_name(self) -> Optional[str]:
        """Get the tool name."""
        # Try different attribute names used by CrewAI
        for attr in ("tool_name", "name", "tool"):
            val = getattr(self._event, attr, None)
            if val:
                if isinstance(val, str):
                    return val
                # Might be a tool object
                return getattr(val, "name", str(val))
        return None

    @property
    def tool_input(self) -> dict[str, Any]:
        """Get the tool input (mutable)."""
        for attr in ("tool_input", "input", "arguments", "args"):
            val = getattr(self._event, attr, None)
            if isinstance(val, dict):
                return val
        return {}

    @property
    def tool_result(self) -> Any:
        """Get the tool result."""
        for attr in ("tool_result", "result", "output", "tool_output"):
            val = getattr(self._event, attr, None)
            if val is not None:
                return val
        return None

    @property
    def agent(self) -> Any:
        """Get the agent."""
        return getattr(self._event, "agent", None)

    @property
    def error(self) -> Any:
        """Get any error."""
        return getattr(self._event, "error", None) or getattr(self._event, "exception", None)

    @property
    def tool(self) -> Any:
        """Get the tool object for ID purposes."""
        return getattr(self._event, "tool", self._event)

