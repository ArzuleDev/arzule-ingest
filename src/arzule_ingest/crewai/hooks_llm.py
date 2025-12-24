"""LLM call hooks for CrewAI instrumentation."""

from __future__ import annotations

from typing import Any, Optional

from ..run import current_run
from .normalize import evt_llm_end, evt_llm_start
from .spanctx import end_span, start_child_span

_hooks_installed = False


def install_llm_hooks() -> None:
    """
    Install before/after LLM call hooks for CrewAI.

    These hooks capture LLM prompts and responses.
    """
    global _hooks_installed
    if _hooks_installed:
        return

    try:
        from crewai.utilities.events.llm_event_listener import LLMEventListener
    except ImportError:
        # CrewAI not installed or version mismatch
        import sys

        print("[arzule] CrewAI LLM hooks not available", file=sys.stderr)
        return

    # Store original methods
    _original_on_llm_started = LLMEventListener.on_llm_started
    _original_on_llm_ended = LLMEventListener.on_llm_ended

    def _before_llm_call(self: Any, event: Any) -> None:
        """Hook called before LLM execution."""
        run = current_run()
        if run:
            context = _LLMContext(event)

            # Start an LLM span
            span_id = start_child_span(run, kind="llm", name="llm_call")

            # Store span_id for correlation
            llm_key = f"llm:{id(event)}"
            run._inflight[llm_key] = span_id

            # Emit start event
            run.emit(evt_llm_start(run, context, span_id))

        # Call original
        return _original_on_llm_started(self, event)

    def _after_llm_call(self: Any, event: Any) -> None:
        """Hook called after LLM execution."""
        run = current_run()
        if run:
            context = _LLMContext(event)

            # Retrieve span_id
            llm_key = f"llm:{id(event)}"
            span_id = run._inflight.pop(llm_key, None)

            # Emit end event
            run.emit(evt_llm_end(run, context, span_id))

            # End span
            if span_id:
                end_span(run, span_id)

        # Call original
        return _original_on_llm_ended(self, event)

    # Monkey-patch the listener
    LLMEventListener.on_llm_started = _before_llm_call
    LLMEventListener.on_llm_ended = _after_llm_call

    _hooks_installed = True


class _LLMContext:
    """Adapter to normalize CrewAI LLM events to a consistent interface."""

    def __init__(self, event: Any) -> None:
        self._event = event

    @property
    def messages(self) -> list[Any]:
        """Get the messages sent to the LLM."""
        for attr in ("messages", "prompt", "prompts"):
            val = getattr(self._event, attr, None)
            if val is not None:
                if isinstance(val, list):
                    return val
                return [val]
        return []

    @property
    def response(self) -> Any:
        """Get the LLM response."""
        for attr in ("response", "result", "output", "completion"):
            val = getattr(self._event, attr, None)
            if val is not None:
                return val
        return None

    @property
    def agent(self) -> Any:
        """Get the agent."""
        return getattr(self._event, "agent", None)

    @property
    def task(self) -> Any:
        """Get the task."""
        return getattr(self._event, "task", None)

    @property
    def error(self) -> Any:
        """Get any error."""
        return getattr(self._event, "error", None) or getattr(self._event, "exception", None)

