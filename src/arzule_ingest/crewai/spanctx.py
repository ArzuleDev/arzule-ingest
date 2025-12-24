"""Span context management using contextvars."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

from ..ids import new_span_id

if TYPE_CHECKING:
    from ..run import ArzuleRun


# Context variable for the current span stack
_span_stack: ContextVar[list[str]] = ContextVar("_span_stack", default=[])


def get_current_span_id() -> Optional[str]:
    """Get the current span ID from the stack."""
    stack = _span_stack.get()
    return stack[-1] if stack else None


def get_parent_span_id() -> Optional[str]:
    """Get the parent span ID (second to last in stack)."""
    stack = _span_stack.get()
    return stack[-2] if len(stack) > 1 else None


def start_child_span(run: "ArzuleRun", kind: str, name: str) -> str:
    """
    Start a new child span.

    Args:
        run: The active ArzuleRun
        kind: Span kind (e.g., "tool", "llm", "task")
        name: Human-readable span name

    Returns:
        The new span ID
    """
    span_id = new_span_id()
    run.push_span(span_id)
    return span_id


def end_span(run: "ArzuleRun", span_id: Optional[str] = None) -> Optional[str]:
    """
    End the current span.

    Args:
        run: The active ArzuleRun
        span_id: Optional span ID to verify (for debugging)

    Returns:
        The ended span ID
    """
    popped = run.pop_span()
    if span_id and popped and popped != span_id:
        import sys

        print(
            f"[arzule] Span mismatch: expected {span_id}, got {popped}",
            file=sys.stderr,
        )
    return popped

