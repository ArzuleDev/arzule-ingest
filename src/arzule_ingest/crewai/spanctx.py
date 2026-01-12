"""Span context management using contextvars.

Supports both sequential and concurrent task execution modes.
For concurrent execution, use the task-based span functions.
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

from ..ids import new_span_id

if TYPE_CHECKING:
    from ..run import ArzuleRun


# Context variable for the current span stack (legacy sequential mode)
_span_stack: ContextVar[list[str]] = ContextVar("_span_stack", default=[])


def get_current_span_id() -> Optional[str]:
    """Get the current span ID from the stack (legacy sequential mode)."""
    stack = _span_stack.get()
    return stack[-1] if stack else None


def get_parent_span_id() -> Optional[str]:
    """Get the parent span ID (second to last in stack)."""
    stack = _span_stack.get()
    return stack[-2] if len(stack) > 1 else None


def start_child_span(run: "ArzuleRun", kind: str, name: str) -> str:
    """
    Start a new child span (legacy sequential mode).

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
    End the current span (legacy sequential mode).

    Args:
        run: The active ArzuleRun
        span_id: Optional span ID to verify (for debugging)

    Returns:
        The ended span ID
    """
    popped = run.pop_span()
    if span_id and popped and popped != span_id:
        print(
            f"[arzule] Span mismatch: expected {span_id}, got {popped}",
            file=sys.stderr,
        )
    return popped


# =============================================================================
# Task-Based Span Management (for concurrent execution)
# =============================================================================


def start_task_child_span(run: "ArzuleRun", task_key: str, kind: str, name: str) -> str:
    """
    Start a child span within a task's span tree (for concurrent mode).

    Args:
        run: The active ArzuleRun
        task_key: The task identifier
        kind: Span kind (e.g., "tool", "llm")
        name: Human-readable span name

    Returns:
        The new span ID
    """
    span_id = new_span_id()
    run.push_task_span(task_key, span_id)
    return span_id


def end_task_child_span(run: "ArzuleRun", task_key: str) -> Optional[str]:
    """
    End a child span within a task's span tree.

    Args:
        run: The active ArzuleRun
        task_key: The task identifier

    Returns:
        The ended span ID
    """
    return run.pop_task_span(task_key)


def get_task_span_id(run: "ArzuleRun", task_key: Optional[str] = None, agent_key: Optional[str] = None) -> Optional[str]:
    """
    Get the current span ID for a task.

    Args:
        run: The active ArzuleRun
        task_key: The task identifier
        agent_key: Alternative - look up by agent key

    Returns:
        The current span ID for the task, or None
    """
    return run.get_task_parent_span(task_key=task_key, agent_key=agent_key)





