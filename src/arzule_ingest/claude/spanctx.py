"""Span context management for Claude Code.

Manages the parent-child span relationships for tool calls and subagents.
Uses a stack-based approach to track nesting within a session.
"""

from __future__ import annotations

from typing import Optional

from .session import get_session_state, update_session_state
from ..ids import new_span_id


def push_span(
    session_id: str,
    *,
    tool_use_id: str,
    span_id: Optional[str] = None,
    subagent_type: Optional[str] = None,
) -> str:
    """
    Push a new span onto the context stack.

    Called when entering a new context (tool call or subagent).

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the call
        span_id: Optional explicit span ID (generated if not provided)
        subagent_type: Subagent type if this is a Task call

    Returns:
        The span ID for this context
    """
    if span_id is None:
        span_id = new_span_id()

    state = get_session_state(session_id)

    # Initialize span stack if needed
    if "span_stack" not in state:
        state["span_stack"] = []

    # Get current parent
    parent_span_id = state["span_stack"][-1]["span_id"] if state["span_stack"] else None

    # Push new span
    span_info = {
        "span_id": span_id,
        "tool_use_id": tool_use_id,
        "parent_span_id": parent_span_id,
        "subagent_type": subagent_type,
    }

    state["span_stack"].append(span_info)

    # Also track by tool_use_id for fast lookup
    if "spans_by_tool_use" not in state:
        state["spans_by_tool_use"] = {}
    state["spans_by_tool_use"][tool_use_id] = span_info

    update_session_state(session_id, {
        "span_stack": state["span_stack"],
        "spans_by_tool_use": state["spans_by_tool_use"],
    })

    return span_id


def pop_span(session_id: str, tool_use_id: Optional[str] = None) -> Optional[dict]:
    """
    Pop a span from the context stack.

    Called when exiting a context (tool call or subagent completes).

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Optional tool use ID to validate the pop

    Returns:
        The popped span info or None if stack is empty
    """
    state = get_session_state(session_id)
    span_stack = state.get("span_stack", [])

    if not span_stack:
        return None

    # If tool_use_id provided, verify it matches
    if tool_use_id is not None:
        # Find and remove the matching span
        for i in range(len(span_stack) - 1, -1, -1):
            if span_stack[i].get("tool_use_id") == tool_use_id:
                popped = span_stack.pop(i)
                # Remove from lookup dict
                spans_by_tool_use = state.get("spans_by_tool_use", {})
                spans_by_tool_use.pop(tool_use_id, None)
                update_session_state(session_id, {
                    "span_stack": span_stack,
                    "spans_by_tool_use": spans_by_tool_use,
                })
                return popped
        # Not found - return None
        return None

    # Pop the top
    popped = span_stack.pop()

    # Remove from lookup dict
    spans_by_tool_use = state.get("spans_by_tool_use", {})
    if popped.get("tool_use_id"):
        spans_by_tool_use.pop(popped["tool_use_id"], None)

    update_session_state(session_id, {
        "span_stack": span_stack,
        "spans_by_tool_use": spans_by_tool_use,
    })

    return popped


def get_current_span(session_id: str) -> Optional[dict]:
    """
    Get the current (topmost) span context.

    Args:
        session_id: Claude Code session identifier

    Returns:
        Current span info or None if no active spans
    """
    state = get_session_state(session_id)
    span_stack = state.get("span_stack", [])
    return span_stack[-1] if span_stack else None


def get_current_span_id(session_id: str) -> Optional[str]:
    """
    Get the current span ID.

    Args:
        session_id: Claude Code session identifier

    Returns:
        Current span ID or None
    """
    span = get_current_span(session_id)
    return span.get("span_id") if span else None


def get_current_parent_span_id(session_id: str) -> Optional[str]:
    """
    Get the parent span ID of the current context.

    Used for setting parent_span_id on new events.

    Args:
        session_id: Claude Code session identifier

    Returns:
        Parent span ID or None if at root
    """
    span = get_current_span(session_id)
    return span.get("parent_span_id") if span else None


def get_span_by_tool_use(session_id: str, tool_use_id: str) -> Optional[dict]:
    """
    Get span info by tool use ID.

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID to look up

    Returns:
        Span info or None
    """
    state = get_session_state(session_id)
    return state.get("spans_by_tool_use", {}).get(tool_use_id)


def get_span_depth(session_id: str) -> int:
    """
    Get the current span nesting depth.

    Args:
        session_id: Claude Code session identifier

    Returns:
        Depth (0 = at root)
    """
    state = get_session_state(session_id)
    return len(state.get("span_stack", []))


def is_in_subagent(session_id: str) -> bool:
    """
    Check if currently executing within a subagent.

    Args:
        session_id: Claude Code session identifier

    Returns:
        True if in a subagent context
    """
    span = get_current_span(session_id)
    return span is not None and span.get("subagent_type") is not None


def get_current_subagent_type(session_id: str) -> Optional[str]:
    """
    Get the current subagent type if in a subagent.

    Args:
        session_id: Claude Code session identifier

    Returns:
        Subagent type or None if not in subagent
    """
    span = get_current_span(session_id)
    return span.get("subagent_type") if span else None


def clear_spans(session_id: str) -> None:
    """
    Clear all span context for a session.

    Called on session end or reset.

    Args:
        session_id: Claude Code session identifier
    """
    update_session_state(session_id, {
        "span_stack": [],
        "spans_by_tool_use": {},
    })
