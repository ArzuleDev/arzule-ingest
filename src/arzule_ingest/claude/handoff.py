"""Handoff lifecycle management for Claude Code.

Manages the correlation of handoff events (proposed → ack → complete)
using deterministic keys derived from session and tool use IDs.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from .session import get_session_state, update_session_state


def generate_handoff_key(session_id: str, tool_use_id: str) -> str:
    """
    Generate deterministic handoff key from session and tool use.

    The handoff key is used to correlate:
    - handoff.proposed (main agent delegates)
    - handoff.ack (subagent starts)
    - handoff.complete (subagent returns)

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID from the Task call

    Returns:
        32-character hex string for handoff correlation
    """
    combined = f"{session_id}:{tool_use_id}"
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def create_handoff(
    session_id: str,
    tool_use_id: str,
    subagent_type: str,
    description: str,
) -> str:
    """
    Create and register a new handoff.

    Called when PreToolUse fires for a Task tool call.

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the Task call
        subagent_type: Type of subagent being delegated to
        description: Task description

    Returns:
        Handoff key for correlation
    """
    handoff_key = generate_handoff_key(session_id, tool_use_id)

    # Store handoff in session state
    state = get_session_state(session_id)
    if "handoffs" not in state:
        state["handoffs"] = {}

    state["handoffs"][handoff_key] = {
        "tool_use_id": tool_use_id,
        "subagent_type": subagent_type,
        "description": description,
        "status": "proposed",
        "proposed_at": _now_iso(),
    }

    update_session_state(session_id, {"handoffs": state["handoffs"]})

    return handoff_key


def ack_handoff(session_id: str, tool_use_id: str) -> Optional[str]:
    """
    Acknowledge a handoff (subagent starts work).

    Called when the subagent session starts.

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the Task call

    Returns:
        Handoff key if found, None otherwise
    """
    handoff_key = generate_handoff_key(session_id, tool_use_id)

    state = get_session_state(session_id)
    handoffs = state.get("handoffs", {})

    if handoff_key in handoffs:
        handoffs[handoff_key]["status"] = "acknowledged"
        handoffs[handoff_key]["acked_at"] = _now_iso()
        update_session_state(session_id, {"handoffs": handoffs})
        return handoff_key

    return None


def complete_handoff(session_id: str, tool_use_id: str) -> Optional[str]:
    """
    Complete a handoff (subagent returns result).

    Called when PostToolUse fires for a Task tool call.

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the Task call

    Returns:
        Handoff key if found, None otherwise
    """
    handoff_key = generate_handoff_key(session_id, tool_use_id)

    state = get_session_state(session_id)
    handoffs = state.get("handoffs", {})

    if handoff_key in handoffs:
        handoffs[handoff_key]["status"] = "completed"
        handoffs[handoff_key]["completed_at"] = _now_iso()
        update_session_state(session_id, {"handoffs": handoffs})
        return handoff_key

    return None


def fail_handoff(session_id: str, tool_use_id: str, error: str) -> Optional[str]:
    """
    Mark a handoff as failed.

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the Task call
        error: Error message

    Returns:
        Handoff key if found, None otherwise
    """
    handoff_key = generate_handoff_key(session_id, tool_use_id)

    state = get_session_state(session_id)
    handoffs = state.get("handoffs", {})

    if handoff_key in handoffs:
        handoffs[handoff_key]["status"] = "failed"
        handoffs[handoff_key]["failed_at"] = _now_iso()
        handoffs[handoff_key]["error"] = error
        update_session_state(session_id, {"handoffs": handoffs})
        return handoff_key

    return None


def get_handoff(session_id: str, tool_use_id: str) -> Optional[dict]:
    """
    Get handoff info by tool_use_id.

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the Task call

    Returns:
        Handoff info dict or None
    """
    handoff_key = generate_handoff_key(session_id, tool_use_id)
    state = get_session_state(session_id)
    return state.get("handoffs", {}).get(handoff_key)


def get_active_handoffs(session_id: str) -> list[dict]:
    """
    Get all active (non-completed) handoffs for a session.

    Args:
        session_id: Claude Code session identifier

    Returns:
        List of active handoff info dicts
    """
    state = get_session_state(session_id)
    handoffs = state.get("handoffs", {})

    return [
        {"key": key, **info}
        for key, info in handoffs.items()
        if info.get("status") not in ("completed", "failed")
    ]


def get_handoff_duration(session_id: str, tool_use_id: str) -> Optional[float]:
    """
    Get the duration of a completed handoff in seconds.

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the Task call

    Returns:
        Duration in seconds or None if not completed
    """
    handoff = get_handoff(session_id, tool_use_id)
    if not handoff:
        return None

    proposed_at = handoff.get("proposed_at")
    completed_at = handoff.get("completed_at") or handoff.get("failed_at")

    if not proposed_at or not completed_at:
        return None

    try:
        from datetime import datetime

        start = datetime.fromisoformat(proposed_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        return (end - start).total_seconds()
    except Exception:
        return None


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
