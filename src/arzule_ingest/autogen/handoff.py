"""Handoff detection for AutoGen agent conversations.

AutoGen uses a message-passing model where agents send messages to each other.
A "handoff" occurs when control transfers from one agent to another, which
happens implicitly through the conversation flow.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Optional

from ..ids import new_span_id

if TYPE_CHECKING:
    from ..run import ArzuleRun


def detect_handoff(
    sender_name: str,
    recipient_name: str,
    message: Any,
) -> Optional[str]:
    """
    Detect if a message represents a handoff between agents.

    In AutoGen, every message from one agent to another is effectively
    a handoff since control transfers to the recipient.

    Args:
        sender_name: Name of the sending agent
        recipient_name: Name of the receiving agent
        message: The message content

    Returns:
        A handoff key if this represents a handoff, None otherwise
    """
    # In AutoGen, any agent-to-agent message is a handoff
    # Generate a unique key for correlation
    if sender_name and recipient_name and sender_name != recipient_name:
        return str(uuid.uuid4())
    return None


def emit_handoff_event(
    run: "ArzuleRun",
    from_agent: str,
    to_agent: str,
    handoff_key: str,
    message_content: Optional[str] = None,
    span_id: Optional[str] = None,
) -> None:
    """
    Emit a handoff event when control transfers between agents.

    Args:
        run: The active ArzuleRun
        from_agent: The agent handing off
        to_agent: The agent receiving control
        handoff_key: Correlation key for this handoff
        message_content: Brief summary of the handoff message
        span_id: Optional span ID
    """
    # Store pending handoff for correlation
    run._handoff_pending[handoff_key] = {
        "from_agent": from_agent,
        "to_agent": to_agent,
        "proposed_at": run.now(),
    }

    summary = f"handoff from {from_agent} to {to_agent}"
    if message_content:
        summary += f": {message_content[:50]}..."

    run.emit({
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": span_id or new_span_id(),
        "parent_span_id": run.current_parent_span_id(),
        "seq": run.next_seq(),
        "ts": run.now(),
        "agent": {
            "id": f"autogen:agent:{from_agent}",
            "role": from_agent,
        },
        "event_type": "handoff.proposed",
        "status": "ok",
        "summary": summary,
        "attrs_compact": {
            "handoff_key": handoff_key,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "framework": "autogen",
        },
        "payload": {},
        "raw_ref": {"storage": "inline"},
    })


def emit_handoff_received(
    run: "ArzuleRun",
    handoff_key: str,
    agent_name: str,
    span_id: Optional[str] = None,
) -> None:
    """
    Emit event when an agent receives a handoff (starts processing).

    Args:
        run: The active ArzuleRun
        handoff_key: The handoff correlation key
        agent_name: The receiving agent's name
        span_id: Optional span ID
    """
    pending = run._handoff_pending.get(handoff_key, {})
    from_agent = pending.get("from_agent", "unknown")

    run.emit({
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": span_id or new_span_id(),
        "parent_span_id": run.current_parent_span_id(),
        "seq": run.next_seq(),
        "ts": run.now(),
        "agent": {
            "id": f"autogen:agent:{agent_name}",
            "role": agent_name,
        },
        "event_type": "handoff.ack",
        "status": "ok",
        "summary": f"handoff acknowledged by {agent_name}",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "from_agent": from_agent,
            "to_agent": agent_name,
            "framework": "autogen",
        },
        "payload": {},
        "raw_ref": {"storage": "inline"},
    })


def emit_handoff_complete(
    run: "ArzuleRun",
    handoff_key: str,
    agent_name: str,
    status: str = "ok",
    result_summary: Optional[str] = None,
    span_id: Optional[str] = None,
) -> None:
    """
    Emit event when a handoff completes (agent finishes processing).

    Args:
        run: The active ArzuleRun
        handoff_key: The handoff correlation key
        agent_name: The agent that completed
        status: Completion status
        result_summary: Brief summary of result
        span_id: Optional span ID
    """
    pending = run._handoff_pending.pop(handoff_key, {})
    from_agent = pending.get("from_agent", "unknown")

    run.emit({
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": span_id or new_span_id(),
        "parent_span_id": run.current_parent_span_id(),
        "seq": run.next_seq(),
        "ts": run.now(),
        "agent": {
            "id": f"autogen:agent:{agent_name}",
            "role": agent_name,
        },
        "event_type": "handoff.complete",
        "status": status,
        "summary": result_summary or f"handoff completed by {agent_name}",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "from_agent": from_agent,
            "to_agent": agent_name,
            "framework": "autogen",
        },
        "payload": {},
        "raw_ref": {"storage": "inline"},
    })

