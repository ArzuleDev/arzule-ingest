"""Handoff detection for LangChain/LangGraph agent delegation."""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..run import ArzuleRun

# Pattern to extract handoff keys from text
HANDOFF_RE = re.compile(r"\[arzule_handoff:([0-9a-f-]{36})\]")

# Patterns that indicate agent-to-agent delegation in LangGraph
# These cover common multi-agent patterns across frameworks
HANDOFF_TOOL_PATTERNS = [
    # Standard delegation verbs
    "delegate",
    "handoff",
    "hand_off",
    "handover",
    "hand_over",
    "transfer",
    # Routing patterns
    "route_to",
    "send_to",
    "forward_to",
    "pass_to",
    # Agent interaction patterns
    "coworker",
    "co_worker",
    "colleague",
    "teammate",
    # Role-based patterns
    "specialist",
    "expert",
    "assistant",
    # Consultation patterns
    "consult",
    "ask_",
    "query_",
    # Assignment patterns
    "assign_to",
    "dispatch_to",
    # Call patterns
    "call_agent",
    "invoke_agent",
]


def is_handoff_tool(tool_name: Optional[str]) -> bool:
    """
    Check if a tool name indicates an agent handoff.

    In LangGraph, common patterns include:
    - transfer_to_<agent_name>
    - handoff_to_<agent_name>
    - Tools that route to other agents
    """
    if not tool_name:
        return False

    name_lower = tool_name.lower()
    for pattern in HANDOFF_TOOL_PATTERNS:
        if pattern in name_lower:
            return True

    return False


def extract_target_agent_from_tool(tool_name: str) -> Optional[str]:
    """
    Extract target agent name from handoff tool name.

    Examples:
        transfer_to_researcher -> researcher
        handoff_to_writer -> writer
        delegate_to_specialist -> specialist
        consult_expert -> expert
    """
    # Patterns with _to_ suffix that commonly have agent names after
    to_patterns = [
        "transfer_to_",
        "handoff_to_",
        "hand_off_to_",
        "handover_to_",
        "hand_over_to_",
        "delegate_to_",
        "route_to_",
        "send_to_",
        "forward_to_",
        "pass_to_",
        "assign_to_",
        "dispatch_to_",
    ]
    
    name_lower = tool_name.lower()
    for pattern in to_patterns:
        if name_lower.startswith(pattern):
            return tool_name[len(pattern):]
    
    # Patterns with _ suffix (e.g., consult_expert)
    action_patterns = [
        "consult_",
        "ask_",
        "query_",
        "call_",
        "invoke_",
    ]
    
    for pattern in action_patterns:
        if name_lower.startswith(pattern):
            return tool_name[len(pattern):]

    return None


def detect_handoff_from_agent_action(
    run: "ArzuleRun",
    action: Any,
    span_id: Optional[str],
) -> Optional[str]:
    """
    Detect and emit handoff.proposed event from agent action.

    In LangGraph, agent handoffs typically occur through:
    1. Tool calls like transfer_to_<agent>
    2. State updates that trigger routing
    3. Explicit handoff tools

    Args:
        run: The active ArzuleRun
        action: The AgentAction object
        span_id: Current span ID

    Returns:
        Handoff key if a handoff was detected, None otherwise
    """
    tool = getattr(action, "tool", None)
    if not tool or not is_handoff_tool(tool):
        return None

    tool_input = getattr(action, "tool_input", {})

    # Generate handoff key
    handoff_key = str(uuid.uuid4())

    # Extract target agent
    target_agent = extract_target_agent_from_tool(tool)
    if not target_agent and isinstance(tool_input, dict):
        target_agent = tool_input.get("agent") or tool_input.get("to") or tool_input.get("target")

    # Store pending handoff
    run._handoff_pending[handoff_key] = {
        "from_role": "agent",  # LangChain doesn't always have agent roles
        "from_agent_id": "langchain:agent",
        "to_agent": target_agent,
        "tool_name": tool,
        "proposed_at": run.now(),
    }

    # Emit handoff.proposed event
    run.emit({
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": span_id,
        "parent_span_id": run.current_parent_span_id(),
        "seq": run.next_seq(),
        "ts": run.now(),
        "agent": {"id": "langchain:agent", "role": "agent"},
        "event_type": "handoff.proposed",
        "status": "ok",
        "summary": f"handoff proposed to {target_agent or 'agent'}",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "from_agent_role": "agent",
            "to_agent": target_agent,
            "tool_name": tool,
        },
        "payload": {
            "tool_input": tool_input if isinstance(tool_input, dict) else {"raw": str(tool_input)},
        },
        "raw_ref": {"storage": "inline"},
    })

    return handoff_key


def extract_handoff_key_from_text(text: Optional[str]) -> Optional[str]:
    """
    Extract a handoff key from text (e.g., in messages or state).

    Args:
        text: Text to search for handoff marker

    Returns:
        The handoff key if found, None otherwise
    """
    if not text:
        return None
    match = HANDOFF_RE.search(text)
    return match.group(1) if match else None


def emit_handoff_ack(
    run: "ArzuleRun",
    handoff_key: str,
    agent_name: Optional[str] = None,
    span_id: Optional[str] = None,
) -> None:
    """
    Emit handoff.ack event when receiving agent starts processing.

    Args:
        run: The active ArzuleRun
        handoff_key: The handoff correlation key
        agent_name: The receiving agent's name
        span_id: Current span ID
    """
    pending = run._handoff_pending.get(handoff_key, {})

    run.emit({
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": span_id,
        "parent_span_id": run.current_parent_span_id(),
        "seq": run.next_seq(),
        "ts": run.now(),
        "agent": {
            "id": f"langchain:agent:{agent_name}" if agent_name else "langchain:agent",
            "role": agent_name or "agent",
        },
        "event_type": "handoff.ack",
        "status": "ok",
        "summary": f"handoff acknowledged by {agent_name or 'agent'}",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "from_agent_role": pending.get("from_role"),
            "to_agent": agent_name,
        },
        "payload": {},
        "raw_ref": {"storage": "inline"},
    })


def emit_handoff_complete(
    run: "ArzuleRun",
    handoff_key: str,
    agent_name: Optional[str] = None,
    span_id: Optional[str] = None,
    status: str = "ok",
    result_summary: Optional[str] = None,
) -> None:
    """
    Emit handoff.complete event when handoff task finishes.

    Args:
        run: The active ArzuleRun
        handoff_key: The handoff correlation key
        agent_name: The agent's name
        span_id: Current span ID
        status: Completion status
        result_summary: Brief summary of result
    """
    pending = run._handoff_pending.pop(handoff_key, {})

    run.emit({
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": span_id,
        "parent_span_id": run.current_parent_span_id(),
        "seq": run.next_seq(),
        "ts": run.now(),
        "agent": {
            "id": f"langchain:agent:{agent_name}" if agent_name else "langchain:agent",
            "role": agent_name or "agent",
        },
        "event_type": "handoff.complete",
        "status": status,
        "summary": result_summary or f"handoff completed by {agent_name or 'agent'}",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "from_agent_role": pending.get("from_role"),
            "to_agent": agent_name,
        },
        "payload": {},
        "raw_ref": {"storage": "inline"},
    })


