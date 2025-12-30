"""Handoff detection and correlation for agent delegation."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..run import ArzuleRun


def _compute_payload_hash(payload: Any) -> Optional[str]:
    """
    Compute a stable hash of handoff payload for contract drift detection.
    
    Returns a 12-character hex hash that can be used to identify
    changes in payload structure over time.
    """
    if payload is None:
        return None
    
    try:
        if isinstance(payload, dict):
            serialized = json.dumps(payload, sort_keys=True, default=str)
        else:
            serialized = str(payload)
        
        return hashlib.md5(serialized.encode()).hexdigest()[:12]
    except Exception:
        return None


def _extract_payload_keys(payload: Any, max_keys: int = 15) -> list[str]:
    """
    Extract top-level keys from handoff payload for schema analysis.
    
    Returns sorted list of keys (capped at max_keys) for detecting
    contract drift across handoffs between the same agent pairs.
    """
    if not isinstance(payload, dict):
        return []
    
    try:
        # Get sorted keys, excluding internal arzule metadata
        keys = [k for k in payload.keys() if k != "arzule"]
        return sorted(keys)[:max_keys]
    except Exception:
        return []

# Pattern to extract handoff keys from task descriptions
HANDOFF_RE = re.compile(r"\[arzule_handoff:([0-9a-f-]{36})\]")

# Exact tool names that indicate delegation (CrewAI standard tools)
DELEGATION_TOOL_NAMES = {
    "delegate_work_to_coworker",
    "ask_question_to_coworker",
    "Delegate work to coworker",
    "Ask question to coworker",
}

# Patterns that indicate delegation/handoff when found in tool names
# These cover common multi-agent patterns across frameworks
DELEGATION_PATTERNS = [
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
    # Call patterns (LangGraph style)
    "call_agent",
    "invoke_agent",
]


def is_delegation_tool(tool_name: Optional[str]) -> bool:
    """
    Check if a tool name indicates agent-to-agent delegation.
    
    Matches:
    - Exact CrewAI tool names
    - Pattern-based detection for common delegation tools
    - Works across CrewAI, LangGraph, custom implementations
    """
    if not tool_name:
        return False
    
    name = tool_name.strip()
    
    # Check exact matches first
    if name in DELEGATION_TOOL_NAMES:
        return True
    
    # Check patterns (case-insensitive)
    name_lower = name.lower()
    for pattern in DELEGATION_PATTERNS:
        if pattern in name_lower:
            return True
    
    return False


def maybe_inject_handoff_key(run: "ArzuleRun", context: Any) -> Optional[str]:
    """
    Inject a handoff key into delegation tool calls.

    Modifies context.tool_input in-place to add the handoff marker.

    Args:
        run: The active ArzuleRun
        context: The tool call context from CrewAI

    Returns:
        The handoff key if injected, None otherwise
    """
    tool_name = getattr(context, "tool_name", None)
    if not is_delegation_tool(tool_name):
        return None

    tool_input = getattr(context, "tool_input", None)
    if not isinstance(tool_input, dict):
        return None

    # Generate handoff key
    handoff_key = str(uuid.uuid4())

    # Inject into arzule metadata namespace
    if "arzule" not in tool_input:
        tool_input["arzule"] = {}
    tool_input["arzule"]["handoff_key"] = handoff_key

    # Inject marker into the delegated work payload so receiving task carries it
    # Try common field names used by CrewAI delegation tools
    for field in ("task", "question", "instructions", "context", "message", "description"):
        if field in tool_input and isinstance(tool_input[field], str):
            marker = f"[arzule_handoff:{handoff_key}] "
            tool_input[field] = marker + tool_input[field]
            break

    # Store pending handoff metadata for later correlation
    # IMPORTANT: Include tool_input so we can compare the original delegation
    # context against the result for context drift detection
    agent = getattr(context, "agent", None)
    run._handoff_pending[handoff_key] = {
        "from_role": getattr(agent, "role", None) if agent else None,
        "from_agent_id": f"crewai:role:{getattr(agent, 'role', 'unknown')}" if agent else None,
        "tool_name": tool_name,
        "proposed_at": run.now(),
        "tool_input": tool_input.copy() if tool_input else {},  # Store for drift detection
    }

    return handoff_key


def maybe_emit_handoff_proposed(run: "ArzuleRun", context: Any, span_id: Optional[str]) -> None:
    """
    Emit handoff.proposed event if this was a delegation call.

    Args:
        run: The active ArzuleRun
        context: The tool call context
        span_id: The tool span ID
    """
    tool_input = getattr(context, "tool_input", None)
    if not isinstance(tool_input, dict):
        return

    arzule_meta = tool_input.get("arzule", {})
    handoff_key = arzule_meta.get("handoff_key")
    if not handoff_key:
        return

    agent = getattr(context, "agent", None)
    tool_name = getattr(context, "tool_name", None)

    # Try to extract target agent from tool input
    to_coworker = tool_input.get("coworker") or tool_input.get("to") or tool_input.get("agent")

    # Compute detection fields for forensics (contract drift detection)
    payload_hash = _compute_payload_hash(tool_input)
    payload_keys = _extract_payload_keys(tool_input)

    attrs = {
        "handoff_key": handoff_key,
        "from_agent_role": getattr(agent, "role", None) if agent else None,
        "to_coworker": to_coworker,
        "tool_name": tool_name,
    }
    
    # Add detection fields if available
    if payload_hash:
        attrs["payload_hash"] = payload_hash
    if payload_keys:
        attrs["payload_keys"] = payload_keys

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
            "id": f"crewai:role:{getattr(agent, 'role', 'unknown')}" if agent else None,
            "role": getattr(agent, "role", None) if agent else None,
        } if agent else None,
        "event_type": "handoff.proposed",
        "status": "ok",
        "summary": f"handoff proposed to {to_coworker or 'coworker'}",
        "attrs_compact": attrs,
        "payload": {"tool_input": tool_input},
        "raw_ref": {"storage": "inline"},
    })


def extract_handoff_key_from_text(text: Optional[str]) -> Optional[str]:
    """
    Extract a handoff key from text (e.g., task description).

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
    task_id: Optional[str] = None,
    agent_role: Optional[str] = None,
    span_id: Optional[str] = None,
) -> None:
    """
    Emit handoff.ack event when receiving agent starts the delegated task.

    Args:
        run: The active ArzuleRun
        handoff_key: The handoff correlation key
        task_id: The task ID
        agent_role: The receiving agent's role
        span_id: The current span ID
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
            "id": f"crewai:role:{agent_role}" if agent_role else None,
            "role": agent_role,
        } if agent_role else None,
        "task_id": task_id,
        "event_type": "handoff.ack",
        "status": "ok",
        "summary": f"handoff acknowledged by {agent_role or 'agent'}",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "from_agent_role": pending.get("from_role"),
            "to_agent_role": agent_role,
        },
        "payload": {},
        "raw_ref": {"storage": "inline"},
    })


def emit_handoff_complete(
    run: "ArzuleRun",
    handoff_key: str,
    task_id: Optional[str] = None,
    agent_role: Optional[str] = None,
    span_id: Optional[str] = None,
    status: str = "ok",
    result_summary: Optional[str] = None,
    result_payload: Optional[Any] = None,
) -> None:
    """
    Emit handoff.complete event when delegated task finishes.

    Args:
        run: The active ArzuleRun
        handoff_key: The handoff correlation key
        task_id: The task ID
        agent_role: The agent's role
        span_id: The current span ID
        status: Completion status
        result_summary: Brief summary of result
        result_payload: Full result data for semantic analysis
    """
    pending = run._handoff_pending.pop(handoff_key, {})
    
    # Build payload for semantic drift detection
    # Include both the original request (from pending) and the result
    payload = {}
    
    # Include original delegation request for comparison
    tool_input = pending.get("tool_input", {})
    if tool_input:
        payload["delegation_request"] = {
            "task": tool_input.get("task"),
            "context": tool_input.get("context"),
            "coworker": tool_input.get("coworker"),
        }
    
    # Include the result
    if result_payload is not None:
        # Truncate if too large
        result_str = str(result_payload)
        if len(result_str) > 2000:
            payload["result"] = result_str[:2000] + "..."
        else:
            payload["result"] = result_str
    elif result_summary:
        payload["result"] = result_summary
    
    # Compute result hash for drift detection
    result_hash = _compute_payload_hash(result_payload or result_summary)

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
            "id": f"crewai:role:{agent_role}" if agent_role else None,
            "role": agent_role,
        } if agent_role else None,
        "task_id": task_id,
        "event_type": "handoff.complete",
        "status": status,
        "summary": result_summary or f"handoff completed by {agent_role or 'agent'}",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "from_agent_role": pending.get("from_role"),
            "to_agent_role": agent_role,
            "result_hash": result_hash,
        },
        "payload": payload,
        "raw_ref": {"storage": "inline"},
    })





