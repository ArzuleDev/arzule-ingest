"""Implicit handoff detection for CrewAI task flows.

Detects handoffs that occur through:
1. Task context dependencies - when task.context includes other tasks
2. Sequential task transitions - when one agent's task completes and
   a different agent's task starts next

This captures agent-to-agent information flow even without explicit
delegation tools, enabling context drift detection for normal crew
execution patterns.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..run import ArzuleRun

# Thread-safe storage for last completed task per run
# Used to detect sequential agent transitions
_last_completed_task: dict[str, dict[str, Any]] = {}
_last_completed_lock = threading.Lock()


def _compute_content_hash(content: Any) -> Optional[str]:
    """Compute hash of content for drift detection."""
    if content is None:
        return None
    try:
        if isinstance(content, dict):
            serialized = json.dumps(content, sort_keys=True, default=str)
        else:
            serialized = str(content)
        return hashlib.md5(serialized.encode()).hexdigest()[:12]
    except Exception:
        return None


def _extract_context_tasks(task: Any) -> list[Any]:
    """
    Extract context tasks from a CrewAI task.
    
    CrewAI tasks can have `context` which is a list of other tasks
    whose outputs should be available to this task.
    """
    context = getattr(task, "context", None)
    if context is None:
        return []
    
    if isinstance(context, (list, tuple)):
        return list(context)
    
    # Single task as context
    return [context] if context else []


def _get_task_output(task: Any) -> Optional[str]:
    """Extract the output from a completed task."""
    # Try various attributes where CrewAI stores task output
    for attr in ("output", "result", "raw_output", "output_raw"):
        output = getattr(task, attr, None)
        if output is not None:
            if hasattr(output, "raw"):
                return str(output.raw)
            return str(output)
    return None


def _get_task_description(task: Any) -> Optional[str]:
    """Get task description/expected output for comparison."""
    desc = getattr(task, "description", None)
    expected = getattr(task, "expected_output", None)
    
    parts = []
    if desc:
        parts.append(str(desc))
    if expected:
        parts.append(f"Expected: {expected}")
    
    return " | ".join(parts) if parts else None


def _get_task_identifier(task: Any) -> str:
    """Get a unique identifier for a task."""
    task_id = getattr(task, "id", None)
    if task_id:
        return str(task_id)
    
    task_name = getattr(task, "name", None)
    if task_name:
        return task_name
    
    # Fall back to description hash
    desc = getattr(task, "description", "")
    return f"task:{hashlib.md5(str(desc)[:100].encode()).hexdigest()[:8]}"


def _get_agent_role(task: Any) -> Optional[str]:
    """Get the agent role assigned to a task."""
    agent = getattr(task, "agent", None)
    if agent:
        return getattr(agent, "role", None)
    return None


def detect_task_context_handoff(
    run: "ArzuleRun",
    task: Any,
    span_id: Optional[str] = None,
) -> list[str]:
    """
    Detect implicit handoffs from context tasks and emit handoff.proposed events.
    
    When a task has context from other tasks, each context task represents
    an implicit handoff of information. We emit handoff.proposed for each.
    
    Args:
        run: The active ArzuleRun
        task: The CrewAI task that is starting
        span_id: The current span ID
        
    Returns:
        List of handoff keys generated for tracking
    """
    context_tasks = _extract_context_tasks(task)
    if not context_tasks:
        return []
    
    handoff_keys = []
    receiving_agent = _get_agent_role(task)
    receiving_task_id = _get_task_identifier(task)
    
    for ctx_task in context_tasks:
        # Get info about the providing task
        providing_agent = _get_agent_role(ctx_task)
        providing_task_id = _get_task_identifier(ctx_task)
        ctx_output = _get_task_output(ctx_task)
        ctx_description = _get_task_description(ctx_task)
        
        # Generate handoff key
        handoff_key = str(uuid.uuid4())
        handoff_keys.append(handoff_key)
        
        # Store pending handoff for correlation
        run._handoff_pending[handoff_key] = {
            "type": "implicit_context",
            "from_role": providing_agent,
            "from_task_id": providing_task_id,
            "to_role": receiving_agent,
            "to_task_id": receiving_task_id,
            "proposed_at": run.now(),
            "context_output": ctx_output,
            "context_description": ctx_description,
        }
        
        # Build payload for semantic analysis
        payload = {
            "context_source": {
                "task_id": providing_task_id,
                "agent_role": providing_agent,
                "description": ctx_description,
            },
            "context_output": ctx_output,
        }
        
        # Compute hash for contract drift detection
        content_hash = _compute_content_hash(ctx_output)
        
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
            "agent": {
                "id": f"crewai:role:{providing_agent}" if providing_agent else None,
                "role": providing_agent,
            } if providing_agent else None,
            "event_type": "handoff.proposed",
            "status": "ok",
            "summary": f"context handoff: {providing_agent or 'task'} -> {receiving_agent or 'task'}",
            "attrs_compact": {
                "handoff_key": handoff_key,
                "handoff_type": "implicit_context",
                "from_agent_role": providing_agent,
                "from_task_id": providing_task_id,
                "to_agent_role": receiving_agent,
                "to_task_id": receiving_task_id,
                "payload_hash": content_hash,
            },
            "payload": payload,
            "raw_ref": {"storage": "inline"},
        })
    
    return handoff_keys


def emit_implicit_handoff_complete(
    run: "ArzuleRun",
    task: Any,
    status: str = "ok",
    span_id: Optional[str] = None,
) -> int:
    """
    Emit handoff.complete events for all implicit handoffs to this task.
    
    Called when a task completes. Looks up any pending implicit handoffs
    (both context-based and sequential) that targeted this task.
    
    Args:
        run: The active ArzuleRun
        task: The completed CrewAI task
        status: Completion status
        span_id: The current span ID
        
    Returns:
        Number of handoff.complete events emitted
    """
    task_id = _get_task_identifier(task)
    agent_role = _get_agent_role(task)
    task_output = _get_task_output(task)
    task_description = _get_task_description(task)
    
    # Find all pending handoffs targeting this task
    completed_count = 0
    keys_to_remove = []
    
    for handoff_key, pending in list(run._handoff_pending.items()):
        # Only process implicit handoffs (context or sequential)
        handoff_type = pending.get("type", "")
        if handoff_type not in ("implicit_context", "implicit_sequential"):
            continue
        if pending.get("to_task_id") != task_id:
            continue
        
        keys_to_remove.append(handoff_key)
        
        # Get the original context that was provided
        context_output = pending.get("context_output") or pending.get("previous_output")
        from_agent = pending.get("from_role")
        from_task = pending.get("from_task_id")
        
        # Build payload with both what was received and what was produced
        payload = {
            "received_context": {
                "from_agent": from_agent,
                "from_task": from_task,
                "content": context_output[:500] if context_output else None,
            },
            "task_result": task_output[:500] if task_output else None,
        }
        
        # Compute result hash
        result_hash = _compute_content_hash(task_output)
        
        # Create result summary
        result_summary = None
        if task_output:
            result_summary = task_output[:100] + "..." if len(task_output) > 100 else task_output
        
        # Emit handoff.complete
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
            "event_type": "handoff.complete",
            "status": status,
            "summary": result_summary or f"context processed by {agent_role or 'task'}",
            "attrs_compact": {
                "handoff_key": handoff_key,
                "handoff_type": handoff_type,
                "from_agent_role": from_agent,
                "to_agent_role": agent_role,
                "result": result_summary,
                "result_hash": result_hash,
            },
            "payload": payload,
            "raw_ref": {"storage": "inline"},
        })
        
        completed_count += 1
    
    # Clean up processed handoffs
    for key in keys_to_remove:
        run._handoff_pending.pop(key, None)
    
    # Store this task as the last completed for sequential tracking
    _store_last_completed_task(run.run_id, task, task_output, agent_role, task_id, task_description)
    
    return completed_count


# =============================================================================
# Sequential Task Transition Tracking
# =============================================================================

def _store_last_completed_task(
    run_id: str,
    task: Any,
    output: Optional[str],
    agent_role: Optional[str],
    task_id: str,
    description: Optional[str],
) -> None:
    """Store info about the last completed task for sequential handoff detection."""
    with _last_completed_lock:
        _last_completed_task[run_id] = {
            "task": task,
            "output": output,
            "agent_role": agent_role,
            "task_id": task_id,
            "description": description,
            "output_hash": _compute_content_hash(output),
        }


def _get_last_completed_task(run_id: str) -> Optional[dict[str, Any]]:
    """Get info about the last completed task for a run."""
    with _last_completed_lock:
        return _last_completed_task.get(run_id)


def _clear_last_completed_task(run_id: str) -> None:
    """Clear last completed task tracking for a run."""
    with _last_completed_lock:
        _last_completed_task.pop(run_id, None)


def detect_sequential_handoff(
    run: "ArzuleRun",
    task: Any,
    span_id: Optional[str] = None,
) -> Optional[str]:
    """
    Detect implicit handoff from sequential task execution.
    
    When a task starts with a DIFFERENT agent than the previous task,
    we treat it as an implicit handoff of the previous task's output.
    
    This captures the common pattern where tasks run sequentially
    and each agent builds on the previous work.
    
    Args:
        run: The active ArzuleRun
        task: The CrewAI task that is starting
        span_id: The current span ID
        
    Returns:
        Handoff key if a sequential handoff was detected, None otherwise
    """
    last_task = _get_last_completed_task(run.run_id)
    if not last_task:
        return None
    
    current_agent = _get_agent_role(task)
    previous_agent = last_task.get("agent_role")
    
    # Only emit handoff if agents are different
    # Same agent continuing work isn't a handoff
    if current_agent == previous_agent:
        return None
    
    # Both agents should be known for meaningful analysis
    if not current_agent or not previous_agent:
        return None
    
    current_task_id = _get_task_identifier(task)
    previous_task_id = last_task.get("task_id")
    previous_output = last_task.get("output")
    previous_description = last_task.get("description")
    
    # Generate handoff key
    handoff_key = str(uuid.uuid4())
    
    # Store pending handoff for completion tracking
    run._handoff_pending[handoff_key] = {
        "type": "implicit_sequential",
        "from_role": previous_agent,
        "from_task_id": previous_task_id,
        "to_role": current_agent,
        "to_task_id": current_task_id,
        "proposed_at": run.now(),
        "previous_output": previous_output,
        "previous_description": previous_description,
    }
    
    # Build payload
    payload = {
        "previous_task": {
            "task_id": previous_task_id,
            "agent_role": previous_agent,
            "description": previous_description,
        },
        "previous_output": previous_output[:1000] if previous_output else None,
        "current_task": {
            "task_id": current_task_id,
            "agent_role": current_agent,
            "description": _get_task_description(task),
        },
    }
    
    # Emit handoff.proposed
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
            "id": f"crewai:role:{previous_agent}",
            "role": previous_agent,
        },
        "event_type": "handoff.proposed",
        "status": "ok",
        "summary": f"sequential handoff: {previous_agent} -> {current_agent}",
        "attrs_compact": {
            "handoff_key": handoff_key,
            "handoff_type": "implicit_sequential",
            "from_agent_role": previous_agent,
            "from_task_id": previous_task_id,
            "to_agent_role": current_agent,
            "to_task_id": current_task_id,
            "payload_hash": last_task.get("output_hash"),
        },
        "payload": payload,
        "raw_ref": {"storage": "inline"},
    })
    
    return handoff_key


def cleanup_run_tracking(run_id: str) -> None:
    """Clean up tracking state when a run ends."""
    _clear_last_completed_task(run_id)

