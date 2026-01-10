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

from ..ids import new_span_id

if TYPE_CHECKING:
    from ..run import ArzuleRun


def _generate_instance_id() -> str:
    """Generate a unique instance identifier for agent tracking.

    Returns:
        A short unique identifier (8 characters) for agent instance.
    """
    return uuid.uuid4().hex[:8]


def _get_agent_instance_id(task: Any) -> Optional[str]:
    """Extract instance_id from a task's agent if available.

    Looks for instance_id in:
    1. agent._arzule_instance_id (set by our instrumentation)
    2. agent.instance_id (if agent tracks it directly)

    Args:
        task: The CrewAI task object.

    Returns:
        The instance_id if found, None otherwise.
    """
    agent = getattr(task, "agent", None)
    if not agent:
        return None

    # Check for arzule-specific instance tracking
    instance_id = getattr(agent, "_arzule_instance_id", None)
    if instance_id:
        return instance_id

    # Check for direct instance_id attribute
    instance_id = getattr(agent, "instance_id", None)
    if instance_id:
        return instance_id

    return None


def _build_agent_id(role: str, instance_id: Optional[str] = None) -> str:
    """Build agent ID using instance-aware format when available.

    Args:
        role: The agent role.
        instance_id: Optional instance ID for uniqueness.

    Returns:
        Agent ID in format:
        - "crewai:role:{role}:instance_{instance_id}" when instance_id is provided
        - "crewai:role:{role}" as fallback for backward compatibility
    """
    if instance_id:
        return f"crewai:role:{role}:instance_{instance_id}"
    return f"crewai:role:{role}"

# =============================================================================
# Payload Size Limits
# =============================================================================
# These limits control how much content is stored in handoff events.
# The semantic analyzer (backend) uses these payloads for context drift detection.
# Limits are set to match the semantic analyzer's LLM context capacity (~100K per field).
#
# IMPORTANT: If you see false positive ContextDrift findings where the LLM claims
# requirements were ignored but the requirements were actually in the truncated portion,
# consider increasing these limits.

# Maximum size for combined context from multiple sources (implicit_context handoffs)
MAX_COMBINED_CONTEXT_CHARS = 100_000

# Maximum size for individual context output (sequential handoffs, single sources)
MAX_CONTEXT_OUTPUT_CHARS = 100_000

# Maximum size for task result/output
MAX_TASK_RESULT_CHARS = 100_000

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
    """Get the agent role assigned to a task.
    
    Tries multiple attributes in order of preference:
    1. agent.role (standard CrewAI)
    2. agent.name (fallback if no role)
    
    Returns None if no agent or agent has no role/name.
    Task identifier is stored separately in from_task_id.
    """
    agent = getattr(task, "agent", None)
    if agent:
        # Try role first (standard), then name as fallback
        role = getattr(agent, "role", None)
        if role:
            return role
        name = getattr(agent, "name", None)
        if name:
            return name
    return None


def _get_task_display_name(task: Any) -> str:
    """Get a human-readable display name for a task.
    
    Used as fallback when agent role is not available.
    Returns the task name, or a truncated description, or 'context task'.
    """
    # Try task name first
    name = getattr(task, "name", None)
    if name:
        return f"task:{name}"
    
    # Fall back to truncated description
    desc = getattr(task, "description", None)
    if desc:
        desc_str = str(desc).strip()
        if len(desc_str) > 30:
            desc_str = desc_str[:27] + "..."
        return f"task:{desc_str}"
    
    return "context task"


def detect_task_context_handoff(
    run: "ArzuleRun",
    task: Any,
    span_id: Optional[str] = None,
) -> list[str]:
    """
    Detect implicit handoffs from context tasks and emit a single aggregated handoff.proposed event.
    
    When a task has context from other tasks, we emit ONE handoff.proposed that includes
    ALL context sources. This ensures drift detection compares the task output against
    the complete combined context, avoiding false positives when a task correctly uses
    data from multiple sources.
    
    Args:
        run: The active ArzuleRun
        task: The CrewAI task that is starting
        span_id: The current span ID
        
    Returns:
        List containing the single handoff key (or empty if no context)
    """
    context_tasks = _extract_context_tasks(task)
    if not context_tasks:
        return []
    
    receiving_agent = _get_agent_role(task)
    receiving_task_id = _get_task_identifier(task)
    # Use task display name as fallback when agent role is unknown
    receiving_display = receiving_agent or _get_task_display_name(task)
    
    # Aggregate ALL context sources into a single handoff
    handoff_key = str(uuid.uuid4())
    
    # Collect all context sources
    context_sources = []
    combined_context_parts = []
    from_agents = []
    first_from_instance_id: Optional[str] = None  # For building agent ID

    for ctx_task in context_tasks:
        # Get info about the providing task
        providing_agent = _get_agent_role(ctx_task)
        providing_task_id = _get_task_identifier(ctx_task)
        ctx_output = _get_task_output(ctx_task)
        ctx_description = _get_task_description(ctx_task)
        providing_instance_id = _get_agent_instance_id(ctx_task)
        # Use task display name as fallback when agent role is unknown
        providing_display = providing_agent or _get_task_display_name(ctx_task)

        # Track first instance_id for the "from" agent in the event
        if first_from_instance_id is None and providing_instance_id:
            first_from_instance_id = providing_instance_id

        source_info = {
            "task_id": providing_task_id,
            "agent_role": providing_display,
            "agent_instance_id": providing_instance_id,
            "description": ctx_description,
            "output": ctx_output,
        }
        context_sources.append(source_info)
        from_agents.append(providing_display)

        if ctx_output:
            combined_context_parts.append(f"[{providing_display}]: {ctx_output}")
    
    # Combine all context outputs for analysis
    combined_context = "\n\n---\n\n".join(combined_context_parts) if combined_context_parts else None
    
    # Compute hash of combined context for drift detection
    content_hash = _compute_content_hash(combined_context)
    
    # Store pending handoff with ALL context task references for re-reading at completion
    run._handoff_pending[handoff_key] = {
        "type": "implicit_context",
        "context_tasks": context_tasks,  # Store task refs for re-reading at completion
        "context_sources": context_sources,
        "from_agents": from_agents,
        "to_role": receiving_display,
        "to_task_id": receiving_task_id,
        "proposed_at": run.now(),
        "combined_context": combined_context,
    }
    
    # Build payload with all context sources for semantic analysis
    payload = {
        "context_sources": context_sources,
        "combined_context": combined_context,
        "context_source_count": len(context_sources),
    }
    
    # Create summary showing all source agents
    if len(from_agents) == 1:
        summary = f"context handoff: {from_agents[0]} -> {receiving_display}"
    else:
        summary = f"context handoff: {len(from_agents)} sources -> {receiving_display}"
    
    # Emit SINGLE aggregated handoff.proposed event
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
        # Use first agent as the "from" agent for the event, but all are in payload
        "agent": {
            "id": _build_agent_id(from_agents[0], first_from_instance_id) if from_agents else None,
            "role": from_agents[0] if from_agents else None,
            "instance_id": first_from_instance_id,
        } if from_agents else None,
        "event_type": "handoff.proposed",
        "status": "ok",
        "summary": summary,
        "attrs_compact": {
            "handoff_key": handoff_key,
            "handoff_type": "implicit_context",
            "context_source_count": len(context_sources),
            "from_agents": from_agents,
            "to_agent_role": receiving_display,
            "to_task_id": receiving_task_id,
            "payload_hash": content_hash,
        },
        "payload": payload,
        "raw_ref": {"storage": "inline"},
    })
    
    return [handoff_key]


def _extract_task_intent(task: Any) -> dict[str, Any]:
    """
    Extract task intent metadata for semantic drift analysis.
    
    This helps the backend understand what the task was supposed to do,
    enabling smarter drift detection that distinguishes between:
    - Data transformation tasks (should use/transform context)
    - Review/coordination tasks (should validate, not repeat data)
    - Pass-through tasks (should forward with minimal changes)
    """
    intent: dict[str, Any] = {}
    
    # Task description
    desc = getattr(task, "description", None)
    if desc:
        intent["description"] = str(desc)[:500]
    
    # Expected output
    expected = getattr(task, "expected_output", None)
    if expected:
        intent["expected_output"] = str(expected)[:300]
    
    # Agent goal (provides context on agent's purpose)
    agent = getattr(task, "agent", None)
    if agent:
        goal = getattr(agent, "goal", None)
        if goal:
            intent["agent_goal"] = str(goal)[:300]
        
        # Agent role for additional context
        role = getattr(agent, "role", None)
        if role:
            intent["agent_role"] = str(role)
    
    # Task name if available
    name = getattr(task, "name", None)
    if name:
        intent["task_name"] = str(name)
    
    return intent


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
    
    For aggregated context handoffs (multiple sources), re-reads all context
    from the original task references to ensure we capture the actual context
    that was available (handles async timing where context may have been
    incomplete at task start).
    
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
    agent_instance_id = _get_agent_instance_id(task)
    task_output = _get_task_output(task)
    task_description = _get_task_description(task)
    # Use task display name as fallback when agent role is unknown
    agent_display = agent_role or _get_task_display_name(task)
    
    # Extract task intent for smarter drift analysis
    task_intent = _extract_task_intent(task)
    
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
        
        # Build payload based on handoff type
        if handoff_type == "implicit_context":
            # Re-read all context sources from task refs (now complete)
            # This ensures we capture actual context even for async tasks
            context_tasks = pending.get("context_tasks", [])
            actual_sources = []
            combined_parts = []
            from_agents = pending.get("from_agents", [])
            
            if context_tasks:
                # New format: re-read from task references
                for ctx_task in context_tasks:
                    ctx_output = _get_task_output(ctx_task)  # Re-read at completion
                    ctx_agent = _get_agent_role(ctx_task) or _get_task_display_name(ctx_task)
                    ctx_description = _get_task_description(ctx_task)
                    actual_sources.append({
                        "agent_role": ctx_agent,
                        "task_id": _get_task_identifier(ctx_task),
                        "description": ctx_description,
                        "output": ctx_output[:MAX_CONTEXT_OUTPUT_CHARS] if ctx_output else None,
                    })
                    if ctx_output:
                        combined_parts.append(f"[{ctx_agent}]: {ctx_output}")
            else:
                # Backward compatibility: use stored context_sources or context_output
                stored_sources = pending.get("context_sources", [])
                if stored_sources:
                    # New aggregated format stored at proposed time
                    for source in stored_sources:
                        actual_sources.append({
                            "agent_role": source.get("agent_role"),
                            "task_id": source.get("task_id"),
                            "description": source.get("description"),
                            "output": source.get("output"),
                        })
                        output = source.get("output")
                        if output:
                            combined_parts.append(f"[{source.get('agent_role', 'unknown')}]: {output}")
                else:
                    # Legacy single-source format
                    legacy_output = pending.get("context_output")
                    legacy_from = pending.get("from_role")
                    legacy_task_id = pending.get("from_task_id")
                    legacy_desc = pending.get("context_description")
                    if legacy_output or legacy_from:
                        actual_sources.append({
                            "agent_role": legacy_from,
                            "task_id": legacy_task_id,
                            "description": legacy_desc,
                            "output": legacy_output[:MAX_CONTEXT_OUTPUT_CHARS] if legacy_output else None,
                        })
                        if legacy_output:
                            combined_parts.append(f"[{legacy_from or 'unknown'}]: {legacy_output}")
                        # Also populate from_agents for attrs if not set
                        if not from_agents and legacy_from:
                            from_agents = [legacy_from]
            
            combined_actual = "\n\n---\n\n".join(combined_parts) if combined_parts else None
            
            payload = {
                "received_context": {
                    "source_count": len(actual_sources),
                    "sources": actual_sources,
                    "combined": combined_actual[:MAX_COMBINED_CONTEXT_CHARS] if combined_actual else None,
                },
                "task_result": task_output[:MAX_TASK_RESULT_CHARS] if task_output else None,
                "task_intent": task_intent if task_intent else None,
            }
            
            # For attrs, use the from_agents list
            from_agent_display = ", ".join(from_agents[:3]) if from_agents else "unknown"
            if len(from_agents) > 3:
                from_agent_display += f" +{len(from_agents) - 3} more"
        else:
            # Sequential handoff - single source (legacy format)
            context_output = pending.get("previous_output")
            from_agent = pending.get("from_role")
            from_task = pending.get("from_task_id")
            
            payload = {
                "received_context": {
                    "from_agent": from_agent,
                    "from_task": from_task,
                    "content": context_output[:MAX_CONTEXT_OUTPUT_CHARS] if context_output else None,
                },
                "task_result": task_output[:MAX_TASK_RESULT_CHARS] if task_output else None,
                "task_intent": task_intent if task_intent else None,
            }
            from_agent_display = from_agent
        
        # Compute result hash
        result_hash = _compute_content_hash(task_output)
        
        # Create result summary
        result_summary = None
        if task_output:
            result_summary = task_output[:100] + "..." if len(task_output) > 100 else task_output
        
        # Build attrs_compact
        attrs_compact = {
            "handoff_key": handoff_key,
            "handoff_type": handoff_type,
            "to_agent_role": agent_display,
            "result": result_summary,
            "result_hash": result_hash,
        }
        
        if handoff_type == "implicit_context":
            attrs_compact["context_source_count"] = len(actual_sources)
            attrs_compact["from_agents"] = from_agents
        else:
            attrs_compact["from_agent_role"] = from_agent_display
        
        # Emit handoff.complete
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
                "id": _build_agent_id(agent_role, agent_instance_id) if agent_role else None,
                "role": agent_role,
                "instance_id": agent_instance_id,
            } if agent_role else None,
            "event_type": "handoff.complete",
            "status": status,
            "summary": result_summary or f"context processed by {agent_display}",
            "attrs_compact": attrs_compact,
            "payload": payload,
            "raw_ref": {"storage": "inline"},
        })
        
        completed_count += 1
    
    # Clean up processed handoffs
    for key in keys_to_remove:
        run._handoff_pending.pop(key, None)
    
    # Store this task as the last completed for sequential tracking
    _store_last_completed_task(run.run_id, task, task_output, agent_role, agent_instance_id, task_id, task_description)
    
    return completed_count


# =============================================================================
# Sequential Task Transition Tracking
# =============================================================================

def _store_last_completed_task(
    run_id: str,
    task: Any,
    output: Optional[str],
    agent_role: Optional[str],
    agent_instance_id: Optional[str],
    task_id: str,
    description: Optional[str],
) -> None:
    """Store info about the last completed task for sequential handoff detection."""
    with _last_completed_lock:
        _last_completed_task[run_id] = {
            "task": task,
            "output": output,
            "agent_role": agent_role,
            "agent_instance_id": agent_instance_id,
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
    previous_agent_instance_id = last_task.get("agent_instance_id")

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
        "previous_output": previous_output[:MAX_CONTEXT_OUTPUT_CHARS] if previous_output else None,
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
        "span_id": span_id or new_span_id(),
        "parent_span_id": run.current_parent_span_id(),
        "seq": run.next_seq(),
        "ts": run.now(),
        "agent": {
            "id": _build_agent_id(previous_agent, previous_agent_instance_id),
            "role": previous_agent,
            "instance_id": previous_agent_instance_id,
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

