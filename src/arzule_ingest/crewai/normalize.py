"""Normalize CrewAI events and hook contexts to TraceEvent format."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Optional

from ..ids import new_span_id
from ..sanitize import sanitize, truncate_string

if TYPE_CHECKING:
    from ..run import ArzuleRun


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute from object."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _compute_input_hash(tool_input: Any) -> Optional[str]:
    """
    Compute a stable hash of tool input for repetition detection.
    
    Returns a 12-character hex hash that can be used to identify
    duplicate tool calls with the same input.
    """
    if tool_input is None:
        return None
    
    try:
        # Serialize with sorted keys for deterministic output
        if isinstance(tool_input, dict):
            serialized = json.dumps(tool_input, sort_keys=True, default=str)
        else:
            serialized = str(tool_input)
        
        return hashlib.md5(serialized.encode()).hexdigest()[:12]
    except Exception:
        return None


def _extract_input_keys(tool_input: Any, max_keys: int = 10) -> list[str]:
    """
    Extract top-level keys from tool input for schema analysis.
    
    Returns sorted list of keys (capped at max_keys) for detecting
    schema drift across tool calls.
    """
    if not isinstance(tool_input, dict):
        return []
    
    try:
        # Get sorted keys, excluding internal arzule metadata
        keys = [k for k in tool_input.keys() if k != "arzule"]
        return sorted(keys)[:max_keys]
    except Exception:
        return []


def _extract_agent_info(agent: Any) -> Optional[dict[str, Any]]:
    """Extract agent info from CrewAI agent object."""
    if not agent:
        return None

    role = _safe_getattr(agent, "role", None) or _safe_getattr(agent, "name", "unknown")
    return {
        "id": f"crewai:role:{role}",
        "role": role,
    }


def extract_agent_info_from_event(event: Any) -> Optional[dict[str, Any]]:
    """Extract agent info dict from a CrewAI event object.
    
    This is a convenience function for the listener to extract agent info
    from event bus events for thread-local agent tracking.
    
    Args:
        event: CrewAI event object (e.g., AgentExecutionStartedEvent)
        
    Returns:
        Agent info dict with 'id' and 'role', or None if no agent
    """
    agent = _safe_getattr(event, "agent", None)
    return _extract_agent_info(agent)


def _extract_task_info(task: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract task ID and description from CrewAI task."""
    if not task:
        return None, None

    task_id = _safe_getattr(task, "id", None) or _safe_getattr(task, "name", None)
    description = _safe_getattr(task, "description", None)
    return task_id, description


def _extract_token_usage(response: Any) -> Optional[dict[str, int]]:
    """Extract token usage from LLM response.
    
    Supports multiple formats:
    - OpenAI/LiteLLM: response.usage.{prompt_tokens, completion_tokens, total_tokens}
    - usage_metadata: response.usage_metadata.{input_tokens, output_tokens}
    - Direct attributes: response.{prompt_tokens, completion_tokens, total_tokens}
    
    Returns:
        Dict with token counts or None if not available
    """
    if not response:
        return None
    
    result: dict[str, int] = {}
    
    # Try response.usage (OpenAI/LiteLLM format)
    usage = _safe_getattr(response, "usage", None)
    if usage:
        prompt = _safe_getattr(usage, "prompt_tokens", None)
        completion = _safe_getattr(usage, "completion_tokens", None)
        total = _safe_getattr(usage, "total_tokens", None)
        
        # Also check for input_tokens/output_tokens naming
        if prompt is None:
            prompt = _safe_getattr(usage, "input_tokens", None)
        if completion is None:
            completion = _safe_getattr(usage, "output_tokens", None)
        
        if prompt is not None:
            result["prompt_tokens"] = int(prompt)
        if completion is not None:
            result["completion_tokens"] = int(completion)
        if total is not None:
            result["total_tokens"] = int(total)
        elif prompt is not None and completion is not None:
            result["total_tokens"] = int(prompt) + int(completion)
    
    # Try response.usage_metadata (some providers)
    if not result:
        usage_meta = _safe_getattr(response, "usage_metadata", None)
        if usage_meta:
            input_tokens = _safe_getattr(usage_meta, "input_tokens", None)
            output_tokens = _safe_getattr(usage_meta, "output_tokens", None)
            
            if input_tokens is not None:
                result["prompt_tokens"] = int(input_tokens)
            if output_tokens is not None:
                result["completion_tokens"] = int(output_tokens)
            if input_tokens is not None and output_tokens is not None:
                result["total_tokens"] = int(input_tokens) + int(output_tokens)
    
    # Try dict-style access (some responses are dicts)
    if not result and isinstance(response, dict):
        usage = response.get("usage", {})
        if usage:
            if "prompt_tokens" in usage:
                result["prompt_tokens"] = int(usage["prompt_tokens"])
            if "completion_tokens" in usage:
                result["completion_tokens"] = int(usage["completion_tokens"])
            if "total_tokens" in usage:
                result["total_tokens"] = int(usage["total_tokens"])
    
    return result if result else None


def _base(
    run: "ArzuleRun",
    *,
    span_id: Optional[str],
    parent_span_id: Optional[str],
    async_id: Optional[str] = None,
    causal_parents: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build base event fields with optional async support."""
    event = {
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": span_id or new_span_id(),
        "parent_span_id": parent_span_id,
        "seq": run.next_seq(),
        "ts": run.now(),
        "workstream_id": None,
        "task_id": None,
        "raw_ref": {"storage": "inline"},
    }
    
    # Add async fields if provided
    if async_id:
        event["async_id"] = async_id
    if causal_parents:
        event["causal_parents"] = causal_parents
    
    return event


# =============================================================================
# Event Listener Normalization (Crew/Agent/Task lifecycle events)
# =============================================================================


def evt_from_crewai_event(
    run: "ArzuleRun",
    event: Any,
    *,
    parent_span_id: Optional[str] = None,
    task_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convert a CrewAI event bus event to a TraceEvent.

    Args:
        run: The active ArzuleRun
        event: CrewAI event object
        parent_span_id: Optional explicit parent span (for concurrent task tracking)
        task_key: Optional task key for correlation in concurrent mode

    Returns:
        TraceEvent dict
    """
    event_class = event.__class__.__name__

    # Map event class names to our event types
    event_type_map = {
        # Crew lifecycle
        "CrewKickoffStartedEvent": "crew.kickoff.start",
        "CrewKickoffCompletedEvent": "crew.kickoff.complete",
        "CrewKickoffFailedEvent": "crew.kickoff.failed",
        # Agent lifecycle
        "AgentExecutionStartedEvent": "agent.execution.start",
        "AgentExecutionCompletedEvent": "agent.execution.complete",
        "AgentExecutionFailedEvent": "agent.execution.failed",
        "AgentExecutionErrorEvent": "agent.execution.error",
        # Task lifecycle
        "TaskStartedEvent": "task.start",
        "TaskCompletedEvent": "task.complete",
        "TaskFailedEvent": "task.failed",
        # Tool usage (CrewAI 1.7.x)
        "ToolUsageStartedEvent": "tool.call.start",
        "ToolUsageFinishedEvent": "tool.call.end",
        "ToolUsageErrorEvent": "tool.call.error",
        "ToolUsageEvent": "tool.call",
        # LLM calls (CrewAI 1.7.x)
        "LLMCallStartedEvent": "llm.call.start",
        "LLMCallCompletedEvent": "llm.call.end",
        "LLMCallFailedEvent": "llm.call.error",
        # Flow events
        "FlowStartedEvent": "flow.start",
        "FlowFinishedEvent": "flow.complete",
        # Memory events
        "MemoryQueryStartedEvent": "memory.query.start",
        "MemoryQueryCompletedEvent": "memory.query.end",
        "MemoryQueryFailedEvent": "memory.query.error",
        "MemorySaveStartedEvent": "memory.save.start",
        "MemorySaveCompletedEvent": "memory.save.end",
        "MemorySaveFailedEvent": "memory.save.error",
        # Knowledge events
        "KnowledgeQueryStartedEvent": "knowledge.query.start",
        "KnowledgeQueryCompletedEvent": "knowledge.query.end",
        "KnowledgeQueryFailedEvent": "knowledge.query.error",
        "KnowledgeRetrievalStartedEvent": "knowledge.retrieval.start",
        "KnowledgeRetrievalCompletedEvent": "knowledge.retrieval.end",
        # Reasoning events
        "AgentReasoningStartedEvent": "agent.reasoning.start",
        "AgentReasoningCompletedEvent": "agent.reasoning.end",
        "AgentReasoningFailedEvent": "agent.reasoning.error",
    }

    event_type = event_type_map.get(event_class, f"crewai.{event_class}")

    # Determine status
    status = "ok"
    if "Failed" in event_class or "Error" in event_class:
        status = "error"

    # Extract agent/task/tool info
    agent = _safe_getattr(event, "agent", None)
    task = _safe_getattr(event, "task", None)
    crew = _safe_getattr(event, "crew", None)

    agent_info = _extract_agent_info(agent)
    task_id, task_desc = _extract_task_info(task)

    # Tool info for tool events
    # CrewAI uses "tool_args" for tool input, fall back to "tool_input" for compatibility
    tool_name = _safe_getattr(event, "tool_name", None)
    tool_input = _safe_getattr(event, "tool_args", None)
    if tool_input is None:
        tool_input = _safe_getattr(event, "tool_input", None)
    tool_output = _safe_getattr(event, "tool_output", None) or _safe_getattr(event, "output", None)

    # Build summary
    summary_parts = [event_type]
    if agent_info:
        summary_parts.append(f"agent={agent_info['role']}")
    if task_id:
        summary_parts.append(f"task={task_id}")
    if tool_name:
        summary_parts.append(f"tool={tool_name}")
    summary = " ".join(summary_parts)

    # Extract result/error for completed/failed events
    payload: dict[str, Any] = {}
    attrs: dict[str, Any] = {}

    result = _safe_getattr(event, "result", None)
    if result is not None:
        payload["result"] = truncate_string(str(result), 1000)

    output = _safe_getattr(event, "output", None)
    if output is not None:
        payload["output"] = truncate_string(str(output), 1000)

    error = _safe_getattr(event, "error", None)
    if error is not None:
        attrs["error"] = truncate_string(str(error), 200)

    # Crew info
    if crew:
        attrs["crew_name"] = _safe_getattr(crew, "name", None)

    # Task info in payload
    if task is not None:
        payload["task"] = sanitize({
            "id": _safe_getattr(task, "id", None),
            "description": _safe_getattr(task, "description", None),
            "expected_output": _safe_getattr(task, "expected_output", None),
            "async_execution": _safe_getattr(task, "async_execution", None),
        })

    # Tool info in payload/attrs
    if tool_name:
        attrs["tool_name"] = tool_name
        
        # Add detection fields for forensics (tool events)
        if tool_input is not None:
            input_hash = _compute_input_hash(tool_input)
            input_keys = _extract_input_keys(tool_input)
            if input_hash:
                attrs["tool_input_hash"] = input_hash
            if input_keys:
                attrs["tool_input_keys"] = input_keys
    
    if tool_input is not None:
        payload["tool_input"] = sanitize(tool_input)
    if tool_output is not None:
        payload["tool_output"] = sanitize(tool_output)

    # LLM event info in payload
    messages = _safe_getattr(event, "messages", None)
    if messages is not None:
        payload["messages"] = _truncate_messages(messages)
        attrs["message_count"] = len(messages) if isinstance(messages, list) else 0

    response = _safe_getattr(event, "response", None)
    if response is not None:
        content = _safe_getattr(response, "content", None)
        if content is None:
            content = str(response)
        payload["response"] = truncate_string(str(content), 2000)
        
        # Extract token usage from LLM response (LiteLLM/OpenAI format)
        token_usage = _extract_token_usage(response)
        if token_usage:
            attrs.update(token_usage)

    # Determine parent span:
    # 1. Use explicit parent_span_id if provided (concurrent mode)
    # 2. Try task-based lookup if task_key provided
    # 3. Fall back to legacy sequential mode
    effective_parent_span = parent_span_id
    if effective_parent_span is None and task_key:
        effective_parent_span = run.get_task_parent_span(task_key=task_key)
    if effective_parent_span is None:
        # Extract agent key for agent-based task lookup
        agent_key = f"agent:{agent_info['role']}" if agent_info and agent_info.get('role') else None
        if agent_key:
            effective_parent_span = run.get_task_parent_span(agent_key=agent_key)
    if effective_parent_span is None:
        effective_parent_span = run.current_parent_span_id()

    return {
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": new_span_id(),
        "parent_span_id": effective_parent_span,
        "seq": run.next_seq(),
        "ts": run.now(),
        "workstream_id": None,
        "agent": agent_info,
        "task_id": task_id,
        "event_type": event_type,
        "status": status,
        "summary": summary,
        "attrs_compact": attrs,
        "payload": payload,
        "raw_ref": {"storage": "inline"},
    }


# =============================================================================
# Tool Hook Normalization
# =============================================================================


def evt_tool_start(run: "ArzuleRun", context: Any, span_id: str) -> dict[str, Any]:
    """
    Create event for tool call start.

    Args:
        run: The active ArzuleRun
        context: CrewAI tool call context
        span_id: The span ID for this tool call

    Returns:
        TraceEvent dict
    """
    tool_name = _safe_getattr(context, "tool_name", "unknown_tool")
    tool_input = _safe_getattr(context, "tool_input", {})
    agent = _safe_getattr(context, "agent", None)

    agent_info = _extract_agent_info(agent)

    # Extract handoff_key if present
    handoff_key = None
    if isinstance(tool_input, dict):
        arz = tool_input.get("arzule", {})
        handoff_key = arz.get("handoff_key") if isinstance(arz, dict) else None

    # Compute detection fields for forensics
    input_hash = _compute_input_hash(tool_input)
    input_keys = _extract_input_keys(tool_input)

    attrs = {
        "tool_name": tool_name,
        "handoff_key": handoff_key,
    }
    
    # Add detection fields if available
    if input_hash:
        attrs["tool_input_hash"] = input_hash
    if input_keys:
        attrs["tool_input_keys"] = input_keys

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "tool.call.start",
        "status": "ok",
        "summary": f"tool call: {tool_name}",
        "attrs_compact": attrs,
        "payload": {
            "tool_input": sanitize(tool_input),
        },
    }


def evt_tool_end(run: "ArzuleRun", context: Any, span_id: Optional[str]) -> dict[str, Any]:
    """
    Create event for tool call end.

    Args:
        run: The active ArzuleRun
        context: CrewAI tool call context
        span_id: The span ID for this tool call

    Returns:
        TraceEvent dict
    """
    tool_name = _safe_getattr(context, "tool_name", "unknown_tool")
    # Try both attribute names for tool output
    tool_result = _safe_getattr(context, "tool_result", None)
    if tool_result is None:
        tool_result = _safe_getattr(context, "tool_output", None)
    tool_error = _safe_getattr(context, "error", None) or _safe_getattr(context, "exception", None)
    agent = _safe_getattr(context, "agent", None)
    tool_input = _safe_getattr(context, "tool_input", {})

    agent_info = _extract_agent_info(agent)
    status = "error" if tool_error else "ok"

    # Extract handoff_key if present
    handoff_key = None
    if isinstance(tool_input, dict):
        arz = tool_input.get("arzule", {})
        handoff_key = arz.get("handoff_key") if isinstance(arz, dict) else None

    # Compute detection fields for forensics
    input_hash = _compute_input_hash(tool_input)
    input_keys = _extract_input_keys(tool_input)

    attrs = {
        "tool_name": tool_name,
        "handoff_key": handoff_key,
    }
    
    # Add detection fields if available
    if input_hash:
        attrs["tool_input_hash"] = input_hash
    if input_keys:
        attrs["tool_input_keys"] = input_keys

    payload: dict[str, Any] = {
        "tool_input": sanitize(tool_input),
    }
    if tool_result is not None:
        payload["tool_result"] = sanitize(tool_result)
    if tool_error:
        payload["error"] = truncate_string(str(tool_error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "tool.call.end",
        "status": status,
        "summary": f"tool result: {tool_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }


# =============================================================================
# LLM Hook Normalization
# =============================================================================


def evt_llm_start(run: "ArzuleRun", context: Any, span_id: str) -> dict[str, Any]:
    """
    Create event for LLM call start.

    Args:
        run: The active ArzuleRun
        context: CrewAI LLM call context
        span_id: The span ID for this LLM call

    Returns:
        TraceEvent dict
    """
    messages = _safe_getattr(context, "messages", [])
    agent = _safe_getattr(context, "agent", None)
    task = _safe_getattr(context, "task", None)

    agent_info = _extract_agent_info(agent)
    task_id, _ = _extract_task_info(task)

    # Summarize messages
    msg_count = len(messages) if isinstance(messages, list) else 0

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "task_id": task_id,
        "event_type": "llm.call.start",
        "status": "ok",
        "summary": f"llm call with {msg_count} messages",
        "attrs_compact": {
            "message_count": msg_count,
        },
        "payload": {
            "messages": _truncate_messages(messages),
        },
    }


def evt_llm_end(run: "ArzuleRun", context: Any, span_id: Optional[str]) -> dict[str, Any]:
    """
    Create event for LLM call end.

    Args:
        run: The active ArzuleRun
        context: CrewAI LLM call context
        span_id: The span ID for this LLM call

    Returns:
        TraceEvent dict
    """
    response = _safe_getattr(context, "response", None)
    agent = _safe_getattr(context, "agent", None)
    task = _safe_getattr(context, "task", None)
    error = _safe_getattr(context, "error", None) or _safe_getattr(context, "exception", None)

    agent_info = _extract_agent_info(agent)
    task_id, _ = _extract_task_info(task)
    status = "error" if error else "ok"

    payload: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    
    if response is not None:
        # Extract content from response object or use as string
        content = _safe_getattr(response, "content", None)
        if content is None:
            content = str(response)
        payload["response"] = truncate_string(str(content), 2000)
        
        # Extract token usage for per-agent tracking
        token_usage = _extract_token_usage(response)
        if token_usage:
            attrs.update(token_usage)

    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "task_id": task_id,
        "event_type": "llm.call.end",
        "status": status,
        "summary": "llm response received",
        "attrs_compact": attrs,
        "payload": payload,
    }


def _truncate_messages(messages: Any, max_messages: int = 10) -> list[dict[str, Any]]:
    """Truncate message list for payload."""
    if not isinstance(messages, list):
        return []

    result = []
    for i, msg in enumerate(messages[:max_messages]):
        if isinstance(msg, dict):
            result.append({
                "role": msg.get("role", "unknown"),
                "content": truncate_string(str(msg.get("content", "")), 500),
            })
        else:
            # Try to extract from object
            result.append({
                "role": _safe_getattr(msg, "role", "unknown"),
                "content": truncate_string(str(_safe_getattr(msg, "content", msg)), 500),
            })

    if len(messages) > max_messages:
        result.append({"_truncated": f"{len(messages) - max_messages} more messages"})

    return result


# =============================================================================
# Async Boundary Events (for async_execution=True tasks)
# =============================================================================


def evt_async_spawn(
    run: "ArzuleRun",
    task_key: str,
    async_id: str,
    parent_span_id: Optional[str],
    task_description: Optional[str] = None,
    agent_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create async.spawn event when parent spawns an async child task.

    The causal_parents links this spawn to the spawning context.

    Args:
        run: The active ArzuleRun
        task_key: Unique identifier for the async task
        async_id: The async correlation ID
        parent_span_id: The parent span that spawned this async task
        task_description: Optional task description
        agent_info: Optional agent info dict

    Returns:
        TraceEvent dict
    """
    span_id = new_span_id()
    causal_parents = [parent_span_id] if parent_span_id else []

    return {
        **_base(
            run,
            span_id=span_id,
            parent_span_id=parent_span_id,
            async_id=async_id,
            causal_parents=causal_parents,
        ),
        "agent": agent_info,
        "event_type": "async.spawn",
        "status": "ok",
        "summary": f"async task spawned: {task_key}",
        "attrs_compact": {
            "async_id": async_id,
            "task_key": task_key,
            "causal_parents": causal_parents,
        },
        "payload": {
            "task_description": truncate_string(task_description, 500) if task_description else None,
        },
    }


def evt_async_join(
    run: "ArzuleRun",
    task_key: str,
    async_id: str,
    parent_span_id: Optional[str],
    status: str = "ok",
    result_summary: Optional[str] = None,
    agent_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create async.join event when an async context completes.

    Args:
        run: The active ArzuleRun
        task_key: The task identifier
        async_id: The async correlation ID
        parent_span_id: The parent span
        status: Completion status (ok or error)
        result_summary: Optional summary of the result
        agent_info: Optional agent info dict

    Returns:
        TraceEvent dict
    """
    span_id = new_span_id()

    return {
        **_base(
            run,
            span_id=span_id,
            parent_span_id=parent_span_id,
            async_id=async_id,
        ),
        "agent": agent_info,
        "event_type": "async.join",
        "status": status,
        "summary": result_summary or f"async task completed: {task_key}",
        "attrs_compact": {
            "async_id": async_id,
            "task_key": task_key,
        },
        "payload": {},
    }


# =============================================================================
# Flow Events (for multi-crew orchestration)
# =============================================================================


def evt_flow_start(
    run: "ArzuleRun",
    flow_name: str,
    span_id: str,
    parent_span_id: Optional[str],
    inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create event for flow start.

    Args:
        run: The active ArzuleRun
        flow_name: Name of the flow
        span_id: The span ID for this flow
        parent_span_id: The parent span (usually root span)
        inputs: Optional flow inputs

    Returns:
        TraceEvent dict
    """
    attrs = {
        "flow_name": flow_name,
    }
    
    payload: dict[str, Any] = {}
    if inputs:
        payload["inputs"] = sanitize(inputs)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "flow.start",
        "status": "ok",
        "summary": f"flow started: {flow_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }


def evt_flow_end(
    run: "ArzuleRun",
    flow_name: str,
    span_id: str,
    parent_span_id: Optional[str],
    status: str = "ok",
    result: Optional[Any] = None,
    state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create event for flow completion.

    Args:
        run: The active ArzuleRun
        flow_name: Name of the flow
        span_id: The span ID for this event
        parent_span_id: The parent span (flow span)
        status: Completion status (ok or error)
        result: Optional flow result
        state: Optional final flow state

    Returns:
        TraceEvent dict
    """
    attrs = {
        "flow_name": flow_name,
    }
    
    payload: dict[str, Any] = {}
    if result is not None:
        payload["result"] = truncate_string(str(result), 1000)
    if state:
        payload["final_state"] = sanitize(state)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "flow.complete",
        "status": status,
        "summary": f"flow completed: {flow_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }


def evt_method_start(
    run: "ArzuleRun",
    flow_name: str,
    method_name: str,
    span_id: str,
    parent_span_id: Optional[str],
    params: Optional[dict[str, Any]] = None,
    state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create event for flow method execution start.

    Args:
        run: The active ArzuleRun
        flow_name: Name of the flow
        method_name: Name of the method being executed
        span_id: The span ID for this method
        parent_span_id: The parent span (flow span)
        params: Optional method parameters
        state: Optional current flow state

    Returns:
        TraceEvent dict
    """
    attrs = {
        "flow_name": flow_name,
        "method_name": method_name,
    }
    
    payload: dict[str, Any] = {}
    if params:
        payload["params"] = sanitize(params)
    if state:
        payload["state"] = sanitize(state)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "flow.method.start",
        "status": "ok",
        "summary": f"method started: {flow_name}.{method_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }


def evt_method_end(
    run: "ArzuleRun",
    flow_name: str,
    method_name: str,
    span_id: str,
    parent_span_id: Optional[str],
    status: str = "ok",
    result: Optional[Any] = None,
    state: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create event for flow method execution completion.

    Args:
        run: The active ArzuleRun
        flow_name: Name of the flow
        method_name: Name of the method that completed
        span_id: The span ID for this event
        parent_span_id: The parent span (method start span)
        status: Completion status (ok or error)
        result: Optional method result
        state: Optional updated flow state
        error: Optional error message if failed

    Returns:
        TraceEvent dict
    """
    attrs = {
        "flow_name": flow_name,
        "method_name": method_name,
    }
    
    payload: dict[str, Any] = {}
    if result is not None:
        payload["result"] = truncate_string(str(result), 1000)
    if state:
        payload["state"] = sanitize(state)
    if error:
        attrs["error"] = truncate_string(error, 200)

    return {
        **_base(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": None,
        "event_type": "flow.method.complete" if status == "ok" else "flow.method.failed",
        "status": status,
        "summary": f"method {'completed' if status == 'ok' else 'failed'}: {flow_name}.{method_name}",
        "attrs_compact": attrs,
        "payload": payload,
    }

