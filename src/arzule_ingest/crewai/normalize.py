"""Normalize CrewAI events and hook contexts to TraceEvent format."""

from __future__ import annotations

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


def _extract_agent_info(agent: Any) -> Optional[dict[str, Any]]:
    """Extract agent info from CrewAI agent object."""
    if not agent:
        return None

    role = _safe_getattr(agent, "role", None) or _safe_getattr(agent, "name", "unknown")
    return {
        "id": f"crewai:role:{role}",
        "role": role,
    }


def _extract_task_info(task: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract task ID and description from CrewAI task."""
    if not task:
        return None, None

    task_id = _safe_getattr(task, "id", None) or _safe_getattr(task, "name", None)
    description = _safe_getattr(task, "description", None)
    return task_id, description


def _base(run: "ArzuleRun", *, span_id: Optional[str], parent_span_id: Optional[str]) -> dict[str, Any]:
    """Build base event fields."""
    return {
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


# =============================================================================
# Event Listener Normalization (Crew/Agent/Task lifecycle events)
# =============================================================================


def evt_from_crewai_event(run: "ArzuleRun", event: Any) -> dict[str, Any]:
    """
    Convert a CrewAI event bus event to a TraceEvent.

    Args:
        run: The active ArzuleRun
        event: CrewAI event object

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
    tool_name = _safe_getattr(event, "tool_name", None)
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
        })

    # Tool info in payload/attrs
    if tool_name:
        attrs["tool_name"] = tool_name
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

    return {
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": new_span_id(),
        "parent_span_id": run.current_parent_span_id(),
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

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "tool.call.start",
        "status": "ok",
        "summary": f"tool call: {tool_name}",
        "attrs_compact": {
            "tool_name": tool_name,
            "handoff_key": handoff_key,
        },
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
        "attrs_compact": {
            "tool_name": tool_name,
            "handoff_key": handoff_key,
        },
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
    if response is not None:
        # Extract content from response object or use as string
        content = _safe_getattr(response, "content", None)
        if content is None:
            content = str(response)
        payload["response"] = truncate_string(str(content), 2000)

    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "task_id": task_id,
        "event_type": "llm.call.end",
        "status": status,
        "summary": "llm response received",
        "attrs_compact": {},
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

