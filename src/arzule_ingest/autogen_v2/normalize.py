"""Normalize AutoGen v0.7+ events to TraceEvent format."""

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
    """Extract agent info from AutoGen v0.7+ agent object."""
    if not agent:
        return None

    name = _safe_getattr(agent, "name", None) or "unknown"
    description = _safe_getattr(agent, "description", None) or ""
    
    return {
        "id": f"autogen_v2:agent:{name}",
        "role": name,
        "description": description,
    }


def _extract_message_content(message: Any) -> tuple[str, Optional[str]]:
    """
    Extract content from AutoGen v0.7+ message formats.

    Returns:
        Tuple of (content_str, content_type)
    """
    if message is None:
        return "", None
    
    # Check if it's a BaseChatMessage
    if hasattr(message, 'to_text'):
        try:
            content = message.to_text()
            # Determine message type
            msg_type = type(message).__name__
            return content, msg_type
        except Exception:
            pass
    
    # Check if it has content attribute
    if hasattr(message, 'content'):
        content = message.content
        if isinstance(content, str):
            return content, "text"
        return str(content), "unknown"
    
    # Fallback to string representation
    return str(message), "unknown"


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
# Agent Lifecycle Events
# =============================================================================


def evt_agent_start(
    run: "ArzuleRun",
    agent: Any,
    messages: Any,
    span_id: str,
) -> dict[str, Any]:
    """
    Create event for agent starting message processing.

    Args:
        run: The active ArzuleRun
        agent: The agent processing messages
        messages: Input messages
        span_id: The span ID for this agent invocation

    Returns:
        TraceEvent dict
    """
    agent_info = _extract_agent_info(agent)
    agent_name = agent_info["role"] if agent_info else "unknown"
    msg_count = len(messages) if hasattr(messages, '__len__') else 0

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "agent.start",
        "status": "ok",
        "summary": f"{agent_name} processing {msg_count} message(s)",
        "attrs_compact": {
            "agent_name": agent_name,
            "message_count": msg_count,
            "framework": "autogen_v2",
        },
        "payload": {
            "message_count": msg_count,
        },
    }


def evt_agent_end(
    run: "ArzuleRun",
    agent: Any,
    response: Any,
    span_id: str,
    error: Optional[Exception] = None,
) -> dict[str, Any]:
    """
    Create event for agent finishing message processing.

    Args:
        run: The active ArzuleRun
        agent: The agent that processed messages
        response: The response from the agent
        span_id: The span ID for this agent invocation
        error: Optional exception if the call failed

    Returns:
        TraceEvent dict
    """
    agent_info = _extract_agent_info(agent)
    agent_name = agent_info["role"] if agent_info else "unknown"
    status = "error" if error else "ok"

    payload: dict[str, Any] = {}
    
    if response and hasattr(response, 'chat_message'):
        content, _ = _extract_message_content(response.chat_message)
        payload["response_preview"] = sanitize(truncate_string(content, 200))
    
    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "agent.end",
        "status": status,
        "summary": f"{agent_name} completed" if not error else f"{agent_name} failed: {error}",
        "attrs_compact": {
            "agent_name": agent_name,
            "framework": "autogen_v2",
        },
        "payload": payload,
    }


# =============================================================================
# Message Events
# =============================================================================


def evt_chat_message(
    run: "ArzuleRun",
    agent: Any,
    message: Any,
    direction: str,
    span_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create event for a chat message.

    Args:
        run: The active ArzuleRun
        agent: The agent sending/receiving the message
        message: The message content
        direction: "incoming" or "outgoing"
        span_id: Optional span ID

    Returns:
        TraceEvent dict
    """
    agent_info = _extract_agent_info(agent)
    agent_name = agent_info["role"] if agent_info else "unknown"
    content, msg_type = _extract_message_content(message)
    
    # Extract source if available
    source = _safe_getattr(message, 'source', agent_name)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": f"agent.message.{direction}",
        "status": "ok",
        "summary": f"{agent_name} {direction}: {sanitize(truncate_string(content, 50))}",
        "attrs_compact": {
            "agent_name": agent_name,
            "source": source,
            "message_type": msg_type,
            "direction": direction,
            "content_length": len(content),
            "framework": "autogen_v2",
        },
        "payload": {
            "content": sanitize(truncate_string(content, 2000)),
            "message_type": msg_type,
        },
    }


def evt_message_event(
    run: "ArzuleRun",
    agent: Any,
    event: Any,
    span_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create event for an agent event (non-chat message).

    Args:
        run: The active ArzuleRun
        agent: The agent that produced the event
        event: The event object
        span_id: Optional span ID

    Returns:
        TraceEvent dict
    """
    agent_info = _extract_agent_info(agent)
    agent_name = agent_info["role"] if agent_info else "unknown"
    
    # Extract event information
    event_type_name = type(event).__name__
    content, _ = _extract_message_content(event)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "agent.event",
        "status": "ok",
        "summary": f"{agent_name} event: {event_type_name}",
        "attrs_compact": {
            "agent_name": agent_name,
            "event_type_name": event_type_name,
            "framework": "autogen_v2",
        },
        "payload": {
            "event_type": event_type_name,
            "content": sanitize(truncate_string(str(content), 1000)),
        },
    }


# =============================================================================
# Model Call Events
# =============================================================================


def evt_model_call_start(
    run: "ArzuleRun",
    model_client: Any,
    messages: Any,
    span_id: str,
) -> dict[str, Any]:
    """
    Create event for model call start.

    Args:
        run: The active ArzuleRun
        model_client: The model client making the call
        messages: The messages being sent to the model
        span_id: The span ID for this model call

    Returns:
        TraceEvent dict
    """
    model_type = type(model_client).__name__
    msg_count = len(messages) if hasattr(messages, '__len__') else 0

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": None,
        "event_type": "llm.call.start",
        "status": "ok",
        "summary": f"model call ({model_type}) with {msg_count} messages",
        "attrs_compact": {
            "model_type": model_type,
            "message_count": msg_count,
            "framework": "autogen_v2",
        },
        "payload": {
            "message_count": msg_count,
        },
    }


def evt_model_call_end(
    run: "ArzuleRun",
    model_client: Any,
    response: Any,
    span_id: str,
    error: Optional[Exception] = None,
) -> dict[str, Any]:
    """
    Create event for model call end.

    Args:
        run: The active ArzuleRun
        model_client: The model client that made the call
        response: The model response
        span_id: The span ID for this model call
        error: Optional exception if the call failed

    Returns:
        TraceEvent dict
    """
    model_type = type(model_client).__name__
    status = "error" if error else "ok"

    payload: dict[str, Any] = {}

    if response and not error:
        # Extract content from CreateResult
        if hasattr(response, 'content'):
            if isinstance(response.content, str):
                payload["response"] = sanitize(truncate_string(response.content, 2000))
            else:
                payload["response"] = sanitize(response.content)
        
        # Extract usage info
        if hasattr(response, 'usage'):
            usage = response.usage
            if usage:
                payload["usage"] = {
                    "prompt_tokens": _safe_getattr(usage, "prompt_tokens", None),
                    "completion_tokens": _safe_getattr(usage, "completion_tokens", None),
                }
        
        # Extract finish reason
        if hasattr(response, 'finish_reason'):
            payload["finish_reason"] = response.finish_reason

    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": None,
        "event_type": "llm.call.end",
        "status": status,
        "summary": f"model response ({model_type})" if not error else f"model error: {error}",
        "attrs_compact": {
            "model_type": model_type,
            "framework": "autogen_v2",
        },
        "payload": payload,
    }


# =============================================================================
# Tool Call Events
# =============================================================================


def evt_tool_call_start(
    run: "ArzuleRun",
    tool: Any,
    tool_name: str,
    tool_input: Any,
    span_id: str,
) -> dict[str, Any]:
    """
    Create event for tool call start.

    Args:
        run: The active ArzuleRun
        tool: The tool being executed
        tool_name: Name of the tool
        tool_input: Input to the tool
        span_id: The span ID for this tool call

    Returns:
        TraceEvent dict
    """
    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": None,
        "event_type": "tool.call.start",
        "status": "ok",
        "summary": f"tool call: {tool_name}",
        "attrs_compact": {
            "tool_name": tool_name,
            "framework": "autogen_v2",
        },
        "payload": {
            "tool_input": sanitize(tool_input),
        },
    }


def evt_tool_call_end(
    run: "ArzuleRun",
    tool: Any,
    tool_name: str,
    tool_output: Any,
    span_id: str,
    error: Optional[Exception] = None,
) -> dict[str, Any]:
    """
    Create event for tool call end.

    Args:
        run: The active ArzuleRun
        tool: The tool that was executed
        tool_name: Name of the tool
        tool_output: Output from the tool
        span_id: The span ID for this tool call
        error: Optional exception if the call failed

    Returns:
        TraceEvent dict
    """
    status = "error" if error else "ok"

    payload: dict[str, Any] = {}
    if tool_output is not None:
        # Handle ToolResult type
        if hasattr(tool_output, 'content'):
            payload["tool_output"] = sanitize(truncate_string(str(tool_output.content), 2000))
        else:
            payload["tool_output"] = sanitize(tool_output)
    
    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": None,
        "event_type": "tool.call.end",
        "status": status,
        "summary": f"tool result: {tool_name}" if not error else f"tool error: {tool_name}",
        "attrs_compact": {
            "tool_name": tool_name,
            "framework": "autogen_v2",
        },
        "payload": payload,
    }










