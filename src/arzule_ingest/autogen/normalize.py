"""Normalize AutoGen events to TraceEvent format."""

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
    """Extract agent info from AutoGen agent object."""
    if not agent:
        return None

    name = _safe_getattr(agent, "name", None) or "unknown"
    return {
        "id": f"autogen:agent:{name}",
        "role": name,
    }


def _extract_message_content(message: Any) -> tuple[str, Optional[str]]:
    """
    Extract content from various AutoGen message formats.

    Returns:
        Tuple of (content_str, content_type)
    """
    if message is None:
        return "", None
    
    if isinstance(message, str):
        return message, "text"
    
    if isinstance(message, dict):
        # Standard message dict format
        content = message.get("content", "")
        if isinstance(content, str):
            return content, "text"
        if isinstance(content, list):
            # Multi-modal content (text + images)
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            return " ".join(text_parts), "multimodal"
        return str(content), "unknown"
    
    if isinstance(message, list):
        # List of messages
        return f"[{len(message)} messages]", "list"
    
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
# Message Events
# =============================================================================


def evt_message_send(
    run: "ArzuleRun",
    sender: Any,
    recipient: Any,
    message: Any,
    span_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create event for agent sending a message.

    Args:
        run: The active ArzuleRun
        sender: The sending agent
        recipient: The receiving agent
        message: The message content
        span_id: Optional span ID

    Returns:
        TraceEvent dict
    """
    sender_info = _extract_agent_info(sender)
    recipient_info = _extract_agent_info(recipient)
    content, content_type = _extract_message_content(message)

    sender_name = sender_info["role"] if sender_info else "unknown"
    recipient_name = recipient_info["role"] if recipient_info else "unknown"

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": sender_info,
        "event_type": "agent.message.send",
        "status": "ok",
        "summary": f"{sender_name} -> {recipient_name}: {truncate_string(content, 50)}",
        "attrs_compact": {
            "sender": sender_name,
            "recipient": recipient_name,
            "content_type": content_type,
            "content_length": len(content),
            "framework": "autogen",
        },
        "payload": {
            "content": sanitize(truncate_string(content, 2000)),
            "message": sanitize(message) if isinstance(message, dict) else None,
        },
    }


def evt_message_receive(
    run: "ArzuleRun",
    sender: Any,
    recipient: Any,
    message: Any,
    request_reply: Optional[bool] = None,
    span_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create event for agent receiving a message.

    Args:
        run: The active ArzuleRun
        sender: The sending agent
        recipient: The receiving agent
        message: The message content
        request_reply: Whether a reply is requested
        span_id: Optional span ID

    Returns:
        TraceEvent dict
    """
    sender_info = _extract_agent_info(sender)
    recipient_info = _extract_agent_info(recipient)
    content, content_type = _extract_message_content(message)

    sender_name = sender_info["role"] if sender_info else "unknown"
    recipient_name = recipient_info["role"] if recipient_info else "unknown"

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": recipient_info,
        "event_type": "agent.message.receive",
        "status": "ok",
        "summary": f"{recipient_name} <- {sender_name}: {truncate_string(content, 50)}",
        "attrs_compact": {
            "sender": sender_name,
            "recipient": recipient_name,
            "content_type": content_type,
            "content_length": len(content),
            "request_reply": request_reply,
            "framework": "autogen",
        },
        "payload": {
            "content": sanitize(truncate_string(content, 2000)),
            "message": sanitize(message) if isinstance(message, dict) else None,
        },
    }


# =============================================================================
# LLM Events
# =============================================================================


def evt_llm_start(
    run: "ArzuleRun",
    agent: Any,
    messages: list[dict[str, Any]],
    span_id: str,
) -> dict[str, Any]:
    """
    Create event for LLM call start.

    Args:
        run: The active ArzuleRun
        agent: The agent making the LLM call
        messages: The messages being sent to the LLM
        span_id: The span ID for this LLM call

    Returns:
        TraceEvent dict
    """
    agent_info = _extract_agent_info(agent)
    msg_count = len(messages) if isinstance(messages, list) else 0

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "llm.call.start",
        "status": "ok",
        "summary": f"llm call with {msg_count} messages",
        "attrs_compact": {
            "message_count": msg_count,
            "framework": "autogen",
        },
        "payload": {
            "messages": _truncate_messages(messages),
        },
    }


def evt_llm_end(
    run: "ArzuleRun",
    agent: Any,
    response: Any,
    span_id: Optional[str] = None,
    error: Optional[Exception] = None,
) -> dict[str, Any]:
    """
    Create event for LLM call end.

    Args:
        run: The active ArzuleRun
        agent: The agent that made the LLM call
        response: The LLM response
        span_id: The span ID for this LLM call
        error: Optional exception if the call failed

    Returns:
        TraceEvent dict
    """
    agent_info = _extract_agent_info(agent)
    status = "error" if error else "ok"

    payload: dict[str, Any] = {}

    if response is not None:
        # Handle various response formats
        if hasattr(response, "choices"):
            # OpenAI-style response
            choices = response.choices
            if choices and len(choices) > 0:
                choice = choices[0]
                if hasattr(choice, "message"):
                    content = _safe_getattr(choice.message, "content", "")
                    payload["response"] = truncate_string(str(content), 2000)
                    payload["finish_reason"] = _safe_getattr(choice, "finish_reason", None)
        elif isinstance(response, str):
            payload["response"] = sanitize(truncate_string(response, 2000))
        elif isinstance(response, dict):
            payload["response"] = sanitize(response)
        else:
            payload["response"] = sanitize(truncate_string(str(response), 2000))

        # Extract usage info if available
        if hasattr(response, "usage"):
            usage = response.usage
            payload["usage"] = {
                "prompt_tokens": _safe_getattr(usage, "prompt_tokens", None),
                "completion_tokens": _safe_getattr(usage, "completion_tokens", None),
                "total_tokens": _safe_getattr(usage, "total_tokens", None),
            }

    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "llm.call.end",
        "status": status,
        "summary": "llm response received" if not error else f"llm call failed: {error}",
        "attrs_compact": {
            "framework": "autogen",
        },
        "payload": payload,
    }


# =============================================================================
# Tool/Code Execution Events
# =============================================================================


def evt_tool_start(
    run: "ArzuleRun",
    agent: Any,
    tool_name: str,
    tool_input: Any,
    span_id: str,
) -> dict[str, Any]:
    """
    Create event for tool/function call start.

    Args:
        run: The active ArzuleRun
        agent: The agent executing the tool
        tool_name: Name of the tool/function
        tool_input: Input to the tool
        span_id: The span ID for this tool call

    Returns:
        TraceEvent dict
    """
    agent_info = _extract_agent_info(agent)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "tool.call.start",
        "status": "ok",
        "summary": f"tool call: {tool_name}",
        "attrs_compact": {
            "tool_name": tool_name,
            "framework": "autogen",
        },
        "payload": {
            "tool_input": sanitize(tool_input),
        },
    }


def evt_tool_end(
    run: "ArzuleRun",
    agent: Any,
    tool_name: str,
    tool_output: Any,
    span_id: Optional[str] = None,
    error: Optional[Exception] = None,
) -> dict[str, Any]:
    """
    Create event for tool/function call end.

    Args:
        run: The active ArzuleRun
        agent: The agent that executed the tool
        tool_name: Name of the tool/function
        tool_output: Output from the tool
        span_id: The span ID for this tool call
        error: Optional exception if the call failed

    Returns:
        TraceEvent dict
    """
    agent_info = _extract_agent_info(agent)
    status = "error" if error else "ok"

    payload: dict[str, Any] = {}
    if tool_output is not None:
        payload["tool_output"] = sanitize(tool_output)
    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "tool.call.end",
        "status": status,
        "summary": f"tool result: {tool_name}" if not error else f"tool error: {tool_name}",
        "attrs_compact": {
            "tool_name": tool_name,
            "framework": "autogen",
        },
        "payload": payload,
    }


def evt_code_execution(
    run: "ArzuleRun",
    agent: Any,
    code: str,
    output: str,
    exit_code: int,
    span_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create event for code execution.

    Args:
        run: The active ArzuleRun
        agent: The agent executing code
        code: The code that was executed
        output: Execution output
        exit_code: Process exit code
        span_id: Optional span ID

    Returns:
        TraceEvent dict
    """
    agent_info = _extract_agent_info(agent)
    status = "ok" if exit_code == 0 else "error"

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": agent_info,
        "event_type": "code.execution",
        "status": status,
        "summary": f"code execution (exit={exit_code})",
        "attrs_compact": {
            "exit_code": exit_code,
            "code_length": len(code),
            "output_length": len(output),
            "framework": "autogen",
        },
        "payload": {
            "code": sanitize(truncate_string(code, 2000)),
            "output": sanitize(truncate_string(output, 2000)),
        },
    }


# =============================================================================
# Conversation Events
# =============================================================================


def evt_conversation_start(
    run: "ArzuleRun",
    initiator: Any,
    participants: list[Any],
    initial_message: Any,
    span_id: str,
) -> dict[str, Any]:
    """
    Create event for conversation/chat start.

    Args:
        run: The active ArzuleRun
        initiator: The agent that started the conversation
        participants: List of participating agents
        initial_message: The initial message
        span_id: The span ID for this conversation

    Returns:
        TraceEvent dict
    """
    initiator_info = _extract_agent_info(initiator)
    participant_names = [
        _safe_getattr(p, "name", "unknown") for p in participants
    ]
    content, _ = _extract_message_content(initial_message)

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": initiator_info,
        "event_type": "conversation.start",
        "status": "ok",
        "summary": f"conversation started: {participant_names}",
        "attrs_compact": {
            "initiator": initiator_info["role"] if initiator_info else "unknown",
            "participant_count": len(participants),
            "participants": participant_names[:10],  # Limit for display
            "framework": "autogen",
        },
        "payload": {
            "initial_message": sanitize(truncate_string(content, 500)),
        },
    }


def evt_conversation_end(
    run: "ArzuleRun",
    initiator: Any,
    result: Any,
    message_count: int,
    span_id: Optional[str] = None,
    status: str = "ok",
) -> dict[str, Any]:
    """
    Create event for conversation/chat end.

    Args:
        run: The active ArzuleRun
        initiator: The agent that started the conversation
        result: The final result/output
        message_count: Total messages exchanged
        span_id: The span ID for this conversation
        status: Completion status

    Returns:
        TraceEvent dict
    """
    initiator_info = _extract_agent_info(initiator)

    payload: dict[str, Any] = {
        "message_count": message_count,
    }

    if result is not None:
        if hasattr(result, "summary"):
            payload["summary"] = truncate_string(str(result.summary), 1000)
        elif hasattr(result, "chat_history"):
            payload["final_message_count"] = len(result.chat_history)
        else:
            payload["result"] = sanitize(truncate_string(str(result), 1000))

    return {
        **_base(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": initiator_info,
        "event_type": "conversation.end",
        "status": status,
        "summary": f"conversation ended ({message_count} messages)",
        "attrs_compact": {
            "message_count": message_count,
            "framework": "autogen",
        },
        "payload": payload,
    }


# =============================================================================
# Helpers
# =============================================================================


def _truncate_messages(messages: Any, max_messages: int = 10) -> list[dict[str, Any]]:
    """Truncate and sanitize message list for payload."""
    if not isinstance(messages, list):
        return []

    result = []
    for msg in messages[:max_messages]:
        if isinstance(msg, dict):
            result.append({
                "role": msg.get("role", "unknown"),
                # Apply sanitization to redact secrets and PII from message content
                "content": sanitize(truncate_string(str(msg.get("content", "")), 500)),
            })
        else:
            # Try to extract from object
            result.append({
                "role": _safe_getattr(msg, "role", "unknown"),
                "content": sanitize(truncate_string(str(_safe_getattr(msg, "content", msg)), 500)),
            })

    if len(messages) > max_messages:
        result.append({"_truncated": f"{len(messages) - max_messages} more messages"})

    return result











