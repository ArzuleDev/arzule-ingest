"""Normalize Agent SDK events to TraceEvent format.

Converts Anthropic Agent SDK events and internal events to the standard TraceEvent
schema used by the Arzule backend. Supports Claude Agent SDK tool calls, prompts,
messages, and session lifecycle events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from ..ids import new_span_id
from ..sanitize import sanitize, truncate_string

if TYPE_CHECKING:
    from ..run import ArzuleRun


# =============================================================================
# Helper Functions
# =============================================================================


def _make_agent_id(session_id: str, subagent_type: Optional[str] = None) -> str:
    """Generate an agent ID for Claude Agent SDK.

    Args:
        session_id: The Agent SDK session identifier.
        subagent_type: Optional subagent type for specialized agents.

    Returns:
        Agent ID in format "claude_agent_sdk:main:{session_id}" or
        "claude_agent_sdk:subagent:{type}:{session_id}" for subagents.
    """
    if subagent_type:
        return f"claude_agent_sdk:subagent:{subagent_type}:{session_id}"
    return f"claude_agent_sdk:main:{session_id}"


def _truncate_for_summary(text: str, max_len: int = 100) -> str:
    """Truncate text for use in event summaries.

    Args:
        text: The text to truncate.
        max_len: Maximum length for the summary text.

    Returns:
        Truncated text with ellipsis if needed.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitize payload by removing sensitive data.

    Applies standard sanitization to redact secrets, PII, and truncate
    large values while preserving structure for debugging.

    Args:
        payload: The payload dict to sanitize.

    Returns:
        Sanitized payload dict safe for storage and display.
    """
    return sanitize(payload) if payload else {}


def _extract_tool_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Generate human-readable summary for a tool call.

    Args:
        tool_name: Name of the tool being called.
        tool_input: Tool input parameters.

    Returns:
        Human-readable summary of the tool call.
    """
    if not tool_input:
        return f"Calling {tool_name}"

    # Handle common Claude tools
    if tool_name == "computer":
        action = tool_input.get("action", "unknown")
        return f"Computer: {action}"

    if tool_name == "bash":
        command = tool_input.get("command", "")
        return f"Bash: {_truncate_for_summary(command, 80)}"

    if tool_name == "text_editor":
        command = tool_input.get("command", "unknown")
        path = tool_input.get("path", "")
        if path:
            return f"Editor {command}: {_truncate_for_summary(path, 60)}"
        return f"Editor: {command}"

    if tool_name == "str_replace_editor":
        command = tool_input.get("command", "unknown")
        path = tool_input.get("path", "")
        if path:
            return f"Editor {command}: {_truncate_for_summary(path, 60)}"
        return f"Editor: {command}"

    # MCP tools
    if tool_name.startswith("mcp_"):
        parts = tool_name.split("_", 2)
        if len(parts) >= 3:
            server = parts[1]
            method = parts[2]
            return f"MCP {server}: {method}"

    # Generic fallback
    return f"Calling {tool_name}"


def _base_event(
    run: "ArzuleRun",
    *,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build base event fields common to all trace events.

    Args:
        run: The active ArzuleRun context.
        span_id: Optional explicit span ID (generated if not provided).
        parent_span_id: Optional parent span ID for hierarchy.

    Returns:
        Dict with common event fields populated.
    """
    return {
        "schema_version": "trace_event.v0_1",
        "run_id": run.run_id,
        "tenant_id": run.tenant_id,
        "project_id": run.project_id,
        "trace_id": run.trace_id,
        "span_id": span_id or new_span_id(),
        "parent_span_id": parent_span_id or run.current_parent_span_id(),
        "seq": run.next_seq(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "workstream_id": None,
        "task_id": None,
        "raw_ref": {"storage": "inline"},
    }


# =============================================================================
# Tool Event Normalization
# =============================================================================


def normalize_tool_start(
    input_data: dict[str, Any],
    tool_use_id: str,
    run: "ArzuleRun",
    *,
    session_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize a tool call start event from Agent SDK.

    Args:
        input_data: Tool input data containing tool_name and tool_input.
        tool_use_id: Unique identifier for this tool use.
        run: The active ArzuleRun context.
        session_id: Optional session ID for agent identification.
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        model: Optional model identifier.

    Returns:
        Normalized TraceEvent dict for tool.call.start.
    """
    tool_name = input_data.get("tool_name", "unknown_tool")
    tool_input = input_data.get("tool_input", {})

    # Generate summary
    summary = _extract_tool_summary(tool_name, tool_input)

    # Build agent info
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id or run.run_id),
        "role": "main",
    }
    if model:
        agent_info["model"] = model

    # Build attrs
    attrs: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
    }

    # Build payload with sanitized input
    payload: dict[str, Any] = {
        "tool_input": _sanitize_payload(tool_input) if isinstance(tool_input, dict) else sanitize(tool_input),
    }

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": "tool.call.start",
        "status": "ok",
        "summary": summary,
        "attrs_compact": attrs,
        "payload": payload,
    }


def normalize_tool_end(
    input_data: dict[str, Any],
    tool_use_id: str,
    run: "ArzuleRun",
    duration_ms: float,
    *,
    session_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize a tool call end event from Agent SDK.

    Args:
        input_data: Tool data containing tool_name, tool_input, and tool_output.
        tool_use_id: Unique identifier for this tool use.
        run: The active ArzuleRun context.
        duration_ms: Duration of the tool call in milliseconds.
        session_id: Optional session ID for agent identification.
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        model: Optional model identifier.
        error: Optional error message if tool call failed.

    Returns:
        Normalized TraceEvent dict for tool.call.end.
    """
    tool_name = input_data.get("tool_name", "unknown_tool")
    tool_input = input_data.get("tool_input", {})
    tool_output = input_data.get("tool_output")

    # Determine status
    status = "error" if error else "ok"
    summary = f"{tool_name} {'failed' if error else 'completed'}"

    # Build agent info
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id or run.run_id),
        "role": "main",
    }
    if model:
        agent_info["model"] = model

    # Build attrs
    attrs: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "duration_ms": round(duration_ms, 2),
    }

    # Build payload
    payload: dict[str, Any] = {
        "tool_input": _sanitize_payload(tool_input) if isinstance(tool_input, dict) else sanitize(tool_input),
    }

    if tool_output is not None:
        # Truncate large outputs while preserving useful info
        if isinstance(tool_output, str):
            payload["tool_output"] = truncate_string(sanitize(tool_output), 20000)
        elif isinstance(tool_output, dict):
            payload["tool_output"] = _sanitize_payload(tool_output)
        else:
            payload["tool_output"] = sanitize(tool_output)

    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": "tool.call.end",
        "status": status,
        "summary": summary,
        "attrs_compact": attrs,
        "payload": payload,
    }


# =============================================================================
# Prompt/Message Event Normalization
# =============================================================================


def normalize_prompt(
    input_data: dict[str, Any],
    run: "ArzuleRun",
    *,
    session_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize a user prompt event from Agent SDK.

    Args:
        input_data: Prompt data containing the user message.
        run: The active ArzuleRun context.
        session_id: Optional session ID for agent identification.
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        model: Optional model identifier.

    Returns:
        Normalized TraceEvent dict for prompt event.
    """
    # Extract prompt content
    prompt_text = input_data.get("prompt", input_data.get("content", ""))
    if isinstance(prompt_text, list):
        # Handle multi-part prompts
        parts = []
        for part in prompt_text:
            if isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            else:
                parts.append(str(part))
        prompt_text = " ".join(parts)

    # Generate summary
    summary = f"User prompt: {_truncate_for_summary(prompt_text)}"

    # Build agent info - prompts come from user, not agent
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id or run.run_id),
        "role": "main",
    }
    if model:
        agent_info["model"] = model

    # Build attrs
    attrs: dict[str, Any] = {
        "prompt_length": len(prompt_text) if isinstance(prompt_text, str) else 0,
    }

    # Build payload
    payload: dict[str, Any] = {
        "prompt": sanitize(truncate_string(str(prompt_text), 10000)),
    }

    # Include any additional context from input_data
    if "system_prompt" in input_data:
        payload["system_prompt"] = sanitize(truncate_string(input_data["system_prompt"], 5000))

    if "context" in input_data:
        payload["context"] = _sanitize_payload(input_data["context"]) if isinstance(input_data["context"], dict) else sanitize(input_data["context"])

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": "prompt",
        "status": "ok",
        "summary": summary,
        "attrs_compact": attrs,
        "payload": payload,
    }


def normalize_message(
    msg: dict[str, Any],
    run: "ArzuleRun",
    *,
    session_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize an assistant message event from Agent SDK.

    Args:
        msg: Message data containing role, content, and optional metadata.
        run: The active ArzuleRun context.
        session_id: Optional session ID for agent identification.
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        model: Optional model identifier from message.

    Returns:
        Normalized TraceEvent dict for message event.
    """
    role = msg.get("role", "assistant")
    content = msg.get("content", "")

    # Handle content blocks (Claude API format)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    text_parts.append(f"[Tool: {block.get('name', 'unknown')}]")
            else:
                text_parts.append(str(block))
        content = " ".join(text_parts)

    # Extract model from message if present
    msg_model = msg.get("model") or model

    # Generate summary
    if role == "assistant":
        summary = f"Assistant: {_truncate_for_summary(str(content))}"
    elif role == "user":
        summary = f"User: {_truncate_for_summary(str(content))}"
    else:
        summary = f"{role.capitalize()}: {_truncate_for_summary(str(content))}"

    # Build agent info
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id or run.run_id),
        "role": "main",
    }
    if msg_model:
        agent_info["model"] = msg_model

    # Build attrs
    attrs: dict[str, Any] = {
        "message_role": role,
    }

    # Add token usage if present
    usage = msg.get("usage", {})
    if usage:
        if "input_tokens" in usage:
            attrs["prompt_tokens"] = usage["input_tokens"]
        if "output_tokens" in usage:
            attrs["completion_tokens"] = usage["output_tokens"]
        if "input_tokens" in usage and "output_tokens" in usage:
            attrs["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

    # Add stop reason if present
    stop_reason = msg.get("stop_reason")
    if stop_reason:
        attrs["stop_reason"] = stop_reason

    # Build payload
    payload: dict[str, Any] = {
        "content": sanitize(truncate_string(str(content), 10000)),
        "role": role,
    }

    # Include raw content blocks if available
    raw_content = msg.get("content")
    if isinstance(raw_content, list):
        # Store truncated version of content blocks
        payload["content_blocks"] = sanitize(raw_content[:20])  # Limit to 20 blocks

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": "message",
        "status": "ok",
        "summary": summary,
        "attrs_compact": attrs,
        "payload": payload,
    }


def normalize_result(
    result_msg: dict[str, Any],
    run: "ArzuleRun",
    *,
    session_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
    status: str = "ok",
) -> dict[str, Any]:
    """Normalize a final result/response event from Agent SDK.

    Args:
        result_msg: Result message containing final output and metadata.
        run: The active ArzuleRun context.
        session_id: Optional session ID for agent identification.
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        model: Optional model identifier.
        status: Result status (ok or error).

    Returns:
        Normalized TraceEvent dict for result event.
    """
    # Extract result content
    content = result_msg.get("content", result_msg.get("result", ""))
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        content = " ".join(text_parts)

    # Extract model
    result_model = result_msg.get("model") or model

    # Generate summary
    summary = f"Result: {_truncate_for_summary(str(content))}"

    # Build agent info
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id or run.run_id),
        "role": "main",
    }
    if result_model:
        agent_info["model"] = result_model

    # Build attrs
    attrs: dict[str, Any] = {}

    # Add token usage if present
    usage = result_msg.get("usage", {})
    if usage:
        if "input_tokens" in usage:
            attrs["prompt_tokens"] = usage["input_tokens"]
        if "output_tokens" in usage:
            attrs["completion_tokens"] = usage["output_tokens"]
        if "input_tokens" in usage and "output_tokens" in usage:
            attrs["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

    # Add stop reason
    stop_reason = result_msg.get("stop_reason")
    if stop_reason:
        attrs["stop_reason"] = stop_reason

    # Build payload
    payload: dict[str, Any] = {
        "result": sanitize(truncate_string(str(content), 10000)),
    }

    # Include error if status is error
    error = result_msg.get("error")
    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": "result",
        "status": status,
        "summary": summary,
        "attrs_compact": attrs,
        "payload": payload,
    }


# =============================================================================
# Session Lifecycle Event Normalization
# =============================================================================


def normalize_session_start(
    session_id: str,
    run: "ArzuleRun",
    *,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
    tools: Optional[list[str]] = None,
    system_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize a session start event for Agent SDK.

    Args:
        session_id: The Agent SDK session identifier.
        run: The active ArzuleRun context.
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        model: Optional model identifier.
        tools: Optional list of available tool names.
        system_prompt: Optional system prompt for the session.

    Returns:
        Normalized TraceEvent dict for session.start.
    """
    # Build agent info
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id),
        "role": "main",
    }
    if model:
        agent_info["model"] = model
    if tools:
        agent_info["tools"] = tools

    # Build attrs
    attrs: dict[str, Any] = {
        "session_id": session_id,
    }
    if tools:
        attrs["tool_count"] = len(tools)

    # Build payload
    payload: dict[str, Any] = {}
    if system_prompt:
        payload["system_prompt"] = sanitize(truncate_string(system_prompt, 5000))
    if tools:
        payload["tools"] = tools

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": "session.start",
        "status": "ok",
        "summary": f"Agent SDK session started: {session_id[:8]}...",
        "attrs_compact": attrs,
        "payload": payload,
    }


def normalize_session_end(
    session_id: str,
    run: "ArzuleRun",
    status: str,
    *,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
    error: Optional[str] = None,
    total_tokens: Optional[int] = None,
    duration_ms: Optional[float] = None,
) -> dict[str, Any]:
    """Normalize a session end event for Agent SDK.

    Args:
        session_id: The Agent SDK session identifier.
        run: The active ArzuleRun context.
        status: Session completion status (ok, error, cancelled).
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        model: Optional model identifier.
        error: Optional error message if session failed.
        total_tokens: Optional total tokens used in session.
        duration_ms: Optional session duration in milliseconds.

    Returns:
        Normalized TraceEvent dict for session.end.
    """
    # Build agent info
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id),
        "role": "main",
    }
    if model:
        agent_info["model"] = model

    # Build attrs
    attrs: dict[str, Any] = {
        "session_id": session_id,
    }
    if total_tokens is not None:
        attrs["total_tokens"] = total_tokens
    if duration_ms is not None:
        attrs["duration_ms"] = round(duration_ms, 2)

    # Build payload
    payload: dict[str, Any] = {}
    if error:
        payload["error"] = truncate_string(str(error), 500)

    # Generate summary
    if status == "ok":
        summary = f"Agent SDK session completed: {session_id[:8]}..."
    elif status == "error":
        summary = f"Agent SDK session failed: {session_id[:8]}..."
    else:
        summary = f"Agent SDK session ended ({status}): {session_id[:8]}..."

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": "session.end",
        "status": status,
        "summary": summary,
        "attrs_compact": attrs,
        "payload": payload,
    }


# =============================================================================
# LLM Call Event Normalization
# =============================================================================


def normalize_llm_start(
    run: "ArzuleRun",
    *,
    session_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
    messages: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Normalize an LLM call start event from Agent SDK.

    Args:
        run: The active ArzuleRun context.
        session_id: Optional session ID for agent identification.
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        model: Optional model identifier.
        messages: Optional list of messages being sent to the LLM.

    Returns:
        Normalized TraceEvent dict for llm.call.start.
    """
    # Build agent info
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id or run.run_id),
        "role": "main",
    }
    if model:
        agent_info["model"] = model

    # Build attrs
    message_count = len(messages) if messages else 0
    attrs: dict[str, Any] = {
        "message_count": message_count,
    }
    if model:
        attrs["llm_model"] = model

    # Build payload with truncated messages
    payload: dict[str, Any] = {}
    if messages:
        truncated_messages = []
        for msg in messages[:10]:  # Limit to 10 messages
            truncated_msg = {
                "role": msg.get("role", "unknown"),
                "content": sanitize(truncate_string(str(msg.get("content", "")), 500)),
            }
            truncated_messages.append(truncated_msg)
        if len(messages) > 10:
            truncated_messages.append({"_truncated": f"{len(messages) - 10} more messages"})
        payload["messages"] = truncated_messages

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": "llm.call.start",
        "status": "ok",
        "summary": f"LLM call with {message_count} messages",
        "attrs_compact": attrs,
        "payload": payload,
    }


def normalize_llm_end(
    run: "ArzuleRun",
    *,
    session_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
    response: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
) -> dict[str, Any]:
    """Normalize an LLM call end event from Agent SDK.

    Args:
        run: The active ArzuleRun context.
        session_id: Optional session ID for agent identification.
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        model: Optional model identifier.
        response: Optional response data from the LLM.
        error: Optional error message if LLM call failed.
        duration_ms: Optional call duration in milliseconds.

    Returns:
        Normalized TraceEvent dict for llm.call.end.
    """
    status = "error" if error else "ok"

    # Build agent info
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id or run.run_id),
        "role": "main",
    }
    if model:
        agent_info["model"] = model

    # Build attrs
    attrs: dict[str, Any] = {}
    if model:
        attrs["llm_model"] = model
    if duration_ms is not None:
        attrs["duration_ms"] = round(duration_ms, 2)

    # Extract token usage from response
    if response:
        usage = response.get("usage", {})
        if "input_tokens" in usage:
            attrs["prompt_tokens"] = usage["input_tokens"]
        if "output_tokens" in usage:
            attrs["completion_tokens"] = usage["output_tokens"]
        if "input_tokens" in usage and "output_tokens" in usage:
            attrs["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        stop_reason = response.get("stop_reason")
        if stop_reason:
            attrs["stop_reason"] = stop_reason

    # Build payload
    payload: dict[str, Any] = {}
    if response:
        content = response.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = " ".join(text_parts)
        payload["response"] = sanitize(truncate_string(str(content), 2000))

    if error:
        payload["error"] = truncate_string(str(error), 500)

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": "llm.call.end",
        "status": status,
        "summary": "LLM response received" if not error else "LLM call failed",
        "attrs_compact": attrs,
        "payload": payload,
    }


# =============================================================================
# Generic Event Normalization
# =============================================================================


def normalize_event(
    run: "ArzuleRun",
    *,
    event_type: str,
    session_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    status: str = "ok",
    summary: str = "",
    agent_role: str = "main",
    subagent_type: Optional[str] = None,
    model: Optional[str] = None,
    tools: Optional[list[str]] = None,
    attrs: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Normalize a generic Agent SDK event to TraceEvent format.

    This is a flexible normalization function for custom event types
    that don't fit the specialized normalizers above.

    Args:
        run: The active ArzuleRun context.
        event_type: Type of event (e.g., "custom.event").
        session_id: Optional session ID for agent identification.
        span_id: Optional explicit span ID.
        parent_span_id: Optional parent span ID.
        status: Event status (ok, error).
        summary: Human-readable event summary.
        agent_role: Agent role (main, or subagent type).
        subagent_type: Optional subagent type for specialized agents.
        model: Optional model identifier.
        tools: Optional list of available tools.
        attrs: Optional additional attributes.
        payload: Optional event payload.

    Returns:
        Normalized TraceEvent dict.
    """
    # Build agent info
    agent_info: dict[str, Any] = {
        "id": _make_agent_id(session_id or run.run_id, subagent_type),
        "role": agent_role,
    }
    if subagent_type:
        agent_info["subagent_type"] = subagent_type
    if model:
        agent_info["model"] = model
    if tools:
        agent_info["tools"] = tools

    return {
        **_base_event(run, span_id=span_id, parent_span_id=parent_span_id),
        "agent": agent_info,
        "event_type": event_type,
        "status": status,
        "summary": truncate_string(summary, 200),
        "attrs_compact": _sanitize_payload(attrs) if attrs else {},
        "payload": _sanitize_payload(payload) if payload else {},
    }
