"""Normalize Claude Code events to TraceEvent format.

Converts hook input data and internal events to the standard TraceEvent
schema used by the Arzule backend.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from ..ids import new_span_id
from ..sanitize import sanitize, truncate_string


def normalize_event(
    run: Any,
    *,
    event_type: str,
    agent_id: str,
    agent_role: str,
    summary: str,
    status: str = "ok",
    attrs: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    subagent_type: Optional[str] = None,
    tools: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Normalize a Claude Code event to TraceEvent format.

    Args:
        run: The active ArzuleRun
        event_type: Type of event (session.start, tool.call.start, etc.)
        agent_id: Full agent ID (claude_code:main:xxx or claude_code:subagent:type:id)
        agent_role: Agent role (main, Explore, Plan, Bash, etc.)
        summary: Human-readable event summary
        status: Event status (ok, error)
        attrs: Additional attributes for attrs_compact
        payload: Event payload data
        span_id: Optional explicit span ID
        parent_span_id: Optional parent span ID
        subagent_type: Subagent type if applicable
        tools: List of tools available to the agent
        model: Model identifier if known

    Returns:
        TraceEvent dict ready for emission
    """
    # Build agent info
    agent_info: dict[str, Any] = {
        "id": agent_id,
        "role": agent_role,
    }

    if subagent_type:
        agent_info["subagent_type"] = subagent_type

    if tools:
        agent_info["tools"] = tools

    if model:
        agent_info["model"] = model

    # Build event
    event = {
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
        "agent": agent_info,
        "task_id": None,
        "event_type": event_type,
        "status": status,
        "summary": truncate_string(summary, 200),
        "attrs_compact": sanitize(attrs) if attrs else {},
        "payload": sanitize(payload) if payload else {},
        "raw_ref": {"storage": "inline"},
    }

    return event


def normalize_tool_start(
    run: Any,
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_use_id: str,
    agent_id: str,
    agent_role: str,
    span_id: str,
    parent_span_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Normalize a tool call start event.

    Args:
        run: The active ArzuleRun
        tool_name: Name of the tool being called
        tool_input: Tool input parameters
        tool_use_id: Unique tool use identifier
        agent_id: Current agent ID
        agent_role: Current agent role
        span_id: Span ID for this tool call
        parent_span_id: Optional parent span

    Returns:
        TraceEvent dict
    """
    # Build summary based on tool
    summary = _build_tool_summary(tool_name, tool_input)

    return normalize_event(
        run,
        event_type="tool.call.start",
        agent_id=agent_id,
        agent_role=agent_role,
        summary=summary,
        span_id=span_id,
        parent_span_id=parent_span_id,
        attrs={
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
        },
        payload={
            "tool_input": _sanitize_tool_input(tool_name, tool_input),
        },
    )


def normalize_tool_end(
    run: Any,
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any,
    tool_use_id: str,
    agent_id: str,
    agent_role: str,
    span_id: str,
    parent_span_id: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """
    Normalize a tool call end event.

    Args:
        run: The active ArzuleRun
        tool_name: Name of the tool
        tool_input: Tool input parameters
        tool_output: Tool output/result
        tool_use_id: Unique tool use identifier
        agent_id: Current agent ID
        agent_role: Current agent role
        span_id: Span ID for this tool call
        parent_span_id: Optional parent span
        error: Optional error message

    Returns:
        TraceEvent dict
    """
    status = "error" if error else "ok"
    summary = f"{tool_name} {'failed' if error else 'completed'}"

    payload: dict[str, Any] = {
        "tool_input": _sanitize_tool_input(tool_name, tool_input),
    }

    if tool_output is not None:
        payload["tool_output"] = _sanitize_tool_output(tool_name, tool_output)

    if error:
        payload["error"] = truncate_string(str(error), 500)

    return normalize_event(
        run,
        event_type="tool.call.end",
        agent_id=agent_id,
        agent_role=agent_role,
        summary=summary,
        status=status,
        span_id=span_id,
        parent_span_id=parent_span_id,
        attrs={
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
        },
        payload=payload,
    )


def normalize_handoff_proposed(
    run: Any,
    *,
    session_id: str,
    tool_use_id: str,
    subagent_type: str,
    description: str,
    prompt: str,
    handoff_key: str,
    span_id: str,
    parent_span_id: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Normalize a handoff.proposed event (Task tool called).

    Args:
        run: The active ArzuleRun
        session_id: Claude Code session ID
        tool_use_id: Tool use ID for the Task call
        subagent_type: Type of subagent (Explore, Plan, etc.)
        description: Task description
        prompt: Full prompt for the subagent
        handoff_key: Correlation key for handoff lifecycle
        span_id: Span ID for this handoff
        parent_span_id: Optional parent span
        model: Optional model hint

    Returns:
        TraceEvent dict
    """
    to_agent_id = f"claude_code:subagent:{subagent_type}:{tool_use_id}"

    return normalize_event(
        run,
        event_type="handoff.proposed",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary=f"Delegating to {subagent_type}: {truncate_string(description, 100)}",
        span_id=span_id,
        parent_span_id=parent_span_id,
        attrs={
            "handoff_key": handoff_key,
            "to_agent": to_agent_id,
            "to_agent_role": subagent_type,
            "subagent_type": subagent_type,
            "description": truncate_string(description, 200),
            "is_handoff": True,
        },
        payload={
            "prompt": truncate_string(prompt, 5000),
            "description": description,
            "subagent_type": subagent_type,
            "tool_use_id": tool_use_id,
            "model": model,
        },
    )


def normalize_handoff_ack(
    run: Any,
    *,
    session_id: str,
    tool_use_id: str,
    subagent_type: str,
    handoff_key: str,
    span_id: str,
    parent_span_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Normalize a handoff.ack event (subagent starts work).

    Args:
        run: The active ArzuleRun
        session_id: Claude Code session ID
        tool_use_id: Tool use ID for the Task call
        subagent_type: Type of subagent
        handoff_key: Correlation key for handoff lifecycle
        span_id: Span ID for this event
        parent_span_id: Optional parent span

    Returns:
        TraceEvent dict
    """
    agent_id = f"claude_code:subagent:{subagent_type}:{tool_use_id}"

    return normalize_event(
        run,
        event_type="handoff.ack",
        agent_id=agent_id,
        agent_role=subagent_type,
        summary=f"{subagent_type} agent started",
        span_id=span_id,
        parent_span_id=parent_span_id,
        subagent_type=subagent_type,
        tools=_get_subagent_tools(subagent_type),
        attrs={
            "handoff_key": handoff_key,
            "from_agent": f"claude_code:main:{session_id}",
        },
        payload={
            "tool_use_id": tool_use_id,
        },
    )


def normalize_handoff_complete(
    run: Any,
    *,
    session_id: str,
    tool_use_id: str,
    subagent_type: str,
    result: Any,
    handoff_key: str,
    span_id: str,
    parent_span_id: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """
    Normalize a handoff.complete event (subagent returns result).

    Args:
        run: The active ArzuleRun
        session_id: Claude Code session ID
        tool_use_id: Tool use ID for the Task call
        subagent_type: Type of subagent
        result: Result returned by subagent
        handoff_key: Correlation key for handoff lifecycle
        span_id: Span ID for this event
        parent_span_id: Optional parent span
        error: Optional error message

    Returns:
        TraceEvent dict
    """
    status = "error" if error else "ok"
    from_agent_id = f"claude_code:subagent:{subagent_type}:{tool_use_id}"

    # Build result summary
    if error:
        result_summary = f"Error: {truncate_string(error, 100)}"
    elif isinstance(result, str):
        result_summary = truncate_string(result, 150)
    elif isinstance(result, dict):
        result_summary = truncate_string(str(result.get("result", result)), 150)
    else:
        result_summary = truncate_string(str(result), 150)

    return normalize_event(
        run,
        event_type="handoff.complete",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary=f"{subagent_type} completed: {result_summary}",
        status=status,
        span_id=span_id,
        parent_span_id=parent_span_id,
        attrs={
            "handoff_key": handoff_key,
            "from_agent": from_agent_id,
            "from_agent_role": subagent_type,
        },
        payload={
            "result": _sanitize_tool_output("Task", result),
            "tool_use_id": tool_use_id,
            "error": error,
        },
    )


def _build_tool_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Build a human-readable summary for a tool call."""
    if tool_name == "Read":
        file_path = tool_input.get("file_path", "unknown")
        return f"Reading {_truncate_path(file_path)}"

    if tool_name == "Write":
        file_path = tool_input.get("file_path", "unknown")
        return f"Writing {_truncate_path(file_path)}"

    if tool_name == "Edit":
        file_path = tool_input.get("file_path", "unknown")
        return f"Editing {_truncate_path(file_path)}"

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        return f"Running: {truncate_string(command, 100)}"

    if tool_name == "Glob":
        pattern = tool_input.get("pattern", "")
        return f"Globbing: {pattern}"

    if tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        return f"Searching: {truncate_string(pattern, 80)}"

    if tool_name == "WebFetch":
        url = tool_input.get("url", "")
        return f"Fetching: {truncate_string(url, 80)}"

    if tool_name == "WebSearch":
        query = tool_input.get("query", "")
        return f"Searching: {truncate_string(query, 80)}"

    if tool_name == "Task":
        description = tool_input.get("description", "")
        return f"Task: {truncate_string(description, 100)}"

    if tool_name == "TodoWrite":
        return "Updating todo list"

    if tool_name == "NotebookEdit":
        notebook = tool_input.get("notebook_path", "unknown")
        return f"Editing notebook: {_truncate_path(notebook)}"

    # MCP tools
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        if len(parts) >= 3:
            server = parts[1]
            method = parts[2]
            return f"MCP {server}: {method}"
        return f"MCP call: {tool_name}"

    return f"Calling {tool_name}"


def _truncate_path(path: str, max_len: int = 50) -> str:
    """Truncate a file path, keeping the filename."""
    if len(path) <= max_len:
        return path

    # Keep filename and truncate directory
    parts = path.rsplit("/", 1)
    if len(parts) == 2:
        filename = parts[1]
        if len(filename) >= max_len - 4:
            return f".../{filename[:max_len-4]}"
        return f".../{filename}"

    return truncate_string(path, max_len)


def _sanitize_tool_input(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Sanitize tool input for safe storage."""
    if not isinstance(tool_input, dict):
        return {"_value": truncate_string(str(tool_input), 5000)}

    sanitized = {}
    for key, value in tool_input.items():
        if isinstance(value, str):
            # Truncate large strings (file contents, prompts)
            max_len = 5000 if key in ("content", "prompt", "new_source") else 10000
            sanitized[key] = truncate_string(value, max_len)
        elif isinstance(value, (dict, list)):
            sanitized[key] = sanitize(value)
        else:
            sanitized[key] = value

    return sanitized


def _sanitize_tool_output(tool_name: str, tool_output: Any) -> Any:
    """Sanitize tool output for safe storage."""
    if tool_output is None:
        return None

    if isinstance(tool_output, str):
        # Truncate large outputs
        return truncate_string(tool_output, 50000)

    if isinstance(tool_output, dict):
        return sanitize(tool_output)

    if isinstance(tool_output, list):
        return sanitize(tool_output)

    return truncate_string(str(tool_output), 50000)


def _get_subagent_tools(subagent_type: str) -> list[str]:
    """Get the tools available to a subagent type."""
    tools_map = {
        "Explore": ["Glob", "Grep", "Read"],
        "Plan": ["Glob", "Grep", "Read", "Task"],
        "Bash": ["Bash"],
        "general-purpose": ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task", "WebFetch", "WebSearch"],
        "claude-code-guide": ["Glob", "Grep", "Read", "WebFetch", "WebSearch"],
    }
    return tools_map.get(subagent_type, [])
