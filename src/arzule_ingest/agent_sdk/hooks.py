"""Hook callbacks for Claude Agent SDK tracing.

This module provides hook callback functions that integrate with the
Claude Agent SDK's hook system to emit trace events to Arzule.

The hooks follow the Agent SDK callback signature:
    async def hook_callback(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext
    ) -> HookJSONOutput

Event Types Emitted:
- tool.call.start: When a tool begins execution (PreToolUse)
- tool.call.end: When a tool completes (PostToolUse)
- prompt.submit: When a user prompt is submitted (UserPromptSubmit)
- turn.end: When a conversation turn ends (Stop)
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from ..ids import new_span_id
from ..sanitize import sanitize, truncate_string

if TYPE_CHECKING:
    from ..run import ArzuleRun

# Type aliases for SDK types (to avoid hard dependency)
HookInput = dict[str, Any]
HookContext = dict[str, Any]
HookJSONOutput = dict[str, Any]
HookCallback = Callable[[HookInput, str | None, HookContext], Awaitable[HookJSONOutput]]


# =============================================================================
# Session State Type
# =============================================================================


class SessionState:
    """Thread-safe state container for tracking active tool calls within a session.

    Attributes:
        active_tools: Maps tool_use_id to tracking info (start_time, span_id, etc.)
        turn_count: Number of turns in this session
        current_turn_span_id: Span ID for the current turn
        session_id: Claude session identifier
    """

    __slots__ = ("active_tools", "turn_count", "current_turn_span_id", "session_id")

    def __init__(self, session_id: str | None = None) -> None:
        self.active_tools: dict[str, dict[str, Any]] = {}
        self.turn_count: int = 0
        self.current_turn_span_id: str | None = None
        self.session_id: str | None = session_id


# =============================================================================
# Event Normalization Functions
# =============================================================================


def _base_event(
    run: "ArzuleRun",
    *,
    span_id: str | None = None,
    parent_span_id: str | None = None,
) -> dict[str, Any]:
    """Build base event fields for a trace event."""
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


def _build_agent_info(
    session_id: str | None,
    agent_role: str = "main",
) -> dict[str, Any]:
    """Build agent info dict for trace events."""
    agent_id = f"claude_sdk:main:{session_id or 'unknown'}"
    return {
        "id": agent_id,
        "role": agent_role,
        "framework": "claude_agent_sdk",
    }


def normalize_tool_start(
    input_data: HookInput,
    tool_use_id: str | None,
    run: "ArzuleRun",
    state: SessionState,
    span_id: str,
) -> dict[str, Any]:
    """Normalize a PreToolUse hook input to a tool.call.start event.

    Args:
        input_data: The PreToolUseHookInput from the SDK
        tool_use_id: Unique identifier for this tool use
        run: The active ArzuleRun
        state: Session state for tracking
        span_id: Span ID for this tool call

    Returns:
        TraceEvent dict for tool.call.start
    """
    tool_name = input_data.get("tool_name", "unknown")
    tool_input = input_data.get("tool_input", {})
    session_id = input_data.get("session_id") or state.session_id

    # Build human-readable summary
    summary = _build_tool_summary(tool_name, tool_input)

    return {
        **_base_event(run, span_id=span_id, parent_span_id=state.current_turn_span_id),
        "agent": _build_agent_info(session_id),
        "event_type": "tool.call.start",
        "status": "ok",
        "summary": truncate_string(summary, 200),
        "attrs_compact": sanitize({
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "framework": "claude_agent_sdk",
        }),
        "payload": sanitize({
            "tool_input": _sanitize_tool_input(tool_name, tool_input),
        }),
    }


def normalize_tool_end(
    input_data: HookInput,
    tool_use_id: str | None,
    run: "ArzuleRun",
    state: SessionState,
    span_id: str,
    start_time: float,
) -> dict[str, Any]:
    """Normalize a PostToolUse hook input to a tool.call.end event.

    Args:
        input_data: The PostToolUseHookInput from the SDK
        tool_use_id: Unique identifier for this tool use
        run: The active ArzuleRun
        state: Session state for tracking
        span_id: Span ID for this tool call (should match start)
        start_time: Unix timestamp when tool started

    Returns:
        TraceEvent dict for tool.call.end
    """
    tool_name = input_data.get("tool_name", "unknown")
    tool_input = input_data.get("tool_input", {})
    tool_response = input_data.get("tool_response")
    session_id = input_data.get("session_id") or state.session_id

    # Calculate duration
    duration_ms = int((time.time() - start_time) * 1000)

    # Determine status from response
    status = "ok"
    error_msg = None
    if isinstance(tool_response, dict):
        if tool_response.get("is_error"):
            status = "error"
            error_msg = str(tool_response.get("content", ""))[:500]

    summary = f"{tool_name} {'failed' if status == 'error' else 'completed'}"

    payload: dict[str, Any] = {
        "tool_input": _sanitize_tool_input(tool_name, tool_input),
        "duration_ms": duration_ms,
    }

    if tool_response is not None:
        payload["tool_output"] = _sanitize_tool_output(tool_name, tool_response)

    if error_msg:
        payload["error"] = error_msg

    return {
        **_base_event(run, span_id=span_id, parent_span_id=state.current_turn_span_id),
        "agent": _build_agent_info(session_id),
        "event_type": "tool.call.end",
        "status": status,
        "summary": truncate_string(summary, 200),
        "attrs_compact": sanitize({
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "duration_ms": duration_ms,
            "framework": "claude_agent_sdk",
        }),
        "payload": sanitize(payload),
    }


def normalize_prompt(
    input_data: HookInput,
    run: "ArzuleRun",
    state: SessionState,
    span_id: str,
) -> dict[str, Any]:
    """Normalize a UserPromptSubmit hook input to a prompt.submit event.

    Args:
        input_data: The UserPromptSubmitHookInput from the SDK
        run: The active ArzuleRun
        state: Session state for tracking
        span_id: Span ID for this turn

    Returns:
        TraceEvent dict for prompt.submit
    """
    prompt = input_data.get("prompt", "")
    session_id = input_data.get("session_id") or state.session_id

    # Build summary from prompt preview
    prompt_preview = truncate_string(prompt, 100)
    summary = f"User prompt: {prompt_preview}"

    return {
        **_base_event(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": _build_agent_info(session_id),
        "event_type": "prompt.submit",
        "status": "ok",
        "summary": truncate_string(summary, 200),
        "attrs_compact": sanitize({
            "turn_number": state.turn_count,
            "prompt_length": len(prompt),
            "framework": "claude_agent_sdk",
        }),
        "payload": sanitize({
            "prompt": truncate_string(prompt, 5000),
        }),
    }


def normalize_turn_end(
    input_data: HookInput,
    run: "ArzuleRun",
    state: SessionState,
    span_id: str,
) -> dict[str, Any]:
    """Normalize a Stop hook input to a turn.end event.

    Args:
        input_data: The StopHookInput from the SDK
        run: The active ArzuleRun
        state: Session state for tracking
        span_id: Span ID for this turn (should match prompt span)

    Returns:
        TraceEvent dict for turn.end
    """
    session_id = input_data.get("session_id") or state.session_id
    stop_hook_active = input_data.get("stop_hook_active", False)

    summary = f"Turn {state.turn_count} completed"
    if stop_hook_active:
        summary += " (stop hook active)"

    return {
        **_base_event(run, span_id=span_id, parent_span_id=run.current_parent_span_id()),
        "agent": _build_agent_info(session_id),
        "event_type": "turn.end",
        "status": "ok",
        "summary": truncate_string(summary, 200),
        "attrs_compact": sanitize({
            "turn_number": state.turn_count,
            "stop_hook_active": stop_hook_active,
            "tools_called": len(state.active_tools),
            "framework": "claude_agent_sdk",
        }),
        "payload": {},
    }


# =============================================================================
# Hook Callback Functions
# =============================================================================


async def _pre_tool_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
    run: "ArzuleRun",
    state: SessionState,
) -> HookJSONOutput:
    """Handle PreToolUse hook - emit tool.call.start event.

    Args:
        input_data: Hook input data from SDK
        tool_use_id: Unique tool use identifier
        context: Hook context (currently unused)
        run: The active ArzuleRun
        state: Session state for tracking

    Returns:
        Empty dict to pass through (no modifications to tool behavior)
    """
    try:
        # Generate span ID for this tool call
        span_id = new_span_id()

        # Track this tool call
        effective_tool_use_id = tool_use_id or new_span_id()
        state.active_tools[effective_tool_use_id] = {
            "start_time": time.time(),
            "span_id": span_id,
            "tool_name": input_data.get("tool_name", "unknown"),
        }

        # Emit start event
        event = normalize_tool_start(input_data, effective_tool_use_id, run, state, span_id)
        run.emit(event)

        # Push span onto stack for nested tracking
        run.push_span(span_id)

    except Exception as e:
        print(f"[arzule] Error in pre_tool_hook: {e}", file=sys.stderr)

    # Always return empty dict to pass through
    return {}


async def _post_tool_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
    run: "ArzuleRun",
    state: SessionState,
) -> HookJSONOutput:
    """Handle PostToolUse hook - emit tool.call.end event.

    Args:
        input_data: Hook input data from SDK
        tool_use_id: Unique tool use identifier
        context: Hook context (currently unused)
        run: The active ArzuleRun
        state: Session state for tracking

    Returns:
        Empty dict to pass through (no modifications to tool behavior)
    """
    try:
        effective_tool_use_id = tool_use_id or "unknown"

        # Look up tracking info from start
        tool_info = state.active_tools.pop(effective_tool_use_id, None)

        if tool_info:
            span_id = tool_info["span_id"]
            start_time = tool_info["start_time"]

            # Pop the span we pushed in pre_tool_hook
            run.pop_span()
        else:
            # Fallback if we missed the start event
            span_id = new_span_id()
            start_time = time.time()

        # Emit end event
        event = normalize_tool_end(
            input_data, effective_tool_use_id, run, state, span_id, start_time
        )
        run.emit(event)

    except Exception as e:
        print(f"[arzule] Error in post_tool_hook: {e}", file=sys.stderr)

    return {}


async def _prompt_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
    run: "ArzuleRun",
    state: SessionState,
) -> HookJSONOutput:
    """Handle UserPromptSubmit hook - emit prompt.submit event.

    This marks the start of a new turn in the conversation.

    Args:
        input_data: Hook input data from SDK
        tool_use_id: Not used for this hook type
        context: Hook context (currently unused)
        run: The active ArzuleRun
        state: Session state for tracking

    Returns:
        Empty dict to pass through (no modifications to behavior)
    """
    try:
        # Increment turn count
        state.turn_count += 1

        # Update session ID if provided
        if input_data.get("session_id"):
            state.session_id = input_data["session_id"]

        # Generate span ID for this turn
        span_id = new_span_id()
        state.current_turn_span_id = span_id

        # Clear any stale active tools from previous turn
        if state.active_tools:
            print(
                f"[arzule] Warning: {len(state.active_tools)} unclosed tool calls "
                f"from previous turn",
                file=sys.stderr,
            )
            state.active_tools.clear()

        # Emit prompt event
        event = normalize_prompt(input_data, run, state, span_id)
        run.emit(event)

        # Push turn span for tool call parenting
        run.push_span(span_id)

    except Exception as e:
        print(f"[arzule] Error in prompt_hook: {e}", file=sys.stderr)

    return {}


async def _stop_hook(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
    run: "ArzuleRun",
    state: SessionState,
) -> HookJSONOutput:
    """Handle Stop hook - emit turn.end event.

    This marks the end of a turn in the conversation.

    Args:
        input_data: Hook input data from SDK
        tool_use_id: Not used for this hook type
        context: Hook context (currently unused)
        run: The active ArzuleRun
        state: Session state for tracking

    Returns:
        Empty dict to pass through (no modifications to behavior)
    """
    try:
        # Use the turn span from prompt if available
        span_id = state.current_turn_span_id or new_span_id()

        # Emit turn end event
        event = normalize_turn_end(input_data, run, state, span_id)
        run.emit(event)

        # Pop turn span
        if state.current_turn_span_id:
            run.pop_span()
            state.current_turn_span_id = None

        # Warn about unclosed tool calls
        if state.active_tools:
            print(
                f"[arzule] Warning: {len(state.active_tools)} unclosed tool calls "
                f"at turn end: {list(state.active_tools.keys())[:5]}",
                file=sys.stderr,
            )

    except Exception as e:
        print(f"[arzule] Error in stop_hook: {e}", file=sys.stderr)

    return {}


# =============================================================================
# Hook Factory
# =============================================================================


def create_tracing_hooks(
    run: "ArzuleRun",
    session_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Create hook configuration for Agent SDK with tracing callbacks.

    This function returns a hooks configuration dict that can be passed to
    ClaudeAgentOptions.hooks. The hooks will emit trace events to the
    provided ArzuleRun.

    Args:
        run: The ArzuleRun instance to emit events to
        session_id: Optional session ID for agent identification

    Returns:
        Dict mapping hook event names to list of HookMatcher configs,
        compatible with ClaudeAgentOptions.hooks

    Example:
        ```python
        from arzule_ingest import ArzuleRun
        from arzule_ingest.agent_sdk.hooks import create_tracing_hooks
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

        with ArzuleRun(tenant_id="...", project_id="...", sink=sink) as run:
            hooks = create_tracing_hooks(run)
            options = ClaudeAgentOptions(hooks=hooks)
            client = ClaudeSDKClient()
            async for msg in client.query("Hello", options=options):
                print(msg)
        ```
    """
    # Create shared session state
    state = SessionState(session_id=session_id)

    # Create bound hook callbacks with run and state captured
    pre_tool_callback: HookCallback = partial(_pre_tool_hook, run=run, state=state)
    post_tool_callback: HookCallback = partial(_post_tool_hook, run=run, state=state)
    prompt_callback: HookCallback = partial(_prompt_hook, run=run, state=state)
    stop_callback: HookCallback = partial(_stop_hook, run=run, state=state)

    # Return hooks config matching ClaudeAgentOptions.hooks structure
    # Each hook event maps to a list of HookMatcher objects
    return {
        "PreToolUse": [
            {
                "matcher": None,  # Match all tools
                "hooks": [pre_tool_callback],
                "timeout": 60.0,
            }
        ],
        "PostToolUse": [
            {
                "matcher": None,  # Match all tools
                "hooks": [post_tool_callback],
                "timeout": 60.0,
            }
        ],
        "UserPromptSubmit": [
            {
                "matcher": None,
                "hooks": [prompt_callback],
                "timeout": 60.0,
            }
        ],
        "Stop": [
            {
                "matcher": None,
                "hooks": [stop_callback],
                "timeout": 60.0,
            }
        ],
    }


# =============================================================================
# Helper Functions
# =============================================================================


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
    """Sanitize tool output for safe storage.

    Truncates very large outputs to keep payloads reasonable.
    """
    MAX_OUTPUT_SIZE = 20000

    if tool_output is None:
        return None

    if isinstance(tool_output, str):
        if len(tool_output) > MAX_OUTPUT_SIZE:
            return (
                tool_output[:MAX_OUTPUT_SIZE]
                + f"\n\n[... truncated, full length: {len(tool_output)} chars]"
            )
        return tool_output

    if isinstance(tool_output, dict):
        sanitized = sanitize(tool_output)
        serialized = str(sanitized)
        if len(serialized) > MAX_OUTPUT_SIZE:
            return {
                "_truncated": True,
                "_full_length": len(serialized),
                "_preview": serialized[:MAX_OUTPUT_SIZE],
            }
        return sanitized

    if isinstance(tool_output, list):
        sanitized = sanitize(tool_output)
        serialized = str(sanitized)
        if len(serialized) > MAX_OUTPUT_SIZE:
            return {
                "_truncated": True,
                "_full_length": len(serialized),
                "_preview": serialized[:MAX_OUTPUT_SIZE],
            }
        return sanitized

    output_str = str(tool_output)
    if len(output_str) > MAX_OUTPUT_SIZE:
        return (
            output_str[:MAX_OUTPUT_SIZE]
            + f"\n\n[... truncated, full length: {len(output_str)} chars]"
        )
    return output_str
