"""Main hook handler for Claude Code instrumentation.

This module is invoked by Claude Code hooks and processes events
to emit trace data to Arzule.

Usage:
    python -m arzule_ingest.claude.hook

Hook input is read from stdin as JSON.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from ..ids import new_span_id
from .session import (
    get_or_create_session,
    close_session,
    get_session_state,
    update_session_state,
    set_active_subagent,
    remove_active_subagent,
    get_active_subagent,
)
from .normalize import (
    normalize_event,
    normalize_tool_start,
    normalize_tool_end,
    normalize_handoff_proposed,
    normalize_handoff_ack,
    normalize_handoff_complete,
)
from .handoff import (
    create_handoff,
    complete_handoff,
    get_handoff,
)
from .spanctx import (
    push_span,
    pop_span,
    get_current_span,
    get_current_span_id,
    clear_spans,
)
from .security import (
    validate_tool_input,
    should_emit_security_event,
    get_security_event_details,
)
from .transcript import (
    extract_session_summary,
    parse_transcript,
    get_transcript_stats,
)


def handle_hook() -> None:
    """
    Main hook handler - reads from stdin, emits trace events.

    This is the entry point called by Claude Code hooks.
    """
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Invalid input - exit silently to not block Claude Code
        return

    event_name = input_data.get("hook_event_name")
    session_id = input_data.get("session_id")

    if not event_name or not session_id:
        return

    try:
        # Get or create Arzule run for this session
        run = get_or_create_session(session_id)

        # Route to appropriate handler
        handlers = {
            "SessionStart": handle_session_start,
            "SessionEnd": handle_session_end,
            "PreToolUse": handle_pre_tool_use,
            "PostToolUse": handle_post_tool_use,
            "SubagentStop": handle_subagent_stop,
            "UserPromptSubmit": handle_user_prompt,
            "Stop": handle_stop,
            "PreCompact": handle_pre_compact,
            "Notification": handle_notification,
        }

        handler = handlers.get(event_name)
        if handler:
            handler(run, input_data)
    except Exception as e:
        # Log errors but don't block Claude Code
        _log_error(f"Hook handler error: {e}")


def handle_session_start(run: Any, input_data: dict) -> None:
    """Handle SessionStart hook - emit session.start event."""
    session_id = input_data["session_id"]

    # Initialize session state
    update_session_state(session_id, {
        "started_at": _now_iso(),
        "cwd": input_data.get("cwd"),
        "permission_mode": input_data.get("permission_mode"),
    })

    # Emit session start event
    run.emit(normalize_event(
        run,
        event_type="session.start",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary="Claude Code session started",
        attrs={
            "cwd": input_data.get("cwd"),
            "permission_mode": input_data.get("permission_mode"),
        },
        payload={
            "session_id": session_id,
            "cwd": input_data.get("cwd"),
        },
    ))


def handle_session_end(run: Any, input_data: dict) -> None:
    """Handle SessionEnd hook - emit session.end and flush."""
    session_id = input_data["session_id"]
    transcript_path = input_data.get("transcript_path")

    # Extract summary from transcript
    summary = None
    stats = None
    if transcript_path and os.path.exists(transcript_path):
        summary = extract_session_summary(transcript_path)
        stats = get_transcript_stats(transcript_path)

    # Emit session end event
    run.emit(normalize_event(
        run,
        event_type="session.end",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary=summary or "Claude Code session ended",
        attrs={
            "reason": input_data.get("reason"),
        },
        payload={
            "transcript_path": transcript_path,
            "stats": stats,
        },
    ))

    # Cleanup
    clear_spans(session_id)
    close_session(session_id)


def handle_pre_tool_use(run: Any, input_data: dict) -> None:
    """Handle PreToolUse hook - emit tool.call.start or handoff.proposed."""
    tool_name = input_data.get("tool_name")
    tool_input = input_data.get("tool_input", {})
    tool_use_id = input_data.get("tool_use_id")
    session_id = input_data["session_id"]

    # Security validation
    is_safe, severity, reason = validate_tool_input(tool_name, tool_input)

    if not is_safe:
        # Emit blocked event
        run.emit(normalize_event(
            run,
            event_type="tool.call.blocked",
            agent_id=_get_current_agent_id(session_id),
            agent_role=_get_current_agent_role(session_id),
            summary=f"Blocked {tool_name}: {reason}",
            status="error",
            attrs={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "block_reason": reason,
            },
            payload={
                "tool_input": tool_input,
            },
        ))
        # Note: We emit the event but don't actually block -
        # Claude Code's permission system handles actual blocking
        # This is observability, not enforcement

    elif severity == "warn":
        # Emit warning event for sensitive operations
        run.emit(normalize_event(
            run,
            event_type="tool.call.warning",
            agent_id=_get_current_agent_id(session_id),
            agent_role=_get_current_agent_role(session_id),
            summary=f"Sensitive operation: {reason}",
            attrs={
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "warning_reason": reason,
            },
            payload={},
        ))

    # Check if this is a Task tool (subagent delegation)
    if tool_name == "Task":
        _handle_task_start(run, session_id, tool_use_id, tool_input)
    else:
        _handle_regular_tool_start(run, session_id, tool_name, tool_use_id, tool_input)


def _handle_task_start(run: Any, session_id: str, tool_use_id: str, tool_input: dict) -> None:
    """Handle Task tool call (subagent delegation)."""
    subagent_type = tool_input.get("subagent_type", "general-purpose")
    description = tool_input.get("description", "")
    prompt = tool_input.get("prompt", "")
    model = tool_input.get("model")

    # Create handoff tracking
    handoff_key = create_handoff(session_id, tool_use_id, subagent_type, description)

    # Track active subagent
    set_active_subagent(session_id, tool_use_id, subagent_type)

    # Create span for the delegation
    span_id = push_span(session_id, tool_use_id=tool_use_id, subagent_type=subagent_type)

    # Emit handoff.proposed
    run.emit(normalize_handoff_proposed(
        run,
        session_id=session_id,
        tool_use_id=tool_use_id,
        subagent_type=subagent_type,
        description=description,
        prompt=prompt,
        handoff_key=handoff_key,
        span_id=span_id,
        model=model,
    ))

    # Emit handoff.ack (subagent acknowledges)
    ack_span_id = new_span_id()
    run.emit(normalize_handoff_ack(
        run,
        session_id=session_id,
        tool_use_id=tool_use_id,
        subagent_type=subagent_type,
        handoff_key=handoff_key,
        span_id=ack_span_id,
        parent_span_id=span_id,
    ))


def _handle_regular_tool_start(
    run: Any,
    session_id: str,
    tool_name: str,
    tool_use_id: str,
    tool_input: dict,
) -> None:
    """Handle regular (non-Task) tool call."""
    # Create span for the tool call
    span_id = push_span(session_id, tool_use_id=tool_use_id)

    # Get current agent context
    agent_id = _get_current_agent_id(session_id)
    agent_role = _get_current_agent_role(session_id)

    # Emit tool.call.start
    run.emit(normalize_tool_start(
        run,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_use_id=tool_use_id,
        agent_id=agent_id,
        agent_role=agent_role,
        span_id=span_id,
    ))


def handle_post_tool_use(run: Any, input_data: dict) -> None:
    """Handle PostToolUse hook - emit tool.call.end or handoff.complete."""
    tool_name = input_data.get("tool_name")
    tool_use_id = input_data.get("tool_use_id")
    session_id = input_data["session_id"]
    tool_output = input_data.get("tool_output")
    error = input_data.get("error")

    # Check if this is a Task tool completion
    if tool_name == "Task":
        _handle_task_complete(run, session_id, tool_use_id, tool_output, error)
    else:
        _handle_regular_tool_complete(
            run, session_id, tool_name, tool_use_id,
            input_data.get("tool_input", {}), tool_output, error
        )


def _handle_task_complete(
    run: Any,
    session_id: str,
    tool_use_id: str,
    tool_output: Any,
    error: Optional[str],
) -> None:
    """Handle Task tool completion (subagent returns)."""
    # Get handoff info
    handoff = get_handoff(session_id, tool_use_id)
    handoff_key = complete_handoff(session_id, tool_use_id)

    # Get subagent info
    subagent_info = remove_active_subagent(session_id, tool_use_id)
    subagent_type = subagent_info.get("type", "unknown") if subagent_info else "unknown"

    # Pop span
    span_info = pop_span(session_id, tool_use_id)
    span_id = span_info.get("span_id") if span_info else new_span_id()

    # Extract result
    result = None
    if isinstance(tool_output, dict):
        result = tool_output.get("result", tool_output)
    else:
        result = tool_output

    # Emit handoff.complete
    run.emit(normalize_handoff_complete(
        run,
        session_id=session_id,
        tool_use_id=tool_use_id,
        subagent_type=subagent_type,
        result=result,
        handoff_key=handoff_key or "",
        span_id=new_span_id(),
        parent_span_id=span_id,
        error=error,
    ))


def _handle_regular_tool_complete(
    run: Any,
    session_id: str,
    tool_name: str,
    tool_use_id: str,
    tool_input: dict,
    tool_output: Any,
    error: Optional[str],
) -> None:
    """Handle regular (non-Task) tool completion."""
    # Pop span
    span_info = pop_span(session_id, tool_use_id)
    span_id = span_info.get("span_id") if span_info else new_span_id()

    # Get current agent context
    agent_id = _get_current_agent_id(session_id)
    agent_role = _get_current_agent_role(session_id)

    # Emit tool.call.end
    run.emit(normalize_tool_end(
        run,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        tool_use_id=tool_use_id,
        agent_id=agent_id,
        agent_role=agent_role,
        span_id=span_id,
        error=error,
    ))


def handle_subagent_stop(run: Any, input_data: dict) -> None:
    """Handle SubagentStop hook - emit agent.end for subagent."""
    session_id = input_data["session_id"]
    transcript_path = input_data.get("transcript_path")

    # SubagentStop fires for the parent session when a child completes
    # The actual result is captured in PostToolUse for the Task tool
    # This is mainly for logging/debugging

    run.emit(normalize_event(
        run,
        event_type="agent.end",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary="Subagent execution completed",
        attrs={},
        payload={
            "transcript_path": transcript_path,
        },
    ))


def handle_user_prompt(run: Any, input_data: dict) -> None:
    """Handle UserPromptSubmit hook - track user input."""
    session_id = input_data["session_id"]

    # Track that a new turn started (we don't emit user prompts for privacy)
    update_session_state(session_id, {
        "last_user_prompt_ts": _now_iso(),
    })

    # Emit a lightweight turn event
    run.emit(normalize_event(
        run,
        event_type="user.prompt",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary="User prompt received",
        attrs={},
        payload={},  # No content for privacy
    ))


def handle_stop(run: Any, input_data: dict) -> None:
    """Handle Stop hook - main agent response completed."""
    session_id = input_data["session_id"]
    transcript_path = input_data.get("transcript_path")

    # Emit response complete event
    run.emit(normalize_event(
        run,
        event_type="agent.response.complete",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary="Main agent response completed",
        attrs={},
        payload={},
    ))

    # Optionally capture full transcript
    if transcript_path and os.path.exists(transcript_path):
        transcript = parse_transcript(transcript_path)
        if transcript:
            run.emit(normalize_event(
                run,
                event_type="session.transcript",
                agent_id=f"claude_code:main:{session_id}",
                agent_role="main",
                summary=f"Captured {len(transcript)} messages",
                attrs={
                    "message_count": len(transcript),
                },
                payload={
                    "transcript": transcript,
                },
            ))

    # Flush events
    run.flush()


def handle_pre_compact(run: Any, input_data: dict) -> None:
    """Handle PreCompact hook - context window compaction."""
    session_id = input_data["session_id"]

    run.emit(normalize_event(
        run,
        event_type="context.compact",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary="Context window compacted due to token limit",
        attrs={
            "reason": input_data.get("reason", "token_limit"),
        },
        payload={},
    ))


def handle_notification(run: Any, input_data: dict) -> None:
    """Handle Notification hook - user notification/interaction event."""
    session_id = input_data["session_id"]
    notification = input_data.get("notification", {})

    run.emit(normalize_event(
        run,
        event_type="notification",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary="User notification",
        attrs={},
        payload={
            "notification": notification,
        },
    ))


# Helper functions

def _get_current_agent_id(session_id: str) -> str:
    """Get the current agent ID based on span context."""
    span = get_current_span(session_id)
    if span and span.get("subagent_type"):
        return f"claude_code:subagent:{span['subagent_type']}:{span['tool_use_id']}"
    return f"claude_code:main:{session_id}"


def _get_current_agent_role(session_id: str) -> str:
    """Get the current agent role based on span context."""
    span = get_current_span(session_id)
    if span:
        return span.get("subagent_type", "main")
    return "main"


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _log_error(message: str) -> None:
    """Log error to stderr (doesn't block Claude Code)."""
    try:
        print(f"[arzule] {message}", file=sys.stderr)
    except Exception:
        pass


if __name__ == "__main__":
    handle_hook()
