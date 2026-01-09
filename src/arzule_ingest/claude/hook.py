"""Main hook handler for Claude Code instrumentation.

This module is invoked by Claude Code hooks and processes events
to emit trace data to Arzule.

KEY CONCEPT: Turn-based architecture
Unlike CrewAI/LangGraph which have discrete "runs", Claude Code sessions
are long-running conversational contexts. A "turn" (UserPrompt -> Stop)
is the correct unit of work to treat as a run.

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
from .turn import (
    start_turn,
    end_turn,
    get_current_turn,
    get_or_create_run,
    update_turn_state,
    push_span,
    pop_span,
    get_current_span,
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


def _load_config() -> None:
    """
    Load Arzule configuration into environment variables.

    Priority order (higher priority first):
    1. Already set environment variables (never overwritten)
    2. ~/.arzule/config (user-level config from 'arzule configure')
    3. Project .env file (for development/testing)

    This ensures pip-installed users get their config from ~/.arzule/config
    while still allowing project-level overrides via .env for development.
    """
    from pathlib import Path

    def parse_env_file(path: Path) -> dict:
        """Parse a .env file and return key-value pairs."""
        env_vars = {}
        if not path.exists():
            return env_vars
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Handle export VAR=value and VAR=value
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    env_vars[key] = value
        except Exception:
            pass
        return env_vars

    home = Path.home()

    # PRIMARY: Load ~/.arzule/config (user-level config)
    # This is the recommended way for pip-installed users
    arzule_config = home / ".arzule" / "config"
    if arzule_config.exists():
        env_vars = parse_env_file(arzule_config)
        for key, value in env_vars.items():
            if key.startswith("ARZULE_") and key not in os.environ:
                os.environ[key] = value


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
    cwd = input_data.get("cwd", "")

    if not event_name or not session_id:
        return

    # Load configuration (hooks don't inherit shell env vars)
    _load_config()

    try:
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
            handler(input_data)

    except Exception as e:
        # Log errors but don't block Claude Code
        _log_error(f"Hook handler error: {e}")


def handle_session_start(input_data: dict) -> None:
    """Handle SessionStart hook - emit session.start event."""
    session_id = input_data["session_id"]
    
    # Get or create a session-level run (not a turn)
    run = get_or_create_run(session_id)

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
    
    _flush_run(run, session_id)


def handle_session_end(input_data: dict) -> None:
    """Handle SessionEnd hook - emit session.end and cleanup."""
    session_id = input_data["session_id"]
    transcript_path = input_data.get("transcript_path")

    # End any active turn first
    active_turn = get_current_turn(session_id)
    if active_turn:
        end_turn(session_id, summary="Session ended")
    
    # Get session-level run
    run = get_or_create_run(session_id)

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

    _flush_run(run, session_id)


def handle_user_prompt(input_data: dict) -> None:
    """
    Handle UserPromptSubmit hook - START a new turn.
    
    This is the KEY event in Claude Code. Each user prompt starts a new
    "turn" which we treat as a separate run for analysis purposes.
    """
    session_id = input_data["session_id"]
    prompt = input_data.get("prompt", "")
    
    # Create prompt summary (first 100 chars, for logging only)
    prompt_summary = prompt[:100] + "..." if len(prompt) > 100 else prompt
    
    # End any existing turn (in case Stop wasn't called)
    existing_turn = get_current_turn(session_id)
    if existing_turn:
        end_turn(session_id, summary="New prompt received")
    
    # Start a new turn
    turn_info = start_turn(session_id, prompt_summary)
    run = turn_info["run"]
    
    # Emit turn.start event
    run.emit(normalize_event(
        run,
        event_type="turn.start",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary="New conversation turn started",
        attrs={
            "turn_id": turn_info["turn_id"],
        },
        payload={
            "prompt_length": len(prompt),
            # Note: We don't include prompt content for privacy
        },
    ))
    
    _flush_run(run, session_id)


def handle_stop(input_data: dict) -> None:
    """
    Handle Stop hook - END the current turn.
    
    This fires after every agent response, marking the end of a turn.
    We flush and close the current turn's run here.
    """
    session_id = input_data["session_id"]
    transcript_path = input_data.get("transcript_path")
    
    # Get current turn
    turn_info = get_current_turn(session_id)
    if not turn_info:
        # No active turn - this can happen on SessionStart
        return
    
    run = turn_info["run"]
    
    # Optionally capture transcript stats
    stats = None
    if transcript_path and os.path.exists(transcript_path):
        stats = get_transcript_stats(transcript_path)
    
    # Emit turn.end event
    run.emit(normalize_event(
        run,
        event_type="turn.end",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary="Conversation turn completed",
        attrs={
            "turn_id": turn_info["turn_id"],
            "tool_count": len(turn_info.get("tool_calls", [])),
        },
        payload={
            "stats": stats,
        },
    ))
    
    # End the turn (flushes the run)
    end_turn(session_id, summary="Turn completed")


def handle_pre_tool_use(input_data: dict) -> None:
    """Handle PreToolUse hook - emit tool.call.start or handoff.proposed."""
    tool_name = input_data.get("tool_name")
    tool_input = input_data.get("tool_input", {})
    tool_use_id = input_data.get("tool_use_id")
    session_id = input_data["session_id"]

    # Get current turn's run
    turn_info = get_current_turn(session_id)
    if not turn_info:
        # No active turn - create one implicitly
        turn_info = start_turn(session_id, f"Tool: {tool_name}")
    
    run = turn_info["run"]
    
    # Track tool call and persist to disk for subsequent hooks
    turn_info.setdefault("tool_calls", []).append({
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "started_at": _now_iso(),
    })
    update_turn_state(session_id, {"tool_calls": turn_info["tool_calls"]})

    # Security validation
    is_safe, severity, reason = validate_tool_input(tool_name, tool_input)

    if not is_safe:
        # Emit blocked event
        run.emit(normalize_event(
            run,
            event_type="tool.call.blocked",
            agent_id=_get_current_agent_id(session_id, turn_info),
            agent_role=_get_current_agent_role(turn_info),
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

    elif severity == "warn":
        # Emit warning event for sensitive operations
        run.emit(normalize_event(
            run,
            event_type="tool.call.warning",
            agent_id=_get_current_agent_id(session_id, turn_info),
            agent_role=_get_current_agent_role(turn_info),
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
        _handle_task_start(run, session_id, tool_use_id, tool_input, turn_info)
    else:
        _handle_regular_tool_start(run, session_id, tool_name, tool_use_id, tool_input, turn_info)
    
    _flush_run(run, session_id)


def _handle_task_start(run: Any, session_id: str, tool_use_id: str, tool_input: dict, turn_info: dict) -> None:
    """Handle Task tool call (subagent delegation)."""
    subagent_type = tool_input.get("subagent_type", "general-purpose")
    description = tool_input.get("description", "")
    prompt = tool_input.get("prompt", "")
    model = tool_input.get("model")

    # Create handoff tracking
    handoff_key = create_handoff(session_id, tool_use_id, subagent_type, description)

    # Track active subagent and persist to disk for subsequent hooks
    turn_info.setdefault("active_subagents", {})[tool_use_id] = {
        "type": subagent_type,
        "started_at": _now_iso(),
    }
    update_turn_state(session_id, {"active_subagents": turn_info["active_subagents"]})

    # Create span for the delegation (push_span already persists)
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
    turn_info: dict,
) -> None:
    """Handle regular (non-Task) tool call."""
    # Create span for the tool call
    span_id = push_span(session_id, tool_use_id=tool_use_id)

    # Get current agent context
    agent_id = _get_current_agent_id(session_id, turn_info)
    agent_role = _get_current_agent_role(turn_info)

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


def handle_post_tool_use(input_data: dict) -> None:
    """Handle PostToolUse hook - emit tool.call.end or handoff.complete."""
    tool_name = input_data.get("tool_name")
    tool_use_id = input_data.get("tool_use_id")
    session_id = input_data["session_id"]
    tool_output = input_data.get("tool_output")
    error = input_data.get("error")

    # Get current turn's run
    turn_info = get_current_turn(session_id)
    if not turn_info:
        return
    
    run = turn_info["run"]

    # Check if this is a Task tool completion
    if tool_name == "Task":
        _handle_task_complete(run, session_id, tool_use_id, tool_output, error, turn_info)
    else:
        _handle_regular_tool_complete(
            run, session_id, tool_name, tool_use_id,
            input_data.get("tool_input", {}), tool_output, error, turn_info
        )
    
    _flush_run(run, session_id)


def _handle_task_complete(
    run: Any,
    session_id: str,
    tool_use_id: str,
    tool_output: Any,
    error: Optional[str],
    turn_info: dict,
) -> None:
    """Handle Task tool completion (subagent returns)."""
    # Get handoff info
    handoff = get_handoff(session_id, tool_use_id)
    handoff_key = complete_handoff(session_id, tool_use_id)

    # Get subagent info and persist the removal to disk
    active_subagents = turn_info.get("active_subagents", {})
    subagent_info = active_subagents.pop(tool_use_id, None)
    subagent_type = subagent_info.get("type", "unknown") if subagent_info else "unknown"
    # Persist the updated active_subagents to disk for subsequent hooks
    update_turn_state(session_id, {"active_subagents": active_subagents})

    # Pop span (pop_span already persists to disk)
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
    turn_info: dict,
) -> None:
    """Handle regular (non-Task) tool completion."""
    # Pop span
    span_info = pop_span(session_id, tool_use_id)
    span_id = span_info.get("span_id") if span_info else new_span_id()

    # Get current agent context
    agent_id = _get_current_agent_id(session_id, turn_info)
    agent_role = _get_current_agent_role(turn_info)

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


def handle_subagent_stop(input_data: dict) -> None:
    """Handle SubagentStop hook - emit agent.end for subagent."""
    session_id = input_data["session_id"]
    transcript_path = input_data.get("transcript_path")

    turn_info = get_current_turn(session_id)
    if not turn_info:
        return
    
    run = turn_info["run"]

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
    
    _flush_run(run, session_id)


def handle_pre_compact(input_data: dict) -> None:
    """Handle PreCompact hook - context window compaction."""
    session_id = input_data["session_id"]

    turn_info = get_current_turn(session_id)
    if not turn_info:
        return
    
    run = turn_info["run"]

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
    
    _flush_run(run, session_id)


def handle_notification(input_data: dict) -> None:
    """Handle Notification hook - user notification/interaction event."""
    session_id = input_data["session_id"]
    notification = input_data.get("notification", {})

    turn_info = get_current_turn(session_id)
    if not turn_info:
        return
    
    run = turn_info["run"]

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
    
    _flush_run(run, session_id)


# =============================================================================
# Helper functions
# =============================================================================

def _get_current_agent_id(session_id: str, turn_info: Optional[dict] = None) -> str:
    """Get the current agent ID based on span context."""
    if turn_info:
        span = get_current_span(session_id)
        if span and span.get("subagent_type"):
            return f"claude_code:subagent:{span['subagent_type']}:{span['tool_use_id']}"
    return f"claude_code:main:{session_id}"


def _get_current_agent_role(turn_info: Optional[dict] = None) -> str:
    """Get the current agent role based on span context."""
    if turn_info:
        spans = turn_info.get("spans", [])
        if spans:
            return spans[-1].get("subagent_type", "main")
    return "main"


def _flush_run(run: Any, session_id: str = None) -> None:
    """Flush the run's sink and persist current seq counter.
    
    IMPORTANT: This also persists the current seq counter to disk
    to prevent duplicate seq numbers across hook invocations.
    """
    try:
        run.sink.flush()
    except Exception:
        pass
    
    # Persist the current seq counter to disk for the next hook invocation
    # This is critical to prevent duplicate seq errors
    if session_id:
        try:
            from .turn import _persist_turn_state, _active_turns
            if session_id in _active_turns:
                _persist_turn_state(session_id, _active_turns[session_id])
        except Exception:
            pass


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
