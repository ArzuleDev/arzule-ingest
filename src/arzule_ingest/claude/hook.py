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
import time
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
    emit_with_seq_sync,
)
from .normalize import (
    normalize_event,
    normalize_tool_start,
    normalize_tool_end,
    normalize_handoff_proposed,
    normalize_handoff_complete,
)
from .handoff import (
    create_handoff,
    complete_handoff,
    get_handoff,
    get_active_handoffs,
    generate_handoff_key,
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
            "SubagentStart": handle_subagent_start,
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
    """Handle SessionEnd hook - emit session.end and cleanup.

    This handler ensures both turn-level and session-level runs are properly
    closed with run.end events, which transitions their status from 'receiving'
    to 'indexed' on the backend.
    """
    session_id = input_data["session_id"]
    transcript_path = input_data.get("transcript_path")

    # End any active turn first - this emits run.end for the turn-level run
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

    # Properly close the session-level run by calling __exit__
    # This emits run.end which transitions the run status from 'receiving' to 'indexed'
    # on the backend, so the session shows as "Ended" instead of "Active"
    try:
        run.__exit__(None, None, None)
    except Exception:
        # Fallback: at least flush the sink
        _flush_run(run, session_id)


def handle_user_prompt(input_data: dict) -> None:
    """
    Handle UserPromptSubmit hook - START a new turn.

    This is the KEY event in Claude Code. Each user prompt starts a new
    "turn" which we treat as a separate run for analysis purposes.

    IMPORTANT: When a subagent fires UserPromptSubmit, we must NOT end the
    parent's turn, as that would destroy the Task span context needed to
    attribute the subagent's tool calls correctly.
    """
    session_id = input_data["session_id"]
    prompt = input_data.get("prompt", "")

    # Create prompt summary (first 100 chars, for logging only)
    prompt_summary = prompt[:100] + "..." if len(prompt) > 100 else prompt

    # Check if there are active handoffs (Task spans) - if so, this is a subagent prompt
    # Subagent prompts should NOT create a new turn because:
    # 1. The parent's Task span needs to remain in the span stack
    # 2. The subagent's tool calls need to inherit the subagent context
    active_handoffs = get_active_handoffs(session_id)
    existing_turn = get_current_turn(session_id)

    if active_handoffs and existing_turn:
        # This is a SUBAGENT prompt - don't create a new turn!
        # The subagent's tool calls will use the parent's turn which has the Task span.
        _log_error(f"UserPromptSubmit: SUBAGENT detected ({len(active_handoffs)} active handoffs), reusing parent turn")
        _log_error(f"UserPromptSubmit: active handoffs: {[h.get('subagent_type') for h in active_handoffs]}")

        # CRITICAL FIX: Track which subagent is currently executing
        # For parallel Tasks, we need to identify the correct one
        # Match by finding the handoff whose prompt matches (stored in active_subagents)
        turn_info = existing_turn
        active_subagents = turn_info.get("active_subagents", {})
        matched_tool_use_id = None

        # Try to match by prompt content
        for tool_use_id, info in active_subagents.items():
            stored_prompt = info.get("prompt", "")
            # Check if prompts match (subagent prompts typically include the Task prompt)
            if stored_prompt and prompt and (stored_prompt in prompt or prompt in stored_prompt):
                matched_tool_use_id = tool_use_id
                _log_error(f"UserPromptSubmit: MATCHED subagent by prompt, tool_use_id={tool_use_id}")
                break

        # Fallback: if only one active handoff, use that
        if not matched_tool_use_id and len(active_handoffs) == 1:
            matched_tool_use_id = active_handoffs[0].get("tool_use_id")
            _log_error(f"UserPromptSubmit: Using single active handoff, tool_use_id={matched_tool_use_id}")

        # Store the currently executing subagent for tool call attribution
        if matched_tool_use_id:
            update_turn_state(session_id, {"current_executing_subagent": matched_tool_use_id})
            _log_error(f"UserPromptSubmit: Set current_executing_subagent={matched_tool_use_id}")

        # Just flush and return - don't create a new turn
        run = existing_turn["run"]
        _flush_run(run, session_id)
        return

    # This is a TOP-LEVEL user prompt - create a new turn
    _log_error(f"UserPromptSubmit: TOP-LEVEL prompt, creating new turn")

    # End any existing turn (in case Stop wasn't called)
    if existing_turn:
        end_turn(session_id, summary="New prompt received")

    # Start a new turn
    turn_info = start_turn(session_id, prompt_summary)
    run = turn_info["run"]

    # Emit turn.start event with atomic seq sync
    # Include FULL prompt in payload for dashboard display (not truncated)
    emit_with_seq_sync(session_id, run, normalize_event(
        run,
        event_type="turn.start",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary=prompt_summary or "New conversation turn started",
        attrs={
            "turn_id": turn_info["turn_id"],
        },
        payload={
            "prompt_length": len(prompt),
            "user_prompt": prompt,  # Include FULL prompt for dashboard display
        },
    ))

    _flush_run(run, session_id)


def handle_stop(input_data: dict) -> None:
    """
    Handle Stop hook - END the current turn.

    This fires after every agent response, marking the end of a turn.
    We flush and close the current turn's run here.

    IMPORTANT: If there are active handoffs, this Stop might be from a subagent.
    We should NOT end the turn in that case.
    """
    session_id = input_data["session_id"]
    transcript_path = input_data.get("transcript_path")

    # Check if there are active handoffs - if so, this might be a subagent Stop
    # In that case, don't end the turn (parent's turn must remain active)
    active_handoffs = get_active_handoffs(session_id)
    if active_handoffs:
        _log_error(f"handle_stop: SUBAGENT Stop detected ({len(active_handoffs)} active handoffs), NOT ending turn")
        _log_error(f"handle_stop: active handoffs: {[h.get('subagent_type') for h in active_handoffs]}")
        # Just return - don't emit turn.end or end the turn
        return

    # Get current turn
    turn_info = get_current_turn(session_id)
    if not turn_info:
        # No active turn - this can happen on SessionStart
        return

    run = turn_info["run"]

    # Optionally capture transcript stats and final response
    stats = None
    final_response = None
    _log_error(f"handle_stop: transcript_path={transcript_path}, exists={os.path.exists(transcript_path) if transcript_path else 'N/A'}")
    if transcript_path and os.path.exists(transcript_path):
        stats = get_transcript_stats(transcript_path)
        # Extract the final assistant message as the turn's result
        final_response = _extract_final_response(transcript_path)
        _log_error(f"handle_stop: final_response length={len(final_response) if final_response else 0}")

    # Build payload with final response if available
    payload = {"stats": stats}
    if final_response:
        # Include full response for real-time visibility
        payload["result_preview"] = final_response
        payload["result_length"] = len(final_response)

    # Debug: log payload before emission
    _log_error(f"handle_stop: payload keys={list(payload.keys())}, has_result_preview={'result_preview' in payload}")

    # Emit turn.end event with atomic seq sync
    emit_with_seq_sync(session_id, run, normalize_event(
        run,
        event_type="turn.end",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary="Conversation turn completed",
        attrs={
            "turn_id": turn_info["turn_id"],
            "tool_count": len(turn_info.get("tool_calls", [])),
        },
        payload=payload,
    ))

    # End the turn (flushes the run)
    end_turn(session_id, summary="Turn completed")


def handle_pre_tool_use(input_data: dict) -> None:
    """Handle PreToolUse hook - emit tool.call.start or handoff.proposed."""
    tool_name = input_data.get("tool_name")
    tool_input = input_data.get("tool_input", {})
    tool_use_id = input_data.get("tool_use_id")
    session_id = input_data["session_id"]
    transcript_path = input_data.get("transcript_path")

    # Debug: log all keys to see if there's subagent correlation info
    _log_error(f"PreToolUse[{tool_name}]: input_data keys={sorted(input_data.keys())}, transcript_path={transcript_path}")

    # Get current turn's run
    turn_info = get_current_turn(session_id)
    if not turn_info:
        # No active turn - skip tracking tool calls that happen during startup
        # or other background activities (not user-initiated)
        return

    run = turn_info["run"]

    # Track tool call with tool_input for retrieval in PostToolUse
    # (Claude Code doesn't send tool_input in PostToolUse, so we need to store it)
    tool_call_info = {
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "tool_input": tool_input,  # Store for PostToolUse retrieval
        "started_at": _now_iso(),
    }
    turn_info.setdefault("tool_calls", []).append(tool_call_info)

    # Also store in a lookup dict for fast retrieval by tool_use_id
    turn_info.setdefault("tool_inputs", {})[tool_use_id] = tool_input

    update_turn_state(session_id, {
        "tool_calls": turn_info["tool_calls"],
        "tool_inputs": turn_info.get("tool_inputs", {}),
    })

    # Security validation
    is_safe, severity, reason = validate_tool_input(tool_name, tool_input)

    if not is_safe:
        # Emit blocked event with atomic seq sync
        emit_with_seq_sync(session_id, run, normalize_event(
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
        # Emit warning event for sensitive operations with atomic seq sync
        emit_with_seq_sync(session_id, run, normalize_event(
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

    _log_error(f"_handle_task_start: tool_use_id={tool_use_id}, subagent_type={subagent_type}")
    _log_error(f"_handle_task_start: BEFORE active_subagents={list(turn_info.get('active_subagents', {}).keys())}")

    # Create handoff tracking
    handoff_key = create_handoff(session_id, tool_use_id, subagent_type, description)

    # Track active subagent and persist to disk for subsequent hooks
    # Store the prompt so we can correlate SubagentStop with the correct tool_use_id
    # CRITICAL: Pass ONLY the new entry to update_turn_state, not the whole dict
    # This ensures proper atomic merge with entries from parallel processes
    new_subagent_entry = {
        tool_use_id: {
            "type": subagent_type,
            "prompt": prompt,  # Used for correlation in SubagentStop
            "started_at": _now_iso(),
        }
    }
    # Also update local turn_info for consistency within this process
    turn_info.setdefault("active_subagents", {})[tool_use_id] = new_subagent_entry[tool_use_id]
    _log_error(f"_handle_task_start: AFTER active_subagents={list(turn_info['active_subagents'].keys())}")
    # Pass only the NEW entry, not the whole dict - update_turn_state will merge with disk
    update_turn_state(session_id, {"active_subagents": new_subagent_entry})
    _log_error(f"_handle_task_start: PERSISTED active_subagents for {tool_use_id}")

    # Create span for the delegation (push_span already persists)
    span_id = push_span(session_id, tool_use_id=tool_use_id, subagent_type=subagent_type)

    # Emit handoff.proposed with atomic seq sync
    # NOTE: We do NOT emit handoff.ack here for Claude Code.
    # Unlike CrewAI/LangGraph where there's a real async queue and the subagent
    # might not pick up work, Claude Code's Task tool is synchronous - when
    # PreToolUse fires, the subagent IS starting immediately.
    # Emitting both proposed and ack at the same timestamp caused:
    # 1. Weird graph visualization with confusing edge routing
    # 2. False positive detections in forensics
    # The handoff lifecycle for Claude Code is: proposed -> complete
    # The ack is implicit in the synchronous Task execution model.
    emit_with_seq_sync(session_id, run, normalize_handoff_proposed(
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

    # Get current agent context (pass tool_input for content-based matching with parallel subagents)
    agent_id = _get_current_agent_id(session_id, turn_info, tool_input)
    agent_role = _get_current_agent_role(turn_info, tool_input)

    # Find the parent span for this tool call
    # If inside a subagent (Task), the parent should be the Task span so frontend
    # can walk the parent_span_id chain to find the handoff context
    spans = turn_info.get("spans", [])
    subagent_spans = [s for s in spans if s.get("subagent_type")]
    parent_span_id = None
    handoff_key = None
    subagent_type = None

    if subagent_spans:
        matched_span = None

        # SINGLE SUBAGENT: Use it directly
        if len(subagent_spans) == 1:
            matched_span = subagent_spans[0]
            _log_error(f"PreToolUse[{tool_name}]: single subagent, using span {matched_span.get('tool_use_id')}")

        # MULTIPLE SUBAGENTS: Try content-based matching first (with tool_name for affinity)
        elif tool_input:
            matched_span = _match_tool_to_subagent(tool_input, turn_info, tool_name)
            if matched_span:
                _log_error(f"PreToolUse[{tool_name}]: content-matched to subagent {matched_span.get('tool_use_id')}")

        # FALLBACK: Try current_executing_subagent
        if not matched_span:
            current_executing = turn_info.get("current_executing_subagent")
            if current_executing:
                for span in subagent_spans:
                    if span.get("tool_use_id") == current_executing:
                        matched_span = span
                        _log_error(f"PreToolUse[{tool_name}]: using current_executing_subagent={current_executing}")
                        break

        # LAST RESORT: Use most recent subagent span (with warning for ambiguity)
        if not matched_span:
            matched_span = subagent_spans[-1]
            _log_error(
                f"PreToolUse[{tool_name}]: AMBIGUOUS FALLBACK - using most recent span "
                f"{matched_span.get('tool_use_id')} out of {len(subagent_spans)} candidates. "
                f"Tool attribution may be incorrect!"
            )

        # Extract values from matched span
        if matched_span:
            parent_span_id = matched_span.get("span_id")
            subagent_type = matched_span.get("subagent_type")
            span_tool_use_id = matched_span.get("tool_use_id")
            if span_tool_use_id:
                handoff_key = generate_handoff_key(session_id, span_tool_use_id)

    # Debug: log agent context for tool calls (helps verify subagent attribution)
    _log_error(f"PreToolUse[{tool_name}]: agent_id={agent_id}, agent_role={agent_role}, spans_count={len(spans)}, parent_span_id={parent_span_id}, handoff_key={handoff_key}")
    if spans:
        subagent_spans = [s for s in spans if s.get("subagent_type")]
        if subagent_spans:
            _log_error(f"PreToolUse[{tool_name}]: subagent spans: {[(s.get('subagent_type'), s.get('tool_use_id'), s.get('span_id')) for s in subagent_spans]}")

    # Debug: log when tool_input is empty (helps diagnose missing payloads)
    if not tool_input:
        _log_error(f"PreToolUse[{tool_name}]: tool_input is empty, tool_use_id={tool_use_id}")
    else:
        _log_error(f"PreToolUse[{tool_name}]: tool_input has {len(tool_input)} keys: {list(tool_input.keys())}")

    # Emit tool.call.start with atomic seq sync (critical for parallel tool calls)
    # Pass parent_span_id, handoff_key, and subagent_type for frontend correlation
    emit_with_seq_sync(session_id, run, normalize_tool_start(
        run,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_use_id=tool_use_id,
        agent_id=agent_id,
        agent_role=agent_role,
        span_id=span_id,
        parent_span_id=parent_span_id,
        handoff_key=handoff_key,
        subagent_type=subagent_type,
    ))


def handle_post_tool_use(input_data: dict) -> None:
    """Handle PostToolUse hook - emit tool.call.end or handoff.complete."""
    tool_name = input_data.get("tool_name")
    tool_use_id = input_data.get("tool_use_id")
    session_id = input_data["session_id"]
    # Claude Code 2.1+ uses "tool_response", older versions use "tool_result"
    tool_output = (
        input_data.get("tool_response") or
        input_data.get("tool_result") or
        input_data.get("tool_output")
    )
    error = input_data.get("error")
    # Also check for transcript_path directly in input_data (Claude Code may provide it)
    transcript_path = input_data.get("transcript_path")

    # Debug: log available keys to understand what Claude Code actually sends
    _log_error(
        f"PostToolUse[{tool_name}]: keys={sorted(input_data.keys())}, "
        f"tool_input_present={'tool_input' in input_data}, "
        f"tool_input_empty={not input_data.get('tool_input')}, "
        f"tool_result_present={'tool_result' in input_data}, "
        f"tool_response_present={'tool_response' in input_data}, "
        f"tool_use_id={tool_use_id}, "
        f"IS_TASK={tool_name == 'Task'}"
    )

    # Get current turn's run
    turn_info = get_current_turn(session_id)
    if not turn_info:
        return

    run = turn_info["run"]

    # Get tool_input: prefer what Claude Code sends, fall back to stored from PreToolUse
    # Per Claude Code docs, tool_input SHOULD be in PostToolUse, but in practice it may be missing
    tool_input_from_hook = input_data.get("tool_input")
    stored_tool_inputs = turn_info.get("tool_inputs", {})
    stored_tool_input = stored_tool_inputs.get(tool_use_id)

    # Use what's available, preferring the hook data if present and non-empty
    if tool_input_from_hook:
        tool_input = tool_input_from_hook
    elif stored_tool_input:
        tool_input = stored_tool_input
        _log_error(f"PostToolUse[{tool_name}]: using stored tool_input (hook didn't provide it)")
    else:
        tool_input = {}
        _log_error(f"PostToolUse[{tool_name}]: NO tool_input available (not in hook, not stored)")

    # Check if this is a Task tool completion
    if tool_name == "Task":
        _handle_task_complete(run, session_id, tool_use_id, tool_output, error, turn_info, transcript_path)
    else:
        _handle_regular_tool_complete(
            run, session_id, tool_name, tool_use_id,
            tool_input, tool_output, error, turn_info
        )

    _flush_run(run, session_id)


def _handle_task_complete(
    run: Any,
    session_id: str,
    tool_use_id: str,
    tool_output: Any,
    error: Optional[str],
    turn_info: dict,
    direct_transcript_path: Optional[str] = None,
) -> None:
    """Handle Task tool completion (subagent returns).

    Result extraction priority:
    1. tool_output from PostToolUse (if Claude Code provides it - rare for Task)
    2. Stored result from SubagentStop (extracted from subagent transcript)
    3. Main session transcript (fallback, race condition prone)
    """
    _log_error(f"_handle_task_complete CALLED: tool_use_id={tool_use_id}, session={session_id}")

    # Get handoff info
    handoff = get_handoff(session_id, tool_use_id)
    handoff_key = complete_handoff(session_id, tool_use_id)

    # Get subagent info and persist the removal to disk
    active_subagents = turn_info.get("active_subagents", {})
    subagent_info = active_subagents.pop(tool_use_id, None)
    subagent_type = subagent_info.get("type", "unknown") if subagent_info else "unknown"

    # Clear current_executing_subagent if it matches this tool_use_id
    current_executing = turn_info.get("current_executing_subagent")
    if current_executing == tool_use_id:
        _log_error(f"TaskComplete: clearing current_executing_subagent={tool_use_id}")
        update_turn_state(session_id, {
            "active_subagents": active_subagents,
            "current_executing_subagent": None
        })
    else:
        # Persist the updated active_subagents to disk for subsequent hooks
        update_turn_state(session_id, {"active_subagents": active_subagents})

    # Pop span (pop_span already persists to disk)
    span_info = pop_span(session_id, tool_use_id)
    span_id = span_info.get("span_id") if span_info else new_span_id()

    # Extract result - try multiple sources
    result = None

    # Source 1: tool_output from PostToolUse (rare for Task tools)
    if tool_output is not None:
        if isinstance(tool_output, dict):
            result = tool_output.get("result", tool_output)
        else:
            result = tool_output
        _log_error(f"TaskComplete: got result from tool_output, length={len(str(result)) if result else 0}")

    # Source 2: Result stored by SubagentStop (most reliable)
    if result is None:
        result = turn_info.get("last_subagent_result")
        if result:
            _log_error(f"TaskComplete: got result from SubagentStop, length={len(result)}")
            # Clear the stored result
            turn_info.pop("last_subagent_result", None)
            update_turn_state(session_id, {"last_subagent_result": None})

    # Source 3: Extract from main session transcript (fallback, race condition prone)
    if result is None:
        transcript_path = direct_transcript_path or turn_info.get("last_subagent_transcript")

        # Fallback: construct the default Claude Code transcript path
        if not transcript_path:
            from pathlib import Path
            claude_projects_dir = Path.home() / ".claude" / "projects"
            # Find the session transcript file
            for p in claude_projects_dir.glob(f"*/{session_id}.jsonl"):
                transcript_path = str(p)
                break

        _log_error(f"TaskComplete: falling back to transcript extraction, path={transcript_path}, tool_use_id={tool_use_id}")
        if transcript_path:
            result = _extract_tool_result_from_transcript(transcript_path, tool_use_id)
            _log_error(f"TaskComplete: extracted result length={len(result) if result else 0}")

    # Emit handoff.complete with atomic seq sync
    emit_with_seq_sync(session_id, run, normalize_handoff_complete(
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
    # Pop span - this gets the span_id from tool.call.start
    span_info = pop_span(session_id, tool_use_id)
    parent_span_id = span_info.get("span_id") if span_info else None

    # tool.call.end needs its OWN span_id to avoid overwriting tool.call.start
    # in the database (which has UNIQUE constraint on run_id, span_id)
    end_span_id = new_span_id()

    # Get current agent context
    agent_id = _get_current_agent_id(session_id, turn_info)
    agent_role = _get_current_agent_role(turn_info)

    # Debug: log what we're emitting to help diagnose payload issues
    _log_error(
        f"tool.call.end[{tool_name}]: tool_input_keys={list(tool_input.keys()) if tool_input else []}, "
        f"tool_output_present={tool_output is not None}, "
        f"tool_output_len={len(str(tool_output)) if tool_output else 0}"
    )

    # Emit tool.call.end with atomic seq sync (critical for parallel tool calls)
    emit_with_seq_sync(session_id, run, normalize_tool_end(
        run,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        tool_use_id=tool_use_id,
        agent_id=agent_id,
        agent_role=agent_role,
        span_id=end_span_id,
        parent_span_id=parent_span_id,
        error=error,
    ))

    # Clean up stored tool_input to prevent memory growth
    tool_inputs = turn_info.get("tool_inputs", {})
    if tool_use_id in tool_inputs:
        del tool_inputs[tool_use_id]
        update_turn_state(session_id, {"tool_inputs": tool_inputs})


def handle_subagent_stop(input_data: dict) -> None:
    """Handle SubagentStop hook - emit agent.end for subagent.

    IMPORTANT: Per Claude Code CHANGELOG 2.0.42, SubagentStop provides:
    - agent_id: The subagent's ID (e.g., "ac9ca42")
    - agent_transcript_path: Path to the SUBAGENT's transcript (not main session)

    The subagent's result is the last assistant message in its transcript.

    RACE CONDITION FIX: For parallel same-type subagents, SubagentStart marks them as
    "pending_prompt_match". Here we resolve the mapping using compute_prompt_similarity()
    to match the subagent's first user message against the Task prompts stored in active_subagents.
    """
    session_id = input_data["session_id"]

    # Debug: Log all input data keys to understand what Claude Code sends
    _log_error(f"SubagentStop: input_data keys={list(input_data.keys())}")

    # Use the correct field names from Claude Code 2.0.42+
    agent_id = input_data.get("agent_id")
    agent_transcript_path = input_data.get("agent_transcript_path")
    # Fallback for older versions
    transcript_path = input_data.get("transcript_path") or agent_transcript_path

    _log_error(f"SubagentStop: agent_id={agent_id}, agent_transcript_path={agent_transcript_path}, transcript_path={transcript_path}")

    # Try to get turn_info, but don't fail if missing
    # SubagentStop can still extract tool_use_ids from agent_transcript_path
    turn_info = get_current_turn(session_id)
    if not turn_info:
        _log_error(f"SubagentStop: no turn_info for session {session_id}, using get_or_create_run")
        # Create a minimal turn_info with just what we need
        run = get_or_create_run(session_id)
        turn_info = {"run": run, "active_subagents": {}, "agent_to_task_mapping": {}, "pending_prompt_matches": {}}
    else:
        run = turn_info["run"]

    # Look up subagent type from multiple sources
    # SubagentStop fires BEFORE PostToolUse for Task, so handoff info should still exist
    subagent_type = "Subagent"  # Fallback
    matched_tool_use_id = None

    # STRATEGY 0 (BEST): Use agent_to_task_mapping built in SubagentStart
    # This is the most reliable method when FIFO was unambiguous (single subagent of type)
    agent_to_task_mapping = turn_info.get("agent_to_task_mapping", {})
    if agent_id and agent_id in agent_to_task_mapping:
        matched_tool_use_id = agent_to_task_mapping[agent_id]
        # Look up subagent_type from active_subagents
        active_subagents = turn_info.get("active_subagents", {})
        if matched_tool_use_id in active_subagents:
            subagent_type = active_subagents[matched_tool_use_id].get("type", "Subagent")
        _log_error(f"SubagentStop: MAPPED via SubagentStart! agent_id={agent_id} -> tool_use_id={matched_tool_use_id}, type={subagent_type}")

    # STRATEGY 1 (NEW): Resolve pending_prompt_match using fuzzy prompt matching
    # This handles parallel same-type subagents where FIFO was ambiguous
    if not matched_tool_use_id:
        pending_prompt_matches = turn_info.get("pending_prompt_matches", {})
        if agent_id and agent_id in pending_prompt_matches:
            pending_info = pending_prompt_matches[agent_id]
            candidate_tool_use_ids = pending_info.get("candidate_tool_use_ids", [])
            _log_error(f"SubagentStop: resolving pending_prompt_match for agent_id={agent_id}, candidates={candidate_tool_use_ids}")

            # Extract subagent's first user message from its transcript
            subagent_prompt = None
            if agent_transcript_path and os.path.exists(agent_transcript_path):
                subagent_prompt = _extract_first_user_message(agent_transcript_path)
                _log_error(f"SubagentStop: extracted subagent_prompt length={len(subagent_prompt) if subagent_prompt else 0}")

            if subagent_prompt and candidate_tool_use_ids:
                active_subagents = turn_info.get("active_subagents", {})
                best_match_id = None
                best_similarity = 0.0

                # Score each candidate Task prompt against the subagent's actual prompt
                for tool_use_id in candidate_tool_use_ids:
                    info = active_subagents.get(tool_use_id, {})
                    task_prompt = info.get("prompt", "")
                    if not task_prompt:
                        continue

                    similarity = compute_prompt_similarity(subagent_prompt, task_prompt)
                    _log_error(f"SubagentStop: similarity({tool_use_id})={similarity:.3f}")

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = tool_use_id

                # Use prompt match if similarity is above threshold (0.1 = 10% keyword overlap)
                if best_match_id and best_similarity >= 0.1:
                    matched_tool_use_id = best_match_id
                    info = active_subagents.get(matched_tool_use_id, {})
                    subagent_type = info.get("type", "Subagent")
                    info["assigned_agent_id"] = agent_id
                    _log_error(f"SubagentStop: PROMPT MATCHED agent_id={agent_id} -> tool_use_id={matched_tool_use_id}, similarity={best_similarity:.3f}")

                    # Update agent_to_task_mapping now that we have a definitive match
                    agent_to_task_mapping[agent_id] = matched_tool_use_id
                    # Remove from pending
                    del pending_prompt_matches[agent_id]
                    update_turn_state(session_id, {
                        "active_subagents": active_subagents,
                        "agent_to_task_mapping": agent_to_task_mapping,
                        "pending_prompt_matches": pending_prompt_matches,
                    })
                else:
                    _log_error(f"SubagentStop: prompt match below threshold (best={best_similarity:.3f}), falling back to FIFO")
                    # FIFO fallback: use first candidate
                    if candidate_tool_use_ids:
                        matched_tool_use_id = candidate_tool_use_ids[0]
                        info = active_subagents.get(matched_tool_use_id, {})
                        subagent_type = info.get("type", "Subagent")
                        info["assigned_agent_id"] = agent_id
                        agent_to_task_mapping[agent_id] = matched_tool_use_id
                        del pending_prompt_matches[agent_id]
                        update_turn_state(session_id, {
                            "active_subagents": active_subagents,
                            "agent_to_task_mapping": agent_to_task_mapping,
                            "pending_prompt_matches": pending_prompt_matches,
                        })
                        _log_error(f"SubagentStop: FIFO FALLBACK agent_id={agent_id} -> tool_use_id={matched_tool_use_id}")

    # STRATEGY 2: Use get_active_handoffs from session state (fallback for single subagent)
    if not matched_tool_use_id:
        active_handoffs = get_active_handoffs(session_id)
        _log_error(f"SubagentStop: found {len(active_handoffs)} active handoffs")

        if len(active_handoffs) == 1:
            # Single subagent case - use this handoff's type
            handoff = active_handoffs[0]
            subagent_type = handoff.get("subagent_type", "Subagent")
            matched_tool_use_id = handoff.get("tool_use_id")
            _log_error(f"SubagentStop: single handoff, type={subagent_type}")
        elif len(active_handoffs) > 1:
            # Multiple parallel subagents - try prompt matching (fallback if strategies 0/1 failed)
            _log_error(f"SubagentStop: {len(active_handoffs)} parallel subagents, trying prompt match")
            if agent_transcript_path and os.path.exists(agent_transcript_path):
                subagent_prompt = _extract_first_user_message(agent_transcript_path)
                if subagent_prompt:
                    # Get prompts from active_subagents in turn_info (includes stored prompts)
                    active_subagents = turn_info.get("active_subagents", {})
                    best_match_id = None
                    best_similarity = 0.0

                    for tool_use_id, info in active_subagents.items():
                        stored_prompt = info.get("prompt", "")
                        if not stored_prompt:
                            continue

                        # Use fuzzy matching instead of substring check
                        similarity = compute_prompt_similarity(subagent_prompt, stored_prompt)
                        _log_error(f"SubagentStop: fallback similarity({tool_use_id})={similarity:.3f}")

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_id = tool_use_id

                    if best_match_id and best_similarity >= 0.1:
                        subagent_type = active_subagents[best_match_id].get("type", "Subagent")
                        matched_tool_use_id = best_match_id
                        _log_error(f"SubagentStop: fallback PROMPT MATCHED, type={subagent_type}, similarity={best_similarity:.3f}")

            # Fallback for parallel: use first active handoff (LAST RESORT)
            if not matched_tool_use_id:
                handoff = active_handoffs[0]
                subagent_type = handoff.get("subagent_type", "Subagent")
                matched_tool_use_id = handoff.get("tool_use_id")
                _log_error(f"SubagentStop: LAST RESORT fallback to first handoff, type={subagent_type}")

    # Extract the subagent's result from its transcript (WITH RETRY for race condition)
    # The transcript may not be fully written to disk when SubagentStop fires
    subagent_result = None
    subagent_tool_use_ids = []
    if agent_transcript_path:
        # Use retry logic - file may not exist or be incomplete yet
        subagent_result = _extract_subagent_result(agent_transcript_path)
        # CRITICAL: Extract ALL tool_use_ids from the subagent transcript
        # This is the AUTHORITATIVE source for tool call attribution
        if os.path.exists(agent_transcript_path):
            subagent_tool_use_ids = _extract_tool_use_ids_from_transcript(agent_transcript_path)
        _log_error(f"SubagentStop: extracted result length={len(subagent_result) if subagent_result else 0} from {agent_transcript_path}")
        _log_error(f"SubagentStop: extracted {len(subagent_tool_use_ids)} tool_use_ids: {subagent_tool_use_ids}")

        # FALLBACK: If subagent_type is still generic, try to extract from transcript
        if subagent_type == "Subagent" and os.path.exists(agent_transcript_path):
            extracted_type = _extract_agent_type_from_transcript(agent_transcript_path)
            if extracted_type:
                subagent_type = extracted_type
                _log_error(f"SubagentStop: extracted agent_type from transcript: {subagent_type}")

    # Store both the transcript path AND the extracted result for _handle_task_complete
    # This is critical because PostToolUse for Task doesn't include tool_output
    if agent_transcript_path or subagent_result:
        turn_info["last_subagent_transcript"] = agent_transcript_path
        turn_info["last_subagent_result"] = subagent_result
        turn_info["last_subagent_id"] = agent_id
        update_turn_state(session_id, {
            "last_subagent_transcript": agent_transcript_path,
            "last_subagent_result": subagent_result,
            "last_subagent_id": agent_id,
        })
        _log_error(f"SubagentStop: stored agent_id={agent_id}, transcript={agent_transcript_path}")

    # Build the full agent_id with type: claude_code:subagent:{type}:{id}
    if agent_id:
        full_agent_id = f"claude_code:subagent:{subagent_type}:{agent_id}"
    else:
        full_agent_id = f"claude_code:main:{session_id}"

    # Use the actual subagent_type for agent_role (not generic "subagent")
    # This ensures correct lane assignment on the frontend timeline
    # Truncate very large results to keep payloads reasonable
    # WAF SizeRestrictions_BODY is excluded for ingest, so we can go larger
    MAX_RESULT_SIZE = 20000  # 20KB - captures most subagent results
    result_preview = None
    result_truncated = False
    if subagent_result:
        if len(subagent_result) > MAX_RESULT_SIZE:
            result_preview = subagent_result[:MAX_RESULT_SIZE] + f"\n\n[... truncated, full length: {len(subagent_result)} chars]"
            result_truncated = True
        else:
            result_preview = subagent_result

    # CRITICAL: Emit tool.call.attribute events for EACH tool call this subagent made
    # This corrects the initial attribution from PreToolUse (which used LAST RESORT fallback)
    # The frontend should use these events to re-attribute tool calls to the correct subagent lane
    if subagent_tool_use_ids and matched_tool_use_id:
        handoff_key = generate_handoff_key(session_id, matched_tool_use_id)
        _log_error(f"SubagentStop: emitting tool.call.attribute for {len(subagent_tool_use_ids)} tools -> {full_agent_id}")
        for tool_uid in subagent_tool_use_ids:
            emit_with_seq_sync(session_id, run, normalize_event(
                run,
                event_type="tool.call.attribute",
                agent_id=full_agent_id,
                agent_role=subagent_type if agent_id else "main",
                summary=f"Tool {tool_uid} belongs to {subagent_type}",
                attrs={
                    "tool_use_id": tool_uid,
                    "subagent_id": agent_id,
                    "subagent_type": subagent_type,
                    "task_tool_use_id": matched_tool_use_id,
                    "handoff_key": handoff_key,
                },
                payload={},
            ))

    emit_with_seq_sync(session_id, run, normalize_event(
        run,
        event_type="agent.end",
        agent_id=full_agent_id,
        agent_role=subagent_type if agent_id else "main",
        summary="Subagent execution completed",
        attrs={
            "subagent_id": agent_id,
            "subagent_type": subagent_type,
            "result_truncated": result_truncated,
            "result_full_length": len(subagent_result) if subagent_result else 0,
            # CRITICAL: Include the matched Task tool_use_id for parent correlation
            "task_tool_use_id": matched_tool_use_id,
        },
        payload={
            "agent_transcript_path": agent_transcript_path,
            "result_preview": result_preview,
            # AUTHORITATIVE: List of ALL tool_use_ids made by this subagent
            # Frontend can use this to definitively attribute tool calls to subagent lanes
            "tool_use_ids": subagent_tool_use_ids,
        },
    ))

    _flush_run(run, session_id)


def handle_subagent_start(input_data: dict) -> None:
    """Handle SubagentStart hook - subagent begins execution.

    Added in Claude Code v2.0.43. This hook fires when a subagent starts executing.

    CRITICAL: SubagentStart provides agent_id and agent_type, but NOT tool_use_id.
    We use TIMING CORRELATION to match this subagent to its parent Task:
    1. PreToolUse[Task] fires first -> creates handoff with tool_use_id + subagent_type
    2. SubagentStart fires shortly after -> provides agent_id + agent_type

    RACE CONDITION FIX: When multiple same-type subagents run in parallel, FIFO matching
    is unreliable because hook delivery order is not guaranteed. In this case:
    - Mark the agent as "pending_prompt_match"
    - Defer definitive mapping to SubagentStop where we have the subagent's transcript
    - SubagentStop will use compute_prompt_similarity() to match tool events to Task prompts

    For single subagents of a type, FIFO matching is used (reliable when unambiguous).
    """
    session_id = input_data["session_id"]

    # SubagentStart provides: agent_id, agent_type (NOT tool_use_id or subagent_type!)
    agent_id = input_data.get("agent_id")
    agent_type = input_data.get("agent_type")  # This is what Claude Code sends

    _log_error(f"SubagentStart: agent_id={agent_id}, agent_type={agent_type}")

    if not agent_id or not agent_type:
        _log_error(f"SubagentStart: missing agent_id or agent_type, cannot build mapping")
        return

    turn_info = get_current_turn(session_id)
    if not turn_info:
        _log_error(f"SubagentStart: no turn_info for session {session_id}")
        return

    # TIMING CORRELATION: Match to unassigned Task handoff of matching type
    # active_subagents tracks Tasks that haven't been assigned an agent_id yet
    active_subagents = turn_info.get("active_subagents", {})
    agent_to_task_mapping = turn_info.get("agent_to_task_mapping", {})
    pending_prompt_matches = turn_info.get("pending_prompt_matches", {})

    # Count unassigned Tasks of matching type to detect ambiguous case
    unassigned_same_type = []
    for tool_use_id, info in active_subagents.items():
        subagent_type = info.get("type", "")
        already_assigned = info.get("assigned_agent_id")
        if subagent_type == agent_type and not already_assigned:
            unassigned_same_type.append(tool_use_id)

    _log_error(f"SubagentStart: found {len(unassigned_same_type)} unassigned Tasks of type={agent_type}: {unassigned_same_type}")

    matched_tool_use_id = None

    # SINGLE SUBAGENT of this type: Use FIFO matching (reliable when unambiguous)
    if len(unassigned_same_type) == 1:
        matched_tool_use_id = unassigned_same_type[0]
        info = active_subagents[matched_tool_use_id]
        info["assigned_agent_id"] = agent_id
        _log_error(f"SubagentStart: FIFO MATCHED (single) agent_id={agent_id} to tool_use_id={matched_tool_use_id}")

        # Store the mapping: agent_id -> task_tool_use_id
        agent_to_task_mapping[agent_id] = matched_tool_use_id
        update_turn_state(session_id, {
            "active_subagents": active_subagents,
            "agent_to_task_mapping": agent_to_task_mapping,
        })
        _log_error(f"SubagentStart: stored FIFO mapping agent_id={agent_id} -> tool_use_id={matched_tool_use_id}")

    # MULTIPLE SUBAGENTS of same type: Defer to SubagentStop for prompt matching
    elif len(unassigned_same_type) > 1:
        _log_error(f"SubagentStart: AMBIGUOUS - {len(unassigned_same_type)} same-type Tasks, deferring to prompt match")

        # Mark this agent as pending prompt match - SubagentStop will resolve it
        pending_prompt_matches[agent_id] = {
            "agent_type": agent_type,
            "candidate_tool_use_ids": unassigned_same_type,
            "started_at": _now_iso(),
        }
        update_turn_state(session_id, {
            "pending_prompt_matches": pending_prompt_matches,
        })
        _log_error(f"SubagentStart: marked agent_id={agent_id} as pending_prompt_match, candidates={unassigned_same_type}")

    else:
        _log_error(f"SubagentStart: NO MATCH found for agent_type={agent_type}, active_subagents={list(active_subagents.keys())}")


def handle_pre_compact(input_data: dict) -> None:
    """Handle PreCompact hook - context window compaction.

    Context compaction is a critical signal for potential context drift.
    When compaction occurs:
    1. Earlier conversation context is summarized/lost
    2. The agent may lose track of earlier goals or constraints
    3. Repeated compactions indicate very long tasks (higher drift risk)

    We track compaction count per turn to help detect drift patterns.
    """
    session_id = input_data["session_id"]

    turn_info = get_current_turn(session_id)
    if not turn_info:
        return

    run = turn_info["run"]

    # Track compaction count for drift detection
    compaction_count = turn_info.get("compaction_count", 0) + 1
    turn_info["compaction_count"] = compaction_count
    update_turn_state(session_id, {"compaction_count": compaction_count})

    # Determine drift risk level based on compaction count
    if compaction_count >= 3:
        drift_risk = "high"
        summary = f"Context compaction #{compaction_count} - HIGH drift risk"
    elif compaction_count == 2:
        drift_risk = "medium"
        summary = f"Context compaction #{compaction_count} - potential drift"
    else:
        drift_risk = "low"
        summary = "Context window compacted due to token limit"

    emit_with_seq_sync(session_id, run, normalize_event(
        run,
        event_type="context.compact",
        agent_id=f"claude_code:main:{session_id}",
        agent_role="main",
        summary=summary,
        status="warn" if drift_risk in ("medium", "high") else "ok",
        attrs={
            "reason": input_data.get("reason", "token_limit"),
            "compaction_count": compaction_count,
            "drift_risk": drift_risk,
        },
        payload={
            "tool_count_so_far": len(turn_info.get("tool_calls", [])),
            "active_subagents": list(turn_info.get("active_subagents", {}).keys()),
        },
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

    emit_with_seq_sync(session_id, run, normalize_event(
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

def _extract_final_response(transcript_path: str) -> Optional[str]:
    """
    Extract the final assistant response from the main session transcript.

    This is used to capture Claude's final response for a turn.

    Args:
        transcript_path: Path to the session's JSONL transcript file

    Returns:
        The final text content from the last assistant message, or None
    """
    # Reuse the same extraction logic as subagent results
    return _extract_subagent_result(transcript_path)


def _extract_subagent_result(
    transcript_path: str,
    max_retries: int = 5,
    retry_delays: tuple[float, ...] = (0.0, 0.1, 0.2, 0.5, 1.0),
) -> Optional[str]:
    """
    Extract the subagent's final result from its transcript.

    The subagent transcript is a JSONL file where each line is a message.
    The result is the text content of the last assistant message.

    NOTE: There's a race condition where SubagentStop fires BEFORE the transcript
    is fully written to disk. We retry with exponential backoff to handle this.
    The transcript file may exist but be empty or incomplete on first attempts.

    Args:
        transcript_path: Path to the subagent's JSONL transcript file
        max_retries: Maximum number of retry attempts (default 5)
        retry_delays: Tuple of delays in seconds for each retry attempt
                      (default exponential backoff: 0, 0.1, 0.2, 0.5, 1.0)

    Returns:
        The final text content, or None if not found after all retries
    """
    if not transcript_path:
        _log_error("_extract_subagent_result: transcript_path is empty/None")
        return None

    for attempt in range(max_retries):
        # Apply delay for this attempt (first attempt has 0 delay)
        delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
        if delay > 0:
            _log_error(f"_extract_subagent_result: retry {attempt + 1}/{max_retries} after {delay}s delay")
            time.sleep(delay)

        # Check if file exists on each attempt (may not exist yet due to race condition)
        if not os.path.exists(transcript_path):
            _log_error(f"_extract_subagent_result: file does not exist (attempt {attempt + 1}/{max_retries}): {transcript_path}")
            if attempt < max_retries - 1:
                continue
            else:
                _log_error(f"_extract_subagent_result: file still missing after {max_retries} retries")
                return None

        last_assistant_text = None
        line_count = 0

        try:
            # Re-open file each attempt to get fresh content (no caching)
            with open(transcript_path, "r") as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Look for assistant messages
                    if msg.get("type") != "assistant":
                        continue

                    message = msg.get("message", {})
                    content = message.get("content", [])

                    if not isinstance(content, list):
                        continue

                    # Extract text blocks from content
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                texts.append(text)

                    # Update last_assistant_text with this message's text
                    if texts:
                        last_assistant_text = "\n".join(texts)

            # If we found a result, return it
            if last_assistant_text:
                _log_error(f"_extract_subagent_result: found result ({len(last_assistant_text)} chars) on attempt {attempt + 1}")
                return last_assistant_text

            # File exists but is empty or has no assistant messages yet
            _log_error(f"_extract_subagent_result: file exists but no assistant content found (attempt {attempt + 1}/{max_retries}, {line_count} lines read)")

            if attempt >= max_retries - 1:
                _log_error(f"_extract_subagent_result: no result after {max_retries} retries, giving up")
                return None

        except Exception as e:
            _log_error(f"_extract_subagent_result: error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt >= max_retries - 1:
                _log_error(f"Failed to extract subagent result from {transcript_path}: {e}")
                return None

    return None


def _extract_tool_use_ids_from_transcript(transcript_path: str) -> list[str]:
    """
    Extract all tool_use_ids from a subagent transcript.

    This is the AUTHORITATIVE source for which tool calls a subagent made.
    The subagent transcript contains all tool_use entries with their ids.

    Args:
        transcript_path: Path to the subagent's JSONL transcript file

    Returns:
        List of tool_use_ids found in the transcript
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return []

    tool_use_ids = []

    try:
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Look for tool_use in message content
                message = msg.get("message", {})
                content = message.get("content", [])

                if not isinstance(content, list):
                    continue

                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_id = item.get("id")
                        if tool_id and tool_id not in tool_use_ids:
                            tool_use_ids.append(tool_id)

    except Exception as e:
        _log_error(f"Failed to extract tool_use_ids from {transcript_path}: {e}")

    return tool_use_ids


def _extract_agent_type_from_transcript(transcript_path: str) -> Optional[str]:
    """
    Extract the agent type from a subagent's transcript.

    Looks for subagent_type field or parses the prompt to determine type.

    Args:
        transcript_path: Path to the subagent's JSONL transcript file

    Returns:
        The agent type (e.g., "search-specialist", "research-analyst"), or None
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Look for subagent_type or agent_type fields directly
                for field in ["subagent_type", "agent_type", "agentType"]:
                    if field in msg:
                        return msg[field]

                # Check in nested message
                message = msg.get("message", {})
                for field in ["subagent_type", "agent_type"]:
                    if field in message:
                        return message[field]

                # Look for first user message content that might contain type info
                # The Task prompt usually contains the subagent_type
                if msg.get("type") == "user":
                    content = msg.get("content", "") or msg.get("message", {}).get("content", "")
                    if isinstance(content, str) and content.strip():
                        # First line of user message might be the agent role description
                        first_line = content.split('\n')[0].lower()
                        # Common agent types
                        agent_types = [
                            "search-specialist", "research-analyst", "market-researcher",
                            "code-reviewer", "security-auditor", "debugger", "Explore",
                            "frontend-developer", "backend-developer", "fullstack-developer",
                        ]
                        for at in agent_types:
                            if at.lower() in first_line:
                                return at
                    break  # Only check first user message

    except Exception as e:
        _log_error(f"Failed to extract agent type from {transcript_path}: {e}")

    return None


def _extract_first_user_message(transcript_path: str) -> Optional[str]:
    """
    Extract the first user message from a transcript.

    This is used to correlate SubagentStop with the correct Task tool call
    by matching the subagent's first user message against the Task prompts.

    Args:
        transcript_path: Path to the subagent's JSONL transcript file

    Returns:
        The first user message text, or None if not found
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Look for the first user message
                if msg.get("type") == "user":
                    # User message content can be a string or in message.content
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content

                    message = msg.get("message", {})
                    content = message.get("content", [])

                    if isinstance(content, str):
                        return content

                    if isinstance(content, list):
                        texts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                if text:
                                    texts.append(text)
                            elif isinstance(item, str):
                                texts.append(item)
                        if texts:
                            return "\n".join(texts)

                    # Return after first user message
                    return None

    except Exception as e:
        _log_error(f"Failed to extract first user message from {transcript_path}: {e}")

    return None


def _extract_tool_result_from_transcript(
    transcript_path: str,
    tool_use_id: str,
    max_retries: int = 5,
    retry_delay: float = 0.2,
) -> Optional[str]:
    """
    Extract the result for a specific tool_use_id from the transcript.

    Claude Code's PostToolUse hook for Task tools doesn't include tool_output.
    The result is only available in the transcript as a tool_result message.

    NOTE: There's a race condition where the transcript may not have been fully
    written to disk when this function is called. We retry with small delays
    to handle this.

    Args:
        transcript_path: Path to the JSONL transcript file
        tool_use_id: The tool_use_id to find the result for
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        The tool result text, or None if not found
    """
    if not transcript_path:
        _log_error(f"_extract: transcript_path is empty/None")
        return None
    if not os.path.exists(transcript_path):
        _log_error(f"_extract: file does not exist: {transcript_path}")
        return None

    _log_error(f"_extract: searching {transcript_path} for {tool_use_id}")

    for attempt in range(max_retries):
        line_count = 0
        result = None

        try:
            # Re-open file each attempt to get fresh content (no caching)
            with open(transcript_path, "r") as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Look for tool_result in message content
                    message = msg.get("message", {})
                    content = message.get("content", [])

                    if not isinstance(content, list):
                        continue

                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "tool_result" and item.get("tool_use_id") == tool_use_id:
                                _log_error(f"_extract: FOUND at line {line_count} (attempt {attempt+1})")
                                # Found the tool result
                                result_content = item.get("content", [])
                                if isinstance(result_content, list):
                                    # Extract text from content blocks
                                    texts = []
                                    for block in result_content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            texts.append(block.get("text", ""))
                                    if texts:
                                        result = "\n".join(texts)
                                        _log_error(f"_extract: returning {len(result)} chars")
                                        return result
                                elif isinstance(result_content, str):
                                    _log_error(f"_extract: returning string of {len(result_content)} chars")
                                    return result_content

                    # Also check toolUseResult field (alternative location)
                    tool_use_result = msg.get("toolUseResult")
                    if tool_use_result and msg.get("sourceToolAssistantUUID"):
                        # Check if this message is for our tool_use_id by looking at content
                        for item in content:
                            if isinstance(item, dict) and item.get("tool_use_id") == tool_use_id:
                                _log_error(f"_extract: FOUND via toolUseResult at line {line_count} (attempt {attempt+1})")
                                result_content = tool_use_result.get("content", [])
                                if isinstance(result_content, list):
                                    texts = []
                                    for block in result_content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            texts.append(block.get("text", ""))
                                    if texts:
                                        result = "\n".join(texts)
                                        _log_error(f"_extract: returning {len(result)} chars")
                                        return result
                                break

            if attempt < max_retries - 1:
                _log_error(f"_extract: NOT FOUND after {line_count} lines (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                _log_error(f"_extract: NOT FOUND after {line_count} lines (final attempt)")

        except Exception as e:
            _log_error(f"Failed to extract tool result from transcript (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return None


def _get_current_agent_id(
    session_id: str,
    turn_info: Optional[dict] = None,
    tool_input: Optional[dict] = None,
) -> str:
    """Get the current agent ID based on span context.

    If we're inside a subagent delegation (Task span), return the subagent's ID.
    For parallel subagents, we use content-based matching to identify the correct one.

    Args:
        session_id: Claude Code session ID
        turn_info: Current turn state
        tool_input: Tool input dict (used for content-based matching with parallel subagents)
    """
    if turn_info:
        spans = turn_info.get("spans", [])
        subagent_spans = [s for s in spans if s.get("subagent_type")]

        if not subagent_spans:
            return f"claude_code:main:{session_id}"

        # SINGLE SUBAGENT: Just use it
        if len(subagent_spans) == 1:
            span = subagent_spans[0]
            return f"claude_code:subagent:{span['subagent_type']}:{span['tool_use_id']}"

        # MULTIPLE SUBAGENTS: Try content-based matching
        if tool_input and len(subagent_spans) > 1:
            matched_span = _match_tool_to_subagent(tool_input, turn_info)
            if matched_span:
                return f"claude_code:subagent:{matched_span['subagent_type']}:{matched_span['tool_use_id']}"

        # FALLBACK: Use current_executing_subagent if set
        current_executing = turn_info.get("current_executing_subagent")
        if current_executing:
            for span in subagent_spans:
                if span.get("tool_use_id") == current_executing:
                    return f"claude_code:subagent:{span['subagent_type']}:{span['tool_use_id']}"

        # LAST RESORT: Use most recent subagent span
        span = subagent_spans[-1]
        return f"claude_code:subagent:{span['subagent_type']}:{span['tool_use_id']}"

    return f"claude_code:main:{session_id}"


def _match_tool_to_subagent(
    tool_input: dict,
    turn_info: dict,
    tool_name: Optional[str] = None,
) -> Optional[dict]:
    """Match a tool call to the most relevant subagent based on content.

    Uses keyword matching between the tool input and subagent prompts to
    determine which subagent most likely made this tool call. Also considers
    tool-type affinity and recency weighting for disambiguation.

    FIX for HIGH severity bug: Parallel subagent misattribution
    - Increased score threshold from 2 to 3 for more reliable matching
    - Added tool-type affinity scoring (e.g., Read/Write prefer "file" subagents)
    - Added recency weighting (prefer more recently started subagents on tie)
    - Better logging for ambiguous matching scenarios

    Args:
        tool_input: The tool's input parameters
        turn_info: Current turn state containing active_subagents and spans
        tool_name: Name of the tool (e.g., "Bash", "Read", "Write") for affinity scoring

    Returns:
        The matched span dict, or None if no good match found
    """
    active_subagents = turn_info.get("active_subagents", {})
    spans = turn_info.get("spans", [])

    if not active_subagents or not spans:
        return None

    # Build a string from tool_input for matching
    tool_text = _extract_text_from_tool_input(tool_input).lower()
    if not tool_text:
        return None

    best_match = None
    best_score = 0
    best_started_at = None

    for tool_use_id, info in active_subagents.items():
        prompt = info.get("prompt", "").lower()
        if not prompt:
            continue

        # Calculate base match score based on keyword overlap
        score = _calculate_match_score(tool_text, prompt)

        # Add tool-type affinity bonus
        if tool_name:
            affinity_bonus = _calculate_tool_type_affinity(tool_name, prompt)
            score += affinity_bonus

        # Get started_at for recency comparison
        started_at = info.get("started_at")

        # Prefer higher scores, break ties by recency (more recent = preferred)
        is_better = False
        if score > best_score:
            is_better = True
        elif score == best_score and score > 0:
            # Same score - prefer more recently started subagent
            if started_at and best_started_at:
                if started_at > best_started_at:
                    is_better = True
                    _log_error(
                        f"_match_tool_to_subagent: tie-break by recency, "
                        f"preferring {tool_use_id} (started {started_at}) over previous (started {best_started_at})"
                    )
            elif started_at and not best_started_at:
                # Prefer the one with timestamp info
                is_better = True

        if is_better:
            best_score = score
            best_started_at = started_at
            # Find the corresponding span
            for span in spans:
                if span.get("tool_use_id") == tool_use_id:
                    best_match = span
                    break

    # Increased threshold from 2 to 3 for more reliable matching
    # Score >= 3 means at least 3 keyword group matches OR 2 groups + affinity bonus
    if best_score >= 3:
        _log_error(
            f"_match_tool_to_subagent: matched with score={best_score}, "
            f"tool_use_id={best_match.get('tool_use_id') if best_match else None}"
        )
        return best_match

    # Log when score is too low for matching (helps debug misattribution)
    if best_score > 0:
        _log_error(
            f"_match_tool_to_subagent: best_score={best_score} below threshold (3), "
            f"returning None for ambiguous match"
        )

    return None


def _calculate_tool_type_affinity(tool_name: str, prompt: str) -> int:
    """Calculate affinity bonus based on tool type and subagent description.

    Returns a bonus score if the subagent's prompt indicates it should use
    this type of tool. This helps disambiguate when 2+ parallel subagents
    run the same tool type (e.g., both running Bash or Read).

    Args:
        tool_name: Name of the tool (e.g., "Bash", "Read", "Write", "Grep", "Glob")
        prompt: The subagent's prompt text (lowercase)

    Returns:
        Affinity bonus score (0-2)
    """
    # Tool-type affinity mappings
    # If subagent prompt mentions certain keywords, it has affinity for related tools
    affinities = {
        # File operation tools
        "Read": ["file", "read", "document", "content", "config", "yaml", "json", "text", "code", "source"],
        "Write": ["file", "write", "create", "update", "modify", "edit", "save", "output", "generate"],
        "Edit": ["edit", "modify", "update", "change", "fix", "refactor", "code", "patch"],
        "Glob": ["file", "find", "search", "pattern", "directory", "folder", "locate", "list"],
        "Grep": ["search", "find", "grep", "pattern", "content", "text", "code", "match"],
        # Command execution tools
        "Bash": ["command", "shell", "terminal", "execute", "run", "script", "npm", "pip", "git", "docker", "kubectl", "test", "build"],
        # Web/API tools
        "WebFetch": ["web", "http", "api", "fetch", "url", "request", "download", "endpoint"],
        "WebSearch": ["search", "web", "google", "internet", "find", "query", "research"],
        # Task tool (subagent delegation)
        "Task": ["delegate", "subagent", "agent", "specialist", "expert"],
    }

    # Normalize tool name
    tool_key = tool_name

    if tool_key not in affinities:
        return 0

    keywords = affinities[tool_key]
    matches = sum(1 for kw in keywords if kw in prompt)

    # Return bonus: 1 point for 1-2 matches, 2 points for 3+ matches
    if matches >= 3:
        return 2
    elif matches >= 1:
        return 1
    return 0


def _extract_text_from_tool_input(tool_input: dict) -> str:
    """Extract searchable text from tool input."""
    texts = []
    for _, value in tool_input.items():
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            texts.append(_extract_text_from_tool_input(value))
    return " ".join(texts)


def _calculate_match_score(tool_text: str, prompt: str) -> int:
    """Calculate how well tool_text matches a subagent prompt.

    Uses keyword matching for common AWS/security terms.
    """
    # Common keywords that indicate specific subagent domains
    keyword_groups = [
        # IAM related
        ["iam", "user", "role", "policy", "permission", "access key", "mfa"],
        # S3 related
        ["s3", "bucket", "object", "storage", "versioning"],
        # Database related
        ["rds", "database", "aurora", "dynamodb", "mysql", "postgres"],
        # Network related
        ["vpc", "subnet", "security group", "nacl", "route", "gateway", "elb", "alb"],
        # Logging/monitoring related
        ["cloudtrail", "cloudwatch", "guardduty", "log", "alarm", "metric", "config"],
        # Lambda/API related
        ["lambda", "api gateway", "function", "serverless"],
        # KMS/Secrets related
        ["kms", "key", "encrypt", "secret", "credential"],
    ]

    score = 0
    for keywords in keyword_groups:
        # Check if any keyword from this group appears in BOTH tool_text and prompt
        tool_matches = any(kw in tool_text for kw in keywords)
        prompt_matches = any(kw in prompt for kw in keywords)
        if tool_matches and prompt_matches:
            score += 1

    return score


def compute_prompt_similarity(prompt1: str, prompt2: str) -> float:
    """Compute similarity between two prompts using keyword extraction and Jaccard similarity.

    This is used to match subagent tool events to their parent Task prompts when
    multiple same-type subagents are running in parallel (FIFO matching is ambiguous).

    Args:
        prompt1: First prompt text
        prompt2: Second prompt text

    Returns:
        Similarity score between 0.0 and 1.0 (Jaccard similarity of extracted keywords)
    """
    import re

    if not prompt1 or not prompt2:
        return 0.0

    def extract_keywords(text: str) -> set:
        """Extract significant keywords from text.

        Filters out common stop words and short words, keeping nouns/phrases
        that are likely to be distinctive.
        """
        # Common stop words to filter out
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "it", "its", "they", "them", "their",
            "we", "us", "our", "you", "your", "i", "me", "my", "he", "she", "his",
            "her", "what", "which", "who", "whom", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "same", "so", "than", "too", "very",
            "just", "also", "now", "here", "there", "then", "once", "if", "else",
            "use", "using", "used", "make", "makes", "made", "get", "gets", "got",
            "any", "about", "into", "over", "after", "before", "between", "under",
            "again", "further", "then", "once", "during", "while", "through",
        }

        # Normalize text: lowercase and extract words
        text_lower = text.lower()
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-z][a-z0-9_-]*\b', text_lower)

        # Filter: keep words that are not stop words and have length >= 3
        keywords = {w for w in words if w not in stop_words and len(w) >= 3}

        return keywords

    keywords1 = extract_keywords(prompt1)
    keywords2 = extract_keywords(prompt2)

    if not keywords1 or not keywords2:
        return 0.0

    # Jaccard similarity: |intersection| / |union|
    intersection = keywords1 & keywords2
    union = keywords1 | keywords2

    similarity = len(intersection) / len(union) if union else 0.0

    return similarity


def _get_current_agent_role(
    turn_info: Optional[dict] = None,
    tool_input: Optional[dict] = None,
) -> str:
    """Get the current agent role based on span context.

    If we're inside a subagent delegation, return that subagent's role.
    For parallel subagents, we use content-based matching to identify the correct one.
    """
    if turn_info:
        spans = turn_info.get("spans", [])
        subagent_spans = [s for s in spans if s.get("subagent_type")]

        if not subagent_spans:
            return "main"

        # SINGLE SUBAGENT: Just use it
        if len(subagent_spans) == 1:
            return subagent_spans[0]["subagent_type"]

        # MULTIPLE SUBAGENTS: Try content-based matching
        if tool_input and len(subagent_spans) > 1:
            matched_span = _match_tool_to_subagent(tool_input, turn_info)
            if matched_span:
                return matched_span["subagent_type"]

        # FALLBACK: Use current_executing_subagent if set
        current_executing = turn_info.get("current_executing_subagent")
        if current_executing:
            for span in subagent_spans:
                if span.get("tool_use_id") == current_executing:
                    return span["subagent_type"]

        # LAST RESORT: Use most recent subagent span
        return subagent_spans[-1]["subagent_type"]

    return "main"


def _flush_run(run: Any, _session_id: str = None) -> None:
    """Flush the run's sink.

    NOTE: We do NOT persist turn state here anymore!
    emit_with_seq_sync already persists current_seq to disk atomically.
    Persisting from _flush_run caused race conditions in parallel tool calls
    because each process has its own in-memory _active_turns with stale seq values.
    """
    try:
        run.sink.flush()
    except Exception:
        pass


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _log_error(message: str) -> None:
    """Log error to file (hooks run as subprocesses, stderr doesn't reach terminal)."""
    try:
        from pathlib import Path
        log_file = Path.home() / ".arzule" / "hook_debug.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"[{_now_iso()}] {message}\n")
    except Exception:
        pass


if __name__ == "__main__":
    handle_hook()
