"""Session state management for Claude Code instrumentation.

Manages ArzuleRun instances and session state across hook invocations.
State is persisted to disk since each hook is a separate process invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Optional

# Lazy imports to avoid circular dependencies
_ArzuleRun = None
_HttpBatchSink = None
_JsonlFileSink = None

_session_lock = Lock()
_sessions: dict[str, Any] = {}
_session_state: dict[str, dict] = {}

# State file for persistence across hook invocations
STATE_DIR = Path.home() / ".arzule" / "claude_sessions"


def _get_imports():
    """Lazy import to avoid circular dependencies."""
    global _ArzuleRun, _HttpBatchSink, _JsonlFileSink
    if _ArzuleRun is None:
        from ..run import ArzuleRun
        from ..sinks import HttpBatchSink, JsonlFileSink
        _ArzuleRun = ArzuleRun
        _HttpBatchSink = HttpBatchSink
        _JsonlFileSink = JsonlFileSink


def get_or_create_session(session_id: str) -> Any:
    """
    Get existing run or create new one for session.

    Args:
        session_id: Claude Code session identifier

    Returns:
        ArzuleRun instance for this session
    """
    _get_imports()

    with _session_lock:
        if session_id in _sessions:
            return _sessions[session_id]

        # Load persisted state if exists
        state = _load_session_state(session_id)

        # Create run with appropriate sink
        api_key = os.environ.get("ARZULE_API_KEY")
        tenant_id = os.environ.get("ARZULE_TENANT_ID")
        project_id = os.environ.get("ARZULE_PROJECT_ID")

        if api_key and tenant_id and project_id:
            # Production endpoint (hardcoded, can override via env var for testing)
            endpoint = os.environ.get(
                "ARZULE_ENDPOINT",
                "https://dh43xnx5e03pq.cloudfront.net/ingest"
            )
            sink = _HttpBatchSink(
                endpoint_url=endpoint,
                api_key=api_key,
            )
        else:
            # Fallback to local file
            traces_dir = Path.home() / ".arzule" / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
            sink = _JsonlFileSink(str(traces_dir / f"claude_{session_id}.jsonl"))

        # Create ArzuleRun directly for Claude Code sessions
        run_id = state.get("run_id") or _session_to_run_id(session_id)
        run = _ArzuleRun(
            run_id=run_id,
            tenant_id=tenant_id or "local",
            project_id=project_id or "claude_code",
            sink=sink,
        )
        # Enter the run context to initialize it
        run.__enter__()

        _sessions[session_id] = run
        _session_state[session_id] = state

        return run


def close_session(session_id: str) -> None:
    """
    Close and cleanup a session.

    Args:
        session_id: Claude Code session identifier
    """
    with _session_lock:
        if session_id in _sessions:
            run = _sessions.pop(session_id)
            try:
                # Flush the sink to send any buffered events
                run.sink.flush()
            except Exception:
                pass

        if session_id in _session_state:
            # Keep state file for potential resume
            _session_state.pop(session_id)


def get_session_state(session_id: str) -> dict:
    """
    Get session state (handoffs, spans, etc.).

    Args:
        session_id: Claude Code session identifier

    Returns:
        Session state dictionary
    """
    if session_id not in _session_state:
        _session_state[session_id] = _load_session_state(session_id)
    return _session_state.get(session_id, {})


def update_session_state(session_id: str, updates: dict) -> None:
    """
    Update session state and persist.

    Args:
        session_id: Claude Code session identifier
        updates: Dictionary of updates to merge into state
    """
    if session_id not in _session_state:
        _session_state[session_id] = _load_session_state(session_id)

    _session_state[session_id].update(updates)
    _persist_session_state(session_id, _session_state[session_id])


def set_active_subagent(session_id: str, tool_use_id: str, subagent_type: str) -> None:
    """
    Track an active subagent for the session.

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the Task call
        subagent_type: Type of subagent (Explore, Plan, Bash, etc.)
    """
    state = get_session_state(session_id)
    if "active_subagents" not in state:
        state["active_subagents"] = {}

    state["active_subagents"][tool_use_id] = {
        "type": subagent_type,
        "started_at": _now_iso(),
    }

    update_session_state(session_id, {"active_subagents": state["active_subagents"]})


def get_active_subagent(session_id: str, tool_use_id: str) -> Optional[dict]:
    """
    Get active subagent info.

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the Task call

    Returns:
        Subagent info dict or None
    """
    state = get_session_state(session_id)
    return state.get("active_subagents", {}).get(tool_use_id)


def remove_active_subagent(session_id: str, tool_use_id: str) -> Optional[dict]:
    """
    Remove an active subagent (when completed).

    Args:
        session_id: Claude Code session identifier
        tool_use_id: Tool use ID for the Task call

    Returns:
        Removed subagent info or None
    """
    state = get_session_state(session_id)
    if "active_subagents" in state and tool_use_id in state["active_subagents"]:
        info = state["active_subagents"].pop(tool_use_id)
        update_session_state(session_id, {"active_subagents": state["active_subagents"]})
        return info
    return None


def _session_to_run_id(session_id: str) -> str:
    """
    Convert Claude Code session_id to Arzule run_id (UUID format).

    Creates a deterministic UUID from the session_id for consistent
    mapping across hook invocations.

    Args:
        session_id: Claude Code session identifier

    Returns:
        UUID string for run_id
    """
    hash_bytes = hashlib.sha256(session_id.encode()).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))


def _load_session_state(session_id: str) -> dict:
    """Load persisted session state."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"{session_id}.json"

    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            return {}
    return {}


def _persist_session_state(session_id: str, state: dict) -> None:
    """Persist session state to disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"{session_id}.json"

    try:
        state_file.write_text(json.dumps(state, default=str))
    except Exception:
        pass  # Best effort persistence


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
