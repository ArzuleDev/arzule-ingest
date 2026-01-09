"""Turn-based run management for Claude Code.

A "turn" is a UserPromptSubmit -> Stop cycle. Each turn becomes a separate
Arzule run, allowing proper boundaries for analysis.

This replaces the session-based approach where entire sessions (which can
span hours/days) were treated as a single run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

# State directory for persistence across hook invocations
STATE_DIR = Path.home() / ".arzule" / "claude_state"

_turn_lock = Lock()
_active_turns: dict[str, Any] = {}  # session_id -> current turn info


def _get_imports():
    """Lazy import to avoid circular dependencies."""
    from ..run import ArzuleRun
    from ..sinks import HttpBatchSink, JsonlFileSink, MultiSink
    from .stream_sink import StreamingSink
    return ArzuleRun, HttpBatchSink, JsonlFileSink, MultiSink, StreamingSink


def start_turn(session_id: str, prompt_summary: str = "") -> dict:
    """
    Start a new turn (run) for a Claude Code session.
    
    Called when UserPromptSubmit is received. Creates a new ArzuleRun
    for this turn, allowing each user interaction to be a discrete unit.
    
    Args:
        session_id: Claude Code session identifier
        prompt_summary: Optional summary of the user prompt (for logging)
        
    Returns:
        Turn info dict with run, turn_id, etc.
    """
    ArzuleRun, HttpBatchSink, JsonlFileSink, MultiSink, StreamingSink = _get_imports()
    
    with _turn_lock:
        # Generate unique turn_id
        turn_ts = int(time.time() * 1000)
        turn_id = f"{session_id}:{turn_ts}"
        run_id = _turn_to_run_id(turn_id)
        
        # Load config
        api_key = os.environ.get("ARZULE_API_KEY")
        tenant_id = os.environ.get("ARZULE_TENANT_ID")
        project_id = os.environ.get("ARZULE_PROJECT_ID")
        stream_url = os.environ.get("ARZULE_STREAM_URL")  # Optional local streaming
        
        sinks = []
        
        # Primary sink: Arzule backend (if configured)
        if api_key and tenant_id and project_id:
            endpoint = os.environ.get(
                "ARZULE_ENDPOINT",
                "https://dh43xnx5e03pq.cloudfront.net/ingest"
            )
            sinks.append(HttpBatchSink(
                endpoint_url=endpoint,
                api_key=api_key,
            ))
        
        # Secondary sink: Local streaming server (like reference repo)
        if stream_url:
            sinks.append(StreamingSink(stream_url, session_id))
        
        # Fallback: Local file
        if not sinks:
            traces_dir = Path.home() / ".arzule" / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
            sinks.append(JsonlFileSink(str(traces_dir / f"turn_{turn_id.replace(':', '_')}.jsonl")))
        
        # Use MultiSink if multiple sinks
        if len(sinks) == 1:
            sink = sinks[0]
        else:
            sink = MultiSink(sinks)
        
        # Create ArzuleRun for this turn
        run = ArzuleRun(
            run_id=run_id,
            tenant_id=tenant_id or "local",
            project_id=project_id or "claude_code",
            sink=sink,
        )
        run.__enter__()
        
        # Store turn info (including seq counter for persistence)
        turn_info = {
            "turn_id": turn_id,
            "run_id": run_id,
            "run": run,
            "session_id": session_id,
            "started_at": _now_iso(),
            "prompt_summary": prompt_summary,
            "tool_calls": [],
            "spans": [],
            "current_seq": 0,  # Track seq counter for cross-process persistence
        }
        
        _active_turns[session_id] = turn_info
        _persist_turn_state(session_id, turn_info)
        
        return turn_info


def get_current_turn(session_id: str) -> Optional[dict]:
    """
    Get the current active turn for a session.
    
    Returns None if no turn is active (session started but no prompt yet).
    """
    with _turn_lock:
        if session_id in _active_turns:
            return _active_turns[session_id]
        
        # Try to load from disk (hooks are separate processes)
        turn_info = _load_turn_state(session_id)
        if turn_info and "turn_id" in turn_info:
            # Recreate the run object
            turn_info["run"] = _recreate_run(turn_info)
            _active_turns[session_id] = turn_info
            return turn_info
        
        return None


def end_turn(session_id: str, summary: str = "") -> Optional[dict]:
    """
    End the current turn and flush all events.
    
    Called when Stop is received. Properly closes the run context manager
    which emits run.end, flushes the sink, and unregisters from the global registry.
    
    Args:
        session_id: Claude Code session identifier
        summary: Optional summary of the turn
        
    Returns:
        Completed turn info or None if no active turn
    """
    with _turn_lock:
        turn_info = _active_turns.pop(session_id, None)
        
        if turn_info:
            turn_info["ended_at"] = _now_iso()
            turn_info["summary"] = summary
            
            # Properly close the run context manager
            # This calls __exit__ which:
            # 1. Emits run.end event
            # 2. Marks run as closed
            # 3. Flushes the sink
            # 4. Unregisters from global registry
            run = turn_info.get("run")
            if run:
                try:
                    run.__exit__(None, None, None)
                except Exception:
                    # Fallback: at least try to flush
                    try:
                        run.sink.flush()
                    except Exception:
                        pass
            
            # Clear persisted state
            _clear_turn_state(session_id)
            
            return turn_info
        
        return None


def get_or_create_run(session_id: str) -> Any:
    """
    Get the current turn's run, or create a session-level placeholder.
    
    This maintains backward compatibility with code that expects a run
    to always exist. If no turn is active, returns a session-level run.
    """
    turn = get_current_turn(session_id)
    if turn:
        return turn["run"]
    
    # No active turn - create session-level run for SessionStart/SessionEnd
    ArzuleRun, HttpBatchSink, JsonlFileSink, _ = _get_imports()
    
    api_key = os.environ.get("ARZULE_API_KEY")
    tenant_id = os.environ.get("ARZULE_TENANT_ID")
    project_id = os.environ.get("ARZULE_PROJECT_ID")
    
    if api_key and tenant_id and project_id:
        endpoint = os.environ.get(
            "ARZULE_ENDPOINT",
            "https://dh43xnx5e03pq.cloudfront.net/ingest"
        )
        sink = HttpBatchSink(endpoint_url=endpoint, api_key=api_key)
    else:
        traces_dir = Path.home() / ".arzule" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        sink = JsonlFileSink(str(traces_dir / f"session_{session_id}.jsonl"))
    
    run = ArzuleRun(
        run_id=_session_to_run_id(session_id),
        tenant_id=tenant_id or "local",
        project_id=project_id or "claude_code",
        sink=sink,
    )
    run.__enter__()
    return run


def update_turn_state(session_id: str, updates: dict) -> None:
    """Update the current turn's state."""
    with _turn_lock:
        if session_id in _active_turns:
            _active_turns[session_id].update(updates)
            _persist_turn_state(session_id, _active_turns[session_id])


def push_span(session_id: str, **span_info) -> str:
    """Push a span onto the current turn's span stack and persist to disk."""
    from ..ids import new_span_id
    
    span_id = new_span_id()
    span_info["span_id"] = span_id
    
    with _turn_lock:
        if session_id in _active_turns:
            _active_turns[session_id].setdefault("spans", []).append(span_info)
            # Persist to disk so subsequent hooks see this span
            _persist_turn_state(session_id, _active_turns[session_id])
    
    return span_id


def pop_span(session_id: str, tool_use_id: str = None) -> Optional[dict]:
    """Pop a span from the current turn's span stack and persist to disk."""
    with _turn_lock:
        if session_id in _active_turns:
            spans = _active_turns[session_id].get("spans", [])
            result = None
            if tool_use_id:
                for i, span in enumerate(spans):
                    if span.get("tool_use_id") == tool_use_id:
                        result = spans.pop(i)
                        break
            elif spans:
                result = spans.pop()
            
            if result is not None:
                # Persist to disk so subsequent hooks see the updated stack
                _persist_turn_state(session_id, _active_turns[session_id])
            return result
    return None


def get_current_span(session_id: str) -> Optional[dict]:
    """Get the current (topmost) span."""
    with _turn_lock:
        if session_id in _active_turns:
            spans = _active_turns[session_id].get("spans", [])
            if spans:
                return spans[-1]
    return None


# =============================================================================
# Internal helpers
# =============================================================================

def _turn_to_run_id(turn_id: str) -> str:
    """Convert turn_id to UUID format for run_id."""
    hash_bytes = hashlib.sha256(turn_id.encode()).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))


def _session_to_run_id(session_id: str) -> str:
    """Convert session_id to UUID format for run_id."""
    hash_bytes = hashlib.sha256(f"session:{session_id}".encode()).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))


def _recreate_run(turn_info: dict) -> Any:
    """Recreate ArzuleRun from persisted turn info.
    
    IMPORTANT: Restores the seq counter from persisted state to avoid
    duplicate seq numbers across hook invocations (separate processes).
    """
    ArzuleRun, HttpBatchSink, JsonlFileSink, MultiSink, StreamingSink = _get_imports()
    
    api_key = os.environ.get("ARZULE_API_KEY")
    tenant_id = os.environ.get("ARZULE_TENANT_ID")
    project_id = os.environ.get("ARZULE_PROJECT_ID")
    stream_url = os.environ.get("ARZULE_STREAM_URL")
    
    sinks = []
    
    if api_key and tenant_id and project_id:
        endpoint = os.environ.get(
            "ARZULE_ENDPOINT",
            "https://dh43xnx5e03pq.cloudfront.net/ingest"
        )
        sinks.append(HttpBatchSink(endpoint_url=endpoint, api_key=api_key))
    
    if stream_url:
        sinks.append(StreamingSink(stream_url, turn_info["session_id"]))
    
    if not sinks:
        traces_dir = Path.home() / ".arzule" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        turn_id = turn_info.get("turn_id", "unknown")
        sinks.append(JsonlFileSink(str(traces_dir / f"turn_{turn_id.replace(':', '_')}.jsonl")))
    
    sink = sinks[0] if len(sinks) == 1 else MultiSink(sinks)
    
    run = ArzuleRun(
        run_id=turn_info.get("run_id", _turn_to_run_id(turn_info.get("turn_id", ""))),
        tenant_id=tenant_id or "local",
        project_id=project_id or "claude_code",
        sink=sink,
    )
    
    # Restore the seq counter from persisted state to avoid duplicate seq numbers
    # This is critical for cross-process hook invocations
    restored_seq = turn_info.get("current_seq", 0)
    if restored_seq > 0:
        run._seq = restored_seq
    
    run.__enter__()
    return run


def _persist_turn_state(session_id: str, turn_info: dict) -> None:
    """Persist turn state to disk (for cross-process access)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"turn_{session_id}.json"
    
    # Don't persist the run object itself, but DO persist the seq counter
    state = {k: v for k, v in turn_info.items() if k != "run"}
    
    # Extract current seq from the run object for persistence
    run = turn_info.get("run")
    if run and hasattr(run, "_seq"):
        state["current_seq"] = run._seq
    
    try:
        state_file.write_text(json.dumps(state, default=str))
    except Exception:
        pass


def _load_turn_state(session_id: str) -> Optional[dict]:
    """Load turn state from disk."""
    state_file = STATE_DIR / f"turn_{session_id}.json"
    
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            pass
    return None


def _clear_turn_state(session_id: str) -> None:
    """Clear persisted turn state."""
    state_file = STATE_DIR / f"turn_{session_id}.json"
    try:
        state_file.unlink(missing_ok=True)
    except Exception:
        pass


def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

