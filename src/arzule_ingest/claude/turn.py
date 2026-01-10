"""Turn-based run management for Claude Code.

A "turn" is a UserPromptSubmit -> Stop cycle. Each turn becomes a separate
Arzule run, allowing proper boundaries for analysis.

This replaces the session-based approach where entire sessions (which can
span hours/days) were treated as a single run.

IMPORTANT: Claude Code runs hooks as SEPARATE PROCESSES. When multiple tool
calls happen in parallel (common with WebSearch etc.), we need FILE-LEVEL
locking to prevent race conditions where parallel processes overwrite each
other's state. The threading Lock only protects in-memory access within a
single process.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

# State directory for persistence across hook invocations
STATE_DIR = Path.home() / ".arzule" / "claude_state"

_turn_lock = Lock()  # In-process lock (protects in-memory dict)
_active_turns: dict[str, Any] = {}  # session_id -> current turn info


@contextmanager
def _file_lock(session_id: str):
    """
    Acquire an exclusive file lock for cross-process synchronization.
    
    This is CRITICAL for parallel hook invocations - without this,
    concurrent processes will overwrite each other's state updates.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = STATE_DIR / f"turn_{session_id}.lock"
    
    fd = None
    try:
        fd = open(lock_file, "w")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fd:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
                fd.close()
            except Exception:
                pass


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
        
        # IMPORTANT: Initialize current_seq from run._seq AFTER __enter__()
        # because __enter__() may emit run.start which uses run.next_seq().
        # 
        # next_seq() semantics: increments _seq THEN returns it.
        # So after run.start (seq 1), _seq = 1. The _seq value IS the last returned.
        # 
        # emit_with_seq_sync uses disk_seq as "last used" and emits disk_seq + 1.
        # So current_seq should be the last used seq (i.e., run._seq directly).
        initial_seq = run._seq if hasattr(run, '_seq') else 0
        
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
            "current_seq": initial_seq,  # Start from where run left off
            "root_span_id": run._root_span_id,  # Persist for cross-process recreation
        }
        
        _active_turns[session_id] = turn_info
        _persist_turn_state(session_id, turn_info)
        
        return turn_info


def get_current_turn(session_id: str) -> Optional[dict]:
    """
    Get the current active turn for a session.
    
    Returns None if no turn is active (session started but no prompt yet).
    Uses file locking to ensure consistent state across parallel hook invocations.
    
    Lock ordering: file_lock (outer) -> turn_lock (inner) to prevent deadlocks.
    """
    # File lock is always outer to ensure consistent lock ordering
    with _file_lock(session_id):
        with _turn_lock:
            if session_id in _active_turns:
                # Even if in memory, refresh from disk to get parallel updates
                disk_state = _load_turn_state_unlocked(session_id)
                if disk_state and "turn_id" in disk_state:
                    # Merge disk state into memory
                    existing = _active_turns[session_id]
                    existing["spans"] = disk_state.get("spans", [])
                    existing["tool_inputs"] = disk_state.get("tool_inputs", {})
                    existing["tool_calls"] = disk_state.get("tool_calls", [])
                    existing["current_seq"] = disk_state.get("current_seq", 0)
                    # Also copy any other fields that might have been updated
                    for key in ["last_subagent_result", "last_subagent_transcript", 
                               "last_subagent_id", "active_subagents", "compaction_count"]:
                        if key in disk_state:
                            existing[key] = disk_state[key]
                return _active_turns[session_id]
            
            # Try to load from disk (hooks are separate processes)
            turn_info = _load_turn_state_unlocked(session_id)
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
    ArzuleRun, HttpBatchSink, JsonlFileSink, _, _ = _get_imports()
    
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
    """Update the current turn's state.
    
    Uses file locking to handle parallel hook invocations safely.
    This is critical for parallel tool calls (e.g., multiple WebSearch).
    
    For list fields (tool_calls, spans), we MERGE by adding new items.
    For dict fields (tool_inputs, active_subagents), we MERGE by combining keys.
    For scalar fields, we use the update value.
    """
    # Use file lock to prevent parallel processes from overwriting each other
    with _file_lock(session_id):
        with _turn_lock:
            if session_id not in _active_turns:
                return
            
            existing = _active_turns[session_id]
            
            # Reload from disk to get any changes from parallel processes
            disk_state = _load_turn_state_unlocked(session_id)
            if disk_state and "turn_id" in disk_state:
                # Start with disk state as the base for mergeable fields
                existing["spans"] = disk_state.get("spans", [])
                existing["current_seq"] = disk_state.get("current_seq", 0)
                
                # For dicts, merge disk + memory
                for dict_key in ["tool_inputs", "active_subagents"]:
                    disk_dict = disk_state.get(dict_key, {})
                    mem_dict = existing.get(dict_key, {})
                    existing[dict_key] = {**disk_dict, **mem_dict}
                
                # For tool_calls list, we need to merge by tool_use_id
                disk_tool_calls = disk_state.get("tool_calls", [])
                existing["tool_calls"] = disk_tool_calls
            
            # Now apply updates with proper merging
            for key, value in updates.items():
                if key == "tool_calls" and isinstance(value, list):
                    # Merge tool_calls by tool_use_id (avoid duplicates)
                    existing_calls = existing.get("tool_calls", [])
                    existing_ids = {tc.get("tool_use_id") for tc in existing_calls}
                    for tc in value:
                        if tc.get("tool_use_id") not in existing_ids:
                            existing_calls.append(tc)
                    existing["tool_calls"] = existing_calls
                    
                elif key == "spans" and isinstance(value, list):
                    # Merge spans by span_id (avoid duplicates)
                    existing_spans = existing.get("spans", [])
                    existing_span_ids = {s.get("span_id") for s in existing_spans}
                    for span in value:
                        if span.get("span_id") not in existing_span_ids:
                            existing_spans.append(span)
                    existing["spans"] = existing_spans
                    
                elif key in ["tool_inputs", "active_subagents"] and isinstance(value, dict):
                    # Merge dicts
                    existing_dict = existing.get(key, {})
                    existing_dict.update(value)
                    existing[key] = existing_dict
                    
                else:
                    # Scalar value - just update
                    existing[key] = value
            
            _persist_turn_state_unlocked(session_id, existing)


def push_span(session_id: str, **span_info) -> str:
    """Push a span onto the current turn's span stack and persist to disk.
    
    Uses file locking to handle parallel hook invocations safely.
    """
    from ..ids import new_span_id
    
    span_id = new_span_id()
    span_info["span_id"] = span_id
    
    # Use file lock to prevent parallel processes from overwriting each other
    with _file_lock(session_id):
        with _turn_lock:
            # Reload from disk to get any changes from parallel processes
            disk_state = _load_turn_state_unlocked(session_id)
            if disk_state and "turn_id" in disk_state:
                # Merge disk state into memory
                if session_id in _active_turns:
                    _active_turns[session_id]["spans"] = disk_state.get("spans", [])
                    _active_turns[session_id]["current_seq"] = disk_state.get("current_seq", 0)
            
            if session_id in _active_turns:
                _active_turns[session_id].setdefault("spans", []).append(span_info)
                # Persist to disk so subsequent hooks see this span
                _persist_turn_state_unlocked(session_id, _active_turns[session_id])
    
    return span_id


def pop_span(session_id: str, tool_use_id: str = None) -> Optional[dict]:
    """Pop a span from the current turn's span stack and persist to disk.
    
    Uses file locking to handle parallel hook invocations safely.
    """
    # Use file lock to prevent parallel processes from overwriting each other
    with _file_lock(session_id):
        with _turn_lock:
            # Reload from disk to get any changes from parallel processes
            disk_state = _load_turn_state_unlocked(session_id)
            if disk_state and "turn_id" in disk_state:
                if session_id in _active_turns:
                    _active_turns[session_id]["spans"] = disk_state.get("spans", [])
                    _active_turns[session_id]["current_seq"] = disk_state.get("current_seq", 0)
            
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
                    _persist_turn_state_unlocked(session_id, _active_turns[session_id])
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
    
    IMPORTANT: This recreates the run object for an EXISTING turn.
    We must NOT emit run.start again - that was already done when
    the turn started. We only need to:
    1. Create the run object with the same run_id
    2. Initialize internal state (managers, spans)
    3. Restore the seq counter from disk
    
    This is critical for cross-process hook invocations where each
    hook process needs its own run object but shares the same run_id.
    """
    ArzuleRun, HttpBatchSink, JsonlFileSink, MultiSink, StreamingSink = _get_imports()
    from ..run_managers.span_manager import SpanManager
    from ..run_managers.task_manager import TaskManager
    from ..run import _active_run, register_run
    from ..ids import new_span_id
    
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
    
    # Initialize run state WITHOUT emitting run.start (that was already done)
    # This is a partial __enter__ - we need the managers but not the event
    _active_run.set(run)
    register_run(run)
    run._root_span_id = turn_info.get("root_span_id") or new_span_id()
    run._span_manager = SpanManager(root_span_id=run._root_span_id)
    run._task_manager = TaskManager(root_span_id=run._root_span_id)
    
    return run


def _persist_turn_state_unlocked(session_id: str, turn_info: dict) -> None:
    """Persist turn state to disk (WITHOUT acquiring file lock).
    
    Caller must hold the file lock via _file_lock(session_id).
    
    Uses atomic write (write to temp file, then rename) to prevent
    partial reads by parallel processes.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"turn_{session_id}.json"
    temp_file = STATE_DIR / f"turn_{session_id}.json.tmp"
    
    # Don't persist the run object itself, but DO persist the seq counter
    state = {k: v for k, v in turn_info.items() if k != "run"}
    
    # IMPORTANT: Do NOT override current_seq from run._seq!
    # emit_with_seq_sync carefully manages current_seq to track the LAST USED seq.
    # run._seq is the NEXT seq to use, which is 1 ahead. Using it here was causing
    # sequence numbers to be skipped (every other seq missing).
    # The current_seq in turn_info is already correct from emit_with_seq_sync.
    
    try:
        # Atomic write: write to temp file first, then rename
        # This prevents other processes from reading partial/corrupted JSON
        temp_file.write_text(json.dumps(state, default=str))
        temp_file.rename(state_file)
    except Exception:
        # Clean up temp file if rename failed
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            pass


def _load_turn_state_unlocked(session_id: str) -> Optional[dict]:
    """Load turn state from disk (WITHOUT acquiring file lock).
    
    Caller must hold the file lock via _file_lock(session_id).
    
    If JSON parsing fails, retries once after a brief delay in case
    another process was mid-write (though atomic writes should prevent this).
    """
    state_file = STATE_DIR / f"turn_{session_id}.json"
    
    if not state_file.exists():
        return None
    
    for attempt in range(2):  # Retry once if parsing fails
        try:
            content = state_file.read_text()
            if not content.strip():
                # Empty file - treat as no state
                return None
            return json.loads(content)
        except json.JSONDecodeError as e:
            if attempt == 0:
                # Brief delay before retry (file might be mid-write)
                time.sleep(0.01)
                continue
            # Log the error for debugging
            _log_debug(f"Failed to parse turn state for {session_id}: {e}")
            return None
        except Exception as e:
            _log_debug(f"Error reading turn state for {session_id}: {e}")
            return None
    
    return None


def _persist_turn_state(session_id: str, turn_info: dict) -> None:
    """Persist turn state to disk (with file locking for cross-process safety)."""
    with _file_lock(session_id):
        _persist_turn_state_unlocked(session_id, turn_info)


def _load_turn_state(session_id: str) -> Optional[dict]:
    """Load turn state from disk (with file locking for cross-process safety)."""
    with _file_lock(session_id):
        return _load_turn_state_unlocked(session_id)


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


def emit_with_seq_sync(session_id: str, run: Any, event: dict) -> None:
    """
    Emit an event with atomic seq number synchronization across processes.
    
    This is CRITICAL for parallel hook invocations. Without this, multiple
    processes could emit events with the same seq number, causing duplicates
    that get deduplicated by the backend.
    
    Flow:
    1. Acquire file lock
    2. Load current seq from disk (source of truth for cross-process coordination)
    3. Compute next seq = disk_seq + 1 (ALWAYS increment from disk)
    4. Set event["seq"] manually and update run._seq
    5. Persist new seq to disk BEFORE emit
    6. Emit event
    7. Release file lock
    """
    event_type = event.get("event_type", "unknown")
    tool_use_id = event.get("attributes", {}).get("tool_use_id", "none")
    
    _log_debug(f"emit_with_seq_sync[{event_type}]: acquiring lock for {tool_use_id}")
    
    with _file_lock(session_id):
        # Load current seq from disk - this is the LAST seq used by any process
        disk_state = _load_turn_state_unlocked(session_id)
        
        if disk_state:
            disk_seq = disk_state.get("current_seq", 0)
        else:
            # Disk state missing or corrupted - fall back to run's in-memory seq
            # This is a safeguard against data loss; run._seq tracks last used seq
            disk_seq = getattr(run, '_seq', 0)
            if disk_seq > 0:
                _log_debug(f"emit_with_seq_sync: disk state missing, using run._seq={disk_seq}")
        
        # ALWAYS increment from disk - this ensures unique seq across processes
        # Even if run._seq is higher (shouldn't happen), we trust disk as source of truth
        this_seq = disk_seq + 1
        
        span_id = event.get("span_id", "no-span")
        _log_debug(f"emit_with_seq_sync[{event_type}]: disk_seq={disk_seq}, this_seq={this_seq}, span_id={span_id[:12]}...")
        
        # Set the event's seq manually - don't rely on run.emit() to do it
        event["seq"] = this_seq
        
        # Update run._seq to stay in sync for any non-synced emits
        # IMPORTANT: next_seq() does `_seq += 1; return _seq`, so _seq should be
        # the LAST USED seq, not the next to use. Set to this_seq so next_seq() 
        # returns this_seq + 1.
        run._seq = this_seq
        
        # Persist this_seq to disk BEFORE emit - next process will read this and use this_seq + 1
        if session_id in _active_turns:
            _active_turns[session_id]["current_seq"] = this_seq
            _persist_turn_state_unlocked(session_id, _active_turns[session_id])
        elif disk_state:
            disk_state["current_seq"] = this_seq
            _persist_turn_state_unlocked(session_id, disk_state)
        else:
            # No state yet - create minimal state with just seq
            _persist_turn_state_unlocked(session_id, {"current_seq": this_seq})
        
        _log_debug(f"emit_with_seq_sync[{event_type}]: persisted seq={this_seq}, emitting")
        
        # Now emit the event - seq is already set on the event
        run.emit(event)
        
        _log_debug(f"emit_with_seq_sync[{event_type}]: emitted, releasing lock")


def _log_debug(message: str) -> None:
    """Log debug info to file."""
    try:
        from pathlib import Path
        log_file = Path.home() / ".arzule" / "hook_debug.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"[{_now_iso()}] {message}\n")
    except Exception:
        pass

