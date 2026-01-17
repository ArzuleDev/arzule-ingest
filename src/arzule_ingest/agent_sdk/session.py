"""Session state persistence for trace resumption support.

This module provides session state management for the Agent SDK integration,
allowing customers to resume traces using a session ID. Session state is stored
in a sidecar directory next to the trace file for easy correlation.

Storage layout:
    Trace file: ./traces.jsonl
    Session state: ./traces.jsonl.sessions/{session_id}.json

Example usage:
    >>> from pathlib import Path
    >>> from arzule_ingest.agent_sdk.session import (
    ...     save_session_state,
    ...     load_session_state,
    ...     update_session_state,
    ...     list_sessions,
    ... )
    >>>
    >>> trace_file = Path("./traces.jsonl")
    >>> state = {
    ...     "session_id": "sess_abc123",
    ...     "run_id": "run_xyz",
    ...     "trace_id": "trace_001",
    ...     "last_seq": 42,
    ...     "root_span_id": "span_root",
    ...     "status": "active",
    ... }
    >>> save_session_state("sess_abc123", state, trace_file)
    >>> loaded = load_session_state("sess_abc123", trace_file)
    >>> update_session_state("sess_abc123", {"last_seq": 43}, trace_file)
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class SessionStatus(str, Enum):
    """Status of a session.

    Attributes:
        ACTIVE: Session is currently in progress.
        COMPLETED: Session has finished successfully.
        ERROR: Session ended with an error.
    """

    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SessionState:
    """Session state data structure.

    This dataclass represents the persistent state of a trace session,
    enabling resumption of traces across process restarts.

    Attributes:
        session_id: Unique identifier for the session.
        run_id: Associated run identifier.
        trace_id: Associated trace identifier.
        last_seq: Last processed sequence number.
        root_span_id: ID of the root span in the trace.
        created_at: ISO 8601 timestamp of session creation.
        last_updated_at: ISO 8601 timestamp of last update.
        total_events: Total number of events in the session.
        total_tool_calls: Total number of tool calls in the session.
        status: Current session status (active, completed, error).
    """

    session_id: str
    run_id: str
    trace_id: str
    last_seq: int = 0
    root_span_id: str = ""
    created_at: str = field(default_factory=lambda: _now_iso())
    last_updated_at: str = field(default_factory=lambda: _now_iso())
    total_events: int = 0
    total_tool_calls: int = 0
    status: SessionStatus = SessionStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        """Convert session state to dictionary.

        Returns:
            Dictionary representation of the session state.
        """
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "last_seq": self.last_seq,
            "root_span_id": self.root_span_id,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "total_events": self.total_events,
            "total_tool_calls": self.total_tool_calls,
            "status": self.status.value if isinstance(self.status, SessionStatus) else self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        """Create session state from dictionary.

        Args:
            data: Dictionary containing session state fields.

        Returns:
            SessionState instance.
        """
        status_value = data.get("status", SessionStatus.ACTIVE.value)
        if isinstance(status_value, str):
            try:
                status = SessionStatus(status_value)
            except ValueError:
                status = SessionStatus.ACTIVE
        else:
            status = status_value

        return cls(
            session_id=data.get("session_id", ""),
            run_id=data.get("run_id", ""),
            trace_id=data.get("trace_id", ""),
            last_seq=data.get("last_seq", 0),
            root_span_id=data.get("root_span_id", ""),
            created_at=data.get("created_at", _now_iso()),
            last_updated_at=data.get("last_updated_at", _now_iso()),
            total_events=data.get("total_events", 0),
            total_tool_calls=data.get("total_tool_calls", 0),
            status=status,
        )


class SessionPersistenceError(Exception):
    """Exception raised when session persistence operations fail."""

    pass


def _now_iso() -> str:
    """Get current timestamp in ISO 8601 format.

    Returns:
        ISO 8601 formatted timestamp string with timezone.
    """
    return datetime.now(timezone.utc).isoformat()


def _get_sessions_dir(trace_file: Path) -> Path:
    """Get the sessions directory for a trace file.

    The sessions directory is a sidecar directory named {trace_file}.sessions
    located next to the trace file.

    Args:
        trace_file: Path to the trace file.

    Returns:
        Path to the sessions directory.
    """
    return trace_file.parent / f"{trace_file.name}.sessions"


def _get_session_file(session_id: str, trace_file: Path) -> Path:
    """Get the session state file path for a session.

    Args:
        session_id: Unique session identifier.
        trace_file: Path to the trace file.

    Returns:
        Path to the session state JSON file.
    """
    sessions_dir = _get_sessions_dir(trace_file)
    return sessions_dir / f"{session_id}.json"


def _get_lock_file(session_id: str, trace_file: Path) -> Path:
    """Get the lock file path for a session.

    Args:
        session_id: Unique session identifier.
        trace_file: Path to the trace file.

    Returns:
        Path to the lock file.
    """
    sessions_dir = _get_sessions_dir(trace_file)
    return sessions_dir / f".{session_id}.lock"


@contextmanager
def _file_lock(session_id: str, trace_file: Path) -> Iterator[None]:
    """Acquire an exclusive file lock for cross-process synchronization.

    This is critical for concurrent access safety - without this,
    parallel processes could corrupt session state by simultaneous writes.

    Note: Uses fcntl which is Unix-only. Windows support would require
    a different approach (e.g., msvcrt.locking).

    Args:
        session_id: Unique session identifier.
        trace_file: Path to the trace file.

    Yields:
        None when lock is acquired.

    Raises:
        SessionPersistenceError: If lock acquisition fails.
    """
    sessions_dir = _get_sessions_dir(trace_file)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    lock_file = _get_lock_file(session_id, trace_file)
    fd = None

    try:
        fd = open(lock_file, "w")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        yield
    except OSError as e:
        raise SessionPersistenceError(f"Failed to acquire lock for session {session_id}: {e}") from e
    finally:
        if fd:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
                fd.close()
            except OSError:
                pass


def _atomic_write(file_path: Path, content: str) -> None:
    """Write content to file atomically using temp file and rename.

    This ensures that readers never see partial writes. The write is
    performed to a temporary file in the same directory, then renamed
    to the target path (which is atomic on POSIX systems).

    Args:
        file_path: Target file path.
        content: Content to write.

    Raises:
        SessionPersistenceError: If write operation fails.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Write to temp file in same directory (for atomic rename)
        fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f".{file_path.stem}_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            os.replace(temp_path, file_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
    except OSError as e:
        raise SessionPersistenceError(f"Failed to write session state to {file_path}: {e}") from e


def save_session_state(session_id: str, state: dict[str, Any], trace_file: Path) -> None:
    """Save session state to persistent storage.

    Writes session state to a JSON file in the sessions directory with
    atomic write semantics and file locking for concurrent access safety.

    Args:
        session_id: Unique session identifier.
        state: Session state dictionary to persist.
        trace_file: Path to the trace file.

    Raises:
        SessionPersistenceError: If save operation fails.

    Example:
        >>> state = {
        ...     "session_id": "sess_123",
        ...     "run_id": "run_456",
        ...     "trace_id": "trace_789",
        ...     "last_seq": 10,
        ...     "root_span_id": "span_root",
        ...     "status": "active",
        ... }
        >>> save_session_state("sess_123", state, Path("./traces.jsonl"))
    """
    trace_file = Path(trace_file)

    # Ensure required fields
    state = dict(state)
    state["session_id"] = session_id
    state.setdefault("created_at", _now_iso())
    state["last_updated_at"] = _now_iso()

    with _file_lock(session_id, trace_file):
        session_file = _get_session_file(session_id, trace_file)
        content = json.dumps(state, indent=2, default=str)
        _atomic_write(session_file, content)


def load_session_state(session_id: str, trace_file: Path) -> dict[str, Any] | None:
    """Load session state from persistent storage.

    Reads session state from the JSON file in the sessions directory.
    Uses file locking for concurrent access safety.

    Args:
        session_id: Unique session identifier.
        trace_file: Path to the trace file.

    Returns:
        Session state dictionary if found, None otherwise.

    Raises:
        SessionPersistenceError: If read operation fails (other than file not found).

    Example:
        >>> state = load_session_state("sess_123", Path("./traces.jsonl"))
        >>> if state:
        ...     print(f"Last seq: {state['last_seq']}")
    """
    trace_file = Path(trace_file)
    session_file = _get_session_file(session_id, trace_file)

    if not session_file.exists():
        return None

    with _file_lock(session_id, trace_file):
        try:
            content = session_file.read_text(encoding="utf-8")
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise SessionPersistenceError(
                f"Failed to parse session state from {session_file}: {e}"
            ) from e
        except OSError as e:
            raise SessionPersistenceError(
                f"Failed to read session state from {session_file}: {e}"
            ) from e


def get_session_id_from_trace_file(trace_file: Path) -> str | None:
    """Extract session ID from the last event in a trace file.

    Reads the last line of the trace file and extracts the session_id
    field from the JSON event. This is useful for resuming a trace
    when the session ID is not known.

    Args:
        trace_file: Path to the trace file.

    Returns:
        Session ID if found, None otherwise.

    Example:
        >>> session_id = get_session_id_from_trace_file(Path("./traces.jsonl"))
        >>> if session_id:
        ...     state = load_session_state(session_id, trace_file)
    """
    trace_file = Path(trace_file)

    if not trace_file.exists():
        return None

    try:
        # Read last line efficiently
        with open(trace_file, "rb") as f:
            # Seek to end
            f.seek(0, 2)
            file_size = f.tell()

            if file_size == 0:
                return None

            # Read backwards to find last newline
            buffer_size = min(4096, file_size)
            f.seek(max(0, file_size - buffer_size))
            content = f.read().decode("utf-8")

            # Find last complete line
            lines = content.strip().split("\n")
            if not lines:
                return None

            last_line = lines[-1].strip()
            if not last_line:
                return None

            # Parse JSON and extract session_id
            event = json.loads(last_line)

            # Check common field locations for session_id
            session_id = event.get("session_id")
            if session_id:
                return session_id

            # Check in metadata
            metadata = event.get("metadata", {})
            session_id = metadata.get("session_id")
            if session_id:
                return session_id

            # Check in context
            context = event.get("context", {})
            session_id = context.get("session_id")
            if session_id:
                return session_id

            return None

    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def update_session_state(session_id: str, updates: dict[str, Any], trace_file: Path) -> None:
    """Update session state with partial updates.

    Loads existing session state, merges updates, and persists the result.
    Uses file locking for concurrent access safety.

    Args:
        session_id: Unique session identifier.
        updates: Dictionary of fields to update.
        trace_file: Path to the trace file.

    Raises:
        SessionPersistenceError: If update operation fails.

    Example:
        >>> update_session_state(
        ...     "sess_123",
        ...     {"last_seq": 11, "total_events": 100},
        ...     Path("./traces.jsonl"),
        ... )
    """
    trace_file = Path(trace_file)

    with _file_lock(session_id, trace_file):
        session_file = _get_session_file(session_id, trace_file)

        # Load existing state
        existing_state: dict[str, Any] = {}
        if session_file.exists():
            try:
                content = session_file.read_text(encoding="utf-8")
                existing_state = json.loads(content)
            except (json.JSONDecodeError, OSError):
                pass

        # Merge updates
        existing_state.update(updates)
        existing_state["session_id"] = session_id
        existing_state["last_updated_at"] = _now_iso()

        # Persist
        content = json.dumps(existing_state, indent=2, default=str)
        _atomic_write(session_file, content)


def list_sessions(trace_file: Path) -> list[dict[str, Any]]:
    """List all sessions for a trace file.

    Scans the sessions directory and loads all session state files.
    Results are sorted by last_updated_at in descending order (most recent first).

    Args:
        trace_file: Path to the trace file.

    Returns:
        List of session state dictionaries.

    Example:
        >>> sessions = list_sessions(Path("./traces.jsonl"))
        >>> for session in sessions:
        ...     print(f"{session['session_id']}: {session['status']}")
    """
    trace_file = Path(trace_file)
    sessions_dir = _get_sessions_dir(trace_file)

    if not sessions_dir.exists():
        return []

    sessions: list[dict[str, Any]] = []

    try:
        for session_file in sessions_dir.glob("*.json"):
            # Skip hidden files
            if session_file.name.startswith("."):
                continue

            try:
                content = session_file.read_text(encoding="utf-8")
                state = json.loads(content)
                sessions.append(state)
            except (json.JSONDecodeError, OSError):
                # Skip corrupted files
                continue

    except OSError:
        return []

    # Sort by last_updated_at descending
    sessions.sort(
        key=lambda s: s.get("last_updated_at", ""),
        reverse=True,
    )

    return sessions


def delete_session(session_id: str, trace_file: Path) -> bool:
    """Delete a session state file.

    Removes the session state file and its lock file from the sessions directory.

    Args:
        session_id: Unique session identifier.
        trace_file: Path to the trace file.

    Returns:
        True if session was deleted, False if it did not exist.

    Raises:
        SessionPersistenceError: If deletion fails (other than file not found).

    Example:
        >>> deleted = delete_session("sess_123", Path("./traces.jsonl"))
        >>> print("Deleted" if deleted else "Not found")
    """
    trace_file = Path(trace_file)
    session_file = _get_session_file(session_id, trace_file)
    lock_file = _get_lock_file(session_id, trace_file)

    if not session_file.exists():
        return False

    with _file_lock(session_id, trace_file):
        try:
            session_file.unlink()
        except OSError as e:
            raise SessionPersistenceError(
                f"Failed to delete session state {session_file}: {e}"
            ) from e

    # Clean up lock file (best effort)
    try:
        lock_file.unlink()
    except OSError:
        pass

    return True


def cleanup_stale_sessions(
    trace_file: Path,
    max_age_hours: int = 24,
    statuses: list[SessionStatus] | None = None,
) -> int:
    """Clean up stale session state files.

    Removes session files that are older than the specified age and
    optionally match specific statuses.

    Args:
        trace_file: Path to the trace file.
        max_age_hours: Maximum age in hours for sessions to keep.
        statuses: Optional list of statuses to clean up. If None, only
            ERROR and COMPLETED sessions are cleaned.

    Returns:
        Number of sessions cleaned up.

    Example:
        >>> cleaned = cleanup_stale_sessions(
        ...     Path("./traces.jsonl"),
        ...     max_age_hours=48,
        ...     statuses=[SessionStatus.ERROR],
        ... )
        >>> print(f"Cleaned up {cleaned} sessions")
    """
    if statuses is None:
        statuses = [SessionStatus.ERROR, SessionStatus.COMPLETED]

    status_values = {s.value for s in statuses}
    trace_file = Path(trace_file)
    cutoff = datetime.now(timezone.utc)

    cleaned = 0
    sessions = list_sessions(trace_file)

    for session in sessions:
        session_status = session.get("status", "")
        if session_status not in status_values:
            continue

        last_updated = session.get("last_updated_at", "")
        if not last_updated:
            continue

        try:
            updated_dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            age_hours = (cutoff - updated_dt).total_seconds() / 3600

            if age_hours > max_age_hours:
                session_id = session.get("session_id", "")
                if session_id and delete_session(session_id, trace_file):
                    cleaned += 1
        except (ValueError, TypeError):
            continue

    return cleaned
