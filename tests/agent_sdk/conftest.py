"""Shared fixtures for Agent SDK tests."""

from __future__ import annotations

import time
from typing import Any, Optional
from unittest.mock import Mock, MagicMock

import pytest


@pytest.fixture
def mock_run():
    """Create a mock ArzuleRun for testing."""
    run = Mock()
    run.run_id = "test-run-id"
    run.tenant_id = "test-tenant"
    run.project_id = "test-project"
    run.trace_id = "a" * 32
    run._root_span_id = "b" * 16
    run._closed = False
    run._seq = 0

    def next_seq():
        run._seq += 1
        return run._seq

    run.next_seq = next_seq
    run.emit = Mock()
    run.current_parent_span_id = Mock(return_value="c" * 16)
    run.push_span = Mock()
    run.pop_span = Mock()

    def make_event(
        event_type: str,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        status: str = "ok",
        summary: str = "",
        agent: Optional[dict[str, Any]] = None,
        workstream_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attrs_compact: Optional[dict[str, Any]] = None,
        payload: Optional[dict[str, Any]] = None,
        raw_ref: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "trace_event.v0_1",
            "run_id": run.run_id,
            "tenant_id": run.tenant_id,
            "project_id": run.project_id,
            "trace_id": run.trace_id,
            "span_id": span_id or "d" * 16,
            "parent_span_id": parent_span_id,
            "seq": run.next_seq(),
            "ts": "2025-01-16T00:00:00+00:00",
            "agent": agent,
            "workstream_id": workstream_id,
            "task_id": task_id,
            "event_type": event_type,
            "status": status,
            "summary": summary,
            "attrs_compact": attrs_compact or {},
            "payload": payload or {},
            "raw_ref": raw_ref or {"storage": "inline"},
        }

    run._make_event = make_event
    return run


@pytest.fixture
def sample_pre_tool_input():
    """Sample input for PreToolUse hook event."""
    return {
        "hook_event_name": "PreToolUse",
        "session_id": "sess_test123",
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la"},
        "transcript_path": "/tmp/transcript",
        "cwd": "/home/user",
        "tool_use_id": "tool_use_12345",
    }


@pytest.fixture
def sample_post_tool_input():
    """Sample input for PostToolUse hook event."""
    return {
        "hook_event_name": "PostToolUse",
        "session_id": "sess_test123",
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la"},
        "tool_output": "file1.txt\nfile2.txt\n",
        "transcript_path": "/tmp/transcript",
        "cwd": "/home/user",
        "tool_use_id": "tool_use_12345",
        "duration_ms": 150,
    }


@pytest.fixture
def sample_prompt_input():
    """Sample input for UserPrompt hook event."""
    return {
        "hook_event_name": "UserPrompt",
        "session_id": "sess_test123",
        "prompt": "Please analyze the codebase and suggest improvements",
        "transcript_path": "/tmp/transcript",
        "cwd": "/home/user",
    }


@pytest.fixture
def sample_stop_input():
    """Sample input for Stop hook event."""
    return {
        "hook_event_name": "Stop",
        "session_id": "sess_test123",
        "transcript_path": "/tmp/transcript",
        "cwd": "/home/user",
        "stop_reason": "end_turn",
    }


@pytest.fixture
def sample_session_start_input():
    """Sample input for session start event."""
    return {
        "session_id": "sess_test123",
        "cwd": "/home/user/project",
        "model": "claude-sonnet-4-20250514",
        "permissions": ["read", "write", "bash"],
    }


@pytest.fixture
def sample_session_end_input():
    """Sample input for session end event."""
    return {
        "session_id": "sess_test123",
        "cwd": "/home/user/project",
        "duration_ms": 120000,
        "tool_count": 15,
        "turn_count": 3,
    }


@pytest.fixture
def sample_tool_event():
    """Sample tool event data for normalization tests."""
    return {
        "tool_name": "Read",
        "tool_input": {
            "file_path": "/home/user/project/src/main.py",
        },
        "tool_output": "def main():\n    print('Hello')\n",
        "tool_use_id": "tool_use_abc123",
        "duration_ms": 50,
    }


@pytest.fixture
def sample_tool_event_with_secrets():
    """Sample tool event containing sensitive data."""
    return {
        "tool_name": "Bash",
        "tool_input": {
            "command": "curl -H 'Authorization: Bearer sk-test12345' https://api.example.com",
        },
        "tool_output": "{'api_key': 'secret123', 'password': 'hunter2'}",
        "tool_use_id": "tool_use_secret",
        "duration_ms": 200,
    }


@pytest.fixture
def mock_time(monkeypatch):
    """Mock time.time for predictable duration calculations."""
    current_time = 1705363200.0  # 2024-01-16 00:00:00 UTC

    def mock_time_fn():
        return current_time

    monkeypatch.setattr(time, "time", mock_time_fn)
    return current_time


@pytest.fixture
def mock_datetime(monkeypatch):
    """Mock datetime for predictable timestamps."""
    from datetime import datetime, timezone

    fixed_dt = datetime(2025, 1, 16, 0, 0, 0, tzinfo=timezone.utc)

    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return fixed_dt

    monkeypatch.setattr("datetime.datetime", MockDatetime)
    return fixed_dt
