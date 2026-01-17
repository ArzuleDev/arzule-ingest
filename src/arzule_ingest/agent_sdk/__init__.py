"""Claude Agent SDK integration for Arzule tracing.

This module provides a wrapper around the claude_agent_sdk that automatically
captures telemetry events and sends them to the Arzule backend.

Usage:
    ```python
    from arzule_ingest.agent_sdk import TracedClaudeClient, traced_claude_session

    # Using context manager
    async with TracedClaudeClient(
        trace_file="./traces/session.jsonl"
    ) as client:
        await client.query("Hello, Claude!")
        async for msg in client.receive_response():
            print(msg)

    # Or using the convenience function
    async with traced_claude_session(
        endpoint="https://ingest.arzule.com/v1/traces",
        api_key="your-api-key",
    ) as client:
        await client.query("Hello!")
    ```

Session Resumption:
    ```python
    # First session
    async with TracedClaudeClient() as client:
        await client.query("Start a task")
        session_id = client.session_id  # Save this

    # Later, resume the session
    async with TracedClaudeClient(resume=session_id) as client:
        await client.query("Continue the task")
    ```
"""

from __future__ import annotations

from .client import (
    TracedClaudeClient,
    traced_claude_session,
    # Exceptions
    TracingError,
    SessionNotFoundError,
    ClientNotConnectedError,
    # Session state helpers from client
    delete_session_state,
)
from .hooks import create_tracing_hooks
from .normalize import (
    normalize_event,
    normalize_llm_end,
    normalize_llm_start,
    normalize_message,
    normalize_prompt,
    normalize_result,
    normalize_session_end,
    normalize_session_start,
    normalize_tool_end,
    normalize_tool_start,
)
from .session import (
    SessionState,
    SessionStatus,
    get_session_id_from_trace_file,
    list_sessions,
    load_session_state,
    save_session_state,
    update_session_state,
)

__all__ = [
    # Main client
    "TracedClaudeClient",
    "traced_claude_session",
    # Exceptions
    "TracingError",
    "SessionNotFoundError",
    "ClientNotConnectedError",
    # Hooks
    "create_tracing_hooks",
    # Tool events
    "normalize_tool_start",
    "normalize_tool_end",
    # Prompt/Message events
    "normalize_prompt",
    "normalize_message",
    "normalize_result",
    # Session lifecycle
    "normalize_session_start",
    "normalize_session_end",
    # LLM calls
    "normalize_llm_start",
    "normalize_llm_end",
    # Generic
    "normalize_event",
    # Session state management
    "SessionState",
    "SessionStatus",
    "save_session_state",
    "load_session_state",
    "delete_session_state",
    "get_session_id_from_trace_file",
    "update_session_state",
    "list_sessions",
]
