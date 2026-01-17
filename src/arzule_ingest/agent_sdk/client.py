"""TracedClaudeClient - Wrapper for ClaudeSDKClient with automatic tracing.

This module provides a wrapper around the claude_agent_sdk.ClaudeSDKClient that
automatically captures telemetry events and sends them to the Arzule backend.

Requires the claude-agent-sdk package to be installed:
    pip install claude-agent-sdk
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# Lazy import flag for claude_agent_sdk (optional dependency)
_CLAUDE_SDK_AVAILABLE = False
_CLAUDE_SDK_IMPORT_ERROR: Optional[str] = None

try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, HookMatcher
    from claude_agent_sdk.types import (
        Message,
        ResultMessage,
        AssistantMessage,
        UserMessage,
        SystemMessage,
        StreamEvent,
        HookEvent,
        HookInput,
        HookJSONOutput,
        HookContext,
        PreToolUseHookInput,
        PostToolUseHookInput,
        UserPromptSubmitHookInput,
        StopHookInput,
    )
    _CLAUDE_SDK_AVAILABLE = True
except ImportError as e:
    _CLAUDE_SDK_IMPORT_ERROR = str(e)
    # Define placeholder types for static analysis when SDK not installed
    ClaudeSDKClient = None  # type: ignore[misc, assignment]
    ClaudeAgentOptions = None  # type: ignore[misc, assignment]
    HookMatcher = None  # type: ignore[misc, assignment]
    Message = Any  # type: ignore[misc, assignment]
    ResultMessage = Any  # type: ignore[misc, assignment]
    AssistantMessage = Any  # type: ignore[misc, assignment]
    UserMessage = Any  # type: ignore[misc, assignment]
    SystemMessage = Any  # type: ignore[misc, assignment]
    StreamEvent = Any  # type: ignore[misc, assignment]
    HookEvent = Any  # type: ignore[misc, assignment]
    HookInput = Any  # type: ignore[misc, assignment]
    HookJSONOutput = Any  # type: ignore[misc, assignment]
    HookContext = Any  # type: ignore[misc, assignment]
    PreToolUseHookInput = Any  # type: ignore[misc, assignment]
    PostToolUseHookInput = Any  # type: ignore[misc, assignment]
    UserPromptSubmitHookInput = Any  # type: ignore[misc, assignment]
    StopHookInput = Any  # type: ignore[misc, assignment]

from ..ids import new_run_id, new_span_id, new_trace_id
from ..sinks.base import TelemetrySink
from ..sinks.file_jsonl import JsonlFileSink
from ..sinks.http_batch import HttpBatchSink
from ..endpoints import get_claude_ingest_url

if TYPE_CHECKING:
    from ..run import ArzuleRun


def _check_sdk_available() -> None:
    """Check if claude_agent_sdk is available, raise helpful error if not."""
    if not _CLAUDE_SDK_AVAILABLE:
        raise ImportError(
            f"claude-agent-sdk is required for TracedClaudeClient. "
            f"Install with: pip install claude-agent-sdk\n"
            f"Original error: {_CLAUDE_SDK_IMPORT_ERROR}"
        )


# =============================================================================
# Exceptions
# =============================================================================


class TracingError(Exception):
    """Base exception for tracing errors."""

    pass


class SessionNotFoundError(TracingError):
    """Raised when attempting to resume a non-existent session."""

    pass


class ClientNotConnectedError(TracingError):
    """Raised when performing operations on a disconnected client."""

    pass


# =============================================================================
# Session State Management
# =============================================================================


@dataclass
class SessionState:
    """Persistent state for session resumption.

    This state is saved to disk to allow resuming sessions across process
    restarts or reconnections.
    """

    session_id: str
    run_id: str
    trace_id: str
    created_at: str
    last_active_at: str
    prompt_count: int = 0
    message_count: int = 0
    total_cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _get_session_state_path(session_id: str) -> Path:
    """Get the path for session state file."""
    state_dir = Path.home() / ".arzule" / "agent_sdk_sessions"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{session_id}.json"


def save_session_state(state: SessionState) -> None:
    """Save session state to disk.

    Args:
        state: The session state to persist
    """
    import json

    path = _get_session_state_path(state.session_id)
    data = {
        "session_id": state.session_id,
        "run_id": state.run_id,
        "trace_id": state.trace_id,
        "created_at": state.created_at,
        "last_active_at": state.last_active_at,
        "prompt_count": state.prompt_count,
        "message_count": state.message_count,
        "total_cost_usd": state.total_cost_usd,
        "metadata": state.metadata,
    }
    path.write_text(json.dumps(data, indent=2))


def load_session_state(session_id: str) -> Optional[SessionState]:
    """Load session state from disk.

    Args:
        session_id: The session ID to load

    Returns:
        SessionState if found, None otherwise
    """
    import json

    path = _get_session_state_path(session_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return SessionState(
            session_id=data["session_id"],
            run_id=data["run_id"],
            trace_id=data["trace_id"],
            created_at=data["created_at"],
            last_active_at=data["last_active_at"],
            prompt_count=data.get("prompt_count", 0),
            message_count=data.get("message_count", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            metadata=data.get("metadata", {}),
        )
    except (json.JSONDecodeError, KeyError):
        return None


def delete_session_state(session_id: str) -> bool:
    """Delete session state from disk.

    Args:
        session_id: The session ID to delete

    Returns:
        True if deleted, False if not found
    """
    path = _get_session_state_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


# =============================================================================
# Hook Factories
# =============================================================================


def _create_tracing_hooks(
    client: "TracedClaudeClient",
) -> dict[HookEvent, list[HookMatcher]]:
    """Create tracing hooks that capture telemetry events.

    Args:
        client: The TracedClaudeClient instance to emit events to

    Returns:
        Dictionary mapping hook events to matchers with callbacks
    """

    async def on_pre_tool_use(
        hook_input: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Capture tool invocation start events."""
        # Check for PreToolUseHookInput by key presence (TypedDict doesn't support isinstance)
        if not isinstance(hook_input, dict) or "tool_name" not in hook_input or "tool_input" not in hook_input:
            return {}

        span_id = new_span_id()
        client._inflight_spans[tool_use_id or span_id] = span_id

        client._emit_event(
            event_type="tool.start",
            span_id=span_id,
            parent_span_id=client._current_turn_span,
            summary=f"Tool: {hook_input['tool_name']}",
            attrs_compact={
                "tool_name": hook_input["tool_name"],
                "tool_input": _truncate_value(hook_input.get("tool_input", {})),
            },
        )

        return {}

    async def on_post_tool_use(
        hook_input: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Capture tool completion events."""
        # Check for PostToolUseHookInput by key presence (TypedDict doesn't support isinstance)
        if not isinstance(hook_input, dict) or "tool_name" not in hook_input or "tool_response" not in hook_input:
            return {}

        # Look up the span we started
        span_id = client._inflight_spans.pop(tool_use_id, None) or new_span_id()

        # Determine status from response
        response = hook_input.get("tool_response")
        is_error = False
        if isinstance(response, dict):
            is_error = response.get("is_error", False)

        client._emit_event(
            event_type="tool.end",
            span_id=span_id,
            parent_span_id=client._current_turn_span,
            status="error" if is_error else "ok",
            summary=f"Tool completed: {hook_input['tool_name']}",
            attrs_compact={
                "tool_name": hook_input["tool_name"],
                "tool_response": _truncate_value(response),
            },
        )

        return {}

    async def on_user_prompt_submit(
        hook_input: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Capture user prompt submission events."""
        # Check for UserPromptSubmitHookInput by key presence (TypedDict doesn't support isinstance)
        if not isinstance(hook_input, dict) or "prompt" not in hook_input:
            return {}

        # Start a new turn span
        client._current_turn_span = new_span_id()
        client._state.prompt_count += 1

        client._emit_event(
            event_type="turn.start",
            span_id=client._current_turn_span,
            parent_span_id=client._root_span_id,
            summary="User prompt submitted",
            attrs_compact={
                "prompt_preview": _truncate_value(hook_input.get("prompt", ""), max_chars=500),
                "prompt_index": client._state.prompt_count,
            },
        )

        return {}

    async def on_stop(
        hook_input: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Capture session stop events."""
        # StopHookInput is a dict - just check it's a dict (TypedDict doesn't support isinstance)
        if not isinstance(hook_input, dict):
            return {}

        # End the current turn if one is active
        if client._current_turn_span:
            client._emit_event(
                event_type="turn.end",
                span_id=new_span_id(),
                parent_span_id=client._current_turn_span,
                summary="Turn completed",
            )
            client._current_turn_span = None

        return {}

    return {
        "PreToolUse": [HookMatcher(matcher=None, hooks=[on_pre_tool_use])],
        "PostToolUse": [HookMatcher(matcher=None, hooks=[on_post_tool_use])],
        "UserPromptSubmit": [HookMatcher(matcher=None, hooks=[on_user_prompt_submit])],
        "Stop": [HookMatcher(matcher=None, hooks=[on_stop])],
    }


def _truncate_value(value: Any, max_chars: int = 2000) -> Any:
    """Truncate large values to prevent excessive payload sizes.

    Args:
        value: The value to truncate
        max_chars: Maximum characters for string values

    Returns:
        Truncated value
    """
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"
        return value
    elif isinstance(value, dict):
        return {k: _truncate_value(v, max_chars) for k, v in value.items()}
    elif isinstance(value, list):
        return [_truncate_value(v, max_chars) for v in value[:100]]  # Limit list items
    return value


# =============================================================================
# Main TracedClaudeClient Class
# =============================================================================


class TracedClaudeClient:
    """Wrapper around ClaudeSDKClient with automatic tracing.

    This client wraps the Claude Agent SDK client and automatically:
    - Registers tracing hooks to capture tool use and prompt events
    - Intercepts the message stream to capture response events
    - Manages trace lifecycle (start/end events)
    - Supports session resumption via session_id

    Usage:
        ```python
        async with TracedClaudeClient(
            trace_file="./traces/session.jsonl",
            tenant_id="my-tenant",
            project_id="my-project",
        ) as client:
            await client.query("Hello, Claude!")
            async for msg in client.receive_response():
                print(msg)
        ```

    For session resumption:
        ```python
        # Resume a previous session
        async with TracedClaudeClient(
            resume="previous-session-id",
            endpoint="https://ingest.arzule.com/v1/traces",
            api_key="...",
        ) as client:
            await client.query("Continue our conversation...")
        ```
    """

    def __init__(
        self,
        trace_file: str | Path | None = None,
        endpoint: str | None = None,
        sink: TelemetrySink | None = None,
        api_key: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
        resume: str | None = None,
        **sdk_options: Any,
    ) -> None:
        """Initialize the traced Claude client.

        Args:
            trace_file: Path to write trace events as JSONL (local mode)
            endpoint: HTTP endpoint for trace ingestion (cloud mode)
            sink: Custom telemetry sink (overrides trace_file and endpoint)
            api_key: API key for authentication with the endpoint
            tenant_id: Tenant identifier for multi-tenant tracing
            project_id: Project identifier for organizing traces
            resume: Session ID to resume (loads previous state)
            **sdk_options: Additional options passed to ClaudeSDKClient

        Raises:
            ImportError: If claude-agent-sdk is not installed
        """
        # Check SDK availability early to provide helpful error message
        _check_sdk_available()

        self._trace_file = trace_file
        self._endpoint = endpoint
        self._api_key = api_key
        self._tenant_id = tenant_id or os.environ.get("ARZULE_TENANT_ID", "local")
        self._project_id = project_id or os.environ.get("ARZULE_PROJECT_ID", "agent_sdk")
        self._resume_session_id = resume
        self._sdk_options = sdk_options

        # State management
        self._state: Optional[SessionState] = None
        self._sink: Optional[TelemetrySink] = sink
        self._owns_sink = sink is None  # Track if we should close the sink

        # Span tracking
        self._root_span_id: Optional[str] = None
        self._current_turn_span: Optional[str] = None
        self._inflight_spans: dict[str, str] = {}  # tool_use_id -> span_id
        self._seq: int = 0

        # Underlying SDK client
        self._client: Optional[ClaudeSDKClient] = None
        self._connected = False

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def session_id(self) -> str:
        """Get the current session ID.

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._state:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")
        return self._state.session_id

    @property
    def run_id(self) -> str:
        """Get the current run ID.

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._state:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")
        return self._state.run_id

    @property
    def trace_id(self) -> str:
        """Get the current trace ID.

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._state:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")
        return self._state.trace_id

    @property
    def is_connected(self) -> bool:
        """Check if the client is currently connected."""
        return self._connected

    @property
    def total_cost_usd(self) -> float:
        """Get the total cost in USD for this session."""
        if not self._state:
            return 0.0
        return self._state.total_cost_usd

    @property
    def message_count(self) -> int:
        """Get the total number of messages in this session."""
        if not self._state:
            return 0
        return self._state.message_count

    # =========================================================================
    # Sink Setup
    # =========================================================================

    def _setup_sink(self) -> TelemetrySink:
        """Create the appropriate telemetry sink based on configuration.

        Returns:
            Configured TelemetrySink instance
        """
        # Use provided sink if available
        if self._sink:
            return self._sink

        # Prefer explicit endpoint
        if self._endpoint:
            api_key = self._api_key or os.environ.get("ARZULE_API_KEY")
            return HttpBatchSink(
                endpoint_url=self._endpoint,
                api_key=api_key,
            )

        # Use trace file for local mode
        if self._trace_file:
            return JsonlFileSink(path=self._trace_file)

        # Check environment for cloud mode
        api_key = self._api_key or os.environ.get("ARZULE_API_KEY")
        if api_key:
            endpoint = get_claude_ingest_url()
            return HttpBatchSink(
                endpoint_url=endpoint,
                api_key=api_key,
            )

        # Fallback to local file
        traces_dir = Path.home() / ".arzule" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        return JsonlFileSink(path=traces_dir / f"agent_sdk_{new_run_id()}.jsonl")

    # =========================================================================
    # Event Emission
    # =========================================================================

    def _now(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _next_seq(self) -> int:
        """Get next sequence number."""
        self._seq += 1
        return self._seq

    def _emit_event(
        self,
        event_type: str,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        status: str = "ok",
        summary: str = "",
        attrs_compact: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit a trace event to the configured sink.

        Args:
            event_type: Type of event (e.g., "run.start", "tool.end")
            span_id: Span identifier for this event
            parent_span_id: Parent span identifier
            status: Event status ("ok" or "error")
            summary: Human-readable summary
            attrs_compact: Compact attributes for the event
            payload: Additional payload data
        """
        if not self._sink or not self._state:
            return

        # Build agent ID in format: claude_agent_sdk:main:{session_id}
        agent_id = f"claude_agent_sdk:main:{self._state.session_id}"

        event = {
            "schema_version": "trace_event.v0_1",
            "run_id": self._state.run_id,
            "tenant_id": self._tenant_id,
            "project_id": self._project_id,
            "trace_id": self._state.trace_id,
            "span_id": span_id or new_span_id(),
            "parent_span_id": parent_span_id,
            "seq": self._next_seq(),
            "ts": self._now(),
            "agent": {
                "id": agent_id,
                "role": "main",
            },
            "event_type": event_type,
            "status": status,
            "summary": summary,
            "attrs_compact": attrs_compact or {},
            "payload": payload or {},
            "raw_ref": {"storage": "inline"},
        }

        self._sink.write(event)

    # =========================================================================
    # SDK Client Methods
    # =========================================================================

    async def connect(self, prompt: str | None = None) -> None:
        """Connect to Claude with optional initial prompt.

        Args:
            prompt: Optional initial prompt to send on connection

        Raises:
            SessionNotFoundError: If resuming a session that doesn't exist
        """
        # Setup sink
        if not self._sink:
            self._sink = self._setup_sink()

        # Load or create session state
        if self._resume_session_id:
            self._state = load_session_state(self._resume_session_id)
            if not self._state:
                raise SessionNotFoundError(
                    f"Session '{self._resume_session_id}' not found. "
                    "Cannot resume non-existent session."
                )
            # Update last active timestamp
            self._state.last_active_at = self._now()
        else:
            # Create new session
            now = self._now()
            self._state = SessionState(
                session_id=new_run_id(),  # Use run_id format for session_id
                run_id=new_run_id(),
                trace_id=new_trace_id(),
                created_at=now,
                last_active_at=now,
            )

        # Start root span
        self._root_span_id = new_span_id()
        self._emit_event(
            event_type="run.start",
            span_id=self._root_span_id,
            parent_span_id=None,
            summary="Agent SDK session started",
            attrs_compact={
                "session_id": self._state.session_id,
                "resumed": self._resume_session_id is not None,
            },
        )

        # Create hooks for tracing
        tracing_hooks = _create_tracing_hooks(self)

        # Merge with any user-provided hooks
        user_hooks = self._sdk_options.pop("hooks", None)
        if user_hooks:
            for event, matchers in user_hooks.items():
                if event in tracing_hooks:
                    tracing_hooks[event].extend(matchers)
                else:
                    tracing_hooks[event] = matchers

        # Build SDK options with tracing hooks
        options = ClaudeAgentOptions(
            hooks=tracing_hooks,
            resume=self._resume_session_id,
            **self._sdk_options,
        )

        # Create and connect the underlying client
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect(prompt)
        self._connected = True

        # Save state for potential resumption
        save_session_state(self._state)

    async def disconnect(self) -> None:
        """Disconnect from Claude and finalize tracing."""
        if not self._connected or not self._client:
            return

        try:
            await self._client.disconnect()
        finally:
            self._connected = False

            # Emit run end event
            self._emit_event(
                event_type="run.end",
                span_id=new_span_id(),
                parent_span_id=self._root_span_id,
                summary="Agent SDK session ended",
                attrs_compact={
                    "session_id": self._state.session_id if self._state else None,
                    "total_cost_usd": self._state.total_cost_usd if self._state else 0,
                    "message_count": self._state.message_count if self._state else 0,
                },
            )

            # Flush and close sink if we own it
            if self._sink:
                self._sink.flush()
                if self._owns_sink:
                    self._sink.close()

            # Save final state
            if self._state:
                self._state.last_active_at = self._now()
                save_session_state(self._state)

    async def query(self, prompt: str, **kwargs: Any) -> None:
        """Send a query to Claude.

        Args:
            prompt: The prompt to send
            **kwargs: Additional arguments passed to the SDK client

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._connected or not self._client:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")

        await self._client.query(prompt, **kwargs)

    async def receive_response(self) -> AsyncIterator[Message]:
        """Receive messages from Claude until ResultMessage.

        Yields:
            Message: Each message received from Claude

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._connected or not self._client:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")

        async for message in self._client.receive_response():
            # Track message count
            if self._state:
                self._state.message_count += 1

            # Emit message event based on type
            self._emit_message_event(message)

            # Track cost from result messages
            if isinstance(message, ResultMessage):
                if self._state and message.total_cost_usd:
                    self._state.total_cost_usd += message.total_cost_usd

                # End current turn span if active
                if self._current_turn_span:
                    self._emit_event(
                        event_type="turn.end",
                        span_id=new_span_id(),
                        parent_span_id=self._current_turn_span,
                        summary="Turn completed",
                        attrs_compact={
                            "duration_ms": message.duration_ms,
                            "cost_usd": message.total_cost_usd,
                            "num_turns": message.num_turns,
                        },
                    )
                    self._current_turn_span = None

            yield message

    async def receive_messages(self) -> AsyncIterator[Message]:
        """Receive all messages from Claude (raw stream).

        Yields:
            Message: Each message received from Claude

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._connected or not self._client:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")

        async for message in self._client.receive_messages():
            # Track message count
            if self._state:
                self._state.message_count += 1

            # Emit message event based on type
            self._emit_message_event(message)

            yield message

    def _emit_message_event(self, message: Message) -> None:
        """Emit a trace event for a message.

        Args:
            message: The message to emit an event for
        """
        if isinstance(message, AssistantMessage):
            # Extract text content for summary
            text_content = ""
            for block in message.content:
                if hasattr(block, "text"):
                    text_content = block.text[:200] + "..." if len(block.text) > 200 else block.text
                    break

            self._emit_event(
                event_type="llm.response",
                span_id=new_span_id(),
                parent_span_id=self._current_turn_span,
                summary=f"Assistant response ({message.model})",
                attrs_compact={
                    "model": message.model,
                    "content_preview": text_content,
                    "content_blocks": len(message.content),
                },
            )

        elif isinstance(message, UserMessage):
            content = message.content if isinstance(message.content, str) else "[complex content]"
            self._emit_event(
                event_type="user.message",
                span_id=new_span_id(),
                parent_span_id=self._current_turn_span,
                summary="User message",
                attrs_compact={
                    "content_preview": content[:200] if isinstance(content, str) else content,
                },
            )

        elif isinstance(message, SystemMessage):
            self._emit_event(
                event_type="system.message",
                span_id=new_span_id(),
                parent_span_id=self._current_turn_span,
                summary=f"System: {message.subtype}",
                attrs_compact={
                    "subtype": message.subtype,
                    "data": message.data,
                },
            )

        elif isinstance(message, ResultMessage):
            self._emit_event(
                event_type="result",
                span_id=new_span_id(),
                parent_span_id=self._root_span_id,
                status="error" if message.is_error else "ok",
                summary=f"Result ({message.subtype})",
                attrs_compact={
                    "subtype": message.subtype,
                    "duration_ms": message.duration_ms,
                    "duration_api_ms": message.duration_api_ms,
                    "num_turns": message.num_turns,
                    "total_cost_usd": message.total_cost_usd,
                    "is_error": message.is_error,
                },
            )

    async def interrupt(self) -> None:
        """Send interrupt signal to Claude.

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._connected or not self._client:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")

        self._emit_event(
            event_type="interrupt",
            span_id=new_span_id(),
            parent_span_id=self._current_turn_span,
            summary="Interrupt requested",
        )

        await self._client.interrupt()

    async def set_permission_mode(self, mode: str) -> None:
        """Change permission mode during conversation.

        Args:
            mode: The permission mode ('default', 'acceptEdits', 'bypassPermissions')

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._connected or not self._client:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")

        await self._client.set_permission_mode(mode)

    async def set_model(self, model: str | None = None) -> None:
        """Change the AI model during conversation.

        Args:
            model: The model to use, or None for default

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._connected or not self._client:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")

        await self._client.set_model(model)

    async def get_server_info(self) -> dict[str, Any] | None:
        """Get server initialization info.

        Returns:
            Dictionary with server info, or None if not available

        Raises:
            ClientNotConnectedError: If client is not connected
        """
        if not self._connected or not self._client:
            raise ClientNotConnectedError("Client not connected. Call connect() first.")

        return await self._client.get_server_info()

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "TracedClaudeClient":
        """Enter async context - connects to Claude."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Exit async context - disconnects from Claude."""
        # Record error in trace if one occurred
        if exc:
            self._emit_event(
                event_type="error",
                span_id=new_span_id(),
                parent_span_id=self._root_span_id,
                status="error",
                summary=f"Session error: {type(exc).__name__}",
                attrs_compact={
                    "exc_type": type(exc).__name__,
                    "exc_msg": str(exc)[:500],
                },
            )

        await self.disconnect()


# =============================================================================
# Convenience Functions
# =============================================================================


@asynccontextmanager
async def traced_claude_session(
    trace_file: str | Path | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    tenant_id: str | None = None,
    project_id: str | None = None,
    resume: str | None = None,
    **sdk_options: Any,
) -> AsyncIterator[TracedClaudeClient]:
    """Context manager for a traced Claude session.

    This is a convenience function that creates and manages a TracedClaudeClient.

    Args:
        trace_file: Path to write trace events as JSONL (local mode)
        endpoint: HTTP endpoint for trace ingestion (cloud mode)
        api_key: API key for authentication
        tenant_id: Tenant identifier
        project_id: Project identifier
        resume: Session ID to resume
        **sdk_options: Additional options for ClaudeSDKClient

    Yields:
        TracedClaudeClient: Connected and ready-to-use client

    Example:
        ```python
        async with traced_claude_session(
            trace_file="./traces/session.jsonl"
        ) as client:
            await client.query("Hello!")
            async for msg in client.receive_response():
                print(msg)
        ```
    """
    client = TracedClaudeClient(
        trace_file=trace_file,
        endpoint=endpoint,
        api_key=api_key,
        tenant_id=tenant_id,
        project_id=project_id,
        resume=resume,
        **sdk_options,
    )

    async with client:
        yield client
