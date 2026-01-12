"""Claude Code instrumentation for Arzule observability.

This module provides observability instrumentation for Claude Code (Anthropic's CLI agent).
It captures session lifecycle, tool calls, subagent delegations (handoffs), and context events.

Example usage:
    # RECOMMENDED: Use the wrapper command (captures hooks + OTel metrics)
    $ arzule-claude "your prompt"
    
    # The wrapper automatically:
    # - Configures OTel to export token/cost metrics to Arzule
    # - Passes through all arguments to Claude Code
    # - Uses your ~/.arzule/config credentials

    # Alternative: Install hooks only (no OTel metrics)
    $ arzule-claude-install install

    # Environment variables (set via 'arzule configure' or ~/.arzule/config):
    # ARZULE_API_KEY - Your Arzule API key
    # ARZULE_TENANT_ID - Your tenant ID
    # ARZULE_PROJECT_ID - Your project ID

    # Optional: Enable real-time streaming
    # ARZULE_STREAM_URL - Local observability server URL (e.g., http://localhost:4000/events)

Architecture:
    Claude Code sessions are long-running conversational contexts (can span hours/days).
    A "turn" (UserPromptSubmit -> Stop) is the atomic unit of work we treat as a run.
    
    Session "abc123":
      Turn 1 (run_id: turn-001):
        UserPromptSubmit -> PreToolUse(Read) -> PostToolUse(Read) -> Stop
      Turn 2 (run_id: turn-002):
        UserPromptSubmit -> PreToolUse(Bash) -> PostToolUse(Bash) -> Stop
      ...

Agent ID Schema:
    claude_code:main:{session_id} - Main orchestrator agent
    claude_code:subagent:{type}:{tool_use_id} - Subagents (Explore, Plan, Bash, etc.)

Event Types:
    session.start - Session initialized
    session.end - Session terminated
    turn.start - New conversation turn started (user prompt received)
    turn.end - Conversation turn completed (agent response finished)
    tool.call.start - Tool invocation begins
    tool.call.end - Tool invocation completes
    tool.call.blocked - Dangerous command blocked
    handoff.proposed - Main agent delegates to subagent
    handoff.ack - Subagent begins work
    handoff.complete - Subagent returns result
    context.compact - Context window compacted
    notification - User notification event
"""

from .install import install_claude_code, uninstall_claude_code, is_installed
from .wrapper import run_claude_with_otel
from .turn import (
    start_turn,
    end_turn,
    get_current_turn,
    get_or_create_run,
    update_turn_state,
    emit_with_seq_sync,
)
# Backwards compatibility - deprecated, use turn-based API
from .session import get_or_create_session, get_session_state

__all__ = [
    # Wrapper (recommended entry point)
    "run_claude_with_otel",
    # Installation utilities
    "install_claude_code",
    "uninstall_claude_code",
    "is_installed",
    # Turn-based API (recommended)
    "start_turn",
    "end_turn",
    "get_current_turn",
    "get_or_create_run",
    "update_turn_state",
    # Session management (deprecated, for backwards compatibility)
    "get_or_create_session",
    "get_session_state",
]
