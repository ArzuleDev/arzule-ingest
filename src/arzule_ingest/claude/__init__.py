"""Claude Code instrumentation for Arzule observability.

This module provides observability instrumentation for Claude Code (Anthropic's CLI agent).
It captures session lifecycle, tool calls, subagent delegations (handoffs), and context events.

Example usage:
    # Option 1: Install hooks via CLI
    $ arzule-claude install

    # Option 2: Manual hook configuration in ~/.claude/settings.json
    # See install.py for the hook configuration

    # Environment variables required:
    # ARZULE_API_KEY - Your Arzule API key
    # ARZULE_TENANT_ID - Your tenant ID
    # ARZULE_PROJECT_ID - Your project ID

Agent ID Schema:
    claude_code:main:{session_id} - Main orchestrator agent
    claude_code:subagent:{type}:{tool_use_id} - Subagents (Explore, Plan, Bash, etc.)

Event Types:
    session.start - Session initialized
    session.end - Session terminated
    session.transcript - Full conversation captured
    tool.call.start - Tool invocation begins
    tool.call.end - Tool invocation completes
    tool.call.blocked - Dangerous command blocked
    handoff.proposed - Main agent delegates to subagent
    handoff.ack - Subagent begins work
    handoff.complete - Subagent returns result
    agent.response.complete - Main agent response finished
    context.compact - Context window compacted
    notification - User notification event
"""

from .install import install_claude_code, uninstall_claude_code, is_installed

__all__ = ["install_claude_code", "uninstall_claude_code", "is_installed"]
