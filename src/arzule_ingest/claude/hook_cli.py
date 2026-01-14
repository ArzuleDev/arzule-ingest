"""Lightweight CLI entry point for Claude Code hooks.

This module provides the `arzule-hook` command which is invoked by Claude Code
hooks configured in .claude/settings.local.json.

The hook reads JSON input from stdin and emits trace events to Arzule.

Usage:
    echo '{"hook_event_name": "...", ...}' | arzule-hook
"""

from __future__ import annotations


def main() -> None:
    """
    CLI entry point for arzule-hook command.

    Imports are deferred to keep startup fast since hooks are invoked
    frequently during Claude Code sessions.
    """
    from .hook import handle_hook
    handle_hook()


if __name__ == "__main__":
    main()
