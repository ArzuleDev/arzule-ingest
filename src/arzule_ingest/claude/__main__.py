"""CLI entry point for Claude Code hook handler.

This module enables running the hook handler via:
    python -m arzule_ingest.claude.hook

This is the command that gets registered in Claude Code's settings.json.
"""

from .hook import handle_hook

if __name__ == "__main__":
    handle_hook()
