"""Installation utilities for Claude Code hooks.

Provides functions to install/uninstall Arzule instrumentation hooks
in Claude Code's settings.json configuration.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


def _get_hook_command(portable: bool = True) -> str:
    """
    Generate the hook command for Claude Code.

    Args:
        portable: If True (default), returns a portable command that works
                  across machines. If False, uses the current Python executable.

    Returns:
        Command string for hook configuration.

    The portable command tries arzule-hook first (fast, if installed),
    then falls back to uvx for zero-install experience.
    """
    if portable:
        # Portable: tries arzule-hook first, falls back to uvx
        return 'command -v arzule-hook >/dev/null 2>&1 && arzule-hook || uvx arzule-ingest hook'
    else:
        # Legacy: absolute path to current Python (not portable)
        python_exe = sys.executable
        return f'{python_exe} -m arzule_ingest.claude.hook'


def _build_hook_config(portable: bool = True) -> dict:
    """Build hook configuration with dynamic command.

    Args:
        portable: If True, uses portable command (arzule-hook with uvx fallback).
                  If False, uses absolute Python path (legacy).
    """
    command = _get_hook_command(portable=portable)

    return {
        "SessionStart": [
            {"hooks": [{"type": "command", "command": command}]}
        ],
        "SessionEnd": [
            {"hooks": [{"type": "command", "command": command}]}
        ],
        "UserPromptSubmit": [
            # CRITICAL: This starts a new turn - enables per-turn tracking
            {"hooks": [{"type": "command", "command": command}]}
        ],
        "PreToolUse": [
            {"matcher": ".*", "hooks": [{"type": "command", "command": command}]}
        ],
        "PostToolUse": [
            {"matcher": ".*", "hooks": [{"type": "command", "command": command}]}
        ],
        "SubagentStart": [
            # CRITICAL: Captures subagent start for agent_id mapping (v2.0.43+)
            {"hooks": [{"type": "command", "command": command}]}
        ],
        "SubagentStop": [
            # CRITICAL: Captures agent_id and agent_transcript_path for definitive attribution
            {"hooks": [{"type": "command", "command": command}]}
        ],
        "Stop": [
            # CRITICAL: This ends the current turn
            {"hooks": [{"type": "command", "command": command}]}
        ],
        "PreCompact": [
            {"hooks": [{"type": "command", "command": command}]}
        ],
        "Notification": [
            {"hooks": [{"type": "command", "command": command}]}
        ],
    }


# Hook configuration (built dynamically with portable command)
HOOK_CONFIG = _build_hook_config(portable=True)

# Identifiers to recognize Arzule hooks (old and new formats)
ARZULE_HOOK_MARKERS = [
    "arzule_ingest.claude.hook",  # Legacy: python -m arzule_ingest.claude.hook
    "arzule-hook",                 # New: arzule-hook CLI
    "uvx arzule-ingest",           # Fallback: uvx arzule-ingest hook
]


def get_settings_paths() -> list[Path]:
    """
    Get possible locations for Claude Code settings.

    Returns:
        List of potential settings.json paths (global and local)
    """
    paths = []

    # Global settings: ~/.claude/settings.json
    global_path = Path.home() / ".claude" / "settings.json"
    paths.append(global_path)

    # Local settings: .claude/settings.json in current directory
    local_path = Path.cwd() / ".claude" / "settings.json"
    if local_path != global_path:
        paths.append(local_path)

    return paths


def find_settings_file() -> Optional[Path]:
    """
    Find the first existing settings file.

    Returns:
        Path to settings file or None if not found
    """
    for path in get_settings_paths():
        if path.exists():
            return path
    return None


def get_global_settings_path() -> Path:
    """Get the global settings path (~/.claude/settings.json)."""
    return Path.home() / ".claude" / "settings.json"


def get_local_settings_path() -> Path:
    """Get the local/project settings path (.claude/settings.local.json).

    This is the recommended location for project-level hooks.
    Using settings.local.json ensures hooks are user-specific and git-ignored.
    """
    return Path.cwd() / ".claude" / "settings.local.json"


def get_default_settings_path() -> Path:
    """Get the default settings path for installation.

    Returns project-level path if in a project with .claude folder,
    otherwise returns global path.
    """
    # If we're in a project with .claude folder, use local
    claude_dir = Path.cwd() / ".claude"
    if claude_dir.exists() and claude_dir.is_dir():
        return get_local_settings_path()

    # Otherwise, use global
    return get_global_settings_path()


def load_settings(settings_path: Path) -> dict:
    """
    Load settings from file.

    Args:
        settings_path: Path to settings.json

    Returns:
        Settings dict (empty if file doesn't exist)
    """
    if not settings_path.exists():
        return {}

    try:
        with open(settings_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_settings(settings_path: Path, settings: dict) -> bool:
    """
    Save settings to file.

    Args:
        settings_path: Path to settings.json
        settings: Settings dict to save

    Returns:
        True if successful
    """
    try:
        # Ensure parent directory exists
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except IOError:
        return False


def install_claude_code(
    settings_path: Optional[Path] = None,
    *,
    events: Optional[list[str]] = None,
    force: bool = False,
    portable: bool = True,
) -> bool:
    """
    Install Arzule hooks into Claude Code settings.

    Args:
        settings_path: Optional explicit path to settings file.
                       Defaults to project-level (.claude/settings.local.json) if
                       in a project with .claude folder, otherwise global.
        events: Optional list of specific events to hook (defaults to all)
        force: If True, overwrite existing hooks
        portable: If True (default), uses portable command (arzule-hook with uvx fallback).
                  If False, uses absolute Python path (not recommended for project-level).

    Returns:
        True if installation succeeded
    """
    if settings_path is None:
        settings_path = get_default_settings_path()

    # Build hook config with appropriate command
    hook_config = _build_hook_config(portable=portable)

    settings = load_settings(settings_path)

    # Initialize hooks section if needed
    if "hooks" not in settings:
        settings["hooks"] = {}

    # Determine which events to install
    events_to_install = events or list(hook_config.keys())

    for event_name in events_to_install:
        if event_name not in hook_config:
            continue

        event_hooks = hook_config[event_name]

        if event_name not in settings["hooks"]:
            settings["hooks"][event_name] = []

        # Check if Arzule hook already installed
        existing_hooks = settings["hooks"][event_name]
        arzule_installed = any(
            _is_arzule_hook(hook_entry)
            for hook_entry in existing_hooks
        )

        if arzule_installed and not force:
            # Already installed, skip
            continue

        if arzule_installed and force:
            # Remove existing Arzule hooks
            settings["hooks"][event_name] = [
                hook_entry for hook_entry in existing_hooks
                if not _is_arzule_hook(hook_entry)
            ]

        # Add Arzule hooks
        settings["hooks"][event_name].extend(event_hooks)

    return save_settings(settings_path, settings)


def uninstall_claude_code(settings_path: Optional[Path] = None) -> bool:
    """
    Remove Arzule hooks from Claude Code settings.

    Args:
        settings_path: Optional explicit path to settings.json

    Returns:
        True if uninstallation succeeded
    """
    if settings_path is None:
        settings_path = find_settings_file()
        if settings_path is None:
            return True  # Nothing to uninstall

    settings = load_settings(settings_path)

    if "hooks" not in settings:
        return True  # No hooks, nothing to uninstall

    # Remove Arzule hooks from each event
    for event_name in list(settings["hooks"].keys()):
        hooks = settings["hooks"][event_name]
        settings["hooks"][event_name] = [
            hook_entry for hook_entry in hooks
            if not _is_arzule_hook(hook_entry)
        ]

        # Remove empty event arrays
        if not settings["hooks"][event_name]:
            del settings["hooks"][event_name]

    # Remove empty hooks section
    if not settings["hooks"]:
        del settings["hooks"]

    return save_settings(settings_path, settings)


def is_installed(settings_path: Optional[Path] = None) -> bool:
    """
    Check if Arzule hooks are installed.

    Args:
        settings_path: Optional explicit path to settings.json

    Returns:
        True if at least one Arzule hook is installed
    """
    if settings_path is None:
        settings_path = find_settings_file()
        if settings_path is None:
            return False

    settings = load_settings(settings_path)

    if "hooks" not in settings:
        return False

    for event_name, hooks in settings["hooks"].items():
        for hook_entry in hooks:
            if _is_arzule_hook(hook_entry):
                return True

    return False


def get_installation_status(settings_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Get detailed installation status.

    Args:
        settings_path: Optional explicit path to settings.json

    Returns:
        Dict with installation status details
    """
    if settings_path is None:
        settings_path = find_settings_file()

    status = {
        "installed": False,
        "settings_path": str(settings_path) if settings_path else None,
        "events": {},
    }

    if settings_path is None or not settings_path.exists():
        return status

    settings = load_settings(settings_path)

    if "hooks" not in settings:
        return status

    for event_name in HOOK_CONFIG.keys():
        hooks = settings["hooks"].get(event_name, [])
        arzule_installed = any(_is_arzule_hook(h) for h in hooks)
        status["events"][event_name] = arzule_installed
        if arzule_installed:
            status["installed"] = True

    return status


def _is_arzule_hook(hook_entry: dict) -> bool:
    """Check if a hook entry is an Arzule hook.

    Recognizes both old format (python -m arzule_ingest.claude.hook)
    and new format (arzule-hook with uvx fallback).
    """
    hooks = hook_entry.get("hooks", [])
    for hook in hooks:
        command = hook.get("command", "")
        for marker in ARZULE_HOOK_MARKERS:
            if marker in command:
                return True
    return False


def generate_settings_json() -> str:
    """
    Generate the hooks section for manual installation.

    Returns:
        JSON string with hook configuration
    """
    config = {"hooks": HOOK_CONFIG}
    return json.dumps(config, indent=2)


def print_installation_instructions() -> None:
    """Print manual installation instructions."""
    print("=" * 60)
    print("Arzule Claude Code Instrumentation")
    print("=" * 60)
    print()
    print("RECOMMENDED: Use the wrapper command for full observability:")
    print()
    print("  $ arzule-claude \"your prompt\"")
    print()
    print("This captures BOTH hooks data AND OTel metrics (tokens, costs).")
    print()
    print("-" * 60)
    print("Alternative: Hooks-only installation")
    print("-" * 60)
    print()
    print("Add the following to your ~/.claude/settings.json file:")
    print()
    print(generate_settings_json())
    print()
    print("=" * 60)
    print("Configuration (run 'arzule configure' to set up):")
    print("  ARZULE_API_KEY - Your Arzule API key")
    print("  ARZULE_TENANT_ID - Your tenant ID")
    print("  ARZULE_PROJECT_ID - Your project ID")
    print()
    print("Config file: ~/.arzule/config")
    print("=" * 60)


# CLI entry point
def main():
    """CLI entry point for installation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Install/uninstall Arzule instrumentation for Claude Code"
    )
    parser.add_argument(
        "command",
        choices=["install", "uninstall", "status", "show"],
        help="Command to run"
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Explicit path to settings file (overrides --global/--local)"
    )

    # Location flags (mutually exclusive)
    location_group = parser.add_mutually_exclusive_group()
    location_group.add_argument(
        "--global", "-g",
        dest="use_global",
        action="store_true",
        help="Use global settings (~/.claude/settings.json)"
    )
    location_group.add_argument(
        "--local", "-l",
        dest="use_local",
        action="store_true",
        help="Use project-level settings (.claude/settings.local.json) [default if .claude exists]"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstallation even if already installed"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy absolute Python path instead of portable command (not recommended)"
    )

    args = parser.parse_args()

    # Determine settings path
    if args.path:
        settings_path = args.path
    elif args.use_global:
        settings_path = get_global_settings_path()
    elif args.use_local:
        settings_path = get_local_settings_path()
    else:
        settings_path = get_default_settings_path()

    if args.command == "install":
        portable = not args.legacy
        if install_claude_code(settings_path, force=args.force, portable=portable):
            print("Arzule hooks installed successfully")
            status = get_installation_status(settings_path)
            print(f"  Settings: {status['settings_path']}")

            # Show location info
            is_local = "settings.local.json" in str(settings_path)
            if is_local:
                print("  Type: Project-level (user-specific, git-ignored)")
            else:
                print("  Type: Global (applies to all projects)")

            print()
            print("Events hooked:")
            for event, installed in status["events"].items():
                icon = "[x]" if installed else "[ ]"
                print(f"  {icon} {event}")
            print()
            print("Now run 'claude' in this project to start tracing!")
        else:
            print("Failed to install hooks", file=sys.stderr)
            sys.exit(1)

    elif args.command == "uninstall":
        if uninstall_claude_code(settings_path):
            print(f"Arzule hooks uninstalled from {settings_path}")
        else:
            print("Failed to uninstall hooks", file=sys.stderr)
            sys.exit(1)

    elif args.command == "status":
        # Check both locations
        global_status = get_installation_status(get_global_settings_path())
        local_status = get_installation_status(get_local_settings_path())

        print("Arzule Hook Installation Status")
        print("=" * 40)
        print()

        # Global status
        print(f"Global (~/.claude/settings.json):")
        if global_status["installed"]:
            print("  Status: Installed")
            for event, installed in global_status["events"].items():
                if installed:
                    print(f"    [x] {event}")
        else:
            print("  Status: Not installed")
        print()

        # Local status
        print(f"Project (.claude/settings.local.json):")
        if local_status["installed"]:
            print("  Status: Installed")
            for event, installed in local_status["events"].items():
                if installed:
                    print(f"    [x] {event}")
        else:
            print("  Status: Not installed")
        print()

        # Recommendation
        if global_status["installed"] and not local_status["installed"]:
            print("Tip: Consider migrating to project-level hooks:")
            print("  arzule-claude-install uninstall --global")
            print("  arzule-claude-install install --local")
        elif not global_status["installed"] and not local_status["installed"]:
            print("No Arzule hooks installed.")
            print()
            print("Run 'arzule-claude-install install' to install hooks")
            print("Or use 'arzule init' to initialize this project")

    elif args.command == "show":
        print_installation_instructions()


if __name__ == "__main__":
    main()
