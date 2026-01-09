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

# Hook configuration for each event type
HOOK_CONFIG = {
    "SessionStart": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m arzule_ingest.claude.hook"
                }
            ]
        }
    ],
    "SessionEnd": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m arzule_ingest.claude.hook"
                }
            ]
        }
    ],
    "PreToolUse": [
        {
            "matcher": ".*",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m arzule_ingest.claude.hook"
                }
            ]
        }
    ],
    "PostToolUse": [
        {
            "matcher": ".*",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m arzule_ingest.claude.hook"
                }
            ]
        }
    ],
    "SubagentStop": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m arzule_ingest.claude.hook"
                }
            ]
        }
    ],
    "Stop": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m arzule_ingest.claude.hook"
                }
            ]
        }
    ],
    "PreCompact": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m arzule_ingest.claude.hook"
                }
            ]
        }
    ],
    "Notification": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{sys.executable} -m arzule_ingest.claude.hook"
                }
            ]
        }
    ],
}

# Identifier to recognize Arzule hooks
ARZULE_HOOK_MARKER = "arzule_ingest.claude.hook"


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
    """Get the global settings path."""
    return Path.home() / ".claude" / "settings.json"


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
) -> bool:
    """
    Install Arzule hooks into Claude Code settings.

    Args:
        settings_path: Optional explicit path to settings.json (uses global if not specified)
        events: Optional list of specific events to hook (defaults to all)
        force: If True, overwrite existing hooks

    Returns:
        True if installation succeeded
    """
    if settings_path is None:
        settings_path = get_global_settings_path()

    settings = load_settings(settings_path)

    # Initialize hooks section if needed
    if "hooks" not in settings:
        settings["hooks"] = {}

    # Determine which events to install
    events_to_install = events or list(HOOK_CONFIG.keys())

    for event_name in events_to_install:
        if event_name not in HOOK_CONFIG:
            continue

        hook_config = HOOK_CONFIG[event_name]

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
        settings["hooks"][event_name].extend(hook_config)

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
    """Check if a hook entry is an Arzule hook."""
    hooks = hook_entry.get("hooks", [])
    for hook in hooks:
        command = hook.get("command", "")
        if ARZULE_HOOK_MARKER in command:
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
    print("Arzule Claude Code Instrumentation - Manual Installation")
    print("=" * 60)
    print()
    print("Add the following to your ~/.claude/settings.json file:")
    print()
    print(generate_settings_json())
    print()
    print("=" * 60)
    print("Environment Variables Required:")
    print("  ARZULE_API_KEY - Your Arzule API key")
    print("  ARZULE_TENANT_ID - Your tenant ID")
    print("  ARZULE_PROJECT_ID - Your project ID")
    print()
    print("Without these variables, traces will be saved locally to:")
    print("  ~/.arzule/traces/claude_{session_id}.jsonl")
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
        help="Path to settings.json (defaults to ~/.claude/settings.json)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstallation even if already installed"
    )

    args = parser.parse_args()

    if args.command == "install":
        if install_claude_code(args.path, force=args.force):
            print("✓ Arzule hooks installed successfully")
            status = get_installation_status(args.path)
            print(f"  Settings: {status['settings_path']}")
        else:
            print("✗ Failed to install hooks", file=sys.stderr)
            sys.exit(1)

    elif args.command == "uninstall":
        if uninstall_claude_code(args.path):
            print("✓ Arzule hooks uninstalled")
        else:
            print("✗ Failed to uninstall hooks", file=sys.stderr)
            sys.exit(1)

    elif args.command == "status":
        status = get_installation_status(args.path)
        if status["installed"]:
            print("✓ Arzule hooks are installed")
            print(f"  Settings: {status['settings_path']}")
            print("  Events:")
            for event, installed in status["events"].items():
                icon = "✓" if installed else "✗"
                print(f"    {icon} {event}")
        else:
            print("✗ Arzule hooks are not installed")

    elif args.command == "show":
        print_installation_instructions()


if __name__ == "__main__":
    main()
