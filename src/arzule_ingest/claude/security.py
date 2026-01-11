"""Security validation for Claude Code tool calls.

Validates tool inputs before execution to block dangerous commands
and emit security events for sensitive operations.

Inspired by: https://github.com/disler/claude-code-hooks-multi-agent-observability
"""

from __future__ import annotations

import re
from typing import Tuple

# Dangerous command patterns to block
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"rm\s+-rf\s+/", "Recursive delete from root"),
    (r"rm\s+-rf\s+~", "Recursive delete from home"),
    (r"rm\s+-rf\s+\*", "Recursive delete wildcard"),
    (r">\s*/dev/sd", "Write to block device"),
    (r"dd\s+if=.*of=/dev/", "Direct disk write"),
    (r"mkfs\.", "Filesystem format"),
    (r"chmod\s+777\s+/", "Overly permissive root chmod"),
    (r"curl.*\|\s*(?:ba)?sh", "Pipe URL to shell"),
    (r"wget.*\|\s*(?:ba)?sh", "Pipe URL to shell"),
    (r":()\s*\{\s*:\|\:&\s*\}", "Fork bomb"),
    (r">\s*/etc/passwd", "Overwrite passwd"),
    (r">\s*/etc/shadow", "Overwrite shadow"),
    (r"cat\s+/etc/shadow", "Read shadow file"),
    (r"base64\s+-d.*\|\s*(?:ba)?sh", "Decode and execute"),
    (r"python.*-c.*exec\s*\(", "Python exec injection"),
    (r"eval\s*\$\(", "Eval command substitution"),
    (r"\$\(.*curl.*\)", "Command substitution with curl"),
]

# Sensitive file patterns to warn about
SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    (r"\.env", "Environment file access"),
    (r"\.ssh/", "SSH directory access"),
    (r"id_rsa", "SSH private key"),
    (r"\.aws/credentials", "AWS credentials"),
    (r"\.git/config", "Git config (may contain tokens)"),
    (r"\.npmrc", "NPM config (may contain tokens)"),
    (r"\.pypirc", "PyPI config (may contain tokens)"),
    (r"credentials\.json", "Credentials file"),
    (r"secrets\.json", "Secrets file"),
    (r"\.kube/config", "Kubernetes config"),
]

# System paths that should not be written to
PROTECTED_WRITE_PATHS: list[str] = [
    "/etc/",
    "/bin/",
    "/sbin/",
    "/usr/bin/",
    "/usr/sbin/",
    "/boot/",
    "/sys/",
    "/proc/",
    "/dev/",
]


def validate_command(command: str) -> Tuple[bool, str, str]:
    """
    Validate a bash command for dangerous patterns.

    Args:
        command: The bash command to validate

    Returns:
        Tuple of (is_safe, severity, reason)
        - is_safe: True if command should be allowed
        - severity: "block", "warn", or "ok"
        - reason: Human-readable reason if not safe
    """
    if not command:
        return True, "ok", ""

    # Check for dangerous patterns (block)
    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, "block", reason

    # Check for sensitive patterns (warn but allow)
    for pattern, reason in SENSITIVE_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True, "warn", reason

    return True, "ok", ""


def validate_file_write(file_path: str) -> Tuple[bool, str, str]:
    """
    Validate a file write operation.

    Args:
        file_path: The path being written to

    Returns:
        Tuple of (is_safe, severity, reason)
    """
    if not file_path:
        return True, "ok", ""

    # Block writes to protected system paths
    for protected_path in PROTECTED_WRITE_PATHS:
        if file_path.startswith(protected_path):
            return False, "block", f"Write to protected path: {protected_path}"

    # Warn about sensitive file writes
    for pattern, reason in SENSITIVE_PATTERNS:
        if re.search(pattern, file_path, re.IGNORECASE):
            return True, "warn", reason

    return True, "ok", ""


def validate_tool_input(tool_name: str, tool_input: dict) -> Tuple[bool, str, str]:
    """
    Validate tool input before execution.

    Args:
        tool_name: Name of the tool being called
        tool_input: Tool input parameters

    Returns:
        Tuple of (is_safe, severity, reason)
    """
    if not tool_input:
        return True, "ok", ""

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        return validate_command(command)

    if tool_name == "Write":
        file_path = tool_input.get("file_path", "")
        return validate_file_write(file_path)

    if tool_name == "Edit":
        file_path = tool_input.get("file_path", "")
        return validate_file_write(file_path)

    # Task tool - validate the prompt for injection attempts
    if tool_name == "Task":
        prompt = tool_input.get("prompt", "")
        # Check if the prompt contains dangerous commands
        # This catches prompt injection attempts
        for pattern, reason in DANGEROUS_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True, "warn", f"Prompt contains pattern: {reason}"

    return True, "ok", ""


def should_emit_security_event(tool_name: str, tool_input: dict) -> bool:
    """
    Check if a security event should be emitted for this tool call.

    Args:
        tool_name: Name of the tool
        tool_input: Tool input parameters

    Returns:
        True if a security event should be emitted
    """
    is_safe, severity, _ = validate_tool_input(tool_name, tool_input)
    return severity in ("block", "warn")


def get_security_event_details(
    tool_name: str,
    tool_input: dict,
) -> dict:
    """
    Get details for a security event.

    Args:
        tool_name: Name of the tool
        tool_input: Tool input parameters

    Returns:
        Dict with security event details
    """
    is_safe, severity, reason = validate_tool_input(tool_name, tool_input)

    return {
        "tool_name": tool_name,
        "severity": severity,
        "reason": reason,
        "blocked": not is_safe,
    }
