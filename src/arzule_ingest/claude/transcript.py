"""Transcript parsing utilities for Claude Code.

Parses JSONL transcript files to extract conversation history,
tool calls, and agent interactions for trace reconstruction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..sanitize import truncate_string


def parse_transcript(transcript_path: str) -> Optional[list[dict]]:
    """
    Parse JSONL transcript file into list of messages.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        List of message dicts or None if parsing failed
    """
    try:
        path = Path(transcript_path)
        if not path.exists():
            return None

        messages = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return messages if messages else None
    except Exception:
        return None


def extract_session_summary(transcript_path: str) -> Optional[str]:
    """
    Extract a summary from the transcript (last assistant message).

    Args:
        transcript_path: Path to the transcript file

    Returns:
        Summary string or None
    """
    try:
        path = Path(transcript_path)
        if not path.exists():
            return None

        with open(path, "r") as f:
            lines = f.readlines()

        # Find last assistant message
        for line in reversed(lines):
            try:
                msg = json.loads(line.strip())
                if msg.get("type") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return truncate_string(content, 200)
                    elif isinstance(content, list):
                        # Handle content blocks
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                return truncate_string(block.get("text", ""), 200)
            except json.JSONDecodeError:
                continue

        return None
    except Exception:
        return None


def extract_tool_calls(transcript_path: str) -> list[dict]:
    """
    Extract all tool calls from the transcript.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        List of tool call dicts with name, input, output
    """
    messages = parse_transcript(transcript_path)
    if not messages:
        return []

    tool_calls = []
    pending_tools: dict[str, dict] = {}

    for msg in messages:
        msg_type = msg.get("type")

        # Assistant message with tool_use
        if msg_type == "assistant":
            tool_uses = msg.get("tool_use", [])
            if not isinstance(tool_uses, list):
                tool_uses = [tool_uses] if tool_uses else []

            for tool_use in tool_uses:
                if isinstance(tool_use, dict):
                    tool_id = tool_use.get("id") or tool_use.get("tool_use_id")
                    if tool_id:
                        pending_tools[tool_id] = {
                            "name": tool_use.get("name"),
                            "input": tool_use.get("input"),
                            "id": tool_id,
                        }

        # Tool result
        elif msg_type == "tool_result":
            tool_id = msg.get("tool_use_id")
            if tool_id and tool_id in pending_tools:
                tool_call = pending_tools.pop(tool_id)
                tool_call["output"] = msg.get("content")
                tool_call["error"] = msg.get("error")
                tool_calls.append(tool_call)

    return tool_calls


def extract_subagent_delegations(transcript_path: str) -> list[dict]:
    """
    Extract Task tool calls (subagent delegations) from transcript.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        List of delegation dicts with subagent_type, description, prompt, result
    """
    tool_calls = extract_tool_calls(transcript_path)

    delegations = []
    for call in tool_calls:
        if call.get("name") == "Task":
            tool_input = call.get("input", {})
            if isinstance(tool_input, dict):
                delegations.append({
                    "tool_use_id": call.get("id"),
                    "subagent_type": tool_input.get("subagent_type", "general-purpose"),
                    "description": tool_input.get("description", ""),
                    "prompt": tool_input.get("prompt", ""),
                    "model": tool_input.get("model"),
                    "result": call.get("output"),
                    "error": call.get("error"),
                })

    return delegations


def extract_message_count(transcript_path: str) -> int:
    """
    Count total messages in transcript.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        Number of messages
    """
    messages = parse_transcript(transcript_path)
    return len(messages) if messages else 0


def extract_user_messages(transcript_path: str) -> list[str]:
    """
    Extract user messages from transcript.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        List of user message strings
    """
    messages = parse_transcript(transcript_path)
    if not messages:
        return []

    user_messages = []
    for msg in messages:
        if msg.get("type") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_messages.append(content)

    return user_messages


def extract_assistant_turns(transcript_path: str) -> int:
    """
    Count assistant turns (responses) in transcript.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        Number of assistant turns
    """
    messages = parse_transcript(transcript_path)
    if not messages:
        return 0

    return sum(1 for msg in messages if msg.get("type") == "assistant")


def get_transcript_stats(transcript_path: str) -> dict[str, Any]:
    """
    Get statistics from a transcript.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        Dict with transcript statistics
    """
    messages = parse_transcript(transcript_path)
    if not messages:
        return {}

    tool_calls = extract_tool_calls(transcript_path)
    delegations = extract_subagent_delegations(transcript_path)

    # Count by type
    type_counts: dict[str, int] = {}
    for msg in messages:
        msg_type = msg.get("type", "unknown")
        type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

    # Count tool usage
    tool_counts: dict[str, int] = {}
    for call in tool_calls:
        tool_name = call.get("name", "unknown")
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

    return {
        "total_messages": len(messages),
        "message_types": type_counts,
        "tool_calls": len(tool_calls),
        "tool_usage": tool_counts,
        "subagent_delegations": len(delegations),
        "subagent_types": [d.get("subagent_type") for d in delegations],
    }
