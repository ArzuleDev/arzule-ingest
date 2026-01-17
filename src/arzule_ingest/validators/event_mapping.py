"""Event mapping utilities for validators.

This module provides functions to map between Claude Code hook events
and validator API data structures. It handles:

1. Extracting relevant data from hook input
2. Building spawn requests from PreToolUse events
3. Correlating SubagentStop with pending validations
4. Mapping validation results to hook responses

The mapping logic handles the complexities of:
- Parallel subagent execution (multiple Tasks running concurrently)
- Missing or incomplete data in hook events
- Different Claude Code versions with varying event schemas
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .types import (
    SpawnRequest,
    ValidationResult,
    ValidationDecision,
    HookContext,
)


@dataclass
class TaskToolInput:
    """Parsed input from a Task tool call.

    Attributes:
        subagent_type: Type of subagent to spawn.
        description: Short task description (3-5 words).
        prompt: Full task prompt/instructions.
        model: Optional model override.
        run_in_background: Whether to run asynchronously.
    """

    subagent_type: str
    description: str
    prompt: str
    model: Optional[str] = None
    run_in_background: bool = False

    @classmethod
    def from_hook_input(cls, tool_input: dict[str, Any]) -> TaskToolInput:
        """Create TaskToolInput from hook tool_input dict.

        Args:
            tool_input: The tool_input field from PreToolUse hook.

        Returns:
            Parsed TaskToolInput.
        """
        return cls(
            subagent_type=tool_input.get("subagent_type", "general-purpose"),
            description=tool_input.get("description", ""),
            prompt=tool_input.get("prompt", ""),
            model=tool_input.get("model"),
            run_in_background=tool_input.get("run_in_background", False),
        )


def extract_pre_tool_use_context(
    input_data: dict[str, Any]
) -> Optional[HookContext]:
    """Extract context from a PreToolUse (Task) hook event.

    This function extracts the relevant fields from a PreToolUse hook
    event for use in validator spawning and correlation.

    Args:
        input_data: Raw hook input data from stdin.

    Returns:
        HookContext if this is a Task tool event, None otherwise.
    """
    if input_data.get("hook_event_name") != "PreToolUse":
        return None

    if input_data.get("tool_name") != "Task":
        return None

    tool_input = input_data.get("tool_input", {})

    return HookContext(
        session_id=input_data.get("session_id", ""),
        tool_use_id=input_data.get("tool_use_id"),
        subagent_type=tool_input.get("subagent_type", "general-purpose"),
    )


def extract_subagent_stop_context(
    input_data: dict[str, Any]
) -> Optional[HookContext]:
    """Extract context from a SubagentStop hook event.

    This function extracts the relevant fields from a SubagentStop hook
    event for use in result retrieval and correlation.

    Args:
        input_data: Raw hook input data from stdin.

    Returns:
        HookContext if this is a SubagentStop event, None otherwise.
    """
    if input_data.get("hook_event_name") != "SubagentStop":
        return None

    return HookContext(
        session_id=input_data.get("session_id", ""),
        agent_id=input_data.get("agent_id"),
    )


def build_spawn_request(
    context: HookContext,
    tool_input: TaskToolInput,
    validators: list[str],
    transcript_path: str = "",
    validation_level: str = "standard",
    model: str = "claude-3-5-haiku-20241022",
) -> SpawnRequest:
    """Build a SpawnRequest from hook context and tool input.

    Args:
        context: HookContext with session and agent info.
        tool_input: Parsed TaskToolInput from PreToolUse.
        validators: List of validator types to run.
        transcript_path: Path to subagent transcript (if available).
        validation_level: Validation intensity level.
        model: Claude model for validation.

    Returns:
        SpawnRequest ready to send to the API.

    Raises:
        ValueError: If required context fields are missing.
    """
    project_id = os.environ.get("ARZULE_PROJECT_ID", "")

    if not context.session_id:
        raise ValueError("session_id is required")

    if not context.agent_id:
        raise ValueError("agent_id is required for spawning validators")

    return SpawnRequest(
        project_id=project_id,
        session_id=context.session_id,
        agent_id=context.agent_id,
        subagent_type=tool_input.subagent_type,
        description=tool_input.description,
        prompt=tool_input.prompt,
        transcript_path=transcript_path,
        validators=validators,
        validation_level=validation_level,
        model=model,
        metadata={
            "tool_use_id": context.tool_use_id,
            "model_override": tool_input.model,
        },
    )


def map_result_to_hook_response(
    result: ValidationResult,
    include_details: bool = True,
) -> dict[str, Any]:
    """Map a ValidationResult to a hook response dict.

    This function converts a ValidationResult into the format expected
    by Claude Code SubagentStop hook responses.

    Args:
        result: ValidationResult from the API.
        include_details: Whether to include detailed issue info.

    Returns:
        Dict suitable for hook JSON output.
    """
    # Determine continue/block decision
    should_continue = not result.should_block()

    response: dict[str, Any] = {
        "continue": should_continue,
    }

    # Add decision reason if blocking
    if not should_continue:
        response["decision"] = "block"
        response["reason"] = result.get_summary()

    # Add validation details if requested
    if include_details:
        response["validation"] = {
            "decision": result.decision.value,
            "aggregate_score": result.aggregate_score,
            "passed": result.passed,
            "validators_completed": result.validators_completed,
            "validators_total": result.validators_total,
            "critical_issues_count": len(result.critical_issues),
            "high_issues_count": len(result.high_issues),
        }

        # Include critical issue summaries
        if result.critical_issues:
            response["validation"]["critical_issues"] = [
                {
                    "validator_type": issue.validator_type,
                    "category": issue.category,
                    "description": issue.description[:200],
                }
                for issue in result.critical_issues[:5]
            ]

    return response


def correlate_subagent_to_task(
    agent_id: str,
    session_state: dict[str, Any],
    agent_transcript_path: Optional[str] = None,
) -> Optional[str]:
    """Correlate a subagent to its originating Task tool call.

    When SubagentStop fires, we need to find the corresponding Task
    tool_use_id to retrieve the right validation context. This function
    implements correlation strategies:

    1. Direct mapping (if available from SubagentStart)
    2. Prompt matching (compare transcript prompts)
    3. FIFO ordering (most recent unprocessed Task)

    Args:
        agent_id: Subagent ID from SubagentStop.
        session_state: Persisted session state with pending validations.
        agent_transcript_path: Path to subagent's transcript.

    Returns:
        tool_use_id of the corresponding Task, or None if not found.
    """
    pending = session_state.get("pending_validations", {})
    agent_to_task = session_state.get("agent_to_task_mapping", {})

    # Strategy 1: Direct mapping from SubagentStart
    if agent_id in agent_to_task:
        return agent_to_task[agent_id]

    # Strategy 2: Prompt matching
    if agent_transcript_path:
        subagent_prompt = _extract_first_user_message(agent_transcript_path)
        if subagent_prompt:
            best_match = None
            best_score = 0.0

            for tool_use_id, info in pending.items():
                if info.get("processed"):
                    continue

                task_prompt = info.get("prompt", "")
                if not task_prompt:
                    continue

                score = _compute_prompt_similarity(subagent_prompt, task_prompt)
                if score > best_score:
                    best_score = score
                    best_match = tool_use_id

            if best_match and best_score >= 0.1:
                return best_match

    # Strategy 3: FIFO (most recent unprocessed)
    for tool_use_id, info in pending.items():
        if not info.get("processed"):
            return tool_use_id

    return None


def _extract_first_user_message(transcript_path: str) -> Optional[str]:
    """Extract the first user message from a transcript file.

    Args:
        transcript_path: Path to JSONL transcript file.

    Returns:
        First user message text, or None if not found.
    """
    import json

    try:
        path = Path(transcript_path)
        if not path.exists():
            return None

        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("role") == "user":
                        content = entry.get("content", "")
                        if isinstance(content, list):
                            # Handle structured content
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    return item.get("text", "")
                        elif isinstance(content, str):
                            return content
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return None


def _compute_prompt_similarity(prompt_a: str, prompt_b: str) -> float:
    """Compute similarity score between two prompts.

    Uses keyword overlap as a simple similarity metric.

    Args:
        prompt_a: First prompt text.
        prompt_b: Second prompt text.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    # Extract keywords (simple word tokenization)
    words_a = set(re.findall(r"\b\w+\b", prompt_a.lower()))
    words_b = set(re.findall(r"\b\w+\b", prompt_b.lower()))

    if not words_a or not words_b:
        return 0.0

    # Jaccard similarity
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)

    return intersection / union if union > 0 else 0.0


def get_transcript_path_for_agent(
    session_id: str,
    agent_id: str,
) -> Optional[str]:
    """Get the transcript path for a subagent.

    Claude Code stores subagent transcripts in predictable locations.
    This function attempts to find the transcript for a given agent.

    Args:
        session_id: Claude Code session ID.
        agent_id: Subagent ID.

    Returns:
        Path to transcript file if found, None otherwise.
    """
    # Claude Code transcript locations
    claude_projects_dir = Path.home() / ".claude" / "projects"

    # Try to find by agent_id pattern
    for path in claude_projects_dir.glob(f"*/agent-{agent_id}.jsonl"):
        return str(path)

    # Try session-based path
    for path in claude_projects_dir.glob(f"*/{session_id}-{agent_id}.jsonl"):
        return str(path)

    return None
