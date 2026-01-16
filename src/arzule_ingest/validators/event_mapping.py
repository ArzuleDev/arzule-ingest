"""Event to validator mapping configuration.

This module defines which validators should be triggered for different event types,
and at which validation levels.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .types import ValidationLevel, ValidatorType


# Map event types to validator types that should be triggered
VALIDATOR_TRIGGER_EVENTS: Dict[str, List[str]] = {
    # Tool execution events
    "tool.call.end": ["security", "correctness"],
    "tool.call.error": ["security", "correctness"],

    # Agent execution events
    "agent.execution.complete": ["scope", "correctness"],
    "agent.execution.error": ["scope", "correctness"],

    # Task lifecycle events
    "task.complete": ["compliance", "groundedness"],
    "task.error": ["compliance"],

    # Run lifecycle events (comprehensive validation)
    "run.end": ["security", "correctness", "scope", "compliance", "groundedness"],
    "run.error": ["security", "correctness", "scope"],

    # LLM events
    "llm.call.end": ["groundedness"],
    "llm.call.error": ["groundedness"],

    # Message events
    "message.send": ["security", "compliance"],
    "message.receive": ["security"],

    # Code execution events
    "code.execution.end": ["security", "correctness"],
    "code.execution.error": ["security"],

    # CrewAI-specific events (from listener.py)
    "crewai.agent.execution.completed": ["crewai_agent", "security", "correctness"],
    "crewai.task.completed": ["crewai_agent", "scope", "correctness"],
    "crewai.tool.usage.finished": ["crewai_tool_usage", "security"],
    "crewai.a2a.delegation.completed": ["crewai_delegation", "security", "scope"],
    "crewai.flow.method.finished": ["crewai_flow", "scope"],

    # LangGraph-specific events (from callback_handler.py)
    "langgraph.node.end": ["langgraph_node", "security", "correctness"],
    "langgraph.edge.traverse": ["langgraph_edge", "scope"],
    "langgraph.state.update": ["langgraph_state", "security"],
    "langgraph.parallel.fanout": ["langgraph_send", "scope", "security"],
    "langgraph.handoff.complete": ["langgraph_node", "correctness"],
}


# Validators to run at each level (cumulative - each level includes previous)
LEVEL_VALIDATORS: Dict[ValidationLevel, Set[str]] = {
    ValidationLevel.MINIMAL: {"security"},
    ValidationLevel.STANDARD: {
        "security",
        "correctness",
        "scope",
        # CrewAI validators for standard level
        "crewai_agent",
        "crewai_tool_usage",
        # LangGraph validators for standard level
        "langgraph_node",
        "langgraph_edge",
    },
    ValidationLevel.THOROUGH: {
        "security",
        "correctness",
        "scope",
        "compliance",
        "groundedness",
        # All CrewAI validators
        "crewai_agent",
        "crewai_delegation",
        "crewai_tool_usage",
        "crewai_flow",
        # All LangGraph validators
        "langgraph_node",
        "langgraph_edge",
        "langgraph_state",
        "langgraph_send",
    },
}


# Event types that should always trigger validators regardless of level
ALWAYS_VALIDATE_EVENTS: Set[str] = {
    "run.end",
    "run.error",
    "code.execution.end",
    "code.execution.error",
}


# Event types that should skip validation entirely
SKIP_VALIDATION_EVENTS: Set[str] = {
    "run.start",
    "agent.start",
    "task.start",
    "tool.call.start",
    "llm.call.start",
}


def should_trigger_validators(
    event_type: str,
    level: str,
) -> Tuple[bool, List[str]]:
    """Determine if validators should be triggered for an event.

    Args:
        event_type: The type of event (e.g., "tool.call.end").
        level: The validation level ("minimal", "standard", or "thorough").

    Returns:
        Tuple of (should_trigger, validators_to_run):
            - should_trigger: Boolean indicating if validation should occur
            - validators_to_run: List of validator types to run
    """
    # Skip validation for start events
    if event_type in SKIP_VALIDATION_EVENTS:
        return (False, [])

    # Check if this event type has validators configured
    if event_type not in VALIDATOR_TRIGGER_EVENTS:
        return (False, [])

    # Get the validators for this event type
    event_validators = set(VALIDATOR_TRIGGER_EVENTS[event_type])

    # Parse the validation level
    try:
        validation_level = ValidationLevel(level.lower())
    except ValueError:
        # Default to standard if invalid level
        validation_level = ValidationLevel.STANDARD

    # Get validators allowed at this level
    level_validators = LEVEL_VALIDATORS.get(validation_level, LEVEL_VALIDATORS[ValidationLevel.STANDARD])

    # For always-validate events, use all configured validators
    if event_type in ALWAYS_VALIDATE_EVENTS:
        validators_to_run = list(event_validators)
    else:
        # Filter to only validators allowed at this level
        validators_to_run = list(event_validators & level_validators)

    if not validators_to_run:
        return (False, [])

    return (True, validators_to_run)


def get_validators_for_event(event_type: str) -> List[str]:
    """Get all possible validators for an event type.

    Args:
        event_type: The type of event.

    Returns:
        List of validator types configured for this event, or empty list.
    """
    return VALIDATOR_TRIGGER_EVENTS.get(event_type, [])


def get_events_for_validator(validator_type: str) -> List[str]:
    """Get all event types that trigger a specific validator.

    Args:
        validator_type: The validator type to look up.

    Returns:
        List of event types that trigger this validator.
    """
    events = []
    for event_type, validators in VALIDATOR_TRIGGER_EVENTS.items():
        if validator_type in validators:
            events.append(event_type)
    return events


def get_level_validators(level: str) -> List[str]:
    """Get all validators enabled at a specific level.

    Args:
        level: The validation level ("minimal", "standard", or "thorough").

    Returns:
        List of validator types enabled at this level.
    """
    try:
        validation_level = ValidationLevel(level.lower())
    except ValueError:
        validation_level = ValidationLevel.STANDARD

    return list(LEVEL_VALIDATORS.get(validation_level, set()))


def is_critical_event(event_type: str) -> bool:
    """Check if an event type requires validation regardless of level.

    Args:
        event_type: The type of event.

    Returns:
        True if this event should always be validated.
    """
    return event_type in ALWAYS_VALIDATE_EVENTS


def validate_event_type(event_type: str) -> bool:
    """Check if an event type is recognized.

    Args:
        event_type: The type of event.

    Returns:
        True if this is a known event type with validator configuration.
    """
    return event_type in VALIDATOR_TRIGGER_EVENTS or event_type in SKIP_VALIDATION_EVENTS


def get_all_event_types() -> List[str]:
    """Get all configured event types.

    Returns:
        List of all event types that have validator configuration.
    """
    return list(VALIDATOR_TRIGGER_EVENTS.keys())


def get_all_validator_types() -> List[str]:
    """Get all validator types.

    Returns:
        List of all validator type names.
    """
    return [v.value for v in ValidatorType]
