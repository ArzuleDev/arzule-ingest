"""Validators module for Arzule SDK.

This module provides types, hooks, client, and streaming for interacting with
the Arzule Validator API.

Example:
    from arzule_ingest.validators import (
        ValidatorClient,
        ValidatorHook,
        DefaultValidatorHook,
        ValidationResult,
        ValidationLevel,
        ValidatorType,
        ValidatorResultStream,
    )

    # Create a client
    client = ValidatorClient(
        endpoint="https://api.arzule.com/v1/validators",
        api_key="your-api-key",
        project_id="your-project-id",
    )

    # Spawn validators for an event
    client.spawn_validators(
        session_id="session-123",
        agent_id="agent-456",
        event_type="tool.call.end",
        event_data={"tool_name": "search"},
        validators=["security", "correctness"],
        level=ValidationLevel.STANDARD,
    )

    # Register hooks for validation events
    hook = DefaultValidatorHook()
    hook.register_result_callback(lambda data: print(f"Result: {data}"))

    # Or use SSE streaming for real-time results
    stream = ValidatorResultStream(
        endpoint="https://api.arzule.com",
        api_key="your-api-key",
        session_id="session-123",
    )
"""

from .hooks import (
    CompositeValidatorHook,
    DefaultValidatorHook,
    LoggingValidatorHook,
    ValidatorCallback,
    ValidatorHook,
)
from .client import (
    ValidatorClient,
    ValidatorClientError,
    ValidatorAuthError,
    ValidatorTimeoutError,
)
from .types import (
    AggregatedValidationResult,
    ValidationIssue,
    ValidationLevel,
    ValidationResult,
    ValidatorType,
)
from .event_mapping import (
    VALIDATOR_TRIGGER_EVENTS,
    get_all_event_types,
    get_all_validator_types,
    get_events_for_validator,
    get_level_validators,
    get_validators_for_event,
    is_critical_event,
    should_trigger_validators,
    validate_event_type,
)
from .streaming import ValidatorResultCallback, ValidatorResultStream

__all__ = [
    # Types
    "ValidationLevel",
    "ValidatorType",
    "ValidationIssue",
    "ValidationResult",
    "AggregatedValidationResult",
    # Hooks
    "ValidatorHook",
    "ValidatorCallback",
    "DefaultValidatorHook",
    "LoggingValidatorHook",
    "CompositeValidatorHook",
    # Client
    "ValidatorClient",
    "ValidatorClientError",
    "ValidatorAuthError",
    "ValidatorTimeoutError",
    # Streaming
    "ValidatorResultStream",
    "ValidatorResultCallback",
    # Event mapping
    "VALIDATOR_TRIGGER_EVENTS",
    "should_trigger_validators",
    "get_validators_for_event",
    "get_events_for_validator",
    "get_level_validators",
    "is_critical_event",
    "validate_event_type",
    "get_all_event_types",
    "get_all_validator_types",
]
